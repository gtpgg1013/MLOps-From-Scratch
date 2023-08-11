from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
from torch.optim import Adam
import torch.nn as nn
import torch
import io
import cv2

from fastapi.responses import FileResponse
from typing import Optional, List
import sqlite3
import os
import zipfile

import uuid
import base64

# 모델 정의
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, 10)
optimizer = Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 이미지를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read()))

    # 이미지를 로컬 디스크에 저장
    image.save(f"/path/to/images/{file.filename}")

    # 데이터베이스에 이미지 메타데이터 삽입
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (path) VALUES (?)', (f"/path/to/images/{file.filename}",))
    conn.commit()
    conn.close()

    return {"status": "success"}

@app.get("/datasets")
def datasets():
    # 데이터베이스에서 데이터셋 정보를 조회
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM datasets')
    datasets = cursor.fetchall()
    conn.close()

    return [{"name": dataset[1], "version": dataset[2]} for dataset in datasets]

@app.post('/datasets')
def create_dataset(name: str, labels: str):
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    
    # datasets 테이블에 데이터셋 정보 저장
    cursor.execute('INSERT INTO datasets (name, labels) VALUES (?, ?)', (name, labels))
    
    # labels 테이블에 각 라벨 정보 및 encoding 저장
    for idx, label_name in enumerate(labels.split(',')):
        cursor.execute('INSERT INTO labels (name, encoding, dataset) VALUES (?, ?, ?)', (label_name.strip(), idx, name))

    conn.commit()
    conn.close()
    return {}

@app.post('/images/upload/{dataset}')
async def upload_images(dataset: str, img_data: dict):
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()

    # 이미지 데이터 디코딩
    img_str = img_data['image']
    label = img_data['label']
    original_file_name = f"{label}.jpg"

    # 중복 확인
    cursor.execute('SELECT COUNT(*) FROM images WHERE path LIKE ?', (f'/path/to/images/{dataset}_{original_file_name}',))
    count = cursor.fetchone()[0]
    
    # 중복된 경우 UUID를 파일명에 추가
    if count > 0:
        file_name = f"{label}_{uuid.uuid4().hex}.jpg"
    else:
        file_name = original_file_name

    path = f'/path/to/images/{dataset}_{file_name}'

    # 이미지 저장
    with open(path, 'wb') as f:
        f.write(base64.b64decode(img_str))

    # 메타데이터를 데이터베이스에 저장
    cursor.execute('INSERT INTO images (path, label, dataset) VALUES (?, ?, ?)', (path, label, dataset))

    conn.commit()
    conn.close()
    return {}



@app.post("/train")
def train(dataset: str):
    # 데이터베이스에서 선택된 데이터셋의 이미지들을 조회
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images WHERE dataset = ?', (dataset,))
    images = cursor.fetchall()
    conn.close()

    # 이미지들을 사용하여 모델 학습
    for image in images:
        path = image[1]
        img = cv2.imread(path)  # 이미지 로드
        img = ToTensor()(img).unsqueeze(0)  # 이미지를 텐서로 변환
        label = torch.tensor([0])  # 라벨을 텐서로 변환, 0으로 가정

        pred = model(img)  # 예측
        loss = loss_fn(pred, label)  # 손실 계산

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 학습된 모델 및 코드를 디스크에 저장
    model_path = '/path/to/model.pth'
    torch.save(model.state_dict(), model_path)

    # 데이터베이스에 모델 메타데이터 삽입
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO models (name, version, dataset, path) VALUES (?, ?, ?, ?)', ('model', 'v1', dataset, model_path))
    conn.commit()
    conn.close()

    return {"status": "success"}

@app.get("/images")
def images(dataset: str, page: int = 1):
    # 데이터베이스에서 선택된 데이터셋의 이미지들을 조회
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()
    offset = (page - 1) * 100
    cursor.execute('SELECT * FROM images WHERE dataset = ? LIMIT 100 OFFSET ?', (dataset, offset))
    images = cursor.fetchall()
    conn.close()

    return [{"id": image[0], "path": image[1], "label": image[2]} for image in images]

@app.get('/images/count')
def count_images(dataset: Optional[str] = None):
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()

    if dataset is not None:
        cursor.execute(f'SELECT COUNT(*) FROM images WHERE dataset="{dataset}"')
    else:
        cursor.execute('SELECT COUNT(*) FROM images')

    count = cursor.fetchone()[0]
    conn.close()
    return {'count': count}

@app.get('/labels')
def labels():
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM labels')
    labels = [{'id': id, 'name': name, 'dataset': dataset, 'encoding': encoding} for id, name, encoding, dataset in cursor.fetchall()]

    conn.close()

    return labels

@app.post("/upload_zip/{dataset}")
async def upload_zip(dataset: str, file: UploadFile = File(...)):
    temp_dir = "/path/to/temp"
    extracted_dir = "/path/to/temp/extracted"

    # 필요한 디렉터리가 있는지 확인하고, 없다면 생성
    for dir_path in [temp_dir, extracted_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # ZIP 파일을 로컬 디스크에 일시적으로 저장
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, 'wb') as buffer:
        buffer.write(await file.read())
    
    # ZIP 파일 압축 해제
    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
        zip_ref.extractall("/path/to/temp/extracted")

    # 압축 해제된 모든 파일 처리
    extracted_path = "/path/to/temp/extracted"
    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # 여기서 파일 이름에서 레이블을 추출하려고 시도합니다. 파일 이름이 'label_imagename.jpg' 형식이라고 가정합니다.
            label = filename.split("_")[0]

            # 레이블 유효성 검사. 이는 데이터셋의 레이블과 일치해야 합니다.
            if label not in [label['name'] for label in labels if label['dataset'] == dataset]:
                return {"status": "failure", "reason": f"Invalid label {label} in the ZIP file."}

            # 이미지를 최종 경로로 이동
            final_path = f"/path/to/images/{dataset}_{filename}"
            os.rename(file_path, final_path)

            # 메타데이터를 데이터베이스에 저장
            conn = sqlite3.connect('mlops.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO images (path, label, dataset) VALUES (?, ?, ?)', (final_path, label, dataset))
            conn.commit()
            conn.close()

    # 임시 파일 및 디렉토리 정리
    os.remove(temp_path)
    os.rmdir(extracted_path)

    return {"status": "success"}

@app.delete("/datasets")
def delete_dataset(name: str):
    conn = sqlite3.connect('mlops.db')
    cursor = conn.cursor()

    # 해당 데이터셋의 모든 이미지 삭제
    cursor.execute('SELECT path FROM images WHERE dataset = ?', (name,))
    images_to_delete = cursor.fetchall()
    for image in images_to_delete:
        os.remove(image[0])

    # 데이터베이스에서 해당 데이터셋과 관련된 모든 정보 삭제
    cursor.execute('DELETE FROM images WHERE dataset = ?', (name,))
    cursor.execute('DELETE FROM labels WHERE dataset = ?', (name,))
    cursor.execute('DELETE FROM datasets WHERE name = ?', (name,))
    conn.commit()
    conn.close()

    return {"status": "success"}