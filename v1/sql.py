from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from PIL import Image
import sqlite3
import os

# 필요한 디렉토리 생성
if not os.path.exists('/path/to/images'):
    os.makedirs('/path/to/images')

if not os.path.exists('/path/to'):
    os.makedirs('/path/to')

# SQLite 데이터베이스 연결
conn = sqlite3.connect('mlops.db')
cursor = conn.cursor()

# 필요한 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT,
    label INTEGER,
    dataset TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    labels TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    version TEXT,
    dataset TEXT,
    path TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    encoding INTEGER,
    dataset TEXT
)
''')

# MNIST 데이터셋 다운로드
dataset = MNIST(root='/path/to/mnist', download=True, transform=ToTensor())

# MNIST 데이터셋 정보를 datasets 테이블에 추가
cursor.execute('INSERT INTO datasets (name, labels) VALUES (?, ?)', ('mnist', ','.join(map(str, range(10)))))

# 각 레이블에 대한 숫자 인코딩 생성
for i in range(10):
    cursor.execute('INSERT INTO labels (name, encoding, dataset) VALUES (?, ?, ?)', (str(i), i, 'mnist'))

# 각 이미지를 로컬 디스크에 저장하고, 메타데이터를 데이터베이스에 저장
for i, (image, label) in enumerate(dataset):
    # 이미지 저장
    path = f'/path/to/images/mnist_{i}.jpg'
    image = Image.fromarray(image.mul(255).byte().numpy().squeeze())
    image.save(path)

    # 메타데이터 저장
    cursor.execute('INSERT INTO images (path, label, dataset) VALUES (?, ?, ?)', (path, label, 'mnist'))

conn.commit()
conn.close()
