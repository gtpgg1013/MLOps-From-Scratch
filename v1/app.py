import streamlit as st
from PIL import Image
import requests
import os
import pandas as pd
import base64
from io import BytesIO

# tab 구조로 페이지 구현
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Management", "Model Training", "Dataset Viewer"])

# 홈 페이지
if page == "Home":
    st.title("Home")
    st.write("""
    Welcome to the MLOps system!

    This system allows you to manage your datasets, train machine learning models, and make predictions.
    You can upload your own images, label them, and create datasets. Then, you can use these datasets to train your own models.
    Once a model is trained, you can deploy it and start making predictions.

    Let's get started!
    """)

# 데이터 관리 페이지
elif page == 'Data Management':
    st.write('## Data Management')

    datasets = requests.get('http://localhost:8000/datasets').json()
    labels = requests.get('http://localhost:8000/labels').json()

    action = st.radio('Select an action', ['Create a dataset', 'Add single image to a dataset', 'Upload images via ZIP to a dataset', 'Delete a dataset'])

    if action == 'Create a dataset':
        dataset_name = st.text_input('Enter a name for the new dataset')
        labels_input = st.text_input('Enter labels for the new dataset (separated by commas)')
        
        if st.button('Create dataset'):
            labels_list = [label.strip() for label in labels_input.split(',')]
            response = requests.post(f'http://localhost:8000/datasets?name={dataset_name}&labels={",".join(labels_list)}')
            st.write('Dataset created!')

    elif action == 'Add single image to a dataset':
        dataset_name = st.selectbox('Select a dataset', [dataset['name'] for dataset in datasets])

        # 이미지 업로드
        uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'png', 'jpeg'])
        # st.write(labels)
        label = st.selectbox('Select a label', [label['name'] for label in labels if label['dataset'] == dataset_name])

        # 임시 이미지와 라벨 리스트
        if 'temp_images' not in st.session_state:
            st.session_state.temp_images = []
        
        # 이미지 추가 버튼
        if st.button("Add Image to List"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                st.session_state.temp_images.append({'image': img_str, 'label': label})
                st.write(f'Image added to the list! Current count: {len(st.session_state.temp_images)}')

        # 모든 이미지 제출 버튼
        if st.button("Submit All Images"):
            for img_data in st.session_state.temp_images:
                response = requests.post(f'http://localhost:8000/images/upload/{dataset_name}', json=img_data)
                if response.status_code == 200:
                    st.write(f'Image with label {img_data["label"]} uploaded successfully!')
                else:
                    st.write(f'Error uploading image with label {img_data["label"]}.')
            # 임시 이미지 리스트 초기화
            st.session_state.temp_images = []

    elif action == 'Upload images via ZIP to a dataset':
        dataset_name = st.selectbox('Select a dataset', [dataset['name'] for dataset in datasets])
        uploaded_file = st.file_uploader('Choose a ZIP file', type=['zip'])
        
        if uploaded_file is not None:
            # ZIP 파일을 서버로 전송
            files = {'file': uploaded_file}
            response = requests.post(f'http://localhost:8000/upload_zip/{dataset_name}', files=files)
            
            if response.status_code == 200 and response.json().get("status") == "success":
                st.write('ZIP file successfully uploaded and processed!')
            else:
                st.write('Error processing ZIP file. Ensure the labels in the ZIP file match the dataset labels.')

    elif action == 'Delete a dataset':
        dataset_to_delete = st.selectbox('Select a dataset to delete', [dataset['name'] for dataset in datasets])

        if st.button(f"Delete {dataset_to_delete}"):
            response = requests.delete(f'http://localhost:8000/datasets?name={dataset_to_delete}')
            
            if response.status_code == 200 and response.json().get("status") == "success":
                st.write(f'Dataset {dataset_to_delete} successfully deleted!')
            else:
                st.write(f'Error deleting dataset {dataset_to_delete}.')

# 모델 학습 페이지
elif page == "Model Training":
    st.title("Model Training")

    # 데이터셋 선택
    datasets = requests.get('http://localhost:8000/datasets').json()
    dataset_name = st.selectbox('Select a dataset', [dataset['name'] for dataset in datasets])

    # 학습 시작 버튼
    if st.button('Start Training'):
        # FastAPI 서버에 학습 요청 POST
        response = requests.post(f'http://localhost:8000/train?dataset={dataset_name}')

        # 학습 결과를 표시
        if response.status_code == 200:
            st.write('Successfully started the training.')
        else:
            st.write('Server error')

# 데이터셋 뷰어 페이지
elif page == "Dataset Viewer":
    st.title("Dataset Viewer")

    # 데이터셋 선택
    datasets = requests.get('http://localhost:8000/datasets').json()
    dataset_name = st.selectbox('Select a dataset', [dataset['name'] for dataset in datasets])

    # 총 페이지 수 계산
    total_images = requests.get(f'http://localhost:8000/images/count?dataset={dataset_name}').json()['count']
    total_pages = (total_images + 99) // 100  # 한 페이지에 이미지 100개

    # 페이지 번호 선택
    page_number = st.selectbox('Select a page number', list(range(1, total_pages + 1)))

    # 선택한 데이터셋의 이미지들을 조회
    images = requests.get(f'http://localhost:8000/images?dataset={dataset_name}&page={page_number}').json()

    # 컬럼 머리글 추가 (가운데 정렬, 색상 추가)
    header = st.columns(4)
    header[0].markdown('<div style="text-align:center; padding: 10px; background-color: #f0f0f0;">Image</div>', unsafe_allow_html=True)
    header[1].markdown('<div style="text-align:center; padding: 10px; background-color: #f0f0f0;">Dataset</div>', unsafe_allow_html=True)
    header[2].markdown('<div style="text-align:center; padding: 10px; background-color: #f0f0f0;">File Name</div>', unsafe_allow_html=True)
    header[3].markdown('<div style="text-align:center; padding: 10px; background-color: #f0f0f0;">Label</div>', unsafe_allow_html=True)

    # 각 이미지와 그에 대한 메타데이터를 행으로 표시
    for image in images:
        img = Image.open(image['path'])

        # 이미지를 base64 인코딩
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # 이미지와 메타데이터를 하나의 행으로 표시 (Bootstrap 스타일 적용, 마진 추가, 테두리 굵게)
        cols = st.columns(4)
        cols[0].markdown(f'<div style="border:2px solid #ddd; padding: 10px; height: 100px; margin-bottom: 10px;"><img src="data:image/jpeg;base64,{img_str}" style="width:100%; height:100%; display:block; margin: 0 auto;"></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div style="border:2px solid #ddd; padding: 10px; text-align:center; height: 100px; line-height: 80px; margin-bottom: 10px;">{dataset_name}</div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div style="border:2px solid #ddd; padding: 10px; text-align:center; height: 100px; line-height: 80px; margin-bottom: 10px;">{os.path.basename(image["path"])}</div>', unsafe_allow_html=True)
        cols[3].markdown(f'<div style="border:2px solid #ddd; padding: 10px; text-align:center; height: 100px; line-height: 80px; margin-bottom: 10px;">{image["label"]}</div>', unsafe_allow_html=True)