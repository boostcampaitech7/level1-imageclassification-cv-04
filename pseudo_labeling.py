import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import cv2
import json

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.layer_modification import layer_modification
import csv

import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from PIL import Image

def inference(model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
    model.to(device)
    model.eval()

    all_predictions = []
    all_images = []  # 이미지를 저장할 리스트

    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            logits = F.softmax(logits, dim=1)

            all_predictions.append(logits.cpu().detach().numpy())
            all_images.append(images.cpu().detach())  # 이미지를 CPU로 옮긴 후 저장

    predictions = np.concatenate(all_predictions, axis=0)
    all_images = torch.cat(all_images, dim=0)  # 이미지 리스트를 하나의 텐서로 결합

    return predictions, all_images  # 예측값과 이미지를 모두 반환

# pseudo_labeling_with_inference 함수
def pseudo_labeling_with_inference(model, device, test_loader, confidence_threshold=0.5):
    predictions, all_images = inference(model, device, test_loader)  # Inference 수행 및 이미지 반환
    
    max_probs = np.max(predictions, axis=1)  # 각 예측에서 가장 높은 확률값 
    preds = np.argmax(predictions, axis=1)  # 각 예측에서 가장 높은 확률을 가진 클래스 추출
    
    # confidence_threshold 이상인 데이터만 사용
    confident_mask = max_probs > confidence_threshold
    pseudo_labels = preds[confident_mask]
    
    # confidence_threshold 이상인 데이터의 이미지 추출
    pseudo_data = all_images[confident_mask]

    return pseudo_data, pseudo_labels


def save_pseudo_images_by_label(pseudo_data, pseudo_labels, labels, output_dir="./pseudo", prefix="sketch", csv_filename="output.csv"):
    csv_filepath = os.path.join("./", csv_filename)
    
    # CSV 파일 열기 (쓰기 모드)
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 라벨별 폴더 생성 및 이미지 저장
        label_counters = {}  # 각 라벨별로 파일 번호를 저장하는 딕셔너리
        
        for i, (image, label, target) in enumerate(zip(pseudo_data, pseudo_labels, labels)):
            # label이 리스트일 경우, 문자열로 변환
            if isinstance(label, list):
                label = str(label[0])
            
            # 라벨별로 폴더 생성
            label_folder = os.path.join(output_dir, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            
            # 각 라벨의 이미지 번호를 카운트
            if label not in label_counters:
                label_counters[label] = 1
            else:
                label_counters[label] += 1

            file_name = f"{prefix}_{label_counters[label]}.jpeg"
            file_path = os.path.join(label_folder, file_name)
            
            # 이미지 비정규화 및 변환 (Pillow는 RGB 형식이므로 바로 저장 가능)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
                image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image)  # NumPy 배열을 PIL 이미지로 변환
            elif len(image.shape) == 2:
                image = image.cpu().numpy()
                image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image).convert('RGB')  # 흑백 이미지를 RGB로 변환
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # 이미지 저장 (Pillow는 RGB 형식을 사용하므로 별도 변환 없이 저장 가능)
            img.save(file_path)

            # CSV 파일에 라벨, 경로, 타겟 저장
            csv_row = [label, f"{label}/{file_name}", target]
            writer.writerow(csv_row)
        

def main():
    # Set device
    with open('./imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"

    test_info = pd.read_csv(testdata_info_file)
    num_classes = 500

    # Set up transform
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # Set up test dataset and dataloader
    test_dataset = CustomDataset(root_dir=testdata_dir, info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # Load model
    model_selector = ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    model = model_selector.get_model()
    model = layer_modification(model)

    # Load the best model from ./train_result/best_model.pt
    model_path = os.path.join("./train_result", "model_adamw.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Run pseudo-labeling
    pseudo_data, labels = pseudo_labeling_with_inference(model=model, device=device, test_loader=test_loader, confidence_threshold=0.8)
    # Save results
    pseudo_wordnet_labels = [class_idx[str(label)] for label in labels]
    save_pseudo_images_by_label(pseudo_data, pseudo_wordnet_labels, labels)

if __name__ == "__main__":
    main()
