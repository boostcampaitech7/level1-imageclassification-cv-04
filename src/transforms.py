import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import random
import torch

from PIL import Image

class TransformSelector:
    def __init__(self, transform_type: str):
        if transform_type in ["albumentations"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)
        return transform
"""
        
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),

                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed['image']
"""
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True, cutmix_prob: float = 0.5, beta: float = 1.0):
        self.is_train = is_train
        self.cutmix_prob = cutmix_prob
        self.beta = beta
        
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Blur()
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image, label=None, second_image=None, second_label=None):
        transformed = self.transform(image=image)
        image = transformed['image']

        if self.is_train and second_image is not None and label is not None and second_label is not None:
            # CutMix 적용
            lam = np.random.beta(self.beta, self.beta)
            image, mixed_label = self.cutmix(image, second_image, label, second_label, lam)
            return image, mixed_label
        
        return image, label  # Train 시 라벨도 반환

    def rand_bbox(self, size, lam):
        """랜덤한 bounding box를 생성"""
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 랜덤한 중심 좌표
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, image1, image2, label_a, label_b, lam):

        """CutMix 구현"""
        _, H, W = image1.shape
        
        # 랜덤한 bounding box 생성
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image1.shape, lam)
        
        # bounding box 유효성 체크
        bbx1 = max(0, bbx1)
        bby1 = max(0, bby1)
        bbx2 = min(W, bbx2)
        bby2 = min(H, bby2)

    # CutMix 적용
        if bbx1 < bbx2 and bby1 < bby2:  # 유효한 bounding box인지 확인
            slice1 = image1[:, bby1:bby2, bbx1:bbx2]
            slice2 = image2[:, bby1:bby2, bbx1:bbx2]

            # CutMix 적용
            if slice1.shape == slice2.shape:
                image1[:, bby1:bby2, bbx1:bbx2] = slice2.clone()  # 텐서 복사

        # 라벨 혼합
        mixed_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = H * W
        a = mixed_area / total_area  # 비율 계산

        # One-hot 인코딩 (num_classes는 클래스 수로 설정)
        num_classes = 500  # 예시: 클래스 수에 맞게 설정
        # One-hot 인코딩
        label_a = torch.nn.functional.one_hot(torch.tensor(label_a), num_classes=num_classes).float()
        # if len(label_b.shape) == 0 :
        label_b = torch.nn.functional.one_hot(torch.tensor(label_b), num_classes=num_classes).float()
        # print(f"label_A : {label_a} , label_b : {label_b}")
        # 면적 비율에 따라 라벨 혼합
        mixed_label = (1 - a) * label_a + a * label_b
        mixed_label = torch.argmax(mixed_label)
        print(f"mixed_label {mixed_label}  ")
        return image1, mixed_label