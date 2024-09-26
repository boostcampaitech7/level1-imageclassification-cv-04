import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import numpy as np

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
        
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(448,448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20),
                    A.RandomBrightnessContrast(p=0.25),
                    A.Blur()
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image):
        image = np.array(image)  # numpy 배열로 변환
        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # edges = cv2.Canny(gray_image, 50, 150)
        # _, thresh = cv2.threshold(edges, 70, 150, cv2.THRESH_BINARY_INV)
        # image = thresh
        
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # numpy 배열 상태 유지
        
        # # PIL 변환 부분 제거하고, numpy 배열로 변환된 이미지를 그대로 전달
        transformed = self.transform(image=image)
        
        return transformed['image']
