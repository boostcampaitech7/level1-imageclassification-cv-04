import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from src.prewitt_edges import enhance_edges

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
                   # Albumentations 변환으로 사용
                    A.Lambda(image=lambda x, **kwargs: enhance_edges(x)),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20),
                    A.RandomBrightnessContrast(p=0.25),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.Blur()
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed['image']
