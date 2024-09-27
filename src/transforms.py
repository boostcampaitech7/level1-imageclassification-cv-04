import albumentations as A
from albumentations.pytorch import ToTensorV2
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
                    A.Rotate(limit=10,p=0.5),
                    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    # A.OneOf([
                    #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                    # ], p=0.5),

                    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed['image']
