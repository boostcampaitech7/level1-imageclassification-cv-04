import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from typing import Union, Tuple

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: callable,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)
    """
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
    img_path = os.path.join(self.root_dir, self.image_paths[index])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = self.transform(image)

    if self.is_inference:
        return image
    else:
        target = self.targets[index]
        return image, target    
    
    """
    ###추가
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_inference:
            image = self.transform(image=image)['image']
            return image
        else:
            target = self.targets[index]

            # CutMix 적용을 위해 두 번째 이미지를 랜덤하게 선택
            if hasattr(self.transform, 'cutmix_prob') and random.random() < self.transform.cutmix_prob:
                 # 현재 이미지와 다른 target을 가진 인덱스 랜덤 선택
                while True:
                    second_index = random.randint(0, len(self.image_paths) - 1)
                    if self.targets[second_index] != target:
                        break
                second_img_path = os.path.join(self.root_dir, self.image_paths[second_index])
                second_image = cv2.imread(second_img_path, cv2.IMREAD_COLOR)
                second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
                second_target = self.targets[second_index]

                # transform 호출 시 CutMix 적용
                image, target = self.transform(image=image, label=target, second_image=second_image, second_label=second_target)
            else:
                 image, target = self.transform(image=image, label=target)
            
            
            return image, target
