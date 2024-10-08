import os
import cv2
import torch
import numpy as np
import pandas as pd
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