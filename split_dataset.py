import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

###split_dataset 함수 선언

def split_dataset(dataset, val_split=0.2):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(num_samples * val_split)
    
    train_indices, val_indices = train_test_split(indices, test_size=val_split, shuffle=True, random_state=42)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    return train_subset, val_subset

###아래에 train dataset 경로만 지정해주면 train dataset, val dataset로 나누어줌
###train dataset은 augmentaion, 전처리 끝낸 후의 set 의미
train_dataset, val_dataset = split_dataset(dataset)

