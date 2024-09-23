import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.trainer import Trainer, Loss, Focal_CELoss
from src.layer_modification import layer_modification
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_result"
    wrong_prediction_path = "./wrong_prediction"

    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique())

    # Split data
    train_df, val_df = train_test_split(train_info, test_size=0.2, stratify=train_info['target'])

    # Set up transforms
    transform_selector = TransformSelector(transform_type="albumentations")
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Set up datasets
    train_dataset = CustomDataset(root_dir=traindata_dir, info_df=train_df, transform=train_transform)
    val_dataset = CustomDataset(root_dir=traindata_dir, info_df=val_df, transform=val_transform)

    # Set up dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Set up model
    model_selector = ModelSelector(model_type='timm', num_classes=num_classes, model_name='resnet18', pretrained=True)
    model = model_selector.get_model()
    model = layer_modification(model)
    model.to(device)

    # 스케줄러 초기화
    scheduler_step_size = 30  # 매 30step마다 학습률 감소
    scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소
    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)
    # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
    epochs_per_lr_decay = 2
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    # Set up trainer and train
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=Loss(), 
        epochs=5,
        result_path=save_result_path,
        wrong_path=wrong_prediction_path
    )    
    trainer.train()
if __name__ == "__main__":
    main()