import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import Config
from utils.dataset import CustomDataset
from utils.transforms import TransformSelector
from utils.trainer import Trainer
from models import ModelSelector

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and split data
    train_info = pd.read_csv(Config.TRAIN_INFO_FILE)
    num_classes = len(train_info['target'].unique())
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target']
    )

    # Set up transforms
    transform_selector = TransformSelector(Config.TRANSFORM_TYPE)
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(Config.TRAIN_DIR, train_df, train_transform)
    val_dataset = CustomDataset(Config.TRAIN_DIR, val_df, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Set up model
    model_selector = ModelSelector(
        model_type=Config.MODEL_TYPE,
        num_classes=num_classes,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED
    )
    model = model_selector.get_model().to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    steps_per_epoch = len(train_loader)
    scheduler_step_size = steps_per_epoch * Config.EPOCHS_PER_LR_DECAY
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=Config.SCHEDULER_GAMMA
    )

    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=Config.EPOCHS,
        result_path=Config.SAVE_RESULT_PATH
    )
    trainer.train()

if __name__ == "__main__":
    main()