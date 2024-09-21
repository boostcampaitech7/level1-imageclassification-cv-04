import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import argparse

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.trainer import Trainer, Loss

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_info = pd.read_csv(args.traindata_info_file)
    num_classes = len(train_info['target'].unique())

    # Split data
    train_df, val_df = train_test_split(train_info, test_size=args.val_split, stratify=train_info['target'])

    # Set up transforms
    transform_selector = TransformSelector(transform_type=args.transform_type)
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Set up datasets
    train_dataset = CustomDataset(root_dir=args.traindata_dir, info_df=train_df, transform=train_transform)
    val_dataset = CustomDataset(root_dir=args.traindata_dir, info_df=val_df, transform=val_transform)

    # Set up dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up model
    model_selector = ModelSelector(model_type=args.model_type, num_classes=num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model = model_selector.get_model()
    model.to(device)

    # Scheduler initialization
    steps_per_epoch = len(train_loader)
    scheduler_step_size = steps_per_epoch * args.epochs_per_lr_decay

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=args.scheduler_gamma)

    # Set up trainer and train
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=Loss(), 
        epochs=args.epochs,
        result_path=args.save_result_path
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--traindata_dir", type=str, default="./data/train", help="Path to training data directory")
    parser.add_argument("--traindata_info_file", type=str, default="./data/train.csv", help="Path to training data info file")
    parser.add_argument("--save_result_path", type=str, default="./train_result", help="Path to save training results")
    
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--transform_type", type=str, default="albumentations", help="Type of data transforms to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")

    parser.add_argument("--model_type", type=str, default="timm", help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Name of the model")
    parser.add_argument("--pretrained", type=bool, default=True, help="Whether to use pretrained weights")\
    
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--epochs_per_lr_decay", type=int, default=2, help="Number of epochs before learning rate decay")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Learning rate decay factor")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")

    args = parser.parse_args()
    main(args)