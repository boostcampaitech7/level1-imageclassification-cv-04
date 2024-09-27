import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import argparse
from timm.data.mixup import Mixup

from torch.utils.tensorboard import SummaryWriter  # 추가

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.trainer import Trainer, Loss, Focal_CELoss
from src.layer_modification import layer_modification

def main(args):
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

    ##mixup 추가
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.12, num_classes=num_classes)

    # Set up model
    model_selector = ModelSelector(model_type=args.model_type, num_classes=num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model = model_selector.get_model()
    model = layer_modification(model)
    model.to(device)

    # Scheduler initialization
    steps_per_epoch = len(train_loader)
    scheduler_step_size = steps_per_epoch * args.epochs_per_lr_decay

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=args.scheduler_gamma)

    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.log_dir)

    # Set up trainer and train
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=Loss(mixup_fn=mixup_fn), 
        epochs=args.epochs,
        result_path=args.save_result_path,
        writer=writer,  # 추가
        mixup_fn = mixup_fn,
        wrong_path="./wrong"
    )
    trainer.train()
    writer.close()  # 추가

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--traindata_dir", type=str, default="./data/train", help="Path to training data directory")
    parser.add_argument("--traindata_info_file", type=str, default="./data/train.csv", help="Path to training data info file")
    parser.add_argument("--save_result_path", type=str, default="./train_result", help="Path to save training results")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to save TensorBoard logs")  # 추가
    
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--transform_type", type=str, default="albumentations", help="Type of data transforms to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")

    parser.add_argument("--model_type", type=str, default="timm", help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", help="Name of the model")
    parser.add_argument("--pretrained", type=bool, default=True, help="Whether to use pretrained weights")
    
    # * Mixup params
    parser.add_argument('--mixup', type = float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # parser.add_argument('--smoothing', type=float, default=0,
    #                     help='Label smoothing (default: 0.1)')

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--epochs_per_lr_decay", type=int, default=2, help="Number of epochs before learning rate decay")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Learning rate decay factor")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")

    args = parser.parse_args()
    main(args)