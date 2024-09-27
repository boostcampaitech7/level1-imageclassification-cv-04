import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import argparse
from timm.data.mixup import Mixup

from torch.utils.tensorboard import SummaryWriter

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.trainer import Trainer, Loss, Focal_CELoss
from src.layer_modification import layer_modification

import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import argparse
from timm.data.mixup import Mixup

from torch.utils.tensorboard import SummaryWriter

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.trainer import Trainer, Loss, Focal_CELoss
from src.layer_modification import layer_modification
from src.utils import inference
#앙상블 에측
# 각 fole 에서 학습된 모델의 예측값들의 평륜을 최종 예측으로 결정
def ensemble_inference(models, device, test_loader):
    all_predictions = []
    
    for model in models:
        #model_paths  
        # tmodel_path =
        # tmodel.load_state_dict(torch.load(tmodel_path, map_location=device))
        # tmodel.to(device)
        model.eval()
        predictions = inference(model=model, device=device, test_loader=test_loader)
        all_predictions.append(predictions)
    
    # Convert to numpy array for easier manipulation
    all_predictions = np.array(all_predictions)
    
    # Compute the mean prediction across all models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Convert to class predictions
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    
    return final_predictions

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
        # Load data
    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_KFold_result"
    wrong_prediction_path = "./wrong_KFold_prediction"

    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique())
    
    # Set up transforms
    transform_selector = TransformSelector(transform_type=args.transform_type)
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Set up datasets
    full_dataset = CustomDataset(root_dir=args.traindata_dir, info_df=train_info, transform=train_transform)

    # 레이블을 추출
    labels = train_info['target'].values

    # 1. StratifiedKFold 설정
    n_splits = args.n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    
    models = []  # 각 폴드에서 학습된 모델을 저장할 리스트
    #model_paths = [] #kfold로 학습한 후 저장된 모델들  path 릿트  
    # 2: 폴드별 학습 및 학습된 모델 저장
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset, labels)):
        print(f'Fold {fold+1}')
        
        # Subset을 사용하여 train_idx와 val_idx에 따른 학습 및 검증 데이터 분리
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # Set up DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True , drop_last = True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False , drop_last = True)
        
        
        # Mixup 설정
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

        # Optimizer and scheduler
        steps_per_epoch = len(train_loader)
        scheduler_step_size = steps_per_epoch * args.epochs_per_lr_decay
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=args.scheduler_gamma)

        # TensorBoard setup
        # writer = SummaryWriter(log_dir=f"{args.log_dir}/fold_{fold + 1}")
        writer = SummaryWriter(log_dir=f"{args.log_dir}/fold")
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
            result_path=   f"{args.save_result_path}/fold_{fold + 1}",
            writer=writer,
            mixup_fn=mixup_fn,
            wrong_path=f"./wrong_prediction_path/fold_{fold + 1}"
        )
        trainer.train()
        writer.close()

        # Save the model for this fold
        torch.save(model.state_dict(), f"{args.save_result_path}/model_fold_{fold + 1}.pth")
        #model_paths.append(f"args.save_result_path}/model_fold_{fold + 1}.pth")
        models.append(model) # tensor


    # Test 데이터에 대한 앙상블 예측 수행
    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"
    test_info = pd.read_csv(testdata_info_file)

      # 테스트 데이터용 Transform 설정
    test_transform = transform_selector.get_transform(is_train=False)
    test_dataset = CustomDataset(root_dir=testdata_dir, info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # # Load model
    # tmodel_selector = ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # tmodel = model_selector.get_model()
    # tmodel = layer_modification(model)

    # # Load the best model from ./train_result/best_model.pt
    # tmodel_path = os.path.join("./train_result", "best_model.pt")
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)

    # # Run inference
    # predictions = inference(model=model, device=device, test_loader=test_loader)


    # 앙상블 예측 수행
    predictions = ensemble_inference(models=models, device=device , test_loader=test_loader)

     # 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("kfold_output.csv", index=False)
    print("Ensemble inference completed and results saved to kfold_output.csv")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--traindata_dir", type=str, default="./data/train", help="Path to training data directory")
    parser.add_argument("--traindata_info_file", type=str, default="./data/train.csv", help="Path to training data info file")
    parser.add_argument("--save_result_path", type=str, default="./train_KFold_result", help="Path to save training results")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to save TensorBoard logs")  # 추가
    
    parser.add_argument("--n_splits", type=int, default=3, help="Number of folds for StratifiedKFold") ##추가
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
