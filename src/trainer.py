import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional




from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter  # 추가

from torchvision.utils import save_image ##
##mixup
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

class Loss(nn.Module):
    def __init__(self , mixup_fn=None):
        super(Loss, self).__init__()
        self.mixup_fn = mixup_fn
        #추가 - eva참고
        if self.mixup_fn is not None : 
            #smoothing is hadled with mixup label transform
            self.loss_fn = SoftTargetCrossEntropy()
        # elif args.smoothing > 0.:
        #     LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else : 
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.12) 

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        #
        # print(f"(targets { targets.shape} , {outputs.shape}")
        if targets.dim() >1 :
            targets = targets.argmax(dim=1)  # Convert to class indices
        # elif len(targets.shape) == 1:
        #     # One-hot encoding of targets
        #     targets = targets.view(-1, 1)  # Make it a column vector
        
        return self.loss_fn(outputs, targets)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # CrossEntropyLoss 사용

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-Entropy Loss 계산
        ce_loss = self.ce_loss(outputs, targets)

        # outputs -> softmax로 클래스 확률을 얻음
        probs = torch.softmax(outputs, dim=1)
        probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # 정답 클래스의 확률만 추출

        # Focal Loss 계산
        focal_weight = (1 - probs) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t

        # 최종 Loss 계산
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class Focal_CELoss(nn.Module):
    def __init__(self, focal_gamma=2.0, focal_alpha=None, label_smoothing=0.05, focal_threshold=0.7):
        super(Focal_CELoss, self).__init__()
        # CrossEntropyLoss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Focal Loss 설정
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        # Focal Loss를 사용할 헷갈리는 예제를 선택할 기준 확률 값
        self.focal_threshold = focal_threshold

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-Entropy Loss 계산
        ce_loss = self.ce_loss(outputs, targets)

        # 헷갈리는 예제를 위한 Focal Loss 계산
        probs = torch.softmax(outputs, dim=1)
        max_probs, _ = probs.max(dim=1)  # 각 예제에서 가장 큰 확률 값을 얻음

        # 헷갈리는 예제만 Focal Loss 적용 (최대 확률이 threshold 이하인 경우)
        focal_mask = max_probs < self.focal_threshold
        if focal_mask.sum() > 0:  # focal_mask가 존재하는 경우
            focal_loss = self.focal_loss(outputs[focal_mask], targets[focal_mask])
            # 두 손실을 합쳐줌
            total_loss = ce_loss + focal_loss
        else:
            total_loss = ce_loss  # focal_mask가 없으면 cross-entropy만 사용

        return total_loss


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: nn.Module, 
        epochs: int,
        result_path: str,
        writer: SummaryWriter,  # 추가
        wrong_path: str=None,
        mixup_fn: Optional[Mixup] = None, # 추가
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')
        self.writer = writer  # 추가
        self.wrong_path = wrong_path
        self.mixup_fn= mixup_fn
        

    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

  
            
    def train_epoch(self, epoch) -> float:  # epoch 인자 추가
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        acc_cum = 0
        
        batch_idx = len(progress_bar) # 
        # print(f"batch_idx {batch_idx}")
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            if self.mixup_fn is not None : #and batch_idx %4 == 0 : # 4step 주기로 mixup 적용  
                images , targets = self.mixup_fn(images, targets)
                # save_images(images ,"./augmenatation", batch_idx)
            
            outputs = self.model(images)
            acc_cum += self.accuracy(outputs, targets)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = acc_cum / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)  # 추가
        self.writer.add_scalar('Accuracy/train', avg_acc, epoch)  # 추가
        print(f'train accuracy: {avg_acc}%')
        return avg_loss

    def validate(self, epoch, log_wrong_predictions = False) -> float:  # epoch 인자 추가
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        acc_cum = 0
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                # print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
                
                acc_cum += self.accuracy(outputs, targets)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                if log_wrong_predictions:
                    self.save_wrong_image(images, outputs, targets)
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = acc_cum / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)  # 추가
        self.writer.add_scalar('Accuracy/val', avg_acc, epoch)  # 추가
        print(f'validation accuracy: {avg_acc}%')
        return avg_loss

    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self.train_epoch(epoch)  # epoch 인자 추가
            if epoch < self.epochs - 1:
                val_loss = self.validate(epoch)  # epoch 인자 추가
            else:
                val_loss = self.validate(epoch, log_wrong_predictions=True)  # epoch 인자 추가
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")
            self.save_model(epoch, val_loss)
            self.scheduler.step()
            
    def save_wrong_image(self, images, outputs, targets):
        unnormalize = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                        )
        to_image = transforms.ToPILImage()
        os.makedirs(self.wrong_path, exist_ok=True)
        _, preds = torch.max(outputs, 1)
        for i in range(len(targets)):
            if preds[i] != targets[i]:
                wrong_image = to_image(unnormalize(images[i].cpu()))
                filename = f"정답_{targets[i].cpu().item()}_예측_{preds[i].cpu().item()}.png"
                filepath = os.path.join(self.wrong_path, filename)
                wrong_image.save(filepath)
    
    def cross_validation(self):
        pass
    
    def accuracy(self, outputs, targets):
        #acc = (targets == predictions.argmax(1)).sum().item() / len(targets) * 100
         # predictions의 크기와 targets의 크기를 확인
        
        if targets.dim() >1 : #one hot encodeing 
            targets = targets.argmax(dim=1)

        predictions = outputs.argmax(dim=1)  # 예측 클래스 인덱스
        
        # targets와 predictions의 크기가 일치하는지 확인
        # if predictions.size(0) != targets.size(0):
        #     raise ValueError(f"Prediction size {predictions.size(0)} does not match target size {targets.size(0)}")

        acc = (targets == predictions).sum().item() / targets.size(0) * 100  # 배치 크기로 나누기
        
        
        return acc