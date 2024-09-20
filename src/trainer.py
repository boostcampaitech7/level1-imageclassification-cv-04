import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(outputs, targets)

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
        wrong_path: str=None
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
        self.wrong_path = wrong_path

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

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self, log_wrong_predictions = False) -> float:
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                if log_wrong_predictions:
                    self.save_wrong_image(self.wrong_path, images, outputs, targets)
        return total_loss / len(self.val_loader)

    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self.train_epoch()
            if epoch < self.epochs - 1:
                val_loss = self.validate()
            else:
                val_loss = self.validate(log_wrong_predictions=True)
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