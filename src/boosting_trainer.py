from src.trainer import Trainer
import torch
from tqdm.auto import tqdm
import numpy as np
from src.models import ModelSelector
from src.layer_modification import layer_modification
import os
from src.freeze import freeze

class BoostingTrainer(Trainer):
    def __init__(self, *args, num_models=3, init_boosting_factor = 1, fix_boosting_factor = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_models = num_models # 앙상블에 사용할 모델 수
        #### 임시
        # model_selector = ModelSelector(model_type='timm', num_classes=500, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
        # model = model_selector.get_model()
        # model = layer_modification(model)
        # self.model_path = os.path.join("./train_result/baseth ensemble model", "best_model.pt")
        # model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        # model.to(self.device)
        # self.models = [model]
        
        ####
        self.models = [self.model] # 현재 담겨 있는 self.model이 Base Model이 될 예정
        self.init_boosting_factor = init_boosting_factor
        self.fix_boosting_factor = fix_boosting_factor
        self.model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
        
    def boost_weights(self, outputs, targets):
        '''
        틀린 예측에 가중치를 더하기 위한 함수
        '''
        _, preds = torch.max(outputs, 1)
        weights = torch.ones_like(targets, dtype = torch.float)
        if self.fix_boosting_factor:
            weights[preds != targets] = self.init_boosting_factor
        else:
            weights[preds != targets] = self.set_boosting_factor()
        return weights
    
    def set_boosting_factor(self):
        return np.exp(1/(len(self.models)+0.5))
        
    
    def train_epoch_boosting(self, model, previous_model = None):
        '''
        부스팅을 적용한 학습 (1 에폭)
        '''
        model.train()
        total_loss = 0.0
        acc_cum = 0
        progress_bar = tqdm(self.train_loader, desc='Training with Boosting')
        # print(len(self.models), 'Exists' if previous_model is not None else 'None') # 나중에 지울 부분
        model.to(self.device)
        for i,(images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # 현재 모델 예측
            outputs = model(images)
            acc_cum += self.accuracy(outputs, targets)
            # 틀린 예측에 대한 가중치 적용
            if previous_model is not None: # 이전 모델이 존재하는 경우. 즉, Base 모델이 아닌 경우에 대해서만 Penalty 계산 후, 적용
                previous_model.eval()
                with torch.no_grad():
                    prev_outputs = previous_model(images)
                    # print('이전 모델의 정확도:',self.accuracy(prev_outputs, targets))
                    penalty_weights = self.boost_weights(prev_outputs, targets)
                loss = self.loss_fn(outputs, targets) * penalty_weights.to(self.device)
                loss = loss.mean()
            else:
                loss = self.loss_fn(outputs, targets)
                    
            # print('현재 모델의 정확도:', self.accuracy(outputs, targets))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), acc = acc_cum / (i+1))
            
        print(f"Train Accuracy (Boosting Applied): {acc_cum / len(self.train_loader)}%")
        return total_loss / len(self.train_loader)
    
    def train_boosting(self):
        '''
        부스팅을 적용하여 여러 모델을 학습
        '''
        # 첫 번째 모델을 학습
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self.train_epoch_boosting(self.models[0])
            val_loss = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss: .4f}, Validation Loss: {val_loss: .4f}\n")
            self.save_model(epoch, val_loss, '0')
            self.scheduler.step()
            
        # 부스팅을 적용해 추가 모델 학습
        for i in range(1, self.num_models):
            print(f"\nTraining Boosted Model {i+1}")
            new_model = self._clone_model(self.models[-1], i) # 새로운 모델은 이전 모델과 동일하게 생성

            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs} for Boosted Model {i+1}")
                train_loss = self.train_epoch_boosting(new_model, self.models[-1])
                    
                    
                val_loss = self.validate()
                print(f"Boosted Model {i+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")
                self.save_model(epoch, val_loss, i)
                self.scheduler.step()
            
            self.models.append(new_model)
            
            
            
    def _clone_model(self, model,i):
        '''
        모델을 복제하기 위한 함수
        i: ith ensemble model
        '''
        model_selector = ModelSelector(model_type='timm', num_classes=500, model_name= self.model_name, pretrained=True)
        cloned_model = model_selector.get_model()
        cloned_model = layer_modification(model)
        # cloned_model.load_state_dict(model.state_dict()) # 파라미터를 복사
        model_path = os.path.join(f"./train_result/{i}th ensemble model", "best_model.pt")
        cloned_model.load_state_dict(torch.load(model_path, map_location=self.device))
        cloned_model.to(self.device)
        return cloned_model
    
    
    def validate_ensemble(self):
        best_models = []
        # Load all models
        for i in range(self.num_models):
            model = model_selector = ModelSelector(model_type='timm', num_classes=500, model_name= self.model_name, pretrained=False)
            model = model_selector.get_model()
            model = layer_modification(model)
            # Load the best model from ./train_result/best_model.pt
            model_path = os.path.join(f"./train_result/{i}th model", "best_model.pt")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            best_models.append(model)
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc = 'Final Validation', leave = False)
            all_predictions = []
            all_labels = []
        
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Initialize output with zeros
                outputs = torch.zeros((images.size(0), 500), device=self.device)
                
                # Accumulate predictions from all models
                for i in range(self.num_models):
                    outputs += best_models[i](images)
                
                # Average the predictions
                outputs /= self.num_models
                
                # Store predictions and labels
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
            
            # Concatenate all predictions and labels
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # Calculate accuracy
            acc = self.accuracy(all_predictions, all_labels)
            print(f'Ensemble Accuracy: {acc:.2f}%')
        
        return acc
                
    def save_model(self, epoch, loss, num):
        '''
        모델을 저장하기 위한 함수
        num: 몇 번째 앙상블 모델인지 확인 가능한 인자
        '''
        self.result_path = os.path.join(self.result_path, f'{num}th ensemble model')
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