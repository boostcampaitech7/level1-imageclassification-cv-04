import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

def inference(model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
    model.to(device)
    model.eval()
    
    all_predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            logits = F.softmax(logits, dim=1)
    all_predictions.append(logits.cpu().detach().numpy())
    predictions = np.concatenate(all_predictions, axis = 0)
    
    return predictions