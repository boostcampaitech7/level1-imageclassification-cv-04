import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def inference(model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().detach().numpy())
    
    return predictions