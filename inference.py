import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.utils import inference
from src.layer_modification import layer_modification
import numpy as np

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"
    save_result_path = "./train_result"

    test_info = pd.read_csv(testdata_info_file)
    num_classes = 500

    # Set up transform
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # Set up test dataset and dataloader
    test_dataset = CustomDataset(root_dir=testdata_dir, info_df=test_info, transform=test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    # Load model
        # Load model 1
    model_selector_1 = ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    model_1 = model_selector_1.get_model()
    model_1 = layer_modification(model_1)
    model_1_path = os.path.join("./train_result", "best_model.pt")
    model_1.load_state_dict(torch.load(model_1_path, map_location=device))
    model_1.to(device)
    # Run inference
    predictions_model_1 = inference(model=model_1, device=device, test_loader=test_loader)
    np.save('prewitt_data.npy',predictions_model_1)
    print("1 save")

    # Save results
    # test_info['target'] = predictions
    # test_info = test_info.reset_index().rename(columns={"index": "ID"})
    # test_info.to_csv("output.csv", index=False)
    # print("Inference completed and results saved to output.csv")

if __name__ == "__main__":
    main()