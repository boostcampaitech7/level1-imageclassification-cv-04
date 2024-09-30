import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.utils import inference
from src.layer_modification import layer_modification

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
    
    # # Load model 1
    # model_selector_1 = ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_1 = model_selector_1.get_model()
    # model_1 = layer_modification(model_1)
    # model_1_path = os.path.join("./train_result", "best_model.pt")
    # model_1.load_state_dict(torch.load(model_1_path, map_location=device))
    # model_1.to(device)
    # # Run inference
    # predictions_model_1 = inference(model=model_1, device=device, test_loader=test_loader)
    # np.save('prewitt_data.npy',predictions_model_1)
    # print("1 save")
    # predictions_model_1 = torch.tensor(predictions_model_1, dtype=torch.float32, device=device)

    # # Load model 2
    # model_selector_2 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_2 = model_selector_2.get_model()
    # model_2 = layer_modification(model_2)
    # model_2_path = os.path.join("./train_result", "model_adamw.pt")
    # model_2.load_state_dict(torch.load(model_2_path, map_location=device))
    # model_2.to(device)
    # # Run inference
    # predictions_model_2 = inference(model=model_2, device=device, test_loader=test_loader)
    # np.save('adamW.npy',predictions_model_2)
    # print("2 save")
    # predictions_model_2 = torch.tensor(predictions_model_2, dtype=torch.float32, device=device)

    # # Load model 3
    # model_selector_3 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_3 = model_selector_3.get_model()
    # model_3 = layer_modification(model_3)
    # model_3_path = os.path.join("./train_result", "2nd_ensemble_model_until_7epoch_best_model.pt")
    # model_3.load_state_dict(torch.load(model_3_path, map_location=device))
    # model_3.to(device)
    # # Run inference
    # predictions_model_3 = inference(model=model_3, device=device, test_loader=test_loader)
    # np.save('ensemble_data.npy',predictions_model_3)
    # predictions_model_3 = torch.tensor(predictions_model_3, dtype=torch.float32, device=device)
    # print("3 save")


    # # Load model 4
    # model_selector_4 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_4 = model_selector_4.get_model()
    # model_4 = layer_modification(model_4)
    # model_4_path = os.path.join("./train_result", "Canny.pt")
    # model_4.load_state_dict(torch.load(model_4_path, map_location=device))
    # model_4.to(device)
    # # Run inference
    # predictions_model_4 = inference(model=model_4, device=device, test_loader=test_loader)
    # np.save('canny.npy',predictions_model_4)
    # print("4 save")
    # predictions_model_4 = torch.tensor(predictions_model_4, dtype=torch.float32, device=device)


    # # Load model 5
    # model_selector_5 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_5 = model_selector_5.get_model()
    # model_5 = layer_modification(model_5)
    # model_5_path = os.path.join("./train_result", "aug5_best_model.pt")
    # model_5.load_state_dict(torch.load(model_5_path, map_location=device))
    # model_5.to(device)
    # # Run inference
    # predictions_model_5 = inference(model=model_5, device=device, test_loader=test_loader)
    # np.save('cutmix_data.npy',predictions_model_5)
    # print("5 save")
    # predictions_model_5 = torch.tensor(predictions_model_5, dtype=torch.float32, device=device)

    # #Load model 6
    # model_selector_6 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_6 = model_selector_6.get_model()
    # model_6 = layer_modification(model_6)
    # model_6_path = os.path.join("./train_result", "eva02_labelsmoothing_0.1_batchsize_64.pt")
    # model_6.load_state_dict(torch.load(model_6_path, map_location=device))
    # model_6.to(device)
    # # Run inference
    # predictions_model_6 = inference(model=model_6, device=device, test_loader=test_loader)
    # np.save('labelsmoothing.npy',predictions_model_6)
    # print("6 save")
    # #predictions_model_6 = torch.tensor(predictions_model_6, dtype=torch.float32, device=device)
    
    # #Load model 7
    # model_selector_7 =  ModelSelector(model_type='timm', num_classes=num_classes, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False)
    # model_7 = model_selector_7.get_model()
    # model_7 = layer_modification(model_7)
    # model_7_path = os.path.join("./train_result", "model_eva02_lr_0.0001.pt")
    # model_7.load_state_dict(torch.load(model_7_path, map_location=device))
    # model_7.to(device)
    # # Run inference
    # predictions_model_7 = inference(model=model_7, device=device, test_loader=test_loader)
    # np.save('lr_0.0001.npy',predictions_model_7)
    # print("7 save")
    # predictions_model_7 = torch.tensor(predictions_model_7, dtype=torch.float32, device=device)

    predictions_model_1 = np.load('adamW.npy')
    print(predictions_model_1.shape)
    
    predictions_model_2 = np.load('cutmix_data.npy')
    print(predictions_model_2.shape)
    
    #predictions_model_3 = np.load('ensemble_data.npy')
    #print(predictions_model_3.shape)
    
    predictions_model_4 = np.load('labelsmoothing.npy')
    print(predictions_model_4.shape)
    
    #predictions_model_5 = np.load('lr_0.0001.npy')
    #print(predictions_model_5.shape)
    
    predictions_model_6 = np.load('canny_prewitt_data.npy')
    print(predictions_model_6.shape)
    
    predictions_model_7 = np.load('prewitt_data.npy')
    print(predictions_model_7.shape)
    #predictions_model_8 = np.load('new_ensemble.npy')
    #print(predictions_model_8.shape)

    # # Soft Voting 수행
    soft_voting = (predictions_model_1 + predictions_model_2 + 
                    predictions_model_4 + predictions_model_6 +
                    predictions_model_7) / 5
    print("Soft voting shape:", soft_voting.shape)

    soft_voting = torch.tensor(soft_voting, dtype=torch.float32)
    voting_result = soft_voting.argmax(dim=1)
    print("Voting result shape:", voting_result.shape)

    # Save results
    test_info['target'] = voting_result.tolist()
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output_soft_voting.csv", index=False)
    print("Inference completed and results saved to output.csv")

    # # # 결과 비교
    # # matches = (voting_result == model_2_result).sum().item()
    # # total = voting_result.numel()
    # # print(f"Matches with model_2: {matches}/{total} ({matches/total*100:.2f}%)")
if __name__ == "__main__":
    main()