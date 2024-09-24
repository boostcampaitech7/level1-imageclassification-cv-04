# convert data folder more human readable
import os
from data_dict import data_dict

def rename_folders(base_path):
    for folder_name in os.listdir(base_path):
        if folder_name in data_dict:
            old_path = os.path.join(base_path, folder_name)
            new_name = data_dict[folder_name]
            new_path = os.path.join(base_path, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {folder_name} -> {new_name}")
            except OSError as e:
                print(f"Error renaming {folder_name}: {e}")
        else:
            print(f"No mapping found for: {folder_name}")

if __name__ == "__main__":
    base_path = "./data/train"
    rename_folders(base_path)
    print("Folder renaming complete.")