import os
from dataset import CustomDataset
import pandas as pd
from transforms import TransformSelector
def label_maker():
    labels = []
    predictions = []
    wrong_dir = '../data/train/n02123045'
    image_path = [x for x in os.listdir('../data/train/n02123045') if x.startswith('sketch')]
    # for name in os.listdir(wrong_dir):
    #     n = name.split('_')
    #     labels.append(n[1])
    #     predictions.append(n[-1].split('.')[0])
    transform_selector = TransformSelector(transform_type="albumentations")
    train_transform = transform_selector.get_transform(is_train=True)
    labels = ['n02123045'] * len(image_path)
    df = pd.DataFrame({'image_path': image_path, 'target':labels})
    data = CustomDataset(root_dir = wrong_dir, info_df = df, is_inference = False, transform = train_transform)
    return data