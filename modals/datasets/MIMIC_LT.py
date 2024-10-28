import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class MIMICLT(Dataset):
    def __init__(self, root_dir, mode='train', augmentations=None):
        """
        Args:
            root_dir (string): Directory with all the images and CSV files.
            mode (string): One of 'train', 'valid', or 'test' to specify the dataset split.
            augmentations (callable, optional): Optional augmentation to be applied
            on a sample.
        """
        #tmp fix
        label_root_dir = '/orange/bianjiang/tienyu/MIMIC_CXR/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-1.1.0/cxr-lt-2023/'
        if mode == 'train':
            csv_file = os.path.join(label_root_dir, 'train.csv')
        elif mode == 'valid':
            csv_file = os.path.join(label_root_dir, 'development.csv')
        elif mode == 'test':
            csv_file = os.path.join(label_root_dir, 'test.csv')
        else: 
            raise ValueError(f"Invalid mode: {mode}. Please choose one of 'train', 'valid', or 'test'.")
        self.annotations = pd.read_csv(csv_file)
        self.classes = list(self.annotations.columns[6:])
        self.root_dir = root_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.annotations['path'][idx])
        image = Image.open(img_name).convert('RGB')
        labels = self.annotations.iloc[idx, 6:].values

        if self.augmentations:
            image = self.augmentations(image)

        return image, labels