import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class MIMICLTDataset(Dataset):
    def __init__(self, root_dir, mode='train', augmentation=None):
        """
        Args:
            root_dir (string): Directory with all the images and CSV files.
            mode (string): One of 'train', 'val', or 'test' to specify the dataset split.
            augmentation (callable, optional): Optional augmentation to be applied
            on a sample.
        """
        if mode == 'train':
            csv_file = os.path.join(root_dir, 'train.csv')
        elif mode == 'val':
            csv_file = os.path.join(root_dir, 'val.csv')
        elif mode == 'test':
            csv_file = os.path.join(root_dir, 'test.csv')
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.augmentation = augmentation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.annotations['path'][idx])
        image = Image.open(img_name).convert('RGB')
        labels = self.annotations.iloc[idx, 6:].values

        if self.augmentation:
            image = self.augmentation(image)

        return image, labels