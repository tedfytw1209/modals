import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import datasets
import torchvision.transforms as transforms

class MIMICLT(Dataset):
    def __init__(self, root_dir, mode='train',transfroms=[], augmentations=[],label_transfroms=[]):
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
        self.num_class = len(self.classes)
        image_names, labels = self.annotations['path'], self.annotations.iloc[:, 6:].values
        self.image_names = [image_name for image_name in image_names]
        self.labels = [label for label in labels]
        
        self.channel = 3
        self.root_dir = root_dir
        self.transfroms = transfroms
        self.augmentations = augmentations
        self.label_transfroms = label_transfroms
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = self.loader(img_name)
        labels = self.labels[idx]
        for transfrom in self.transfroms:
            image = transfrom(image)
        for augmentation in self.augmentations:
            image = augmentation(image)
        for label_trans in self.label_transfroms:
            labels = label_trans(labels)

        return image, labels

if __name__ == "__main__":
    root = '/red/bianjiang/VLM_dataset/ReportGeneration/MIMIC-CXR_JPG/'
    dataset = MIMICLT(root_dir=root, mode='train', transfroms=[transforms.ToTensor()],label_transfroms=[torch.tensor])
    print(len(dataset))
    sample = dataset[0]
    print(sample[0].shape,sample[0].mean())