import os
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

print("GTAV-NIGHTRAIN DATASET FILE USED:", __file__)


class DatasetTest(Dataset):
    def __init__(self, root_dir, test_set='set1'):
        super().__init__()
        self.root_dir = root_dir
        self.test_set = test_set
        self.samples = []
        self.target_size = (960, 540)
        self._collect()

    def _collect(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Test directory not found: {self.root_dir}")
        
        test_path = os.path.join(self.root_dir, self.test_set, 'input')
        if not os.path.exists(test_path):
            raise ValueError(f"Test set directory not found: {test_path}")
            
        files = sorted([f for f in os.listdir(test_path) if f.endswith('.png')])
        for f in files:
            img_path = os.path.join(test_path, f)
            img_name = f[:-4]
            self.samples.append((img_path, img_name))
            
        print(f"Loaded {len(self.samples)} test images from {test_path}")
        print(f"Images will be resized to: {self.target_size[0]}×{self.target_size[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.target_size, Image.LANCZOS)
        img = TF.to_tensor(img)
        
        return img, img_name