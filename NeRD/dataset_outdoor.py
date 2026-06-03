import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
print('OUTDOOR DATASET FILE USED:', __file__)
class DatasetTest(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.samples = []
        self.target_size = (640, 360)
        self._collect()
    def _collect(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f'Test directory not found: {self.root_dir}')
        test_path = os.path.join(self.root_dir, 'input')
        if not os.path.exists(test_path):
            raise ValueError(f'Test set directory not found: {test_path}')
        files = sorted([f for f in os.listdir(test_path) if f.endswith('.png') or f.endswith('.jpg')])
        for f in files:
            self.samples.append((os.path.join(test_path, f), f[:-4]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.target_size, Image.LANCZOS)
        return TF.to_tensor(img), img_name
