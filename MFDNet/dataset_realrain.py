import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

print("REALRAIN DATASET FILE USED:", __file__)

class DatasetTest(Dataset):
    def __init__(self, root_dir, variant='realrain_h'):
        super().__init__()
        self.root_dir = root_dir
        self.variant = variant
        self.samples = []
        
        if 'realrain_h' in variant.lower() or 'realrain_l' in variant.lower():
            self.target_size = (768, 512)
            self.resize = True
        else:
            self.target_size = None
            self.resize = False
        
        self._collect()
    
    def _collect(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Test directory not found: {self.root_dir}")
        
        if 'realrain_h' in self.variant.lower():
            input_dir = os.path.join(self.root_dir, 'RealRain-1k', 'RealRain-1k-H', 'test', 'input')
        elif 'realrain_l' in self.variant.lower():
            input_dir = os.path.join(self.root_dir, 'RealRain-1k', 'RealRain-1k-L', 'test', 'input')
        elif 'synrain' in self.variant.lower():
            input_dir = os.path.join(self.root_dir, 'SynRain-13k', 'test', 'input')
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        
        for f in files:
            img_path = os.path.join(input_dir, f)
            img_name = f[:-4]
            self.samples.append((img_path, img_name))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        if self.resize:
            img = img.resize(self.target_size, Image.LANCZOS)
        
        img = TF.to_tensor(img)
        
        return img, img_name
