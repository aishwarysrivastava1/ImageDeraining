import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

print("RAINDS DATASET FILE USED:", __file__)

def get_gt_filename_syn(input_filename, rain_type):
    name = input_filename

    if rain_type == 'raindrop':
        if name.startswith('pie-rd-'):
            return 'pie-norain-' + name[len('pie-rd-'):]
        elif name.startswith('rd-'):
            return 'norain-' + name[len('rd-'):]
    elif rain_type == 'rainstreak':
        if name.startswith('pie-rain-'):
            return 'pie-norain-' + name[len('pie-rain-'):]
        elif name.startswith('rain-'):
            return 'norain-' + name[len('rain-'):]
    elif rain_type == 'rainstreak_raindrop':
        if name.startswith('pie-rd-rain-'):
            return 'pie-norain-' + name[len('pie-rd-rain-'):]
        elif name.startswith('rd-rain-'):
            return 'norain-' + name[len('rd-rain-'):]

    raise ValueError(f"Cannot derive GT filename from '{input_filename}' for rain_type='{rain_type}'")

class DatasetTest(Dataset):
    def __init__(self, root_dir, variant='rainds_real', rain_type='all'):
        super().__init__()
        self.root_dir = root_dir
        self.variant = variant
        self.rain_type = rain_type
        self.samples = []
        
        self.target_size = (640, 360)
        
        self._collect()
    
    def _collect(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Test directory not found: {self.root_dir}")
        
        if 'real' in self.variant.lower():
            test_dir = os.path.join(self.root_dir, 'RainDS_real', 'test_set')
            is_syn = False
        elif 'syn' in self.variant.lower():
            test_dir = os.path.join(self.root_dir, 'RainDS_syn', 'test_set')
            is_syn = True
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")

        gt_dir = os.path.join(test_dir, 'gt')
        if not os.path.exists(gt_dir):
            raise ValueError(f"GT directory not found: {gt_dir}")

        rain_dirs = []
        if self.rain_type == 'all':
            rain_dirs = ['raindrop', 'rainstreak', 'rainstreak_raindrop']
        elif self.rain_type == 'raindrop':
            rain_dirs = ['raindrop']
        elif self.rain_type == 'rainstreak':
            rain_dirs = ['rainstreak']
        elif self.rain_type == 'rainstreak_raindrop':
            rain_dirs = ['rainstreak_raindrop']
        
        for rain_dir in rain_dirs:
            rain_path = os.path.join(test_dir, rain_dir)
            if not os.path.exists(rain_path):
                continue
            
            files = sorted([f for f in os.listdir(rain_path) if f.endswith('.png') or f.endswith('.jpg')])
            
            for f in files:
                img_path = os.path.join(rain_path, f)
                img_name = f[:-4]
                full_name = f"{rain_dir}_{img_name}"

                if is_syn:
                    gt_filename = get_gt_filename_syn(f, rain_dir)
                else:
                    gt_filename = f

                gt_path = os.path.join(gt_dir, gt_filename)

                self.samples.append((img_path, gt_path, full_name, rain_dir))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, gt_path, img_name, rain_type = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.target_size, Image.LANCZOS)
        img = TF.to_tensor(img)

        gt = Image.open(gt_path).convert("RGB")
        gt = gt.resize(self.target_size, Image.LANCZOS)
        gt = TF.to_tensor(gt)

        return img, gt, img_name
