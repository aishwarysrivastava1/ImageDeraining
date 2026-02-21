"""
GTAV-NightRain Dataset Loader for NDR-Restore
Handles set1, set2, and set3 with _00 suffix removal for GT matching
Place at: NDR-Restore/data/Dataset_gtavnightrain.py

GTAV-NightRain structure:
- set1/test/rainy/ (50 images: 0000_00.png to 0049_00.png)
- set1/test/gt/ (50 GT: 0000.png to 0049.png)
- set2/test/rainy/ (50 images: 0000_00.png to 0049_00.png)
- set2/test/gt/ (50 GT: 0000.png to 0049.png)
- set3/test/rainy/ (186 images: 0000_00.png to 0185_00.png)
- set3/test/gt/ (186 GT: 0000.png to 0185.png)

Resolution: 1920×1080 (Full HD) - Will be resized to prevent OOM
"""

import cv2
import torch
import numpy as np
from torch.utils import data as data
from pathlib import Path
import torch.nn.functional as F


class Dataset_GTAVNightRain(data.Dataset):
    """
    GTAV-NightRain dataset loader with automatic resizing
    
    Dataset sets:
    - set1: 50 test images (night rain with default rain asset)
    - set2: 50 test images (night rain with custom rain asset)
    - set3: 186 test images (night rain with heavy rain mod)
    
    All images: 1920×1080 → Resized to prevent OOM on 8GB GPU
    """

    def __init__(self, opt):
        super(Dataset_GTAVNightRain, self).__init__()

        self.opt = opt
        self.gt_paths = []
        self.lq_paths = []
        
        # Get dataset set from config
        if 'dataset_set' in self.opt:
            dataset_set = self.opt['dataset_set']
        else:
            # Try to infer from name
            name = self.opt.get('name', '').lower()
            if 'set1' in name:
                dataset_set = 'set1'
            elif 'set2' in name:
                dataset_set = 'set2'
            elif 'set3' in name:
                dataset_set = 'set3'
            else:
                dataset_set = 'set1'  # Default
        
        self.dataset_set = dataset_set
        
        # Root directory: D:/Datasets/GTAV-NightRain/
        gtav_base = Path('D:/Datasets/GTAV-NightRain')
        
        # Construct paths
        root_dir = gtav_base / dataset_set / 'test'
        rainy_dir = root_dir / 'rainy'
        gt_dir = root_dir / 'gt'
        
        if not rainy_dir.exists():
            raise ValueError(f"Rainy directory not found: {rainy_dir}")
        if not gt_dir.exists():
            raise ValueError(f"GT directory not found: {gt_dir}")
        
        # CRITICAL: Resize Full HD (1920×1080) to prevent OOM
        # Resize to 960×540 (50% reduction, maintains 16:9 aspect ratio)
        self.target_size = (960, 540)  # (width, height)
        
        print(f"\n[GTAV-NightRain-{dataset_set}] Loading from: {root_dir}")
        print(f"  Images will be resized to: {self.target_size[0]}×{self.target_size[1]} for memory optimization")
        
        # Get all rainy images (sorted numerically)
        rainy_files = sorted([f for f in rainy_dir.iterdir() 
                             if f.suffix.lower() == '.png'],
                            key=lambda x: int(x.stem.split('_')[0]))
        
        print(f"  Found {len(rainy_files)} rainy images in rainy/")
        
        # Match with GT images (remove _00 suffix)
        for rainy_file in rainy_files:
            # GT matching: 0000_00.png → 0000.png
            # Extract base number: 0000_00 → 0000
            base_name = rainy_file.stem.replace('_00', '')
            gt_file = gt_dir / f"{base_name}.png"
            
            if gt_file.exists():
                self.lq_paths.append(str(rainy_file))
                self.gt_paths.append(str(gt_file))
            else:
                print(f"  Warning: GT not found for {rainy_file.name} (expected {gt_file.name})")
        
        print(f"\n[GTAV-NightRain-{dataset_set}] Successfully loaded {len(self.lq_paths)} image pairs")
        
        if len(self.lq_paths) == 0:
            raise ValueError(f"No valid image pairs found in {root_dir}")
        
        assert len(self.gt_paths) == len(self.lq_paths), \
            f"GT and LQ count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}"

    def __getitem__(self, index):
        # Load images
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        # CRITICAL: Resize Full HD (1920×1080) → 960×540 to prevent OOM
        img_lq = cv2.resize(img_lq, self.target_size, interpolation=cv2.INTER_AREA)
        img_gt = cv2.resize(img_gt, self.target_size, interpolation=cv2.INTER_AREA)

        # Ensure 3 channels
        img_lq = img_lq[:, :, :3]
        img_gt = img_gt[:, :, :3]

        # Transpose to CHW
        img_lq = img_lq.transpose(2, 0, 1)
        img_gt = img_gt.transpose(2, 0, 1)

        # Convert to tensors
        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)).float()

        # Add padding (8-pixel alignment)
        img_lq = img_lq.unsqueeze(0)
        h, w = img_lq.shape[-2:]
        pad_h = 8 - h % 8 if h % 8 != 0 else 0
        pad_w = 8 - w % 8 if w % 8 != 0 else 0

        img_lq = F.pad(img_lq, (0, pad_w, 0, pad_h), mode='reflect')
        img_lq = img_lq.squeeze(0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': self.lq_paths[index],
            'gt_path': self.gt_paths[index]
        }

    def __len__(self):
        return len(self.lq_paths)