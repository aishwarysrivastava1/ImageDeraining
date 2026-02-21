"""
RainDS Dataset Loader for NDR-Restore (CORRECTED VERSION)
Place at: NDR-Restore/data/Dataset_rainds.py

Handles exact naming conventions:
- RainDS_real: Natural numbers (2-218, not continuous)
- RainDS_syn: Multiple naming patterns (rd-, rain-, rd-rain-, pie-variants)
"""

import cv2
import torch
import numpy as np
from torch.utils import data as data
from pathlib import Path
import glob
import os
import torch.nn.functional as F


class Dataset_RainDS(data.Dataset):
    """
    RainDS dataset loader for both RainDS_real and RainDS_syn
    
    Root: D:/Deraining/RLP/rlp/data/RainDS/
    
    RainDS_real structure:
        RainDS_real/test_set/
        ├── gt/ (2.png, 3.png, ..., 218.png)
        ├── raindrop/ (2.png, 3.png, ..., 218.png)
        ├── rainstreak/ (2.png, 3.png, ..., 218.png)
        └── rainstreak_raindrop/ (2.png, 3.png, ..., 218.png)
    
    RainDS_syn structure:
        RainDS_syn/test_set/
        ├── gt/ (norain-X.png, pie-norain-X.png)
        ├── raindrop/ (rd-X.png, pie-rd-X.png)
        ├── rainstreak/ (rain-X.png, pie-rain-X.png)
        └── rainstreak_raindrop/ (rd-rain-X.png, pie-rd-rain-X.png)
    """

    def __init__(self, opt):
        super(Dataset_RainDS, self).__init__()

        self.opt = opt
        self.gt_paths = []
        self.lq_paths = []
        self.rain_types = []
        
        # Get dataset variant from path
        if 'dataset_variant' in self.opt:
            dataset_variant = self.opt['dataset_variant']
        else:
            # Auto-detect from path (check if it contains 'real' or 'syn')
            dataset_variant = 'real' if 'real' in str(self.opt.get('name', '')).lower() else 'syn'
        
        # Root directory: D:/Deraining/RLP/rlp/data/RainDS/
        rainds_base = Path('D:/Deraining/RLP/rlp/data/RainDS')
        
        # Construct full path
        if dataset_variant == 'real':
            root_dir = rainds_base / 'RainDS_real' / 'test_set'
            self.dataset_name = 'RainDS_real'
        else:
            root_dir = rainds_base / 'RainDS_syn' / 'test_set'
            self.dataset_name = 'RainDS_syn'
        
        gt_dir = root_dir / 'gt'
        
        if not gt_dir.exists():
            raise ValueError(f"GT directory not found: {gt_dir}")
        
        print(f"\n[{self.dataset_name}] Loading from: {root_dir}")
        
        # Rain type folders
        rain_type_folders = ['raindrop', 'rainstreak', 'rainstreak_raindrop']
        
        for rain_type in rain_type_folders:
            rain_type_dir = root_dir / rain_type
            
            if not rain_type_dir.exists():
                print(f"Warning: {rain_type} directory not found at {rain_type_dir}, skipping...")
                continue
            
            # Get all rainy images
            rainy_files = sorted([f for f in rain_type_dir.iterdir() 
                                 if f.suffix.lower() in ['.png', '.jpg']])
            
            print(f"  Found {len(rainy_files)} images in {rain_type}/")
            
            for rainy_file in rainy_files:
                # Find corresponding GT
                gt_file = self._find_gt(rainy_file, gt_dir, self.dataset_name)
                
                if gt_file is not None and gt_file.exists():
                    self.lq_paths.append(str(rainy_file))
                    self.gt_paths.append(str(gt_file))
                    self.rain_types.append(rain_type)
                else:
                    # Print warning for missing GT
                    print(f"  Warning: GT not found for {rainy_file.name}")
        
        print(f"\n[{self.dataset_name}] Successfully loaded {len(self.lq_paths)} image pairs:")
        print(f"  - Raindrop: {self.rain_types.count('raindrop')}")
        print(f"  - Rainstreak: {self.rain_types.count('rainstreak')}")
        print(f"  - Rainstreak+Raindrop: {self.rain_types.count('rainstreak_raindrop')}")
        
        if len(self.lq_paths) == 0:
            raise ValueError(f"No valid image pairs found in {root_dir}")
        
        assert len(self.gt_paths) == len(self.lq_paths), \
            f"GT and LQ count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}"

    def _find_gt(self, rainy_file, gt_dir, dataset_name):
        """
        Find GT for rainy image based on dataset naming convention
        
        RainDS_real: 2.png → 2.png (same name)
        RainDS_syn conversions:
            - rd-5.png → norain-5.png
            - rain-5.png → norain-5.png
            - rd-rain-5.png → norain-5.png
            - pie-rd-5.png → pie-norain-5.png
            - pie-rain-5.png → pie-norain-5.png
            - pie-rd-rain-5.png → pie-norain-5.png
        """
        filename = rainy_file.stem  # Without extension
        ext = rainy_file.suffix
        
        if dataset_name == 'RainDS_real':
            # RainDS_real: Same filename in GT
            gt_filename = filename + ext
        
        else:  # RainDS_syn
            # Handle all RainDS_syn naming patterns
            
            # Pattern 1: pie-rd-rain-X → pie-norain-X
            if filename.startswith('pie-rd-rain-'):
                number = filename.replace('pie-rd-rain-', '')
                gt_filename = f'pie-norain-{number}{ext}'
            
            # Pattern 2: pie-rain-X → pie-norain-X
            elif filename.startswith('pie-rain-'):
                number = filename.replace('pie-rain-', '')
                gt_filename = f'pie-norain-{number}{ext}'
            
            # Pattern 3: pie-rd-X → pie-norain-X
            elif filename.startswith('pie-rd-'):
                number = filename.replace('pie-rd-', '')
                gt_filename = f'pie-norain-{number}{ext}'
            
            # Pattern 4: rd-rain-X → norain-X
            elif filename.startswith('rd-rain-'):
                number = filename.replace('rd-rain-', '')
                gt_filename = f'norain-{number}{ext}'
            
            # Pattern 5: rain-X → norain-X
            elif filename.startswith('rain-'):
                number = filename.replace('rain-', '')
                gt_filename = f'norain-{number}{ext}'
            
            # Pattern 6: rd-X → norain-X
            elif filename.startswith('rd-'):
                number = filename.replace('rd-', '')
                gt_filename = f'norain-{number}{ext}'
            
            else:
                # Unknown pattern - try same filename
                print(f"    Unknown naming pattern: {filename}")
                gt_filename = filename + ext
        
        gt_path = gt_dir / gt_filename
        return gt_path if gt_path.exists() else None

    def __getitem__(self, index):
        # Load images
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        # CRITICAL: Resize RainDS_real images to prevent OOM
        # RainDS_real: 1296x728 → 640x360
        # RainDS_syn: already 640x360
        if self.dataset_name == 'RainDS_real':
            img_lq = cv2.resize(img_lq, (640, 360), interpolation=cv2.INTER_AREA)
            img_gt = cv2.resize(img_gt, (640, 360), interpolation=cv2.INTER_AREA)

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
            'gt_path': self.gt_paths[index],
            'rain_type': self.rain_types[index]
        }

    def __len__(self):
        return len(self.lq_paths)