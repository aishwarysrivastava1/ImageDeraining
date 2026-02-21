"""
RealRain Dataset Loader for NDR-Restore
Handles RealRain-1k-H, RealRain-1k-L, and SynRain-13k
Place at: NDR-Restore/data/Dataset_realrain.py

RealRain structure:
- RealRain-1k-H/test/input/ (rainy images)
- RealRain-1k-H/test/target/ (GT images)
- RealRain-1k-L/test/input/ (rainy images)
- RealRain-1k-L/test/target/ (GT images)
- SynRain-13k/test/input/ (rainy images)
- SynRain-13k/test/target/ (GT images)

Images: Natural numbers (non-continuous), GT has same name as rainy
Average resolution: 1512×973 (RealRain-1k), 482×420 (SynRain-13k)
"""

import cv2
import torch
import numpy as np
from torch.utils import data as data
from pathlib import Path
import torch.nn.functional as F


class Dataset_RealRain(data.Dataset):
    """
    RealRain dataset loader with automatic resizing for memory optimization
    
    Dataset variants:
    - RealRain-1k-H: Heavy rain (224 test images, avg 1512×973)
    - RealRain-1k-L: Light rain (224 test images, avg 1512×973)
    - SynRain-13k: Synthetic (1200 test images, 482×420)
    """

    def __init__(self, opt):
        super(Dataset_RealRain, self).__init__()

        self.opt = opt
        self.gt_paths = []
        self.lq_paths = []
        
        # Get dataset variant from config
        if 'dataset_variant' in self.opt:
            dataset_variant = self.opt['dataset_variant']
        else:
            # Try to infer from name
            name = self.opt.get('name', '').lower()
            if 'realrain-1k-h' in name or 'h' in name:
                dataset_variant = 'RealRain-1k-H'
            elif 'realrain-1k-l' in name or 'l' in name:
                dataset_variant = 'RealRain-1k-L'
            elif 'synrain' in name:
                dataset_variant = 'SynRain-13k'
            else:
                dataset_variant = 'RealRain-1k-L'  # Default
        
        self.dataset_variant = dataset_variant
        
        # Root directory: D:/Datasets/RealRain/
        realrain_base = Path('D:/Datasets/RealRain')
        
        # Construct paths based on variant
        if dataset_variant == 'RealRain-1k-H':
            root_dir = realrain_base / 'RealRain-1k' / 'RealRain-1k-H' / 'test'
            self.target_size = (768, 512)  # Resize to prevent OOM
            self.dataset_name = 'RealRain-1k-H'
        elif dataset_variant == 'RealRain-1k-L':
            root_dir = realrain_base / 'RealRain-1k' / 'RealRain-1k-L' / 'test'
            self.target_size = (768, 512)  # Resize to prevent OOM
            self.dataset_name = 'RealRain-1k-L'
        elif dataset_variant == 'SynRain-13k':
            root_dir = realrain_base / 'SynRain-13k' / 'test'
            self.target_size = None  # No resize needed (already small)
            self.dataset_name = 'SynRain-13k'
        else:
            raise ValueError(f"Unknown dataset variant: {dataset_variant}")
        
        input_dir = root_dir / 'input'
        target_dir = root_dir / 'target'
        
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        if not target_dir.exists():
            raise ValueError(f"Target directory not found: {target_dir}")
        
        print(f"\n[{self.dataset_name}] Loading from: {root_dir}")
        if self.target_size:
            print(f"  Images will be resized to: {self.target_size[0]}×{self.target_size[1]} for memory optimization")
        
        # Get all rainy images (sorted by natural number)
        rainy_files = sorted([f for f in input_dir.iterdir() 
                             if f.suffix.lower() in ['.png', '.jpg']],
                            key=lambda x: int(x.stem) if x.stem.isdigit() else int(''.join(filter(str.isdigit, x.stem))))
        
        print(f"  Found {len(rainy_files)} rainy images in input/")
        
        # Match with GT images (same name)
        for rainy_file in rainy_files:
            gt_file = target_dir / rainy_file.name
            
            if gt_file.exists():
                self.lq_paths.append(str(rainy_file))
                self.gt_paths.append(str(gt_file))
            else:
                print(f"  Warning: GT not found for {rainy_file.name}")
        
        print(f"\n[{self.dataset_name}] Successfully loaded {len(self.lq_paths)} image pairs")
        
        if len(self.lq_paths) == 0:
            raise ValueError(f"No valid image pairs found in {root_dir}")
        
        assert len(self.gt_paths) == len(self.lq_paths), \
            f"GT and LQ count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}"

    def __getitem__(self, index):
        # Load images
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        # CRITICAL: Resize high-resolution images to prevent OOM
        # RealRain-1k: 1512×973 → 768×512
        # SynRain-13k: 482×420 → no resize
        if self.target_size is not None:
            h, w = img_lq.shape[:2]
            # Maintain aspect ratio
            scale = min(self.target_size[0] / w, self.target_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_gt = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_AREA)

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