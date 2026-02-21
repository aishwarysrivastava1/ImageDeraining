"""
GT-Rain Dataset Loader for NDR-Restore
Place at: NDR-Restore/data/Dataset_gtrain.py
"""

import cv2
import torch
import numpy as np
from torch.utils import data as data
from pathlib import Path
import glob
import os
import torch.nn.functional as F


class Dataset_GTRain(data.Dataset):
    """
    GT-Rain dataset loader
    
    Structure:
    GT-RAIN_test/
    ├── scene_1/
    │   ├── scene_1-R-000.png (rainy)
    │   ├── scene_1-R-001.png
    │   └── scene_1-C-000.png (clean GT)
    └── scene_2/
        └── ...
    """

    def __init__(self, opt):
        super(Dataset_GTRain, self).__init__()

        self.opt = opt
        self.gt_paths = []
        self.lq_paths = []
        
        # Collect GT-Rain test images
        root_dir = Path(self.opt['gtrain_root'])
        
        for scene_dir in sorted(root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            # Get all rainy images (-R- pattern)
            rainy_files = sorted([f for f in scene_dir.iterdir() 
                                 if '-R-' in f.name and f.suffix.lower() == '.png'])
            
            for rainy_file in rainy_files:
                # Find corresponding GT
                gt_file = self._find_gt(rainy_file, scene_dir)
                
                if gt_file is not None:
                    self.lq_paths.append(str(rainy_file))
                    self.gt_paths.append(str(gt_file))
        
        print(f"[GT-Rain] Successfully loaded {len(self.lq_paths)} image pairs")
        
        assert len(self.gt_paths) == len(self.lq_paths), \
            f"GT and LQ count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}"

    def _find_gt(self, rainy_path, scene_dir):
        """Find GT for rainy image"""
        # Get all clean files
        clean_files = sorted([f for f in scene_dir.iterdir() 
                             if '-C-' in f.name and f.suffix.lower() == '.png'])
        
        if len(clean_files) == 0:
            return None
        
        # Extract pattern from rainy file
        rainy_name = rainy_path.stem  # e.g., scene_1-R-000 or scene_1-Webcam-R-000
        
        # Try exact match with -C-
        gt_name = rainy_name.replace('-R-', '-C-')
        gt_path = scene_dir / f"{gt_name}.png"
        
        if gt_path.exists():
            return gt_path
        
        # Try with -C-000 (single GT case)
        base_name = rainy_name.split('-R-')[0]
        gt_path = scene_dir / f"{base_name}-C-000.png"
        
        if gt_path.exists():
            return gt_path
        
        # Fallback: use first clean file
        return clean_files[0] if clean_files else None

    def __getitem__(self, index):
        # Load images
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

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
