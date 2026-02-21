"""
Outdoor-Rain Dataset Loader for NDR-Restore
Handles multiple rainy images per single ground truth
Place at: NDR-Restore/data/Dataset_outdoor.py

Outdoor-Rain structure:
- gt/ (600 GT images: im_0001.png to im_0600.png)
- input/ (9000 rainy images: im_0001_s80_a04.png, im_0001_s80_a05.png, etc.)

Naming convention:
- Rainy: im_0001_s80_a04.png → GT: im_0001.png
- Rainy: im_0042_s95_a06.png → GT: im_0042.png
- Match by first 7 characters: "im_0001" → "im_0001.png"

Paper: 9000 training samples, 1500 validation samples
"""

import cv2
import torch
import numpy as np
from torch.utils import data as data
from pathlib import Path
import torch.nn.functional as F


class Dataset_Outdoor(data.Dataset):
    """
    Outdoor-Rain dataset loader
    
    Features:
    - Multiple rainy images per single GT
    - Prefix matching (first 7 chars)
    - Auto-resizing for memory optimization
    - Paper: Heavy rain with accumulation effects
    """

    def __init__(self, opt):
        super(Dataset_Outdoor, self).__init__()

        self.opt = opt
        self.gt_paths = []
        self.lq_paths = []
        
        # Root directory: D:/Datasets/Outdoor/
        outdoor_base = Path('D:/Datasets/Outdoor')
        
        gt_dir = outdoor_base / 'gt'
        input_dir = outdoor_base / 'input'
        
        if not gt_dir.exists():
            raise ValueError(f"GT directory not found: {gt_dir}")
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # CRITICAL: Resize for memory optimization
        # Paper doesn't specify resolution, but outdoor images are typically high-res
        # Assuming 720×480 to 1024×768 range, resize to 640×480
        self.target_size = (640, 480)  # (width, height)
        
        print(f"\n[Outdoor-Rain] Loading from: {outdoor_base}")
        print(f"  Images will be resized to: {self.target_size[0]}×{self.target_size[1]} for memory optimization")
        
        # Get all GT images (sorted)
        gt_files = sorted([f for f in gt_dir.iterdir() 
                          if f.suffix.lower() == '.png'],
                         key=lambda x: int(x.stem.split('_')[1]) if len(x.stem.split('_')) > 1 else 0)
        
        print(f"  Found {len(gt_files)} GT images in gt/")
        
        # Get all rainy images
        rainy_files = sorted([f for f in input_dir.iterdir() 
                             if f.suffix.lower() == '.png'])
        
        print(f"  Found {len(rainy_files)} rainy images in input/")
        
        # Match rainy images to GT by prefix (first 7 characters)
        # Example: im_0001_s80_a04.png → prefix "im_0001" → GT im_0001.png
        gt_dict = {f.stem: f for f in gt_files}
        
        matched_count = 0
        unmatched_count = 0
        
        for rainy_file in rainy_files:
            # Extract prefix (first 7 characters: "im_0001")
            rainy_name = rainy_file.stem
            
            # Get prefix: "im_0001_s80_a04" → "im_0001"
            if len(rainy_name) >= 7:
                prefix = rainy_name[:7]  # "im_0001"
                
                # Find matching GT
                if prefix in gt_dict:
                    self.lq_paths.append(str(rainy_file))
                    self.gt_paths.append(str(gt_dict[prefix]))
                    matched_count += 1
                else:
                    unmatched_count += 1
                    if unmatched_count <= 5:  # Show first few warnings
                        print(f"  Warning: No GT found for {rainy_file.name} (prefix: {prefix})")
            else:
                unmatched_count += 1
                if unmatched_count <= 5:
                    print(f"  Warning: Invalid filename format: {rainy_file.name}")
        
        if unmatched_count > 5:
            print(f"  ... and {unmatched_count - 5} more unmatched files")
        
        print(f"\n[Outdoor-Rain] Successfully loaded {len(self.lq_paths)} rainy-GT pairs")
        print(f"  Matched: {matched_count}, Unmatched: {unmatched_count}")
        
        if len(self.lq_paths) == 0:
            raise ValueError(f"No valid image pairs found in {outdoor_base}")
        
        assert len(self.gt_paths) == len(self.lq_paths), \
            f"GT and LQ count mismatch: {len(self.gt_paths)} vs {len(self.lq_paths)}"

    def __getitem__(self, index):
        # Load images
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        # Get original size
        h_orig, w_orig = img_lq.shape[:2]
        
        # CRITICAL: Resize to target size for memory optimization
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