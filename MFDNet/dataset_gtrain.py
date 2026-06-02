import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

print("GTRain DATASET FILE USED:", __file__)

class DatasetTrain(Dataset):
    def __init__(self, root_dir, img_options=None):
        super().__init__()
        self.root_dir = root_dir
        self.img_options = img_options or {}
        self.patch_size = self.img_options.get("patch_size", 256)

        self.image_pairs = []
        self._collect_pairs()

    def _collect_pairs(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Dataset directory not found: {self.root_dir}")

        scene_dirs = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        for scene in sorted(scene_dirs):
            scene_path = os.path.join(self.root_dir, scene)
            files = os.listdir(scene_path)

            rainy_files = sorted(f for f in files if "-R-" in f and f.endswith(".png"))
            clean_files = sorted(f for f in files if "-C-" in f and f.endswith(".png"))

            if not rainy_files or not clean_files:
                continue

            # Case 1: single clean image
            if len(clean_files) == 1:
                clean_path = os.path.join(scene_path, clean_files[0])
                for rf in rainy_files:
                    self.image_pairs.append((
                        os.path.join(scene_path, rf),
                        clean_path,
                        scene
                    ))
            else:
                # Case 2: match by frame index
                clean_map = {
                    f.split("-C-")[1].split(".")[0]: f
                    for f in clean_files
                }

                for rf in rainy_files:
                    idx = rf.split("-R-")[1].split(".")[0]
                    if idx in clean_map:
                        self.image_pairs.append((
                            os.path.join(scene_path, rf),
                            os.path.join(scene_path, clean_map[idx]),
                            scene
                        ))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rainy_path, clean_path, scene = self.image_pairs[idx]

        rainy = TF.to_tensor(Image.open(rainy_path).convert("RGB"))
        clean = TF.to_tensor(Image.open(clean_path).convert("RGB"))

        rainy, clean = self._augment(rainy, clean)
        return rainy, clean, scene

    def _augment(self, rainy, clean):
        ps = self.patch_size
        _, H, W = rainy.shape

        if H < ps or W < ps:
            rainy = TF.resize(rainy, (max(ps, H), max(ps, W)))
            clean = TF.resize(clean, (max(ps, H), max(ps, W)))

        i = random.randint(0, rainy.shape[1] - ps)
        j = random.randint(0, rainy.shape[2] - ps)

        rainy = TF.crop(rainy, i, j, ps, ps)
        clean = TF.crop(clean, i, j, ps, ps)

        if random.random() > 0.5:
            rainy = TF.hflip(rainy)
            clean = TF.hflip(clean)

        if random.random() > 0.5:
            rainy = TF.vflip(rainy)
            clean = TF.vflip(clean)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            rainy = TF.rotate(rainy, angle)
            clean = TF.rotate(clean, angle)

        return rainy, clean

class DatasetTest(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.samples = []
        self._collect()

    def _collect(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Test directory not found: {self.root_dir}")

        for scene in sorted(os.listdir(self.root_dir)):
            scene_path = os.path.join(self.root_dir, scene)
            if not os.path.isdir(scene_path):
                continue

            for f in sorted(os.listdir(scene_path)):
                if "-R-" in f and f.endswith(".png"):
                    self.samples.append((
                        os.path.join(scene_path, f),
                        f"{scene}_{f[:-4]}"
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        pad_w = (8 - W % 8) % 8
        pad_h = (8 - H % 8) % 8
        if pad_w != 0 or pad_h != 0:
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect')
        
        rainy = TF.to_tensor(img)
        return rainy, name
