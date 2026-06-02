# MFDNet: Multi-Scale Fusion and Decomposition Network for Single Image Deraining

<div align="center">

Implementation repository for **MFDNet** with a practical training/evaluation pipeline across multiple state-of-the-art deraining datasets.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-11.x-76b900)](#installation)
[![Status](https://img.shields.io/badge/Project-Research%20Implementation-success)](#)

</div>

---

## Overview

This repository contains an implementation of **MFDNet** for single-image deraining, adapted for experimentation on different modern deraining benchmarks.

- Implements core model in `MFDNet.py`
- Includes training pipeline in `train.py`
- Includes dataset-specific testing and evaluation scripts
- Organizes utility functions in `utils/`

Reference implementation source: [qwangg/MFDNet](https://github.com/qwangg/MFDNet)  
This repo is an implementation and extension-oriented working version based on the corresponding research direction.

---

## Repository Structure

```text
MFDNet/
├── MFDNet.py                  # Main network architecture
├── restormer_block.py         # Building blocks used by the network
├── train.py                   # Training script
├── losses.py                  # Charbonnier, edge, and related losses
├── SSIM.py                    # SSIM loss/metric module
├── dataset_gtrain.py          # Training dataset loader
├── dataset_gtav.py            # GTAV test loader
├── dataset_outdoor.py         # OutdoorRain test loader
├── dataset_rainds.py          # RainDS test loader
├── dataset_realrain.py        # RealRain test loader
├── test_*.py                  # Dataset-specific inference scripts
├── evaluate_*.py              # Dataset-specific PSNR/SSIM evaluation scripts
├── utils/                     # File, image, model, and dataset utilities
├── mfd.yml                    # Conda environment file
├── requirement.txt            # Pip dependencies snapshot
└── README.md
```

---

## Installation

### Option A: Conda (recommended)

```bash
conda env create -f mfd.yml
conda activate mfd
```

### Option B: Pip

```bash
pip install -r requirement.txt
```

> Recommended runtime: NVIDIA GPU + CUDA-enabled PyTorch.

---

## Datasets

This implementation is designed around the following datasets/scripts:

- **GTRain** (training + testing)
- **GTAV-NightRain**
- **OutdoorRain**
- **RainDS**
- **RealRain**

Place datasets in your local data root and pass the paths explicitly with command-line arguments (examples below).

---

## Training

Train on GTRain with custom paths and hyperparameters:

```bash
python train.py \
  --train_dir "./data/GT-RAIN_train" \
  --epochs 100 \
  --batch_size 4 \
  --lr 4e-4 \
  --patch_size 128 \
  --save_freq 50 \
  --gpus 0
```

Useful flags:

- `--resume` to continue from latest checkpoint in `logs/`
- `--save_freq` to control periodic checkpoint saving

---

## Inference

Generated images are saved to `results/MFDNet/<DatasetName>/` (created automatically when scripts run).

---

## Evaluation (PSNR / SSIM)

After inference, compute metrics using dataset-specific evaluators:

```bash
python evaluate_gtrain.py   --result_dir "./results/MFDNet/GTRain"   --gt_dir "D:/Deraining/RLP/rlp/data/GT-RAIN_test"
python evaluate_gtav.py     --result_dir "./results/MFDNet/GTAV"     --gt_dir "D:/Deraining/RLP/rlp/data/GTAV-NightRain"
python evaluate_outdoor.py  --result_dir "./results/MFDNet/Outdoor"  --gt_dir "D:/Deraining/RLP/rlp/data/Outdoor"
python evaluate_rainds.py   --result_dir "./results/MFDNet/RainDS"   --gt_dir "D:/Deraining/RLP/rlp/data/RainDS"
python evaluate_realrain.py --result_dir "./results/MFDNet/RealRain" --gt_dir "D:/Deraining/RLP/rlp/data/RealRain"
```

Metric CSVs are written to `results/MFDNet/`.

---

## Citation and Credits

If you use this repository in research or applied work, please cite the original MFDNet paper and acknowledge the upstream implementation:

- Upstream repository: [qwangg/MFDNet](https://github.com/qwangg/MFDNet)

- Some scripts ship with Windows-style default paths; override them with your local directories.
- For reproducible results, keep dataset organization consistent with script expectations.
- For publication-grade reporting, use a fixed train/val/test split and record checkpoint + config used for each run.
