<div align="center">
  
# 🌧️ NeRD-Rain Extended

**State-of-the-Art Neural Representation for Image Deraining**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An extended and robust implementation of **[NeRD: Neural Representation for Rain Drop Removal](https://github.com/cschenxiang/NeRD-Rain)**, specifically optimized and expanded for evaluating state-of-the-art image deraining datasets.

</div>

<br>

## 📖 Overview

This repository extends the original NeRD multi-scale transformer-based architecture by providing a unified, clean, and GitHub-ready framework for training and evaluating deraining models across a diverse array of modern rain datasets. It leverages Implicit Neural Representations (INR) combined with spatial transformers to effectively restore rain-corrupted images.

### ✨ Key Features
- **Multi-Scale Transformer Architecture:** High-fidelity image restoration using multi-scale embeddings and fusion mechanisms.
- **Extensive Dataset Support:** Out-of-the-box data loading, testing, and evaluation scripts for leading datasets:
  - `GTAV-NightRain`
  - `GT-RAIN`
  - `Outdoor-Rain`
  - `RainDS`
  - `RealRain`
- **Comprehensive Evaluation:** Dedicated evaluation scripts computing `PSNR` and `SSIM` metrics seamlessly, saving results to organized CSV reports.
- **Optimized Training:** Features gradual warmup learning rates, robust loss functions (Charbonnier, Edge, FFT, L1), and gradient accumulation support.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NeRD-Rain-Extended.git
   cd NeRD-Rain-Extended
   ```

2. **Install dependencies:**
   Make sure you have PyTorch installed with CUDA support. Then, install the remaining requirements:
   ```bash
   pip install torch torchvision numpy opencv-python scikit-image einops kornia tqdm
   ```

*(Note: This repository includes a customized `pytorch-gradual-warmup-lr` as a submodule/local directory. You do not need to install it separately).*

---

## 🚀 Usage

### 1. Training the Model
To train the model on a specific dataset (e.g., GT-RAIN), run the main `train.py` script. The script is configured to use a hybrid loss function and cosine annealing with warm restarts.

```bash
python train.py \
    --train_dir /path/to/dataset/train \
    --epochs 60 \
    --batch_size 2 \
    --patch_size 128 \
    --lr 1e-4 \
    --gpus "0"
```
Checkpoints will be saved automatically to the `./logs` directory.

### 2. Testing
Test scripts generate restored images from your trained weights. There is a dedicated test script for each dataset to handle specific dataset folder structures natively.

Example for GTAV:
```bash
python test_gtav.py \
    --input_dir /path/to/GTAV-NightRain \
    --result_dir ./results/NeRD/GTAV \
    --weights ./logs/model_latest.pth \
    --gpus "0"
```

### 3. Evaluation
After testing, run the corresponding evaluation script to compute quantitative metrics (PSNR, SSIM) against the ground truth target images.

```bash
python evaluate_gtav.py \
    --result_dir ./results/NeRD/GTAV \
    --gt_dir /path/to/GTAV-NightRain \
    --csv_dir ./results/NeRD
```
The results will be logged to the console and automatically exported as a timestamped `.csv` file.

---

## 📁 Repository Structure

```text
📦 NeRD-Rain-Extended
 ┣ 📂 pytorch-gradual-warmup-lr/ # Warmup scheduler dependency
 ┣ 📂 utils/                  # Helper utilities (I/O, image metrics, dir management)
 ┣ 📜 train.py                # Main training loop
 ┣ 📜 model.py                # Core NeRD architecture (MultiscaleNet)
 ┣ 📜 mlp.py                  # Implicit Neural Representation (INR) components
 ┣ 📜 losses.py               # Edge, FFT, and Charbonnier loss functions
 ┣ 📜 nerd_inference.py       # Reusable model inference and windowing logic
 ┣ 📜 dataset_*.py            # Dataloaders for specific datasets
 ┣ 📜 test_*.py               # Inference scripts per dataset
 ┗ 📜 evaluate_*.py           # PSNR/SSIM evaluation scripts per dataset
```

---

## 🙏 Acknowledgements

This project is built upon the foundational research of [NeRD-Rain](https://github.com/cschenxiang/NeRD-Rain). I extend my gratitude to the original authors for their pioneering work in Neural Representations for Rain Drop Removal.

If you use this codebase, please ensure you cite the original NeRD paper.

