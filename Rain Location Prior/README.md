# Rain Location Prior (RLP) for Image Deraining

This repository contains an advanced implementation of the **Rain Location Prior (RLP)** framework for image deraining, as described in the research paper [Rain Location Prior](https://github.com/zkawfanx/RLP). This implementation extends the original work with modular architecture support, hardware-aware optimizations, and a unified pipeline for multiple state-of-the-art datasets.

## 🚀 Key Features

*   **Modular RLP & RPIM**: 
    *   **Rain Location Prior (RLP)**: Extracts precise rain-related information using a recurrently updated prior map (modified PReNet).
    *   **Rain Prior Injection Module (RPIM)**: A sophisticated modulation module (inspired by SAM from MPRNet) that adaptively injects the rain prior into the deep restoration backbone.
*   **Versatile Backbone Support**: Seamlessly integrate RLP with different architectures:
    *   **UNet**: A classic encoder-decoder for efficient restoration.
    *   **Uformer**: A cutting-edge transformer-based restoration network.
*   **Multi-Dataset Implementation**: Ready-to-use scripts and data loaders for:
    *   **GTAV-NightRain**: Synthetic night rain dataset.
    *   **RainDS**: Comprehensive rainy day and night series.
    *   **Real Rain & Outdoor Rain**: Benchmarks for real-world generalization.
*   **Performance Optimizations**:
    *   **Mixed Precision Training (AMP)**: Uses `torch.cuda.amp` to reduce training time and VRAM usage.
    *   **Advanced Scheduling**: Warmup periods with Cosine Annealing for stable convergence.
    *   **Loss Function**: Optimized with **Charbonnier Loss** for sharper reconstruction.

## 🛠️ Architecture Overview

The core model `RLP_NightRain` (defined in `models/rlp.py`) is designed to be plug-and-play. It consists of:
1.  **Prior Extractor**: A modified PReNet that generates a rain mask from the input image.
2.  **Modulation**: The RPIM block that refines the prior and original features.
3.  **Restoration Module**: The main deraining network (UNet or Uformer) which receives the prior-enhanced features.

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+ (CUDA support recommended)
- `pip install -r requirements.txt` (including `timm`, `warmup_scheduler`, `tqdm`)

## 🚄 Usage

### 1. Data Preparation
Organize your datasets in the following structure:
```text
/datasets/
  └── GTAV-NightRain/
      ├── train/
      │   ├── input/
      │   └── gt/
      └── test/
```

### 2. Training
Run `train.py` with your desired configuration.

**Train Uformer with RLP and RPIM:**
```bash
python train.py --arch Uformer_B --use_rlp --use_rpim --dataset GTAV-NightRain --train_dir /path/to/data --batch_size 4 --nepoch 250 --warmup
```

**Train Vanilla UNet:**
```bash
python train.py --arch UNet --dataset GTAV-NightRain --train_dir /path/to/data
```

### 3. Evaluation
Use the dataset-specific evaluation scripts for accurate metrics (PSNR, SSIM).

**Evaluate on RainDS:**
```bash
python evaluate_rainds.py --dataset RainDS --weights /path/to/model.pth
```

## 📊 Technical Specifications & Optimizations

| Specification | Implementation Detail |
| :--- | :--- |
| **Optimizer** | AdamW with Weight Decay (0.02) |
| **Scheduler** | Cosine Annealing with 3-epoch Warmup |
| **Precision** | FP16/FP32 Mixed Precision (AMP) |
| **Loss** | Charbonnier Loss (L1-proxy) |
| **GPU Support** | Multi-GPU support via `DataParallel` |

## 📂 Repository Structure

- `models/`: Architecture definitions (RLP, RPIM, UNet, Uformer).
- `dataset_*.py`: Specialized data loaders for each dataset benchmark.
- `train.py`: Central training script with AMP and modular logging.
- `options.py`: Configuration management and hyperparameters.
- `evaluate_*.py`: Evaluation logic for PSNR/SSIM calculation.
- `test_*.py`: Testing scripts for visual result generation.

## 📝 Acknowledgements
This implementation is based on the research work of the RLP paper authors. Special thanks to the open-source community for providing the `Uformer` and `MPRNet` components integrated into this project.
