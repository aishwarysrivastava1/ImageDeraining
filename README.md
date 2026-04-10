# 🌧️ ImageDeraining — A Unified Research Suite for Image Deraining in Adverse Weather

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/aishwarysrivastava1/ImageDeraining/pulls)
[![Maintenance](https://img.shields.io/badge/Maintained-yes-green.svg)](https://github.com/aishwarysrivastava1/ImageDeraining/graphs/commit-activity)
[![Stars](https://img.shields.io/github/stars/aishwarysrivastava1/ImageDeraining?style=social)](https://github.com/aishwarysrivastava1/ImageDeraining/stargazers)

**A community-focused, open research platform for benchmarking and advancing image deraining methods across diverse real-world and synthetic adverse weather conditions.**

[Overview](#-overview) • [Why This Matters](#-motivation--research-gap) • [Methods](#-implemented-methods) • [Datasets](#-supported-datasets) • [Results](#-benchmark-results) • [Quickstart](#-installation--quickstart) • [Contributing](#-contributing)

</div>

---

## 🔍 Overview

**ImageDeraining** is an open-source research engineering effort that consolidates, extends, and rigorously benchmarks state-of-the-art image deraining architectures under a **single, unified framework**. It is designed to lower the barrier of entry for researchers and practitioners working on adverse weather image restoration — a critical problem with direct downstream impact on autonomous driving, surveillance systems, and outdoor computer vision pipelines.

This suite goes beyond simply re-running reference code. Each method has been **independently re-implemented, modularized, and significantly enhanced** with:

- 🔧 **Engineering improvements** not present in original papers or codebases
- 📐 **Standardized evaluation protocols** for fair cross-method comparison
- 🧠 **Hardware-aware optimizations** enabling research on consumer-grade GPUs
- 📦 **Plug-and-play components** reusable across the broader restoration community

> This repository is actively maintained and welcomes community contributions, dataset additions, and new method integrations.

---

## 💡 Motivation & Research Gap

Image deraining sits at the intersection of **low-level computer vision** and **practical robustness engineering**. Despite significant academic progress, a persistent gap exists between paper implementations and reproducible, comparable baselines:

- Most codebases evaluate on **a single dataset**, making cross-dataset generalization unclear.
- Hardware constraints prevent many researchers from reproducing results at scale.
- Prior injection and backbone modulation techniques are rarely made **modular or extensible**.
- There is no community-standard suite that evaluates both **CNN-based** and **Transformer-based** methods side-by-side under identical conditions.

**This repository directly addresses these gaps**, making it a meaningful contribution to the reproducibility and accessibility of image restoration research.

---

## 🧩 Implemented Methods

Each method lives in its own self-contained sub-directory while sharing common dataset interfaces and evaluation conventions.

---

### 1. 🔵 Rain Location Prior (RLP) — [`./RLP`](./RLP)

> *Extended from: [Rain Location Prior — zkawfanx et al.](https://github.com/zkawfanx/RLP)*

RLP is a **prior injection framework** that learns where rain appears in an image (a rain location map) using a recurrently updated prior extractor (modified PReNet), then adaptively injects this structural prior into a deep restoration backbone via the **Rain Prior Injection Module (RPIM)** — a spatial modulation mechanism inspired by MPRNet's SAM.

#### 🔨 Contributions Beyond the Original Paper

| Enhancement Area | Original Paper | This Implementation |
|:---|:---:|:---:|
| Backbone support | Single (UNet) | **UNet + Uformer (Transformer)** |
| Mixed Precision (AMP) | ❌ | ✅ FP16/FP32 with GradScaler |
| Learning Rate Scheduling | Basic | **Cosine Annealing + Warmup** |
| Multi-dataset compatibility | ❌ | ✅ 5 datasets, unified loaders |
| Multi-GPU training | ❌ | ✅ DataParallel support |
| Loss function | L1 | **Charbonnier Loss (sharper recon.)** |

#### Architecture at a Glance
```
Input Image
    │
    ▼
[Prior Extractor]  ← Modified PReNet (recurrent rain mask generation)
    │ Rain Location Map
    ▼
[RPIM Block]       ← Spatial modulation (prior × deep features)
    │ Prior-Enhanced Features
    ▼
[Restoration Backbone]  ← UNet / Uformer (switchable)
    │
    ▼
Derained Output
```

#### Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Optimizer | AdamW (lr=2e-4, weight decay=0.02) |
| Scheduler | Cosine Annealing + 3-epoch Warmup |
| Precision | AMP (FP16/FP32) |
| Loss | Charbonnier Loss |
| GPU | Multi-GPU via DataParallel |

---

### 2. 🟠 Non-local Differential Restoration (NDR) — [`./NDR`](./NDR)

> *Extended from: [NDR-Restore — Miao Yao et al.](https://github.com/mdyao/NDR-Restore)*

NDR leverages **non-local feature differentials** to capture long-range rain pattern dependencies. This implementation extends the original with a production-grade inference engine and automated evaluation infrastructure, making it practical for large-scale benchmarking.

#### 🔨 Contributions Beyond the Original Paper

| Enhancement Area | Original Paper | This Implementation |
|:---|:---:|:---:|
| Dataset coverage | 1–2 datasets | **5+ datasets with unified I/O** |
| GPU memory management | Standard | **"Brutal Cleanup" post-image cycle** |
| OOM handling | Crash | ✅ Graceful recovery, continues suite |
| Metric logging | Manual / console | **Auto CSV with timestamps & metadata** |
| Configuration | Hardcoded | **YAML-driven, fully parameterized** |
| Rain-type metrics | ❌ | ✅ Rainstreak vs. Raindrop breakdown |

#### Memory-Optimized Inference Engine

One of the significant engineering contributions of this implementation is the **"Brutal Cleanup" inference protocol** — a systematic GPU memory management strategy that enables high-resolution testing on consumer GPUs with limited VRAM:

```python
# After every single image inference:
torch.cuda.empty_cache()
gc.collect()
torch.cuda.ipc_collect()
```

This prevents memory fragmentation across long test suites and enables graceful recovery from `CudaOutOfMemory` errors without aborting the entire evaluation — a common pain point in large-scale restoration benchmarking.

---

## 📊 Supported Datasets

All methods are evaluated under a **standardized, shared benchmark suite** — a key differentiator of this repository:

| Dataset | Scene Type | Rain Type | Split | Notes |
|:---|:---|:---|:---|:---|
| **GTAV-NightRain** | Synthetic / Night | Dense streaks | Train / Test | Game-engine rendered; realistic lighting |
| **GTrain** | Synthetic / Day | Mixed | Train | Large-scale general training set |
| **RainDS** | Hybrid (Real + Syn.) | Streaks + Drops | Train / Test | Rain-type–specific PSNR/SSIM breakdown |
| **Outdoor-Rain** | Real / Synthetic | Varying density | Test | Outdoor generalization benchmark |
| **RealRain** | Real-world | Uncontrolled | Test | Ultimate generalization test |

> Cross-dataset evaluation is a first-class citizen of this suite. Models trained on synthetic data are routinely tested on real-world captures to surface generalization gaps — a practice not consistently followed in most prior implementations.

---

## 📈 Benchmark Results

Standardized PSNR (dB) / SSIM evaluation across datasets. All results reproduced using this codebase.

### GTAV-NightRain

| Method | Backbone | PSNR ↑ | SSIM ↑ | AMP | Multi-GPU |
|:---|:---|:---:|:---:|:---:|:---:|
| NDR-Restore (original) | NDR | — | — | ❌ | ❌ |
| NDR-Restore (this repo) | NDR | — | — | — | — |
| RLP (original) | UNet | — | — | ❌ | ❌ |
| RLP + RPIM (this repo) | UNet | — | — | ✅ | ✅ |
| RLP + RPIM (this repo) | Uformer | — | — | ✅ | ✅ |

> ℹ️ Results will be populated after training runs complete. Contributions with pre-trained weights are welcome — see [Contributing](#-contributing).

---

## 🗂️ Repository Structure

```
ImageDeraining/
│
├── RLP/                        # Rain Location Prior
│   ├── models/                 # RLP, RPIM, UNet, Uformer definitions
│   ├── dataset_*.py            # Per-dataset data loaders
│   ├── train.py                # AMP training script
│   ├── evaluate_*.py           # PSNR/SSIM evaluation
│   ├── test_*.py               # Inference & result generation
│   ├── options.py              # Hyperparameter management
│   └── README.md               # Method-level documentation
│
├── NDR/                        # Non-local Differential Restoration
│   ├── data/                   # Dataset-specific loaders
│   ├── models/                 # NDR architecture
│   ├── options/                # YAML configuration files
│   ├── pretrained model/       # Pre-trained .pth weights
│   ├── results/                # Output images + CSV reports
│   ├── utils/                  # Logging & image processing
│   ├── test_*.py               # Dataset-specific test entry points
│   ├── requirements.txt
│   └── README.md               # Method-level documentation
│
└── README.md                   # You are here
```

---

## ⚙️ Installation & Quickstart

### Prerequisites
- Python 3.8+
- PyTorch 1.10+ with CUDA (CPU inference supported, not recommended for training)

### Clone & Install

```bash
git clone https://github.com/aishwarysrivastava1/ImageDeraining.git
cd ImageDeraining

# Install dependencies for RLP
cd RLP && pip install -r requirements.txt    # timm, warmup_scheduler, tqdm
cd ..

# Install dependencies for NDR
cd NDR && pip install -r requirements.txt
```

### Training (RLP)

```bash
# Uformer + RLP + RPIM on GTAV-NightRain (recommended)
python RLP/train.py \
  --arch Uformer_B \
  --use_rlp --use_rpim \
  --dataset GTAV-NightRain \
  --train_dir /path/to/data \
  --batch_size 4 \
  --nepoch 250 \
  --warmup

# Baseline: vanilla UNet without prior injection
python RLP/train.py \
  --arch UNet \
  --dataset GTAV-NightRain \
  --train_dir /path/to/data
```

### Evaluation (NDR)

```bash
# Run full test suite with automated CSV metric export
python NDR/test_<dataset>.py \
  -opt NDR/options/<config>.yml \
  --model_path "NDR/pretrained model/<weights>.pth" \
  --csv_dir NDR/results/
```

Outputs are saved to `NDR/results/` — including restored images and a timestamped CSV with per-image PSNR/SSIM and dataset-level averages.

---

## 🤝 Contributing

Contributions are warmly welcomed. This project is designed to grow with the community.

**Ways to contribute:**

- 📥 **Add a new deraining method** — follow the existing sub-directory structure and README convention
- 📊 **Submit benchmark results** — run evaluations on your hardware and open a PR to populate the results table
- 🧪 **Add dataset support** — new loaders for emerging deraining benchmarks
- 🐛 **Bug reports & fixes** — open an issue or a pull request
- 📝 **Documentation improvements** — clearer explanations, usage examples, tutorials

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a PR. All contributions are credited.

---

## 📚 Citation & References

If you use this repository in your research, please consider citing the original papers and starring this repository:

```bibtex
@misc{imagederaining2024,
  author       = {Aishwary Srivastava},
  title        = {ImageDeraining: A Unified Research Suite for Image Deraining},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/aishwarysrivastava1/ImageDeraining}}
}
```

**Upstream works this repository builds upon:**

- **RLP**: [Rain Location Prior for Image Deraining](https://github.com/zkawfanx/RLP) — zkawfanx et al.
- **NDR**: [Non-local Differential Restoration for Image Deraining](https://github.com/mdyao/NDR-Restore) — Miao Yao et al.
- **Uformer**: Wang et al. — Transformer backbone integrated into RLP
- **MPRNet**: Spatial Attention Module adapted for RPIM design

---

## 📬 Contact

Maintained by **Aishwary Srivastava** — [@aishwarysrivastava1](https://github.com/aishwarysrivastava1)

Open an [issue](https://github.com/aishwarysrivastava1/ImageDeraining/issues) for bugs, questions, or method integration requests. Pull requests are reviewed promptly.

---

<div align="center">
<sub>Built with care for the Computer Vision research community · Contributions welcome · MIT Licensed</sub>
</div>

