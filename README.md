# 🌧️ ImageDeraining: A Deep Dive into State-of-the-Art Single Image Deraining Architectures

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/aishwarysrivastava1/ImageDeraining?style=social)](https://github.com/aishwarysrivastava1/ImageDeraining/stargazers)

**An extensive, independently developed research repository dedicated to the rigorous exploration, enhancement, and evaluation of leading neural architectures for single-image deraining.**

[Introduction](#-introduction--background) • [Motivation](#-personal-motivation--research-scope) • [Architectural Analysis](#-comprehensive-architectural-analysis--implementations) • [Datasets](#-evaluation-methodology--datasets) • [Usage](#-extensive-setup--usage-guide) • [Acknowledgements](#-acknowledgements--bibliography)

</div>

---

## 📖 Introduction & Background

Adverse weather conditions, particularly rain, introduce severe complex degradations into optical signals. Rain manifests not just as simple, opaque streaks, but as a combination of multi-scale phenomena: dense background streaks (veiling effect), localized foreground raindrops, and atmospheric scattering. This degradation fundamentally impairs the performance of critical downstream computer vision tasks, ranging from autonomous navigation and advanced driver-assistance systems (ADAS) to intelligent video surveillance and outdoor robotics.

Single Image Deraining (SID) is a highly ill-posed inverse problem. The goal is to recover the latent clean background from a single corrupted observation without any temporal cues. While deep learning has revolutionized this field, the transition from theoretical models in academic papers to robust, generalizable tools remains a significant challenge.

This repository represents my comprehensive, personal research effort to understand the intricate mechanics of SOTA deraining networks. It is a documented journey of unravelling complex architectures, pushing their boundaries, and engineering them into a state where they can be evaluated fairly and consistently.

---

## 🎯 Personal Motivation & Research Scope

When exploring the literature on image deraining, I encountered a landscape fragmented by isolated codebases and disparate evaluation metrics. My primary motivation for building this repository was to create a strictly standardized, highly optimized laboratory for my own experiments. 

**Key challenges I aimed to address in my research:**
1. **The Generalization Gap**: Models trained on synthetic data frequently fail on real-world captures. I wanted a framework that seamlessly ran cross-dataset evaluations to measure true generalization.
2. **Hardware & Memory Bottlenecks**: High-resolution inference with modern Transformers often leads to `CUDA OutOfMemory` errors on consumer hardware. I needed to engineer robust memory management protocols.
3. **Architectural Rigidity**: Original implementations are often monolithic. I wanted the flexibility to decouple powerful ideas (like prior injection or non-local differentials) from their original backbones and apply them to modern vision transformers.

**To this end, my scope of work involved:**
- Re-implementing four distinct SOTA architectures from scratch or heavily modifying their upstream source.
- Introducing PyTorch Automatic Mixed Precision (AMP) for accelerated training.
- Integrating complex, hybrid loss functions (Charbonnier, Edge, SSIM, FFT) to drive perceptually superior restoration.
- Building a unified, automated metric tracking system to eliminate human error in benchmarking.

---

## 🧠 Comprehensive Architectural Analysis & Implementations

I selected four distinct architectures that represent different theoretical paradigms in the field of image restoration. I have significantly modified and extended each one.

### 1. Rain Location Prior (RLP) — [`./Rain Location Prior`](./Rain%20Location%20Prior)
*Based on the research by zkawfanx et al.*

**Theoretical Paradigm**: Explicit Prior Injection. 
Instead of forcing the network to blindly learn the mapping from a rainy to a clean image, RLP first explicitly models *where* the rain is. It uses a recurrent sub-network to generate a spatial rain map. This map acts as a prior, which is adaptively injected into the main restoration backbone via a Rain Prior Injection Module (RPIM), guiding the network's attention to heavily corrupted regions.

**My Enhancements & Engineering:**
- **Transformer Backbone Leap**: The original work relied on a standard UNet. I engineered the RPIM to be compatible with **Uformer** (a state-of-the-art hierarchical vision transformer), effectively marrying explicit spatial priors with self-attention mechanisms.
- **Training Acceleration**: I introduced full FP16/FP32 Automatic Mixed Precision via `torch.cuda.amp`.
- **Advanced Optimization**: Replaced basic step-schedulers with Cosine Annealing and Warmup strategies to stabilize transformer training.
- **Loss Formulation**: Transitioned from standard L1 loss to Charbonnier Loss for sharper edge reconstruction.

### 2. Multi-Scale Fusion and Decomposition Network (MFDNet) — [`./MFDNet`](./MFDNet)
*Based on the research by qwangg et al.*

**Theoretical Paradigm**: Multi-Scale Feature Decomposition.
Rain is inherently multi-scale (large foreground drops vs. tiny background streaks). MFDNet tackles this by explicitly decomposing the latent representation into distinct frequency/scale bands. By separating complex rain layers from clean structural background information at various scales, the network achieves highly detailed structural recovery.

**My Enhancements & Engineering:**
- **Codebase Modularization**: I heavily refactored the original monolithic code, isolating the core `MFDNet.py` and `restormer_block.py` components to make them plug-and-play for future experiments.
- **Universal Evaluation Pipeline**: Engineered new data loaders and evaluation scripts (`evaluate_*.py`) to map MFDNet seamlessly to my 5-dataset benchmark suite.
- **Custom Loss Integration**: Implemented a tri-fold loss function explicitly weighing Charbonnier (pixel-wise), Edge (high-frequency structure), and SSIM (perceptual quality).

### 3. Neural Representation for Rain Drop Removal (NeRD) — [`./NeRD`](./NeRD)
*Based on the research by cschenxiang et al.*

**Theoretical Paradigm**: Implicit Neural Representations (INR) + Spatial Transformers.
NeRD is a paradigm shift. Rather than relying solely on discrete pixel grids, it utilizes INRs to learn a continuous, coordinate-based representation of the image. When combined with multi-scale transformers, NeRD demonstrates an exceptional ability to handle massive, non-uniform degradations like heavy raindrops on a lens.

**My Enhancements & Engineering:**
- **Hybrid Multi-Loss Function**: Transformers are notoriously data-hungry and difficult to regularize. I engineered a robust hybrid loss integrating Charbonnier, Edge, L1, and notably FFT (Fast Fourier Transform) loss to enforce consistency in the frequency domain.
- **Stabilized Training Dynamics**: I integrated a customized `pytorch-gradual-warmup-lr` module to implement a gentle warmup phase followed by Cosine Annealing with Restarts, preventing early-stage divergence.
- **Out-of-the-box Generalization**: Built dedicated inference wrappers tailored for immediate, painless testing across synthetic and real-world datasets.

### 4. Non-local Differential Restoration (NDR) — [`./NDR`](./NDR)
*Based on the research by Miao Yao et al.*

**Theoretical Paradigm**: Non-Local Feature Differentials.
Rain streaks often exhibit strong self-similarity across an image (i.e., they fall in the same direction). NDR leverages non-local operations to capture these long-range dependencies, identifying repetitive rain patterns by computing feature differentials across distant spatial regions.

**My Enhancements & Engineering:**
- **The "Brutal Cleanup" Inference Engine**: Non-local operations are extraordinarily memory-intensive. I developed a systematic memory management protocol—forcing strict `torch.cuda.empty_cache()` and `gc.collect()` cycles after every single image during inference. This prevents VRAM fragmentation and allows massive, high-resolution test suites to run on consumer hardware without crashing.
- **Automated Metric Analytics**: Engineered a fully automated reporting engine that dumps timestamped CSVs mapping exact PSNR and SSIM values, categorized by specific rain types.

---

## 📊 Evaluation Methodology & Datasets

To ensure rigorous validation of my enhancements, I evaluate all models against a standardized suite containing five highly diverse datasets. This includes both synthetic datasets (for controlled quantitative analysis) and real-world datasets (for qualitative generalization testing).

| Dataset | Scene Type | Rain Profile | My Specific Use Case in this Research |
|:---|:---|:---|:---|
| **GTAV-NightRain** | Synthetic (Game Engine) | Dense nighttime streaks | Testing restoration performance under the difficult conditions of artificial night lighting and glare. |
| **GTrain** | Synthetic / Day | Mixed | Serving as the massive, primary training corpus for all models to ensure a fair starting point. |
| **RainDS** | Hybrid (Real + Syn.) | Streaks + Drops | Quantifying how well a single network can dynamically handle fundamentally different artifact types simultaneously. |
| **Outdoor-Rain** | Real / Synthetic | Varying density | Benchmarking generalization capabilities on complex outdoor natural scenes. |
| **RealRain** | Real-world | Uncontrolled | The ultimate, unconstrained qualitative test to see if theoretical models actually work in the real world. |

---

## 📈 Quantitative Performance Metrics

All models within this repository are evaluated using standard quantitative image quality metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-wise absolute accuracy.
- **SSIM (Structural Similarity Index)**: Measures the perceived structural and textural integrity of the reconstructed image.

*Detailed numerical benchmarks are actively being compiled into cross-comparative tables across all 5 datasets and will be published in subsequent updates to this repository.*

---

## 🗂️ Detailed Repository Architecture

To maintain sanity while managing four massive neural network architectures, I have enforced a strict, isolated, yet consistent directory structure:

```text
ImageDeraining/
│
├── MFDNet/                     # Multi-Scale Fusion and Decomposition Network
│   ├── MFDNet.py               # Core architecture
│   ├── restormer_block.py      # Fundamental building blocks
│   └── (Dataset loaders, training scripts, evaluation scripts...)
│
├── NeRD/                       # Neural Representation for Rain Drop Removal
│   ├── model.py                # Multi-scale transformer core
│   ├── mlp.py                  # Implicit Neural Representation modules
│   └── (Inference wrappers, warmup schedulers, loss definitions...)
│
├── Rain Location Prior/        # Rain Location Prior (RLP)
│   ├── models/                 # Uformer, UNet, RPIM definitions
│   ├── options.py              # Centralized hyperparameter routing
│   └── (AMP training logic, DataParallel handlers...)
│
├── NDR/                        # Non-local Differential Restoration
│   ├── models/                 # Non-local differential blocks
│   ├── options/                # YAML-driven configuration system
│   └── (Brutal cleanup inference scripts, CSV metric loggers...)
│
└── README.md                   # You are here
```

Every sub-directory operates as an independent workspace, complete with its own specific `README.md`, localized data handling logic, and most contain their own Python dependencies (`requirements.txt` or `.yml`).

---

## ⚙️ Extensive Setup & Usage Guide

### 1. System Requirements
- **OS**: Linux or Windows 10/11
- **Python**: Version 3.8 or higher
- **Framework**: PyTorch 1.10+ (CUDA is **strongly** recommended. While CPU inference is technically possible, it is computationally prohibitive for these models).

### 2. Initialization
First, clone the master repository to your local machine:
```bash
git clone https://github.com/aishwarysrivastava1/ImageDeraining.git
cd ImageDeraining
```

### 3. Environment & Execution per Architecture
Because each architecture requires specific library versions (e.g., Einops for NeRD, Timm for RLP's Uformer), you must navigate into the desired directory and configure its specific environment.

**Example: Training and Evaluating RLP with Uformer**
```bash
# Enter the workspace
cd "Rain Location Prior"

# Install isolated dependencies (e.g. timm, warmup_scheduler, tqdm)
pip install -r requirements.txt

# Launch AMP-enabled training on GTAV-NightRain
python train.py \
  --arch Uformer_B \
  --use_rlp --use_rpim \
  --dataset GTAV-NightRain \
  --train_dir /path/to/your/local/data \
  --batch_size 4 \
  --nepoch 250 \
  --warmup
```

**Example: Running Inference on NeRD**
```bash
cd ../NeRD
pip install -r requirements.txt

# Run inference and generate restored images
python test_gtav.py \
    --input_dir /path/to/GTAV-NightRain \
    --result_dir ./results/NeRD/GTAV \
    --weights ./logs/model_latest.pth

# Calculate PSNR/SSIM against Ground Truth
python evaluate_gtav.py \
    --result_dir ./results/NeRD/GTAV \
    --gt_dir /path/to/GTAV-NightRain \
    --csv_dir ./results/NeRD
```

*For highly detailed, script-specific flags and YAML configurations, please consult the inner `README.md` file located inside each architecture's folder.*

---

## 📚 Acknowledgements & Bibliography

This comprehensive repository is the result of my personal passion for computational photography and image restoration. However, it is fundamentally built upon the brilliant foundational research of others. I extend my utmost respect and deepest gratitude to the original authors of these architectures. 

If my implementations, refactors, or optimizations aid in your understanding or research, I ask that you direct all formal academic citations to the following pioneering works:

- **MFDNet**: *Multi-Scale Fusion and Decomposition Network for Single Image Deraining.* Upstream codebase: [qwangg/MFDNet](https://github.com/qwangg/MFDNet)
- **NeRD**: *Neural Representation for Rain Drop Removal.* Upstream codebase: [cschenxiang/NeRD-Rain](https://github.com/cschenxiang/NeRD-Rain)
- **RLP**: *Rain Location Prior for Image Deraining.* Upstream codebase: [zkawfanx/RLP](https://github.com/zkawfanx/RLP)
- **NDR**: *Non-local Differential Restoration for Image Deraining.* Upstream codebase: [mdyao/NDR-Restore](https://github.com/mdyao/NDR-Restore)

---

<div align="center">
<b>Implementation by Aishwary Srivastava</b> <br><br>
<sub>Authored for rigorous personal research in Deep Learning and Low-Level Computer Vision · MIT Licensed</sub>
</div>
