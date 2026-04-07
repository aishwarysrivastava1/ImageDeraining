# NDR-Restore: Enhanced Multidataset Implementation

This repository is a refined and expanded implementation of the **NDR (Non-local Differential Restoration)** for image deraining, based on the research paper [Non-local Differential Restoration for Image Deraining](https://github.com/mdyao/NDR-Restore).

This version includes several critical enhancements for robustness, memory efficiency, and large-scale evaluation across multiple datasets.

## 🚀 Key Features (Extended Implementation)

Apart from the core NDR logic, this implementation introduces:

*   **Multidataset Support**: Native integration for evaluating across 5+ major deraining datasets including:
    *   **GTAV-NightRain**: High-fidelity nighttime synthetic rain.
    *   **GTrain**: Generalized synthetic rain training sets.
    *   **Outdoor-Rain**: Outdoor scenes with varying rain density.
    *   **RainDS**: Real and synthetic rain combinations, including **rain-type specific metrics** (Rainstreak vs. Raindrop).
    *   **RealRain**: Evaluation on real-world captured rain.
*   **Memory-Optimized Inference Engine**: Innovative "Brutal Cleanup" system that manages GPU memory after *every* single image inference. This allows for:
    *   High-resolution testing on consumer GPUs with limited VRAM.
    *   Robust handling of `CudaOutOfMemory` errors without crashing the entire test suite.
    *   Automated `gc.collect()` and `torch.cuda.ipc_collect()` cycles.
*   **Automated Evaluation & Reporting**:
    *   Real-time PSNR and SSIM metric calculation.
    *   Detailed **CSV Logging**: Automatically exports timestamped results including per-image metrics and overall averages.
    *   Unified logging system with console and file output.
*   **YAML-Driven Configuration**: Easily switch between datasets and model versions using structured `.yml` files in the `options/` directory.

## 📁 Project Structure

```text
NDR/
├── data/               # Dataset-specific loaders (GTAV, GTrain, etc.)
├── models/             # NDR Architecture and model logic
├── options/            # YAML configuration files for testing
├── pretrained model/   # Directory for .pth model weights
├── results/            # Automated output (Images and CSV metrics)
├── utils/              # Image processing and logging utilities
├── test_*.py           # Dataset-specific test entry points
└── requirements.txt    # Project dependencies
```

### Parameters:
*   `-opt`: Path to the YAML configuration file defining data paths and network settings.
*   `--model_path`: Path to the pre-trained `.pth` file.
*   `--csv_dir`: (Optional) Custom directory to save result CSVs.

## 📈 Results and Metrics

After running a test, results are stored in the `results/` directory:
1.  **Saved Images**: Processed images showing the restored output.
2.  **CSV Reports**: Detailed spreadsheets containing:
    *   Model and Dataset metadata.
    *   Average PSNR (dB) and SSIM across the whole set.
    *   Individual PSNR/SSIM for every image in the dataset.

## 🧠 Specifications

*   **Framework**: PyTorch
*   **Backbone**: NDR (Non-local Differential Restoration)
*   **Optimization**: Memory-optimized for seamless testing on high-resolution images.
*   **Precision**: Full evaluation support for PSNR/SSIM in RGB space.

## 🙏 Acknowledgements

This implementation is based on the original work of **Miao Yao et al.** on NDR-Restore. Special thanks to the researchers for their contribution to the field of image restoration.

