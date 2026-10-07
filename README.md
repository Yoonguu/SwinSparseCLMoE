# SwinSparseCLMoE: Sparsely-Gated Mixture-of-Experts with Contrastive Learning for Efficient Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-v1.0+-blue.svg)](https://monai.io/)
[![Release](https://img.shields.io/badge/Release-v1.0--Reproducibility-blue)](https://github.com/Yoonguu/SwinSparseCLMoE/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **SwinSparseCLMoE**, a highly accurate and parameter-efficient 3D framework designed for multi-parametric MRI (mpMRI) brain tumor segmentation across the **BraTS 2021** and **BraTS 2024** benchmarks.

---

## 🌟 Key Features & Contributions
* **Bottleneck Contrastive Refinement**: Incorporates InfoNCE-based contrastive learning at the Swin encoder bottleneck to yield intensity-invariant, noise-robust anatomical feature routing signals.
* **True Sparse Execution via Dynamic MoE**: Leverages deterministic Top-$k$ ($k=2, N=4$) sparse routing governed by an auxiliary load-balancing loss. We use highly compact $1 \times 1 \times 1$ convolutional experts and dynamically bypass unselected experts completely to ensure genuine operational efficiency (No dense masking).
* **Exceptional Efficiency**: Achieves State-of-the-Art (SOTA) segmentation performance using only **6.16M parameters (4.38M active)** and **245.57 GFLOPs**, substantially outperforming monolithic architectures like nnU-Net (64.65M Params, 603.44 GFLOPs).
* **Strict Reproducibility**: Provides a complete end-to-end pipeline including explicit data split identifiers, automated label conversion, fixed random seeds, training, evaluation, and exact hardware efficiency profiling scripts.

---

## 📊 Performance Highlights

| Dataset | Model | Mean Dice $\uparrow$ | Mean HD95 (mm) $\downarrow$ | Active Params | GFLOPs |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BraTS 2021** | nnU-Net | 0.9030 | 3.7686 | 64.65M | 603.44 |
| | **SwinSparseCLMoE (Ours)** | **0.9072** | **3.1243** | **4.38M** | **245.57** |
| **BraTS 2024** | nnU-Net | 0.8400 | 4.6620 | 64.65M | 603.44 |
| | **SwinSparseCLMoE (Ours)** | **0.8692** | **4.1070** | **4.38M** | **245.57** |

---

## 🚀 Executable Reproduction Procedure

To ensure complete transparency and strict reproducibility as reported in our manuscript, please follow the step-by-step instructions below.

### 1. Installation
Ensure you have Python 3.10+ installed. Install the required dependencies, including `fvcore` for computational efficiency profiling.
```bash
pip install torch torchvision
pip install monai nibabel medpy wandb fvcore


### 2. Dataset Preparation & Split Identifiers
Download the BraTS 2021 and BraTS 2024 Adult Glioma (BraTS-GLI) datasets.
To guarantee the exact same data distribution used in our experiments, we provide the fixed 8:1:1 patient-level split identifiers (Train/Val/Test).

BraTS Label Conversion: Our preprocessing pipeline automatically handles the BraTS label conversion from [0, 1, 2, 4] to [0, 1, 2, 3] during data loading.

Split Identifiers: Please check the splits/ directory for the exact list of case IDs used for training, validation, and testing:

splits/brats21_splits.json

splits/brats24_splits.json

### 3. Training
To reproduce the training process with our fixed random seed (seed=42) and specific MoE hyperparameters (Base channel=24, Top-$k$=2, $N$=4):
```bash
# Example for BraTS 2024
python TrainVT_wPatch_wWT_wnpys_4.py \
    --brats-version 24 \
    --data-dir /path/to/BraTS2024/BraTS-GLI \
    --batch-size 4 \
    --expert-count 4 \
    --top-k 2 \
    --base-channel 24 \
    --seed 42 \
    --gpu-num 0


### 4. Evaluation & Inference
The training script automatically performs sliding-window inference on the test set after the best model is saved. However, to manually evaluate a saved checkpoint and calculate detailed region-wise (WT, TC, ET) metrics (Dice, HD95) and save prediction .nii.gz masks:

```bash
python TrainVT_wPatch_wWT_wnpys_4.py \
    --brats-version 24 \
    --data-dir /path/to/BraTS2024/BraTS-GLI \
    --resume /path/to/checkpoint/best_by_mean_dice.pt \
    --save-predictions



### 5. Computational Efficiency Benchmarking
To strictly reproduce the exact conditions used for our efficiency measurements (GFLOPs, Peak Memory, Latency, Throughput) reported in the paper:

```bash
# This script dynamically loads the model architectures without weights and 
# profiles the operational efficiency using a standard 1x4x128x128x128 AMP input.
python measure_efficiency_all.py

