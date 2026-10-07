# SwinSparseCLMoE: Sparsely-Gated Mixture-of-Experts with Contrastive Learning for Efficient Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-v1.0+-blue.svg)](https://monai.io/)
[![Release](https://img.shields.io/badge/Release-v1.0--Reproducibility-blue)](https://github.com/Yoonguu/SwinSparseCLMoE/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **SwinSparseCLMoE**, a highly accurate and parameter-efficient 3D framework designed for multi-parametric MRI (mpMRI) brain tumor segmentation across the **BraTS 2021** and **BraTS 2024** benchmarks.

---

## 🌟 Key Features & Contributions
* **Bottleneck Contrastive Refinement**: Incorporates InfoNCE-based contrastive learning at the Swin encoder bottleneck to yield intensity-invariant, noise-robust anatomical feature routing signals.
* **True Sparse Execution via Dynamic MoE**: Leverages deterministic Top-$k$ ($k=2, N=4$) sparse routing governed by an auxiliary load-balancing loss, bypassing unselected experts completely to ensure genuine operational efficiency.
* **Exceptional Efficiency**: Achieves State-of-the-Art (SOTA) segmentation performance using only **6.16M parameters (4.38M active)** and **245.57 GFLOPs**, outperforming the monolithic nnU-Net (64.65M Params, 603.44 GFLOPs).
* **Strict Reproducibility**: Complete end-to-end pipeline including data splits, preprocessing, label conversion, training, evaluation, and hardware efficiency profiling.

---

## 📊 Performance Highlights

| Dataset | Model | Mean Dice $\uparrow$ | Mean HD95 (mm) $\downarrow$ | Active Params | GFLOPs |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BraTS 2021** | nnU-Net | 0.9030 | 3.7686 | 64.65M | 603.44 |
| | **SwinSparseCLMoE** | **0.9072** | **3.1243** | **4.38M** | **245.57** |
| **BraTS 2024** | nnU-Net | 0.8400 | 4.6620 | 64.65M | 603.44 |
| | **SwinSparseCLMoE** | **0.8692** | **4.1070** | **4.38M** | **245.57** |

---

## 🚀 Executable Reproduction Procedure

To ensure complete transparency and reproducibility as reported in our study, please follow the step-by-step instructions below.

### 1. Installation
Ensure you have Python 3.10+ installed. Install the required dependencies including `fvcore` for efficiency profiling.
```bash
pip install torch torchvision
pip install monai nibabel medpy wandb fvcore
