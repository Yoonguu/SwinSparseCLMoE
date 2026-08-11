# SwinSparseCLMoE: Sparsely-Gated Mixture-of-Experts with Contrastive Learning for Efficient Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-v1.0+-blue.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **SwinSparseCLMoE**, a highly accurate and parameter-efficient 3D framework designed for multi-parametric MRI (mpMRI) brain tumor segmentation across the **BraTS 2021** and **BraTS 2024** benchmarks.

---

## 🌟 Key Features
* **Bottleneck Contrastive Refinement**: Incorporates InfoNCE-based contrastive learning at the Swin encoder bottleneck to yield intensity-invariant, noise-robust anatomical feature routing signals.
* **Dynamic Sparsely-Gated MoE Head**: Leverages load-balanced Top-$k$ ($k=2, N=4$) dynamic expert routing to specialize in heterogeneous tumor sub-regions (ET, TC, WT) while preventing expert collapse.
* **Exceptional Efficiency**: Achieves State-of-the-Art (SOTA) segmentation performance with only **6.16M parameters** ($\approx 10.5\times$ smaller parameter profile compared to nnU-Net's 64.65M).
* **End-to-End MONAI Pipeline**: Complete multi-parametric 3D training and sliding-window inference pipeline built upon MONAI and PyTorch.

---

## 📊 Performance Highlights

| Dataset | Model | Mean Dice $\uparrow$ | Mean HD95 (mm) $\downarrow$ | Params |
| :--- | :--- | :---: | :---: | :---: |
| **BraTS 2021** | nnU-Net | 0.9030 | 3.7686 | 64.65M |
| | **SwinSparseCLMoE (Ours)** | **0.9072** | **3.1243** | **6.16M** |
| **BraTS 2024** | nnU-Net | 0.8400 | 4.6620 | 64.65M |
| | **SwinSparseCLMoE (Ours)** | **0.8692** | **4.1070** | **6.16M** |

---

## 🚀 Quick Start

### Installation
```bash
pip install torch monai nibabel medpy wandb
