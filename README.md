# ECG-MI-Detection-CTAT
This repository implements the Cross-Task Attention Transfer (CTAT) Framework for improved myocardial infarction (MI) diagnosis from 12-lead ECG signals.

## 🔬 Overview

The CTAT framework incorporates clinical morphological knowledge into deep learning models for ECG-based MI detection. Our approach uses:

- **Teacher Network**: Attention U-Net trained for ECG delineation (P-wave, QRS, T-wave segmentation)
- **Student Network**: ResNet34 with attention mechanisms for MI classification
- **Cross-Task Attention Transfer**: Knowledge distillation using KL divergence to transfer morphological attention from teacher to student

### Key Features

- 🎯 **Clinically-Informed**: Incorporates morphological knowledge that cardiologists use for MI diagnosis
- 🔄 **Attention Transfer**: Novel cross-task attention distillation mechanism
- 📊 **Multi-Scale**: Transfers attention at different resolution levels
- 🏥 **Multi-Dataset**: Evaluated on PTB-XL, SPH, and CPSC datasets
- 📈 **High Performance**: Achieves state-of-the-art results with interpretable predictions



## 🛠️ Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)
- 16GB RAM minimum
- NVIDIA GPU with 8GB+ VRAM (recommended)

## 📋 Dependencies

Key dependencies include:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
pandas>=1.3.0
wfdb>=4.0.0
neurokit2>=0.2.0
