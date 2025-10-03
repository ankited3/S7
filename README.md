# 🧠 CIFAR-10 Neural Network

This project implements a custom Convolutional Neural Network (CNN) for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), with advanced augmentations and novel architectural choices such as **Depthwise Separable Convolutions**, **Dilated Convolutions**, and **Global Average Pooling (GAP)**.

---

## 📊 Dataset

- **Dataset**: CIFAR-10 (32x32 RGB images, 10 classes)
- **Preprocessing**:
  - Calculated **mean** and **standard deviation** for normalization
  - Input layer: **3 channels (RGB)**
  - Output layer: **10 channels (classes)**

---

## 🎛️ Data Augmentation

Data augmentation was performed using the [Albumentations](https://albumentations.ai/) library:

- Horizontal Flip  
- ShiftScaleRotate  
- CoarseDropout  

---

## 🏗️ Neural Network Architecture

# 🚀 Deep Convolutional Neural Network (CNN) Architecture with Progressive Receptive Fields

This document outlines a custom **Deep Convolutional Neural Network (CNN)** designed to aggressively build a large **Receptive Field (RF)** early in the network while managing computational cost through techniques like **Depthwise Separable Convolutions** and **$1 \times 1$ bottlenecks**.

The architecture uses a sequence of **Convolution Blocks** for feature extraction and **Transition Blocks** for spatial downsampling.

---

## 🧱 Block Details & Structure

### CONVOLUTION BLOCK 1 (Feature Extraction & Channel Expansion)

This block focuses on initial feature extraction and rapid RF growth without spatial downsampling.

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $32 \times 32 \times 3$ | **(3x3x3)x32** | $32 \times 32 \times 32$ | $3 \times 3$ |
| $32 \times 32 \times 32$ | **(3x3x32)x32** | $32 \times 32 \times 32$ | $5 \times 5$ |
| $32 \times 32 \times 32$ | **(3x3x32)x32** | $32 \times 32 \times 32$ | $7 \times 7$ |
| $32 \times 32 \times 32$ | **(3x3x32)x32** | $32 \times 32 \times 32$ | $9 \times 9$ |
| $32 \times 32 \times 32$ | **(3x3x32)x64** | $32 \times 32 \times 64$ | $11 \times 11$ |

---

### ↘️ TRANSITION BLOCK 1 (Spatial Downsampling)

The first transition block halves the spatial dimensions using a strided convolution and employs a $1 \times 1$ convolution for channel reduction (bottleneck).

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $16 \times 16 \times 64$ | **(2x2x64)x64** | $16 \times 16 \times 64$ | $12 \times 12$ |
| $16 \times 16 \times 64$ | **(1x1x64)x32** | $16 \times 16 \times 32$ | $12 \times 12$ |

---

### 🧠 CONVOLUTION BLOCK 2 (Depthwise Separable Layer)

This block introduces a **Depthwise Separable Convolution** to gain a significant increase in RF with fewer parameters, followed by channel manipulation.

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $16 \times 16 \times 32$ | **(3x3x32)x32** | $16 \times 16 \times 32$ | $16 \times 16$ |
| $16 \times 16 \times 32$ | **(1x1x64)x32** | $16 \times 16 \times 64$ | $20 \times 20$ |
| $16 \times 16 \times 64$ | **(3x3x32)x64** | $16 \times 16 \times 32$ | $24 \times 24$ |
| $16 \times 16 \times 32$ | **(3x3x32)x32** | $16 \times 16 \times 32$ | $28 \times 28$ |

---

### ↘️ TRANSITION BLOCK 2 (Second Downsampling)

Further spatial reduction to $8 \times 8$ and maintenance of the 32-channel depth.

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $8 \times 8 \times 32$ | **(2x2x32)x32** | $8 \times 8 \times 32$ | $30 \times 30$ |
| $8 \times 8 \times 32$ | **(1x1x32)x32** | $8 \times 8 \times 32$ | $30 \times 30$ |

---

### ⬆️ CONVOLUTION BLOCK 3 & 4 (Deep Feature Extraction)

These blocks prioritize a massive increase in the RF through sustained $3 \times 3$ convolutions while maintaining the $8 \times 8$ and later $3 \times 3$ spatial size.

#### CONVOLUTION BLOCK 3 ($8 \times 8$ Resolution)

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $8 \times 8 \times 32$ | **(3x3x32)x32** | $8 \times 8 \times 32$ | $38 \times 38$ |
| $8 \times 8 \times 32$ | **(3x3x32)x32** | $8 \times 8 \times 32$ | $46 \times 46$ |
| $8 \times 8 \times 32$ | **(3x3x32)x32** | $8 \times 8 \times 32$ | $54 \times 54$ |
| $8 \times 8 \times 32$ | **(3x3x32)x32** | $8 \times 8 \times 32$ | $62 \times 62$ |

#### TRANSITION BLOCK 3 (Dilated Downsampling)

This block uses a **Dilated Convolution with stride 2** to downsample to $3 \times 3$, effectively combining downsampling with an enlarged view of the features.

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $3 \times 3 \times 32$ | **(3x3x32)x32** | $3 \times 3 \times 32$ | $78 \times 78$ |
| $3 \times 3 \times 32$ | **(1x1x32)x32** | $3 \times 3 \times 32$ | $78 \times 78$ |

#### CONVOLUTION BLOCK 4 ($3 \times 3$ Resolution)

The final feature extraction block achieves a very large RF, essential for capturing global context.

| Input Size | Kernel/Operation | Output Size | Receptive Field (RF) |
| :---: | :---: | :---: | :---: |
| $3 \times 3 \times 32$ | **(3x3x32)x32** | $3 \times 3 \times 32$ | $94 \times 94$ |
| $3 \times 3 \times 32$ | **(3x3x32)x32** | $3 \times 3 \times 32$ | $110 \times 110$ |
| $3 \times 3 \times 32$ | **(3x3x32)x32** | $3 \times 3 \times 32$ | $126 \times 126$ |
| $3 \times 3 \times 32$ | **(3x3x32)x32** | $3 \times 3 \times 32$ | $142 \times 142$ |

---


## ⚙️ Model Details

- **Total Parameters**: 195,264  
- **Receptive Field (RF)**: 142  
- **Key Features**:
  - ✅ Depthwise Separable Convolutions  
  - ✅ Dilated Convolutions  
  - ✅ Global Average Pooling (GAP)  
  - ❌ No MaxPooling layers  

---

## 📈 Training & Results

- **Epochs**: 50  
- **Training Behavior**:  
  - Underfitting in first ~40 epochs  
  - Slight overfitting afterwards  

- **Performance**:  
  - Max **Train Accuracy**: 85.18%  
  - Max **Test Accuracy**: 86.20%  

---

## 🚀 Actions / Next Steps

- Experiment with learning rate schedules (Cyclic LR, OneCycle LR)  
- Add dropout or stronger augmentations to reduce overfitting  
- Explore model pruning or quantization for deployment  

---

## 📜 Observations

- Data augmentation (horizontal flip, shiftScaleRotate, coarseDropout) helped generalization.  
- GAP layer enabled direct mapping to **10 classes** without FC layers.  
- Depthwise + Dilated convolutions allowed a larger receptive field without adding excessive parameters.  

---









