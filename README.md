```markdown
# 🔬 Inverse Photonic Design Framework

### Neural Network-Based Inverse Design for Sb2S3 Multilayer Structures

> Fast, data-driven inverse design of multilayer photonic structures using deep learning + physics-guided optimization.

---

## 📌 Overview

This project tackles the **inverse design problem in photonics**:

- **Input:** Desired optical spectrum (244 values)
- **Output:** Multilayer structure (8 layer thicknesses)

Unlike the forward problem, the inverse mapping is **non-unique** — multiple structures can produce the same spectrum.

### 💡 Solution
A **tandem neural network framework**:
1. Train a **forward model** (Structure → Spectrum)
2. Freeze it
3. Train an **inverse model** (Spectrum → Structure) using physics-guided loss

---

## 🧠 Key Features

- ⚡ **<10 sec prediction time**
- 🔁 **Tandem learning to handle non-uniqueness**
- 📉 **100× improvement in forward model accuracy**
- 🎯 **83.6% error reduction with optimization**
- 🔬 Physics-consistent predictions via forward proxy

---

## 🏗️ System Architecture

```

Spectrum (244) → Inverse Net → Structure (8)
↓
Forward Net (Frozen)
↓
Reconstructed Spectrum

Loss = α·Structure Loss + (1-α)·Spectrum Loss

```

---

## 🧪 Physical System

- **8-layer stack** (SiO₂ / Sb₂S₃ alternating)
- Thickness range: **25–200 nm**
- Substrate: SiO₂ (500 nm)

### 🌈 Spectrum Details

- Wavelengths: **400–705 nm** (61 points)
- 4 channels:
  - Transmission (amorphous & crystalline)
  - Reflection (amorphous & crystalline)

Total size = **61 × 4 = 244**

---

## 📊 Dataset

| Property | Value |
|--------|------|
| Total samples | 120,000 |
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Files

```

structures.npy   # (120k, 8)
spectra.npy      # (120k, 244)
s_min.npy        # normalization min
s_max.npy        # normalization max

```

---

## 🧩 Model Architecture

### 🔷 Forward Network (ResNet-based)

- Input: 8 → Output: 244
- 4× Residual Blocks
- Activation: GELU
- Output: Sigmoid

📈 **Val MSE:** `0.000089` (vs 0.009379 baseline)

---

### 🔶 Inverse Network

- Input: 244 → Output: 8
- Fully connected deep network
- Output: Tanh (normalized structure)

### ⚙️ Tandem Loss

```

Loss = α · MSE(structure)

* (1 - α) · MSE(spectrum)

α = 0.3

```

---

## 🚀 Inference Pipeline

1. Predict structure via inverse net
2. Optimize using forward net (no S4 calls)
3. Denormalize + clip values
4. Run **Stanford S4** for verification

⏱️ **Runtime:** < 10 seconds  
📡 **S4 calls:** 1 per prediction  

---

## 📈 Results

### Forward Model

| Model | Val MSE |
|------|--------|
| Basic MLP | 0.009379 |
| ResNet (final) | **0.000089** |

---

### Inverse Model

- **Mean R²:** `0.786`

| Layer | R² |
|------|----|
| Top layers | ~0.98–0.99 |
| Middle layers | ~0.7 |
| Deep layers | ~0.5–0.6 |

📌 Lower accuracy in deeper layers is due to **physical non-uniqueness**

---

### 🔍 S4 Verification (20 samples)

| Stage | MSE | MAE |
|------|-----|-----|
| Before optimization | 0.0175 | 0.0595 |
| After optimization | **0.00286** | **~0.022** |

✅ **83.6% improvement**

---

## 📂 Project Structure

```

.
├── data/
│   ├── structures.npy
│   ├── spectra.npy
│   ├── s_min.npy
│   └── s_max.npy
│
├── models/
│   ├── best_forward_net_v2.pth
│   └── best_inverse_net_v2.pth
│
├── scripts/
│   ├── train_forward.py
│   ├── train_inverse.py
│   ├── verify_pipeline.py
│   └── verify_optimized.py
│
├── s4/
│   ├── verify.lua
│   ├── verify_input.txt
│   └── verified_spectrum.dat
│
└── README.md

````

---

## ⚡ Quick Start

### 1️⃣ Train Inverse Model
```bash
python train_inverse.py
````

### 2️⃣ Run Verification

```bash
python verify_optimized.py
```

---

## 🔮 Scaling to Production

| Component    | Upgrade                |
| ------------ | ---------------------- |
| Dataset      | 120k → 500k+           |
| Expected R²  | 0.78 → ~0.9            |
| Forward Net  | 4 → 6 ResBlocks        |
| Optimization | Differential Evolution |

---

## ⚠️ Limitations

* ❗ Non-uniqueness in deeper layers
* 🔢 Integer rounding (nm resolution)
* 🌈 Fixed wavelength range
* 📐 Single polarization, normal incidence only

---

## 🧠 Key Insights

* Inverse design is fundamentally **ill-posed**
* Direct MSE fails → averaging effect
* **Tandem learning = physics constraint**
* Forward model acts as a **differentiable simulator**

---

## 📚 Tech Stack

* Python 3
* PyTorch
* NumPy
* Stanford S4 (RCWA)

---

## 👨‍🔬 Author

**Mohammad Huda Ansari**
Electronics & Communication Engineering
Focus: Photonics, Probabilistic Computing, ML for Physics
