# Inverse Photonic Design Framework — Technical Documentation

**Project:** Neural Network-Based Inverse Design for Sb2S3 Multilayer Structures  
**Simulator:** Stanford S4 (RCWA solver)  
**Environment:** Python 3, PyTorch, Anaconda, Windows  
**Status:** Validated on 120k dataset  

---

## 1. Problem Statement

### 1.1 The Forward Problem (Conventional)
Given a multilayer photonic structure, compute its optical spectrum using rigorous coupled-wave analysis (RCWA) via S4.

```
Structure (8 layer thicknesses in nm)  ──→  S4 Simulator  ──→  Optical Spectrum (244 values)
```

### 1.2 The Inverse Problem (This Framework)
Given a desired optical spectrum, predict the multilayer structure that produces it.

```
Optical Spectrum (244 values)  ──→  ???  ──→  Structure (8 layer thicknesses in nm)
```

This is fundamentally harder than the forward problem due to **non-uniqueness**: multiple different structures can produce nearly identical spectra. A direct neural network averages these solutions and produces physically wrong predictions.

---

## 2. Physical System Description

### 2.1 Layer Stack
The structure is a 1D multilayer stack of alternating SiO2 and Sb2S3 layers on a SiO2 substrate, sandwiched between vacuum.

```
Vacuum_Above
─────────────────
Layer 1  →  SiO2     (thickness: 25–200 nm)
Layer 2  →  Sb2S3    (thickness: 25–200 nm)
Layer 3  →  SiO2     (thickness: 25–200 nm)
Layer 4  →  Sb2S3    (thickness: 25–200 nm)
Layer 5  →  SiO2     (thickness: 25–200 nm)
Layer 6  →  Sb2S3    (thickness: 25–200 nm)
Layer 7  →  SiO2     (thickness: 25–200 nm)
Layer 8  →  Sb2S3    (thickness: 25–200 nm)
─────────────────
SiO2 substrate (500 nm)
─────────────────
Vacuum_Below
```

### 2.2 Material: Sb2S3 — Why Two Phases?
Sb2S3 is a **phase-change material** with two distinct optical states:
- **Amorphous phase (a):** Lower refractive index, lower absorption
- **Crystalline phase (c):** Higher refractive index, higher absorption

The simulator computes the spectrum for BOTH phases in a single run. This is why the spectrum has 4 channels instead of 2.

### 2.3 Spectrum Structure
The output spectrum is computed at **61 wavelength points** from 400 nm to 705 nm (step = 5 nm).

At each wavelength, 4 values are recorded:

| Channel | Symbol | Description |
|---|---|---|
| 1 | T_amor | Transmission — amorphous phase |
| 2 | R_amor | Reflection — amorphous phase |
| 3 | T_crys | Transmission — crystalline phase |
| 4 | R_crys | Reflection — crystalline phase |

**Data layout:** Values are interleaved: `[T_a_λ1, R_a_λ1, T_c_λ1, R_c_λ1, T_a_λ2, R_a_λ2, ...]`

To extract individual channels from a 244-element array:
```python
T_amor = spectrum[0::4]   # indices 0, 4, 8, ...
R_amor = spectrum[1::4]   # indices 1, 5, 9, ...
T_crys = spectrum[2::4]   # indices 2, 6, 10, ...
R_crys = spectrum[3::4]   # indices 3, 7, 11, ...
```

### 2.4 Input/Output Dimensions Summary

| Variable | Shape | Range | Description |
|---|---|---|---|
| Structure | (8,) | [25, 200] nm | Layer thicknesses |
| Spectrum | (244,) | [0, 1] | T and R values (physically bounded) |

---

## 3. Dataset

### 3.1 Generation
Structures were sampled uniformly from [25, 200] nm for each of the 8 layers. S4 was called for each structure to compute the corresponding spectrum.

```
Total generated:   120,000 (structure, spectrum) pairs
Train split:        84,000  (70%)
Validation split:   18,000  (15%)
Test split:         18,000  (15%)
Split method:       Random permutation, seed=42
```

### 3.2 Files

| File | Shape | Description |
|---|---|---|
| `structures.npy` | (120000, 8) | Layer thicknesses in nm |
| `spectra.npy` | (120000, 244) | Corresponding spectra |
| `s_min.npy` | (8,) | Per-layer min for normalization |
| `s_max.npy` | (8,) | Per-layer max for normalization |

### 3.3 Normalization

**Structures** — normalized to [-1, 1] (better gradient flow for deep networks):
```python
structures_norm = 2 * (structures - s_min) / (s_max - s_min) - 1
```

**Spectra** — used raw, no normalization needed (T and R are physically in [0, 1]).

**Denormalization** (to recover nm values from model output):
```python
structures_nm = (structures_norm + 1) / 2 * (s_max - s_min) + s_min
```

---

## 4. Architecture

The framework consists of two neural networks trained sequentially.

### 4.1 Forward Network (Phase 1)

**Purpose:** Learn the physical mapping Structure → Spectrum. This is the well-posed, deterministic direction.

**Architecture:**
```
Input:  (8,)   — normalized structure
         ↓
  Linear(8 → 512) + LayerNorm + GELU       ← input projection
         ↓
  ResBlock(512) × 4                         ← residual backbone
         ↓
  Linear(512 → 256) + GELU
  Linear(256 → 244) + Sigmoid              ← output in [0,1]
         ↓
Output: (244,)  — predicted spectrum
```

**ResBlock definition:**
```python
class ResBlock(nn.Module):
    def __init__(self, dim):
        self.block = Sequential(
            Linear(dim, dim), LayerNorm(dim), GELU(),
            Linear(dim, dim), LayerNorm(dim))
    def forward(self, x):
        return GELU()(x + self.block(x))   # skip connection
```

**Key design decisions:**

| Decision | Choice | Reason |
|---|---|---|
| Skip connections | Yes (ResBlock) | Sharp spectral peaks need residuals |
| Normalization | LayerNorm | Stable training, no batch size dependency |
| Activation | GELU | Smoother gradients than ReLU for sharp features |
| Output activation | Sigmoid | Forces output to [0,1] matching physical bounds |
| Dropout | None | Forward mapping is deterministic — dropout hurts |
| Width | 512 | 8→244 expansion needs capacity |

**Training configuration:**
```
Optimizer:     AdamW, lr=3e-4, weight_decay=1e-5
Scheduler:     CosineAnnealingLR, T_max=300, eta_min=1e-5
Loss:          MSELoss
Batch size:    1024
Epochs:        300
Parameters:    2,309,108
Best Val MSE:  0.000089   (vs 0.009379 with basic MLP — 100x improvement)
```

**Saved as:** `best_forward_net_v2.pth`

---

### 4.2 Inverse Network — Tandem (Phase 2)

**Purpose:** Learn the inverse mapping Spectrum → Structure. Trained using a tandem approach to handle non-uniqueness.

**Architecture:**
```
Input:  (244,)  — spectrum
         ↓
  Linear(244 → 512) + LayerNorm + GELU
  Linear(512 → 512) + LayerNorm + GELU
  Linear(512 → 256) + LayerNorm + GELU
  Linear(256 → 128) + GELU
  Linear(128 → 8)   + Tanh               ← output in [-1,1]
         ↓
Output: (8,)  — normalized structure
```

**Why Tanh output:** The structure is normalized to [-1, 1]. Tanh directly produces values in this range, avoiding the need for clipping.

**Tandem Training — The Key Idea:**

The non-uniqueness problem means that if you train with direct MSE loss on structures, the network learns to average all valid solutions — producing a prediction that is wrong for every single one of them.

The tandem approach avoids this by also measuring how well the predicted structure *performs* spectrally:

```
Input Spectrum
      ↓
 Inverse Net  →  Predicted Structure
                        ↓
               Frozen Forward Net  →  Reconstructed Spectrum
                        ↓
         Loss = α × MSE(pred_struct, true_struct)       [direct loss]
              + (1-α) × MSE(recon_spec, input_spec)     [tandem loss]
```

The forward net is completely frozen during inverse training. The gradient flows: Reconstructed Spectrum → Forward Net → Predicted Structure → Inverse Net.

**Training configuration:**
```
Optimizer:     AdamW, lr=3e-4, weight_decay=1e-5
Scheduler:     CosineAnnealingLR, T_max=300, eta_min=1e-5
Alpha (α):     0.3   (30% direct, 70% tandem)
Batch size:    1024
Epochs:        300
Parameters:    555,912
Best Val Loss: 0.024671
```

**Saved as:** `best_inverse_net_v2.pth`

---

## 5. Inference Pipeline

At deployment, for a given target spectrum, the full pipeline is:

```
Step 1: Feed spectrum → Inverse Net → Initial structure estimate (ms)
Step 2: Use initial estimate as starting point for Nelder-Mead optimization
        Objective function uses the FROZEN FORWARD NET (not S4) — fast
        500 iterations, bounds: [-1, 1] in normalized space
Step 3: Denormalize optimized structure → clip to [25, 200] nm → round to int
Step 4: Write structure to verify_input.txt → run S4 → get verified spectrum
Step 5: Compute MSE / MAE between verified spectrum and target spectrum
```

**Total time per prediction:** < 10 seconds  
**S4 calls per prediction:** 1 (only for final verification)

---

## 6. Key Files

| File | Purpose |
|---|---|
| `structures.npy` | Full structure dataset |
| `spectra.npy` | Full spectrum dataset |
| `s_min.npy` | Normalization minimum per layer |
| `s_max.npy` | Normalization maximum per layer |
| `best_forward_net_v2.pth` | Trained forward network weights |
| `best_inverse_net_v2.pth` | Trained inverse network weights |
| `verify.lua` | S4 script for single-structure verification |
| `verify_input.txt` | Interface file: Python writes structure, S4 reads it |
| `verified_spectrum.dat` | Interface file: S4 writes spectrum, Python reads it |
| `train_inverse.py` | Training script for inverse network |
| `verify_pipeline.py` | Basic S4 verification loop |
| `verify_optimized.py` | Verification with Nelder-Mead post-optimization |

---

## 7. Results

### 7.1 Forward Network Performance

| Metric | Basic MLP (first attempt) | ResNet (final) |
|---|---|---|
| Val MSE | 0.009379 | 0.000089 |
| Epoch plateaued | 20 | Still improving at 300 |
| Peak reconstruction | Missed entirely | Accurately captured |
| Improvement | — | 100× |

### 7.2 Inverse Network R² per Layer

| Layer | Material | R² | Quality |
|---|---|---|---|
| 1 | SiO2 | 0.986 | Excellent |
| 2 | Sb2S3 | 0.990 | Excellent |
| 3 | SiO2 | 0.951 | Excellent |
| 4 | Sb2S3 | 0.779 | Good |
| 5 | SiO2 | 0.700 | Acceptable |
| 6 | Sb2S3 | 0.645 | Weak |
| 7 | SiO2 | 0.516 | Weak |
| 8 | Sb2S3 | 0.717 | Acceptable |
| **Mean** | | **0.786** | **Good** |

**Why buried layers have lower R²:**
Outer layers (1–3) have the most influence on the spectrum — small changes in their thickness cause large, distinct spectral shifts. Buried layers (5–7) have weaker, less unique spectral signatures. Multiple different combinations of buried layer thicknesses produce similar spectra — this is non-uniqueness, and it is a physical property of the system, not a model failure.

### 7.3 S4 Verification Results (20 test samples)

**Before optimization (raw inverse net prediction):**
```
Mean MSE:  0.017455
Mean MAE:  0.0595
```

**After Nelder-Mead optimization using NN forward proxy:**
```
Mean MSE:  0.002860
Mean MAE:  ~0.022
Improvement: 83.6%
```

**Sample quality breakdown:**

| Category | Threshold | Count | % |
|---|---|---|---|
| Excellent | MSE < 0.001 | ~8/20 | 40% |
| Good | MSE 0.001–0.005 | ~5/20 | 25% |
| Acceptable | MSE 0.005–0.010 | ~4/20 | 20% |
| Poor | MSE > 0.010 | ~3/20 | 15% |

---

## 8. Scaling Up for Production

To go from this validated framework to a production-quality inverse design tool, the following steps are required in order of priority:

### 8.1 Data (Highest Priority)

The single biggest lever. Current: 120k samples. Target: 500k–1M.

```
Current dataset:   120,000 pairs  →  Mean R² = 0.786
Target dataset:    500,000 pairs  →  Expected Mean R² ≈ 0.88–0.92
```

**What to do:**
```python
# Generate structures by uniform random sampling
import numpy as np
n_new = 500000
structures_new = np.random.randint(25, 201, size=(n_new, 8))
np.savetxt('structures_500k.txt', structures_new, fmt='%d', delimiter='\t')
```

Then run S4 on `structures_500k.txt`. The Lua script already handles batch generation — just point it at the new file.

**Expected gains from more data:**
- Layers 6 and 7 (currently R² = 0.52–0.65) will improve most
- Mean R² should reach 0.88+
- Fewer optimization failures (the sample 18 type cases)

**Important:** Regenerate the .npy files and retrain both networks from scratch on the new dataset. Do not mix old and new data.

### 8.2 Forward Network (Minor Changes Needed)

The current ResNet architecture is good. For 500k data, increase slightly:

```python
# Change ResBlock count from 4 to 6
self.res_blocks = nn.Sequential(
    ResBlock(512), ResBlock(512), ResBlock(512),
    ResBlock(512), ResBlock(512), ResBlock(512))   # 4 → 6

# Increase training epochs
for epoch in range(500):   # 300 → 500
```

Target Val MSE after scaling: < 0.000050

### 8.3 Inverse Network (Architecture Upgrade Worth Trying)

With 500k data, increase inverse net capacity:

```python
class InverseNet(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(244, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512),  nn.GELU(),
            nn.Linear(512,  512), nn.LayerNorm(512),  nn.GELU(),
            nn.Linear(512,  256), nn.LayerNorm(256),  nn.GELU(),
            nn.Linear(256,  128), nn.GELU(),
            nn.Linear(128,  8),   nn.Tanh())
```

Also consider tuning alpha (currently 0.3). Higher alpha gives more weight to spectrum reconstruction vs direct structure matching:
```
alpha = 0.2   →  more emphasis on direct structure matching
alpha = 0.5   →  balanced (try this with larger dataset)
```

### 8.4 Optimization Step (Can Improve Without New Data)

Current: 500 Nelder-Mead iterations, single start point.

**Better version — multi-start with more iterations:**
```python
from scipy.optimize import differential_evolution

# Replace Nelder-Mead with differential evolution for global search
bounds = [(-1, 1)] * 8
result = differential_evolution(
    objective,
    bounds,
    maxiter=1000,
    tol=1e-6,
    seed=42,
    x0=pred_norm)   # warm start from inverse net
```

This directly addresses the sample 18 failure type where Nelder-Mead got stuck at a boundary.

Alternatively, multi-start Nelder-Mead:
```python
best_loss = float('inf')
best_result = None
starting_points = [pred_norm]   # inverse net prediction

# Add 4 random restarts near the prediction
for _ in range(4):
    perturbed = pred_norm + np.random.normal(0, 0.1, size=8)
    perturbed = np.clip(perturbed, -1, 1)
    starting_points.append(perturbed)

for x0 in starting_points:
    result = minimize(objective, x0=x0, method="Nelder-Mead",
                      options={"maxiter": 1000})
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result
```

Expected improvement: Success rate from 80% → 92%+

### 8.5 Evaluation Metric for Production

R² on structures is not the right metric for deployment. The correct metric is:

```
For each test sample:
  1. Predict structure from spectrum
  2. Simulate predicted structure with S4
  3. Compute MAE between simulated and target spectrum

Report:
  - Mean MAE across all test samples
  - % samples with MAE < 0.02  (excellent)
  - % samples with MAE < 0.05  (acceptable)
```

Current baseline (120k, 20 samples): Mean MAE = 0.022 after optimization.
Production target: Mean MAE < 0.015, >90% samples under 0.05.

---

## 9. What Was Tried and Discarded

### 9.1 Log Transform on Spectra
Initially applied log scaling to handle near-zero transmission values. Discarded because it distorted the MSE loss and caused the network to plateau early. Raw spectra in [0,1] work better.

### 9.2 Structure Normalization to [0,1]
Initial normalization was [0,1]. Changed to [-1,1] for better gradient flow in deep networks with Tanh activations.

### 9.3 ReduceLROnPlateau Scheduler
Caused premature learning rate decay when the network was still learning slowly. Replaced with CosineAnnealingLR which decays smoothly over the full training period.

### 9.4 Mixture Density Network (MDN)
Tried with K=5 components to model the distribution of valid structures. Results were worse than the tandem network:
```
Tandem inverse:        Mean R² = 0.786
MDN best component:    Mean R² = 0.727   ← worse
MDN best of 20:        Mean R² = 0.571   ← much worse
```
The MDN overfit severely (train NLL -11.37 vs val NLL -7.09). The buried layer degeneracy is a physical property — the MDN found it but the sampling noise overwhelmed the signal. Discarded.

### 9.5 Clustered Datasets (1k and 2k)
Generated 125k structures but only computed spectra for 2k samples (clustered subset). This was used for early validation. The clustering removed continuous coverage that neural networks need. The forward network trained on 2k data completely missed spectral peaks. Discarded — used full 120k instead.

---

## 10. Known Limitations

**Non-uniqueness of buried layers:**
Layers 6 and 7 have R² of 0.52–0.65. Multiple structures produce identical or near-identical spectra. This is a physical property of the system. The predicted structure may not match the "true" structure but may still produce the correct spectrum — which is all that matters for inverse design.

**Integer rounding:**
Layer thicknesses are rounded to the nearest nm for S4 input. This introduces a small discretization error. For very thin layers near 25 nm, a 1 nm rounding error is 4% relative error. Negligible for thicker layers.

**Wavelength range fixed:**
The framework covers 400–705 nm at 5 nm steps. If a different wavelength range or finer resolution is needed, both the S4 Lua script and the network input/output dimensions must change.

**Single polarization and normal incidence:**
The Lua script uses normal incidence and a single polarization. Angle-resolved or polarization-resolved data would provide more information and likely improve buried layer R².

---

## 11. Quick Reference — Running the Pipeline

### Train Forward Network
```bash
# (Already done — weights saved in best_forward_net_v2.pth)
# To retrain on new data:
python train_forward.py
```

### Train Inverse Network
```bash
python train_inverse.py
```

### Run Verification (basic)
```bash
python verify_pipeline.py
```

### Run Verification with Optimization (recommended)
```bash
python verify_optimized.py
```

### Predict structure for a new spectrum
```python
import numpy as np
import torch

s_min = np.load('s_min.npy')
s_max = np.load('s_max.npy')

# Your target spectrum — shape (244,) with values in [0,1]
target_spectrum = np.array([...], dtype=np.float32)

inv_net = InverseNet()
inv_net.load_state_dict(torch.load('best_inverse_net_v2.pth', map_location='cpu'))
inv_net.eval()

with torch.no_grad():
    pred_norm = inv_net(torch.tensor(target_spectrum).unsqueeze(0)).numpy()[0]

pred_nm = (pred_norm + 1) / 2 * (s_max - s_min) + s_min
pred_nm = np.clip(np.round(pred_nm), 25, 200).astype(int)
print("Predicted structure (nm):", pred_nm)
# Then run through verify_optimized.py for refinement and S4 verification
```
