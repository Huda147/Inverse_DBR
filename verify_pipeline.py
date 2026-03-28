
import numpy as np
import torch
import torch.nn as nn
import subprocess
import os
import gc

print("=== S4 Verification Pipeline ===")

# ── Load normalization ─────────────────────────────────────────────────
s_min = np.load("s_min.npy")
s_max = np.load("s_max.npy")

# ── Load test spectra and structures ──────────────────────────────────
spectra    = np.load("spectra.npy").astype(np.float32)
structures = np.load("structures.npy").astype(np.float32)

# Use last 100 samples as test cases (same seed as training split)
n = len(spectra)
idx   = torch.randperm(n, generator=torch.Generator().manual_seed(42))
n_train = int(0.70*n)
n_val   = int(0.15*n)
idx_test = idx[n_train+n_val:].numpy()

# Pick first 20 test samples for verification (S4 is slow)
test_idx      = idx_test[:20]
test_spectra  = spectra[test_idx]
test_structs  = structures[test_idx]
print(f"Verifying {len(test_idx)} test samples via S4")

# ── Rebuild inverse net ────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class ForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(8, 512), nn.LayerNorm(512), nn.GELU())
        self.res_blocks = nn.Sequential(
            ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512))
        self.output_proj = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 244), nn.Sigmoid())
    def forward(self, x):
        return self.output_proj(self.res_blocks(self.input_proj(x)))

class InverseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(244, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 8),   nn.Tanh())
    def forward(self, x): return self.net(x)

device = torch.device("cpu")  # CPU only — stable

inv_net = InverseNet()
inv_net.load_state_dict(torch.load("best_inverse_net_v2.pth", map_location="cpu"))
inv_net.eval()
print("Inverse net loaded")

# ── Predict structures ─────────────────────────────────────────────────
spec_norm = test_spectra.copy()  # spectra already in [0,1]
with torch.no_grad():
    pred_norm = inv_net(torch.tensor(spec_norm)).numpy()

# Denormalize to nm
pred_nm = (pred_norm + 1) / 2 * (s_max - s_min) + s_min
pred_nm = np.clip(np.round(pred_nm), 25, 200).astype(int)
true_nm = test_structs.astype(int)

print("\nPredicted structures (nm):")
for i in range(len(pred_nm)):
    print(f"  Sample {i+1:2d} | True: {true_nm[i]} | Pred: {pred_nm[i]}")

# ── S4 verification loop ───────────────────────────────────────────────
print("\n── Running S4 verification ──")
results = []

for i, (pred_struct, true_spec) in enumerate(zip(pred_nm, test_spectra)):
    # Write structure to verify_input.txt
    line = "\t".join(str(v) for v in pred_struct)
    with open("verify_input.txt", "w") as f:
        f.write(line + "\n")

    # Remove old output
    if os.path.exists("verified_spectrum.dat"):
        os.remove("verified_spectrum.dat")

    # Run S4
    result = subprocess.run(
        ["s4", "verify.lua"],
        capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  Sample {i+1}: S4 FAILED — {result.stderr[:100]}")
        continue

    if not os.path.exists("verified_spectrum.dat"):
        print(f"  Sample {i+1}: output file not created")
        continue

    # Read S4 output
    with open("verified_spectrum.dat", "r") as f:
        line = f.read().strip()

    vals = [float(x) for x in line.split()]
    if len(vals) != 244:
        print(f"  Sample {i+1}: got {len(vals)} values, expected 244")
        continue

    s4_spec = np.array(vals, dtype=np.float32)

    # Compute errors
    mse  = float(np.mean((s4_spec - true_spec)**2))
    mae  = float(np.mean(np.abs(s4_spec - true_spec)))
    rmax = float(np.max(np.abs(s4_spec - true_spec)))

    results.append({
        "sample":    i+1,
        "pred_nm":   pred_struct,
        "true_nm":   true_nm[i],
        "mse":       mse,
        "mae":       mae,
        "max_err":   rmax,
        "s4_spec":   s4_spec,
        "true_spec": true_spec
    })

    print(f"  Sample {i+1:2d} | MSE: {mse:.6f} | MAE: {mae:.4f} | MaxErr: {rmax:.4f}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
if results:
    mses = [r["mse"] for r in results]
    maes = [r["mae"] for r in results]
    print(f"\n── Summary ({len(results)} samples verified) ──")
    print(f"  Mean MSE:    {np.mean(mses):.6f}")
    print(f"  Mean MAE:    {np.mean(maes):.4f}")
    print(f"  Best  MSE:   {np.min(mses):.6f}  (sample {results[np.argmin(mses)]['sample']})")
    print(f"  Worst MSE:   {np.max(mses):.6f}  (sample {results[np.argmax(mses)]['sample']})")

    # Save full results
    with open("verification_results.txt", "w") as f:
        f.write("Sample\tMSE\tMAE\tMaxErr\tTrue_nm\tPred_nm\n")
        for r in results:
            true_str = " ".join(str(v) for v in r["true_nm"])
            pred_str = " ".join(str(v) for v in r["pred_nm"])
            f.write(f"{r['sample']}\t{r['mse']:.6f}\t{r['mae']:.4f}\t"
                    f"{r['max_err']:.4f}\t[{true_str}]\t[{pred_str}]\n")
    print("\nFull results saved to verification_results.txt")

print("=== Done ===")
