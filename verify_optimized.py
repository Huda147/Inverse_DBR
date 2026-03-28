
import numpy as np
import torch
import torch.nn as nn
import subprocess, os, gc
from scipy.optimize import minimize

print("=== S4 Verification with Local Optimization ===")

s_min = np.load("s_min.npy")
s_max = np.load("s_max.npy")
spectra    = np.load("spectra.npy").astype(np.float32)
structures = np.load("structures.npy").astype(np.float32)

n = len(spectra)
idx      = torch.randperm(n, generator=torch.Generator().manual_seed(42))
n_train  = int(0.70*n)
n_val    = int(0.15*n)
idx_test = idx[n_train+n_val:].numpy()
test_idx     = idx_test[:20]
test_spectra = spectra[test_idx]
test_structs = structures[test_idx]

# ── Models ────────────────────────────────────────────────────────────
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

device = torch.device("cpu")

fwd_net = ForwardNet()
fwd_net.load_state_dict(torch.load("best_forward_net_v2.pth", map_location="cpu"))
fwd_net.eval()
for p in fwd_net.parameters():
    p.requires_grad = False

inv_net = InverseNet()
inv_net.load_state_dict(torch.load("best_inverse_net_v2.pth", map_location="cpu"))
inv_net.eval()

# ── Helper: run S4 for one structure ──────────────────────────────────
def run_s4(struct_nm):
    struct_nm = np.clip(np.round(struct_nm), 25, 200).astype(int)
    line = "\t".join(str(v) for v in struct_nm)
    with open("verify_input.txt", "w") as f:
        f.write(line + "\n")
    if os.path.exists("verified_spectrum.dat"):
        os.remove("verified_spectrum.dat")
    result = subprocess.run(
        ["s4", "verify.lua"],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not os.path.exists("verified_spectrum.dat"):
        return None
    with open("verified_spectrum.dat", "r") as f:
        vals = [float(x) for x in f.read().strip().split()]
    if len(vals) != 244:
        return None
    return np.array(vals, dtype=np.float32)

# ── Neural net forward as fast proxy for optimization ─────────────────
def nn_loss(struct_norm, target_spec):
    x = torch.tensor(struct_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = fwd_net(x).numpy()[0]
    return float(np.mean((pred - target_spec)**2))

# ── Main loop ─────────────────────────────────────────────────────────
print("\n── Running verification with NN-guided optimization ──")

results_before, results_after = [], []

for i, (true_spec, true_struct) in enumerate(zip(test_spectra, test_structs)):
    # Step 1: inverse net prediction
    with torch.no_grad():
        pred_norm = inv_net(torch.tensor(true_spec).unsqueeze(0)).numpy()[0]
    pred_nm = (pred_norm + 1) / 2 * (s_max - s_min) + s_min
    pred_nm = np.clip(np.round(pred_nm), 25, 200).astype(int)

    # Step 2: S4 verification BEFORE optimization
    s4_before = run_s4(pred_nm)
    if s4_before is not None:
        mse_b = float(np.mean((s4_before - true_spec)**2))
        mae_b = float(np.mean(np.abs(s4_before - true_spec)))
        results_before.append(mse_b)
    else:
        mse_b, mae_b = None, None

    # Step 3: Local optimization using NN forward as proxy
    # Optimize in normalized [-1,1] space using Nelder-Mead
    def objective(x):
        return nn_loss(x, true_spec)

    opt_result = minimize(
        objective,
        x0    = pred_norm,
        method= "Nelder-Mead",
        options= {"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6})

    opt_norm = np.clip(opt_result.x, -1, 1)
    opt_nm   = (opt_norm + 1) / 2 * (s_max - s_min) + s_min
    opt_nm   = np.clip(np.round(opt_nm), 25, 200).astype(int)

    # Step 4: S4 verification AFTER optimization
    s4_after = run_s4(opt_nm)
    if s4_after is not None:
        mse_a = float(np.mean((s4_after - true_spec)**2))
        mae_a = float(np.mean(np.abs(s4_after - true_spec)))
        results_after.append(mse_a)
    else:
        mse_a, mae_a = None, None

    print(f"  Sample {i+1:2d} | "
          f"Before — MSE: {mse_b:.6f} MAE: {mae_b:.4f} | "
          f"After  — MSE: {mse_a:.6f} MAE: {mae_a:.4f} | "
          f"Pred: {pred_nm} | Opt: {opt_nm}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n── Summary ──")
print(f"  Mean MSE before optimization: {np.mean(results_before):.6f}")
print(f"  Mean MSE after  optimization: {np.mean(results_after):.6f}")
print(f"  Improvement: {(1 - np.mean(results_after)/np.mean(results_before))*100:.1f}%")
print("=== Done ===")
