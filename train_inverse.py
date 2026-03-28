
import numpy as np
import torch
import torch.nn as nn
import gc
from sklearn.metrics import r2_score

print("=== Inverse Training Script ===")

s_min = np.load('s_min.npy')
s_max = np.load('s_max.npy')

structures = np.load('structures.npy')
spectra    = np.load('spectra.npy')
print(f"Loaded: {structures.shape}, {spectra.shape}")

structures_norm = (2*(structures - s_min)/(s_max - s_min) - 1).astype(np.float32)
spectra_norm    = spectra.astype(np.float32)
del structures, spectra
gc.collect()

X = torch.from_numpy(structures_norm)
Y = torch.from_numpy(spectra_norm)
del structures_norm, spectra_norm
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

n = len(X)
idx       = torch.randperm(n, generator=torch.Generator().manual_seed(42))
n_train   = int(0.70*n)
n_val     = int(0.15*n)
n_test    = n - n_train - n_val
idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train+n_val]
idx_test  = idx[n_train+n_val:]
print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, A, B, idx):
        self.A, self.B, self.idx = A, B, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.A[j], self.B[j]

inv_train = torch.utils.data.DataLoader(
    IndexDataset(Y, X, idx_train), batch_size=1024, shuffle=True,  num_workers=0)
inv_val   = torch.utils.data.DataLoader(
    IndexDataset(Y, X, idx_val),   batch_size=1024, num_workers=0)
inv_test  = torch.utils.data.DataLoader(
    IndexDataset(Y, X, idx_test),  batch_size=1024, num_workers=0)

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

fwd_net = ForwardNet().to(device)
fwd_net.load_state_dict(torch.load("best_forward_net_v2.pth", map_location=device))
for p in fwd_net.parameters():
    p.requires_grad = False
fwd_net.eval()
print("Forward net loaded and frozen")

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

inv_net   = InverseNet().to(device)
inv_opt   = torch.optim.AdamW(inv_net.parameters(), lr=3e-4, weight_decay=1e-5)
inv_sched = torch.optim.lr_scheduler.CosineAnnealingLR(inv_opt, T_max=300, eta_min=1e-5)
criterion = nn.MSELoss()
alpha     = 0.3
print(f"Inverse net params: {sum(p.numel() for p in inv_net.parameters()):,}")

def inv_run(loader, opt=None):
    inv_net.train() if opt else inv_net.eval()
    total, td, tt = 0, 0, 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for spec_b, struct_b in loader:
            spec_b, struct_b = spec_b.to(device), struct_b.to(device)
            pred_s = inv_net(spec_b)
            d_loss = criterion(pred_s, struct_b)
            t_loss = criterion(fwd_net(pred_s), spec_b)
            loss   = alpha*d_loss + (1-alpha)*t_loss
            if opt:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(inv_net.parameters(), 1.0)
                opt.step()
            total += loss.item()*len(spec_b)
            td    += d_loss.item()*len(spec_b)
            tt    += t_loss.item()*len(spec_b)
    n = len(loader.dataset)
    return total/n, td/n, tt/n

best_inv = float("inf")
print("\n── Training ──")
for epoch in range(300):
    tr, td, tt = inv_run(inv_train, inv_opt)
    va, vd, vt = inv_run(inv_val)
    inv_sched.step()

    if va < best_inv:
        best_inv = va
        torch.save(inv_net.state_dict(), "best_inverse_net_v2.pth")

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Total: {tr:.6f} | Direct: {td:.6f} | Tandem: {tt:.6f} | Val: {va:.6f}", flush=True)

print(f"\nBest Val Loss: {best_inv:.6f}")

inv_net.load_state_dict(torch.load("best_inverse_net_v2.pth"))
inv_net.eval()

all_true, all_pred = [], []
with torch.no_grad():
    for spec_b, struct_b in inv_test:
        all_pred.append(inv_net(spec_b.to(device)).cpu().numpy())
        all_true.append(struct_b.numpy())

all_true = np.vstack(all_true)
all_pred = np.vstack(all_pred)
true_nm  = (all_true+1)/2 * (s_max-s_min) + s_min
pred_nm  = (all_pred+1)/2 * (s_max-s_min) + s_min

r2 = r2_score(all_true, all_pred, multioutput="raw_values")
print("\nR2 per layer:")
for i, v in enumerate(r2):
    print(f"  Layer {i+1}: {v:.4f}")
print(f"  Mean R2: {r2.mean():.4f}")

np.savetxt("true_structures_nm.txt", true_nm[:100], fmt="%.1f")
np.savetxt("pred_structures_nm.txt", pred_nm[:100], fmt="%.1f")
print("\nSaved true/pred to txt files")
print("=== Done ===")
