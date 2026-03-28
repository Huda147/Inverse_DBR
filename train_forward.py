
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import gc

# ── Load from .npy (10x faster, less memory than loadtxt) ─────────────
structures = np.load('structures.npy')
spectra    = np.load('spectra.npy')
print("Loaded:", structures.shape, spectra.shape)

# ── Normalize ─────────────────────────────────────────────────────────
s_min = structures.min(axis=0)
s_max = structures.max(axis=0)
np.save('s_min.npy', s_min)  # save for later denormalization
np.save('s_max.npy', s_max)

structures_norm = (2 * (structures - s_min) / (s_max - s_min) - 1).astype(np.float32)
spectra_norm    = spectra.astype(np.float32)

del structures, spectra
gc.collect()
print("Memory freed after normalization")

# ── Tensors — keep on CPU, move to GPU only in batches ────────────────
X = torch.from_numpy(structures_norm)
Y = torch.from_numpy(spectra_norm)

del structures_norm, spectra_norm
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Split indices only — avoids data duplication ──────────────────────
n = len(X)
idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train+n_val]
idx_test  = idx[n_train+n_val:]

print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

# ── Simple indexed dataset ────────────────────────────────────────────
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx):
        self.X = X
        self.Y = Y
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.X[j], self.Y[j]

# Forward: input=structure, target=spectrum
fwd_train = DataLoader(IndexDataset(X, Y, idx_train), batch_size=1024, shuffle=True)
fwd_val   = DataLoader(IndexDataset(X, Y, idx_val),   batch_size=1024)
fwd_test  = DataLoader(IndexDataset(X, Y, idx_test),  batch_size=1024)

# ── Model ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
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

fwd_net   = ForwardNet().to(device)
optimizer = torch.optim.AdamW(fwd_net.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
criterion = nn.MSELoss()
print(f"Params: {sum(p.numel() for p in fwd_net.parameters()):,}")

# ── Training ──────────────────────────────────────────────────────────
def run_epoch(model, loader, opt=None):
    model.train() if opt else model.eval()
    total = 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if opt:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(xb)
    return total / len(loader.dataset)

print("\n── Training Forward Network ──")
best_val = float('inf')
train_losses, val_losses = [], []

for epoch in range(300):
    tr = run_epoch(fwd_net, fwd_train, optimizer)
    va = run_epoch(fwd_net, fwd_val)
    scheduler.step()
    train_losses.append(tr)
    val_losses.append(va)

    if va < best_val:
        best_val = va
        torch.save(fwd_net.state_dict(), 'best_forward_net_v2.pth')

    if (epoch+1) % 30 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {tr:.6f} | Val: {va:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

print(f"\nBest Forward Val MSE: {best_val:.6f}")

# # ── Plot loss ─────────────────────────────────────────────────────────
# plt.figure(figsize=(8,4))
# plt.plot(train_losses, label='Train')
# plt.plot(val_losses,   label='Val')
# plt.xlabel('Epoch'); plt.ylabel('MSE')
# plt.title('Forward Network Loss')
# plt.legend(); plt.grid(True); plt.tight_layout()
# plt.savefig('forward_loss.png', dpi=80)
# plt.show()
# plt.close()

# # ── Visual check — 1 sample only ──────────────────────────────────────
# fwd_net.load_state_dict(torch.load('best_forward_net_v2.pth'))
# fwd_net.eval()

# xb, yb = next(iter(fwd_test))
# with torch.no_grad():
#     pred = fwd_net(xb[:1].to(device)).cpu().numpy()[0]
# true = yb[:1].numpy()[0]

# wl = np.linspace(0.40, 0.705, 61)
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i, (ax, lbl) in enumerate(zip(axes, ['T_amor','R_amor','T_crys','R_crys'])):
#     ax.plot(wl, true[i::4], label='True', lw=2)
#     ax.plot(wl, pred[i::4], label='Pred', lw=1.5, ls='--')
#     ax.set_title(lbl); ax.legend(); ax.grid(True)
# plt.suptitle('Forward Net v2: Test Sample')
# plt.tight_layout()
# plt.savefig('forward_check.png', dpi=80)
# plt.show()
# plt.close()
