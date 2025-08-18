# All necessary imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch import cuda
import subprocess
import pandas as pd

# GPU utility functions
def get_num_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        return len(output.strip().split("\n"))
    except Exception as e:
        print("Could not detect GPUs:", e)
        return 0

num_gpus = get_num_gpus()
if num_gpus >= 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
else:
    print("No GPUs found. Running on CPU.")

TARGET_FRACTION = 0.80
_reserve_tensors = []

def reserve_mem_all_gpus(target_frac=TARGET_FRACTION, dtype=torch.float32):
    ndev = cuda.device_count()
    for d in range(ndev):
        with cuda.device(d):
            free_bytes, total_bytes = cuda.mem_get_info()
            want = int(total_bytes * target_frac)
            already = total_bytes - free_bytes
            need = max(want - already, 0)
            if need <= 0:
                continue
            numel = need // (torch.finfo(dtype).bits // 8)
            if numel <= 0:
                continue
            t = torch.empty(numel, dtype=dtype, device=f'cuda:{d}')
            _reserve_tensors.append(t)

def release_reserved_mem():
    global _reserve_tensors
    _reserve_tensors = []
    torch.cuda.empty_cache()

# Load data
path = os.path.expanduser("/g/data/yr31/mp8713/")
data_file = os.path.join(path, 'sct_matrix_transposed.npz')
cell_gene = np.load(data_file)['arr_0']
gene_cell = cell_gene.T.astype(np.float32)
gene_tensor = torch.tensor(gene_cell)
print(f"Input tensor shape: {gene_tensor.shape} (genes x cells)")

# Dataset
class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

# Hybrid Loss (Weighted MSE + BCE)
class WeightedHybridLoss(nn.Module):
    def __init__(self, pos_weight=100.0, bce_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, recon, original):
        mse_weight = torch.where(original > 0, self.pos_weight, 1.0).to(original.device)
        mse_loss = ((recon - original) ** 2 * mse_weight).mean()
        bce_loss = self.bce_loss(recon, (original > 0).float())
        return mse_loss + self.bce_weight * bce_loss

# Two-layer fully-connected autoencoder with unequal, dynamic hidden dims
class TwoLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super().__init__()
        # Encoder: input → hidden_dim1 → hidden_dim2 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim2, latent_dim),
            nn.LeakyReLU(0.01),
        )
        # Decoder: latent_dim → hidden_dim2 → hidden_dim1 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim1, input_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def mean_abs_grad(model: nn.Module) -> float:
    with torch.no_grad():
        vals = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(vals)) if vals else 0.0

def dynamic_acc_thresh(y_true):
    return torch.where(
        y_true == 0, 0.05,
        torch.where((y_true > 0) & (y_true < 1), 0.1, 0.1 * y_true)
    )

def train_AE(dataloader, device, epochs=4000, log_filename="test1_metrics.txt"):
    # Constraint bounds
    min_lr, max_lr = 1e-6, 1e-2
    lr_factor = 0.5

    min_h1, max_h1 = 512, 4096
    min_h2, max_h2 = 256, 2048
    min_latent, max_latent = 64, 512

    min_pw, max_pw = 1.0, 200.0
    min_bw, max_bw = 0.1, 20.0

    # Initial hyperparameters
    sample_x, _ = next(iter(dataloader))
    input_dim = sample_x.shape[1]

    hidden_dim1 = 2048
    hidden_dim2 = 1024
    latent_dim  = 128
    pos_weight  = 100.0
    bce_weight  = 10.0
    lr          = 0.005

    # Clamp initial values
    hidden_dim1 = int(np.clip(hidden_dim1, min_h1, max_h1))
    hidden_dim2 = int(np.clip(hidden_dim2, min_h2, max_h2))
    latent_dim  = int(np.clip(latent_dim, min_latent, max_latent))
    pos_weight  = float(np.clip(pos_weight, min_pw, max_pw))
    bce_weight  = float(np.clip(bce_weight, min_bw, max_bw))
    lr          = float(np.clip(lr, min_lr, max_lr))

    best_acc = 0.0

    def build():
        model = TwoLayerAutoencoder(input_dim, hidden_dim1, hidden_dim2, latent_dim).to(device)
        model.apply(initialize_weights)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        opt = optim.AdamW(model.parameters(), lr=lr)
        sched = OneCycleLR(
            opt, max_lr=lr,
            total_steps=epochs * len(dataloader),
            pct_start=0.1, anneal_strategy='cos',
            div_factor=10.0, final_div_factor=100.0,
            cycle_momentum=False
        )
        return model, opt, sched

    release_reserved_mem()
    model, optimizer, scheduler = build()
    loss_fn = WeightedHybridLoss(pos_weight=pos_weight, bce_weight=bce_weight)
    fixed_orig = gene_tensor[:20].to(device)

    with open(log_filename, "w") as f:
        f.write("Epoch\tLoss\tAcc\tAbsGrad\tLR\n")

    for epoch in range(1, epochs + 1):
        tot_loss = tot_acc = tot_abs = 0.0
        samples = updates = 0

        for x, _ in dataloader:
            x = x.float().to(device, non_blocking=True)
            recon, _ = model(x)
            loss = loss_fn(recon, x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            tot_abs += mean_abs_grad(model)
            updates += 1
            optimizer.step()
            scheduler.step()

            bs = x.size(0)
            tot_loss += loss.item() * bs
            with torch.no_grad():
                acc = (torch.abs(recon - x) < dynamic_acc_thresh(x)).float().mean().item()
                tot_acc += acc * bs
            samples += bs

        avg_loss = tot_loss / samples
        avg_acc = tot_acc / samples
        avg_abs = tot_abs / updates
        curr_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0:
            improved   = avg_acc > best_acc
            stagnation = avg_abs < 1e-5

            if not improved:
                lr *= lr_factor
                hidden_dim1 += 128
                hidden_dim2 += 64
                pos_weight *= 0.9
                bce_weight *= 0.9
            elif stagnation:
                lr /= lr_factor
                hidden_dim1 -= 128
                hidden_dim2 -= 64
                pos_weight *= 1.1
                bce_weight *= 1.1
            else:
                best_acc = avg_acc

            # clamp all
            hidden_dim1 = int(np.clip(hidden_dim1, min_h1, max_h1))
            hidden_dim2 = int(np.clip(hidden_dim2, min_h2, max_h2))
            latent_dim  = int(np.clip(latent_dim, min_latent, max_latent))
            pos_weight  = float(np.clip(pos_weight, min_pw, max_pw))
            bce_weight  = float(np.clip(bce_weight, min_bw, max_bw))
            lr          = float(np.clip(lr, min_lr, max_lr))

            print(f"[Epoch {epoch}] → lr:{lr:.5g} h1:{hidden_dim1} h2:{hidden_dim2} "
                  f"pw:{pos_weight:.1f} bw:{bce_weight:.1f}")

            model, optimizer, scheduler = build()
            loss_fn = WeightedHybridLoss(pos_weight=pos_weight, bce_weight=bce_weight)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} Loss:{avg_loss:.4f} Acc:{avg_acc:.4f} "
                  f"AbsGrad:{avg_abs:.5f} lr:{curr_lr:.5g}")
            with open(log_filename, "a") as f:
                f.write(f"{epoch}\t{avg_loss:.4f}\t{avg_acc:.4f}\t{avg_abs:.5f}\t{curr_lr:.5g}\n")

    return model

if __name__ == "__main__":
    reserve_mem_all_gpus(TARGET_FRACTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ds = IndexedDataset(gene_tensor)
    dl = DataLoader(ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

    model = train_AE(dl, device, epochs=2000, log_filename="test1_two_layer_dynamic.txt")
    torch.save(model.state_dict(), 'test1_two_layer_dynamic.pth')
    print("Saved model.")

    model.eval()
    with torch.no_grad():
        gt = gene_tensor.to(device)
        recon, emb = model(gt)
        emb = emb.cpu()
        recon = recon.cpu().numpy()

    print("Final embeddings shape:", emb.shape)
    torch.save(emb, "test1_two_layer_dynamic_embeddings.pt")
    np.savez_compressed("test1_two_layer_dynamic_recon.npz", reconstructed_matrix=recon)
