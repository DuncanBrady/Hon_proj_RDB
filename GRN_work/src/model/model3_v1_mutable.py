import os, subprocess, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import cuda
import pandas as pd

# ---------------- GPU utils ----------------
def get_num_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        return len(out.strip().split("\n"))
    except Exception:
        return 0

num_gpus = get_num_gpus()
if num_gpus >= 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

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
            if need <= 0: continue
            numel = need // (torch.finfo(dtype).bits // 8)
            if numel <= 0: continue
            t = torch.empty(numel, dtype=dtype, device=f'cuda:{d}')
            _reserve_tensors.append(t)

def release_reserved_mem():
    global _reserve_tensors
    _reserve_tensors = []
    torch.cuda.empty_cache()

# ---------------- Data ----------------
path = os.path.expanduser("C:\\Users\\rdbra\\Documents\\honoursProject\\code_base\\Data\\")
os.chdir(path)
data_file = os.path.join(path, 'sct_matrix_transposed.npz')
cell_gene = np.load(data_file)['arr_0']
gene_cell = cell_gene.T.astype(np.float32)
gene_tensor = torch.tensor(gene_cell)
assert gene_tensor.dim() == 2
print(f"Input tensor shape: {gene_tensor.shape} (genes x cells)")

# ---------------- Dataset ----------------
class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return idx

# ---------------- Huber Loss ----------------
def weighted_huber_loss(reconstructed, original, weight, delta=1.0):
    error = reconstructed - original
    is_small = torch.abs(error) <= delta
    small_loss = 0.5 * error**2
    large_loss = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(is_small, small_loss, large_loss)
    return torch.mean(weight * loss)

def loss_function(reconstructed, original):
    weight = torch.where(original > 0, 1.0, 0.5).to(original.device)
    return weighted_huber_loss(reconstructed, original, weight)

# ---------------- Model ----------------
class NMFReconstructor(nn.Module):
    def __init__(self, n_genes, n_cells, rank=512, mlp_hidden=0):
        super().__init__()
        self.G  = nn.Parameter(torch.randn(n_genes, rank) * 0.1)
        self.C  = nn.Parameter(torch.randn(n_cells, rank) * 0.1)
        self.bg = nn.Parameter(torch.zeros(n_genes))
        self.bc = nn.Parameter(torch.zeros(n_cells))
        if mlp_hidden > 0:
            self.mlp = nn.Sequential(
                nn.Linear(rank, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, 1)
            )
        else:
            self.mlp = None

    def forward(self, gene_idx=None):
        if gene_idx is None:
            base = self.G @ self.C.T
            return base + self.bg[:, None] + self.bc[None, :]
        g = self.G[gene_idx]
        base = g @ self.C.T
        return base + self.bg[gene_idx][:, None] + self.bc[None, :]

# ---------------- Learning Rate Scheduler ----------------
class CustomMultiStageCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, steps_per_epoch, epochs_per_cycle=50, last_epoch=-1):
        self.total_steps = total_epochs * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.epochs_per_cycle = epochs_per_cycle
        self.steps_per_cycle = epochs_per_cycle * steps_per_epoch

        self.stage_cycles = [
            (int(0.1 * self.total_steps), (0.005, 0.001)),
            (int(0.2 * self.total_steps), (0.001, 0.0005)),
            (int(0.3 * self.total_steps), (0.0005, 0.0001)),
            (int(0.4 * self.total_steps), (0.0001, 0.00005))
        ]

        self.stage_boundaries = []
        self.lr_stages = []
        total = 0
        for steps, lr in self.stage_cycles:
            total += steps
            self.stage_boundaries.append(total)
            self.lr_stages.append(lr)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        stage_idx = 0
        offset = 0
        for i, boundary in enumerate(self.stage_boundaries):
            if step < boundary:
                stage_idx = i
                break
            offset = boundary

        lr_start, lr_end = self.lr_stages[stage_idx]
        local_step = step - offset
        cycle_step = local_step % self.steps_per_cycle
        cycle_progress = cycle_step / self.steps_per_cycle
        cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_progress))
        return [lr_end + cosine_factor * (lr_start - lr_end) for _ in self.optimizer.param_groups]

# ---------------- Metrics ----------------
def mean_abs_grad(model):
    with torch.no_grad():
        grads = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(grads)) if grads else 0.0

def dynamic_acc_thresh(y_true):
    return torch.where(
        y_true == 0, 0.05,
        torch.where((y_true > 0) & (y_true < 1), 0.1, 0.1 * y_true)
    )

# ---------------- Training ----------------
def train_AE(dataloader, full_tensor, device, epochs=200, log_filename="nmf_metrices.txt"):
    n_genes, n_cells = full_tensor.shape
    model = NMFReconstructor(n_genes=n_genes, n_cells=n_cells, rank=512)
    model = model.to(device)
    full_tensor = full_tensor.to(device)
    release_reserved_mem()

    optimizer = optim.AdamW(model.parameters())
    scheduler = CustomMultiStageCyclicLR(
        optimizer,
        total_epochs=epochs,
        steps_per_epoch=len(dataloader),
        epochs_per_cycle=50
    )

    with open(log_filename, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\tAbsGrad\tLR\n")

    for epoch in range(1, epochs + 1):
        total_loss = total_acc = total_absG = total_samples = 0
        for batch_idx in dataloader:
            g_idx = batch_idx.to(device)
            x_batch = full_tensor[g_idx]
            recon = model(gene_idx=g_idx)

            loss = loss_function(recon, x_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_absG += mean_abs_grad(model)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                acc_thresh = dynamic_acc_thresh(x_batch)
                acc = (torch.abs(recon - x_batch) < acc_thresh).float().mean().item()
                total_acc += acc * x_batch.numel()
            total_loss += loss.item() * x_batch.numel()
            total_samples += x_batch.numel()

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_absG = total_absG / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f} - Acc: {avg_acc:.4f} - AbsGrad: {avg_absG:.6f} - LR: {current_lr:.6f}")
            with open(log_filename, "a") as f:
                f.write(f"{epoch}\t{avg_loss:.6f}\t{avg_acc:.4f}\t{avg_absG:.6f}\t{current_lr:.6f}\n")

    return model

# ---------------- Run ----------------
if __name__ == "__main__":
    reserve_mem_all_gpus(TARGET_FRACTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}")
    torch.backends.cudnn.benchmark = True

    dataset = IndexedDataset(gene_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    
    model = train_AE(dataloader, gene_tensor, device, log_filename="nmf_metrices.txt")

    # Save model
    torch.save(model.state_dict(), "nmf_model.pth")
    print("Model weights saved to 'nmf_model.pth'")

    # Reconstruct full matrix
    with torch.no_grad():
        X_hat = []
        for s in range(0, gene_tensor.shape[0], 1024):
            idx = torch.arange(s, min(s + 1024, gene_tensor.shape[0]), device=device)
            X_hat.append(model(gene_idx=idx).cpu())
        X_hat = torch.cat(X_hat, dim=0)

    torch.save(X_hat, "X_hat.pt")
    print(f"Reconstructed matrix saved. Shape: {X_hat.shape}")

    # Save latent embeddings
    torch.save(model.G.detach().cpu(), "gene_embeddings_G.pt")
    torch.save(model.C.detach().cpu(), "cell_embeddings_C.pt")
    pd.DataFrame(model.G.detach().cpu().numpy()).to_csv("gene_embeddings_G.csv", index=False)
    pd.DataFrame(model.C.detach().cpu().numpy()).to_csv("cell_embeddings_C.csv", index=False)
    print("Saved gene and cell embeddings (PT + CSV formats)")
