# =========================================================
# VQ Auto-Decoder for SCT-normalised scRNA-seq
# - One unique latent vector per gene (nn.Embedding)
# - VQ quantization on the gene latent (keeps VQ-VAE behaviour)
# - Decoder maps quantized latent -> expression row (cells)
# - Non-negative recon via softplus(x)-ln(2)
# - Weighted MSE (emphasise non-zeros) + BCE(logits) on presence
# - Gradient clipping, OneCycleLR, codebook usage/perplexity logging
# - Saves z_e (prequant), z_q (quantized), code indices, reconstruction
# - Adds fixed-batch preview every 100 epochs and prints full recon at end
# =========================================================

import os, math, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch import cuda
import pandas as pd

# ----------------------------
# GPU helpers
# ----------------------------
def get_num_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        return len([ln for ln in out.strip().split("\n") if ln.strip()])
    except Exception:
        return 0

num_gpus = get_num_gpus()
if num_gpus >= 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
else:
    print("No GPUs found. Running on CPU.")

TARGET_FRACTION = 0.80
_reserve_tensors = []
def reserve_mem_all_gpus(target_frac=TARGET_FRACTION, dtype=torch.float32):
    global _reserve_tensors
    for d in range(cuda.device_count()):
        with cuda.device(d):
            free_bytes, total_bytes = cuda.mem_get_info()
            want = int(total_bytes * target_frac)
            already = total_bytes - free_bytes
            need = max(want - already, 0)
            if need <= 0: continue
            numel = need // (torch.finfo(dtype).bits // 8)
            if numel <= 0: continue
            _reserve_tensors.append(torch.empty(numel, dtype=dtype, device=f"cuda:{d}"))
def release_reserved_mem():
    global _reserve_tensors
    _reserve_tensors = []
    torch.cuda.empty_cache()

def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m

# ----------------------------
# Load SCT-normalised expression: genes x cells (non-negative)
# ----------------------------
path = os.path.expanduser("/g/data/yr31/mp8713/")
data_file = os.path.join(path, "sct_matrix_transposed.npz")  # assumed (cells x genes) transposed previously
cell_gene = np.load(data_file)["arr_0"]
gene_cell = cell_gene.T.astype(np.float32)   # shape: [G, C]
gene_tensor = torch.tensor(gene_cell)        # each sample is one gene row across cells
G, C = gene_tensor.shape
print(f"Expression tensor: {gene_tensor.shape} (genes x cells)")

# ----------------------------
# Dataset
# ----------------------------
class IndexedDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data
    def __len__(self):
        return int(self.data.shape[0])
    def __getitem__(self, idx):
        return self.data[idx], idx  # return row + its gene id

# ----------------------------
# Loss (same as your SCT setup)
# ----------------------------
class WeightedHybridLoss(nn.Module):
    """
    - Recon values = clamp_min(softplus(logits) - ln2, 0)  (maps logits≈0 -> 0)
    - Weighted MSE (heavier on non-zeros)
    - BCEWithLogits on presence (pos_weight)
    """
    def __init__(self, pos_weight=100.0, bce_weight=10.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.bce_weight = float(bce_weight)
        self.register_buffer("pos_weight_tensor", torch.tensor(self.pos_weight, dtype=torch.float32))

    @staticmethod
    def vals_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return (F.softplus(logits) - math.log(2.0)).clamp_min(0.0)

    # alias to match your snippet name
    @staticmethod
    def _nonneg_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return WeightedHybridLoss.vals_from_logits(logits)

    def forward(self, recon_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon_vals = self.vals_from_logits(recon_logits)
        w = torch.where(target > 0, self.pos_weight, 1.0).to(target.device)
        mse = ((recon_vals - target) ** 2 * w).mean()
        bce = F.binary_cross_entropy_with_logits(
            recon_logits, (target > 0).float(),
            pos_weight=self.pos_weight_tensor.to(target.device)
        )
        return mse + self.bce_weight * bce, mse, bce

# ----------------------------
# Simple metrics
# ----------------------------
def mean_abs_grad(model: nn.Module) -> float:
    with torch.no_grad():
        vals = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(vals)) if vals else 0.0

def dynamic_acc_thresh(y_true: torch.Tensor) -> torch.Tensor:
    # tighter at zero, looser for large values (same as before)
    return torch.where(
        y_true == 0, 0.05,
        torch.where((y_true > 0) & (y_true < 1), 0.1, 0.1 * y_true)
    )

# ----------------------------
# Vector Quantizer
# ----------------------------
class VectorQuantizer(nn.Module):
    """VQ layer with per-sample loss + straight-through estimator."""
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e: torch.Tensor):
        # z_e: [B, D]
        flat = z_e.view(-1, self.embedding_dim)
        e = self.embedding.weight                            # [K, D]
        z_sq = (flat ** 2).sum(dim=1, keepdim=True)          # [B, 1]
        e_sq = (e ** 2).sum(dim=1)                           # [K]
        ze = flat @ e.t()                                    # [B, K]
        dist = z_sq - 2 * ze + e_sq.unsqueeze(0)             # [B, K]

        indices = torch.argmin(dist, dim=1)                  # [B]
        z_q = e[indices]                                     # [B, D]

        # Losses
        embed_loss_vec  = ((z_q - z_e.detach()) ** 2).mean(dim=1)   # [B]
        commit_loss_vec = ((z_e - z_q.detach()) ** 2).mean(dim=1)   # [B]
        vq_loss_vec = embed_loss_vec + self.beta * commit_loss_vec  # [B]

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, vq_loss_vec, indices

# ----------------------------
# VQ Auto-Decoder (no encoder MLP, per-gene nn.Embedding -> VQ -> decoder)
# ----------------------------
class VQAutoDecoder(nn.Module):
    def __init__(self, num_genes: int, num_cells: int,
                 latent_dim: int = 128, hidden_dim1: int = 2048, hidden_dim2: int = 1024,
                 codebook_size: int = 1024, vq_beta: float = 0.25):
        super().__init__()
        # One unique continuous latent per gene
        self.gene_embed = nn.Embedding(num_genes, latent_dim)
        nn.init.normal_(self.gene_embed.weight, mean=0.0, std=0.02)
        self.ln = nn.LayerNorm(latent_dim)

        # VQ layer on the latent
        self.vq = VectorQuantizer(codebook_size, latent_dim, beta=vq_beta)

        # Decoder (latent -> expression row logits over cells)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim2, hidden_dim1), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim1, num_cells),   # logits; map with softplus-ln2 in the loss/metrics
        )

    def forward(self, gene_idx: torch.Tensor):
        # gene_idx: [B] long
        z_e = self.ln(self.gene_embed(gene_idx))                # [B, D]
        z_q, vq_loss_vec, code_idx = self.vq(z_e)               # [B, D], [B], [B]
        recon_logits = self.dec(z_q)                            # [B, C]
        return recon_logits, z_q, vq_loss_vec, z_e, code_idx

# ----------------------------
# Reconstruct full matrix using the decoder only
# ----------------------------
@torch.no_grad()
def reconstruct_full(model: nn.Module, device: torch.device, batch_size: int = 256):
    model.eval()
    G, C = gene_tensor.shape
    recon = np.empty((G, C), dtype=np.float32)
    loader = DataLoader(IndexedDataset(gene_tensor), batch_size=batch_size, shuffle=False)
    for _, idx in loader:
        idx = idx.to(device)
        logits, *_ = model(idx)
        vals = WeightedHybridLoss.vals_from_logits(logits).cpu().numpy()
        recon[idx.cpu().numpy(), :] = vals
    return recon

# ----------------------------
# Training
# ----------------------------
def train_vq_autodecoder(dataloader, device, epochs=800, log_filename="vq_autodecoder_metrics.txt"):
    G, C = gene_tensor.shape
    # Model
    model = VQAutoDecoder(num_genes=G, num_cells=C,
                          latent_dim=128, hidden_dim1=2048, hidden_dim2=1024,
                          codebook_size=1024, vq_beta=0.25).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optim/sched
    lr = 0.005
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * len(dataloader),
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10.0, final_div_factor=100.0,
        cycle_momentum=False
    )
    clip_norm = 1.0

    # Loss
    loss_fn = WeightedHybridLoss(pos_weight=200.0, bce_weight=20.0)
    vq_weight_start, vq_weight_end, warmup_epochs = 1e-3, 1.0, int(0.25 * epochs)

    # ----- Fixed subset for preview -----
    fixed_idx = torch.arange(min(20, G), dtype=torch.long)       # first 20 genes as a stable preview
    fixed_orig = gene_tensor[fixed_idx].clone()                   # original rows (CPU tensor)

    with open(log_filename, "w") as f:
        f.write("Epoch\tLoss\tRecon\tMSE\tBCE\tVQ\tAcc\tBinAcc\tAbsGrad\tPerp\tVQw\tLR\n")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {k: 0.0 for k in ["loss","recon","mse","bce","vq","acc","binacc","abs"]}
        samples = updates = 0
        code_counts = torch.zeros(unwrap(model).vq.num_embeddings, device=device)

        # VQ warmup
        t = min(1.0, epoch / max(1, warmup_epochs))
        vq_weight = vq_weight_start + t * (vq_weight_end - vq_weight_start)

        for x, idx in dataloader:
            # x: [B, C] targets; idx: [B] gene ids
            x   = x.float().to(device, non_blocking=True)
            idx = idx.to(device)

            logits, z_q, vq_loss_vec, z_e, code_idx = model(idx)
            recon_loss, mse, bce = loss_fn(logits, x)
            vq_loss = vq_loss_vec.mean()

            loss = recon_loss + vq_weight * vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step(); scheduler.step()

            # metrics
            with torch.no_grad():
                vals = WeightedHybridLoss.vals_from_logits(logits)
                acc = (torch.abs(vals - x) < dynamic_acc_thresh(x)).float().mean().item()
                bin_pred = (logits > 0).float()
                bin_true = (x > 0).float()
                bin_acc = (bin_pred == bin_true).float().mean().item()

                binc = torch.bincount(code_idx.view(-1).to(torch.int64),
                                      minlength=unwrap(model).vq.num_embeddings).float()
                code_counts += binc

            bs = x.size(0)
            totals["loss"]   += float(loss.item()) * bs
            totals["recon"]  += float(recon_loss.item()) * bs
            totals["mse"]    += float(mse.item()) * bs
            totals["bce"]    += float(bce.item()) * bs
            totals["vq"]     += float(vq_loss.item()) * bs
            totals["acc"]    += acc * bs
            totals["binacc"] += bin_acc * bs
            totals["abs"]    += mean_abs_grad(model)
            samples += bs
            updates += 1

        # epoch summary
        avg = {k: totals[k] / samples for k in ["loss","recon","mse","bce","vq","acc","binacc"]}
        avg["abs"] = totals["abs"] / max(1, updates)
        curr_lr = scheduler.get_last_lr()[0]

        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = torch.exp(-(probs * (probs.add(1e-8).log())).sum()).item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} "
                  f"Loss:{avg['loss']:.4f} Recon:{avg['recon']:.4f} (MSE:{avg['mse']:.4f} BCE:{avg['bce']:.4f}) "
                  f"VQ:{avg['vq']:.4f} Acc:{avg['acc']:.4f} BinAcc:{avg['binacc']:.4f} "
                  f"AbsGrad:{avg['abs']:.5f} Perp:{perplexity:.1f} VQw:{vq_weight:.3f} lr:{curr_lr:.5g}")
            with open(log_filename, "a") as f:
                f.write(f"{epoch}\t{avg['loss']:.4f}\t{avg['recon']:.4f}\t{avg['mse']:.4f}\t{avg['bce']:.4f}\t"
                        f"{avg['vq']:.4f}\t{avg['acc']:.4f}\t{avg['binacc']:.4f}\t{avg['abs']:.5f}\t"
                        f"{perplexity:.2f}\t{vq_weight:.3f}\t{curr_lr:.5g}\n")

        # ----- Fixed subset preview + (optional) full matrix print -----
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                # forward using fixed indices
                fixed_logits, *_ = model(fixed_idx.to(device))
                recon_values = WeightedHybridLoss._nonneg_from_logits(fixed_logits)

                print("Original [0:20, 0:6]:")
                print(pd.DataFrame(fixed_orig[:, :6].cpu().numpy()))
                print("Reconstructed [0:20, 0:6]:")
                print(pd.DataFrame(recon_values[:, :6].cpu().numpy()))

                # full reconstruction print (can be large/slow)
                full_recon = reconstruct_full(model, device, batch_size=256)
                print("Full reconstructed matrix (genes x cells):")
                print(pd.DataFrame(full_recon))
            model.train()

        if avg["acc"] > best_acc:
            best_acc = avg["acc"]

    return model

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    reserve_mem_all_gpus(TARGET_FRACTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ds = IndexedDataset(gene_tensor)
    dl = DataLoader(ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

    model = train_vq_autodecoder(dl, device, epochs=4000, log_filename="vqvae2_metrics.txt")

    # Save weights
    torch.save(model.state_dict(), "vqvae2_model.pth")
    print("Saved model.")

    # Recon + save
    recon = reconstruct_full(model, device, batch_size=256)
    np.savez_compressed("vqvae2_recon.npz", reconstructed_matrix=recon)

    # Also print the full reconstructed matrix once at the end
    print("Final full reconstructed matrix (genes x cells):")
    print(pd.DataFrame(recon))

    # Export latents for every gene
    model = unwrap(model)
    with torch.no_grad():
        gene_ids = torch.arange(G, device=device)
        z_e = model.ln(model.gene_embed(gene_ids))          # [G, D]
        z_q, _, code_idx = model.vq(z_e)
    torch.save(z_e.cpu(),   "vqvae2_latent_prequant.pt")     # continuous per-gene vectors
    torch.save(z_q.cpu(),   "vqvae2_latent_quantised.pt")    # quantized per-gene vectors
    torch.save(code_idx.cpu(), "vqvae2_code_indices.pt")
    print("Saved reconstructions and latent representations.")
