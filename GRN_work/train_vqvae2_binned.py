import os
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import Dict, Tuple

from src.model.MehransModels.vqvae2 import (
    VQAutoDecoder,
    WeightedHybridLoss,
    VectorQuantizer,
    IndexedDataset,
    reconstruct_full,
    load_expression_tensor,
    unwrap,
    reserve_mem_all_gpus,
    TARGET_FRACTION,
)
from src.preprocess.binning import term_freq_bin

# ------------------------------
# Config
# ------------------------------
class Config:
    data_file: str = os.environ.get("GENE_EXPR_FILE", "C:/Users/rdbra/Documents/honoursProject/code_base/data/sct_matrix_transposed.npz")
    out_dir: str = "vqvae2_binned_results"
    num_bins: int = 5
    batch_size: int = 64
    epochs: int = 200
    lr: float = 5e-3
    vq_beta: float = 0.25
    codebook_size: int = 1024
    latent_dim: int = 128
    hidden_dim1: int = 2048
    hidden_dim2: int = 1024
    pos_weight: float = 200.0
    bce_weight: float = 20.0
    vq_w_start: float = 1e-3
    vq_w_end: float = 1.0
    vq_warmup_frac: float = 0.25
    num_workers: int = 4

CFG = Config()

# ------------------------------
# Utilities for binning metrics
# ------------------------------

def compute_bin_edges(non_zero_vals: np.ndarray, num_bins: int) -> np.ndarray:
    return np.linspace(non_zero_vals.min(), non_zero_vals.max(), num_bins + 1)

def apply_bin_edges(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    b = np.digitize(values, bin_edges, right=False)
    b[values == values.max()] = len(bin_edges) - 1  # ensure max lands in last bin
    return b

def tensor_to_bins(t: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    # t shape (...)
    cpu = t.detach().cpu().numpy()
    non_zero_mask = cpu > 0
    binned = np.zeros_like(cpu, dtype=np.int64)
    nz_vals = cpu[non_zero_mask]
    if nz_vals.size > 0:
        binned_vals = apply_bin_edges(nz_vals, bin_edges)
        binned[non_zero_mask] = binned_vals
    return torch.from_numpy(binned)

class ConfusionMatrixAccumulator:
    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.matrix = torch.zeros((num_bins+1, num_bins+1), dtype=torch.int64)  # include zero bin
    def update(self, y_true_bins: torch.Tensor, y_pred_bins: torch.Tensor):
        # flatten
        t = y_true_bins.view(-1)
        p = y_pred_bins.view(-1)
        for i in range(self.num_bins+1):
            for j in range(self.num_bins+1):
                self.matrix[i, j] += torch.sum((t == i) & (p == j))
    def as_dict(self):
        return self.matrix.tolist()

# ------------------------------
# Dataset wrapping original tensor (genes x cells)
# ------------------------------
class GeneDataset(Dataset):
    def __init__(self, gene_tensor: torch.Tensor):
        self.data = gene_tensor
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], idx

# ------------------------------
# Training with bin metrics
# ------------------------------

def train():
    os.makedirs(CFG.out_dir, exist_ok=True)
    # Load and bin data (cells x genes -> genes x cells)
    expr_tensor = load_expression_tensor(CFG.data_file)  # (G, C)
    G, C = expr_tensor.shape

    # Build bin edges using non-zero values of entire dataset
    arr = expr_tensor.numpy()
    non_zero_vals = arr[arr > 0]
    if non_zero_vals.size == 0:
        raise ValueError("Dataset has no non-zero values; cannot bin.")
    bin_edges = compute_bin_edges(non_zero_vals, CFG.num_bins)
    with open(os.path.join(CFG.out_dir, "bin_edges.json"), "w") as f:
        json.dump(bin_edges.tolist(), f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GeneDataset(expr_tensor)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=CFG.num_workers)

    model = VQAutoDecoder(num_genes=G, num_cells=C, latent_dim=CFG.latent_dim, hidden_dim1=CFG.hidden_dim1,
                          hidden_dim2=CFG.hidden_dim2, codebook_size=CFG.codebook_size, vq_beta=CFG.vq_beta).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    total_steps = CFG.epochs * len(dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, total_steps=total_steps,
                                                    pct_start=0.1, anneal_strategy='cos', div_factor=10.0,
                                                    final_div_factor=100.0, cycle_momentum=False)

    loss_fn = WeightedHybridLoss(pos_weight=CFG.pos_weight, bce_weight=CFG.bce_weight)
    warmup_epochs = int(CFG.vq_warmup_frac * CFG.epochs)
    clip_norm = 1.0

    metrics_path = os.path.join(CFG.out_dir, "training_metrics.tsv")
    with open(metrics_path, "w") as f:
        f.write("epoch\tloss\trecon\tmse\tbce\tvq\tacc\tbin_acc\tperp\tvq_w\tlr\tmacro_bin_acc\n")

    cm_acc = ConfusionMatrixAccumulator(CFG.num_bins)

    def dynamic_acc_thresh(y_true: torch.Tensor) -> torch.Tensor:
        return torch.where(
            y_true == 0, 0.05,
            torch.where((y_true > 0) & (y_true < 1), 0.1, 0.1 * y_true)
        )

    best_macro_bin_acc = 0.0

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        totals = {k: 0.0 for k in ["loss","recon","mse","bce","vq","acc","binacc"]}
        samples = updates = 0
        code_counts = torch.zeros(unwrap(model).vq.num_embeddings, device=device)

        t = min(1.0, epoch / max(1, warmup_epochs))
        vq_weight = CFG.vq_w_start + t * (CFG.vq_w_end - CFG.vq_w_start)

        for x, idx in dl:
            x = x.float().to(device)
            idx = idx.to(device)

            logits, z_q, vq_loss_vec, z_e, code_idx = model(idx)
            recon_loss, mse, bce = loss_fn(logits, x)
            vq_loss = vq_loss_vec.mean()
            loss = recon_loss + vq_weight * vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step(); scheduler.step()

            with torch.no_grad():
                vals = WeightedHybridLoss.vals_from_logits(logits)
                acc = (torch.abs(vals - x) < dynamic_acc_thresh(x)).float().mean().item()
                bin_pred = (logits > 0).float()
                bin_true = (x > 0).float()
                bin_acc = (bin_pred == bin_true).float().mean().item()

                # bin-level confusion (multi-bin) using predicted continuous values mapped to bins
                pred_bins = tensor_to_bins(vals, bin_edges)
                true_bins = tensor_to_bins(x, bin_edges)
                cm_acc.update(true_bins, pred_bins)

                binc = torch.bincount(code_idx.view(-1).to(torch.int64), minlength=unwrap(model).vq.num_embeddings).float()
                code_counts += binc

            bs = x.size(0)
            totals["loss"] += loss.item() * bs
            totals["recon"] += recon_loss.item() * bs
            totals["mse"] += mse.item() * bs
            totals["bce"] += bce.item() * bs
            totals["vq"] += vq_loss.item() * bs
            totals["acc"] += acc * bs
            totals["binacc"] += bin_acc * bs
            samples += bs
            updates += 1

        avg = {k: totals[k] / samples for k in ["loss","recon","mse","bce","vq","acc","binacc"]}
        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = torch.exp(-(probs * (probs.add(1e-8).log())).sum()).item()
        lr = scheduler.get_last_lr()[0]

        # macro bin accuracy from confusion matrix (ignore zero-rows with no samples)
        cm = cm_acc.matrix.float()
        diag = torch.diag(cm)
        support = cm.sum(dim=1)
        with torch.no_grad():
            per_bin_acc = torch.where(support > 0, diag / support, torch.zeros_like(support))
            macro_bin_acc = per_bin_acc.mean().item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{CFG.epochs} Loss:{avg['loss']:.4f} Recon:{avg['recon']:.4f} VQ:{avg['vq']:.4f} Acc:{avg['acc']:.4f} BinAcc:{avg['binacc']:.4f} MacroBinAcc:{macro_bin_acc:.4f} Perp:{perplexity:.1f} vqW:{vq_weight:.3f} lr:{lr:.5g}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch}\t{avg['loss']:.4f}\t{avg['recon']:.4f}\t{avg['mse']:.4f}\t{avg['bce']:.4f}\t{avg['vq']:.4f}\t{avg['acc']:.4f}\t{avg['binacc']:.4f}\t{perplexity:.2f}\t{vq_weight:.3f}\t{lr:.5g}\t{macro_bin_acc:.4f}\n")

        # Save interim confusion matrix every 50 epochs
        if epoch % 50 == 0:
            torch.save(cm_acc.matrix, os.path.join(CFG.out_dir, f"confusion_matrix_epoch{epoch}.pt"))

        if macro_bin_acc > best_macro_bin_acc:
            best_macro_bin_acc = macro_bin_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'macro_bin_acc': macro_bin_acc}, os.path.join(CFG.out_dir, 'best_model.pt'))

    # Final artifacts
    torch.save(cm_acc.matrix, os.path.join(CFG.out_dir, "confusion_matrix_final.pt"))
    # full reconstruction
    recon = reconstruct_full(model, device, expr_tensor, batch_size=256)
    np.savez_compressed(os.path.join(CFG.out_dir, "reconstruction_full.npz"), reconstructed_matrix=recon)

    # save latents
    model_u = unwrap(model)
    with torch.no_grad():
        gene_ids = torch.arange(G, device=device)
        z_e = model_u.ln(model_u.gene_embed(gene_ids))
        z_q, _, code_idx = model_u.vq(z_e)
    torch.save(z_e.cpu(), os.path.join(CFG.out_dir, "latent_prequant.pt"))
    torch.save(z_q.cpu(), os.path.join(CFG.out_dir, "latent_quantised.pt"))
    torch.save(code_idx.cpu(), os.path.join(CFG.out_dir, "vq_code_indices.pt"))

    # dump summary json
    summary = {
        'epochs': CFG.epochs,
        'num_bins': CFG.num_bins,
        'best_macro_bin_acc': best_macro_bin_acc,
        'bin_edges': bin_edges.tolist(),
        'model_state_dict': 'best_model.pt'
    }
    with open(os.path.join(CFG.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Training complete. Artifacts saved in', CFG.out_dir)

if __name__ == '__main__':
    reserve_mem_all_gpus(TARGET_FRACTION)
    torch.backends.cudnn.benchmark = True
    train()
