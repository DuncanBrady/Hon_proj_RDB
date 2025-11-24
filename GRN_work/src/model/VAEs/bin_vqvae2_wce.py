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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch import cuda
import pandas as pd
import argparse
import re
import sys
try:
    import psutil
except Exception:
    psutil = None

# Ensure the repository root (parent of `src`) is on sys.path so imports like
# `preprocess.binning` or `src.preprocess.binning` work when running this file
# directly (python path/to/bin_vqvae2.py).
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Try to import the term_freq_bin function from the project's preprocess module.
try:
    # Prefer the package-style import if the `src` package is available
    from src.preprocess.binning import term_freq_bin
except Exception:
    try:
        # Fallback to top-level import if running with src on sys.path
        from preprocess.binning import term_freq_bin
    except Exception:
        term_freq_bin = None
        print("Warning: could not import term_freq_bin from preprocess.binning; falling back to simple linear binning")

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


def choose_cell_chunk(device: torch.device, batch_size: int, bins: int, num_cells: int,
                      safety_fraction: float = 0.15, min_chunk: int = 16, max_chunk: int = 16384) -> int:
    """
    Heuristic to pick a cell chunk size so that allocating logits for a chunk
    (batch_size * chunk * bins * 4 bytes) fits comfortably into available memory.
    Uses CUDA free bytes when running on cuda, psutil available RAM for cpu, or a
    conservative default if neither is available.
    """
    # bytes per float32
    bytes_per_elem = torch.finfo(torch.float32).bits // 8
    available = None
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            free, total = cuda.mem_get_info()
            available = int(free)
        else:
            if psutil is not None:
                available = int(psutil.virtual_memory().available)
    except Exception:
        available = None

    if available is None:
        # conservative default 2GB
        available = 2 * 1024 ** 3

    # target using only a fraction of available memory
    target_bytes = int(available * float(safety_fraction))
    # avoid zero division
    denom = max(1, batch_size * max(1, bins) * bytes_per_elem)
    chunk = int(target_bytes // denom)
    # clamp and ensure at least min_chunk
    chunk = max(min_chunk, chunk)
    chunk = min(max_chunk, chunk)
    chunk = min(chunk, num_cells)
    return int(chunk)
# ----------------------------
# Data processing
# ----------------------------
def calculate_class_weights(binned_matrix: np.ndarray):
    # calculate the inverse frequency of each bin across the entire matrix
    # compute inverse-frequency weights (higher weight for rare bins)
    bins_present, counts = np.unique(binned_matrix, return_counts=True)
    total_counts = binned_matrix.size
    # raw inverse-frequency: proportional to 1 / freq
    class_weights = np.zeros(int(binned_matrix.max()) + 1, dtype=np.float64)
    for bin_val, count in zip(bins_present, counts):
        class_weights[int(bin_val)] = float(total_counts) / float(max(1, count))
    # scale weights so that the expected per-sample weight is ~1.0. For inverse-frequency
    # weights `total_counts/counts`, the expectation across the empirical distribution
    # equals the number of classes, so divide by that to keep typical loss magnitudes
    # comparable to the unweighted cross-entropy.
    num_classes = float(max(1, class_weights.size))
    return class_weights / num_classes

def load_data(file_path: str, bins: int = 7, return_class_weights=False):
    data = np.load(file_path)
    sc_matrix = data['data'] if 'data' in data.keys() else data[0]
    genes_counts = len(data["genes"]) if "genes" in data.keys() else None
    cell_counts = len(data["cells"]) if "cells" in data.keys() else None
    print(f"Gene counts: {genes_counts}, Cell counts: {cell_counts}, Raw data shape: {sc_matrix.shape}")
    bin_matrix = term_freq_bin(sc_matrix.copy(), bins)
    #Check the min and max of the binned matrix
    print(f"Binned matrix min: {bin_matrix.min()}, max: {bin_matrix.max()}")
    transpose = bin_matrix.shape == (cell_counts, genes_counts) if genes_counts and cell_counts else False
    class_weights = calculate_class_weights(bin_matrix) if return_class_weights else None
    if transpose:
        print("Data in cells x genes format; transposing to genes x cells.")
        gene_cell = bin_matrix.T   # shape: [G, C] 
    else:
        gene_cell = bin_matrix
    # Always return integer bin indices (0..bins-1) for binned-only model
    gene_tensor = torch.tensor(gene_cell.astype(np.int64))  # each sample is one gene row across cells (int bins)
    print(f"Expression tensor: {gene_tensor.shape} (genes x cells)")
    return gene_tensor, class_weights
    

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
# Note: This file is binned-only: we use categorical cross-entropy over bins.
# The previous NegativeBinomial loss has been removed.


class WeightedCrossEntropyBinnedLoss(nn.Module):
    """
    Weighted cross-entropy for multi-class logits.

    Expects logits shaped [N, K] and targets shaped [N] (long ints).
    The `weight` argument (optional) should be an iterable or tensor of length K
    containing per-class weights. Returns (loss, mean_loss, aux) where `loss`
    is the reduced value according to `reduction` (sum/mean).
    """
    def __init__(self, weight=None, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be one of 'none','mean','sum'")
        self.reduction = reduction
        # store as tensor if provided; move to device at call time
        if weight is None:
            self.register_buffer('_weight', None)
        else:
            # If caller passed a torch Tensor, copy it safely via detach().clone()
            # to avoid the UserWarning about constructing a tensor from a tensor.
            if isinstance(weight, torch.Tensor):
                w = weight.detach().clone().to(torch.float32)
            else:
                w = torch.tensor(np.asarray(weight), dtype=torch.float32)
            self.register_buffer('_weight', w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: [N, K], targets: [N]
        if logits.dim() != 2:
            raise ValueError("WeightedCrossEntropyBinnedLoss expects logits with shape [N, K]")
        weight = None
        if getattr(self, '_weight', None) is not None:
            # ensure weight tensor length matches number of classes (K)
            w = self._weight.to(logits.device)
            K = logits.size(1)
            if w.numel() == K:
                weight = w
            elif w.numel() > K:
                # truncate if provided weight longer than logits' classes
                weight = w[:K]
            else:
                # pad with ones if provided weight shorter than logits' classes
                pad = torch.ones(K - w.numel(), dtype=w.dtype, device=w.device)
                weight = torch.cat([w, pad], dim=0)

        loss_vec = F.cross_entropy(logits, targets.long(), weight=weight, reduction='none')
        if self.reduction == 'sum':
            return loss_vec.sum(), loss_vec.mean(), torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'mean':
            return loss_vec.mean(), loss_vec.mean(), torch.tensor(0.0, device=logits.device)
        else:
            return loss_vec, loss_vec, torch.tensor(0.0, device=logits.device)

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
                 codebook_size: int = 1024, vq_beta: float = 0.25,
                 init_dispersion: float = 1.0, bins: int = 7):
        super().__init__()
        # One unique continuous latent per gene
        self.gene_embed = nn.Embedding(num_genes, latent_dim)
        nn.init.normal_(self.gene_embed.weight, mean=0.0, std=0.02)
        self.ln = nn.LayerNorm(latent_dim)

        # VQ layer on the latent
        self.vq = VectorQuantizer(codebook_size, latent_dim, beta=vq_beta)


        # Decoder (latent -> per-gene logits over bins)
        # To avoid a huge final linear mapping to num_cells * bins we predict per-gene
        # logits over bins (K) and learn a small per-cell-per-bin bias matrix of shape [C, K].
        # Final per-gene-per-cell logits are computed as p_bk + cell_bin_bias[c,k]. This
        # keeps model parameters small (C*K bias parameters) and allows computing loss
        # in cell-wise chunks to avoid allocating B*C*K at once.
        self.bins = int(bins)
        self._num_cells = int(num_cells)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim2, hidden_dim1), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim1, self.bins),   # per-gene logits over K bins
        )

        # Learnable per-cell per-bin biases (C x K). Small: e.g. 50k cells * 7 bins = 350k params
        # which is far smaller than a full dense output.
        # Initialize with tiny random noise instead of all-zeros to break symmetry and
        # encourage the model to learn per-cell differences early (helps avoid stagnation).
        self.cell_bin_bias = nn.Parameter(torch.randn(self._num_cells, self.bins) * 1e-2)

    def forward(self, gene_idx: torch.Tensor):
        # gene_idx: [B] long
        z_e = self.ln(self.gene_embed(gene_idx))                # [B, D]
        z_q, vq_loss_vec, code_idx = self.vq(z_e)               # [B, D], [B], [B]
        # per-gene logits over bins: [B, K]
        per_gene_bin_logits = self.dec(z_q)
        # return per-gene logits (B, K) and let the training loop add per-cell bias in chunks
        return per_gene_bin_logits, z_q, vq_loss_vec, z_e, code_idx

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
        per_gene_bin_logits, *_ = model(idx)
        # For each gene in the batch compute per-cell argmax without allocating full [B,C,K]
        # per_gene_bin_logits: [B, K]
        cell_bias = unwrap(model).cell_bin_bias.to(per_gene_bin_logits.device)
        for i_in_batch, gidx in enumerate(idx):
            p_bk = per_gene_bin_logits[i_in_batch]      # [K]
            # compute per-cell logits: p_bk (1,K) + cell_bias (C,K) -> (C,K)
            sums = (p_bk.unsqueeze(0) + cell_bias).cpu().numpy()   # (C,K)
            pred_bins = np.argmax(sums, axis=1).astype(np.float32)  # (C,)
            recon[gidx.cpu().numpy(), :] = pred_bins
    return recon

# ----------------------------
# Training
# ----------------------------
def train_vq_autodecoder(dataloader, device, epochs=800, log_filename="vq_autodecoder_metrics.txt",
                         init_dispersion: float = 1.0, lr: float = 0.005, bins: int = 7,
                         checkpoint_freq: int = 50, result_dir: str = None,
                         class_weights=None, weight_mode: str = 'cap', weight_cap: float = 1e4):
    G, C = gene_tensor.shape
    # Model (binned-only)
    # Determine actual number of bins/classes from the loaded data to avoid target-out-of-range
    bins_used = int(gene_tensor.max().item()) + 1
    assert bins_used > 0, "Computed bins_used must be > 0"
    model = VQAutoDecoder(num_genes=G, num_cells=C,
                          latent_dim=128, hidden_dim1=2048, hidden_dim2=1024,
                          codebook_size=1024, vq_beta=0.25, init_dispersion=init_dispersion,
                          bins=bins_used).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optim/sched
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * len(dataloader),
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10.0, final_div_factor=100.0,
        cycle_momentum=False
    )
    clip_norm = 1.0

    # Loss (categorical weighted CE over bins)
    # Prefer class_weights passed in from the caller (e.g. computed by load_data).
    # If None, we leave weight_tensor as None so the loss uses no class reweighting.
    if class_weights is None:
        weight_tensor = None
    else:
        # accept numpy array, list, or torch tensor and convert to torch tensor
        if isinstance(class_weights, torch.Tensor):
            weight_tensor = class_weights.clone().detach().to(torch.float32)
        else:
            weight_tensor = torch.tensor(np.asarray(class_weights, dtype=np.float32))

    # Apply weight transform to avoid extreme gradients from very rare bins
    if weight_tensor is not None:
        if weight_mode == 'cap':
            # cap very large weights
            weight_tensor = torch.clamp(weight_tensor, max=float(weight_cap))
            print(f"Applied weight cap: {weight_cap}")
        elif weight_mode == 'log':
            # compress dynamic range using log1p
            weight_tensor = torch.log1p(weight_tensor)
            print("Applied log1p transform to class weights")
        else:
            # 'none' -> leave as-is
            pass

    loss_fn = WeightedCrossEntropyBinnedLoss(weight=(weight_tensor if weight_tensor is not None else None),
                                            reduction='sum')
    vq_weight_start, vq_weight_end, warmup_epochs = 1e-3, 1.0, int(0.25 * epochs)
    # AMP: use mixed precision on CUDA to save memory and speed up training
    use_amp = (device.type == 'cuda') and torch.cuda.is_available()
    # Create a GradScaler in a way that's compatible across PyTorch versions.
    scaler = None
    if use_amp:
        try:
            # Newer torch.amp API (PyTorch >= 2.x)
            scaler = torch.amp.GradScaler(device_type='cuda')
        except TypeError:
            # Fallback to the older cuda-specific API available on older versions
            scaler = torch.cuda.amp.GradScaler()

    # ----- Fixed subset for preview -----
    fixed_idx = torch.arange(min(20, G), dtype=torch.long)       # first 20 genes as a stable preview
    fixed_orig = gene_tensor[fixed_idx].clone()                   # original rows (CPU tensor)

    with open(log_filename, "w") as f:
        f.write("Epoch\tLoss\tRecon\tCE\tVQ\tAcc\tBinAcc\tAbsGrad\tPerp\tVQw\tLR\tCellChunk\n")

    # Log AMP usage and an initial cell_chunk suggestion so it's recorded in the run log
    bs = getattr(dataloader, 'batch_size', 1) or 1
    try:
        init_cell_chunk = choose_cell_chunk(device, batch_size=bs, bins=bins_used, num_cells=unwrap(model)._num_cells)
    except Exception:
        init_cell_chunk = None
    print(f"AMP enabled: {use_amp}, initial cell_chunk: {init_cell_chunk}")
    with open(log_filename, "a") as f:
        f.write(f"#AMP:{use_amp}\tinit_cell_chunk:{init_cell_chunk}\n")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_cell_chunk = None
        totals = {k: 0.0 for k in ["loss", "recon", "ce", "vq", "acc", "binacc", "abs"]}
        samples = updates = 0
        code_counts = torch.zeros(unwrap(model).vq.num_embeddings, device=device)

        # VQ warmup
        t = min(1.0, epoch / max(1, warmup_epochs))
        vq_weight = vq_weight_start + t * (vq_weight_end - vq_weight_start)

        for x, idx in dataloader:
            # x: [B, C] integer bin targets; idx: [B] gene ids
            x = x.long().to(device, non_blocking=True)
            idx = idx.to(device)

            # Use autocast for the forward + loss computation when AMP is enabled
            # Use the newer torch.amp API which supports device_type argument
            amp_device = 'cuda' if use_amp else 'cpu'
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                per_gene_bin_logits, z_q, vq_loss_vec, z_e, code_idx = model(idx)
                # per_gene_bin_logits: [B, K]
                # Compute cross-entropy over all cells in manageable cell-chunks so we don't
                # allocate a full [B, C, K] tensor. We'll sum cross-entropy (reduction='sum')
                # over chunks and divide by total elements to get mean loss.
                K = per_gene_bin_logits.size(1)
                total_ce_sum = per_gene_bin_logits.new_zeros(())
                total_ce_sum_weighted = per_gene_bin_logits.new_zeros(())
                total_elems = 0
                # Prepare a device-aware weight vector for cross-entropy (pad/truncate to K)
                if weight_tensor is None:
                    w_for_ce = None
                else:
                    w_for_ce = weight_tensor.to(per_gene_bin_logits.device)
                    if w_for_ce.numel() == K:
                        pass
                    elif w_for_ce.numel() > K:
                        w_for_ce = w_for_ce[:K]
                    else:
                        pad = torch.ones(K - w_for_ce.numel(), dtype=w_for_ce.dtype, device=w_for_ce.device)
                        w_for_ce = torch.cat([w_for_ce, pad], dim=0)
                # choose a reasonable cell chunk size (auto-tuned based on available memory)
                num_cells = unwrap(model)._num_cells
                bs = x.size(0)
                cell_chunk = choose_cell_chunk(device, batch_size=bs, bins=K, num_cells=num_cells)
                # record the first chosen chunk for this epoch so we can log it
                if epoch_cell_chunk is None:
                    try:
                        epoch_cell_chunk = int(cell_chunk)
                    except Exception:
                        epoch_cell_chunk = None
                # make sure cell_bin_bias is on the device
                cell_bias = unwrap(model).cell_bin_bias.to(per_gene_bin_logits.device)
                for start in range(0, num_cells, cell_chunk):
                    end = min(num_cells, start + cell_chunk)
                    bias_chunk = cell_bias[start:end]           # [chunk, K]
                    # broadcasts: per_gene_bin_logits: [B, K] -> [B, chunk, K]
                    logits_chunk = per_gene_bin_logits.unsqueeze(1) + bias_chunk.unsqueeze(0)
                    logits_flat = logits_chunk.reshape(-1, K)  # [B*chunk, K]
                    targets_chunk = x[:, start:end].reshape(-1)
                    # compute unweighted CE for the chunk (used for reporting and stable
                    # gradient magnitudes) and also the weighted CE (if class weights
                    # were provided). We keep both so logging shows a meaningful CE value
                    # even when class reweighting heavily scales the loss.
                    uw_sum = F.cross_entropy(logits_flat, targets_chunk.long(), reduction='sum')
                    if w_for_ce is not None:
                        w_sum = F.cross_entropy(logits_flat, targets_chunk.long(), weight=w_for_ce, reduction='sum')
                    else:
                        w_sum = uw_sum
                    total_ce_sum = total_ce_sum + uw_sum
                    total_ce_sum_weighted = total_ce_sum_weighted + w_sum
                    total_elems += logits_flat.size(0)
                # recon_loss (logged as CE) is the unweighted mean CE per sample; the
                # weighted CE (if used) is available as `weighted_ce` for diagnostics.
                recon_loss = total_ce_sum / float(total_elems)
                weighted_ce = total_ce_sum_weighted / float(total_elems)
                ce_loss = recon_loss
                vq_loss = vq_loss_vec.mean()

                loss = recon_loss + vq_weight * vq_loss

            # Backprop: use GradScaler when AMP is enabled
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

            # Step the scheduler after the optimizer step
            scheduler.step()

            # metrics (compute in same cell-chunked manner to avoid building full tensor)
            with torch.no_grad():
                total_correct = 0
                total_count = 0
                # reuse the same chunk choice used for loss computation to keep metrics consistent
                cell_bias = unwrap(model).cell_bin_bias.to(per_gene_bin_logits.device)
                for start in range(0, num_cells, cell_chunk):
                    end = min(num_cells, start + cell_chunk)
                    bias_chunk = cell_bias[start:end]
                    logits_chunk = per_gene_bin_logits.unsqueeze(1) + bias_chunk.unsqueeze(0)  # [B, chunk, K]
                    preds = logits_chunk.argmax(dim=2)
                    targ = x[:, start:end]
                    total_correct += (preds == targ).sum().item()
                    total_count += preds.numel()
                acc = float(total_correct) / max(1, total_count)
                # bin presence accuracy (treat bin>0 as non-zero)
                # compute predictions for presence similarly but re-use preds per chunk
                total_bin_correct = 0
                for start in range(0, num_cells, cell_chunk):
                    end = min(num_cells, start + cell_chunk)
                    bias_chunk = cell_bias[start:end]
                    logits_chunk = per_gene_bin_logits.unsqueeze(1) + bias_chunk.unsqueeze(0)
                    preds = logits_chunk.argmax(dim=2)
                    bin_pred = (preds > 0).float()
                    bin_true = (x[:, start:end] > 0).float()
                    total_bin_correct += (bin_pred == bin_true).sum().item()
                bin_acc = float(total_bin_correct) / max(1, total_count)

                binc = torch.bincount(code_idx.view(-1).to(torch.int64),
                                      minlength=unwrap(model).vq.num_embeddings).float()
                code_counts += binc

            bs = x.size(0)
            totals["loss"] += float(loss.item()) * bs
            totals["recon"] += float(recon_loss.item()) * bs
            totals["ce"] += float(ce_loss.item()) * bs
            totals["vq"] += float(vq_loss.item()) * bs
            totals["acc"] += acc * bs
            totals["binacc"] += bin_acc * bs
            totals["abs"] += mean_abs_grad(model)
            samples += bs
            updates += 1

        # epoch summary
        avg = {k: totals[k] / samples for k in ["loss", "recon", "ce", "vq", "acc", "binacc"]}
        avg["abs"] = totals["abs"] / max(1, updates)
        curr_lr = scheduler.get_last_lr()[0]

        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = torch.exp(-(probs * (probs.add(1e-8).log())).sum()).item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs} "
                f"Loss:{avg['loss']:.4f} Recon:{avg['recon']:.4f} (CE:{avg['ce']:.4f}) "
                f"VQ:{avg['vq']:.4f} Acc:{avg['acc']:.4f} BinAcc:{avg['binacc']:.4f} "
                f"AbsGrad:{avg['abs']:.5f} Perp:{perplexity:.1f} VQw:{vq_weight:.3f} lr:{curr_lr:.5g}"
            )
            with open(log_filename, "a") as f:
                f.write(
                    f"{epoch}\t{avg['loss']:.4f}\t{avg['recon']:.4f}\t{avg['ce']:.4f}\t{avg['vq']:.4f}\t"
                    f"{avg['acc']:.4f}\t{avg['binacc']:.4f}\t{avg['abs']:.5f}\t"
                    f"{perplexity:.2f}\t{vq_weight:.3f}\t{curr_lr:.5g}\t{epoch_cell_chunk}\n"
                )

        # Periodic checkpointing
        if result_dir is not None and checkpoint_freq and epoch % int(checkpoint_freq) == 0:
            ckpt_path = os.path.join(result_dir, f"checkpoint_epoch_{epoch}.pth")
            try:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler is not None:
                    try:
                        save_dict['scaler_state_dict'] = scaler.state_dict()
                    except Exception:
                        # older PyTorch scaler may not expose state_dict(); ignore
                        save_dict['scaler_state_dict'] = None
                torch.save(save_dict, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"Warning: failed to save checkpoint to {ckpt_path}: {e}")

        # ----- Fixed subset preview + (optional) full matrix print -----
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                # forward using fixed indices -> per-gene logits [B, K]
                fixed_logits, *_ = model(fixed_idx.to(device))
                # To get a small preview of reconstructed bins for a few cells,
                # add the per-cell bias for the first few cells and argmax over K.
                preview_cells = min(6, unwrap(model)._num_cells)
                cell_bias = unwrap(model).cell_bin_bias.to(fixed_logits.device)
                # logits: [B, preview_cells, K]
                preview_logits = fixed_logits.unsqueeze(1) + cell_bias[:preview_cells].unsqueeze(0)
                pred_bins = preview_logits.argmax(dim=2)
                recon_values = pred_bins.cpu().numpy()

                print("Original [0:20, 0:6]:")
                print(pd.DataFrame(fixed_orig[:, :preview_cells].cpu().numpy()))
                print("Reconstructed [0:20, 0:6]:")
                print(pd.DataFrame(recon_values[:, :preview_cells]))

                # full reconstruction skipped here to avoid huge log output; final
                # reconstruction is saved once at the end of training.
            model.train()

        if avg["acc"] > best_acc:
            best_acc = avg["acc"]

    return model

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ auto-decoder with categorical cross-entropy over bins")
    parser.add_argument("--data", type=str, default=None, help="Path to SCT-normalised expression .npz file")
    parser.add_argument("--result" , type=str, default="vqvae2_results", help="Output result directory")
    parser.add_argument("--bins", type=int, default=7,
                        help="Number of discrete bins used for each cell (binned-only mode).")
    parser.add_argument("--dry-run", action="store_true", help="Create result subdirectory and minimal log then exit (no training)")
    parser.add_argument("--init-dispersion", type=float, default=1.0,
                        help="Initial dispersion (theta) for NB; stored as log_theta and learned")
    parser.add_argument("--pos-weight", type=float, default=200.0, help="pos_weight for BCE term")
    parser.add_argument("--bce-weight", type=float, default=20.0, help="multiplier for BCE term")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--log-file", type=str, default="vqvae2_metrics.txt")
    parser.add_argument("--class-weights", type=str, default=None,
                        help="Path to .npy file or comma-separated list of class weights; if omitted, inverse-frequency weights are used by default")
    parser.add_argument("--weight-mode", type=str, default="cap", choices=["none","cap","log"],
                        help="How to transform class weights to avoid extremes: 'none' leave as-is, 'cap' clamp to --weight-cap, 'log' apply log1p")
    parser.add_argument("--weight-cap", type=float, default=1e4,
                        help="Maximum weight when --weight-mode=cap (default 1e4)")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="If >0, truncate each gene row to the first N cells (useful for smoke tests)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"],
                        help="Device to run on: 'auto' respects availability, or force 'cpu'/'cuda'")
    args = parser.parse_args()

    reserve_mem_all_gpus(TARGET_FRACTION)
    # Allow forcing device via CLI for smoke tests
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Determine dataset name (used to form result subdirectory). If not provided,
    # allow dry-run to proceed using a 'dryrun' tag.
    if args.data is None:
        if not args.dry_run:
            raise ValueError("Please provide --data path to the .npz file containing expression data")
        data_name = "dryrun"
    else:
        data_name = os.path.splitext(os.path.basename(args.data))[0]

    # Create result subdirectory: {dataSetName}_{epochCount}_{NumBin}_vqvae2_<model>_{timeStamp}
    # Include model identifier to make WCE results easy to find
    subdir_name = f"{data_name}_{args.epochs}_{args.bins}_vqvae2_wce_{int(time.time())}"
    result_dir = os.path.join(args.result, subdir_name)
    os.makedirs(result_dir, exist_ok=False)

    # If dry-run was requested, write minimal log and exit without touching torch models
    log_path = os.path.join(result_dir, args.log_file)
    if args.dry_run:
        with open(log_path, "w") as f:
            f.write("Epoch\tLoss\tRecon\tCE\tVQ\tAcc\tBinAcc\tAbsGrad\tPerp\tVQw\tLR\n")
            f.write("DRYRUN\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n")
        print(f"Dry-run: created result directory and log at {result_dir}")
        raise SystemExit(0)

    # Load the data and prepare dataloader, check for class_weights arg 
    return_class_weights = args.class_weights is None
    gene_tensor, class_weights = load_data(args.data, bins=args.bins, return_class_weights=return_class_weights)  # shape: [G, C]
    print(gene_tensor.max().item(), "is the max bin value in the loaded data")
    # Optional truncation for smoke tests
    if args.max_cells and args.max_cells > 0:
        gene_tensor = gene_tensor[:, :args.max_cells].contiguous()
    G, C = gene_tensor.shape

    ds = IndexedDataset(gene_tensor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # parse class weights CLI arg: path to .npy or comma-separated list
    if args.class_weights:
        s = args.class_weights.strip()
        if os.path.exists(s):
            try:
                arr = np.load(s)
                class_weights = arr if isinstance(arr, np.ndarray) else np.array(arr)
            except Exception:
                # fallback: read text and split
                with open(s, 'r') as fh:
                    txt = fh.read().strip()
                parts = re.split(r"[,\s]+", txt.strip('[]'))
                class_weights = [float(p) for p in parts if p != '']
        else:
            # assume comma-separated list
            parts = re.split(r"[,\s]+", s.strip('[]'))
            class_weights = [float(p) for p in parts if p != '']

    # pass full path for log file so training writes into the result dir
    model = train_vq_autodecoder(dl, device, epochs=args.epochs, log_filename=log_path,
                                 init_dispersion=args.init_dispersion, lr=args.lr, bins=args.bins,
                                 checkpoint_freq=50, result_dir=result_dir,
                                 class_weights=class_weights,
                                 weight_mode=args.weight_mode, weight_cap=args.weight_cap)

    # Save weights
    torch.save(model.state_dict(), os.path.join(result_dir, "vqvae2_model.pth"))
    print(f"Saved model to {result_dir}.")

    # Recon + save
    recon = reconstruct_full(model, device, batch_size=256)
    np.savez_compressed(os.path.join(result_dir, "vqvae2_recon.npz"), reconstructed_matrix=recon)

    # Do not print the full reconstructed matrix to stdout (it can be huge).
    # The reconstruction is saved to disk above as a compressed .npz file.

    # Export latents for every gene
    model = unwrap(model)
    with torch.no_grad():
        gene_ids = torch.arange(G, device=device)
        z_e = model.ln(model.gene_embed(gene_ids))          # [G, D]
        z_q, _, code_idx = model.vq(z_e)
    torch.save(z_e.cpu(),   os.path.join(result_dir, "vqvae2_latent_prequant.pt"))     # continuous per-gene vectors
    torch.save(z_q.cpu(),   os.path.join(result_dir, "vqvae2_latent_quantised.pt"))    # quantized per-gene vectors
    torch.save(code_idx.cpu(), os.path.join(result_dir, "vqvae2_code_indices.pt"))
    print(f"Saved reconstructions and latent representations to {result_dir}.")
