"""Train a VAE on binned single-cell RNA data.

Usage:
    python -m src.train.train_vae_binned --data PATH_TO_NPZ --num_bins 8

The script will:
- load the matrix (expects shape genes x cells or cells x genes),
- bin non-zero values using term_freq_bin from src.preprocess.binning,
- split dataset into 10 equal parts, reserve one part as final test (10%),
- run 5 folds where each fold uses a different 10% as validation and the remaining
  80% as training (so 80/10/10),
- compute and log metrics: loss, overall accuracy, non-zero accuracy, binary accuracy.
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

try:
    # prefer package import if environment configured
    from src.preprocess.binning import term_freq_bin
except Exception:
    # fallback: load by path relative to this file
    import importlib.util
    spec = importlib.util.spec_from_file_location('binning',
        Path(__file__).parents[2] / 'preprocess' / 'binning.py')
    binmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binmod)
    term_freq_bin = binmod.term_freq_bin

from src.model.vae_binned import VAE_binned


class BinnedDataset(Dataset):
    def __init__(self, arr, num_bins, normalize_input=True, one_hot=False):
        # arr: (n_samples, n_genes) integers 0..(num_bins-1)
        # one_hot: if True, return flattened one-hot vectors per sample of length num_genes * num_bins
        self.arr = arr.astype(np.int64)
        self.num_bins = num_bins
        self.normalize_input = normalize_input
        self.one_hot = one_hot

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        target = self.arr[idx]  # int labels per gene
        if self.one_hot:
            # create one-hot per gene and flatten to (num_genes * num_bins,)
            # target values are 0..num_bins-1
            G = target.shape[0]
            oh = np.zeros((G, self.num_bins), dtype=np.float32)
            # advanced indexing to set ones
            genes_idx = np.arange(G)
            oh[genes_idx, target] = 1.0
            inp = oh.reshape(-1)
        else:
            if self.normalize_input:
                inp = target.astype(np.float32) / float(max(1, self.num_bins))
            else:
                inp = target.astype(np.float32)
        return inp, target


def collate_batch(batch):
    inputs = np.stack([b[0] for b in batch], axis=0)
    targets = np.stack([b[1] for b in batch], axis=0)
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).long()


def compute_class_weights(targets, num_bins, device=None):
    # targets: numpy array of shape (n_samples, n_genes)
    flat = targets.flatten()
    counts = np.bincount(flat, minlength=num_bins).astype(np.float32)
    # avoid zeros
    counts = counts + 1.0
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * num_bins
    w = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        w = w.to(device)
    return w


def evaluate(model, dataloader, device, num_bins, kl_weight=1.0):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    nsamples = 0

    overall_correct = 0
    overall_total = 0
    nonzero_correct = 0
    nonzero_total = 0
    binary_correct = 0
    binary_total = 0

    # per-class stats for precision/recall/F1
    class_tp = np.zeros(num_bins, dtype=np.int64)
    class_pred = np.zeros(num_bins, dtype=np.int64)
    class_actual = np.zeros(num_bins, dtype=np.int64)

    ce_loss = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits, mu, logvar = model(x)
            B = logits.shape[0]
            # logits: (B, G, C)
            G = logits.shape[1]
            C = logits.shape[2]
            logits_flat = logits.view(B * G, C)
            targets_flat = y.view(B * G)
            recon = ce_loss(logits_flat, targets_flat)
            kl = model.kl_divergence(mu, logvar).sum()
            loss = recon + kl_weight * kl

            total_loss += float(loss)
            total_recon += float(recon)
            total_kl += float(kl)
            nsamples += B

            preds = logits.argmax(dim=2)  # (B, G)
            overall_correct += int((preds == y).sum().item())
            overall_total += int(y.numel())

            nz_mask = (y > 0)
            nz_total = int(nz_mask.sum().item())
            if nz_total > 0:
                nonzero_correct += int(((preds == y) & nz_mask).sum().item())
                nonzero_total += nz_total

            preds_bin = (preds > 0)
            targets_bin = (y > 0)
            binary_correct += int((preds_bin == targets_bin).sum().item())
            binary_total += int(targets_bin.numel())

            # update per-class counts
            # flatten
            preds_flat = preds.view(-1).cpu().numpy()
            targets_flat_np = y.view(-1).cpu().numpy()
            for k in range(num_bins):
                class_tp[k] += int(((preds_flat == k) & (targets_flat_np == k)).sum())
                class_pred[k] += int((preds_flat == k).sum())
                class_actual[k] += int((targets_flat_np == k).sum())

    metrics = {
        'loss': total_loss / max(1, nsamples),
        'recon_loss': total_recon / max(1, nsamples),
        'kl_sum': total_kl / max(1, nsamples),
        'overall_acc': overall_correct / overall_total if overall_total > 0 else 0.0,
        'nonzero_acc': nonzero_correct / nonzero_total if nonzero_total > 0 else None,
        'binary_acc': binary_correct / binary_total if binary_total > 0 else 0.0,
    }

    # per-class precision/recall/F1 and macro-F1
    per_class_prec = []
    per_class_rec = []
    per_class_f1 = []
    for k in range(num_bins):
        tp = class_tp[k]
        pred_k = class_pred[k]
        actual_k = class_actual[k]
        prec = tp / pred_k if pred_k > 0 else 0.0
        rec = tp / actual_k if actual_k > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_prec.append(prec)
        per_class_rec.append(rec)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1)) if len(per_class_f1) > 0 else 0.0
    metrics.update({
        'per_class_precision': per_class_prec,
        'per_class_recall': per_class_rec,
        'per_class_f1': per_class_f1,
        'macro_f1': macro_f1,
    })

    # non-zero precision/recall (binary)
    # true positive for non-zero = predicted non-zero and actual non-zero
    # predicted_nonzero, actual_nonzero
    # compute from class counts: non-zero classes are 1..(num_bins-1)
    pred_nonzero = int(class_pred[1:].sum())
    actual_nonzero = int(class_actual[1:].sum())
    tp_nonzero = int(sum(class_tp[1:].tolist()))
    nonzero_prec = tp_nonzero / pred_nonzero if pred_nonzero > 0 else 0.0
    nonzero_rec = tp_nonzero / actual_nonzero if actual_nonzero > 0 else 0.0
    metrics['nonzero_precision'] = nonzero_prec
    metrics['nonzero_recall'] = nonzero_rec
    return metrics


def train_fold(model, device, train_loader, val_loader, num_bins, epochs, lr, kl_weight, fold_dir, patience=5, lr_patience=2, min_lr=1e-6, eval_interval=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)

    # compute class weights on training set for CrossEntropy
    # gather training targets
    train_targets = []
    for _, t in train_loader:
        train_targets.append(t.numpy())
    train_targets = np.concatenate(train_targets, axis=0)
    class_weights = compute_class_weights(train_targets, num_bins, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    no_improve = 0

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_path = fold_dir / 'epoch_metrics.json'
    # record start time for ETA calculations
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        # track simple training accuracy during the epoch for lightweight reporting
        train_overall_correct = 0
        train_overall_total = 0
        # also track non-zero and binary accuracy on training batches (lightweight)
        train_nonzero_correct = 0
        train_nonzero_total = 0
        train_binary_correct = 0
        train_binary_total = 0
        running_recon = 0.0
        running_kl = 0.0
        running_loss = 0.0
        nsamples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            B = logits.shape[0]
            G = logits.shape[1]
            C = logits.shape[2]
            logits_flat = logits.view(B * G, C)
            targets_flat = y.view(B * G)
            recon = ce_loss(logits_flat, targets_flat) / float(B)  # per-batch normalized
            kl = model.kl_divergence(mu, logvar).sum() / float(B)
            loss = recon + kl_weight * kl
            loss.backward()
            optimizer.step()

            # lightweight accuracy tracking on training batches
            preds_train = logits.argmax(dim=2)
            train_overall_correct += int((preds_train == y).sum().item())
            train_overall_total += int(y.numel())

            # non-zero accuracy: only consider positions where target > 0
            nz_mask = (y > 0)
            nz_total = int(nz_mask.sum().item())
            if nz_total > 0:
                train_nonzero_correct += int(((preds_train == y) & nz_mask).sum().item())
                train_nonzero_total += nz_total

            # binary accuracy (zero vs non-zero)
            preds_bin = (preds_train > 0)
            targets_bin = (y > 0)
            train_binary_correct += int((preds_bin == targets_bin).sum().item())
            train_binary_total += int(targets_bin.numel())

            running_recon += float(recon.detach() if hasattr(recon, 'detach') else recon)
            running_kl += float(kl)
            running_loss += float(loss.detach() if hasattr(loss, 'detach') else loss)
            nsamples += 1

        # decide whether to run full evaluation this epoch
        do_eval = (epoch % eval_interval == 0) or (epoch == epochs)

        if do_eval:
            train_metrics = evaluate(model, train_loader, device, num_bins, kl_weight=kl_weight)
            val_metrics = evaluate(model, val_loader, device, num_bins, kl_weight=kl_weight)
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            # step scheduler on val loss
            scheduler.step(val_metrics['loss'])

            # save best model by val loss
            if val_metrics['loss'] < best_val_loss - 1e-8:
                best_val_loss = val_metrics['loss']
                model_path = str(fold_dir / 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                no_improve = 0
            else:
                no_improve += 1

            epoch_record = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': optimizer.param_groups[0]['lr']
            }
            # per-epoch accuracies — prefer validation metrics when available
            epoch_record['accuracy'] = float(val_metrics.get('overall_acc', 0.0))
            # validation non-zero and binary accuracies may be None if no non-zero values
            epoch_record['nonzero_accuracy'] = float(val_metrics['nonzero_acc']) if (val_metrics.get('nonzero_acc') is not None) else None
            epoch_record['binary_accuracy'] = float(val_metrics.get('binary_acc', 0.0))
            print(f"Epoch {epoch}/{epochs}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, overall_val_acc={val_metrics['overall_acc']:.4f}, lr={optimizer.param_groups[0]['lr']:.6g}")
        else:
            # lightweight train-only stats when skipping full evaluation
            train_loss = running_loss / max(1, nsamples)
            train_recon = running_recon / max(1, nsamples)
            train_kl = running_kl / max(1, nsamples)
            train_metrics = {'loss': train_loss, 'recon_loss': train_recon, 'kl_sum': train_kl, 'overall_acc': (train_overall_correct / train_overall_total) if train_overall_total > 0 else 0.0}
            history['train'].append(train_metrics)
            epoch_record = {
                'epoch': epoch,
                'train': train_metrics,
                'val': None,
                'lr': optimizer.param_groups[0]['lr']
            }
            # when validation is skipped, report training accuracy as the epoch accuracy
            epoch_record['accuracy'] = float(train_metrics.get('overall_acc', 0.0))
            epoch_record['nonzero_accuracy'] = float(train_nonzero_correct / train_nonzero_total) if train_nonzero_total > 0 else None
            epoch_record['binary_accuracy'] = float(train_binary_correct / train_binary_total) if train_binary_total > 0 else None
            print(f"Epoch {epoch}/{epochs}: train_loss={train_metrics['loss']:.4f}, val=skipped, lr={optimizer.param_groups[0]['lr']:.6g}")

        # append to JSON list on disk
        if epoch_log_path.exists():
            existing = json.loads(epoch_log_path.read_text())
        else:
            existing = []
        # add timing and ETA info to epoch_record
        total_elapsed = time.time() - start_time
        epochs_done = epoch
        avg_epoch = total_elapsed / epochs_done if epochs_done > 0 else total_elapsed
        remaining_epochs = max(0, epochs - epoch)
        eta_seconds = remaining_epochs * avg_epoch
        epoch_record['elapsed_seconds'] = total_elapsed
        epoch_record['eta_seconds'] = eta_seconds
        existing.append(epoch_record)
        epoch_log_path.write_text(json.dumps(existing, indent=2))

        # early stopping only considered on evaluation epochs
        if do_eval and (no_improve >= patience):
            print(f"Early stopping at epoch {epoch}, no improvement in {no_improve} epochs")
            break

    return history


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    npz = np.load(args.data)
    # try common keys
    if 'arr_0' in npz:
        mat = npz['arr_0']
    else:
        # take first array
        mat = npz[list(npz.files)[0]]

    # Ensure samples as rows (n_samples, n_genes)
    if mat.shape[0] == args.num_genes and mat.shape[1] != args.num_genes:
        # likely genes x cells -> transpose
        mat = mat.T

    n_samples, n_genes = mat.shape
    assert n_genes == args.num_genes, f"Expected {args.num_genes} genes, found {n_genes}"

    print(f"Loaded data: samples={n_samples}, genes={n_genes}")

    # Bin data in-place using provided term_freq_bin
    print("Binning non-zero values with term_freq_bin(num_bins={})".format(args.num_bins))
    binned = term_freq_bin(mat.copy(), args.num_bins)
    # term_freq_bin produces bins in 1..num_bins for non-zero values and 0 for zeros.
    # convert to 0..(num_bins-1) so we have exactly num_bins classes where 0==zero-bin
    binned = binned.astype(np.int64)
    binned[binned > 0] = binned[binned > 0] - 1
    print('After reindexing bins, unique labels:', np.unique(binned)[:20])

    # shuffle indices and split into 10 equal parts
    rng = np.random.RandomState(args.seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    parts = np.array_split(indices, 10)

    # Reserve final part as held-out test
    test_part = parts[-1]
    remaining_parts = parts[:-1]

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = results_root / f"vae_binned_{timestamp}"
    run_dir.mkdir()

    fold_histories = {}

    # We'll use first 5 of remaining_parts as validation folds (gives five 10% val sets)
    # allow running a single fold (for array-job style runs) or all 5 folds
    fold_ids = [args.fold_id] if getattr(args, 'fold_id', None) is not None else list(range(5))

    for fold_i in fold_ids:
        val_part = remaining_parts[fold_i]
        train_parts = [p for j, p in enumerate(remaining_parts) if j != fold_i]
        train_idx = np.concatenate(train_parts, axis=0)
        val_idx = val_part

        X_train = binned[train_idx]
        X_val = binned[val_idx]
        X_test = binned[test_part]

        train_ds = BinnedDataset(X_train, args.num_bins, normalize_input=not args.one_hot_input, one_hot=args.one_hot_input)
        val_ds = BinnedDataset(X_val, args.num_bins, normalize_input=not args.one_hot_input, one_hot=args.one_hot_input)
        test_ds = BinnedDataset(X_test, args.num_bins, normalize_input=not args.one_hot_input, one_hot=args.one_hot_input)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        model = VAE_binned(num_genes=n_genes, num_bins=args.num_bins, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, one_hot_input=args.one_hot_input).to(device)

        fold_dir = run_dir / f"fold_{fold_i}"
        fold_dir.mkdir()

        print(f"Starting fold {fold_i}: train={len(train_idx)} val={len(val_idx)} test={len(test_part)}")

        history = train_fold(model, device, train_loader, val_loader, args.num_bins, args.epochs, args.lr, args.kl_weight, str(fold_dir), patience=args.patience, lr_patience=args.lr_patience, min_lr=args.min_lr, eval_interval=args.eval_interval)

        # load best model and evaluate on test and save reconstructions
        best_model_path = fold_dir / 'best_model.pth'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device, args.num_bins, kl_weight=args.kl_weight)
        print(f"Fold {fold_i} test metrics: {test_metrics}")

        # Save final best reconstructions (preds, probs (optional), targets)
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits, mu, logvar = model(x)
                probs = torch.softmax(logits, dim=2)
                preds = logits.argmax(dim=2)
                if args.save_probs:
                    all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        recon_path = fold_dir / 'best_reconstructions.npz'
        # save compressed; include probs only if requested to avoid large files
        save_dict = {'preds': all_preds, 'targets': all_targets}
        if args.save_probs:
            if len(all_probs) > 0:
                all_probs = np.concatenate(all_probs, axis=0)
                save_dict['probs'] = all_probs
        np.savez_compressed(str(recon_path), **save_dict)

        fold_histories[f'fold_{fold_i}'] = {
            'history': history,
            'test_metrics': test_metrics,
            'best_reconstructions': str(recon_path)
        }

    # save overall metadata
    meta = {
        'num_genes': n_genes,
        'num_samples': n_samples,
        'num_bins': args.num_bins,
        'args': vars(args),
        'folds': fold_histories,
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(meta, f, indent=2, default=lambda o: (o if isinstance(o, (int, float, str, bool, type(None))) else str(o)))

    print(f"Run complete, results saved to {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to npz file containing expression matrix')
    parser.add_argument('--num_genes', type=int, default=5000, help='Number of genes (features)')
    parser.add_argument('--num_bins', type=int, default=8, help='Number of bins for discretisation')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--eval_interval', type=int, default=10, help='Run full evaluation every N epochs (and always on final epoch)')
    parser.add_argument('--save_probs', action='store_true', help='Save per-bin probabilities in reconstructions (can be very large). Defaults to False')
    parser.add_argument('--fold_id', type=int, default=None, help='If set, run only this fold index (0-4). Useful for job arrays')
    parser.add_argument('--one_hot_input', action='store_true', help='Use flattened one-hot encoding for inputs (increases input dim to num_genes * num_bins)')
    # set patience high by default so folds run the full --epochs unless explicitly requested
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience (epochs without val improvement). Set high to effectively disable early stopping during the requested epoch budget.')
    parser.add_argument('--lr_patience', type=int, default=3, help='LR scheduler patience for ReduceLROnPlateau')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for LR scheduler')
    args = parser.parse_args()
    main(args)
