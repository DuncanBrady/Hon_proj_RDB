
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch import cuda
import subprocess


import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import subprocess
from torch import cuda

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
            numel = need // torch.finfo(dtype).bits * 8
            if numel <= 0:
                continue
            t = torch.empty(numel, dtype=dtype, device=f'cuda:{d}')
            _reserve_tensors.append(t)

def release_reserved_mem():
    global _reserve_tensors
    _reserve_tensors = []
    torch.cuda.empty_cache()

# Set working directory and load data
path = os.path.expanduser("C:\\Users\\rdbra\\Documents\\honoursProject\\code_base\\Data\\")
os.chdir(path)
data_file = os.path.join(path, 'sct_matrix_transposed.npz')
cell_gene = np.load(data_file)['arr_0']
gene_cell = cell_gene.T
gene_tensor = torch.tensor(gene_cell, dtype=torch.float32)
assert gene_tensor.dim() == 2, "Input tensor must be 2D"
print(f"Input tensor shape: {gene_tensor.shape} (genes x cells)")




# ---------------- Huber Loss ----------------
#def weighted_huber_loss(reconstructed, original, weight, delta=1.0):
#    error = reconstructed - original
#    is_small = torch.abs(error) <= delta
#    small_loss = 0.5 * error**2
#    large_loss = delta * (torch.abs(error) - 0.5 * delta)https://chatgpt.com/c/689bf8db-bc54-8326-88de-a48891456079
#    loss = torch.where(is_small, small_loss, large_loss)
#    return torch.mean(weight * loss)

#def loss_function(reconstructed, original):
#    weight = torch.where(original > 0, 1.0, 0.5).to(original.device)
#    return weighted_huber_loss(reconstructed, original, weight)


def weighted_mse_loss(reconstructed, original, weight):
    return torch.mean(weight * (reconstructed - original)**2)

def loss_function(reconstructed, original):
    weight = torch.where(original > 0, 1.0, 0.5).to(original.device)
    return weighted_mse_loss(reconstructed, original, weight)


class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

class FCAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[2048, 512], latent_dim=128):
        super(FCAutoencoder, self).__init__()
        encoder_layers = []
        current_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h))
            encoder_layers.append(nn.LayerNorm(h))
            encoder_layers.append(nn.ReLU())
            current_dim = h
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        encoder_layers.append(nn.LayerNorm(latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        current_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h))
            decoder_layers.append(nn.ReLU())
            current_dim = h
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def mean_abs_grad(model: nn.Module) -> float:
    with torch.no_grad():
        vals = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(vals)) if vals else 0.0

def dynamic_acc_thresh(y_true):
    thresh = torch.where(
        y_true == 0, 0.05,
        torch.where(
            (y_true > 0) & (y_true < 1), 0.1,
            0.1 * y_true
        )
    )
    return thresh



class CustomMultiStageCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, steps_per_epoch, epochs_per_cycle=50, last_epoch=-1):
        self.total_steps = total_epochs * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.epochs_per_cycle = epochs_per_cycle
        self.steps_per_cycle = epochs_per_cycle * steps_per_epoch

        self.stage_cycles = [
            (int(0.1 * self.total_steps), (0.005, 0.001)),     # 10% of total steps
            (int(0.2 * self.total_steps), (0.001, 0.0005)),   # 20%
            (int(0.3 * self.total_steps), (0.0005, 0.0001)),  # 30%
            (int(0.4 * self.total_steps), (0.0001, 0.00005))  # 40%
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



def train_AE(dataloader, device, epochs=2000, log_filename="model2_metrices.txt"):
    sample_x, _ = next(iter(dataloader))
    input_dim = sample_x.size(1)

    model = FCAutoencoder(input_dim)
    model.apply(initialize_weights)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    model = model.cuda()
    release_reserved_mem()

    
    optimizer = optim.AdamW(model.parameters())

    
    scheduler = CustomMultiStageCyclicLR(
    optimizer,
    total_epochs=2000,
    steps_per_epoch=len(dataloader),
    epochs_per_cycle=50 )



    with open(log_filename, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\tAbsGrad\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        total_absG = 0.0
        n_updates = 0
        total_samples = 0

        for batch_x, _ in dataloader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            decoded, _ = model(batch_x)
            loss = loss_function(decoded, batch_x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_absG += mean_abs_grad(model)
            n_updates += 1
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            with torch.no_grad():
                acc_thresh_dynamic = dynamic_acc_thresh(batch_x)
                acc = (torch.abs(decoded - batch_x) < acc_thresh_dynamic).float().mean().item()
                total_acc += acc * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_absG = total_absG / max(n_updates, 1)

        if epoch % 50 == 0 and torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(d) / 1024**2
                print(f"GPU{d} alloc: {alloc:.1f} MiB")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}, Acc: {avg_acc:.4f}, AbsGrad: {avg_absG:.6f}, lr: {current_lr:.6f}")
            with open(log_filename, "a") as f:
                f.write(f"{epoch}\t{avg_loss:.6f}\t{avg_acc:.4f}\t{avg_absG:.6f}\n")

    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    reserve_mem_all_gpus(TARGET_FRACTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}")
    torch.backends.cudnn.benchmark = True

    ds = IndexedDataset(gene_tensor)
    dl = DataLoader(ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

    model = train_AE(dl, device, log_filename="model2_gvae_metrices.txt")
    torch.save(model.state_dict(), 'model2_gvae_gene_cell.pth')
    print("Model weights saved to 'model2_gvae_gene_cell.pth'")

    model.eval()
    with torch.no_grad():
        gene_tensor = gene_tensor.to(device)
        _, gene_embeddings = model(gene_tensor)
        gene_embeddings = gene_embeddings.cpu()

    print(f"Final gene embeddings shape: {gene_embeddings.shape}")
    torch.save(gene_embeddings, "model2_gvae_gene_embeddings.pt")
    print("Saved gene embeddings to 'model2_gvae_gene_embeddings.pt'")




