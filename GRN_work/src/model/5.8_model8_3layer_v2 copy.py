
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset



# Calculate the fraction of total memory that equals 20 GiB
total_memory_in_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GiB
memory_limit_fraction = 20 / total_memory_in_gb  

# Set per-process memory fraction (this will allocate approximately 20 GiB)
torch.cuda.set_per_process_memory_fraction(memory_limit_fraction, 0)  # 0 is the device ID


# Set working directory and load data
path = '/media/rokny/DATA1/Mehran/'
os.chdir(path)
data_path = path + 'Integrated_matrix.npy'

gene_cell = np.load(data_path)
print(gene_cell.shape)

path = '/media/rokny/DATA1/Mehran/model8/'
os.chdir(path)

# Define Huber Loss Function for Sparse Data
def weighted_huber_loss(reconstructed_expression, original_expression, weight, delta=1.0):
    error = reconstructed_expression - original_expression
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    huber_loss = torch.where(is_small_error, squared_loss, linear_loss)
    return torch.mean(weight * huber_loss)

def loss_function(reconstructed_expression, original_expression):
    weight = torch.where(original_expression > 0, 1.0, 0.1)
    return weighted_huber_loss(reconstructed_expression, original_expression, weight)

# Define IndexedDataset for the DataLoader
class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Reshape input to add channel dimension for Conv1d (batch_size, 1, num_features)
        x = x.unsqueeze(1)  # Add a channel dimension (N, 1, L), where L is the number of features (genes)
        
        # Encode and decode
        latent_rep = self.encoder(x)
        reconstructed_expression = self.decoder(latent_rep)

        # Remove the channel dimension after decoding (N, L)
        reconstructed_expression = reconstructed_expression.squeeze(1)
        
        # Ensure output matches input size
        if reconstructed_expression.size(1) != x.size(2):
            if reconstructed_expression.size(1) > x.size(2):
                reconstructed_expression = reconstructed_expression[:, :x.size(2)]  # Slice to match the original size
            else:
                padding = x.size(2) - reconstructed_expression.size(1)
                reconstructed_expression = F.pad(reconstructed_expression, (0, padding))  # Pad to match the size
        
        return reconstructed_expression, latent_rep



def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Training function for the Convolutional Autoencoder with Weighted Huber Loss
def train_AE(dataloader, device, log_filename):
    model = ConvAutoencoder().to(device)
    
    # Apply weight initialization to the model
    model.apply(initialize_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)  # Using AdamW optimizer
    scheduler = OneCycleLR(optimizer, max_lr=1e-5, steps_per_epoch=len(dataloader), epochs=1000)  # OneCycleLR scheduler
    
    EPOCH_AE = 1000
    accuracy_threshold = 0.05
    encoder = None
    final_reconstructed = None

    with open(log_filename, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tAccuracy\n")

        for epoch in range(EPOCH_AE):
            total_loss = 0
            total_accuracy = 0
            total_samples = 0

            for batch_x, _ in dataloader:
                batch_x = batch_x.float().to(device)
                decoded, encoded = model(batch_x)
                loss = loss_function(decoded, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                with torch.no_grad():
                    correct = torch.abs(decoded - batch_x) < accuracy_threshold
                    accuracy = correct.float().mean().item()
                    total_accuracy += accuracy * batch_x.size(0)
                    total_samples += batch_x.size(0)

                # Only keep the final epoch's encoder and reconstructed results
                #if epoch == EPOCH_AE - 1:
                #    encoder = encoded if encoder is None else torch.cat((encoder, encoded), dim=0)
                #    final_reconstructed = decoded if final_reconstructed is None else torch.cat((final_reconstructed, decoded), dim=0)
                      
                if epoch == EPOCH_AE - 1:
                    encoder = encoded.cpu() if encoder is None else torch.cat((encoder.cpu(), encoded.cpu()), dim=0)
                    final_reconstructed = decoded.cpu() if final_reconstructed is None else torch.cat((final_reconstructed.cpu(), decoded.cpu()), dim=0)


            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / total_samples

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1} | Train Loss: {avg_loss:.6f} | Accuracy: {avg_accuracy:.4f}')
                log_file.write(f"{epoch + 1}\t{avg_loss:.6f}\t{avg_accuracy:.4f}\n")
                
        torch.cuda.empty_cache()


    return model, encoder, final_reconstructed


def reduction_AE(gene_cell, device):
    # Train on gene-cell matrix
    gene_tensor = torch.tensor(gene_cell, dtype=torch.float32).to(device)
    dataset = IndexedDataset(gene_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model, gene_cell_encoder, gene_cell_reconstructed = train_AE(dataloader, device, log_filename="model8_v2_1000epoch_gene_cell_training_log.txt")

    # Train on cell-gene matrix
    cell_gene_tensor = torch.tensor(gene_cell.T, dtype=torch.float32).to(device)
    dataset_transposed = IndexedDataset(cell_gene_tensor)
    dataloader_transposed = DataLoader(dataset_transposed, batch_size=128, shuffle=True)
    model_transposed, cell_gene_encoder, cell_gene_reconstructed = train_AE(dataloader_transposed, device, log_filename="model8_v2_1000epoch_cell_gene_training_log.txt")
    
    return (model, gene_cell_encoder, gene_cell_reconstructed,
            model_transposed, cell_gene_encoder, cell_gene_reconstructed)


# Calling the function with gene-cell data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, gene_cell_encoder, gene_cell_reconstructed_matrix, model_transposed, cell_gene_encoder, cell_gene_reconstructed_matrix = reduction_AE(gene_cell, device=device)

# Save models, encoders, and reconstructed matrices
torch.save(model.state_dict(), 'model8_v2_1000epoch.pth')
torch.save(model_transposed.state_dict(), 'model8_v2_1000epoch_transposed.pth')
torch.save(gene_cell_encoder, 'model8_v2_1000epoch_gene_cell_encoder.pt')
torch.save(cell_gene_encoder, 'model8_v2_1000epoch_cell_gene_encoder.pt')
np.save('model8_v2_1000epoch_gene_cell_reconstructed_matrix.npy', gene_cell_reconstructed_matrix.detach().cpu().numpy())
np.save('model8_v2_1000epoch_cell_gene_reconstructed_matrix.npy', cell_gene_reconstructed_matrix.detach().cpu().numpy())


