# This file defines a variational autoencoder (VAE) model reconstructs binned single-cell data.
# This occurs at cell level, each cell is a sample and gene expression for that cell are encoded.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_cell_emb(nn.Module):
    """Variational Autoencoder for binned single-cell data.

    Decoder outputs logits per gene for each bin. The reconstruction loss
    should be computed with CrossEntropyLoss (see training script).
    """
    def __init__(self, num_genes: int, num_bins: int, hidden_dim: int = 1024, latent_dim: int = 64, one_hot_input: bool = False):
        super().__init__()
        self.num_genes = num_genes
        self.num_bins = num_bins
        # if using one-hot input, input dimension expands to num_genes * num_bins
        self.one_hot_input = one_hot_input
        self.input_dim = num_genes * num_bins if one_hot_input else num_genes

        # Encoder
        self.enc_fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.enc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        # Final projection to logits for each gene and bin
        # We'll reshape outputs to (batch, num_genes, num_bins)
        self.dec_out = nn.Linear(hidden_dim, num_genes * num_bins)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h1 = F.relu(self.enc_fc1(x))
        h2 = F.relu(self.enc_fc2(h1))
        mu = self.enc_mu(h2)
        logvar = self.enc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z):
        h1 = F.relu(self.dec_fc1(z))
        h2 = F.relu(self.dec_fc2(h1))
        out = self.dec_out(h2)
        # logits shape: (batch, num_genes * num_bins)
        logits = out.view(-1, self.num_genes, self.num_bins)
        return logits

    def forward(self, x):
        """Forward returns logits, mu, logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

    @staticmethod
    def kl_divergence(mu, logvar):
        # KL divergence between N(mu, sigma) and N(0,1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
