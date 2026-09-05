import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class L1SparseAutoencoder(nn.Module):
    def __init__(self, n_bands, latent_dim=2048, l1_coeff=3e-3, tied_weights=True):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.encoder = nn.Linear(n_bands, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_bands, bias=True)
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t().clone())
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.div_(norms)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon = F.mse_loss(x_hat, x)
        sparsity = z.abs().mean()
        return recon + self.l1_coeff * sparsity, recon.item(), sparsity.item()


class TopKSparseAutoencoder(nn.Module):
    """Gao et al. 2024 style: hard top-k activation removes the need to
    tune an L1 coefficient and empirically yields cleaner single-concept
    features — worth running as a cross-check against the L1 variant."""

    def __init__(self, n_bands, latent_dim=2048, k=32):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(n_bands, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_bands, bias=True)

    def forward(self, x):
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon = F.mse_loss(x_hat, x)
        return recon, recon.item(), 0.0  # sparsity is exact by construction


def train_sae(model, X_train, X_val, epochs=60, batch_size=512, lr=1e-3,
              device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xval = torch.tensor(X_val, dtype=torch.float32).to(device)

    n = Xtr.shape[0]
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        model.train()
        ep_recon, ep_sparse = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device)
            x_hat, z = model(xb)
            loss, recon, sparse = model.loss(xb, x_hat, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if isinstance(model, L1SparseAutoencoder):
                model._normalize_decoder()
            ep_recon += recon * xb.shape[0]
            ep_sparse += sparse * xb.shape[0]

        model.eval()
        with torch.no_grad():
            x_hat_val, z_val = model(Xval)
            val_loss, val_recon, _ = model.loss(Xval, x_hat_val, z_val)
            active_frac = (z_val > 0).float().mean().item()

        history.append(dict(epoch=epoch, train_recon=ep_recon / n,
                             val_recon=val_recon, active_frac=active_frac))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train_recon={ep_recon/n:.5f}  "
                  f"val_recon={val_recon:.5f}  active_frac={active_frac:.3f}")
    return model, history


@torch.no_grad()
def encode(model, X, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    if isinstance(model, TopKSparseAutoencoder):
        pre = model.encoder(Xt)
        topk_vals, topk_idx = torch.topk(pre, model.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
    else:
        z = F.relu(model.encoder(Xt))
    return z.cpu().numpy()


if __name__ == "__main__":
    # Smoke test with random data (Colab will substitute real/simulated spectra)
    n_bands, n = 421, 4000
    X = np.random.rand(n, n_bands).astype(np.float32)
    model = L1SparseAutoencoder(n_bands, latent_dim=512, l1_coeff=1e-3)
    model, hist = train_sae(model, X[:3000], X[3000:], epochs=5, verbose=True)
    print("Smoke test OK")
