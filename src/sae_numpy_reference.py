"""
sae_numpy_reference.py
========================
Pure NumPy re-implementation of the SAME L1-Sparse Autoencoder math as
spectral_sae.py's L1SparseAutoencoder — hand-derived forward/backward pass,
no autograd. This exists SOLELY because this sandbox has no internet access
to install PyTorch, and I will not fake a "torch ran successfully" result.

This lets me:
  - actually execute and validate the full pipeline right now, on CPU,
    with zero external dependencies beyond numpy
  - produce real (not hypothetical) numbers, plots, and a feature dashboard
    in this conversation
  - give you high confidence the Colab PyTorch notebook (spectral_sae.py)
    will behave the same way, since it's the identical loss function,
    architecture, and update rule — just accelerated + autograd'd.

Model:  x -> ReLU(W_enc x + b_enc) = z  -> W_dec z + b_dec = x_hat
Loss :  ||x - x_hat||^2  +  l1_coeff * mean(|z|)
Update: vanilla Adam, decoder columns renormalized to unit norm each step
        (exactly matching the tied/normalized-decoder trick used in the
        torch version, which prevents the trivial "shrink activations,
        grow decoder norm" degenerate solution).
"""

import numpy as np


class NumpyAdam:
    def __init__(self, shapes, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros(s) for k, s in shapes.items()}
        self.v = {k: np.zeros(s) for k, s in shapes.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class NumpySparseAutoencoder:
    def __init__(self, n_bands, latent_dim=512, l1_coeff=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        scale_enc = np.sqrt(2.0 / n_bands)
        scale_dec = np.sqrt(2.0 / latent_dim)
        self.params = dict(
            W_enc=rng.normal(0, scale_enc, size=(n_bands, latent_dim)),
            b_enc=np.zeros(latent_dim),
            W_dec=rng.normal(0, scale_dec, size=(latent_dim, n_bands)),
            b_dec=np.zeros(n_bands),
        )
        self._normalize_decoder()
        self.l1_coeff = l1_coeff
        self.opt = NumpyAdam({k: v.shape for k, v in self.params.items()})

    def _normalize_decoder(self):
        norms = np.linalg.norm(self.params["W_dec"], axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        self.params["W_dec"] /= norms

    def forward(self, X):
        pre = X @ self.params["W_enc"] + self.params["b_enc"]
        z = np.maximum(pre, 0)
        x_hat = z @ self.params["W_dec"] + self.params["b_dec"]
        return pre, z, x_hat

    def loss_and_grad(self, X):
        n = X.shape[0]
        pre, z, x_hat = self.forward(X)
        resid = x_hat - X
        recon_loss = np.mean(np.sum(resid ** 2, axis=1))
        sparsity_loss = np.mean(np.abs(z))
        total = recon_loss + self.l1_coeff * sparsity_loss

        # gradients
        d_xhat = 2 * resid / n                       # (n, bands)
        grad_W_dec = z.T @ d_xhat                     # (latent, bands)
        grad_b_dec = d_xhat.sum(axis=0)

        d_z_from_recon = d_xhat @ self.params["W_dec"].T   # (n, latent)
        d_z_from_l1 = (self.l1_coeff / n) * np.sign(z)
        d_z = d_z_from_recon + d_z_from_l1
        d_pre = d_z * (pre > 0)

        grad_W_enc = X.T @ d_pre
        grad_b_enc = d_pre.sum(axis=0)

        grads = dict(W_enc=grad_W_enc, b_enc=grad_b_enc,
                     W_dec=grad_W_dec, b_dec=grad_b_dec)
        return total, recon_loss, sparsity_loss, grads

    def encode(self, X):
        _, z, _ = self.forward(X)
        return z


def train(model, X_train, X_val, epochs=80, batch_size=256, verbose=True):
    n = X_train.shape[0]
    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        ep_recon = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            total, recon, sparse, grads = model.loss_and_grad(xb)
            model.opt.step(model.params, grads)
            model._normalize_decoder()
            ep_recon += recon * len(idx)
        _, z_val, x_hat_val = model.forward(X_val)
        val_recon = np.mean(np.sum((x_hat_val - X_val) ** 2, axis=1))
        active_frac = np.mean(z_val > 0)
        history.append(dict(epoch=epoch, train_recon=ep_recon / n,
                             val_recon=val_recon, active_frac=active_frac))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train_recon={ep_recon/n:.5f}  "
                  f"val_recon={val_recon:.5f}  active_frac={active_frac:.3f}")
    return model, history
