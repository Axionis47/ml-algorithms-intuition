"""
VARIATIONAL AUTOENCODER — Paradigm: GENERATIVE (Deep Latent Variable Model)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A deep generative model with:
    - Encoder: maps data x → latent distribution q(z|x)
    - Latent space: low-dimensional representation z
    - Decoder: maps z → reconstructed data p(x|z)

Unlike regular autoencoders, VAE learns a DISTRIBUTION in latent space,
enabling generation of new samples.

===============================================================
THE KEY INSIGHT: ELBO AND REPARAMETERIZATION
===============================================================

We want to maximize log p(x), but it's intractable.
Instead, maximize the ELBO (Evidence Lower BOund):

    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction + Regularization

Reconstruction term: decoded samples should match input
KL term: posterior q(z|x) should be close to prior p(z) = N(0,I)

THE REPARAMETERIZATION TRICK:
    Instead of sampling z ~ q(z|x) = N(μ, σ²) (not differentiable!)
    Sample ε ~ N(0,1), then z = μ + σ × ε (differentiable!)

This allows gradients to flow through the sampling operation.

===============================================================
WHY KL MATTERS
===============================================================

Without KL term:
    - Encoder maps each x to a POINT (like regular autoencoder)
    - Latent space has "holes" (can't generate there)
    - Sampling produces garbage

With KL term:
    - Encoder maps x to distributions near N(0,I)
    - Latent space is DENSE (every z generates something)
    - Smooth interpolation between data points

KL = 0.5 × Σ(μ² + σ² - log(σ²) - 1)

===============================================================
LATENT SPACE PROPERTIES
===============================================================

1. CONTINUITY: nearby z → similar x
2. COMPLETENESS: every z → valid x
3. DISENTANGLEMENT (ideally): each z_i controls one factor

This enables:
    - Generation: sample z ~ N(0,I), decode to x
    - Interpolation: z = α×z1 + (1-α)×z2
    - Manipulation: change one z dimension

===============================================================
VAE vs AUTOENCODER
===============================================================

AUTOENCODER:
    - Deterministic encoding
    - No principled generation
    - Latent space has holes

VAE:
    - Probabilistic encoding
    - Principled generation via sampling
    - Smooth, complete latent space

The KL term is what makes VAE a proper generative model.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Latent factors exist and are Gaussian
2. Decoder can reconstruct from latent
3. KL regularization enforces structure
4. Bottleneck dimension matters

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


def create_vae_dataset(n_samples=1000, dataset='mnist_like'):
    """
    Create simple image dataset for VAE.
    """
    np.random.seed(42)

    if dataset == 'mnist_like':
        # Simple 8x8 digit-like patterns
        images = []
        labels = []

        for i in range(n_samples):
            img = np.zeros((8, 8))
            digit = i % 4  # 0, 1, 2, 3

            if digit == 0:  # Circle-ish (O)
                img[1:7, 2] = 1
                img[1:7, 5] = 1
                img[2, 2:6] = 1
                img[5, 2:6] = 1
            elif digit == 1:  # Vertical line (1)
                img[1:7, 4] = 1
                img[2, 3] = 1
            elif digit == 2:  # L shape
                img[2:6, 2] = 1
                img[5, 2:6] = 1
            else:  # Cross (+)
                img[1:7, 3:5] = 1
                img[3:5, 1:7] = 1

            # Add noise
            img += np.random.randn(8, 8) * 0.1
            img = np.clip(img, 0, 1)

            images.append(img.flatten())
            labels.append(digit)

        X = np.array(images)
        y = np.array(labels)

    elif dataset == '2d':
        # 2D data for visualization
        X = np.vstack([
            np.random.randn(n_samples//4, 2) * 0.5 + [-2, -2],
            np.random.randn(n_samples//4, 2) * 0.5 + [2, -2],
            np.random.randn(n_samples//4, 2) * 0.5 + [-2, 2],
            np.random.randn(n_samples//4, 2) * 0.5 + [2, 2],
        ])
        y = np.array([0]*(n_samples//4) + [1]*(n_samples//4) +
                    [2]*(n_samples//4) + [3]*(n_samples//4))

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


class VAE:
    """
    Variational Autoencoder.

    Encoder: x → (μ, log σ²)
    Sampling: z = μ + σ × ε, where ε ~ N(0,I)
    Decoder: z → x_reconstructed
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=0.001):
        """
        Parameters:
        -----------
        input_dim : Dimension of input data
        hidden_dim : Dimension of hidden layers
        latent_dim : Dimension of latent space
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate

        # Encoder weights
        # x → hidden
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b_enc1 = np.zeros(hidden_dim)

        # hidden → μ
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_mu = np.zeros(latent_dim)

        # hidden → log σ²
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        # z → hidden
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b_dec1 = np.zeros(hidden_dim)

        # hidden → x
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_dec2 = np.zeros(input_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x):
        """
        Encode x to latent distribution parameters.

        Returns: μ, log σ², hidden activation (for backprop)
        """
        # Hidden layer
        h_enc = self.relu(x @ self.W_enc1 + self.b_enc1)

        # Latent distribution parameters
        mu = h_enc @ self.W_mu + self.b_mu
        logvar = h_enc @ self.W_logvar + self.b_logvar

        return mu, logvar, h_enc

    def reparameterize(self, mu, logvar):
        """
        THE REPARAMETERIZATION TRICK!

        Instead of z ~ N(μ, σ²), use z = μ + σ × ε where ε ~ N(0,I)
        This allows gradients to flow through sampling.
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)  # Sample from N(0,I)
        z = mu + std * eps
        return z, eps

    def decode(self, z):
        """
        Decode latent z to reconstruction.

        Returns: reconstruction, hidden activation (for backprop)
        """
        h_dec = self.relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = self.sigmoid(h_dec @ self.W_dec2 + self.b_dec2)
        return x_recon, h_dec

    def forward(self, x):
        """
        Full forward pass.

        Returns all intermediate values for backprop.
        """
        mu, logvar, h_enc = self.encode(x)
        z, eps = self.reparameterize(mu, logvar)
        x_recon, h_dec = self.decode(z)

        return x_recon, mu, logvar, z, eps, h_enc, h_dec

    def compute_loss(self, x, x_recon, mu, logvar):
        """
        Compute ELBO loss = Reconstruction + KL divergence.

        Reconstruction: Binary cross-entropy (for [0,1] data)
        KL: Closed form for Gaussian q(z|x) vs N(0,I) prior
        """
        batch_size = x.shape[0]

        # Reconstruction loss: binary cross-entropy
        # -[x log(x_recon) + (1-x) log(1-x_recon)]
        x_recon = np.clip(x_recon, 1e-10, 1 - 1e-10)
        recon_loss = -np.sum(x * np.log(x_recon) + (1 - x) * np.log(1 - x_recon)) / batch_size

        # KL divergence: KL(N(μ, σ²) || N(0, I))
        # = 0.5 × Σ(μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / batch_size

        return recon_loss + kl_loss, recon_loss, kl_loss

    def backward(self, x, x_recon, mu, logvar, z, eps, h_enc, h_dec):
        """
        Backpropagation through VAE.
        """
        batch_size = x.shape[0]

        # Gradient of reconstruction loss w.r.t. x_recon
        x_recon = np.clip(x_recon, 1e-10, 1 - 1e-10)
        dx_recon = (x_recon - x) / batch_size  # sigmoid + BCE gradient

        # Backprop through decoder
        # h_dec was computed as relu(z @ W_dec1 + b_dec1)
        # Need to use the pre-activation for relu_grad
        pre_activation_dec = z @ self.W_dec1 + self.b_dec1
        dh_dec = (dx_recon @ self.W_dec2.T) * self.relu_grad(pre_activation_dec)
        dW_dec2 = h_dec.T @ dx_recon
        db_dec2 = np.sum(dx_recon, axis=0)

        dz = dh_dec @ self.W_dec1.T
        dW_dec1 = z.T @ dh_dec
        db_dec1 = np.sum(dh_dec, axis=0)

        # Gradient of KL loss w.r.t. μ and log σ²
        dmu_kl = mu / batch_size
        dlogvar_kl = 0.5 * (np.exp(logvar) - 1) / batch_size

        # Backprop through reparameterization
        # z = μ + exp(0.5 logvar) × ε
        std = np.exp(0.5 * logvar)
        dmu_reparam = dz
        dlogvar_reparam = dz * eps * 0.5 * std

        # Combine gradients
        dmu = dmu_reparam + dmu_kl
        dlogvar = dlogvar_reparam + dlogvar_kl

        # Backprop through encoder output layers
        dW_mu = h_enc.T @ dmu
        db_mu = np.sum(dmu, axis=0)

        dW_logvar = h_enc.T @ dlogvar
        db_logvar = np.sum(dlogvar, axis=0)

        # Backprop to hidden
        dh_enc = dmu @ self.W_mu.T + dlogvar @ self.W_logvar.T
        dh_enc *= self.relu_grad(h_enc)

        # Backprop through encoder input layer
        dW_enc1 = x.T @ dh_enc
        db_enc1 = np.sum(dh_enc, axis=0)

        # Update weights with gradient descent
        self.W_dec2 -= self.lr * dW_dec2
        self.b_dec2 -= self.lr * db_dec2
        self.W_dec1 -= self.lr * dW_dec1
        self.b_dec1 -= self.lr * db_dec1
        self.W_mu -= self.lr * dW_mu
        self.b_mu -= self.lr * db_mu
        self.W_logvar -= self.lr * dW_logvar
        self.b_logvar -= self.lr * db_logvar
        self.W_enc1 -= self.lr * dW_enc1
        self.b_enc1 -= self.lr * db_enc1

    def fit(self, X, epochs=100, batch_size=32, verbose=True):
        """Train the VAE."""
        n_samples = len(X)
        losses = {'total': [], 'recon': [], 'kl': []}

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for i in range(0, n_samples, batch_size):
                x_batch = X_shuffled[i:i+batch_size]

                # Forward
                x_recon, mu, logvar, z, eps, h_enc, h_dec = self.forward(x_batch)

                # Loss
                loss, recon, kl = self.compute_loss(x_batch, x_recon, mu, logvar)
                epoch_loss += loss * len(x_batch)
                epoch_recon += recon * len(x_batch)
                epoch_kl += kl * len(x_batch)

                # Backward
                self.backward(x_batch, x_recon, mu, logvar, z, eps, h_enc, h_dec)

            epoch_loss /= n_samples
            epoch_recon /= n_samples
            epoch_kl /= n_samples

            losses['total'].append(epoch_loss)
            losses['recon'].append(epoch_recon)
            losses['kl'].append(epoch_kl)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f} "
                      f"(Recon={epoch_recon:.4f}, KL={epoch_kl:.4f})")

        return losses

    def generate(self, n_samples=10):
        """Generate new samples by sampling z ~ N(0,I) and decoding."""
        z = np.random.randn(n_samples, self.latent_dim)
        x_gen, _ = self.decode(z)
        return x_gen

    def reconstruct(self, x):
        """Reconstruct input through encoder-decoder."""
        mu, logvar, _ = self.encode(x)
        z, _ = self.reparameterize(mu, logvar)
        x_recon, _ = self.decode(z)
        return x_recon

    def encode_mean(self, x):
        """Get latent mean (for visualization)."""
        mu, _, _ = self.encode(x)
        return mu


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X, y = create_vae_dataset(n_samples=800, dataset='mnist_like')

    # -------- Experiment 1: Latent Dimension --------
    print("\n1. EFFECT OF LATENT DIMENSION")
    print("-" * 40)
    for latent_dim in [1, 2, 4, 8, 16]:
        vae = VAE(input_dim=64, hidden_dim=32, latent_dim=latent_dim, learning_rate=0.01)
        losses = vae.fit(X, epochs=100, verbose=False)
        print(f"latent_dim={latent_dim:<3} final_loss={losses['total'][-1]:.4f} "
              f"(recon={losses['recon'][-1]:.4f}, kl={losses['kl'][-1]:.4f})")
    print("→ Larger latent = lower recon error, but higher KL")
    print("→ Trade-off between expressiveness and regularization")

    # -------- Experiment 2: KL Weight (β-VAE) --------
    print("\n2. EFFECT OF KL WEIGHT (β-VAE style)")
    print("-" * 40)
    print("Standard VAE uses β=1. β>1 encourages disentanglement.")

    class BetaVAE(VAE):
        def __init__(self, *args, beta=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta

        def compute_loss(self, x, x_recon, mu, logvar):
            batch_size = x.shape[0]
            x_recon = np.clip(x_recon, 1e-10, 1 - 1e-10)
            recon_loss = -np.sum(x * np.log(x_recon) + (1 - x) * np.log(1 - x_recon)) / batch_size
            kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / batch_size
            return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        vae = BetaVAE(input_dim=64, hidden_dim=32, latent_dim=4, beta=beta, learning_rate=0.01)
        losses = vae.fit(X, epochs=100, verbose=False)
        print(f"β={beta:<4} final_loss={losses['total'][-1]:.4f} "
              f"(recon={losses['recon'][-1]:.4f}, kl={losses['kl'][-1]:.4f})")
    print("→ Higher β = stronger regularization, more 'Gaussian' latent")
    print("→ Lower β = better reconstruction, but less structured latent")

    # -------- Experiment 3: Without KL Term --------
    print("\n3. VAE vs AUTOENCODER (No KL Term)")
    print("-" * 40)

    # VAE with KL
    vae = VAE(input_dim=64, hidden_dim=32, latent_dim=4, learning_rate=0.01)
    vae.fit(X, epochs=100, verbose=False)

    # "VAE" without KL (autoencoder)
    ae = BetaVAE(input_dim=64, hidden_dim=32, latent_dim=4, beta=0.0, learning_rate=0.01)
    ae.fit(X, epochs=100, verbose=False)

    # Check latent space structure
    vae_z = vae.encode_mean(X)
    ae_z = ae.encode_mean(X)

    print(f"VAE latent mean:  {np.mean(vae_z):.3f}, std: {np.std(vae_z):.3f}")
    print(f"AE latent mean:   {np.mean(ae_z):.3f}, std: {np.std(ae_z):.3f}")
    print("→ VAE latent is near N(0,1) due to KL term")
    print("→ AE latent can be arbitrary → poor generation")

    # -------- Experiment 4: Generation Quality --------
    print("\n4. GENERATION QUALITY")
    print("-" * 40)

    # Generate samples
    vae_samples = vae.generate(10)
    ae_samples = ae.generate(10)

    # Check if generated samples look like training data
    vae_sample_range = (vae_samples.min(), vae_samples.max())
    ae_sample_range = (ae_samples.min(), ae_samples.max())
    train_range = (X.min(), X.max())

    print(f"Training data range:  [{train_range[0]:.2f}, {train_range[1]:.2f}]")
    print(f"VAE generated range:  [{vae_sample_range[0]:.2f}, {vae_sample_range[1]:.2f}]")
    print(f"AE generated range:   [{ae_sample_range[0]:.2f}, {ae_sample_range[1]:.2f}]")
    print("→ VAE generates samples in training range")
    print("→ AE (no KL) generates unrealistic samples")

    # -------- Experiment 5: Hidden Layer Size --------
    print("\n5. EFFECT OF HIDDEN LAYER SIZE")
    print("-" * 40)
    for hidden_dim in [16, 32, 64, 128]:
        vae = VAE(input_dim=64, hidden_dim=hidden_dim, latent_dim=4, learning_rate=0.01)
        losses = vae.fit(X, epochs=100, verbose=False)
        n_params = 64*hidden_dim + hidden_dim*4 + hidden_dim*4 + 4*hidden_dim + hidden_dim*64
        print(f"hidden={hidden_dim:<4} ~params={n_params:<6} loss={losses['total'][-1]:.4f}")
    print("→ Larger hidden = more capacity, lower loss")

    # -------- Experiment 6: Reconstruction Quality --------
    print("\n6. RECONSTRUCTION QUALITY BY CLASS")
    print("-" * 40)
    vae = VAE(input_dim=64, hidden_dim=32, latent_dim=4, learning_rate=0.01)
    vae.fit(X, epochs=100, verbose=False)

    X_recon = vae.reconstruct(X)
    mse_per_class = []
    for c in range(4):
        mask = y == c
        mse = np.mean((X[mask] - X_recon[mask])**2)
        mse_per_class.append(mse)
        print(f"Class {c} MSE: {mse:.4f}")
    print("→ Some patterns may be easier to reconstruct than others")


def visualize_vae():
    """Visualize VAE latent space and generations."""
    print("\n" + "="*60)
    print("VAE VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    X, y = create_vae_dataset(n_samples=800, dataset='mnist_like')

    vae = VAE(input_dim=64, hidden_dim=32, latent_dim=2, learning_rate=0.01)
    losses = vae.fit(X, epochs=200, verbose=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Plot 1: Latent space
    ax = axes[0, 0]
    z = vae.encode_mean(X)
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='viridis', alpha=0.5, s=20)
    ax.set_title('Latent Space (colored by class)')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    plt.colorbar(scatter, ax=ax)

    # Plot 2: Training loss
    ax = axes[0, 1]
    ax.plot(losses['total'], label='Total')
    ax.plot(losses['recon'], label='Reconstruction')
    ax.plot(losses['kl'], label='KL')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Plot 3: Original vs Reconstructed
    ax = axes[0, 2]
    n_show = 5
    for i in range(n_show):
        # Original
        ax.imshow(X[i*100].reshape(8, 8), cmap='gray',
                  extent=[i*1.1, i*1.1+1, 1.1, 2.1])
        # Reconstructed
        x_recon = vae.reconstruct(X[i*100:i*100+1])
        ax.imshow(x_recon.reshape(8, 8), cmap='gray',
                  extent=[i*1.1, i*1.1+1, 0, 1])
    ax.set_xlim(-0.1, n_show*1.1)
    ax.set_ylim(-0.1, 2.2)
    ax.set_title('Original (top) vs Reconstructed (bottom)')
    ax.axis('off')

    # Plot 4: Generated samples
    ax = axes[1, 0]
    samples = vae.generate(25)
    for i in range(25):
        row, col = i // 5, i % 5
        ax.imshow(samples[i].reshape(8, 8), cmap='gray',
                  extent=[col*1.1, col*1.1+1, (4-row)*1.1, (4-row)*1.1+1])
    ax.set_xlim(-0.1, 5.5)
    ax.set_ylim(-0.1, 5.5)
    ax.set_title('Generated Samples (z ~ N(0,I))')
    ax.axis('off')

    # Plot 5: Latent space interpolation
    ax = axes[1, 1]
    # Interpolate between two points in latent space
    z1 = vae.encode_mean(X[0:1])
    z2 = vae.encode_mean(X[200:201])
    alphas = np.linspace(0, 1, 8)

    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z1 + alpha * z2
        x_interp, _ = vae.decode(z_interp)
        ax.imshow(x_interp.reshape(8, 8), cmap='gray',
                  extent=[i*1.1, i*1.1+1, 0, 1])
    ax.set_xlim(-0.1, 8.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Latent Space Interpolation')
    ax.axis('off')

    # Plot 6: Latent space grid
    ax = axes[1, 2]
    # Generate from a grid in latent space
    n_grid = 5
    z_range = np.linspace(-2, 2, n_grid)
    for i, z1_val in enumerate(z_range):
        for j, z2_val in enumerate(z_range):
            z = np.array([[z1_val, z2_val]])
            x_gen, _ = vae.decode(z)
            ax.imshow(x_gen.reshape(8, 8), cmap='gray',
                      extent=[j*1.1, j*1.1+1, (n_grid-1-i)*1.1, (n_grid-1-i)*1.1+1])
    ax.set_xlim(-0.1, n_grid*1.1)
    ax.set_ylim(-0.1, n_grid*1.1)
    ax.set_title('Decoded Grid (z₁, z₂ ∈ [-2, 2])')
    ax.axis('off')

    plt.suptitle('VARIATIONAL AUTOENCODER\n'
                 'Smooth latent space enables generation and interpolation',
                 fontsize=12)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("VAE — Variational Autoencoder")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Encoder: x → q(z|x) = N(μ, σ²)
    Decoder: z → p(x|z)
    A deep GENERATIVE model with structured latent space.

THE KEY INSIGHT:
    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
    Reconstruction + Regularization

    KL term forces latent to be ~N(0,I) → smooth, complete space

REPARAMETERIZATION TRICK:
    z = μ + σ × ε,  ε ~ N(0,I)
    Allows gradients to flow through sampling!

VAE vs AUTOENCODER:
    AE: deterministic, no principled generation
    VAE: probabilistic, smooth latent space, can generate

LATENT SPACE PROPERTIES:
    - Continuous: nearby z → similar x
    - Complete: every z → valid x
    - Enables: generation, interpolation, manipulation
    """)

    ablation_experiments()

    fig = visualize_vae()
    save_path = '/Users/sid47/ML Algorithms/18_vae.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. VAE = encoder (to latent) + decoder (from latent)
2. ELBO = reconstruction loss + KL divergence
3. Reparameterization trick enables backprop through sampling
4. KL term makes latent space smooth and complete
5. Without KL: just an autoencoder, poor generation
6. Latent dim controls trade-off: reconstruction vs regularity
    """)
