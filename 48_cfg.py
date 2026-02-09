"""
CLASSIFIER-FREE GUIDANCE (CFG) — Paradigm: GUIDED GENERATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Control WHAT the diffusion model generates without a separate classifier.

THE PROBLEM:
    Unconditional diffusion: generates random samples from data distribution
    We want: generate samples matching a CONDITION (class, text, etc.)

CLASSIFIER GUIDANCE (the old way):
    - Train a classifier p(y|x_t) on noisy images
    - At sampling: ε̃ = ε_θ - s × ∇_{x_t} log p(y|x_t)
    - Problem: Need separate classifier, trained on noisy data!

CLASSIFIER-FREE GUIDANCE (the elegant solution):
    - Train ONE model that can be conditional OR unconditional
    - At training: Drop condition randomly (e.g., 10% of time)
    - At sampling: Interpolate between conditional and unconditional

THE CFG EQUATION:
    ε̃ = ε_θ(x_t, ∅) + w × (ε_θ(x_t, c) - ε_θ(x_t, ∅))
       = (1-w) × ε_θ(x_t, ∅) + w × ε_θ(x_t, c)

    w = 1.0: Pure conditional (normal conditioning)
    w > 1.0: Push AWAY from unconditional (stronger guidance)
    w = 0.0: Pure unconditional
    w < 0.0: Negative guidance (avoid the condition)

===============================================================
THE MATHEMATICS
===============================================================

CLASSIFIER GUIDANCE DERIVATION:
    We want to sample from p(x|y) ∝ p(x)p(y|x)
    Score: ∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)

    In diffusion terms:
    ε̃(x_t, t, y) = ε_θ(x_t, t) - σ_t × s × ∇_{x_t} log p_φ(y|x_t)
                   \_________/         \__________________________/
                   unconditional        classifier gradient

CLASSIFIER-FREE TRICK:
    Train ε_θ(x_t, t, c) with c = class OR c = ∅ (null token)

    Key insight: The difference between conditional and unconditional
    is LIKE the classifier gradient!

    ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅) ≈ -σ_t × ∇_{x_t} log p(c|x_t)

    So: ε̃ = ε_θ(∅) + w × (ε_θ(c) - ε_θ(∅))
           = ε_θ(∅) + w × (-σ_t × ∇_x log p(c|x))
           ≈ Classifier guidance without classifier!

GUIDANCE SCALE (w):
    w = 1: Normal conditional model
    w > 1: Amplify the condition (push away from unconditional)

    Typical values: w ∈ [3, 15] for images
    Higher w → more aligned to condition, but less diversity

===============================================================
INDUCTIVE BIAS
===============================================================

1. UNIFIED MODEL: Same network for conditional and unconditional
   - Saves compute vs separate classifier
   - No domain mismatch (classifier sees same noisy data)

2. DROPOUT CONDITIONING: Random condition dropping during training
   - 10-20% drop rate typical
   - Model learns both modes

3. GUIDANCE SCALE: Controllable fidelity-diversity tradeoff
   - High w: High fidelity, low diversity
   - Low w: Low fidelity, high diversity
   - w > 1 "extrapolates" beyond training distribution

4. IMPLICIT CLASSIFIER: Learns to discriminate conditions
   - Never explicitly trained to classify
   - But conditional vs unconditional difference acts like gradient

WHEN CFG EXCELS:
- Class-conditional generation (ImageNet classes)
- Text-to-image (Stable Diffusion uses CFG)
- Any conditional generation task
- When you want to control strength of conditioning

WHEN CFG STRUGGLES:
- Very rare conditions (poor unconditional baseline)
- Too high guidance → artifacts, oversaturation
- Multiple conditions can conflict

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

# Import from previous files
from importlib import import_module
diffusion_fundamentals = import_module('45_diffusion_fundamentals')

# Import what we need
cosine_schedule = diffusion_fundamentals.cosine_schedule
get_schedule_params = diffusion_fundamentals.get_schedule_params
make_moons = diffusion_fundamentals.make_moons
make_gaussian_mixture = diffusion_fundamentals.make_gaussian_mixture


# =============================================================================
# CONDITIONAL DENOISING NETWORK
# =============================================================================

class ConditionalDenoisingMLP:
    """
    MLP that predicts noise, CONDITIONED on class label.

    Architecture:
        [x_t, t_emb, c_emb] → Linear → SiLU → Linear → SiLU → Linear → SiLU → Linear → ε_pred

    Condition c is embedded via learned embedding table.
    Null token (unconditional) is a special class.
    """

    def __init__(self, data_dim=2, hidden_dim=128, time_emb_dim=32,
                 num_classes=4, cond_emb_dim=16, null_class_idx=None):
        """
        Args:
            data_dim: Dimension of data (2 for 2D points)
            hidden_dim: Hidden layer size
            time_emb_dim: Timestep embedding dimension
            num_classes: Number of classes (excluding null token)
            cond_emb_dim: Class embedding dimension
            null_class_idx: Index for null/unconditional token (default: num_classes)
        """
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.cond_emb_dim = cond_emb_dim

        # Null class is the last class
        self.null_class_idx = null_class_idx if null_class_idx is not None else num_classes
        self.total_classes = num_classes + 1  # +1 for null token

        # Class embedding table
        self.class_emb = np.random.randn(self.total_classes, cond_emb_dim) * 0.1

        # Input: data + time_emb + cond_emb
        input_dim = data_dim + time_emb_dim + cond_emb_dim

        # Network weights (3 hidden layers like DenoisingMLP)
        scale1 = np.sqrt(2. / input_dim)
        scale2 = np.sqrt(2. / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b3 = np.zeros(hidden_dim)

        self.W4 = np.random.randn(hidden_dim, data_dim) * scale2
        self.b4 = np.zeros(data_dim)

        # Time embedding frequencies (sinusoidal)
        self.time_freqs = np.exp(np.linspace(0, np.log(1000), time_emb_dim // 2))

    def silu(self, x):
        """SiLU activation: x * sigmoid(x)"""
        return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))

    def time_embedding(self, t):
        """Sinusoidal time embedding."""
        t = np.atleast_1d(t).reshape(-1, 1)  # (B, 1)
        args = t * self.time_freqs  # (B, time_emb_dim/2)
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)  # (B, time_emb_dim)
        return emb

    def get_class_embedding(self, c):
        """Get class embedding for class indices."""
        return self.class_emb[c]  # (B, cond_emb_dim)

    def forward(self, x_t, t, c):
        """
        Forward pass: predict noise given noisy input, timestep, and class.

        Args:
            x_t: Noisy input (B, data_dim)
            t: Timestep indices (B,)
            c: Class indices (B,), use self.null_class_idx for unconditional

        Returns:
            ε_pred: Predicted noise (B, data_dim)
        """
        # Get embeddings
        t_emb = self.time_embedding(t)  # (B, time_emb_dim)
        c_emb = self.get_class_embedding(c)  # (B, cond_emb_dim)

        # Concatenate inputs
        self.x_in = np.concatenate([x_t, t_emb, c_emb], axis=-1)  # (B, input_dim)

        # Forward through network
        self.h1 = self.silu(self.x_in @ self.W1 + self.b1)
        self.h2 = self.silu(self.h1 @ self.W2 + self.b2)
        self.h3 = self.silu(self.h2 @ self.W3 + self.b3)
        self.out = self.h3 @ self.W4 + self.b4

        return self.out

    def backward(self, grad_out, lr=1e-3):
        """Backward pass and weight update."""
        batch_size = grad_out.shape[0]

        # Layer 4
        grad_h3 = grad_out @ self.W4.T
        grad_W4 = self.h3.T @ grad_out / batch_size
        grad_b4 = np.mean(grad_out, axis=0)

        # Layer 3 (SiLU)
        sigmoid_h3 = 1 / (1 + np.exp(-np.clip(self.h2 @ self.W3 + self.b3, -500, 500)))
        grad_h3_pre = grad_h3 * (sigmoid_h3 + (self.h2 @ self.W3 + self.b3) * sigmoid_h3 * (1 - sigmoid_h3))
        grad_h2 = grad_h3_pre @ self.W3.T
        grad_W3 = self.h2.T @ grad_h3_pre / batch_size
        grad_b3 = np.mean(grad_h3_pre, axis=0)

        # Layer 2 (SiLU)
        sigmoid_h2 = 1 / (1 + np.exp(-np.clip(self.h1 @ self.W2 + self.b2, -500, 500)))
        grad_h2_pre = grad_h2 * (sigmoid_h2 + (self.h1 @ self.W2 + self.b2) * sigmoid_h2 * (1 - sigmoid_h2))
        grad_h1 = grad_h2_pre @ self.W2.T
        grad_W2 = self.h1.T @ grad_h2_pre / batch_size
        grad_b2 = np.mean(grad_h2_pre, axis=0)

        # Layer 1 (SiLU)
        sigmoid_h1 = 1 / (1 + np.exp(-np.clip(self.x_in @ self.W1 + self.b1, -500, 500)))
        grad_h1_pre = grad_h1 * (sigmoid_h1 + (self.x_in @ self.W1 + self.b1) * sigmoid_h1 * (1 - sigmoid_h1))
        grad_W1 = self.x_in.T @ grad_h1_pre / batch_size
        grad_b1 = np.mean(grad_h1_pre, axis=0)

        # Gradient for class embedding
        grad_x_in = grad_h1_pre @ self.W1.T
        grad_c_emb = grad_x_in[:, -self.cond_emb_dim:]  # Last cond_emb_dim dimensions

        # Update weights
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W3 -= lr * grad_W3
        self.b3 -= lr * grad_b3
        self.W4 -= lr * grad_W4
        self.b4 -= lr * grad_b4

        return np.mean(grad_out ** 2)


# =============================================================================
# CFG TRAINER
# =============================================================================

class CFGTrainer:
    """
    Classifier-Free Guidance trainer.

    Key innovation: During training, randomly drop condition with probability p_uncond.
    This teaches the model to generate both conditional and unconditional.
    """

    def __init__(self, model, T=1000, schedule='cosine', p_uncond=0.1):
        """
        Args:
            model: ConditionalDenoisingMLP
            T: Number of diffusion timesteps
            schedule: 'linear' or 'cosine'
            p_uncond: Probability of dropping condition (using null token)
        """
        self.model = model
        self.T = T
        self.p_uncond = p_uncond

        # Get noise schedule
        if schedule == 'linear':
            betas = np.linspace(1e-4, 0.02, T)
        else:
            betas = cosine_schedule(T)

        self.params = get_schedule_params(betas)
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = self.params['alphas_cumprod']

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: Sample x_t from q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = np.sqrt(self.alpha_bars[t]).reshape(-1, 1)
        sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def train_step(self, x_0, c, lr=1e-3):
        """
        Single training step with classifier-free dropout.

        Args:
            x_0: Clean data (B, data_dim)
            c: Class labels (B,)
            lr: Learning rate

        Returns:
            loss: MSE loss between predicted and true noise
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = np.random.randint(0, self.T, size=batch_size)

        # Sample noise and get x_t
        noise = np.random.randn(*x_0.shape)
        x_t, _ = self.q_sample(x_0, t, noise)

        # CLASSIFIER-FREE DROPOUT: Replace some conditions with null token
        c_dropped = c.copy()
        drop_mask = np.random.rand(batch_size) < self.p_uncond
        c_dropped[drop_mask] = self.model.null_class_idx

        # Predict noise
        noise_pred = self.model.forward(x_t, t, c_dropped)

        # Loss: MSE between predicted and true noise
        loss = np.mean((noise_pred - noise) ** 2)

        # Backward pass
        grad = 2 * (noise_pred - noise) / batch_size
        self.model.backward(grad, lr)

        return loss

    def train(self, X, labels, epochs=1000, batch_size=128, lr=1e-3, verbose=True):
        """
        Train the model.

        Args:
            X: Training data (N, data_dim)
            labels: Class labels (N,)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            losses: List of losses per epoch
        """
        N = len(X)
        losses = []

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(N)
            X_shuffled = X[idx]
            labels_shuffled = labels[idx]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, N, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]

                loss = self.train_step(batch_X, batch_labels, lr)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses


# =============================================================================
# CFG SAMPLER
# =============================================================================

class CFGSampler:
    """
    Sample from a CFG-trained diffusion model.

    The key: Combine conditional and unconditional predictions with guidance scale.
    """

    def __init__(self, trainer):
        """
        Args:
            trainer: CFGTrainer with trained model
        """
        self.model = trainer.model
        self.T = trainer.T
        self.params = trainer.params
        self.betas = trainer.betas
        self.alphas = trainer.alphas
        self.alpha_bars = trainer.alpha_bars

    def p_mean_variance(self, x_t, t, c, guidance_scale=1.0):
        """
        Compute mean for reverse step using CFG.

        ε̃ = ε_uncond + w × (ε_cond - ε_uncond)
           = (1-w) × ε_uncond + w × ε_cond

        Args:
            x_t: Current noisy sample
            t: Timestep
            c: Class condition
            guidance_scale: CFG weight (w)

        Returns:
            mean: Posterior mean for x_{t-1}
        """
        # Get batch dimension
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
        batch_size = x_t.shape[0]

        # Create timestep array
        t_arr = np.full(batch_size, t, dtype=np.int32)

        # Unconditional prediction (null token)
        c_uncond = np.full(batch_size, self.model.null_class_idx, dtype=np.int32)
        eps_uncond = self.model.forward(x_t.copy(), t_arr, c_uncond)

        if guidance_scale == 0.0:
            # Pure unconditional
            eps_pred = eps_uncond
        elif guidance_scale == 1.0:
            # Pure conditional (no guidance needed, just conditional)
            eps_pred = self.model.forward(x_t.copy(), t_arr, c)
        else:
            # CFG: interpolate/extrapolate
            eps_cond = self.model.forward(x_t.copy(), t_arr, c)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Compute posterior mean
        # μ_θ = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        sqrt_alpha = np.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar_t)

        mean = (1 / sqrt_alpha) * (x_t - (beta_t / sqrt_one_minus_alpha_bar) * eps_pred)

        return mean

    def sample(self, class_label, n_samples=100, guidance_scale=3.0, seed=None):
        """
        Generate samples conditioned on class with CFG.

        Args:
            class_label: Class to generate (int)
            n_samples: Number of samples
            guidance_scale: CFG weight (higher = stronger conditioning)
            seed: Random seed for reproducibility

        Returns:
            samples: Generated samples (n_samples, data_dim)
        """
        if seed is not None:
            np.random.seed(seed)

        # Start from pure noise
        x_t = np.random.randn(n_samples, self.model.data_dim)

        # Class condition
        c = np.full(n_samples, class_label, dtype=np.int32)

        # Reverse diffusion
        for t in range(self.T - 1, -1, -1):
            z = np.random.randn(*x_t.shape) if t > 0 else np.zeros_like(x_t)

            mean = self.p_mean_variance(x_t, t, c, guidance_scale)

            # Add noise (variance)
            sigma_t = np.sqrt(self.betas[t])
            x_t = mean + sigma_t * z

        return x_t

    def sample_unconditional(self, n_samples=100, seed=None):
        """Sample unconditionally (guidance_scale=0)."""
        return self.sample(self.model.null_class_idx, n_samples, guidance_scale=0.0, seed=seed)


# =============================================================================
# DATASET: LABELED CLUSTERS
# =============================================================================

def make_labeled_clusters(n_samples=1000, n_clusters=4, std=0.3, seed=42):
    """
    Create labeled 2D clusters for conditional generation.

    Args:
        n_samples: Total number of samples
        n_clusters: Number of clusters (classes)
        std: Standard deviation of clusters
        seed: Random seed

    Returns:
        X: Data points (n_samples, 2)
        labels: Cluster labels (n_samples,)
    """
    np.random.seed(seed)

    samples_per_cluster = n_samples // n_clusters
    X = []
    labels = []

    # Place clusters in a circle
    for i in range(n_clusters):
        angle = 2 * np.pi * i / n_clusters
        center = np.array([np.cos(angle) * 2, np.sin(angle) * 2])

        cluster_points = np.random.randn(samples_per_cluster, 2) * std + center
        X.append(cluster_points)
        labels.append(np.full(samples_per_cluster, i))

    X = np.vstack(X)
    labels = np.concatenate(labels)

    return X, labels


def make_labeled_moons(n_samples=1000, noise=0.1, seed=42):
    """
    Create labeled moons dataset (2 classes).

    Args:
        n_samples: Total samples (half per moon)
        noise: Noise level
        seed: Random seed

    Returns:
        X: Data points (n_samples, 2)
        labels: Moon labels (n_samples,), 0 or 1
    """
    np.random.seed(seed)

    n_per_moon = n_samples // 2

    # Moon 1 (top)
    theta1 = np.linspace(0, np.pi, n_per_moon)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    moon1 = np.column_stack([x1, y1]) + np.random.randn(n_per_moon, 2) * noise

    # Moon 2 (bottom, offset)
    theta2 = np.linspace(0, np.pi, n_per_moon)
    x2 = 1 - np.cos(theta2)
    y2 = -np.sin(theta2) - 0.5
    moon2 = np.column_stack([x2, y2]) + np.random.randn(n_per_moon, 2) * noise

    X = np.vstack([moon1, moon2])
    labels = np.array([0] * n_per_moon + [1] * n_per_moon)

    return X, labels


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def visualize_cfg_samples(trainer, guidance_scales=[0.0, 1.0, 3.0, 7.0],
                          n_samples=200, num_classes=4, save_path=None):
    """
    Visualize effect of different guidance scales.
    """
    sampler = CFGSampler(trainer)

    fig, axes = plt.subplots(num_classes, len(guidance_scales), figsize=(4*len(guidance_scales), 4*num_classes))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for row, class_idx in enumerate(range(num_classes)):
        for col, w in enumerate(guidance_scales):
            ax = axes[row, col] if num_classes > 1 else axes[col]

            # Generate samples
            samples = sampler.sample(class_idx, n_samples=n_samples, guidance_scale=w, seed=42)

            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, c=[colors[class_idx]])
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')

            if row == 0:
                ax.set_title(f'w = {w}', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Class {class_idx}', fontsize=12)

    plt.suptitle('Classifier-Free Guidance: Effect of Guidance Scale\n'
                 'w=0: unconditional | w=1: pure conditional | w>1: amplified guidance',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def visualize_training_data(X, labels, title="Training Data", save_path=None):
    """Visualize labeled training data."""
    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def visualize_guidance_comparison(trainer, X_train, labels_train, guidance_scales=[0.0, 1.0, 5.0, 10.0],
                                   n_samples=300, save_path=None):
    """
    Compare training data vs generated samples at different guidance scales.
    """
    sampler = CFGSampler(trainer)
    num_classes = trainer.model.num_classes

    fig, axes = plt.subplots(2, len(guidance_scales), figsize=(4*len(guidance_scales), 8))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Top row: Generated samples
    for col, w in enumerate(guidance_scales):
        ax = axes[0, col]

        for class_idx in range(num_classes):
            samples = sampler.sample(class_idx, n_samples=n_samples//num_classes, guidance_scale=w, seed=42+class_idx)
            ax.scatter(samples[:, 0], samples[:, 1], c=[colors[class_idx]], alpha=0.6, s=10, label=f'Class {class_idx}')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(f'Generated (w={w})', fontsize=12)
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('Generated', fontsize=12)

    # Bottom row: Training data (repeated for comparison)
    for col in range(len(guidance_scales)):
        ax = axes[1, col]

        for i in range(num_classes):
            mask = labels_train == i
            ax.scatter(X_train[mask, 0], X_train[mask, 1], c=[colors[i]], alpha=0.6, s=10)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title('Training Data', fontsize=12)
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('Real Data', fontsize=12)

    plt.suptitle('CFG: Generated Samples vs Training Data\n'
                 'Higher guidance → samples cluster more tightly around class centers',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def ablation_p_uncond(X, labels, p_uncond_values=[0.0, 0.1, 0.2, 0.5],
                      epochs=500, save_path=None):
    """
    Ablation: Effect of unconditional dropout rate.
    """
    num_classes = len(np.unique(labels))
    fig, axes = plt.subplots(num_classes, len(p_uncond_values),
                              figsize=(4*len(p_uncond_values), 4*num_classes))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for col, p_uncond in enumerate(p_uncond_values):
        print(f"\nTraining with p_uncond={p_uncond}")

        # Train model
        model = ConditionalDenoisingMLP(
            data_dim=2, hidden_dim=128, time_emb_dim=32,
            num_classes=num_classes, cond_emb_dim=16
        )
        trainer = CFGTrainer(model, T=500, schedule='cosine', p_uncond=p_uncond)
        trainer.train(X, labels, epochs=epochs, lr=1e-3, verbose=False)

        # Sample
        sampler = CFGSampler(trainer)

        for row, class_idx in enumerate(range(num_classes)):
            ax = axes[row, col] if num_classes > 1 else axes[col]

            # Use guidance scale w=5 to see conditioning effect
            samples = sampler.sample(class_idx, n_samples=150, guidance_scale=5.0, seed=42)

            ax.scatter(samples[:, 0], samples[:, 1], c=[colors[class_idx]], alpha=0.6, s=10)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')

            if row == 0:
                ax.set_title(f'p_uncond = {p_uncond}', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Class {class_idx}', fontsize=12)

    plt.suptitle('Ablation: Unconditional Dropout Rate (p_uncond)\n'
                 'p=0: No dropout (pure conditional) | p=0.1-0.2: Good CFG | p=0.5: Too much dropout',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def ablation_guidance_interpolation(trainer, class_from=0, class_to=1,
                                      weights=np.linspace(-1, 2, 7), save_path=None):
    """
    Ablation: Interpolation between classes using guidance.

    Shows what happens with negative guidance and extrapolation.
    """
    sampler = CFGSampler(trainer)

    fig, axes = plt.subplots(1, len(weights), figsize=(3*len(weights), 3))

    for col, w in enumerate(weights):
        ax = axes[col]

        # Generate samples with varying guidance
        samples = sampler.sample(class_from, n_samples=150, guidance_scale=w, seed=42)

        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, c='blue')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f'w = {w:.1f}', fontsize=10)

    plt.suptitle(f'Guidance Scale Interpolation (Class {class_from})\n'
                 'w<0: Negative guidance | w=0: Unconditional | w=1: Conditional | w>1: Amplified',
                 fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN: DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CLASSIFIER-FREE GUIDANCE (CFG) — Paradigm: GUIDED GENERATION")
    print("="*70)

    # Create labeled dataset
    print("\n1. Creating labeled clusters dataset...")
    X, labels = make_labeled_clusters(n_samples=2000, n_clusters=4, std=0.4, seed=42)
    num_classes = len(np.unique(labels))
    print(f"   Data shape: {X.shape}, Classes: {num_classes}")

    # Visualize training data
    visualize_training_data(X, labels, title="Training Data: 4 Labeled Clusters",
                            save_path='/Users/sid47/ML Algorithms/48_cfg_training_data.png')

    # Create model and trainer
    print("\n2. Creating conditional diffusion model...")
    model = ConditionalDenoisingMLP(
        data_dim=2,
        hidden_dim=256,
        time_emb_dim=32,
        num_classes=num_classes,
        cond_emb_dim=32
    )

    trainer = CFGTrainer(
        model,
        T=500,
        schedule='cosine',
        p_uncond=0.1  # Drop condition 10% of time
    )

    # Train
    print("\n3. Training with classifier-free dropout (p_uncond=0.1)...")
    losses = trainer.train(X, labels, epochs=1500, batch_size=128, lr=1e-3)

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CFG Training Loss')
    plt.savefig('/Users/sid47/ML Algorithms/48_cfg_training_loss.png', dpi=150, bbox_inches='tight')
    print("Saved: /Users/sid47/ML Algorithms/48_cfg_training_loss.png")
    plt.close()

    # Visualize guidance scale effect
    print("\n4. Visualizing guidance scale effect...")
    visualize_cfg_samples(
        trainer,
        guidance_scales=[0.0, 1.0, 3.0, 7.0],
        n_samples=200,
        num_classes=num_classes,
        save_path='/Users/sid47/ML Algorithms/48_cfg_guidance_scales.png'
    )

    # Compare to training data
    print("\n5. Comparing generated vs training data...")
    visualize_guidance_comparison(
        trainer, X, labels,
        guidance_scales=[0.0, 1.0, 5.0, 10.0],
        n_samples=400,
        save_path='/Users/sid47/ML Algorithms/48_cfg_comparison.png'
    )

    # Ablation: p_uncond
    print("\n6. Ablation: Effect of unconditional dropout rate...")
    ablation_p_uncond(
        X, labels,
        p_uncond_values=[0.0, 0.1, 0.2, 0.5],
        epochs=800,
        save_path='/Users/sid47/ML Algorithms/48_cfg_ablation_puncond.png'
    )

    # Ablation: Guidance interpolation
    print("\n7. Ablation: Guidance interpolation (including negative guidance)...")
    ablation_guidance_interpolation(
        trainer,
        class_from=0,
        weights=np.linspace(-1, 3, 9),
        save_path='/Users/sid47/ML Algorithms/48_cfg_ablation_interpolation.png'
    )

    # Summary
    print("\n" + "="*70)
    print("CLASSIFIER-FREE GUIDANCE — SUMMARY")
    print("="*70)
    print("""
KEY INSIGHTS:

1. UNIFIED TRAINING: One model learns both conditional and unconditional
   - Random dropout of condition during training (p_uncond)
   - No separate classifier needed

2. GUIDANCE SCALE (w):
   - w = 0: Unconditional (no class information)
   - w = 1: Standard conditional
   - w > 1: Amplified conditioning (more focused on class)
   - w < 0: Negative guidance (avoid the class)

3. THE CFG EQUATION:
   ε̃ = ε_uncond + w × (ε_cond - ε_uncond)

   This INTERPOLATES between unconditional and conditional predictions.
   When w > 1, it EXTRAPOLATES beyond pure conditioning.

4. p_uncond MATTERS:
   - p = 0: No unconditional learning (CFG won't work)
   - p = 0.1-0.2: Good balance (typical)
   - p = 0.5: Too much dropout (weak conditional signal)

5. APPLICATIONS:
   - Text-to-image (Stable Diffusion uses CFG)
   - Class-conditional generation (ImageNet)
   - Any conditional generation with controllable strength
    """)

    print("\nGenerated visualizations:")
    print("  - 48_cfg_training_data.png: Labeled training clusters")
    print("  - 48_cfg_training_loss.png: Training convergence")
    print("  - 48_cfg_guidance_scales.png: Effect of guidance scale per class")
    print("  - 48_cfg_comparison.png: Generated vs real data comparison")
    print("  - 48_cfg_ablation_puncond.png: Effect of dropout rate")
    print("  - 48_cfg_ablation_interpolation.png: Guidance interpolation/extrapolation")
