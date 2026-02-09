"""
DIFFUSION FUNDAMENTALS — Paradigm: ITERATIVE DENOISING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Destroy data with noise, then learn to REVERSE the process.

Think of it like this:
    1. Take a clear image
    2. Add noise, add noise, add noise... until it's pure static
    3. Train a neural network to UNDO each noise step
    4. Start from static → reverse step by step → get realistic image!

FORWARD PROCESS (fixed, no learning):
    x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)

    Each step adds Gaussian noise:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

REVERSE PROCESS (learned):
    x_T → x_{T-1} → ... → x_0

    Neural network predicts the noise to remove:
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

===============================================================
THE MATHEMATICS
===============================================================

FORWARD PROCESS (adding noise):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

REPARAMETERIZATION TRICK (jump to any timestep):
    x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,  where ε ~ N(0,I)

    where:
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^t α_s  (cumulative product)

THREE EQUIVALENT TRAINING OBJECTIVES:
    1. PREDICT x_0:     L = ||x_0 - x̂_0(x_t, t)||²
    2. PREDICT ε:       L = ||ε - ε_θ(x_t, t)||²  ← Most common!
    3. PREDICT SCORE:   L = ||∇_x log p(x_t) - s_θ(x_t, t)||²

REVERSE PROCESS (removing noise):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    where:
    μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t))

===============================================================
INDUCTIVE BIAS
===============================================================

1. GAUSSIAN NOISE: Assumes Gaussian perturbations are sufficient
   - Works because Gaussian is maximum entropy for given variance
   - Central limit theorem: sum of many small noises → Gaussian

2. MARKOV CHAIN: Each step only depends on previous state
   - Simplifies training (can sample any t independently)
   - Limits expressivity (no long-range dependencies in diffusion)

3. SHARED NETWORK: Single network handles ALL timesteps
   - t is an INPUT to the network (typically via embedding)
   - Forces network to learn noise at all scales
   - Efficient: one model instead of T models

4. SLOW SAMPLING: Need many steps for quality
   - T=1000 typical for training
   - Each step only removes a little noise
   - Tradeoff: more steps = better quality, slower inference

5. ISOTROPIC NOISE: Same noise scale in all dimensions
   - Real data may have different variance per dimension
   - Can be addressed with learned variance or preprocessing

WHEN DIFFUSION WORKS WELL:
- High-dimensional data (images, audio)
- When diversity matters (many valid outputs)
- When you have lots of compute for sampling

WHEN DIFFUSION STRUGGLES:
- Discrete data (text) — need modifications
- Real-time generation — too slow
- Low-data regime — needs lots of training data

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# NOISE SCHEDULES
# ============================================================

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Linear noise schedule.

    β_t increases linearly from beta_start to beta_end.
    Simple but not optimal — too much noise added early.
    """
    return np.linspace(beta_start, beta_end, T)


def cosine_schedule(T, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    ᾱ_t = cos²((t/T + s) / (1 + s) × π/2)

    Smoother than linear, better for images.
    The 's' offset prevents β_t from being too small at t=0.
    """
    steps = np.arange(T + 1)
    f_t = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def get_schedule_params(betas):
    """
    Compute all derived quantities from beta schedule.

    Returns dict with:
        betas: β_t
        alphas: α_t = 1 - β_t
        alphas_cumprod: ᾱ_t = ∏_{s=1}^t α_s
        sqrt_alphas_cumprod: √ᾱ_t (for forward process)
        sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t) (for forward process)
        sqrt_recip_alphas: 1/√α_t (for reverse process)
        posterior_variance: σ_t² (for reverse process)
    """
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.concatenate([[1.], alphas_cumprod[:-1]])

    # For forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    # For reverse process
    sqrt_recip_alphas = 1. / np.sqrt(alphas)

    # Posterior variance: σ_t² = β_t × (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }


# ============================================================
# FORWARD PROCESS (Adding Noise)
# ============================================================

def forward_diffusion_sample(x_0, t, schedule_params, noise=None):
    """
    Sample x_t from q(x_t | x_0) using reparameterization.

    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

    This lets us jump DIRECTLY to any timestep without iterating!

    Args:
        x_0: Original data, shape (batch, ...)
        t: Timestep indices, shape (batch,)
        schedule_params: Dict from get_schedule_params
        noise: Optional pre-sampled noise

    Returns:
        x_t: Noised data at timestep t
        noise: The noise that was added
    """
    if noise is None:
        noise = np.random.randn(*x_0.shape)

    sqrt_alphas_cumprod_t = schedule_params['sqrt_alphas_cumprod'][t]
    sqrt_one_minus_alphas_cumprod_t = schedule_params['sqrt_one_minus_alphas_cumprod'][t]

    # Reshape for broadcasting
    while sqrt_alphas_cumprod_t.ndim < x_0.ndim:
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., np.newaxis]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., np.newaxis]

    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    return x_t, noise


# ============================================================
# SIMPLE DENOISING NETWORK (MLP for 2D data)
# ============================================================

class SinusoidalPositionEmbedding:
    """
    Sinusoidal embedding for timestep t.

    Same idea as Transformer positional encoding:
    - Different frequencies capture different scales
    - Allows network to distinguish timesteps smoothly

    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, t):
        """
        Args:
            t: Timestep, shape (batch,) or scalar
        Returns:
            Embedding, shape (batch, dim)
        """
        t = np.atleast_1d(t).astype(float)
        half_dim = self.dim // 2

        # Frequencies: 1, 1/10000^(1/half_dim), 1/10000^(2/half_dim), ...
        freqs = np.exp(-np.log(10000) * np.arange(half_dim) / half_dim)

        # Outer product: (batch, 1) × (half_dim,) → (batch, half_dim)
        args = t[:, np.newaxis] * freqs[np.newaxis, :]

        # Interleave sin and cos
        embedding = np.concatenate([np.sin(args), np.cos(args)], axis=-1)

        return embedding


class DenoisingMLP:
    """
    Simple but effective MLP for denoising 2D data.

    Architecture:
        [x_t, time_emb] → Linear → SiLU → Linear → SiLU → Linear → ε_pred

    Key design choices:
    1. CONCATENATE x and time embedding (simple, works well for 2D)
    2. SiLU/Swish activation (smoother than ReLU)
    3. 3 hidden layers with skip connection from input

    For images, you'd use a UNet instead. But for understanding
    the fundamentals, an MLP on 2D points is clearer.
    """

    def __init__(self, data_dim=2, hidden_dim=128, time_emb_dim=32):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim

        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)

        # Network layers
        input_dim = data_dim + time_emb_dim

        # Layer 1
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)

        # Layer 2
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(hidden_dim)

        # Layer 3
        self.W3 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b3 = np.zeros(hidden_dim)

        # Output layer
        self.W4 = np.random.randn(hidden_dim, data_dim) * np.sqrt(2. / hidden_dim)
        self.b4 = np.zeros(data_dim)

        # Collect params for optimizer
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.adam_t = 0

    def silu(self, x):
        """SiLU/Swish activation: x * sigmoid(x)"""
        return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))

    def silu_backward(self, x, d_out):
        """Backward for SiLU"""
        sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return d_out * (sig + x * sig * (1 - sig))

    def forward(self, x_t, t):
        """
        Predict the noise ε given noisy data x_t and timestep t.
        """
        # Get time embedding
        t_emb = self.time_embedding(t)

        # Concatenate input and time embedding
        self.x_in = np.concatenate([x_t, t_emb], axis=-1)

        # Forward through network
        self.z1 = self.x_in @ self.W1 + self.b1
        self.h1 = self.silu(self.z1)

        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = self.silu(self.z2)

        self.z3 = self.h2 @ self.W3 + self.b3
        self.h3 = self.silu(self.z3)

        self.out = self.h3 @ self.W4 + self.b4

        return self.out

    def backward(self, d_out):
        """Backward pass to compute gradients."""
        batch_size = d_out.shape[0]

        # Layer 4
        self.dW4 = self.h3.T @ d_out / batch_size
        self.db4 = np.mean(d_out, axis=0)
        dh3 = d_out @ self.W4.T

        # Layer 3
        dz3 = self.silu_backward(self.z3, dh3)
        self.dW3 = self.h2.T @ dz3 / batch_size
        self.db3 = np.mean(dz3, axis=0)
        dh2 = dz3 @ self.W3.T

        # Layer 2
        dz2 = self.silu_backward(self.z2, dh2)
        self.dW2 = self.h1.T @ dz2 / batch_size
        self.db2 = np.mean(dz2, axis=0)
        dh1 = dz2 @ self.W2.T

        # Layer 1
        dz1 = self.silu_backward(self.z1, dh1)
        self.dW1 = self.x_in.T @ dz1 / batch_size
        self.db1 = np.mean(dz1, axis=0)

        self.grads = [self.dW1, self.db1, self.dW2, self.db2, self.dW3, self.db3, self.dW4, self.db4]

    def update(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update."""
        self.adam_t += 1

        for i, (param, grad) in enumerate(zip(self.params, self.grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - beta1 ** self.adam_t)
            v_hat = self.v[i] / (1 - beta2 ** self.adam_t)

            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update references
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = self.params


# ============================================================
# TRAINING
# ============================================================

def train_diffusion(model, data, schedule_params, n_epochs=1000, batch_size=128, lr=1e-3):
    """
    Train the diffusion model.

    THE TRAINING ALGORITHM:
        1. Sample x_0 from data
        2. Sample t ~ Uniform(1, T)
        3. Sample ε ~ N(0, I)
        4. Compute x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        5. Predict ε̂ = model(x_t, t)
        6. Loss = ||ε - ε̂||²
    """
    T = len(schedule_params['betas'])
    n_samples = len(data)
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            # Get batch
            idx = perm[i:i+batch_size]
            x_0 = data[idx]
            batch_len = len(x_0)

            # Sample random timesteps
            t = np.random.randint(0, T, size=batch_len)

            # Forward diffusion: add noise
            x_t, noise = forward_diffusion_sample(x_0, t, schedule_params)

            # Predict noise
            noise_pred = model.forward(x_t, t)

            # MSE loss
            loss = np.mean((noise - noise_pred) ** 2)
            epoch_loss += loss
            n_batches += 1

            # Backward pass
            d_loss = 2 * (noise_pred - noise) / noise_pred.size
            model.backward(d_loss)
            model.update(lr=lr)

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    return losses


# ============================================================
# SAMPLING (Reverse Process)
# ============================================================

def sample_ddpm(model, schedule_params, n_samples=100, data_dim=2):
    """
    Generate samples using DDPM reverse process.

    THE SAMPLING ALGORITHM:
        1. Sample x_T ~ N(0, I)
        2. For t = T-1, T-2, ..., 0:
            - Predict noise: ε̂ = model(x_t, t)
            - Compute mean: μ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε̂)
            - Sample: x_{t-1} = μ + σ_t × z, where z ~ N(0,I)
        3. Return x_0
    """
    T = len(schedule_params['betas'])

    # Start from pure noise
    x = np.random.randn(n_samples, data_dim)

    # Store trajectory for visualization
    trajectory = [x.copy()]

    # Reverse process
    for t in reversed(range(T)):
        t_batch = np.full(n_samples, t)

        # Predict noise
        noise_pred = model.forward(x, t_batch)

        # Get schedule parameters for this timestep
        beta_t = schedule_params['betas'][t]
        sqrt_recip_alpha_t = schedule_params['sqrt_recip_alphas'][t]
        sqrt_one_minus_alpha_cumprod_t = schedule_params['sqrt_one_minus_alphas_cumprod'][t]

        # Compute mean
        # μ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε̂)
        mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * noise_pred)

        # Add noise (except at t=0)
        if t > 0:
            noise = np.random.randn(*x.shape)
            sigma_t = np.sqrt(schedule_params['posterior_variance'][t])
            x = mean + sigma_t * noise
        else:
            x = mean

        # Store some steps for visualization
        if t % (T // 10) == 0 or t < 10:
            trajectory.append(x.copy())

    return x, trajectory


# ============================================================
# DATASETS
# ============================================================

def make_swiss_roll(n_samples=1000):
    """Swiss roll in 2D."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    data = data / np.max(np.abs(data))  # Normalize to [-1, 1]
    return data.astype(np.float32)


def make_moons(n_samples=1000):
    """Two interleaving half-moons."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.stack([outer_circ_x, outer_circ_y], axis=1),
        np.stack([inner_circ_x, inner_circ_y], axis=1)
    ])
    X += 0.05 * np.random.randn(*X.shape)
    X = X - X.mean(axis=0)
    X = X / np.max(np.abs(X))
    return X.astype(np.float32)


def make_circles(n_samples=1000):
    """Two concentric circles."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = 2 * np.pi * np.random.rand(n_outer)
    theta_inner = 2 * np.pi * np.random.rand(n_inner)

    X_outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    X_inner = 0.5 * np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1)

    X = np.vstack([X_outer, X_inner])
    X += 0.05 * np.random.randn(*X.shape)
    return X.astype(np.float32)


def make_gaussian_mixture(n_samples=1000, n_components=8):
    """Mixture of Gaussians arranged in a circle."""
    samples_per_component = n_samples // n_components

    X = []
    for i in range(n_components):
        angle = 2 * np.pi * i / n_components
        center = np.array([np.cos(angle), np.sin(angle)]) * 0.7
        samples = center + 0.1 * np.random.randn(samples_per_component, 2)
        X.append(samples)

    X = np.vstack(X)
    return X.astype(np.float32)


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_forward_process(data, schedule_params, n_steps=10):
    """Visualize how data gets progressively noisier."""
    T = len(schedule_params['betas'])
    timesteps = np.linspace(0, T-1, n_steps).astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, t in enumerate(timesteps):
        t_batch = np.full(len(data), t)
        x_t, _ = forward_diffusion_sample(data, t_batch, schedule_params)

        axes[i].scatter(x_t[:, 0], x_t[:, 1], s=1, alpha=0.5)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_title(f't = {t}\nᾱ_t = {schedule_params["alphas_cumprod"][t]:.3f}')
        axes[i].set_aspect('equal')

    plt.suptitle('FORWARD PROCESS: Progressively Adding Noise\n'
                 'x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_reverse_process(trajectory):
    """Visualize how samples evolve during denoising."""
    n_steps = len(trajectory)
    n_show = min(10, n_steps)
    indices = np.linspace(0, n_steps-1, n_show).astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        x = trajectory[idx]
        axes[i].scatter(x[:, 0], x[:, 1], s=1, alpha=0.5)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        if idx == 0:
            axes[i].set_title('Pure Noise (x_T)')
        elif idx == n_steps - 1:
            axes[i].set_title('Final Sample (x_0)')
        else:
            axes[i].set_title(f'Step {idx}')
        axes[i].set_aspect('equal')

    plt.suptitle('REVERSE PROCESS: Iterative Denoising\n'
                 'x_T → x_{T-1} → ... → x_0', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_noise_schedules(T=1000):
    """Compare different noise schedules."""
    betas_linear = linear_schedule(T)
    betas_cosine = cosine_schedule(T)

    params_linear = get_schedule_params(betas_linear)
    params_cosine = get_schedule_params(betas_cosine)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # β_t
    axes[0].plot(betas_linear, label='Linear', linewidth=2)
    axes[0].plot(betas_cosine, label='Cosine', linewidth=2)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('β_t')
    axes[0].set_title('Noise Added per Step (β_t)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ᾱ_t
    axes[1].plot(params_linear['alphas_cumprod'], label='Linear', linewidth=2)
    axes[1].plot(params_cosine['alphas_cumprod'], label='Cosine', linewidth=2)
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('ᾱ_t')
    axes[1].set_title('Signal Remaining (ᾱ_t)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SNR
    snr_linear = params_linear['alphas_cumprod'] / (1 - params_linear['alphas_cumprod'] + 1e-10)
    snr_cosine = params_cosine['alphas_cumprod'] / (1 - params_cosine['alphas_cumprod'] + 1e-10)
    axes[2].semilogy(snr_linear, label='Linear', linewidth=2)
    axes[2].semilogy(snr_cosine, label='Cosine', linewidth=2)
    axes[2].set_xlabel('Timestep t')
    axes[2].set_ylabel('SNR (log scale)')
    axes[2].set_title('Signal-to-Noise Ratio')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('NOISE SCHEDULES: Linear vs Cosine\n'
                 'Cosine preserves signal longer, better for images', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_training_and_samples(data, samples, losses, dataset_name):
    """Compare original data with generated samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, c='blue')
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].set_title(f'Original Data ({dataset_name})')
    axes[0].set_aspect('equal')

    # Generated samples
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_title('Generated Samples')
    axes[1].set_aspect('equal')

    # Training loss
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE Loss')
    axes[2].set_title('Training Loss')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('DIFFUSION MODEL TRAINING RESULTS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_num_timesteps():
    """
    Ablation: How does the number of diffusion steps affect quality?

    Hypothesis:
    - More steps = better quality (finer denoising)
    - But also slower sampling
    - Sweet spot usually T=100-1000
    """
    print("\n" + "="*60)
    print("ABLATION: Number of Diffusion Steps (T)")
    print("="*60)

    np.random.seed(42)
    data = make_moons(1000)

    timestep_options = [10, 50, 100, 500]
    results = {}

    fig, axes = plt.subplots(2, len(timestep_options), figsize=(16, 8))

    for i, T in enumerate(timestep_options):
        print(f"\nT = {T}:")

        betas = cosine_schedule(T)
        params = get_schedule_params(betas)

        model = DenoisingMLP(data_dim=2, hidden_dim=128)
        losses = train_diffusion(model, data, params, n_epochs=500, batch_size=128, lr=1e-3)

        samples, trajectory = sample_ddpm(model, params, n_samples=500)

        results[T] = {
            'final_loss': losses[-1],
            'samples': samples
        }

        # Plot samples
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
        axes[0, i].set_xlim(-2, 2)
        axes[0, i].set_ylim(-2, 2)
        axes[0, i].set_title(f'T = {T}')
        axes[0, i].set_aspect('equal')

        # Plot loss
        axes[1, i].plot(losses)
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].set_title(f'Final Loss: {losses[-1]:.4f}')
        axes[1, i].grid(True, alpha=0.3)

        print(f"   Final loss: {losses[-1]:.4f}")

    # Add original data for reference
    axes[0, 0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, c='green', label='Original')
    axes[0, 0].legend()

    plt.suptitle('ABLATION: Effect of Number of Diffusion Steps (T)\n'
                 'More steps = finer denoising, better quality', fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig, results


def ablation_schedule_comparison():
    """
    Ablation: Linear vs Cosine noise schedule.

    Hypothesis:
    - Cosine preserves signal longer
    - Better for learning fine details
    - Linear may destroy information too fast
    """
    print("\n" + "="*60)
    print("ABLATION: Linear vs Cosine Noise Schedule")
    print("="*60)

    np.random.seed(42)
    data = make_gaussian_mixture(1000)
    T = 200

    schedules = {
        'Linear': linear_schedule(T),
        'Cosine': cosine_schedule(T)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, (name, betas) in enumerate(schedules.items()):
        print(f"\n{name} schedule:")

        params = get_schedule_params(betas)
        model = DenoisingMLP(data_dim=2, hidden_dim=128)
        losses = train_diffusion(model, data, params, n_epochs=500, batch_size=128, lr=1e-3)

        samples, _ = sample_ddpm(model, params, n_samples=500)

        # Plot samples
        axes[0, i].scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, c='green', label='Original')
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='blue', label='Generated')
        axes[0, i].set_xlim(-2, 2)
        axes[0, i].set_ylim(-2, 2)
        axes[0, i].set_title(f'{name} Schedule - Samples')
        axes[0, i].legend()
        axes[0, i].set_aspect('equal')

        # Plot loss
        axes[1, i].plot(losses)
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].set_title(f'{name} - Loss (final: {losses[-1]:.4f})')
        axes[1, i].grid(True, alpha=0.3)

        print(f"   Final loss: {losses[-1]:.4f}")

    plt.suptitle('ABLATION: Linear vs Cosine Noise Schedule\n'
                 'Cosine typically works better for complex data', fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def ablation_network_size():
    """
    Ablation: Effect of network capacity.

    Hypothesis:
    - Larger network = better denoising
    - But also more prone to overfitting
    - Need to balance capacity with data complexity
    """
    print("\n" + "="*60)
    print("ABLATION: Network Capacity")
    print("="*60)

    np.random.seed(42)
    data = make_swiss_roll(1000)
    T = 200

    betas = cosine_schedule(T)
    params = get_schedule_params(betas)

    hidden_sizes = [64, 128, 256, 512]

    fig, axes = plt.subplots(2, len(hidden_sizes), figsize=(16, 8))

    for i, hidden_dim in enumerate(hidden_sizes):
        print(f"\nHidden dim = {hidden_dim}:")

        model = DenoisingMLP(data_dim=2, hidden_dim=hidden_dim)
        losses = train_diffusion(model, data, params, n_epochs=500, batch_size=128, lr=1e-3)

        samples, _ = sample_ddpm(model, params, n_samples=500)

        # Plot samples
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
        axes[0, i].set_xlim(-2, 2)
        axes[0, i].set_ylim(-2, 2)
        axes[0, i].set_title(f'Hidden = {hidden_dim}')
        axes[0, i].set_aspect('equal')

        # Plot loss
        axes[1, i].plot(losses)
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].set_title(f'Final Loss: {losses[-1]:.4f}')
        axes[1, i].grid(True, alpha=0.3)

        print(f"   Final loss: {losses[-1]:.4f}")

    axes[0, 0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, c='green', label='Original')
    axes[0, 0].legend()

    plt.suptitle('ABLATION: Effect of Network Capacity\n'
                 'Larger networks can model more complex noise patterns', fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("""
======================================================================
DIFFUSION FUNDAMENTALS — Paradigm: ITERATIVE DENOISING
======================================================================

THE CORE IDEA:
    1. DESTROY data by adding noise step by step
    2. LEARN to reverse each step
    3. GENERATE by starting from noise and denoising

This file demonstrates:
    • Forward process (adding noise)
    • Reverse process (removing noise)
    • Training a denoising network
    • Different noise schedules
    • Ablation experiments
    """)

    # Set random seed for reproducibility
    np.random.seed(42)

    # ========================================
    # VISUALIZE NOISE SCHEDULES
    # ========================================
    print("\n" + "="*60)
    print("VISUALIZING NOISE SCHEDULES")
    print("="*60)

    fig_schedules = visualize_noise_schedules(T=1000)
    fig_schedules.savefig('45_diffusion_schedules.png', dpi=150, bbox_inches='tight')
    print("Saved: 45_diffusion_schedules.png")

    # ========================================
    # TRAIN ON DIFFERENT DATASETS
    # ========================================
    datasets = {
        'Moons': make_moons(1000),
        'Circles': make_circles(1000),
        'Swiss Roll': make_swiss_roll(1000),
        'Gaussian Mixture': make_gaussian_mixture(1000)
    }

    T = 200
    betas = cosine_schedule(T)
    params = get_schedule_params(betas)

    for name, data in datasets.items():
        print(f"\n" + "="*60)
        print(f"TRAINING ON: {name}")
        print("="*60)

        # Visualize forward process
        fig_forward = visualize_forward_process(data, params)
        fig_forward.savefig(f'45_diffusion_forward_{name.lower().replace(" ", "_")}.png',
                           dpi=150, bbox_inches='tight')
        plt.close(fig_forward)

        # Train model (256 hidden units, 2000 epochs for good quality)
        model = DenoisingMLP(data_dim=2, hidden_dim=256)
        losses = train_diffusion(model, data, params, n_epochs=2000, batch_size=128, lr=1e-3)

        # Generate samples
        samples, trajectory = sample_ddpm(model, params, n_samples=500)

        # Visualize reverse process
        fig_reverse = visualize_reverse_process(trajectory)
        fig_reverse.savefig(f'45_diffusion_reverse_{name.lower().replace(" ", "_")}.png',
                           dpi=150, bbox_inches='tight')
        plt.close(fig_reverse)

        # Visualize results
        fig_results = visualize_training_and_samples(data, samples, losses, name)
        fig_results.savefig(f'45_diffusion_results_{name.lower().replace(" ", "_")}.png',
                           dpi=150, bbox_inches='tight')
        plt.close(fig_results)

        print(f"Saved visualizations for {name}")

    # ========================================
    # ABLATION EXPERIMENTS
    # ========================================
    print("\n" + "="*60)
    print("RUNNING ABLATION EXPERIMENTS")
    print("="*60)

    fig_ablation_T, _ = ablation_num_timesteps()
    fig_ablation_T.savefig('45_diffusion_ablation_timesteps.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ablation_T)
    print("Saved: 45_diffusion_ablation_timesteps.png")

    fig_ablation_schedule = ablation_schedule_comparison()
    fig_ablation_schedule.savefig('45_diffusion_ablation_schedule.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ablation_schedule)
    print("Saved: 45_diffusion_ablation_schedule.png")

    fig_ablation_network = ablation_network_size()
    fig_ablation_network.savefig('45_diffusion_ablation_network.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ablation_network)
    print("Saved: 45_diffusion_ablation_network.png")

    # ========================================
    # SUMMARY
    # ========================================
    print("""
======================================================================
SUMMARY
======================================================================

KEY INSIGHTS FROM DIFFUSION FUNDAMENTALS:

1. FORWARD PROCESS IS FIXED
   - Just add Gaussian noise according to schedule
   - Can jump to any timestep directly: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

2. REVERSE PROCESS IS LEARNED
   - Neural network predicts the noise to remove
   - Single network handles ALL timesteps (t is an input)

3. NOISE SCHEDULE MATTERS
   - Cosine > Linear for most applications
   - Controls how fast information is destroyed

4. MORE STEPS = BETTER QUALITY
   - But also slower sampling
   - Tradeoff between quality and speed

5. TRAINING IS SIMPLE
   - Just predict the noise that was added
   - MSE loss: ||ε - ε_θ(x_t, t)||²

NEXT: 46_ddpm.py — Full DDPM with all the details
      47_ddim.py — Fast sampling without sacrificing quality

======================================================================
    """)
