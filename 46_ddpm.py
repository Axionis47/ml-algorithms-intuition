"""
DENOISING DIFFUSION PROBABILISTIC MODELS (DDPM) — Paradigm: PROBABILISTIC DENOISING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

DDPM (Ho et al., 2020) is THE foundational diffusion paper.

The key insight: Frame diffusion as a VARIATIONAL problem.
    - Forward: q(x_t | x_0) has CLOSED FORM
    - Reverse: p_θ(x_{t-1} | x_t) is learned
    - Training: Maximize ELBO ≈ Predict noise

FORWARD PROCESS (closed form, no iteration needed!):
    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε,  where ε ~ N(0,I)

    This lets us jump DIRECTLY to any timestep!
    No need to iterate through t-1 steps.

REVERSE PROCESS (learned):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    The network predicts μ_θ, but we reparameterize to predict ε.

===============================================================
THE MATHEMATICS
===============================================================

VARIANCE SCHEDULE:
    β_t ∈ (0, 1) controls noise added at step t
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^t α_s  (cumulative product)

FORWARD PROCESS:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

    CLOSED FORM (the magic!):
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

    So: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

REVERSE PROCESS:
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    POSTERIOR (true reverse, if we knew x_0):
    q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t(x_t, x_0), β̃_t I)

    where:
    μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
    β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) × β_t

TRAINING OBJECTIVE (simplified):
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t, t)||²]

    "Just predict the noise that was added!"

SAMPLING:
    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t)) + σ_t z

===============================================================
INDUCTIVE BIAS
===============================================================

1. GAUSSIAN TRANSITIONS: Both forward and reverse are Gaussian
   - Makes math tractable
   - May not be optimal for all distributions

2. FIXED VARIANCE: σ_t² is fixed (not learned in basic DDPM)
   - Simplifies training
   - Improved DDPM learns variance too

3. MARKOV CHAIN: Each step only depends on previous
   - Enables simple training
   - Requires many steps (T=1000 typical)

4. NOISE PREDICTION: Network predicts ε, not x_0 or score
   - Empirically works best
   - Equivalent to score matching up to scaling

5. SHARED WEIGHTS: Same network for all timesteps
   - t is an input (via embedding)
   - Efficient but may limit per-timestep expressivity

WHEN DDPM EXCELS:
- High-quality generation (state-of-the-art for images)
- Diverse outputs (mode coverage is excellent)
- Training is stable (unlike GANs)

WHEN DDPM STRUGGLES:
- Slow sampling (need 1000 steps)
- Discrete data (text needs modifications)
- Conditional generation (need guidance)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

# Import from fundamentals
from importlib import import_module
diffusion_fundamentals = import_module('45_diffusion_fundamentals')

# Import what we need
cosine_schedule = diffusion_fundamentals.cosine_schedule
linear_schedule = diffusion_fundamentals.linear_schedule
get_schedule_params = diffusion_fundamentals.get_schedule_params
forward_diffusion_sample = diffusion_fundamentals.forward_diffusion_sample
DenoisingMLP = diffusion_fundamentals.DenoisingMLP
make_moons = diffusion_fundamentals.make_moons
make_gaussian_mixture = diffusion_fundamentals.make_gaussian_mixture
make_swiss_roll = diffusion_fundamentals.make_swiss_roll


# ============================================================
# DDPM TRAINER
# ============================================================

class DDPMTrainer:
    """
    Full DDPM training and sampling implementation.

    This class encapsulates the complete DDPM algorithm:
    - Training with noise prediction objective
    - Sampling with proper posterior mean and variance
    - Support for different variance schedules
    """

    def __init__(self, model, T=1000, schedule='cosine', beta_start=1e-4, beta_end=0.02):
        """
        Initialize DDPM trainer.

        Args:
            model: Denoising network (predicts noise given x_t and t)
            T: Number of diffusion steps
            schedule: 'linear' or 'cosine'
            beta_start, beta_end: For linear schedule
        """
        self.model = model
        self.T = T

        # Setup schedule
        if schedule == 'linear':
            betas = linear_schedule(T, beta_start, beta_end)
        else:
            betas = cosine_schedule(T)

        self.params = get_schedule_params(betas)

        # Pre-compute coefficients for sampling
        self._precompute_sampling_coefficients()

    def _precompute_sampling_coefficients(self):
        """Pre-compute coefficients used in sampling."""
        alphas = self.params['alphas']
        alphas_cumprod = self.params['alphas_cumprod']
        alphas_cumprod_prev = self.params['alphas_cumprod_prev']
        betas = self.params['betas']

        # Coefficient for x_t in mean computation
        # (1/√α_t)
        self.sqrt_recip_alphas = 1.0 / np.sqrt(alphas)

        # Coefficient for noise prediction in mean computation
        # β_t / √(1 - ᾱ_t)
        self.noise_coeff = betas / np.sqrt(1.0 - alphas_cumprod)

        # Posterior variance: β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) × β_t
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # Clamp to avoid numerical issues at t=0
        self.posterior_variance = np.clip(self.posterior_variance, 1e-20, None)
        self.posterior_log_variance = np.log(self.posterior_variance)

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward process.

        x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

        This is the CLOSED FORM - we can jump to any t directly!
        """
        return forward_diffusion_sample(x_0, t, self.params, noise)

    def p_mean_variance(self, x_t, t):
        """
        Compute mean and variance of p_θ(x_{t-1} | x_t).

        Mean: μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t))
        Variance: σ_t² = β̃_t (posterior variance)
        """
        # Predict noise
        noise_pred = self.model.forward(x_t, t)

        # Get coefficients for this timestep
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        noise_coeff = self.noise_coeff[t]

        # Reshape for broadcasting
        if sqrt_recip_alpha.ndim == 0:
            sqrt_recip_alpha = np.full(len(x_t), sqrt_recip_alpha)
            noise_coeff = np.full(len(x_t), self.noise_coeff[t[0]] if hasattr(t, '__len__') else self.noise_coeff[t])

        sqrt_recip_alpha = sqrt_recip_alpha[:, np.newaxis]
        noise_coeff = noise_coeff[:, np.newaxis]

        # Compute mean
        mean = sqrt_recip_alpha * (x_t - noise_coeff * noise_pred)

        # Get variance
        if hasattr(t, '__len__'):
            variance = self.posterior_variance[t][:, np.newaxis]
            log_variance = self.posterior_log_variance[t][:, np.newaxis]
        else:
            variance = self.posterior_variance[t]
            log_variance = self.posterior_log_variance[t]

        return mean, variance, log_variance

    def p_sample(self, x_t, t):
        """
        Sample from p_θ(x_{t-1} | x_t) - one step of reverse process.

        x_{t-1} = μ_θ + σ_t × z, where z ~ N(0, I)
        """
        mean, variance, _ = self.p_mean_variance(x_t, t)

        # Sample noise (zero for t=0)
        noise = np.random.randn(*x_t.shape)

        # Handle t=0 case (no noise)
        if hasattr(t, '__len__'):
            mask = (t > 0).astype(float)[:, np.newaxis]
        else:
            mask = 1.0 if t > 0 else 0.0

        return mean + mask * np.sqrt(variance) * noise

    def p_sample_loop(self, shape, return_trajectory=False):
        """
        Full sampling loop: x_T → x_{T-1} → ... → x_0

        Args:
            shape: Shape of samples to generate (n_samples, dim)
            return_trajectory: If True, return all intermediate states

        Returns:
            Final samples x_0 (and trajectory if requested)
        """
        # Start from pure noise
        x = np.random.randn(*shape)

        trajectory = [x.copy()] if return_trajectory else None

        # Reverse process
        for t in reversed(range(self.T)):
            t_batch = np.full(shape[0], t)
            x = self.p_sample(x, t_batch)

            if return_trajectory and (t % (self.T // 20) == 0 or t < 10):
                trajectory.append(x.copy())

        if return_trajectory:
            return x, trajectory
        return x

    def train_step(self, x_0):
        """
        One training step of DDPM.

        THE ALGORITHM:
        1. Sample t ~ Uniform(0, T-1)
        2. Sample ε ~ N(0, I)
        3. Compute x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        4. Predict ε̂ = model(x_t, t)
        5. Loss = ||ε - ε̂||²

        Returns:
            loss: MSE loss value
        """
        batch_size = len(x_0)

        # Sample random timesteps
        t = np.random.randint(0, self.T, size=batch_size)

        # Sample noise
        noise = np.random.randn(*x_0.shape)

        # Forward diffusion
        x_t, _ = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.model.forward(x_t, t)

        # MSE loss
        loss = np.mean((noise - noise_pred) ** 2)

        # Backward pass
        d_loss = 2 * (noise_pred - noise) / noise_pred.size
        self.model.backward(d_loss)

        return loss

    def train(self, data, n_epochs=1000, batch_size=128, lr=1e-3, verbose=True):
        """
        Train the DDPM model.

        Args:
            data: Training data, shape (n_samples, dim)
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            losses: List of training losses
        """
        n_samples = len(data)
        losses = []

        for epoch in range(n_epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                x_0 = data[idx]

                loss = self.train_step(x_0)
                epoch_loss += loss
                n_batches += 1

                # Update model
                self.model.update(lr=lr)

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

        return losses


# ============================================================
# DDPM WITH LEARNED VARIANCE (Improved DDPM)
# ============================================================

class DDPMLearnedVariance(DDPMTrainer):
    """
    DDPM with learned variance (Nichol & Dhariwal, 2021).

    Instead of fixed variance σ_t², learn to interpolate between
    β_t and β̃_t. This improves sample quality, especially with
    fewer sampling steps.

    The model outputs:
    - ε_θ(x_t, t): noise prediction (as before)
    - v_θ(x_t, t): variance interpolation coefficient

    Variance: σ_t² = exp(v log β̃_t + (1-v) log β_t)
    """

    def __init__(self, model, T=1000, schedule='cosine'):
        super().__init__(model, T, schedule)

        # Pre-compute log betas for variance interpolation
        self.log_betas = np.log(np.clip(self.params['betas'], 1e-20, None))

    # Note: Full implementation would require modifying the model
    # to output both noise and variance predictions.
    # This is left as a demonstration of the concept.


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_ddpm_training(data, samples, losses, title="DDPM Training"):
    """Visualize DDPM training results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='blue')
    axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0].set_aspect('equal')

    axes[1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c='red')
    axes[1].set_title('DDPM Samples', fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')

    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title(f'Training Loss (final: {losses[-1]:.4f})', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_ddpm_sampling_process(trajectory, n_show=10):
    """Visualize the DDPM sampling (reverse) process."""
    n_steps = len(trajectory)
    indices = np.linspace(0, n_steps-1, min(n_show, n_steps)).astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        x = trajectory[idx]
        axes[i].scatter(x[:, 0], x[:, 1], s=1, alpha=0.5)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)

        if idx == 0:
            axes[i].set_title('x_T (noise)', fontsize=10)
        elif idx == n_steps - 1:
            axes[i].set_title('x_0 (sample)', fontsize=10)
        else:
            progress = 100 * (n_steps - 1 - idx) / (n_steps - 1)
            axes[i].set_title(f'{progress:.0f}% denoised', fontsize=10)
        axes[i].set_aspect('equal')

    plt.suptitle('DDPM SAMPLING: Iterative Denoising\n'
                 'x_T (pure noise) → x_0 (clean sample)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_schedule_effect(data, T_values=[50, 200, 1000]):
    """Ablation: Effect of number of diffusion steps."""
    fig, axes = plt.subplots(1, len(T_values) + 1, figsize=(4*(len(T_values)+1), 4))

    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='green')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].set_aspect('equal')

    for i, T in enumerate(T_values):
        print(f"\nTraining with T={T}...")

        model = DenoisingMLP(data_dim=2, hidden_dim=128)
        trainer = DDPMTrainer(model, T=T, schedule='cosine')
        losses = trainer.train(data, n_epochs=800, batch_size=128, lr=1e-3, verbose=False)

        samples = trainer.p_sample_loop((300, 2))

        axes[i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        axes[i+1].set_title(f'T = {T}\nLoss: {losses[-1]:.4f}', fontsize=11)
        axes[i+1].set_aspect('equal')

    plt.suptitle('ABLATION: Effect of Diffusion Steps (T)\n'
                 'More steps → better quality, but slower sampling',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_schedule_type_effect(data):
    """Ablation: Linear vs Cosine schedule."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='green')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].set_aspect('equal')

    T = 200
    schedules = ['linear', 'cosine']

    for i, schedule in enumerate(schedules):
        print(f"\nTraining with {schedule} schedule...")

        model = DenoisingMLP(data_dim=2, hidden_dim=128)
        trainer = DDPMTrainer(model, T=T, schedule=schedule)
        losses = trainer.train(data, n_epochs=1000, batch_size=128, lr=1e-3, verbose=False)

        samples = trainer.p_sample_loop((300, 2))

        axes[i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        axes[i+1].set_title(f'{schedule.capitalize()} Schedule\nLoss: {losses[-1]:.4f}',
                          fontsize=11)
        axes[i+1].set_aspect('equal')

    plt.suptitle('ABLATION: Linear vs Cosine Noise Schedule',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("""
======================================================================
DENOISING DIFFUSION PROBABILISTIC MODELS (DDPM)
Paradigm: PROBABILISTIC DENOISING
======================================================================

DDPM is the foundational paper for modern diffusion models.

KEY CONTRIBUTIONS:
1. Closed-form forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
2. Simple training objective: predict the noise
3. Principled reverse process with proper variance

This file demonstrates:
    • Full DDPM training algorithm
    • Proper sampling with posterior variance
    • Ablations on timesteps and schedules
    """)

    np.random.seed(42)

    # ========================================
    # TRAIN DDPM ON GAUSSIAN MIXTURE
    # ========================================
    print("\n" + "="*60)
    print("TRAINING DDPM ON GAUSSIAN MIXTURE")
    print("="*60)

    data = make_gaussian_mixture(1000, n_components=8)

    model = DenoisingMLP(data_dim=2, hidden_dim=256)
    trainer = DDPMTrainer(model, T=200, schedule='cosine')

    print("\nTraining...")
    losses = trainer.train(data, n_epochs=1500, batch_size=128, lr=1e-3)

    print("\nSampling...")
    samples, trajectory = trainer.p_sample_loop((500, 2), return_trajectory=True)

    # Visualize
    fig = visualize_ddpm_training(data, samples, losses, "DDPM: Gaussian Mixture")
    fig.savefig('46_ddpm_gaussian_mixture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 46_ddpm_gaussian_mixture.png")

    fig = visualize_ddpm_sampling_process(trajectory)
    fig.savefig('46_ddpm_sampling_process.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 46_ddpm_sampling_process.png")

    # ========================================
    # TRAIN DDPM ON MOONS
    # ========================================
    print("\n" + "="*60)
    print("TRAINING DDPM ON MOONS")
    print("="*60)

    data_moons = make_moons(1000)

    model_moons = DenoisingMLP(data_dim=2, hidden_dim=256)
    trainer_moons = DDPMTrainer(model_moons, T=200, schedule='cosine')

    print("\nTraining...")
    losses_moons = trainer_moons.train(data_moons, n_epochs=2000, batch_size=128, lr=1e-3)

    print("\nSampling...")
    samples_moons = trainer_moons.p_sample_loop((500, 2))

    fig = visualize_ddpm_training(data_moons, samples_moons, losses_moons, "DDPM: Moons")
    fig.savefig('46_ddpm_moons.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 46_ddpm_moons.png")

    # ========================================
    # ABLATION: NUMBER OF TIMESTEPS
    # ========================================
    print("\n" + "="*60)
    print("ABLATION: Number of Diffusion Steps")
    print("="*60)

    data_ablation = make_moons(500)
    fig = visualize_schedule_effect(data_ablation, T_values=[50, 100, 200])
    fig.savefig('46_ddpm_ablation_timesteps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 46_ddpm_ablation_timesteps.png")

    # ========================================
    # ABLATION: SCHEDULE TYPE
    # ========================================
    print("\n" + "="*60)
    print("ABLATION: Linear vs Cosine Schedule")
    print("="*60)

    fig = visualize_schedule_type_effect(data_ablation)
    fig.savefig('46_ddpm_ablation_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 46_ddpm_ablation_schedule.png")

    # ========================================
    # SUMMARY
    # ========================================
    print("""
======================================================================
SUMMARY: DDPM
======================================================================

KEY INSIGHTS:

1. CLOSED-FORM FORWARD PROCESS
   x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
   → Can sample any timestep directly!

2. SIMPLE TRAINING OBJECTIVE
   L = E[||ε - ε_θ(x_t, t)||²]
   → Just predict the noise that was added

3. PROPER REVERSE PROCESS
   μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ)
   → Derived from posterior q(x_{t-1}|x_t,x_0)

4. VARIANCE SCHEDULE MATTERS
   Cosine > Linear for most tasks
   → Preserves signal longer

5. MORE STEPS = BETTER QUALITY
   T=1000 typical for high quality
   → But sampling is slow!

NEXT: 47_ddim.py — Fast sampling (50 steps instead of 1000!)

======================================================================
    """)
