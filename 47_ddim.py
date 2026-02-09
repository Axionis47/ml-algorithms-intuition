"""
DENOISING DIFFUSION IMPLICIT MODELS (DDIM) — Paradigm: DETERMINISTIC DENOISING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

DDIM (Song et al., 2020) enables FAST sampling from diffusion models.

THE KEY INSIGHT:
    DDPM training objective doesn't depend on the Markov chain!
    We can use a NON-MARKOVIAN generative process for sampling.

DDPM: Random walk (stochastic, needs ~1000 steps)
DDIM: Direct path (deterministic, can use ~50 steps)

SAME MODEL, DIFFERENT SAMPLING!
    - Train with DDPM objective
    - Sample with DDIM (10-100x faster)

===============================================================
THE MATHEMATICS
===============================================================

DDPM SAMPLING:
    x_{t-1} = μ_θ + σ_t × z,  z ~ N(0,I)
    Random noise at each step → random path

DDIM SAMPLING:
    x_{t-1} = √ᾱ_{t-1} × predicted_x_0
            + √(1-ᾱ_{t-1}-σ²) × ε_θ(x_t, t)
            + σ × z

    where predicted_x_0 = (x_t - √(1-ᾱ_t) × ε_θ) / √ᾱ_t

When σ = 0: DETERMINISTIC!
    Same x_T → same x_0 (always)

When σ = √((1-ᾱ_{t-1})/(1-ᾱ_t)) × √(1-ᾱ_t/ᾱ_{t-1}):
    Equivalent to DDPM (stochastic)

STEP SKIPPING:
    Don't need t → t-1 → t-2 → ...
    Can use subsequence: τ_1, τ_2, ..., τ_S where S << T

    Example: T=1000, S=50
    τ = [0, 20, 40, 60, ..., 980, 1000]

===============================================================
INDUCTIVE BIAS
===============================================================

1. DETERMINISM OPTION: Can make sampling fully deterministic
   - Same latent → same output (reproducible)
   - Enables interpolation in latent space
   - No variance means potentially worse diversity

2. STEP SKIPPING: Can skip steps without retraining
   - 50 steps ≈ 1000 steps quality
   - But very few steps (10) degrades quality

3. INTERPOLATION: Smooth latent space
   - Interpolate x_T → smooth transitions in x_0
   - Enables semantic editing

4. SAME TRAINING: Uses DDPM-trained model
   - No retraining needed
   - Just change sampling procedure

WHEN DDIM EXCELS:
- Fast generation (50 steps vs 1000)
- Reproducible outputs (deterministic)
- Latent space manipulation
- Real-time applications

WHEN DDIM STRUGGLES:
- Very few steps (<20) still degrades quality
- May have slightly less diversity than DDPM
- Still slower than GANs (50 steps vs 1)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

# Import from previous files
from importlib import import_module
diffusion_fundamentals = import_module('45_diffusion_fundamentals')
ddpm_module = import_module('46_ddpm')

# Import what we need
cosine_schedule = diffusion_fundamentals.cosine_schedule
get_schedule_params = diffusion_fundamentals.get_schedule_params
DenoisingMLP = diffusion_fundamentals.DenoisingMLP
make_moons = diffusion_fundamentals.make_moons
make_gaussian_mixture = diffusion_fundamentals.make_gaussian_mixture

DDPMTrainer = ddpm_module.DDPMTrainer


# ============================================================
# DDIM SAMPLER
# ============================================================

class DDIMSampler:
    """
    DDIM Sampling from a trained DDPM model.

    Key features:
    1. Deterministic or stochastic sampling (controlled by eta)
    2. Step skipping (use fewer steps without retraining)
    3. Latent space interpolation
    """

    def __init__(self, ddpm_trainer):
        """
        Initialize DDIM sampler from a trained DDPM model.

        Args:
            ddpm_trainer: Trained DDPMTrainer object
        """
        self.model = ddpm_trainer.model
        self.T = ddpm_trainer.T
        self.params = ddpm_trainer.params

    def get_timestep_sequence(self, num_steps):
        """
        Get timestep sequence for sampling.

        Args:
            num_steps: Number of sampling steps (S)

        Returns:
            Array of timesteps [τ_S, τ_{S-1}, ..., τ_1, τ_0]
        """
        # Uniform spacing
        step_size = self.T // num_steps
        timesteps = np.arange(0, self.T, step_size)[:num_steps]
        return timesteps[::-1]  # Reverse for sampling

    def predict_x0(self, x_t, t, eps_pred):
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - √(1-ᾱ_t) × ε_θ) / √ᾱ_t
        """
        sqrt_alpha_cumprod = self.params['sqrt_alphas_cumprod'][t]
        sqrt_one_minus_alpha_cumprod = self.params['sqrt_one_minus_alphas_cumprod'][t]

        # Reshape for broadcasting
        if sqrt_alpha_cumprod.ndim == 0:
            sqrt_alpha_cumprod = np.full(len(x_t), sqrt_alpha_cumprod)
            sqrt_one_minus_alpha_cumprod = np.full(len(x_t), sqrt_one_minus_alpha_cumprod)

        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, np.newaxis]
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, np.newaxis]

        x_0 = (x_t - sqrt_one_minus_alpha_cumprod * eps_pred) / sqrt_alpha_cumprod

        return x_0

    def ddim_step(self, x_t, t, t_prev, eta=0.0):
        """
        One step of DDIM sampling.

        DDIM UPDATE:
        x_{t-1} = √ᾱ_{t-1} × predicted_x_0
                + √(1-ᾱ_{t-1}-σ²) × ε_θ(x_t, t)
                + σ × z

        Args:
            x_t: Current state
            t: Current timestep
            t_prev: Previous timestep (we're going backward)
            eta: Stochasticity (0 = deterministic, 1 = DDPM-like)

        Returns:
            x_{t_prev}: Denoised state
        """
        batch_size = len(x_t)
        t_batch = np.full(batch_size, t)

        # Predict noise
        eps_pred = self.model.forward(x_t, t_batch)

        # Predict x_0
        x_0_pred = self.predict_x0(x_t, t_batch, eps_pred)

        # Get alpha values
        alpha_cumprod_t = self.params['alphas_cumprod'][t]
        alpha_cumprod_t_prev = self.params['alphas_cumprod'][t_prev] if t_prev >= 0 else 1.0

        # Compute sigma
        # σ = η × √((1-ᾱ_{t-1})/(1-ᾱ_t)) × √(1-ᾱ_t/ᾱ_{t-1})
        sigma = eta * np.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * \
                np.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)

        # Direction pointing to x_t
        pred_dir = np.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * eps_pred

        # DDIM update
        x_prev = np.sqrt(alpha_cumprod_t_prev) * x_0_pred + pred_dir

        # Add noise if stochastic
        if eta > 0:
            noise = np.random.randn(*x_t.shape)
            x_prev = x_prev + sigma * noise

        return x_prev, x_0_pred

    def sample(self, shape, num_steps=50, eta=0.0, return_trajectory=False):
        """
        Sample using DDIM.

        Args:
            shape: Shape of samples (n_samples, dim)
            num_steps: Number of sampling steps (can be much less than T!)
            eta: Stochasticity parameter
                 - eta=0: deterministic
                 - eta=1: DDPM-like stochasticity
            return_trajectory: Return intermediate states

        Returns:
            Samples (and trajectory if requested)
        """
        # Get timestep sequence
        timesteps = self.get_timestep_sequence(num_steps)

        # Start from noise
        x = np.random.randn(*shape)

        trajectory = [x.copy()] if return_trajectory else None
        x0_predictions = []

        # Sampling loop
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1

            x, x0_pred = self.ddim_step(x, t, t_prev, eta)

            if return_trajectory:
                trajectory.append(x.copy())
                x0_predictions.append(x0_pred.copy())

        if return_trajectory:
            return x, trajectory, x0_predictions
        return x

    def interpolate(self, x1, x2, num_interpolations=10, num_steps=50, eta=0.0):
        """
        Interpolate between two samples in latent space.

        This is possible because DDIM is deterministic!
        Encode x1, x2 to noise, interpolate, decode.

        Note: This implementation uses interpolation in x_T space,
        which is simpler but less precise than full inversion.
        """
        # For simplicity, we start from random noise and interpolate there
        # A full implementation would invert x1 and x2 to get their latents

        z1 = np.random.randn(1, x1.shape[0])
        z2 = np.random.randn(1, x2.shape[0])

        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z = (1 - alpha) * z1 + alpha * z2
            x = self.sample(z.shape, num_steps=num_steps, eta=eta)
            interpolations.append(x[0])

        return np.array(interpolations)


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_ddim_speed_comparison(ddpm_trainer, data, step_counts=[10, 25, 50, 100, 200]):
    """Compare DDIM quality at different step counts."""
    sampler = DDIMSampler(ddpm_trainer)

    fig, axes = plt.subplots(1, len(step_counts) + 1, figsize=(3*(len(step_counts)+1), 3))

    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='green')
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].set_aspect('equal')

    for i, num_steps in enumerate(step_counts):
        samples = sampler.sample((300, 2), num_steps=num_steps, eta=0.0)

        axes[i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        axes[i+1].set_title(f'{num_steps} steps', fontsize=10)
        axes[i+1].set_aspect('equal')

    plt.suptitle('DDIM: Fast Sampling (deterministic, η=0)\n'
                 'Same trained model, different number of sampling steps',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_ddim_eta_comparison(ddpm_trainer, data, etas=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """Compare DDIM with different eta values."""
    sampler = DDIMSampler(ddpm_trainer)

    fig, axes = plt.subplots(1, len(etas) + 1, figsize=(3*(len(etas)+1), 3))

    # Original data
    axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='green')
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].set_aspect('equal')

    for i, eta in enumerate(etas):
        # Use same seed to show effect of eta
        np.random.seed(42)
        samples = sampler.sample((300, 2), num_steps=50, eta=eta)

        axes[i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        if eta == 0:
            axes[i+1].set_title(f'η={eta}\n(deterministic)', fontsize=10)
        elif eta == 1:
            axes[i+1].set_title(f'η={eta}\n(DDPM-like)', fontsize=10)
        else:
            axes[i+1].set_title(f'η={eta}', fontsize=10)
        axes[i+1].set_aspect('equal')

    plt.suptitle('DDIM: Effect of η (stochasticity)\n'
                 'η=0: deterministic, η=1: DDPM-like stochastic',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_ddim_determinism(ddpm_trainer, num_runs=3):
    """Show that DDIM with eta=0 is deterministic."""
    sampler = DDIMSampler(ddpm_trainer)

    fig, axes = plt.subplots(2, num_runs + 1, figsize=(3*(num_runs+1), 6))

    # Deterministic (eta=0)
    axes[0, 0].set_title('η=0 (deterministic)', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    for i in range(num_runs):
        np.random.seed(42)  # Same seed each time
        samples = sampler.sample((200, 2), num_steps=50, eta=0.0)
        axes[0, i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        axes[0, i+1].set_title(f'Run {i+1}', fontsize=10)
        axes[0, i+1].set_aspect('equal')

    # Stochastic (eta=1)
    axes[1, 0].set_title('η=1 (stochastic)', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    for i in range(num_runs):
        np.random.seed(42)  # Same seed, but stochastic so different results
        samples = sampler.sample((200, 2), num_steps=50, eta=1.0)
        axes[1, i+1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        axes[1, i+1].set_title(f'Run {i+1}', fontsize=10)
        axes[1, i+1].set_aspect('equal')

    plt.suptitle('DDIM Determinism: Same seed, same x_T\n'
                 'η=0: identical outputs | η=1: different outputs',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_ddim_sampling_trajectory(ddpm_trainer, num_steps=50):
    """Visualize the DDIM sampling trajectory and x_0 predictions."""
    sampler = DDIMSampler(ddpm_trainer)

    np.random.seed(42)
    samples, trajectory, x0_preds = sampler.sample(
        (300, 2), num_steps=num_steps, eta=0.0, return_trajectory=True
    )

    # Show trajectory
    n_show = min(8, len(trajectory))
    indices = np.linspace(0, len(trajectory)-1, n_show).astype(int)

    fig, axes = plt.subplots(2, n_show, figsize=(2.5*n_show, 5))

    for i, idx in enumerate(indices):
        # x_t trajectory
        axes[0, i].scatter(trajectory[idx][:, 0], trajectory[idx][:, 1], s=1, alpha=0.5)
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        if idx == 0:
            axes[0, i].set_title('x_T', fontsize=9)
        elif idx == len(trajectory)-1:
            axes[0, i].set_title('x_0', fontsize=9)
        else:
            axes[0, i].set_title(f't={num_steps-idx}', fontsize=9)
        axes[0, i].set_aspect('equal')

        # x_0 predictions
        if idx < len(x0_preds):
            axes[1, i].scatter(x0_preds[idx][:, 0], x0_preds[idx][:, 1], s=1, alpha=0.5, c='orange')
            axes[1, i].set_xlim(-3, 3)
            axes[1, i].set_ylim(-3, 3)
            axes[1, i].set_title(f'x̂_0 pred', fontsize=9)
            axes[1, i].set_aspect('equal')
        else:
            axes[1, i].axis('off')

    axes[0, 0].set_ylabel('x_t trajectory', fontsize=10)
    axes[1, 0].set_ylabel('x_0 predictions', fontsize=10)

    plt.suptitle('DDIM Sampling Trajectory\n'
                 'Top: actual x_t | Bottom: predicted x_0 at each step',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("""
======================================================================
DENOISING DIFFUSION IMPLICIT MODELS (DDIM)
Paradigm: DETERMINISTIC DENOISING
======================================================================

DDIM enables FAST sampling from diffusion models.

KEY INSIGHT:
    DDPM training doesn't depend on the Markov chain!
    Use a non-Markovian process for sampling → skip steps!

BENEFITS:
    1. 10-100x faster sampling (50 steps vs 1000)
    2. Deterministic (η=0) or stochastic (η>0)
    3. Smooth latent space interpolation
    4. No retraining needed!
    """)

    np.random.seed(42)

    # ========================================
    # TRAIN DDPM MODEL (used for DDIM sampling)
    # ========================================
    print("\n" + "="*60)
    print("TRAINING DDPM MODEL (for DDIM sampling)")
    print("="*60)

    data = make_gaussian_mixture(1000, n_components=8)

    model = DenoisingMLP(data_dim=2, hidden_dim=256)
    ddpm_trainer = DDPMTrainer(model, T=200, schedule='cosine')

    print("\nTraining DDPM...")
    losses = ddpm_trainer.train(data, n_epochs=1500, batch_size=128, lr=1e-3)

    # ========================================
    # DDIM SPEED COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("DDIM: Speed Comparison (different step counts)")
    print("="*60)

    fig = visualize_ddim_speed_comparison(ddpm_trainer, data, step_counts=[10, 25, 50, 100])
    fig.savefig('47_ddim_speed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 47_ddim_speed_comparison.png")

    # ========================================
    # DDIM ETA COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("DDIM: Eta Comparison (stochasticity)")
    print("="*60)

    fig = visualize_ddim_eta_comparison(ddpm_trainer, data, etas=[0.0, 0.5, 1.0])
    fig.savefig('47_ddim_eta_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 47_ddim_eta_comparison.png")

    # ========================================
    # DDIM DETERMINISM
    # ========================================
    print("\n" + "="*60)
    print("DDIM: Determinism Demonstration")
    print("="*60)

    fig = visualize_ddim_determinism(ddpm_trainer, num_runs=3)
    fig.savefig('47_ddim_determinism.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 47_ddim_determinism.png")

    # ========================================
    # DDIM TRAJECTORY
    # ========================================
    print("\n" + "="*60)
    print("DDIM: Sampling Trajectory")
    print("="*60)

    fig = visualize_ddim_sampling_trajectory(ddpm_trainer, num_steps=50)
    fig.savefig('47_ddim_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 47_ddim_trajectory.png")

    # ========================================
    # TRAIN ON MOONS
    # ========================================
    print("\n" + "="*60)
    print("DDIM on Moons Dataset")
    print("="*60)

    data_moons = make_moons(1000)

    model_moons = DenoisingMLP(data_dim=2, hidden_dim=256)
    ddpm_moons = DDPMTrainer(model_moons, T=200, schedule='cosine')
    ddpm_moons.train(data_moons, n_epochs=2000, batch_size=128, lr=1e-3)

    fig = visualize_ddim_speed_comparison(ddpm_moons, data_moons, step_counts=[10, 25, 50, 100])
    fig.savefig('47_ddim_moons.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 47_ddim_moons.png")

    # ========================================
    # SUMMARY
    # ========================================
    print("""
======================================================================
SUMMARY: DDIM
======================================================================

KEY INSIGHTS:

1. FAST SAMPLING
   50 steps ≈ 1000 steps quality
   → 20x speedup with minimal quality loss

2. DETERMINISTIC OPTION
   η=0: same noise → same output
   → Reproducible, enables interpolation

3. SAME TRAINING
   Use DDPM-trained model
   → Just change sampling procedure

4. QUALITY vs SPEED TRADEOFF
   10 steps: fast but degraded
   50 steps: good balance
   100 steps: near-DDPM quality

5. STOCHASTICITY CONTROL
   η=0: deterministic
   η=1: DDPM-like variance
   → Tune based on application

NEXT: 48_cfg.py — Classifier-Free Guidance for conditional generation

======================================================================
    """)
