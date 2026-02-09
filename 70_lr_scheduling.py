"""
LEARNING RATE SCHEDULING — Paradigm: DYNAMIC LEARNING RATES

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

The learning rate should CHANGE during training.

Early training:  High lr → explore broadly, escape bad regions
Late training:   Low lr  → fine-tune, converge precisely

This is DIFFERENT from adaptive optimizers (64_adaptive_optimizers.py):
    - Adaptive (Adam): per-PARAMETER lr based on gradient history
    - Scheduling: GLOBAL lr changes over time (applied on top of adaptive)

In practice, you use BOTH:
    Adam (adaptive per-param) + Cosine Schedule (global decay)

===============================================================
THE MATHEMATICS
===============================================================

STEP DECAY:
    lr_t = lr_0 * factor^(floor(t / step_size))
    "Drop lr by factor every N steps" (classic deep learning)

EXPONENTIAL DECAY:
    lr_t = lr_0 * gamma^t
    "Smooth continuous decay" (gamma < 1, typically 0.99-0.999)

COSINE ANNEALING (Loshchilov & Hutter, 2016):
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
    "Smooth oscillation from lr_max to lr_min over T steps"
    Most popular modern schedule.

LINEAR WARMUP:
    lr_t = lr_max * (t / warmup_steps)   for t < warmup_steps
    lr_t = lr_max                        for t >= warmup_steps
    "Start from 0, linearly ramp to target lr"
    Critical for Transformers (prevents early instability).

COSINE WITH WARM RESTARTS (SGDR, Loshchilov & Hutter, 2017):
    Like cosine, but RESET periodically.
    Each restart can have longer period (T_mult > 1).
    "Escape local minima by periodically raising lr"

ONE-CYCLE POLICY (Smith, 2018):
    Phase 1: Warmup from lr_min to lr_max (30% of training)
    Phase 2: Cosine decay from lr_max to lr_min (remaining 70%)
    "Super-convergence: train faster with higher peak lr"

===============================================================
INDUCTIVE BIAS — What Scheduling Assumes
===============================================================

1. EARLY = EXPLORATION, LATE = EXPLOITATION
   - High lr explores the loss landscape broadly
   - Low lr fine-tunes near the current basin

2. SMOOTH SCHEDULES > STEP FUNCTIONS
   - Cosine is smoother than step decay → better convergence
   - Sudden drops can destabilize training briefly

3. WARMUP PREVENTS INSTABILITY
   - Large initial gradients (random weights) + high lr = divergence
   - Warmup lets weights settle before ramping up lr

4. RESTARTS CAN HELP
   - Periodic lr increases → escape shallow local minima
   - T_mult > 1 → spend more time in later cycles (refined search)

===============================================================
WHERE IT SHOWS UP IN THIS REPO
===============================================================

- 63_sgd_momentum.py: Fixed lr baseline (no scheduling)
- 64_adaptive_optimizers.py: Per-param adaptation (scheduling adds global control)
- 12_mlp.py: Where scheduling matters in practice
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib

# Import from existing files
sys.path.insert(0, '/Users/sid47/ML Algorithms')
mod_63 = importlib.import_module('63_sgd_momentum')
mod_64 = importlib.import_module('64_adaptive_optimizers')

SGD = mod_63.SGD
rosenbrock = mod_63.rosenbrock
rosenbrock_grad = mod_63.rosenbrock_grad
optimize = mod_63.optimize

Adam = mod_64.Adam
train_tiny_mlp = mod_64.train_tiny_mlp
make_moons = mod_64.make_moons


# ============================================================
# SCHEDULER CLASSES
# ============================================================

class StepDecayScheduler:
    """
    Step Decay: lr *= factor every step_size steps.

    Classic schedule: "Drop lr by 10x every 30 epochs."
    Used in ResNet original paper.
    """

    def __init__(self, lr_init, step_size=100, factor=0.5):
        """
        Args:
            lr_init: Initial learning rate
            step_size: Steps between each decay
            factor: Multiply lr by this factor at each drop
        """
        self.lr_init = lr_init
        self.step_size = step_size
        self.factor = factor

    def get_lr(self, step):
        return self.lr_init * (self.factor ** (step // self.step_size))


class ExponentialDecayScheduler:
    """
    Exponential Decay: lr = lr_init * gamma^step

    Smooth continuous decay. gamma close to 1 = slow decay.
    """

    def __init__(self, lr_init, gamma=0.995):
        """
        Args:
            lr_init: Initial learning rate
            gamma: Decay rate per step (0.99 = fast, 0.999 = slow)
        """
        self.lr_init = lr_init
        self.gamma = gamma

    def get_lr(self, step):
        return self.lr_init * (self.gamma ** step)


class CosineAnnealingScheduler:
    """
    Cosine Annealing: smooth oscillation from lr_max to lr_min.

    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

    The most popular modern schedule. Smooth, no hyperparameter tuning
    beyond T (total steps) and lr_min.
    """

    def __init__(self, lr_max, T_max, lr_min=0.0):
        """
        Args:
            lr_max: Maximum (initial) learning rate
            T_max: Total number of steps for one cycle
            lr_min: Minimum learning rate at end of cycle
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max

    def get_lr(self, step):
        # Clamp step to T_max
        t = min(step, self.T_max)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * t / self.T_max))


class WarmupScheduler:
    """
    Linear Warmup: 0 → lr_max over warmup_steps, then constant.

    Critical for Transformers and large learning rates.
    Without warmup at high lr → gradients explode from random weights.
    """

    def __init__(self, lr_max, warmup_steps):
        """
        Args:
            lr_max: Target learning rate after warmup
            warmup_steps: Number of steps to linearly ramp up
        """
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.lr_max * (step / self.warmup_steps)
        return self.lr_max


class CosineWarmRestartsScheduler:
    """
    Cosine Annealing with Warm Restarts (SGDR).

    Cosine decay, but RESTART periodically.
    Each restart period can be longer (T_mult > 1).

    WHY? Periodic lr increases help escape shallow local minima.
    Longer later cycles → more time for fine-tuning.
    """

    def __init__(self, lr_max, T_0, T_mult=1, lr_min=0.0):
        """
        Args:
            lr_max: Maximum learning rate (at each restart)
            T_0: Length of the first restart period
            T_mult: Multiply period by this after each restart
            lr_min: Minimum learning rate
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult

    def get_lr(self, step):
        # Find which cycle we're in
        if self.T_mult == 1:
            # Simple case: all cycles same length
            t_cur = step % self.T_0
            T_i = self.T_0
        else:
            # Geometric: T_0, T_0*T_mult, T_0*T_mult^2, ...
            T_i = self.T_0
            t_cur = step
            while t_cur >= T_i:
                t_cur -= T_i
                T_i = int(T_i * self.T_mult)

        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * t_cur / T_i))


class OneCycleScheduler:
    """
    One-Cycle Policy (Leslie Smith, 2018): "Super-Convergence."

    Phase 1 (warmup, 30%): lr ramps from lr_min to lr_max
    Phase 2 (decay, 70%):  lr decays from lr_max to lr_min via cosine

    Allows using MUCH higher peak lr → faster training.
    Momentum is inversely varied (low during high lr, high during low lr).
    """

    def __init__(self, lr_max, total_steps, lr_min=None, warmup_frac=0.3):
        """
        Args:
            lr_max: Peak learning rate
            total_steps: Total number of training steps
            lr_min: Minimum lr (default: lr_max/25)
            warmup_frac: Fraction of steps for warmup phase
        """
        self.lr_max = lr_max
        self.lr_min = lr_min if lr_min is not None else lr_max / 25.0
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)

    def get_lr(self, step):
        if step < self.warmup_steps:
            # Phase 1: linear warmup
            progress = step / self.warmup_steps
            return self.lr_min + (self.lr_max - self.lr_min) * progress
        else:
            # Phase 2: cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(np.pi * progress))


# ============================================================
# INTEGRATION FUNCTIONS
# ============================================================

def optimize_with_schedule(func, grad_func, start, optimizer, scheduler, n_steps=500,
                           clip_grad=10.0):
    """
    Optimization with learning rate scheduling.

    At each step: set optimizer.lr = scheduler.get_lr(step), then step.

    Args:
        func: Loss function f(xy) → scalar
        grad_func: Gradient function g(xy) → array
        start: Starting point
        optimizer: SGD or Adam instance
        scheduler: Scheduler instance with get_lr(step) method
        n_steps: Number of optimization steps
        clip_grad: Gradient clipping threshold

    Returns:
        trajectory: List of points visited
        losses: List of loss values
        lrs: List of learning rates used
    """
    params = {'xy': np.array(start, dtype=float)}
    trajectory = [params['xy'].copy()]
    losses = [func(params['xy'])]
    lrs = [scheduler.get_lr(0)]

    optimizer.reset()

    for step in range(n_steps):
        # Update learning rate from scheduler
        optimizer.lr = scheduler.get_lr(step)
        lrs.append(optimizer.lr)

        g = grad_func(params['xy'])
        g = np.clip(g, -clip_grad, clip_grad)
        grads = {'xy': g}
        params = optimizer.step(params, grads)
        trajectory.append(params['xy'].copy())
        losses.append(func(params['xy']))

    return trajectory, losses, lrs


def train_mlp_with_schedule(X, y, optimizer, scheduler, n_epochs=200,
                             hidden_size=16, random_state=42):
    """
    Train a 2-layer MLP with learning rate scheduling.

    Same architecture as train_tiny_mlp in 64, but updates lr each epoch.

    Returns:
        losses: Training loss curve
        lrs: Learning rate at each epoch
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    # Xavier initialization
    W1 = rng.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_size)
    W2 = rng.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(1)

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    optimizer.reset()
    losses = []
    lrs = []

    for epoch in range(n_epochs):
        # Update lr from scheduler
        optimizer.lr = scheduler.get_lr(epoch)
        lrs.append(optimizer.lr)

        # Forward pass
        z1 = X @ params['W1'] + params['b1']
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ params['W2'] + params['b2']
        y_pred = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))

        # Binary cross-entropy loss
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        y_true = y.reshape(-1, 1)
        loss = -np.mean(y_true * np.log(y_pred_clipped) +
                        (1 - y_true) * np.log(1 - y_pred_clipped))

        # Detect divergence
        if np.isnan(loss) or np.isinf(loss):
            losses.append(float('inf'))
            continue

        losses.append(loss)

        # Backward pass
        dz2 = (y_pred_clipped - y_true) / n_samples
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ params['W2'].T
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        # Clip gradients
        for name in grads:
            grads[name] = np.clip(grads[name], -5.0, 5.0)

        params = optimizer.step(params, grads)

    return losses, lrs


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    4 ablations exploring learning rate scheduling.
    """
    np.random.seed(42)
    print("=" * 70)
    print("LEARNING RATE SCHEDULING — ABLATION EXPERIMENTS")
    print("=" * 70)

    # --------------------------------------------------------
    # ABLATION 1: Fixed lr vs Step Decay vs Cosine on Rosenbrock
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 1: Fixed LR vs Step Decay vs Cosine on Rosenbrock")
    print("=" * 60)

    start = np.array([-1.5, 1.5])
    n_steps = 500
    lr = 0.001

    schedules = {
        'Fixed (lr=0.001)': None,
        'Step (drop 0.5x/100)': StepDecayScheduler(lr, step_size=100, factor=0.5),
        'Cosine (→ 0)': CosineAnnealingScheduler(lr, T_max=n_steps, lr_min=0.0),
        'Exponential (0.995)': ExponentialDecayScheduler(lr, gamma=0.995),
    }

    print(f"\n  Rosenbrock function: minimum at (1, 1)")
    print(f"  Start: {start}")
    print(f"  {'Schedule':25s}  {'Final Loss':>12s}  {'Final Point':>20s}  {'Final LR':>10s}")
    print(f"  {'-'*75}")

    for name, sched in schedules.items():
        opt = SGD(lr=lr, momentum=0.9)
        if sched is None:
            # Fixed lr: use optimize from 63
            traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt, n_steps)
            final_lr = lr
        else:
            traj, losses, lrs = optimize_with_schedule(
                rosenbrock, rosenbrock_grad, start, opt, sched, n_steps)
            final_lr = lrs[-1]

        final = traj[-1]
        print(f"  {name:25s}  {losses[-1]:12.6f}  ({final[0]:8.4f}, {final[1]:8.4f})  {final_lr:10.6f}")

    print("\n  KEY INSIGHT: Cosine and step decay converge closer to (1,1)")
    print("  because they reduce lr near the end for fine-tuning.")

    # --------------------------------------------------------
    # ABLATION 2: Warmup Effect at Aggressive LR
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 2: Warmup Effect at Aggressive Learning Rate")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    n_epochs = 200
    aggressive_lr = 0.5

    print(f"\n  MLP on moons dataset, aggressive lr = {aggressive_lr}")
    print(f"  {'Warmup Steps':>14s}  {'Final Loss':>12s}  {'Diverged?':>10s}")
    print(f"  {'-'*40}")

    for warmup in [0, 5, 10, 20, 50]:
        if warmup == 0:
            # No warmup: constant lr
            sched = WarmupScheduler(aggressive_lr, warmup_steps=1)
        else:
            sched = WarmupScheduler(aggressive_lr, warmup_steps=warmup)

        opt = Adam(lr=aggressive_lr)
        losses, lrs = train_mlp_with_schedule(X, y, opt, sched, n_epochs=n_epochs)

        # Check for divergence
        diverged = any(np.isinf(l) or np.isnan(l) for l in losses[-20:])
        final_loss = losses[-1] if not (np.isinf(losses[-1]) or np.isnan(losses[-1])) else float('inf')

        print(f"  {warmup:14d}  {final_loss:12.4f}  {'YES' if diverged else 'no':>10s}")

    print("\n  KEY INSIGHT: Without warmup, aggressive lr can cause instability.")
    print("  Even 10-20 warmup steps are enough to stabilize training.")
    print("  Warmup lets weights develop reasonable magnitudes before high lr kicks in.")

    # --------------------------------------------------------
    # ABLATION 3: Warm Restarts — Period and T_mult
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 3: Cosine Warm Restarts — Period & T_mult")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    n_epochs = 300
    lr = 0.01

    print(f"\n  MLP on moons, {n_epochs} epochs, base lr = {lr}")
    print(f"  {'Config':30s}  {'Final Loss':>12s}  {'Min Loss':>10s}")
    print(f"  {'-'*58}")

    configs = [
        ('Cosine (no restart)', CosineAnnealingScheduler(lr, T_max=n_epochs, lr_min=1e-5)),
        ('Restart T_0=50, mult=1', CosineWarmRestartsScheduler(lr, T_0=50, T_mult=1, lr_min=1e-5)),
        ('Restart T_0=100, mult=1', CosineWarmRestartsScheduler(lr, T_0=100, T_mult=1, lr_min=1e-5)),
        ('Restart T_0=50, mult=2', CosineWarmRestartsScheduler(lr, T_0=50, T_mult=2, lr_min=1e-5)),
        ('Restart T_0=100, mult=2', CosineWarmRestartsScheduler(lr, T_0=100, T_mult=2, lr_min=1e-5)),
    ]

    for name, sched in configs:
        opt = Adam(lr=lr)
        losses, lrs = train_mlp_with_schedule(X, y, opt, sched, n_epochs=n_epochs)
        valid_losses = [l for l in losses if not (np.isinf(l) or np.isnan(l))]
        final = valid_losses[-1] if valid_losses else float('inf')
        min_loss = min(valid_losses) if valid_losses else float('inf')
        print(f"  {name:30s}  {final:12.4f}  {min_loss:10.4f}")

    print("\n  KEY INSIGHT: Warm restarts can help escape shallow local minima.")
    print("  T_mult=2 gives longer later cycles → more time for fine-tuning.")

    # --------------------------------------------------------
    # ABLATION 4: All 6 Schedulers on Same Problem
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 4: All 6 Schedulers on Same MLP Problem")
    print("=" * 60)

    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    n_epochs = 200
    lr = 0.01

    print(f"\n  MLP on moons, {n_epochs} epochs")
    print(f"  {'Scheduler':30s}  {'Final Loss':>12s}  {'Min Loss':>10s}  {'Best Epoch':>12s}")
    print(f"  {'-'*72}")

    all_schedules = [
        ('Fixed (lr=0.01)', WarmupScheduler(lr, warmup_steps=1)),
        ('Step (0.5x/50)', StepDecayScheduler(lr, step_size=50, factor=0.5)),
        ('Exponential (0.99)', ExponentialDecayScheduler(lr, gamma=0.99)),
        ('Cosine', CosineAnnealingScheduler(lr, T_max=n_epochs, lr_min=1e-5)),
        ('Warm Restarts (T=50)', CosineWarmRestartsScheduler(lr, T_0=50, T_mult=2, lr_min=1e-5)),
        ('One-Cycle (peak=0.05)', OneCycleScheduler(lr_max=0.05, total_steps=n_epochs)),
    ]

    for name, sched in all_schedules:
        opt = Adam(lr=lr)
        losses, lrs_used = train_mlp_with_schedule(X, y, opt, sched, n_epochs=n_epochs)
        valid_losses = [l for l in losses if not (np.isinf(l) or np.isnan(l))]
        final = valid_losses[-1] if valid_losses else float('inf')
        min_loss = min(valid_losses) if valid_losses else float('inf')
        best_epoch = np.argmin(valid_losses) if valid_losses else -1
        print(f"  {name:30s}  {final:12.4f}  {min_loss:10.4f}  {best_epoch:12d}")

    print("\n  KEY INSIGHT: Cosine and One-Cycle typically achieve lower final loss.")
    print("  One-Cycle can train faster by using a higher peak lr.")
    print("  Step decay is simple but less smooth than cosine.")

    return True


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_schedule_shapes():
    """
    Show all 6 lr schedule curves over time.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Learning Rate Schedule Shapes\n'
                 'How each schedule varies lr over training',
                 fontsize=14, fontweight='bold')

    total_steps = 300
    lr = 0.01
    steps = np.arange(total_steps)

    schedules = [
        ('Step Decay\n(factor=0.5, every 60)',
         StepDecayScheduler(lr, step_size=60, factor=0.5), '#2196F3'),
        ('Exponential Decay\n(gamma=0.99)',
         ExponentialDecayScheduler(lr, gamma=0.99), '#F44336'),
        ('Cosine Annealing\n(T=300)',
         CosineAnnealingScheduler(lr, T_max=total_steps, lr_min=0.0), '#4CAF50'),
        ('Linear Warmup\n(50 steps)',
         WarmupScheduler(lr, warmup_steps=50), '#FF9800'),
        ('Cosine Warm Restarts\n(T_0=75, T_mult=2)',
         CosineWarmRestartsScheduler(lr, T_0=75, T_mult=2, lr_min=0.0), '#9C27B0'),
        ('One-Cycle\n(peak=0.03, 30% warmup)',
         OneCycleScheduler(lr_max=0.03, total_steps=total_steps), '#795548'),
    ]

    for idx, (name, sched, color) in enumerate(schedules):
        ax = axes[idx // 3, idx % 3]
        lrs = [sched.get_lr(t) for t in steps]
        ax.plot(steps, lrs, color=color, linewidth=2)
        ax.fill_between(steps, 0, lrs, alpha=0.15, color=color)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/70_schedule_shapes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 70_schedule_shapes.png")
    return fig


def visualize_schedule_training():
    """
    Compare all schedules on same MLP problem.
    Shows loss curves and final performance.
    """
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    n_epochs = 200
    lr = 0.01

    schedules = [
        ('Fixed', WarmupScheduler(lr, warmup_steps=1), '#2196F3'),
        ('Step', StepDecayScheduler(lr, step_size=50, factor=0.5), '#F44336'),
        ('Exponential', ExponentialDecayScheduler(lr, gamma=0.99), '#4CAF50'),
        ('Cosine', CosineAnnealingScheduler(lr, T_max=n_epochs, lr_min=1e-5), '#FF9800'),
        ('Warm Restarts', CosineWarmRestartsScheduler(lr, T_0=50, T_mult=2, lr_min=1e-5), '#9C27B0'),
        ('One-Cycle', OneCycleScheduler(lr_max=0.05, total_steps=n_epochs), '#795548'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Learning Rate Schedules: Training Comparison on Moons Dataset',
                 fontsize=14, fontweight='bold')

    all_results = {}
    for name, sched, color in schedules:
        opt = Adam(lr=lr)
        losses, lrs = train_mlp_with_schedule(X, y, opt, sched, n_epochs=n_epochs)
        all_results[name] = {'losses': losses, 'lrs': lrs, 'color': color}

    # Plot 1: Loss curves
    ax = axes[0]
    for name, data in all_results.items():
        valid_losses = [l if not (np.isinf(l) or np.isnan(l)) else None for l in data['losses']]
        ax.plot(valid_losses, color=data['color'], linewidth=1.5, label=name, alpha=0.8)
    ax.set_title('Training Loss Curves', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: Final loss bar chart
    ax = axes[1]
    names = list(all_results.keys())
    final_losses = []
    colors = []
    for name in names:
        valid = [l for l in all_results[name]['losses'] if not (np.isinf(l) or np.isnan(l))]
        final_losses.append(valid[-1] if valid else 1.0)
        colors.append(all_results[name]['color'])

    bars = ax.bar(range(len(names)), final_losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_title('Final Loss Comparison', fontsize=12)
    ax.set_ylabel('Final Loss')

    for bar, val in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=8)

    # Plot 3: LR schedule + loss overlay (for Cosine)
    ax = axes[2]
    cosine_data = all_results['Cosine']
    epochs = np.arange(len(cosine_data['lrs']))
    ax.plot(epochs, cosine_data['lrs'], color='#FF9800', linewidth=2, label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate', color='#FF9800')
    ax.tick_params(axis='y', labelcolor='#FF9800')

    ax2 = ax.twinx()
    valid_losses = [l if not (np.isinf(l) or np.isnan(l)) else None
                    for l in cosine_data['losses']]
    ax2.plot(valid_losses, color='#2196F3', linewidth=1.5, label='Loss', alpha=0.7)
    ax2.set_ylabel('Loss', color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')
    ax.set_title('Cosine Schedule: LR & Loss Overlay', fontsize=12)
    ax.grid(True, alpha=0.2)

    # Combined legend
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color='#FF9800', linewidth=2),
             Line2D([0], [0], color='#2196F3', linewidth=1.5)]
    ax.legend(lines, ['Learning Rate', 'Loss'], loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/70_schedule_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 70_schedule_training.png")
    return fig


def visualize_warmup_effect():
    """
    Show the effect of warmup at aggressive learning rates.
    Without warmup → instability; with warmup → smooth convergence.
    """
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    n_epochs = 200
    aggressive_lr = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Warmup Effect at Aggressive LR = {aggressive_lr}\n'
                 f'Warmup prevents early instability from random weight gradients',
                 fontsize=14, fontweight='bold')

    warmup_configs = [
        (0, 'No Warmup', '#F44336'),
        (10, '10-Step Warmup', '#FF9800'),
        (30, '30-Step Warmup', '#4CAF50'),
    ]

    for col, (warmup, label, color) in enumerate(warmup_configs):
        ax = axes[col]

        if warmup == 0:
            sched = WarmupScheduler(aggressive_lr, warmup_steps=1)
        else:
            sched = WarmupScheduler(aggressive_lr, warmup_steps=warmup)

        opt = Adam(lr=aggressive_lr)
        losses, lrs = train_mlp_with_schedule(X, y, opt, sched, n_epochs=n_epochs)

        epochs = np.arange(len(losses))
        # Plot loss
        valid_losses = [l if not (np.isinf(l) or np.isnan(l)) else None for l in losses]
        ax.plot(epochs, valid_losses, color=color, linewidth=1.5, label='Loss')

        # Shade warmup region
        if warmup > 0:
            ax.axvspan(0, warmup, alpha=0.15, color='yellow', label=f'Warmup ({warmup} steps)')
            ax.axvline(warmup, color='gray', linestyle='--', alpha=0.5)

        # LR overlay
        ax_lr = ax.twinx()
        ax_lr.plot(np.arange(len(lrs)), lrs, color='gray', linewidth=1, alpha=0.5,
                   linestyle='--')
        ax_lr.set_ylabel('LR', color='gray', fontsize=9)
        ax_lr.tick_params(axis='y', labelcolor='gray')

        final = valid_losses[-1] if valid_losses[-1] is not None else float('inf')
        diverged = any(l is None for l in valid_losses[-20:])

        ax.set_title(f'{label}\nFinal loss: {final:.4f}' +
                     (' (DIVERGED!)' if diverged else ''),
                     fontsize=11, color='red' if diverged else 'black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=min(2.0, max(l for l in valid_losses if l is not None) * 1.2))

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/70_warmup_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 70_warmup_effect.png")
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  LEARNING RATE SCHEDULING — Paradigm: DYNAMIC LEARNING RATES")
    print("  Step, Exponential, Cosine, Warmup, Warm Restarts, One-Cycle")
    print("=" * 70)

    # Run ablation experiments
    ablation_experiments()

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    visualize_schedule_shapes()
    visualize_schedule_training()
    visualize_warmup_effect()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Learning Rate Scheduling")
    print("=" * 70)
    print("""
    SCHEDULE          | BEHAVIOR                      | WHEN TO USE
    ------------------+-------------------------------+---------------------------
    Step Decay        | Drop by factor every N steps  | Classic (ResNet, VGG)
    Exponential       | Smooth continuous decay        | When step decay is too abrupt
    Cosine Annealing  | Smooth cosine curve to 0      | Modern default (most papers)
    Linear Warmup     | 0 → lr_max over N steps       | Transformers, high lr
    Warm Restarts     | Cosine + periodic resets       | Escaping local minima
    One-Cycle         | Warmup → peak → cosine down   | Fast training (super-convergence)

    RULES OF THUMB:
    1. DEFAULT: Cosine annealing (works well almost everywhere)
    2. TRANSFORMERS: Linear warmup (1-10% of training) + cosine decay
    3. WANT SPEED: One-Cycle with 3-5x higher peak lr
    4. STUCK IN LOCAL MIN: Try warm restarts (SGDR)
    5. SIMPLE BASELINE: Step decay (divide lr by 10 at 60% and 80% of training)

    REMEMBER: Scheduling is GLOBAL lr control.
    Combine with Adam for per-parameter + global scheduling.
    Adam + Cosine Schedule = industry standard (2024+).
    """)
