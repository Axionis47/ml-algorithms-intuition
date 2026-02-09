"""
SGD & MOMENTUM — Paradigm: GRADIENT DESCENT

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Follow the NEGATIVE GRADIENT downhill. That's it.

    θ ← θ - η × ∇L(θ)

"Move parameters in the direction that reduces loss the fastest."

MOMENTUM adds "velocity" — like a ball rolling downhill:
    v ← γ × v + η × ∇L(θ)      (accumulate velocity)
    θ ← θ - v                    (move with velocity)

Without momentum: zig-zag in ravines (inefficient)
With momentum: roll through ravines (much faster)

NESTEROV MOMENTUM looks ahead before computing gradient:
    Theoretical form:
        v ← γ × v + η × ∇L(θ - γv)  (gradient at look-ahead position)
        θ ← θ - v

    Practical form (Sutskever et al. 2013, used by PyTorch):
        v ← γ × v + η × g           (same velocity update)
        θ ← θ - (γ × v + η × g)     (apply momentum to NEW velocity)

    Both are equivalent reformulations. We implement the practical form.

"Where will I be if I keep going? Compute gradient THERE."

===============================================================
THE MATHEMATICS
===============================================================

VANILLA SGD:
    θ_{t+1} = θ_t - η ∇L(θ_t)

    η = learning rate (the MOST important hyperparameter)
    ∇L = gradient of loss w.r.t. parameters

    Convergence: O(1/√T) for convex, may not converge for non-convex

MOMENTUM:
    v_{t+1} = γ v_t + η ∇L(θ_t)
    θ_{t+1} = θ_t - v_{t+1}

    γ = momentum coefficient (typically 0.9)
    v accumulates past gradients → smooths out oscillations

    Physics analogy:
    - v = velocity of a ball
    - γ = friction (how fast velocity decays)
    - ∇L = force (gravity)

NESTEROV ACCELERATED GRADIENT:
    v_{t+1} = γ v_t + η ∇L(θ_t - γ v_t)
    θ_{t+1} = θ_t - v_{t+1}

    Key difference: gradient evaluated at LOOK-AHEAD position
    This gives better convergence (prescient correction)

MINI-BATCH SGD:
    Instead of full dataset gradient, use random subset:
    ∇L ≈ (1/|B|) Σ_{i∈B} ∇l_i

    Noise from mini-batches:
    - Acts as REGULARIZATION (prevents overfitting)
    - Helps ESCAPE local minima
    - Enables large dataset training

===============================================================
INDUCTIVE BIAS
===============================================================

1. GRADIENT DIRECTION is informative
   - Assumes loss surface is smooth enough
   - Gradient points toward steepest descent (locally)

2. LEARNING RATE is critical
   - Too large → diverge (overshoot minimum)
   - Too small → stuck or incredibly slow
   - Need to tune carefully or use schedules

3. MOMENTUM assumes consistent gradient direction
   - Helps in ravines (same direction)
   - Hurts at sharp turns (overshoots)

4. MINI-BATCH NOISE is a feature, not a bug
   - Regularizes, helps generalization
   - Larger batch → less noise → worse generalization
   - Small batch SGD ≈ implicit regularization

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# OPTIMIZERS
# ============================================================

class SGD:
    """
    Stochastic Gradient Descent with optional Momentum and Nesterov.

    The foundation of all deep learning optimization.
    """

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        """
        Parameters:
        -----------
        lr : float
            Learning rate η.
        momentum : float
            Momentum coefficient γ (0 = vanilla SGD, 0.9 = typical).
        nesterov : bool
            Use Nesterov accelerated gradient.
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def step(self, params, grads):
        """
        One optimization step.

        Args:
            params: dict of parameter name → numpy array
            grads: dict of parameter name → gradient array

        Returns:
            Updated params dict
        """
        for name in params:
            if name not in self.velocities:
                self.velocities[name] = np.zeros_like(params[name])

            if self.momentum > 0:
                if self.nesterov:
                    # Nesterov Accelerated Gradient (Sutskever et al. formulation)
                    # Step 1: Compute look-ahead step using OLD velocity
                    #   theta -= gamma * v_old + lr * grad
                    # Step 2: THEN update velocity for next iteration
                    #   v_new = gamma * v_old + lr * grad
                    #
                    # Equivalent to evaluating gradient at look-ahead position.
                    # This is the standard deep learning form (PyTorch style).
                    v_old = self.velocities[name].copy()
                    self.velocities[name] = self.momentum * v_old + self.lr * grads[name]
                    params[name] = params[name] - (self.momentum * self.velocities[name] +
                                                    self.lr * grads[name])
                else:
                    # Classical momentum: v = gamma * v + lr * grad; theta -= v
                    self.velocities[name] = self.momentum * self.velocities[name] + self.lr * grads[name]
                    params[name] = params[name] - self.velocities[name]
            else:
                # Vanilla SGD
                params[name] = params[name] - self.lr * grads[name]

        return params

    def reset(self):
        """Reset velocities."""
        self.velocities = {}


# ============================================================
# TEST FUNCTIONS (Loss Surfaces)
# ============================================================

def rosenbrock(xy):
    """
    Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²

    Has a narrow curved valley leading to global minimum at (1, 1).
    HARD for vanilla SGD (valley walls are steep, floor is flat).
    Momentum helps cut through the valley.
    """
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_grad(xy):
    x, y = xy[0], xy[1]
    dx = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


def beale(xy):
    """
    Beale function: Multiple flat regions and steep walls.
    Tests ability to navigate complex terrain.
    Global minimum at (3, 0.5).
    """
    x, y = xy[0], xy[1]
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 +
            (2.625 - x + x*y**3)**2)


def beale_grad(xy):
    x, y = xy[0], xy[1]
    t1 = 1.5 - x + x*y
    t2 = 2.25 - x + x*y**2
    t3 = 2.625 - x + x*y**3
    dx = 2*t1*(-1+y) + 2*t2*(-1+y**2) + 2*t3*(-1+y**3)
    dy = 2*t1*(x) + 2*t2*(2*x*y) + 2*t3*(3*x*y**2)
    return np.array([dx, dy])


def saddle_function(xy):
    """
    f(x,y) = x² - y²
    Saddle point at origin. SGD can get stuck here.
    """
    x, y = xy[0], xy[1]
    return x**2 - y**2


def saddle_grad(xy):
    x, y = xy[0], xy[1]
    return np.array([2*x, -2*y])


def himmelblau(xy):
    """
    Himmelblau's function: 4 local minima.
    f(x,y) = (x² + y - 11)² + (x + y² - 7)²
    Tests: does the optimizer find different minima from different starts?
    """
    x, y = xy[0], xy[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def himmelblau_grad(xy):
    x, y = xy[0], xy[1]
    dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([dx, dy])


# ============================================================
# OPTIMIZATION RUNNER
# ============================================================

def optimize(func, grad_func, start, optimizer, n_steps=500, clip_grad=10.0):
    """
    Run optimization and record trajectory.

    Args:
        func: Loss function f(xy) → scalar
        grad_func: Gradient function g(xy) → array
        start: Starting point (2D array)
        optimizer: SGD instance
        n_steps: Number of steps
        clip_grad: Gradient clipping threshold

    Returns:
        trajectory: List of (x, y) points
        losses: List of loss values
    """
    params = {'xy': np.array(start, dtype=float)}
    trajectory = [params['xy'].copy()]
    losses = [func(params['xy'])]

    for _ in range(n_steps):
        g = grad_func(params['xy'])
        # Clip gradients for stability
        g = np.clip(g, -clip_grad, clip_grad)
        grads = {'xy': g}
        params = optimizer.step(params, grads)
        trajectory.append(params['xy'].copy())
        losses.append(func(params['xy']))

    return np.array(trajectory), np.array(losses)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component?
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # -------- Experiment 1: Learning Rate Sweep --------
    print("\n1. EFFECT OF LEARNING RATE")
    print("-" * 40)
    print("On Rosenbrock function (narrow valley), starting at (-1, 1):")

    start = [-1.0, 1.0]
    for lr in [0.00001, 0.0001, 0.001, 0.01]:
        opt = SGD(lr=lr)
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                               n_steps=1000, clip_grad=5.0)
        final = traj[-1]
        print(f"   lr={lr:<10}  final_loss={losses[-1]:.4f}  "
              f"pos=({final[0]:.3f}, {final[1]:.3f})")

    print("→ Too small: barely moves in 1000 steps")
    print("→ Too large: oscillates wildly")
    print("→ Learning rate is THE most important hyperparameter")

    # -------- Experiment 2: Momentum Effect --------
    print("\n2. EFFECT OF MOMENTUM")
    print("-" * 40)
    print("Same Rosenbrock, different momentum coefficients:")

    for mom in [0.0, 0.5, 0.9, 0.99]:
        opt = SGD(lr=0.0001, momentum=mom)
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                               n_steps=1000, clip_grad=5.0)
        final = traj[-1]
        print(f"   momentum={mom:<5}  final_loss={losses[-1]:.4f}  "
              f"pos=({final[0]:.3f}, {final[1]:.3f})")

    print("→ momentum=0: slow, zig-zags in ravine")
    print("→ momentum=0.9: rolls through ravine efficiently")
    print("→ momentum=0.99: too much — overshoots and oscillates")

    # -------- Experiment 3: Nesterov vs Classical --------
    print("\n3. NESTEROV vs CLASSICAL MOMENTUM")
    print("-" * 40)

    for nesterov in [False, True]:
        opt = SGD(lr=0.0001, momentum=0.9, nesterov=nesterov)
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                               n_steps=1000, clip_grad=5.0)
        name = "Nesterov" if nesterov else "Classical"
        print(f"   {name:<12} final_loss={losses[-1]:.4f}  "
              f"pos=({traj[-1][0]:.3f}, {traj[-1][1]:.3f})")

    print("→ Nesterov looks ahead before computing gradient")
    print("→ Better at sharp turns, slightly faster convergence")

    # -------- Experiment 4: Saddle Point Behavior --------
    print("\n4. BEHAVIOR AT SADDLE POINTS")
    print("-" * 40)
    print("f(x,y) = x² - y², saddle at (0,0):")

    for start_point, desc in [([0.1, 0.0], "near saddle x-axis"),
                               ([0.0, 0.1], "near saddle y-axis"),
                               ([0.1, 0.1], "diagonal from saddle")]:
        opt = SGD(lr=0.01, momentum=0.9)
        traj, losses = optimize(saddle_function, saddle_grad, start_point, opt,
                               n_steps=200, clip_grad=5.0)
        print(f"   Start {desc}: ended at ({traj[-1][0]:.3f}, {traj[-1][1]:.3f})  "
              f"loss={losses[-1]:.4f}")

    print("→ Momentum helps escape saddle points (velocity carries through)")
    print("→ Without momentum, SGD can get stuck near saddle points")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_trajectories():
    """
    THE KEY INSIGHT: SGD vs Momentum vs Nesterov on Rosenbrock.
    Shows how momentum cuts through ravines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create contour plot
    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    start = [-1.5, 2.0]
    configs = [
        ('Vanilla SGD\n(no momentum)', SGD(lr=0.0002, momentum=0.0)),
        ('SGD + Momentum\n(γ=0.9)', SGD(lr=0.0002, momentum=0.9)),
        ('Nesterov\n(γ=0.9, look-ahead)', SGD(lr=0.0002, momentum=0.9, nesterov=True)),
    ]

    for ax, (name, opt) in zip(axes, configs):
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.6)
        ax.contourf(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.2)

        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                               n_steps=2000, clip_grad=5.0)

        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', linewidth=1, markersize=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(1, 1, 'r*', markersize=15, label='Global min')
        ax.plot(traj[-1, 0], traj[-1, 1], 'bs', markersize=8, label='End')

        ax.set_title(f'{name}\nFinal loss: {losses[-1]:.2f}', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')

    plt.suptitle('ROSENBROCK FUNCTION: Momentum Cuts Through Ravines\n'
                 'f(x,y) = (1-x)² + 100(y-x²)² | Global minimum at (1,1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_lr_sweep():
    """
    Learning rate effect: too small → too slow, too large → diverges.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    start = [-1.0, 2.0]
    lrs = [0.00001, 0.0001, 0.001, 0.01]

    for col, lr in enumerate(lrs):
        # Top: trajectory on contour
        ax = axes[0, col]
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 15), cmap='viridis', alpha=0.5)

        opt = SGD(lr=lr, momentum=0.9)
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                               n_steps=1000, clip_grad=10.0)

        ax.plot(traj[:, 0], traj[:, 1], 'r.-', linewidth=1, markersize=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6)
        ax.plot(1, 1, 'r*', markersize=10)
        ax.set_title(f'lr = {lr}', fontsize=11, fontweight='bold')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')

        # Bottom: loss curve
        ax = axes[1, col]
        ax.semilogy(losses)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log)')
        ax.set_title(f'Final: {losses[-1]:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('LEARNING RATE SWEEP: The Most Important Hyperparameter\n'
                 'Too small = barely moves | Too large = oscillates | '
                 'Just right = smooth convergence',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_momentum_physics():
    """
    Momentum as "ball rolling downhill": show velocity accumulation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1 & 2: With vs Without momentum on a simple quadratic ravine
    # f(x,y) = 0.5*x² + 50*y² (elongated bowl)
    def ravine(xy):
        return 0.5 * xy[0]**2 + 50 * xy[1]**2

    def ravine_grad(xy):
        return np.array([xy[0], 100 * xy[1]])

    start = [10.0, 1.0]
    x_range = np.linspace(-12, 12, 200)
    y_range = np.linspace(-2, 2, 200)
    Xg, Yg = np.meshgrid(x_range, y_range)
    Zg = 0.5 * Xg**2 + 50 * Yg**2

    # Without momentum
    ax = axes[0, 0]
    ax.contour(Xg, Yg, Zg, levels=20, cmap='viridis', alpha=0.5)
    opt_no = SGD(lr=0.005, momentum=0.0)
    traj_no, _ = optimize(ravine, ravine_grad, start, opt_no, n_steps=200)
    ax.plot(traj_no[:, 0], traj_no[:, 1], 'r.-', linewidth=1.5, markersize=2)
    ax.plot(traj_no[0, 0], traj_no[0, 1], 'go', markersize=8)
    ax.set_title('Without Momentum\n(zig-zag in ravine)', fontsize=11)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-2, 2)

    # With momentum
    ax = axes[0, 1]
    ax.contour(Xg, Yg, Zg, levels=20, cmap='viridis', alpha=0.5)
    opt_yes = SGD(lr=0.005, momentum=0.9)
    traj_yes, _ = optimize(ravine, ravine_grad, start, opt_yes, n_steps=200)
    ax.plot(traj_yes[:, 0], traj_yes[:, 1], 'r.-', linewidth=1.5, markersize=2)
    ax.plot(traj_yes[0, 0], traj_yes[0, 1], 'go', markersize=8)
    ax.set_title('With Momentum (γ=0.9)\n(rolls through ravine)', fontsize=11)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-2, 2)

    # Panel 3: Velocity magnitude over time
    ax = axes[1, 0]

    # Recompute with velocity tracking
    params_no = {'xy': np.array(start, dtype=float)}
    params_yes = {'xy': np.array(start, dtype=float)}
    vel_no_mag = []
    vel_yes_mag = []
    opt_no.reset()
    opt_yes.reset()

    for step in range(200):
        g = ravine_grad(params_no['xy'])
        g = np.clip(g, -10, 10)
        vel_no_mag.append(np.linalg.norm(g) * 0.005)  # "velocity" = lr * grad
        params_no = opt_no.step(params_no, {'xy': g})

        g = ravine_grad(params_yes['xy'])
        g = np.clip(g, -10, 10)
        vel = opt_yes.velocities.get('xy', np.zeros(2))
        vel_yes_mag.append(np.linalg.norm(vel))
        params_yes = opt_yes.step(params_yes, {'xy': g})

    ax.plot(vel_no_mag, label='No momentum', linewidth=2)
    ax.plot(vel_yes_mag, label='Momentum (γ=0.9)', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity magnitude')
    ax.set_title('Velocity Over Time\nMomentum accumulates speed', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Convergence comparison
    ax = axes[1, 1]
    configs = [
        ('No momentum', SGD(lr=0.005, momentum=0.0)),
        ('γ=0.5', SGD(lr=0.005, momentum=0.5)),
        ('γ=0.9', SGD(lr=0.005, momentum=0.9)),
        ('γ=0.99', SGD(lr=0.005, momentum=0.99)),
    ]
    for name, opt in configs:
        _, losses = optimize(ravine, ravine_grad, start, opt, n_steps=200)
        ax.semilogy(losses, label=name, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (log)')
    ax.set_title('Convergence: Momentum Accelerates\n'
                 'But too much (γ=0.99) → overshoots', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('MOMENTUM = Ball Rolling Downhill\n'
                 'Accumulates velocity in consistent gradient direction, '
                 'dampens oscillations',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("SGD & MOMENTUM — Paradigm: GRADIENT DESCENT")
    print("="*60)

    print("""
WHAT THIS IS:
    Follow the negative gradient. Add momentum to go faster.

KEY EQUATIONS:
    Vanilla: θ ← θ - η∇L
    Momentum: v ← γv + η∇L, θ ← θ - v
    Nesterov: v ← γv + η∇L(θ-γv), θ ← θ - v

INDUCTIVE BIAS:
    - GRADIENT points toward steepest descent
    - LEARNING RATE is the most important hyperparameter
    - MOMENTUM helps in ravines, hurts at sharp turns
    - BATCH NOISE acts as regularization
    """)

    # Run ablations
    ablation_experiments()

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_trajectories()
    save_path1 = '/Users/sid47/ML Algorithms/63_sgd_trajectories.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_lr_sweep()
    save_path2 = '/Users/sid47/ML Algorithms/63_sgd_lr_sweep.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_momentum_physics()
    save_path3 = '/Users/sid47/ML Algorithms/63_sgd_momentum_physics.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What SGD & Momentum Reveal")
    print("="*60)
    print("""
1. LEARNING RATE is everything
   → Too small: crawls, too large: diverges
   → There's a narrow "Goldilocks zone"

2. MOMENTUM eliminates zig-zag in ravines
   → γ=0.9 is the standard default
   → Accumulates velocity in consistent direction

3. NESTEROV looks ahead → better at turns
   → Compute gradient at future position
   → Slightly faster convergence than classical

4. SADDLE POINTS are a real problem
   → Vanilla SGD can stall near saddles
   → Momentum carries through with velocity

5. THESE are the FOUNDATIONS
   → Adam = momentum + adaptive learning rates
   → Every modern optimizer builds on these ideas

CONNECTIONS:
    → 12_mlp.py line 320: vanilla SGD (now you know why it's slow)
    → 64_adaptive_optimizers: adds per-parameter learning rates

NEXT: 64_adaptive_optimizers.py — AdaGrad → RMSprop → Adam
      (Why one learning rate for all parameters is not enough)
    """)
