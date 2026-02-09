"""
OPTIMIZER ARENA — Paradigm: COMPARISON

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Head-to-head comparison of gradient-based optimizers on
classic loss surfaces.

SIX OPTIMIZERS, FOUR SURFACES:
    SGD:       Vanilla gradient descent (the baseline)
    SGD+Mom:   Momentum carries through ravines
    AdaGrad:   Per-parameter learning rates (shrink over time)
    RMSProp:   Leaky AdaGrad (forgets old gradients)
    Adam:      RMSProp + Momentum (the default)
    AdamW:     Adam with decoupled weight decay

FOUR SURFACES:
    Rosenbrock:  Narrow curved valley → tests ravine navigation
    Beale:       Flat plateaus + steep walls → tests escape
    Himmelblau:  4 local minima → tests which minimum is found
    Saddle:      Saddle point at origin → tests escape ability

| Optimizer | Best For                         | Weakness              |
|-----------|----------------------------------|-----------------------|
| SGD       | Convex, well-conditioned         | Slow in ravines       |
| SGD+Mom   | Most practical training          | Overshoots at turns   |
| AdaGrad   | Sparse data (NLP embeddings)     | LR decays to zero     |
| RMSProp   | Non-stationary (RL, RNNs)        | No momentum term      |
| Adam      | Default for neural networks      | May not generalize    |
| AdamW     | When using weight decay (LLMs)   | More hyperparameters  |

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module

# Import from 63_sgd_momentum
sgd_module = import_module('63_sgd_momentum')
SGD = sgd_module.SGD
rosenbrock = sgd_module.rosenbrock
rosenbrock_grad = sgd_module.rosenbrock_grad
beale = sgd_module.beale
beale_grad = sgd_module.beale_grad
himmelblau = sgd_module.himmelblau
himmelblau_grad = sgd_module.himmelblau_grad
saddle_function = sgd_module.saddle_function
saddle_grad = sgd_module.saddle_grad
optimize = sgd_module.optimize


# ============================================================
# ADAPTIVE OPTIMIZERS (defined inline for standalone execution)
# See 64_adaptive_optimizers.py for full implementations + ablations
# ============================================================

class AdaGrad:
    """
    AdaGrad: accumulate squared gradients, shrink learning rate per-param.

    h_i += g_i^2
    theta_i -= (lr / sqrt(h_i + eps)) * g_i

    Problem: learning rate monotonically decreases to zero.
    """

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cache = {}

    def step(self, params, grads):
        for name in params:
            if name not in self.cache:
                self.cache[name] = np.zeros_like(params[name])
            self.cache[name] += grads[name] ** 2
            params[name] = params[name] - self.lr * grads[name] / (
                np.sqrt(self.cache[name]) + self.eps)
        return params

    def reset(self):
        self.cache = {}


class RMSProp:
    """
    RMSProp: leaky AdaGrad — use exponential moving average of squared grads.

    h = beta * h + (1 - beta) * g^2
    theta -= (lr / sqrt(h + eps)) * g

    Fixes AdaGrad's dying learning rate problem.
    """

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}

    def step(self, params, grads):
        for name in params:
            if name not in self.cache:
                self.cache[name] = np.zeros_like(params[name])
            self.cache[name] = (self.beta * self.cache[name] +
                                (1 - self.beta) * grads[name] ** 2)
            params[name] = params[name] - self.lr * grads[name] / (
                np.sqrt(self.cache[name]) + self.eps)
        return params

    def reset(self):
        self.cache = {}


class Adam:
    """
    Adam = RMSProp + Momentum with bias correction.

    m = beta1 * m + (1 - beta1) * g        (first moment)
    v = beta2 * v + (1 - beta2) * g^2      (second moment)
    m_hat = m / (1 - beta1^t)              (bias correction)
    v_hat = v / (1 - beta2^t)
    theta -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] = params[name] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


class AdamW:
    """
    AdamW = Adam with DECOUPLED weight decay.

    Standard Adam with L2: adds lambda*theta to gradient BEFORE adaptive scaling.
    AdamW: subtracts lambda*theta AFTER adaptive step.

    This matters because adaptive methods scale gradients differently per-param.
    Decoupled decay applies uniformly, giving better regularization.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            # Decoupled weight decay: applied to params, not to gradient
            params[name] = (params[name] * (1 - self.lr * self.weight_decay) -
                            self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return params

    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0


# ============================================================
# SURFACE CONFIGS
# ============================================================

SURFACES = {
    'Rosenbrock': {
        'func': rosenbrock,
        'grad': rosenbrock_grad,
        'start': [-1.5, 2.0],
        'x_range': (-2, 2),
        'y_range': (-1, 3),
        'levels': np.logspace(-1, 3.5, 25),
        'minimum': (1.0, 1.0),
    },
    'Beale': {
        'func': beale,
        'grad': beale_grad,
        'start': [-1.0, 1.5],
        'x_range': (-2, 4.5),
        'y_range': (-1, 3),
        'levels': np.logspace(-1, 4, 25),
        'minimum': (3.0, 0.5),
    },
    'Himmelblau': {
        'func': himmelblau,
        'grad': himmelblau_grad,
        'start': [-3.0, -2.0],
        'x_range': (-5, 5),
        'y_range': (-5, 5),
        'levels': np.logspace(0, 3, 25),
        'minimum': None,  # 4 local minima
    },
    'Saddle': {
        'func': saddle_function,
        'grad': saddle_grad,
        'start': [0.5, 0.05],
        'x_range': (-2, 2),
        'y_range': (-2, 2),
        'levels': np.linspace(-4, 4, 25),
        'minimum': None,  # saddle at origin
    },
}


def get_optimizers():
    """Return dict of optimizer name -> optimizer instance."""
    return {
        'SGD': SGD(lr=0.0003),
        'SGD+Mom': SGD(lr=0.0003, momentum=0.9),
        'AdaGrad': AdaGrad(lr=0.1),
        'Adam': Adam(lr=0.01),
    }


def get_all_optimizers():
    """Return all six optimizers for the convergence plot."""
    return {
        'SGD': SGD(lr=0.0003),
        'SGD+Mom': SGD(lr=0.0003, momentum=0.9),
        'AdaGrad': AdaGrad(lr=0.1),
        'RMSProp': RMSProp(lr=0.01),
        'Adam': Adam(lr=0.01),
        'AdamW': AdamW(lr=0.01, weight_decay=0.01),
    }


# ============================================================
# MAIN VISUALIZATION: Trajectory Grid
# ============================================================

def visualize_arena():
    """
    Grid: rows = surfaces, columns = optimizers.
    Each cell shows contour + optimizer trajectory.
    """
    surface_names = list(SURFACES.keys())
    opt_configs = get_optimizers()
    opt_names = list(opt_configs.keys())

    n_rows = len(surface_names)
    n_cols = len(opt_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    colors = {'SGD': '#e74c3c', 'SGD+Mom': '#3498db',
              'AdaGrad': '#2ecc71', 'Adam': '#9b59b6'}

    for row, surf_name in enumerate(surface_names):
        cfg = SURFACES[surf_name]
        func, grad_func = cfg['func'], cfg['grad']
        start = cfg['start']
        xr, yr = cfg['x_range'], cfg['y_range']
        levels = cfg['levels']

        # Build contour grid
        xg = np.linspace(xr[0], xr[1], 200)
        yg = np.linspace(yr[0], yr[1], 200)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Zg[i, j] = func(np.array([Xg[i, j], Yg[i, j]]))

        for col, opt_name in enumerate(opt_names):
            ax = axes[row, col]

            # Contour
            if surf_name == 'Saddle':
                ax.contour(Xg, Yg, Zg, levels=levels, cmap='RdBu', alpha=0.6)
                ax.contourf(Xg, Yg, Zg, levels=levels, cmap='RdBu', alpha=0.15)
            else:
                ax.contour(Xg, Yg, Zg, levels=levels, cmap='viridis', alpha=0.6)
                ax.contourf(Xg, Yg, Zg, levels=levels, cmap='viridis', alpha=0.15)

            # Run optimizer
            opt = _make_fresh_optimizer(opt_name)
            traj, losses = optimize(func, grad_func, start, opt,
                                    n_steps=1000, clip_grad=5.0)

            # Plot trajectory
            color = colors[opt_name]
            ax.plot(traj[:, 0], traj[:, 1], '.-', color=color,
                    linewidth=1, markersize=1, alpha=0.7)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color='black',
                    markersize=6, zorder=5)
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color,
                    markersize=7, zorder=5, markeredgecolor='black')

            # Mark minimum if known
            if cfg['minimum'] is not None:
                ax.plot(cfg['minimum'][0], cfg['minimum'][1], 'r*',
                        markersize=12, zorder=5)

            ax.set_xlim(xr)
            ax.set_ylim(yr)

            if row == 0:
                ax.set_title(f'{opt_name}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{surf_name}', fontsize=11, fontweight='bold')

            ax.text(0.05, 0.95, f'loss={losses[-1]:.2e}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('OPTIMIZER ARENA: Trajectories on Classic Loss Surfaces\n'
                 'Black dot = start | Colored square = end | '
                 'Red star = known minimum',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def _make_fresh_optimizer(name):
    """Create a fresh optimizer instance (reset state)."""
    configs = {
        'SGD': lambda: SGD(lr=0.0003),
        'SGD+Mom': lambda: SGD(lr=0.0003, momentum=0.9),
        'AdaGrad': lambda: AdaGrad(lr=0.1),
        'RMSProp': lambda: RMSProp(lr=0.01),
        'Adam': lambda: Adam(lr=0.01),
        'AdamW': lambda: AdamW(lr=0.01, weight_decay=0.01),
    }
    return configs[name]()


# ============================================================
# CONVERGENCE VISUALIZATION
# ============================================================

def visualize_convergence():
    """
    2x2 grid: one subplot per surface.
    All optimizer loss curves overlaid.
    """
    surface_names = list(SURFACES.keys())
    all_opts = list(get_all_optimizers().keys())

    colors = {
        'SGD': '#e74c3c',
        'SGD+Mom': '#3498db',
        'AdaGrad': '#2ecc71',
        'RMSProp': '#f39c12',
        'Adam': '#9b59b6',
        'AdamW': '#1abc9c',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, surf_name in enumerate(surface_names):
        ax = axes[idx]
        cfg = SURFACES[surf_name]

        for opt_name in all_opts:
            opt = _make_fresh_optimizer(opt_name)
            _, losses = optimize(cfg['func'], cfg['grad'], cfg['start'],
                                 opt, n_steps=1000, clip_grad=5.0)

            # Clip for log scale (avoid log of negative)
            losses_plot = np.maximum(losses, 1e-15)
            ax.semilogy(losses_plot, label=opt_name, color=colors[opt_name],
                        linewidth=2, alpha=0.8)

        ax.set_title(f'{surf_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log scale)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1000)

    plt.suptitle('OPTIMIZER CONVERGENCE: Loss vs Steps\n'
                 'Adaptive methods (Adam, RMSProp) converge faster '
                 'but SGD+Momentum is competitive',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison_table():
    """
    Print final loss values for all optimizers on all surfaces.
    """
    surface_names = list(SURFACES.keys())
    all_opts = list(get_all_optimizers().keys())

    print("\n" + "=" * 75)
    print("OPTIMIZER COMPARISON TABLE: Final Loss After 1000 Steps")
    print("=" * 75)

    # Header
    header = f"{'Optimizer':<12}"
    for surf in surface_names:
        header += f"{'|':>2} {surf:<14}"
    print(header)
    print("-" * 75)

    # Each optimizer row
    for opt_name in all_opts:
        row = f"{opt_name:<12}"
        for surf_name in surface_names:
            cfg = SURFACES[surf_name]
            opt = _make_fresh_optimizer(opt_name)
            _, losses = optimize(cfg['func'], cfg['grad'], cfg['start'],
                                 opt, n_steps=1000, clip_grad=5.0)
            final = losses[-1]
            if abs(final) < 0.001:
                row += f"{'|':>2} {final:<14.2e}"
            else:
                row += f"{'|':>2} {final:<14.4f}"
        print(row)

    print("-" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMIZER ARENA — Paradigm: COMPARISON")
    print("=" * 60)

    print("""
WHAT THIS IS:
    Every major gradient optimizer on every loss surface.
    Same start, same steps, different strategies.

THE CONTENDERS:
    SGD:       theta -= lr * grad                  (baseline)
    SGD+Mom:   v = gamma*v + lr*grad; theta -= v   (momentum)
    AdaGrad:   h += grad^2; theta -= lr*grad/sqrt(h) (adaptive)
    RMSProp:   h = decay*h + (1-d)*g^2             (leaky AdaGrad)
    Adam:      momentum + adaptive + bias correction (the default)
    AdamW:     Adam + decoupled weight decay        (for regularization)

THE SURFACES:
    Rosenbrock:  Narrow curved valley (tests ravine navigation)
    Beale:       Flat plateaus + steep walls (tests plateau escape)
    Himmelblau:  4 equivalent minima (tests which one is found)
    Saddle:      x^2 - y^2 (tests saddle point escape)
    """)

    # Print comparison table
    print_comparison_table()

    # Generate trajectory grid
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig1 = visualize_arena()
    save_path1 = '/Users/sid47/ML Algorithms/65_optimizer_arena.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_convergence()
    save_path2 = '/Users/sid47/ML Algorithms/65_optimizer_convergence.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: When to Use What")
    print("=" * 60)
    print("""
1. SGD (vanilla)
   - Simple, well-understood, strong convergence theory
   - Use when: you have lots of time and want guaranteed convergence
   - Weakness: slow in ravines, stuck at saddle points

2. SGD + Momentum (gamma=0.9)
   - The workhorse of deep learning for years
   - Use when: training CNNs, most practical training
   - Momentum carries you through flat regions and ravines

3. AdaGrad
   - Per-parameter learning rates that decay over time
   - Use when: sparse gradients (NLP embeddings, word2vec)
   - Weakness: learning rate goes to zero (dies on long training)

4. RMSProp
   - Leaky AdaGrad — uses exponential moving average
   - Use when: non-stationary problems (RL, RNNs)
   - Fixes AdaGrad's dying learning rate

5. Adam (THE DEFAULT)
   - RMSProp + Momentum + bias correction
   - Use when: starting any neural network project
   - lr=0.001, beta1=0.9, beta2=0.999 works surprisingly often
   - Warning: may generalize worse than SGD+Momentum on some tasks

6. AdamW
   - Adam with decoupled weight decay
   - Use when: training with weight decay (transformers, LLMs)
   - Standard L2 reg in Adam is broken; AdamW fixes it

RULE OF THUMB:
    Start with Adam (lr=0.001) → if it works, ship it
    If generalization matters → try SGD+Momentum with LR schedule
    Sparse features → AdaGrad
    RL / non-stationary → RMSProp or Adam

CONNECTIONS:
    63_sgd_momentum.py → foundations: SGD, momentum, Nesterov
    64_adaptive_optimizers.py → adaptive methods in detail
    12_mlp.py → where these optimizers are actually used
    """)
