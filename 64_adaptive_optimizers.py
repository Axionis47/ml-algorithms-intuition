"""
ADAPTIVE OPTIMIZERS — Paradigm: ADAPTIVE LEARNING RATES

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

One learning rate for ALL parameters is a bad idea.

Consider a neural network with:
- Parameters that get frequent gradients (common features)
- Parameters that get rare gradients (rare features)

If you use the same learning rate:
- Frequent parameters: update too much → overshoot
- Rare parameters: update too little → never learn

SOLUTION: Give each parameter its OWN effective learning rate,
          based on the HISTORY of its gradients.

The progression:
    SGD → AdaGrad → RMSprop → Adam → AdamW
    (one lr)  (accumulate)  (forget)  (momentum   (fix weight
                                       + adapt)     decay)

===============================================================
THE MATHEMATICS
===============================================================

ADAGRAD (Adaptive Gradient):
    G_t = G_{t-1} + g_t^2              (accumulate squared gradients)
    theta_t = theta_{t-1} - (lr / sqrt(G_t + eps)) * g_t

    Each parameter divides by sqrt(sum of all past squared gradients).
    Rare gradients → small G → large effective lr
    Frequent gradients → large G → small effective lr

    PROBLEM: G only grows → effective lr → 0 (learning dies!)

RMSPROP (Root Mean Square Propagation):
    G_t = beta * G_{t-1} + (1 - beta) * g_t^2    (exponential moving average)
    theta_t = theta_{t-1} - (lr / sqrt(G_t + eps)) * g_t

    FIX: Use exponential moving average instead of sum.
    Old gradients are FORGOTTEN → learning rate doesn't die.
    beta = 0.9 means "remember last ~10 gradients worth"

ADAM (Adaptive Moment Estimation):
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t     (1st moment: mean)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2   (2nd moment: variance)
    m_hat = m_t / (1 - beta1^t)                    (bias correction!)
    v_hat = v_t / (1 - beta2^t)                    (bias correction!)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    WHY BIAS CORRECTION?
    m and v are initialized to 0. In early steps:
    m_1 = (1 - beta1) * g_1  ← way too small! (biased toward 0)
    Dividing by (1 - beta1^t) fixes this.
    At t=1 with beta1=0.9: divide by 0.1 → multiply by 10!

    Adam = RMSprop + Momentum + Bias Correction.
    The "default optimizer" for deep learning since 2015.

ADAMW (Adam with Decoupled Weight Decay):
    Same as Adam, but weight decay is SEPARATED from gradient:

    Adam (L2 reg):  g_t = nabla L + lambda * theta  (added to gradient)
    AdamW:          theta_t -= lr * (m_hat / (sqrt(v_hat) + eps)
                                     + weight_decay * theta_t)

    WHY DOES THIS MATTER?
    In Adam, the adaptive scaling ALSO scales the weight decay.
    Parameters with large gradients get less regularization!
    AdamW fixes this by decoupling — every parameter gets equal decay.

===============================================================
INDUCTIVE BIAS
===============================================================

1. GRADIENT MAGNITUDE IS INFORMATIVE
   Parameters with large gradients should take smaller steps
   (they're already learning fast, don't overshoot)

2. GRADIENT HISTORY MATTERS
   Past gradients tell us about loss surface curvature
   High variance → small steps, low variance → large steps

3. EXPONENTIAL FORGETTING (RMSprop, Adam)
   Recent gradients matter more than ancient ones
   Loss surface changes as we train (non-stationary)

4. MOMENTUM HELPS (Adam)
   Smoothing gradient direction reduces noise
   Same principle as SGD momentum, but per-parameter

5. WEIGHT DECAY != L2 REGULARIZATION (AdamW)
   With adaptive methods, L2 and weight decay are NOT equivalent
   Decoupled decay is theoretically cleaner and works better

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
sgd_module = import_module('63_sgd_momentum')

# Import from 63_sgd_momentum.py
SGD = sgd_module.SGD
rosenbrock = sgd_module.rosenbrock
rosenbrock_grad = sgd_module.rosenbrock_grad
beale = sgd_module.beale
beale_grad = sgd_module.beale_grad
optimize = sgd_module.optimize


# ============================================================
# OPTIMIZERS
# ============================================================

class AdaGrad:
    """
    Adaptive Gradient (Duchi et al., 2011).

    Accumulates squared gradients per parameter.
    Rare parameters get larger updates, frequent ones get smaller.
    PROBLEM: Effective learning rate monotonically decreases to zero.
    """

    def __init__(self, lr=0.01, eps=1e-8):
        """
        Parameters:
        -----------
        lr : float
            Base learning rate.
        eps : float
            Small constant for numerical stability.
        """
        self.lr = lr
        self.eps = eps
        self.cache = {}  # Accumulated squared gradients

    def step(self, params, grads):
        """
        One optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array

        Returns:
            Updated params dict
        """
        for name in params:
            if name not in self.cache:
                self.cache[name] = np.zeros_like(params[name])

            # Accumulate squared gradients (only grows!)
            self.cache[name] += grads[name] ** 2

            # Update: lr / sqrt(G + eps) * grad
            params[name] = params[name] - (self.lr / (np.sqrt(self.cache[name]) + self.eps)) * grads[name]

        return params

    def reset(self):
        """Reset accumulated gradients."""
        self.cache = {}


class RMSprop:
    """
    Root Mean Square Propagation (Hinton, 2012).

    Fixes AdaGrad's dying learning rate with exponential moving average.
    beta controls forgetting: 0.9 = remember ~10 steps, 0.99 = ~100 steps.
    """

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        """
        Parameters:
        -----------
        lr : float
            Base learning rate.
        beta : float
            Decay rate for moving average of squared gradients.
        eps : float
            Small constant for numerical stability.
        """
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}  # Running average of squared gradients

    def step(self, params, grads):
        """
        One optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array

        Returns:
            Updated params dict
        """
        for name in params:
            if name not in self.cache:
                self.cache[name] = np.zeros_like(params[name])

            # Exponential moving average of squared gradients
            self.cache[name] = self.beta * self.cache[name] + (1 - self.beta) * grads[name] ** 2

            # Update: lr / sqrt(EMA + eps) * grad
            params[name] = params[name] - (self.lr / (np.sqrt(self.cache[name]) + self.eps)) * grads[name]

        return params

    def reset(self):
        """Reset running averages."""
        self.cache = {}


class Adam:
    """
    Adaptive Moment Estimation (Kingma & Ba, 2015).

    Combines:
    - 1st moment (mean of gradients) = momentum
    - 2nd moment (mean of squared gradients) = adaptive rate
    - Bias correction for both

    The default optimizer for deep learning.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Parameters:
        -----------
        lr : float
            Base learning rate.
        beta1 : float
            Decay rate for 1st moment (momentum). Typical: 0.9.
        beta2 : float
            Decay rate for 2nd moment (adaptive rate). Typical: 0.999.
        eps : float
            Small constant for numerical stability.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 1st moment (mean)
        self.v = {}  # 2nd moment (variance)
        self.t = 0   # Timestep (for bias correction)

    def step(self, params, grads):
        """
        One optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array

        Returns:
            Updated params dict
        """
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update biased 1st moment (momentum)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]

            # Update biased 2nd moment (adaptive rate)
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[name] = params[name] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params

    def reset(self):
        """Reset moments and timestep."""
        self.m = {}
        self.v = {}
        self.t = 0


class AdamW:
    """
    Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2019).

    In standard Adam + L2: weight decay is scaled by adaptive rate.
    In AdamW: weight decay is applied DIRECTLY, not through gradient.

    This matters because:
    - Adam divides gradient by sqrt(v), including L2 term
    - Parameters with large gradients get LESS regularization
    - AdamW ensures equal regularization regardless of gradient scale
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        """
        Parameters:
        -----------
        lr : float
            Base learning rate.
        beta1 : float
            Decay rate for 1st moment.
        beta2 : float
            Decay rate for 2nd moment.
        eps : float
            Small constant for numerical stability.
        weight_decay : float
            Decoupled weight decay coefficient.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        """
        One optimization step.

        Args:
            params: dict of parameter name -> numpy array
            grads: dict of parameter name -> gradient array

        Returns:
            Updated params dict
        """
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update biased 1st moment
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]

            # Update biased 2nd moment
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # DECOUPLED weight decay: shrink params THEN apply Adam step
            # This is the key insight of Loshchilov & Hutter 2019:
            # Standard L2 adds lambda*theta to gradient BEFORE adaptive scaling,
            # but AdamW applies decay DIRECTLY to parameters (uniform shrinkage).
            params[name] = (params[name] * (1 - self.lr * self.weight_decay) -
                            self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

        return params

    def reset(self):
        """Reset moments and timestep."""
        self.m = {}
        self.v = {}
        self.t = 0


# ============================================================
# SIMPLE NEURAL NETWORK (for optimizer comparison)
# ============================================================

def make_moons(n_samples=300, noise=0.2, random_state=42):
    """Generate two interleaving half circles (moons dataset)."""
    rng = np.random.RandomState(random_state)
    n_per_moon = n_samples // 2

    # Upper moon
    theta1 = np.linspace(0, np.pi, n_per_moon)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])

    # Lower moon (shifted)
    theta2 = np.linspace(0, np.pi, n_per_moon)
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    X = np.vstack([x1, x2]) + rng.randn(n_samples, 2) * noise
    y = np.hstack([np.zeros(n_per_moon), np.ones(n_per_moon)])
    return X, y


def train_tiny_mlp(X, y, optimizer, n_epochs=200, hidden_size=16, random_state=42):
    """
    Train a 2-layer MLP (2D -> hidden -> 1) with manual backprop.

    Architecture:
        Input(2) -> Linear(hidden) -> ReLU -> Linear(1) -> Sigmoid

    Loss: Binary cross-entropy

    Returns loss curve.
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

    for epoch in range(n_epochs):
        # ---- Forward pass ----
        z1 = X @ params['W1'] + params['b1']           # (N, hidden)
        a1 = np.maximum(0, z1)                           # ReLU
        z2 = a1 @ params['W2'] + params['b2']           # (N, 1)
        y_pred = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))  # Sigmoid

        # ---- Loss (binary cross-entropy) ----
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        y_true = y.reshape(-1, 1)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        losses.append(loss)

        # ---- Backward pass ----
        # dL/dz2
        dz2 = (y_pred_clipped - y_true) / n_samples     # (N, 1)

        # Gradients for W2, b2
        dW2 = a1.T @ dz2                                # (hidden, 1)
        db2 = np.sum(dz2, axis=0)                        # (1,)

        # dL/da1
        da1 = dz2 @ params['W2'].T                      # (N, hidden)

        # ReLU backward
        dz1 = da1 * (z1 > 0).astype(float)              # (N, hidden)

        # Gradients for W1, b1
        dW1 = X.T @ dz1                                  # (2, hidden)
        db1 = np.sum(dz1, axis=0)                        # (hidden,)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        # Clip gradients for stability
        for name in grads:
            grads[name] = np.clip(grads[name], -5.0, 5.0)

        params = optimizer.step(params, grads)

    return losses


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

    # -------- Experiment 1: AdaGrad Learning Rate Death --------
    print("\n1. ADAGRAD LEARNING RATE ACCUMULATION")
    print("-" * 40)
    print("Track effective lr per parameter on Rosenbrock:")

    start = [-1.0, 1.0]
    opt = AdaGrad(lr=0.5)
    params = {'xy': np.array(start, dtype=float)}

    print(f"   {'Step':<8} {'eff_lr_x':<14} {'eff_lr_y':<14} {'loss':<12}")
    for step in range(1, 501):
        g = rosenbrock_grad(params['xy'])
        g = np.clip(g, -10.0, 10.0)

        # Track effective learning rate BEFORE step
        if 'xy' in opt.cache:
            eff_lr = opt.lr / (np.sqrt(opt.cache['xy']) + opt.eps)
        else:
            eff_lr = np.array([opt.lr, opt.lr])

        grads = {'xy': g}
        params = opt.step(params, grads)
        loss = rosenbrock(params['xy'])

        if step in [1, 10, 50, 100, 200, 500]:
            print(f"   {step:<8} {eff_lr[0]:<14.6f} {eff_lr[1]:<14.6f} {loss:<12.4f}")

    print("→ Effective learning rate MONOTONICALLY DECREASES")
    print("→ After 500 steps, lr is nearly dead — AdaGrad stops learning")
    print("→ This is why RMSprop was invented: use EMA to forget old gradients")

    # -------- Experiment 2: RMSprop Beta Effect --------
    print("\n2. RMSPROP BETA EFFECT (FORGETTING FACTOR)")
    print("-" * 40)
    print("Beta controls how fast we forget old gradients:")
    print("  beta=0.5 → remember ~2 steps")
    print("  beta=0.9 → remember ~10 steps")
    print("  beta=0.99 → remember ~100 steps")

    start = [-1.0, 1.0]
    for beta in [0.5, 0.9, 0.99]:
        opt = RMSprop(lr=0.001, beta=beta)
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                                n_steps=1000, clip_grad=5.0)
        final = traj[-1]
        print(f"   beta={beta:<6}  final_loss={losses[-1]:.4f}  "
              f"pos=({final[0]:.3f}, {final[1]:.3f})")

    print("→ beta=0.5: forgets too fast, noisy effective lr")
    print("→ beta=0.9: good balance (the standard default)")
    print("→ beta=0.99: smooth but slow to adapt to new curvature")

    # -------- Experiment 3: Adam Bias Correction --------
    print("\n3. ADAM BIAS CORRECTION: WITH vs WITHOUT")
    print("-" * 40)
    print("Bias correction matters most in the first ~10 steps:")
    print("At t=1 with beta2=0.999: v_hat = v / (1 - 0.999) = v / 0.001 = 1000 * v")
    print("Without correction, first steps are WAY too large!\n")

    start = [-1.0, 1.0]

    # Adam WITH bias correction (standard)
    opt_corrected = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    params_c = {'xy': np.array(start, dtype=float)}
    losses_c = [rosenbrock(params_c['xy'])]

    # Adam WITHOUT bias correction (manually)
    m_nc = np.zeros(2)
    v_nc = np.zeros(2)
    params_nc = np.array(start, dtype=float)
    losses_nc = [rosenbrock(params_nc)]

    print(f"   {'Step':<8} {'Loss (corrected)':<20} {'Loss (uncorrected)':<20}")
    for step in range(1, 201):
        # Corrected Adam
        g = rosenbrock_grad(params_c['xy'])
        g = np.clip(g, -10.0, 10.0)
        params_c = opt_corrected.step(params_c, {'xy': g})
        losses_c.append(rosenbrock(params_c['xy']))

        # Uncorrected Adam (manual)
        g_nc = rosenbrock_grad(params_nc)
        g_nc = np.clip(g_nc, -10.0, 10.0)
        m_nc = 0.9 * m_nc + 0.1 * g_nc
        v_nc = 0.999 * v_nc + 0.001 * g_nc ** 2
        # NO bias correction: use m and v directly
        params_nc = params_nc - 0.01 * m_nc / (np.sqrt(v_nc) + 1e-8)
        losses_nc.append(rosenbrock(params_nc))

        if step in [1, 5, 10, 50, 100, 200]:
            print(f"   {step:<8} {losses_c[-1]:<20.4f} {losses_nc[-1]:<20.4f}")

    print("→ Without correction, early steps have tiny m and v (biased toward 0)")
    print("→ This makes early updates too small OR too large depending on lr")
    print("→ Bias correction ensures consistent behavior from step 1")

    # -------- Experiment 4: Adam vs AdamW --------
    print("\n4. ADAM (L2) vs ADAMW (DECOUPLED WEIGHT DECAY)")
    print("-" * 40)
    print("On a regularized problem where weight magnitude matters:")

    # Create a problem where regularization matters:
    # Minimize f(x,y) = rosenbrock(x,y) + lambda * (x^2 + y^2)
    lam = 0.1

    def reg_loss(xy):
        return rosenbrock(xy) + lam * np.sum(xy ** 2)

    def reg_grad(xy):
        return rosenbrock_grad(xy) + 2 * lam * xy

    start = [-1.0, 2.0]

    # Adam with L2 in gradient (standard Adam + L2 regularization)
    opt_adam_l2 = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    traj_l2, losses_l2 = optimize(reg_loss, reg_grad, start, opt_adam_l2,
                                   n_steps=1000, clip_grad=5.0)

    # AdamW with decoupled weight decay
    opt_adamw = AdamW(lr=0.01, beta1=0.9, beta2=0.999, weight_decay=2*lam)
    traj_w, losses_w = optimize(rosenbrock, rosenbrock_grad, start, opt_adamw,
                                 n_steps=1000, clip_grad=5.0)

    print(f"   Adam+L2:  final_loss={losses_l2[-1]:.4f}  "
          f"pos=({traj_l2[-1][0]:.3f}, {traj_l2[-1][1]:.3f})  "
          f"weight_norm={np.linalg.norm(traj_l2[-1]):.3f}")
    print(f"   AdamW:    final_loss={losses_w[-1]:.4f}  "
          f"pos=({traj_w[-1][0]:.3f}, {traj_w[-1][1]:.3f})  "
          f"weight_norm={np.linalg.norm(traj_w[-1]):.3f}")

    print("→ Adam+L2: weight decay is scaled by adaptive rate (inconsistent)")
    print("→ AdamW: every parameter gets equal decay (theoretically cleaner)")
    print("→ In practice, AdamW often generalizes better on neural networks")

    # -------- Experiment 5: All Optimizers Comparison --------
    print("\n5. ALL OPTIMIZERS ON ROSENBROCK")
    print("-" * 40)
    print("Starting at (-1, 1), 2000 steps:")

    start = [-1.0, 1.0]
    configs = [
        ('SGD (lr=0.0002)',         SGD(lr=0.0002)),
        ('SGD+Mom (lr=0.0002)',     SGD(lr=0.0002, momentum=0.9)),
        ('AdaGrad (lr=0.1)',        AdaGrad(lr=0.1)),
        ('RMSprop (lr=0.001)',      RMSprop(lr=0.001)),
        ('Adam (lr=0.01)',          Adam(lr=0.01)),
        ('AdamW (lr=0.01, wd=0.01)', AdamW(lr=0.01, weight_decay=0.01)),
    ]

    for name, opt in configs:
        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                                n_steps=2000, clip_grad=5.0)
        final = traj[-1]
        print(f"   {name:<28}  final_loss={losses[-1]:.6f}  "
              f"pos=({final[0]:.4f}, {final[1]:.4f})")

    print("→ SGD: slowest, but most predictable")
    print("→ AdaGrad: starts fast, then stalls (dying lr)")
    print("→ RMSprop: good adaptive behavior")
    print("→ Adam: fast + stable (the default choice)")
    print("→ AdamW: similar to Adam but with explicit weight shrinkage")


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_trajectories():
    """
    Same loss surface (Rosenbrock), overlaid trajectories for
    AdaGrad, RMSprop, Adam, and SGD+Momentum.

    THE KEY INSIGHT: Different optimizers navigate the curved
    valley in fundamentally different ways.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create contour plot data
    x_range = np.linspace(-2, 2, 300)
    y_range = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    start = [-1.5, 2.0]
    configs = [
        ('SGD + Momentum\n(lr=0.0002, mom=0.9)', SGD(lr=0.0002, momentum=0.9), 'red'),
        ('AdaGrad\n(lr=0.1)', AdaGrad(lr=0.1), 'blue'),
        ('RMSprop\n(lr=0.001, beta=0.9)', RMSprop(lr=0.001, beta=0.9), 'green'),
        ('Adam\n(lr=0.01, beta1=0.9, beta2=0.999)', Adam(lr=0.01, beta1=0.9, beta2=0.999), 'orange'),
    ]

    for ax, (name, opt, color) in zip(axes.flat, configs):
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.6)
        ax.contourf(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.15)

        traj, losses = optimize(rosenbrock, rosenbrock_grad, start, opt,
                                n_steps=2000, clip_grad=5.0)

        ax.plot(traj[:, 0], traj[:, 1], '.-', color=color, linewidth=1.2,
                markersize=1, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start', zorder=5)
        ax.plot(1, 1, 'r*', markersize=15, label='Global min (1,1)', zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 'bs', markersize=8, label='End', zorder=5)

        ax.set_title(f'{name}\nFinal loss: {losses[-1]:.2f}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')

    plt.suptitle('ROSENBROCK: How Different Optimizers Navigate the Valley\n'
                 'f(x,y) = (1-x)^2 + 100(y-x^2)^2 | Start: (-1.5, 2.0) | 2000 steps',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_lr_evolution():
    """
    Show how effective learning rate per-parameter evolves over time
    for each optimizer.

    THE KEY INSIGHT: AdaGrad's lr dies, RMSprop's stabilizes,
    Adam's is smooth and corrected from step 1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    start = [-1.0, 1.0]
    n_steps = 500

    # --- AdaGrad ---
    ax = axes[0, 0]
    opt = AdaGrad(lr=0.5)
    params = {'xy': np.array(start, dtype=float)}
    eff_lr_x, eff_lr_y = [], []

    for step in range(n_steps):
        g = rosenbrock_grad(params['xy'])
        g = np.clip(g, -10.0, 10.0)
        params = opt.step(params, {'xy': g})
        eff_x = opt.lr / (np.sqrt(opt.cache['xy'][0]) + opt.eps)
        eff_y = opt.lr / (np.sqrt(opt.cache['xy'][1]) + opt.eps)
        eff_lr_x.append(eff_x)
        eff_lr_y.append(eff_y)

    ax.semilogy(eff_lr_x, label='x parameter', linewidth=2, color='steelblue')
    ax.semilogy(eff_lr_y, label='y parameter', linewidth=2, color='coral')
    ax.set_title('AdaGrad: Learning Rate DIES\n(accumulates forever)', fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective lr (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- RMSprop ---
    ax = axes[0, 1]
    opt = RMSprop(lr=0.001, beta=0.9)
    params = {'xy': np.array(start, dtype=float)}
    eff_lr_x, eff_lr_y = [], []

    for step in range(n_steps):
        g = rosenbrock_grad(params['xy'])
        g = np.clip(g, -10.0, 10.0)
        params = opt.step(params, {'xy': g})
        eff_x = opt.lr / (np.sqrt(opt.cache['xy'][0]) + opt.eps)
        eff_y = opt.lr / (np.sqrt(opt.cache['xy'][1]) + opt.eps)
        eff_lr_x.append(eff_x)
        eff_lr_y.append(eff_y)

    ax.semilogy(eff_lr_x, label='x parameter', linewidth=2, color='steelblue')
    ax.semilogy(eff_lr_y, label='y parameter', linewidth=2, color='coral')
    ax.set_title('RMSprop: Learning Rate STABILIZES\n(exponential forgetting)', fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective lr (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Adam (corrected) ---
    ax = axes[1, 0]
    opt = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    params = {'xy': np.array(start, dtype=float)}
    eff_lr_x, eff_lr_y = [], []

    for step in range(n_steps):
        g = rosenbrock_grad(params['xy'])
        g = np.clip(g, -10.0, 10.0)
        params = opt.step(params, {'xy': g})

        # Effective lr for Adam: lr / (sqrt(v_hat) + eps)
        v_hat = opt.v['xy'] / (1 - opt.beta2 ** opt.t)
        eff_x = opt.lr / (np.sqrt(v_hat[0]) + opt.eps)
        eff_y = opt.lr / (np.sqrt(v_hat[1]) + opt.eps)
        eff_lr_x.append(eff_x)
        eff_lr_y.append(eff_y)

    ax.semilogy(eff_lr_x, label='x parameter', linewidth=2, color='steelblue')
    ax.semilogy(eff_lr_y, label='y parameter', linewidth=2, color='coral')
    ax.set_title('Adam: Smooth + Bias Corrected\n(momentum + adaptive rate)', fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective lr (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Comparison overlay ---
    ax = axes[1, 1]

    # Re-run each and track mean effective lr
    optimizers_to_compare = [
        ('AdaGrad', AdaGrad(lr=0.5)),
        ('RMSprop', RMSprop(lr=0.001, beta=0.9)),
        ('Adam', Adam(lr=0.01)),
    ]

    for opt_name, opt in optimizers_to_compare:
        params = {'xy': np.array(start, dtype=float)}
        mean_eff_lr = []

        for step in range(n_steps):
            g = rosenbrock_grad(params['xy'])
            g = np.clip(g, -10.0, 10.0)
            params = opt.step(params, {'xy': g})

            if opt_name == 'AdaGrad':
                eff = opt.lr / (np.sqrt(opt.cache['xy']) + opt.eps)
            elif opt_name == 'RMSprop':
                eff = opt.lr / (np.sqrt(opt.cache['xy']) + opt.eps)
            else:  # Adam
                v_hat = opt.v['xy'] / (1 - opt.beta2 ** opt.t)
                eff = opt.lr / (np.sqrt(v_hat) + opt.eps)

            mean_eff_lr.append(np.mean(eff))

        ax.semilogy(mean_eff_lr, label=opt_name, linewidth=2)

    ax.set_title('All Optimizers: Mean Effective LR\n(AdaGrad dies, others adapt)', fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean effective lr (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('HOW EFFECTIVE LEARNING RATE EVOLVES PER PARAMETER\n'
                 'AdaGrad accumulates forever (dies) | RMSprop forgets (stabilizes) | '
                 'Adam: smooth + corrected',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_neural_net():
    """
    Simple 2-layer neural net on moons dataset,
    training curves for each optimizer.

    THE KEY INSIGHT: Adam converges faster and more reliably than
    SGD on non-convex neural network loss surfaces.
    """
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Training loss curves ---
    ax = axes[0]
    optimizers = [
        ('SGD (lr=0.1)',           SGD(lr=0.1)),
        ('SGD+Mom (lr=0.1)',       SGD(lr=0.1, momentum=0.9)),
        ('AdaGrad (lr=0.1)',       AdaGrad(lr=0.1)),
        ('RMSprop (lr=0.01)',      RMSprop(lr=0.01)),
        ('Adam (lr=0.01)',         Adam(lr=0.01)),
        ('AdamW (lr=0.01)',        AdamW(lr=0.01, weight_decay=0.01)),
    ]
    colors = ['gray', 'black', 'blue', 'green', 'orange', 'red']

    all_losses = {}
    for (name, opt), color in zip(optimizers, colors):
        losses = train_tiny_mlp(X, y, opt, n_epochs=300, hidden_size=16, random_state=42)
        all_losses[name] = losses
        ax.plot(losses, label=name, linewidth=1.8, color=color, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Cross-Entropy Loss')
    ax.set_title('Training Loss on Moons Dataset\n(2-layer MLP, hidden=16)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: Final loss comparison (bar chart) ---
    ax = axes[1]
    names = [n for n, _ in optimizers]
    final_losses = [all_losses[n][-1] for n in names]
    bar_colors = colors

    bars = ax.barh(range(len(names)), final_losses, color=bar_colors, alpha=0.75)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Loss (lower is better)')
    ax.set_title('Final Loss After 300 Epochs', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add loss values on bars
    for i, (bar, val) in enumerate(zip(bars, final_losses)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=8)

    # --- Panel 3: Early convergence (first 50 epochs) ---
    ax = axes[2]
    for (name, _), color in zip(optimizers, colors):
        losses = all_losses[name][:50]
        ax.plot(losses, label=name, linewidth=1.8, color=color, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Early Training (First 50 Epochs)\nAdaptive methods converge faster', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('NEURAL NETWORK OPTIMIZER COMPARISON\n'
                 'Moons dataset | 2-layer MLP (2 -> 16 -> 1) | Binary cross-entropy',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("ADAPTIVE OPTIMIZERS — Paradigm: ADAPTIVE LEARNING RATES")
    print("="*60)

    print("""
WHAT THIS IS:
    Different parameters need different learning rates.
    Adaptive methods learn the right rate from gradient history.

KEY EQUATIONS:
    AdaGrad:  G += g^2; theta -= lr/sqrt(G+eps) * g
    RMSprop:  G = beta*G + (1-beta)*g^2; theta -= lr/sqrt(G+eps) * g
    Adam:     m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2
              m_hat = m/(1-beta1^t); v_hat = v/(1-beta2^t)
              theta -= lr * m_hat / (sqrt(v_hat) + eps)
    AdamW:    theta -= lr * (m_hat/(sqrt(v_hat)+eps) + wd * theta)

INDUCTIVE BIAS:
    - GRADIENT MAGNITUDE tells us about curvature
    - RECENT gradients matter more than old ones
    - MOMENTUM smooths gradient direction
    - WEIGHT DECAY != L2 regularization in adaptive methods
    """)

    # Run ablations
    ablation_experiments()

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig1 = visualize_trajectories()
    save_path1 = '/Users/sid47/ML Algorithms/64_adaptive_trajectories.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    fig2 = visualize_lr_evolution()
    save_path2 = '/Users/sid47/ML Algorithms/64_adaptive_evolution.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    fig3 = visualize_neural_net()
    save_path3 = '/Users/sid47/ML Algorithms/64_adaptive_neural_net.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Adaptive Optimizers Reveal")
    print("="*60)
    print("""
1. ADAGRAD: First adaptive method
   → Per-parameter learning rates from gradient history
   → FATAL FLAW: accumulated G only grows, lr dies to zero
   → Good for sparse data (NLP embeddings), bad for deep nets

2. RMSPROP: Fixed AdaGrad with forgetting
   → Exponential moving average of squared gradients
   → beta=0.9 forgets gradients ~10 steps old
   → Simple and effective, still widely used

3. ADAM: The default optimizer
   → Momentum (1st moment) + adaptive rate (2nd moment)
   → Bias correction is critical for first ~10 steps
   → Works well out-of-the-box on almost everything

4. ADAMW: The CORRECT way to regularize with Adam
   → L2 regularization in Adam is broken (scaled by adaptive rate)
   → Decoupled weight decay fixes this
   → Used in all modern large model training (GPT, BERT, etc.)

5. THE PROGRESSION MAKES SENSE:
   SGD → needs per-param lr → AdaGrad
   AdaGrad → lr dies → RMSprop (forget old grads)
   RMSprop → add momentum → Adam
   Adam → fix weight decay → AdamW

CONNECTIONS:
    -> 12_mlp.py (line 320): uses vanilla SGD — now you see why
       adaptive methods would train the MLP much faster
    -> 45_diffusion_fundamentals.py: uses Adam for training the
       denoising network — adaptive rates handle varying gradients
       across diffusion timesteps
    -> 63_sgd_momentum.py: the foundation this builds on — SGD
       provides gradient direction, adaptive methods add per-parameter
       scaling on top of that same principle
    """)
