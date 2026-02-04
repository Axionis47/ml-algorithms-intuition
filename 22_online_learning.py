"""
ONLINE LEARNING — Paradigm: STREAMING (Learn on the Fly)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Learn from data ONE SAMPLE AT A TIME, without storing the full dataset.

Traditional (batch) learning: See all data, then learn
Online learning: See one sample, update, forget, repeat

    For each (x_t, y_t):
        1. Make prediction ŷ_t
        2. Observe true y_t
        3. Update model (if wrong)

===============================================================
THE KEY INSIGHT: MISTAKE-DRIVEN LEARNING
===============================================================

The classic PERCEPTRON algorithm:
    If mistake (ŷ_t ≠ y_t):
        w ← w + y_t × x_t

Only update when wrong!

This is elegant:
    - No gradient computation
    - Converges if data is linearly separable
    - Each update is O(d) time and space

===============================================================
WHY ONLINE LEARNING MATTERS
===============================================================

1. STREAMING DATA: Can't store everything
   (Sensor streams, log files, market data)

2. NON-STATIONARITY: Distribution changes over time
   (Online learning adapts continuously)

3. COMPUTATIONAL EFFICIENCY: O(1) per sample
   (Batch methods: O(n) per epoch)

4. THEORETICAL GUARANTEES: Regret bounds
   (Competitive with best fixed model in hindsight)

===============================================================
REGRET: THE ONLINE LEARNING METRIC
===============================================================

Regret = Σₜ loss(model_t, x_t, y_t) - min_w Σₜ loss(w, x_t, y_t)

= Your total loss - Best fixed model's loss (in hindsight)

Good online algorithms have SUBLINEAR regret: O(√T) or O(log T)
→ Average regret → 0 as T → ∞

===============================================================
ONLINE ALGORITHMS
===============================================================

1. PERCEPTRON: Update on mistakes
   w ← w + η × y_t × x_t  (when wrong)

2. ONLINE GRADIENT DESCENT (OGD):
   w ← w - η × ∇loss(w; x_t, y_t)

3. PASSIVE-AGGRESSIVE:
   Update to make current example correct with minimum change
   w ← w + τ × y_t × x_t  where τ = loss / ||x||²

4. ONLINE SVM (Pegasos):
   Stochastic gradient descent on hinge loss

===============================================================
ADAPTIVE LEARNING RATES
===============================================================

ADAGRAD: Per-feature adaptive rates
    η_i = η / √(Σ g_i²)
    Frequently updated features → smaller steps

ADAM: Momentum + adaptive rates
    Combines gradient momentum with second moment

===============================================================
INDUCTIVE BIAS
===============================================================

1. Recency bias: recent samples matter more (with forgetting)
2. Linear models for simplicity (but can kernelize)
3. Mistake-driven: only updates when needed
4. No storage assumption: process and forget

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
accuracy = datasets_module.accuracy


def create_streaming_dataset(n_samples=1000, pattern='stationary', drift_point=500):
    """
    Create datasets for online learning.

    Patterns:
    - stationary: Fixed distribution
    - drift: Distribution changes at drift_point
    - gradual: Slow continuous drift
    - cycling: Distribution cycles between two modes
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples, dtype=int)

    if pattern == 'stationary':
        # Fixed linear boundary
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

    elif pattern == 'drift':
        # Sudden concept drift
        y[:drift_point] = (X[:drift_point, 0] + X[:drift_point, 1] > 0).astype(int)
        y[drift_point:] = (X[drift_point:, 0] - X[drift_point:, 1] > 0).astype(int)

    elif pattern == 'gradual':
        # Gradual drift: boundary rotates
        for t in range(n_samples):
            angle = np.pi * t / n_samples  # 0 to π
            w = np.array([np.cos(angle), np.sin(angle)])
            y[t] = (X[t] @ w > 0).astype(int)

    elif pattern == 'cycling':
        # Cycle between two concepts
        period = n_samples // 4
        for t in range(n_samples):
            if (t // period) % 2 == 0:
                y[t] = (X[t, 0] + X[t, 1] > 0).astype(int)
            else:
                y[t] = (X[t, 0] > 0).astype(int)

    return X, y


class Perceptron:
    """
    Classic Perceptron for online binary classification.

    Update rule: w ← w + η × y × x  (when wrong)
    """

    def __init__(self, n_features, learning_rate=1.0):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate
        self.mistakes = []

    def predict_one(self, x):
        """Predict for single sample."""
        return 1 if (x @ self.w + self.b) >= 0 else -1

    def update(self, x, y):
        """Online update for single sample."""
        y_signed = 2 * y - 1  # Convert {0,1} to {-1,+1}
        pred = self.predict_one(x)

        if pred != y_signed:
            # Mistake! Update weights
            self.w += self.lr * y_signed * x
            self.b += self.lr * y_signed
            self.mistakes.append(1)
        else:
            self.mistakes.append(0)

        return pred != y_signed  # Return whether mistake was made

    def partial_fit(self, X, y):
        """Process multiple samples online."""
        for i in range(len(y)):
            self.update(X[i], y[i])
        return self

    def predict(self, X):
        """Batch prediction."""
        scores = X @ self.w + self.b
        return (scores >= 0).astype(int)


class OnlineGradientDescent:
    """
    Online Gradient Descent for logistic regression.

    Update: w ← w - η × ∇loss
    """

    def __init__(self, n_features, learning_rate=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba_one(self, x):
        """Predict probability for single sample."""
        return self.sigmoid(x @ self.w + self.b)

    def update(self, x, y):
        """Online update using gradient descent."""
        p = self.predict_proba_one(x)

        # Log loss: -[y log(p) + (1-y) log(1-p)]
        loss = -y * np.log(p + 1e-10) - (1 - y) * np.log(1 - p + 1e-10)
        self.losses.append(loss)

        # Gradient: p - y
        grad = p - y
        self.w -= self.lr * grad * x
        self.b -= self.lr * grad

        return loss

    def partial_fit(self, X, y):
        """Process multiple samples online."""
        for i in range(len(y)):
            self.update(X[i], y[i])
        return self

    def predict(self, X):
        """Batch prediction."""
        probs = self.sigmoid(X @ self.w + self.b)
        return (probs >= 0.5).astype(int)


class PassiveAggressive:
    """
    Passive-Aggressive algorithm.

    If correct: do nothing (passive)
    If wrong: update minimally to make it correct (aggressive)

    Update: w ← w + τ × y × x
    where τ = loss / ||x||²
    """

    def __init__(self, n_features, C=1.0):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.C = C  # Aggressiveness parameter
        self.losses = []

    def predict_one(self, x):
        return 1 if (x @ self.w + self.b) >= 0 else -1

    def update(self, x, y):
        """PA update."""
        y_signed = 2 * y - 1

        # Hinge loss: max(0, 1 - y × (w·x + b))
        margin = y_signed * (x @ self.w + self.b)
        loss = max(0, 1 - margin)
        self.losses.append(loss)

        if loss > 0:
            # Compute step size
            x_norm_sq = np.sum(x ** 2) + 1  # +1 for bias
            tau = min(self.C, loss / x_norm_sq)  # PA-I variant

            # Update
            self.w += tau * y_signed * x
            self.b += tau * y_signed

        return loss

    def partial_fit(self, X, y):
        for i in range(len(y)):
            self.update(X[i], y[i])
        return self

    def predict(self, X):
        scores = X @ self.w + self.b
        return (scores >= 0).astype(int)


class OnlineAdaGrad:
    """
    AdaGrad for online learning.

    Adaptive learning rate: η_i = η / √(Σ g_i²)
    Frequently updated features get smaller learning rates.
    """

    def __init__(self, n_features, learning_rate=0.5):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate

        # Accumulated squared gradients
        self.G_w = np.zeros(n_features) + 1e-8
        self.G_b = 1e-8

        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def update(self, x, y):
        """AdaGrad update."""
        p = self.sigmoid(x @ self.w + self.b)

        loss = -y * np.log(p + 1e-10) - (1 - y) * np.log(1 - p + 1e-10)
        self.losses.append(loss)

        # Gradient
        grad_w = (p - y) * x
        grad_b = p - y

        # Accumulate squared gradients
        self.G_w += grad_w ** 2
        self.G_b += grad_b ** 2

        # Adaptive update
        self.w -= self.lr * grad_w / np.sqrt(self.G_w)
        self.b -= self.lr * grad_b / np.sqrt(self.G_b)

        return loss

    def partial_fit(self, X, y):
        for i in range(len(y)):
            self.update(X[i], y[i])
        return self

    def predict(self, X):
        probs = self.sigmoid(X @ self.w + self.b)
        return (probs >= 0.5).astype(int)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # -------- Experiment 1: Online vs Batch Learning --------
    print("\n1. ONLINE vs BATCH LEARNING (Stationary Data)")
    print("-" * 40)

    X, y = create_streaming_dataset(n_samples=1000, pattern='stationary')

    # Online Perceptron
    online_model = Perceptron(n_features=2)
    online_model.partial_fit(X, y)
    online_mistakes = sum(online_model.mistakes)
    online_acc = accuracy(y, online_model.predict(X))

    # Batch (train on all, then predict)
    batch_model = Perceptron(n_features=2)
    # Multiple passes for convergence
    for _ in range(10):
        batch_model.partial_fit(X, y)
    batch_acc = accuracy(y, batch_model.predict(X))

    print(f"Online Perceptron: {online_mistakes} mistakes, final accuracy={online_acc:.3f}")
    print(f"Batch Perceptron (10 passes): final accuracy={batch_acc:.3f}")
    print("→ Online achieves good accuracy in single pass!")

    # -------- Experiment 2: Concept Drift --------
    print("\n2. HANDLING CONCEPT DRIFT")
    print("-" * 40)
    print("Data distribution changes at t=500")

    X_drift, y_drift = create_streaming_dataset(n_samples=1000, pattern='drift')

    # Track accuracy before and after drift
    perceptron = Perceptron(n_features=2)
    ogd = OnlineGradientDescent(n_features=2, learning_rate=0.5)
    pa = PassiveAggressive(n_features=2, C=1.0)

    models = [('Perceptron', perceptron), ('OGD', ogd), ('PA', pa)]

    for name, model in models:
        mistakes_before = 0
        mistakes_after = 0

        for t in range(1000):
            x, y_true = X_drift[t], y_drift[t]

            # Predict
            if name == 'Perceptron':
                pred = 1 if model.predict_one(x) == 1 else 0
            else:
                pred = 1 if model.predict(x.reshape(1, -1))[0] == 1 else 0

            # Track mistakes
            if pred != y_true:
                if t < 500:
                    mistakes_before += 1
                else:
                    mistakes_after += 1

            # Update
            model.update(x, y_true)

        print(f"{name:<12} Before drift: {mistakes_before}/500, After: {mistakes_after}/500")
    print("→ Online methods ADAPT to drift!")

    # -------- Experiment 3: Learning Rate Effect --------
    print("\n3. EFFECT OF LEARNING RATE (OGD)")
    print("-" * 40)

    X, y = create_streaming_dataset(n_samples=500, pattern='stationary')

    for lr in [0.01, 0.1, 0.5, 1.0, 5.0]:
        ogd = OnlineGradientDescent(n_features=2, learning_rate=lr)
        ogd.partial_fit(X, y)
        final_loss = np.mean(ogd.losses[-50:])  # Average of last 50
        acc = accuracy(y, ogd.predict(X))
        print(f"lr={lr:<5} final_avg_loss={final_loss:.4f} accuracy={acc:.3f}")
    print("→ Too small: slow convergence")
    print("→ Too large: unstable")

    # -------- Experiment 4: Algorithm Comparison --------
    print("\n4. ALGORITHM COMPARISON")
    print("-" * 40)

    X, y = create_streaming_dataset(n_samples=1000, pattern='stationary')

    algorithms = {
        'Perceptron': Perceptron(2),
        'OGD': OnlineGradientDescent(2, learning_rate=0.5),
        'PA': PassiveAggressive(2, C=1.0),
        'AdaGrad': OnlineAdaGrad(2, learning_rate=0.5),
    }

    for name, model in algorithms.items():
        model.partial_fit(X, y)
        acc = accuracy(y, model.predict(X))
        if hasattr(model, 'mistakes'):
            metric = f"mistakes={sum(model.mistakes)}"
        else:
            metric = f"final_loss={np.mean(model.losses[-50:]):.4f}"
        print(f"{name:<12} accuracy={acc:.3f} {metric}")

    # -------- Experiment 5: Cumulative Mistake Rate --------
    print("\n5. CUMULATIVE MISTAKE RATE OVER TIME")
    print("-" * 40)

    X, y = create_streaming_dataset(n_samples=500, pattern='stationary')
    perceptron = Perceptron(n_features=2)

    cumulative_mistakes = []
    for t in range(500):
        mistake = perceptron.update(X[t], y[t])
        cumulative_mistakes.append(sum(perceptron.mistakes) / (t + 1))

    print(f"Mistake rate at t=50:  {cumulative_mistakes[49]:.3f}")
    print(f"Mistake rate at t=100: {cumulative_mistakes[99]:.3f}")
    print(f"Mistake rate at t=250: {cumulative_mistakes[249]:.3f}")
    print(f"Mistake rate at t=500: {cumulative_mistakes[499]:.3f}")
    print("→ Mistake rate DECREASES over time (sublinear regret)")

    # -------- Experiment 6: Gradual vs Sudden Drift --------
    print("\n6. GRADUAL vs SUDDEN CONCEPT DRIFT")
    print("-" * 40)

    for pattern in ['stationary', 'drift', 'gradual', 'cycling']:
        X, y = create_streaming_dataset(n_samples=500, pattern=pattern)
        ogd = OnlineGradientDescent(n_features=2, learning_rate=0.5)
        ogd.partial_fit(X, y)
        acc = accuracy(y, ogd.predict(X))
        print(f"pattern={pattern:<12} accuracy={acc:.3f}")
    print("→ Online learning handles all drift types!")


def visualize_online_learning():
    """Visualize online learning dynamics."""
    print("\n" + "="*60)
    print("ONLINE LEARNING VISUALIZATION")
    print("="*60)

    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Plot 1: Perceptron mistakes over time
    ax = axes[0, 0]
    X, y = create_streaming_dataset(n_samples=500, pattern='stationary')
    perceptron = Perceptron(n_features=2)
    perceptron.partial_fit(X, y)

    cumulative = np.cumsum(perceptron.mistakes)
    ax.plot(cumulative, 'b-', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cumulative Mistakes')
    ax.set_title('Perceptron Mistake Bound\n(Sublinear growth)')

    # Plot 2: OGD loss over time
    ax = axes[0, 1]
    ogd = OnlineGradientDescent(n_features=2, learning_rate=0.5)
    ogd.partial_fit(X, y)

    # Moving average
    window = 20
    smoothed = np.convolve(ogd.losses, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, 'r-', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('OGD Loss Over Time\n(Converges to optimal)')

    # Plot 3: Concept drift adaptation
    ax = axes[0, 2]
    X_drift, y_drift = create_streaming_dataset(n_samples=500, pattern='drift', drift_point=250)
    ogd_drift = OnlineGradientDescent(n_features=2, learning_rate=0.5)
    ogd_drift.partial_fit(X_drift, y_drift)

    smoothed = np.convolve(ogd_drift.losses, np.ones(20)/20, mode='valid')
    ax.plot(smoothed, 'g-', alpha=0.7)
    ax.axvline(x=250, color='red', linestyle='--', label='Drift point')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Concept Drift Adaptation\n(Spike at drift, then recovers)')
    ax.legend()

    # Plot 4: Decision boundary evolution (Perceptron)
    ax = axes[1, 0]
    X, y = create_streaming_dataset(n_samples=200, pattern='stationary')

    # Show boundary at different times
    colors = ['blue', 'green', 'orange', 'red']
    times = [10, 50, 100, 200]
    perceptron = Perceptron(n_features=2)

    for i, t in enumerate(times):
        # Train up to time t
        for j in range(len(perceptron.mistakes), t):
            perceptron.update(X[j], y[j])

        # Plot boundary
        w, b = perceptron.w, perceptron.b
        if w[1] != 0:
            x_line = np.linspace(-3, 3, 100)
            y_line = -(w[0] * x_line + b) / (w[1] + 1e-10)
            ax.plot(x_line, y_line, color=colors[i], label=f't={t}', alpha=0.7)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.3, s=20)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title('Perceptron Boundary Evolution\n(Converges to separator)')
    ax.legend(loc='upper right')

    # Plot 5: Algorithm comparison
    ax = axes[1, 1]
    X, y = create_streaming_dataset(n_samples=500, pattern='stationary')

    algorithms = {
        'OGD': OnlineGradientDescent(2, 0.5),
        'PA': PassiveAggressive(2, 1.0),
        'AdaGrad': OnlineAdaGrad(2, 0.5),
    }

    for name, model in algorithms.items():
        model.partial_fit(X, y)
        smoothed = np.convolve(model.losses, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=name, alpha=0.7)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Algorithm Comparison\n(All converge, different speeds)')
    ax.legend()

    # Plot 6: Cycling data
    ax = axes[1, 2]
    X_cycle, y_cycle = create_streaming_dataset(n_samples=500, pattern='cycling')
    ogd_cycle = OnlineGradientDescent(n_features=2, learning_rate=0.5)
    ogd_cycle.partial_fit(X_cycle, y_cycle)

    smoothed = np.convolve(ogd_cycle.losses, np.ones(20)/20, mode='valid')
    ax.plot(smoothed, 'm-', alpha=0.7)

    # Mark concept switches
    for t in [125, 250, 375]:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Cycling Concepts\n(Periodic re-adaptation)')

    plt.suptitle('ONLINE LEARNING\n'
                 'Learning from streaming data one sample at a time',
                 fontsize=12)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("ONLINE LEARNING — Learning on the Fly")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Learn from data ONE SAMPLE AT A TIME.
    For each (x_t, y_t): predict → observe → update

THE KEY INSIGHT:
    MISTAKE-DRIVEN learning.
    Perceptron: w ← w + y × x (only when wrong)

WHY IT MATTERS:
    1. Streaming data (can't store everything)
    2. Non-stationarity (distribution changes)
    3. Efficiency: O(1) per sample

ALGORITHMS:
    - Perceptron: update on mistakes
    - OGD: gradient descent per sample
    - PA: minimal update to correct mistake
    - AdaGrad: adaptive learning rates

REGRET:
    Your loss - Best fixed model's loss
    Good algorithms: sublinear regret (O(√T))
    """)

    ablation_experiments()

    fig = visualize_online_learning()
    save_path = '/Users/sid47/ML Algorithms/22_online_learning.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Process one sample at a time, then forget
2. Perceptron: update only on mistakes
3. OGD: stochastic gradient descent
4. Sublinear regret: converges to optimal
5. Naturally handles concept drift
6. Essential for streaming/non-stationary data
    """)
