"""
CALIBRATION METHODS — Paradigm: UNCERTAINTY (Making Probabilities Honest)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A classifier is CALIBRATED if:
    P(correct | predicted probability p) = p

If model says "80% confident", it should be right 80% of the time.

Modern neural networks are often MISCALIBRATED:
    - Overconfident: say 95% but only right 70%
    - Worse with depth (more parameters = more overconfidence)

Calibration methods POST-PROCESS predictions to fix this.

===============================================================
THE KEY INSIGHT: CALIBRATION ≠ ACCURACY
===============================================================

A model can have HIGH ACCURACY but be POORLY CALIBRATED:
    - Always predict 99% confident
    - Be right 90% of the time
    - Accuracy: 90% ✓
    - Calibration: Terrible ✗

Calibration is about making PROBABILITIES trustworthy.

===============================================================
CALIBRATION METHODS
===============================================================

1. PLATT SCALING:
   Fit a logistic regression to transform logits:
   p_calibrated = σ(a × logit + b)
   Learn a, b on validation set.

2. TEMPERATURE SCALING:
   Special case: p_calibrated = softmax(logits / T)
   Learn single temperature T on validation set.
   T > 1: softer predictions (less confident)
   T < 1: sharper predictions (more confident)

3. ISOTONIC REGRESSION:
   Non-parametric: fit monotonic function
   p_calibrated = isotonic_fit(p_uncalibrated)
   Most flexible, but needs more data.

===============================================================
MEASURING CALIBRATION
===============================================================

EXPECTED CALIBRATION ERROR (ECE):
    Partition predictions into bins by confidence
    ECE = Σ_b (|B_b|/n) × |accuracy(B_b) - confidence(B_b)|

    Perfect calibration: ECE = 0

RELIABILITY DIAGRAM:
    Plot accuracy vs confidence for each bin
    Perfect: diagonal line

===============================================================
WHY MODERN NNs ARE MISCALIBRATED
===============================================================

1. Increased model capacity → overfitting to training confidence
2. Batch normalization changes → worse on test
3. Weight decay regularization → changes calibration
4. Cross-entropy loss → only cares about correct class

Temperature scaling fixes most miscalibration with SINGLE parameter!

===============================================================
INDUCTIVE BIAS
===============================================================

1. Platt scaling: logistic transformation is sufficient
2. Temperature: uniform scaling is sufficient
3. Isotonic: only assumes monotonicity
4. All assume held-out validation set is representative

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
accuracy = datasets_module.accuracy


class SimpleMLP:
    """Simple MLP for calibration experiments."""

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X, return_logits=False):
        current = X
        for i in range(self.n_layers - 1):
            current = np.maximum(0, current @ self.weights[i] + self.biases[i])
        logits = current @ self.weights[-1] + self.biases[-1]

        if return_logits:
            return logits

        return self.softmax(logits)

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def fit(self, X, y, epochs=200, lr=0.1):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        for epoch in range(epochs):
            # Forward
            logits = self.forward(X, return_logits=True)
            probs = self.softmax(logits)

            # One-hot
            y_onehot = np.zeros((n_samples, n_classes))
            y_onehot[np.arange(n_samples), y] = 1

            # Backprop (simplified)
            delta = (probs - y_onehot) / n_samples

            # Update output layer
            activations = [X]
            current = X
            for i in range(self.n_layers - 1):
                current = np.maximum(0, current @ self.weights[i] + self.biases[i])
                activations.append(current)

            for i in range(self.n_layers - 1, -1, -1):
                self.weights[i] -= lr * (activations[i].T @ delta)
                self.biases[i] -= lr * np.sum(delta, axis=0)
                if i > 0:
                    delta = (delta @ self.weights[i].T) * (activations[i] > 0)

        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict_logits(self, X):
        return self.forward(X, return_logits=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error.

    ECE = Σ_b (|B_b|/n) × |accuracy(B_b) - confidence(B_b)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correct[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def compute_reliability_diagram(probs, labels, n_bins=10):
    """Compute data for reliability diagram."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(np.mean(correct[in_bin]))
            bin_counts.append(np.sum(in_bin))

    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


class TemperatureScaling:
    """
    Temperature Scaling calibration.

    Learns a single temperature T to scale logits:
    p_calibrated = softmax(logits / T)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Learn optimal temperature on validation set."""
        n_samples = len(labels)
        n_classes = logits.shape[1]

        # One-hot encode labels
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), labels] = 1

        # Optimize temperature using gradient descent on NLL
        T = 1.0
        for _ in range(max_iter):
            # Forward: softmax(logits / T)
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)

            # NLL loss
            nll = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))

            # Gradient of NLL w.r.t. T
            # d(softmax(x/T))/dT = -1/T² × (softmax - softmax × logits/T)
            # Simplified: use numerical gradient
            eps = 0.001
            scaled_logits_plus = logits / (T + eps)
            probs_plus = self._softmax(scaled_logits_plus)
            nll_plus = -np.mean(np.sum(y_onehot * np.log(probs_plus + 1e-10), axis=1))

            grad = (nll_plus - nll) / eps
            T -= lr * grad
            T = max(0.01, T)  # Keep positive

        self.temperature = T
        return self

    def _softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def calibrate(self, logits):
        """Apply temperature scaling."""
        return self._softmax(logits / self.temperature)


class PlattScaling:
    """
    Platt Scaling calibration.

    Fits logistic regression: p_calibrated = σ(a × logit + b)
    """

    def __init__(self):
        self.a = None  # Shape: (n_classes,)
        self.b = None  # Shape: (n_classes,)

    def fit(self, logits, labels, lr=0.1, max_iter=200):
        """Learn Platt scaling parameters."""
        n_samples, n_classes = logits.shape

        # Initialize parameters
        self.a = np.ones(n_classes)
        self.b = np.zeros(n_classes)

        # One-hot encode labels
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), labels] = 1

        # Optimize using gradient descent
        for _ in range(max_iter):
            # Forward: softmax(a * logits + b)
            scaled_logits = logits * self.a + self.b
            probs = self._softmax(scaled_logits)

            # Gradient of cross-entropy
            grad = probs - y_onehot

            # Update a and b
            self.a -= lr * np.mean(grad * logits, axis=0)
            self.b -= lr * np.mean(grad, axis=0)

        return self

    def _softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def calibrate(self, logits):
        """Apply Platt scaling."""
        return self._softmax(logits * self.a + self.b)


class IsotonicCalibration:
    """
    Isotonic Regression calibration.

    Non-parametric: fits monotonic function for each class.
    """

    def __init__(self):
        self.isotonic_maps = []

    def fit(self, probs, labels):
        """Fit isotonic regression for each class."""
        n_classes = probs.shape[1]
        self.isotonic_maps = []

        for c in range(n_classes):
            # Binary: is this class correct?
            y_binary = (labels == c).astype(float)
            p_class = probs[:, c]

            # Fit isotonic regression
            isotonic_map = self._fit_isotonic(p_class, y_binary)
            self.isotonic_maps.append(isotonic_map)

        return self

    def _fit_isotonic(self, x, y):
        """Pool Adjacent Violators Algorithm (PAVA)."""
        # Sort by x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # PAVA
        n = len(y_sorted)
        y_iso = y_sorted.copy()
        weights = np.ones(n)

        i = 0
        while i < n - 1:
            if y_iso[i] > y_iso[i + 1]:
                # Merge adjacent violators
                y_iso[i] = (y_iso[i] * weights[i] + y_iso[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
                weights[i] = weights[i] + weights[i + 1]
                y_iso = np.delete(y_iso, i + 1)
                weights = np.delete(weights, i + 1)
                x_sorted = np.delete(x_sorted, i + 1)
                n -= 1

                # Check previous
                if i > 0:
                    i -= 1
            else:
                i += 1

        return (x_sorted, y_iso)

    def _apply_isotonic(self, x, isotonic_map):
        """Apply isotonic mapping using linear interpolation."""
        x_sorted, y_iso = isotonic_map

        # Interpolate
        return np.interp(x, x_sorted, y_iso)

    def calibrate(self, probs):
        """Apply isotonic calibration."""
        n_samples, n_classes = probs.shape
        calibrated = np.zeros_like(probs)

        for c in range(n_classes):
            calibrated[:, c] = self._apply_isotonic(probs[:, c], self.isotonic_maps[c])

        # Normalize
        calibrated = np.clip(calibrated, 0, 1)
        calibrated /= np.sum(calibrated, axis=1, keepdims=True)

        return calibrated


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_tr, X_te, y_tr, y_te = datasets['moons']

    # Split train into train + validation (for calibration)
    n_train = int(0.7 * len(X_tr))
    X_train, X_val = X_tr[:n_train], X_tr[n_train:]
    y_train, y_val = y_tr[:n_train], y_tr[n_train:]

    # Train uncalibrated model
    model = SimpleMLP([2, 64, 64, 2])
    model.fit(X_train, y_train, epochs=300)

    # -------- Experiment 1: Uncalibrated Model --------
    print("\n1. UNCALIBRATED MODEL ANALYSIS")
    print("-" * 40)

    logits_test = model.predict_logits(X_te)
    probs_test = model.predict_proba(X_te)

    acc = accuracy(y_te, np.argmax(probs_test, axis=1))
    ece = compute_ece(probs_test, y_te)

    print(f"Accuracy: {acc:.3f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"Mean confidence: {np.mean(np.max(probs_test, axis=1)):.3f}")
    print("→ Neural networks are often OVERCONFIDENT!")

    # -------- Experiment 2: Temperature Scaling --------
    print("\n2. TEMPERATURE SCALING")
    print("-" * 40)

    logits_val = model.predict_logits(X_val)
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(logits_val, y_val)

    print(f"Learned temperature: {temp_scaling.temperature:.3f}")

    probs_temp = temp_scaling.calibrate(logits_test)
    ece_temp = compute_ece(probs_temp, y_te)

    print(f"ECE before: {ece:.4f}")
    print(f"ECE after temperature scaling: {ece_temp:.4f}")
    print(f"Improvement: {(ece - ece_temp) / ece * 100:.1f}%")
    print("→ Single parameter T significantly improves calibration!")

    # -------- Experiment 3: Platt Scaling --------
    print("\n3. PLATT SCALING")
    print("-" * 40)

    platt = PlattScaling()
    platt.fit(logits_val, y_val)

    probs_platt = platt.calibrate(logits_test)
    ece_platt = compute_ece(probs_platt, y_te)

    print(f"Learned a: {platt.a}")
    print(f"Learned b: {platt.b}")
    print(f"ECE after Platt scaling: {ece_platt:.4f}")

    # -------- Experiment 4: Isotonic Regression --------
    print("\n4. ISOTONIC REGRESSION")
    print("-" * 40)

    probs_val = model.predict_proba(X_val)
    isotonic = IsotonicCalibration()
    isotonic.fit(probs_val, y_val)

    probs_isotonic = isotonic.calibrate(probs_test)
    ece_isotonic = compute_ece(probs_isotonic, y_te)

    print(f"ECE after isotonic: {ece_isotonic:.4f}")

    # -------- Experiment 5: Comparison --------
    print("\n5. CALIBRATION METHODS COMPARISON")
    print("-" * 40)

    print(f"{'Method':<25} {'ECE':<10} {'Accuracy':<10}")
    print("-" * 45)

    methods = [
        ('Uncalibrated', probs_test),
        ('Temperature Scaling', probs_temp),
        ('Platt Scaling', probs_platt),
        ('Isotonic Regression', probs_isotonic),
    ]

    for name, probs in methods:
        ece = compute_ece(probs, y_te)
        acc = accuracy(y_te, np.argmax(probs, axis=1))
        print(f"{name:<25} {ece:<10.4f} {acc:<10.3f}")

    print("→ Calibration improves ECE WITHOUT hurting accuracy!")

    # -------- Experiment 6: Temperature Values --------
    print("\n6. EFFECT OF TEMPERATURE VALUE")
    print("-" * 40)

    for T in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        def softmax_T(logits, T):
            exp_l = np.exp(logits / T - np.max(logits / T, axis=1, keepdims=True))
            return exp_l / np.sum(exp_l, axis=1, keepdims=True)

        probs_T = softmax_T(logits_test, T)
        ece_T = compute_ece(probs_T, y_te)
        mean_conf = np.mean(np.max(probs_T, axis=1))

        print(f"T={T:<4} ECE={ece_T:.4f} mean_confidence={mean_conf:.3f}")
    print("→ T > 1 softens predictions (less confident)")
    print("→ T < 1 sharpens predictions (more confident)")


def visualize_calibration():
    """Visualize calibration with reliability diagrams."""
    print("\n" + "="*60)
    print("CALIBRATION VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_tr, X_te, y_tr, y_te = datasets['moons']

    n_train = int(0.7 * len(X_tr))
    X_train, X_val = X_tr[:n_train], X_tr[n_train:]
    y_train, y_val = y_tr[:n_train], y_tr[n_train:]

    # Train model
    model = SimpleMLP([2, 64, 64, 2])
    model.fit(X_train, y_train, epochs=300)

    # Get predictions
    logits_val = model.predict_logits(X_val)
    logits_test = model.predict_logits(X_te)
    probs_uncal = model.predict_proba(X_te)

    # Fit calibration methods
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(logits_val, y_val)
    probs_temp = temp_scaling.calibrate(logits_test)

    platt = PlattScaling()
    platt.fit(logits_val, y_val)
    probs_platt = platt.calibrate(logits_test)

    probs_val = model.predict_proba(X_val)
    isotonic = IsotonicCalibration()
    isotonic.fit(probs_val, y_val)
    probs_isotonic = isotonic.calibrate(probs_uncal)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Reliability diagrams
    methods = [
        ('Uncalibrated', probs_uncal),
        ('Temperature Scaling', probs_temp),
        ('Platt Scaling', probs_platt),
    ]

    for i, (name, probs) in enumerate(methods):
        ax = axes[0, i]

        centers, accs, counts = compute_reliability_diagram(probs, y_te, n_bins=10)
        ece = compute_ece(probs, y_te)

        # Plot bars
        width = 0.08
        ax.bar(centers, accs, width=width, alpha=0.7, label='Accuracy')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{name}\nECE = {ece:.4f}')
        ax.legend(loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Row 2: Confidence distributions
    ax = axes[1, 0]
    conf_uncal = np.max(probs_uncal, axis=1)
    conf_temp = np.max(probs_temp, axis=1)
    ax.hist(conf_uncal, bins=20, alpha=0.5, label='Uncalibrated', density=True)
    ax.hist(conf_temp, bins=20, alpha=0.5, label='Temperature', density=True)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution')
    ax.legend()

    # Gap visualization
    ax = axes[1, 1]
    centers_uncal, accs_uncal, _ = compute_reliability_diagram(probs_uncal, y_te, n_bins=10)
    gaps = accs_uncal - centers_uncal

    colors = ['green' if g >= 0 else 'red' for g in gaps]
    ax.bar(centers_uncal, gaps, width=0.08, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Gap (Accuracy - Confidence)')
    ax.set_title('Calibration Gap\n(Negative = Overconfident)')

    # ECE comparison
    ax = axes[1, 2]
    methods_all = [
        ('Uncalibrated', probs_uncal),
        ('Temperature', probs_temp),
        ('Platt', probs_platt),
        ('Isotonic', probs_isotonic),
    ]
    names = [m[0] for m in methods_all]
    eces = [compute_ece(m[1], y_te) for m in methods_all]

    bars = ax.bar(names, eces, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax.set_ylabel('ECE')
    ax.set_title('ECE Comparison\n(Lower is better)')
    for bar, val in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('CALIBRATION METHODS\n'
                 'Making Neural Network Probabilities Trustworthy',
                 fontsize=12)
    plt.tight_layout()
    return fig


def benchmark_calibration():
    """Benchmark calibration across datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Calibration Across Datasets")
    print("="*60)

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'ECE_uncal':<12} {'ECE_temp':<12} {'Improvement':<12}")
    print("-" * 51)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential', 'clustered']:
            continue

        n_classes = len(np.unique(y_tr))
        n_train = int(0.7 * len(X_tr))
        X_train, X_val = X_tr[:n_train], X_tr[n_train:]
        y_train, y_val = y_tr[:n_train], y_tr[n_train:]

        # Train
        model = SimpleMLP([X_tr.shape[1], 32, 32, n_classes])
        model.fit(X_train, y_train, epochs=200)

        # Uncalibrated
        probs_uncal = model.predict_proba(X_te)
        ece_uncal = compute_ece(probs_uncal, y_te)

        # Temperature scaling
        logits_val = model.predict_logits(X_val)
        logits_test = model.predict_logits(X_te)

        temp = TemperatureScaling()
        temp.fit(logits_val, y_val)
        probs_temp = temp.calibrate(logits_test)
        ece_temp = compute_ece(probs_temp, y_te)

        improvement = (ece_uncal - ece_temp) / ece_uncal * 100 if ece_uncal > 0 else 0

        print(f"{name:<15} {ece_uncal:<12.4f} {ece_temp:<12.4f} {improvement:<12.1f}%")


if __name__ == '__main__':
    print("="*60)
    print("CALIBRATION — Making Probabilities Honest")
    print("="*60)

    print("""
WHAT THIS IS:
    Fix neural network probabilities so:
    P(correct | predicted probability p) = p

THE PROBLEM:
    Modern NNs are OVERCONFIDENT
    Say 95% confident but only right 70% of the time

THE SOLUTION:
    Post-process predictions on a validation set

METHODS:
    1. Temperature Scaling: softmax(logits / T)
       - Single parameter T
       - T > 1 softens, T < 1 sharpens
       - Usually sufficient!

    2. Platt Scaling: softmax(a × logits + b)
       - Per-class scaling parameters

    3. Isotonic Regression
       - Non-parametric, most flexible
       - Needs more validation data

MEASURING CALIBRATION:
    ECE = Expected Calibration Error
    Perfect calibration: ECE = 0
    """)

    ablation_experiments()
    benchmark_calibration()

    fig = visualize_calibration()
    save_path = '/Users/sid47/ML Algorithms/21_calibration.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Calibration ≠ Accuracy (high acc can be poorly calibrated)
2. Modern NNs are overconfident (deeper = worse)
3. Temperature scaling: single T, remarkably effective
4. Platt scaling: per-class a×logit + b
5. Isotonic: non-parametric, needs more data
6. Always calibrate before using probabilities for decisions!
    """)
