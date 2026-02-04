"""
MC DROPOUT — Paradigm: UNCERTAINTY (Approximate Bayesian Inference)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Keep dropout ON during inference. Run multiple forward passes.
The variance in predictions = model uncertainty.

This is approximately equivalent to Bayesian neural networks,
but with ZERO extra cost during training!

===============================================================
THE KEY INSIGHT: DROPOUT ≈ VARIATIONAL INFERENCE
===============================================================

Gal & Ghahramani (2016) showed that:

    Dropout training ≈ Variational inference on a Bayesian NN

Each dropout mask samples a different "subnetwork".
Multiple forward passes with different masks samples from
the approximate posterior over weights.

Mean of predictions = expected prediction
Variance of predictions = epistemic uncertainty

===============================================================
WHY THIS MATTERS
===============================================================

Standard neural networks give OVERCONFIDENT predictions:
    - 99% confident even when wrong
    - No way to say "I don't know"

MC Dropout provides:
    - Uncertainty estimates for FREE (just keep dropout on)
    - Higher uncertainty for out-of-distribution inputs
    - Better calibrated predictions

===============================================================
TYPES OF UNCERTAINTY
===============================================================

EPISTEMIC (Model Uncertainty):
    - Uncertainty about the MODEL
    - Reducible with more data
    - MC Dropout captures this!

ALEATORIC (Data Uncertainty):
    - Inherent noise in the data
    - NOT reducible with more data
    - Requires heteroscedastic models

MC Dropout primarily captures EPISTEMIC uncertainty.

===============================================================
DROPOUT AS REGULARIZATION vs UNCERTAINTY
===============================================================

TRAINING: Dropout prevents co-adaptation (regularization)
INFERENCE: Dropout samples from weight posterior (uncertainty)

Same mechanism, different interpretations!

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes dropout approximates true Bayesian posterior
2. Bernoulli dropout → specific variational family
3. More samples → better uncertainty estimate
4. Dropout rate affects uncertainty magnitude

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
accuracy = datasets_module.accuracy


class MCDropoutMLP:
    """
    MLP with MC Dropout for uncertainty estimation.
    """

    def __init__(self, layer_sizes, dropout_rate=0.5, activation='relu'):
        """
        Parameters:
        -----------
        layer_sizes : List of layer sizes [input, hidden1, ..., output]
        dropout_rate : Probability of dropping a unit
        activation : 'relu' or 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_layers = len(layer_sizes) - 1

        # Initialize weights
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            # Kaiming/He initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        return x

    def _activation_grad(self, x, out):
        if self.activation == 'relu':
            return (out > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - out ** 2
        return np.ones_like(x)

    def _dropout_mask(self, shape, training=True):
        """Generate dropout mask."""
        if training:
            mask = (np.random.rand(*shape) > self.dropout_rate).astype(float)
            # Scale up to maintain expected value
            mask /= (1 - self.dropout_rate)
            return mask
        else:
            # During standard inference (no dropout), return all ones
            return np.ones(shape)

    def forward(self, X, training=True, return_intermediates=False):
        """
        Forward pass with dropout.

        training=True: Apply dropout (for training AND MC inference)
        training=False: No dropout (standard inference)
        """
        activations = [X]
        masks = []

        current = X

        for i in range(self.n_layers - 1):
            # Linear
            z = current @ self.weights[i] + self.biases[i]
            # Activation
            a = self._activation(z)
            # Dropout (apply to hidden layers)
            mask = self._dropout_mask(a.shape, training)
            a = a * mask

            activations.append(a)
            masks.append(mask)
            current = a

        # Output layer (no dropout, no activation for logits)
        logits = current @ self.weights[-1] + self.biases[-1]
        activations.append(logits)

        if return_intermediates:
            return logits, activations, masks
        return logits

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def backward(self, X, y, lr=0.01):
        """Backpropagation."""
        batch_size = X.shape[0]

        # Forward with intermediates
        logits, activations, masks = self.forward(X, training=True, return_intermediates=True)
        probs = self.softmax(logits)

        # Output layer gradient
        n_classes = probs.shape[1]
        y_onehot = np.zeros((batch_size, n_classes))
        y_onehot[np.arange(batch_size), y] = 1

        delta = (probs - y_onehot) / batch_size

        # Backprop through layers
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            dW = activations[i].T @ delta
            db = np.sum(delta, axis=0)

            # Gradient w.r.t. previous layer
            if i > 0:
                delta = delta @ self.weights[i].T
                # Dropout mask
                delta = delta * masks[i - 1]
                # Activation gradient
                delta = delta * self._activation_grad(None, activations[i])

            # Update
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        """Train the network with dropout."""
        n_samples = len(y)
        losses = []

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Single pass (you could add batching)
            self.backward(X_shuffled, y_shuffled, lr)

            # Compute loss (without dropout for stable estimate)
            logits = self.forward(X, training=False)
            probs = self.softmax(logits)
            loss = -np.mean(np.log(probs[np.arange(n_samples), y] + 1e-10))
            losses.append(loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        """Standard prediction (no dropout)."""
        logits = self.forward(X, training=False)
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        """Standard probability prediction (no dropout)."""
        logits = self.forward(X, training=False)
        return self.softmax(logits)

    def mc_predict(self, X, n_samples=50):
        """
        MC Dropout prediction.

        Run n_samples forward passes WITH dropout.
        Returns: mean predictions, uncertainty (std)
        """
        all_probs = []

        for _ in range(n_samples):
            logits = self.forward(X, training=True)  # Keep dropout ON!
            probs = self.softmax(logits)
            all_probs.append(probs)

        all_probs = np.array(all_probs)  # (n_samples, batch, n_classes)

        # Mean prediction
        mean_probs = np.mean(all_probs, axis=0)

        # Predictive uncertainty (std of predicted class probabilities)
        std_probs = np.std(all_probs, axis=0)

        # Epistemic uncertainty: variance in predictions
        # Often use predictive entropy or mutual information
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        return mean_probs, std_probs, predictive_entropy, all_probs

    def mc_predict_class(self, X, n_samples=50):
        """MC Dropout class prediction."""
        mean_probs, _, _, _ = self.mc_predict(X, n_samples)
        return np.argmax(mean_probs, axis=1)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    datasets = get_all_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    # -------- Experiment 1: Number of MC Samples --------
    print("\n1. EFFECT OF NUMBER OF MC SAMPLES")
    print("-" * 40)

    model = MCDropoutMLP([2, 32, 32, 2], dropout_rate=0.2)
    model.fit(X_train, y_train, epochs=200, verbose=False)

    for n_samples in [1, 5, 10, 25, 50, 100]:
        mean_probs, std_probs, entropy, _ = model.mc_predict(X_test, n_samples)
        preds = np.argmax(mean_probs, axis=1)
        acc = accuracy(y_test, preds)
        mean_uncertainty = np.mean(np.max(std_probs, axis=1))
        print(f"n_samples={n_samples:<4} accuracy={acc:.3f} mean_uncertainty={mean_uncertainty:.4f}")
    print("→ More samples → more stable uncertainty estimates")

    # -------- Experiment 2: Dropout Rate --------
    print("\n2. EFFECT OF DROPOUT RATE")
    print("-" * 40)

    for dropout_rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        model = MCDropoutMLP([2, 32, 32, 2], dropout_rate=dropout_rate)
        model.fit(X_train, y_train, epochs=200, verbose=False)

        mean_probs, std_probs, entropy, _ = model.mc_predict(X_test, n_samples=50)
        preds = np.argmax(mean_probs, axis=1)
        acc = accuracy(y_test, preds)
        mean_uncertainty = np.mean(np.max(std_probs, axis=1))
        print(f"dropout={dropout_rate:.1f} accuracy={acc:.3f} mean_uncertainty={mean_uncertainty:.4f}")
    print("→ Higher dropout → higher uncertainty (more variation)")
    print("→ But too high → worse accuracy")

    # -------- Experiment 3: In-Distribution vs Out-of-Distribution --------
    print("\n3. UNCERTAINTY: IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION")
    print("-" * 40)

    model = MCDropoutMLP([2, 32, 32, 2], dropout_rate=0.2)
    model.fit(X_train, y_train, epochs=200, verbose=False)

    # In-distribution: test set
    mean_probs_in, std_probs_in, entropy_in, _ = model.mc_predict(X_test, n_samples=50)

    # Out-of-distribution: far from training data
    X_ood = np.random.randn(100, 2) * 5 + 10  # Shifted far away
    mean_probs_ood, std_probs_ood, entropy_ood, _ = model.mc_predict(X_ood, n_samples=50)

    print(f"In-distribution:")
    print(f"  Mean entropy:      {np.mean(entropy_in):.4f}")
    print(f"  Mean max-std:      {np.mean(np.max(std_probs_in, axis=1)):.4f}")

    print(f"Out-of-distribution:")
    print(f"  Mean entropy:      {np.mean(entropy_ood):.4f}")
    print(f"  Mean max-std:      {np.mean(np.max(std_probs_ood, axis=1)):.4f}")
    print("→ HIGHER uncertainty for OOD data!")
    print("→ MC Dropout knows when it doesn't know")

    # -------- Experiment 4: Uncertainty near Decision Boundary --------
    print("\n4. UNCERTAINTY NEAR DECISION BOUNDARY")
    print("-" * 40)

    # Points clearly in each class
    X_clear = X_test[np.abs(X_test[:, 0]) > 1]
    # Points near boundary (x ≈ 0)
    X_boundary = X_test[np.abs(X_test[:, 0]) < 0.3]

    if len(X_boundary) > 0:
        _, std_clear, entropy_clear, _ = model.mc_predict(X_clear, n_samples=50)
        _, std_boundary, entropy_boundary, _ = model.mc_predict(X_boundary, n_samples=50)

        print(f"Clear region:    mean_entropy={np.mean(entropy_clear):.4f}")
        print(f"Near boundary:   mean_entropy={np.mean(entropy_boundary):.4f}")
        print("→ Higher uncertainty near decision boundary!")

    # -------- Experiment 5: Standard vs MC Dropout Predictions --------
    print("\n5. STANDARD vs MC DROPOUT PREDICTIONS")
    print("-" * 40)

    model = MCDropoutMLP([2, 32, 32, 2], dropout_rate=0.3)
    model.fit(X_train, y_train, epochs=200, verbose=False)

    # Standard prediction
    standard_preds = model.predict(X_test)
    standard_acc = accuracy(y_test, standard_preds)

    # MC prediction
    mc_preds = model.mc_predict_class(X_test, n_samples=50)
    mc_acc = accuracy(y_test, mc_preds)

    disagreement = np.mean(standard_preds != mc_preds)

    print(f"Standard accuracy: {standard_acc:.3f}")
    print(f"MC Dropout accuracy: {mc_acc:.3f}")
    print(f"Disagreement rate: {disagreement:.3f}")
    print("→ MC Dropout often gives same or better predictions")
    print("→ Plus you get uncertainty estimates for FREE!")

    # -------- Experiment 6: Calibration --------
    print("\n6. CALIBRATION: PREDICTED CONFIDENCE vs ACTUAL ACCURACY")
    print("-" * 40)

    mean_probs, _, _, _ = model.mc_predict(X_test, n_samples=50)
    confidences = np.max(mean_probs, axis=1)
    preds = np.argmax(mean_probs, axis=1)
    correct = (preds == y_test)

    # Bin by confidence
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        mask = (confidences >= low) & (confidences < high)
        if np.sum(mask) > 5:
            bin_acc = np.mean(correct[mask])
            bin_conf = np.mean(confidences[mask])
            print(f"Conf [{low:.1f}, {high:.1f}): n={np.sum(mask):3d} "
                  f"avg_conf={bin_conf:.3f} actual_acc={bin_acc:.3f}")
    print("→ Well-calibrated: confidence ≈ accuracy")


def visualize_uncertainty():
    """Visualize MC Dropout uncertainty."""
    print("\n" + "="*60)
    print("MC DROPOUT VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    datasets = get_2d_datasets()
    X_train, X_test, y_train, y_test = datasets['moons']

    model = MCDropoutMLP([2, 32, 32, 2], dropout_rate=0.2)
    model.fit(X_train, y_train, epochs=300, verbose=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Create grid for visualization
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # MC predictions on grid
    mean_probs, std_probs, entropy, all_probs = model.mc_predict(X_grid, n_samples=50)

    # Plot 1: Mean prediction
    ax = axes[0, 0]
    Z = np.argmax(mean_probs, axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolors='black', s=30)
    ax.set_title('Mean Prediction\n(Average of MC samples)')

    # Plot 2: Predictive entropy (uncertainty)
    ax = axes[0, 1]
    Z = entropy.reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=20, cmap='viridis')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='black', s=30)
    plt.colorbar(im, ax=ax)
    ax.set_title('Predictive Entropy\n(Higher = more uncertain)')

    # Plot 3: Std of class 0 probability
    ax = axes[0, 2]
    Z = std_probs[:, 0].reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=20, cmap='magma')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='black', s=30)
    plt.colorbar(im, ax=ax)
    ax.set_title('Std of P(class=0)\n(Epistemic uncertainty)')

    # Plot 4: Sample predictions (show variation)
    ax = axes[1, 0]
    # Show predictions from 5 different MC samples
    for i in range(5):
        Z = np.argmax(all_probs[i*10], axis=1).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], alpha=0.3, colors=['red', 'blue'][i%2])
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolors='black', s=30)
    ax.set_title('Multiple MC Samples\n(Different dropout masks)')

    # Plot 5: Confidence (max probability)
    ax = axes[1, 1]
    Z = np.max(mean_probs, axis=1).reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='black', s=30)
    plt.colorbar(im, ax=ax)
    ax.set_title('Confidence (Max Prob)\n(Higher = more confident)')

    # Plot 6: OOD uncertainty
    ax = axes[1, 2]
    # Extend grid to show OOD
    x_min_ext, x_max_ext = -5, 5
    y_min_ext, y_max_ext = -5, 5
    xx_ext, yy_ext = np.meshgrid(np.linspace(x_min_ext, x_max_ext, 50),
                                  np.linspace(y_min_ext, y_max_ext, 50))
    X_grid_ext = np.c_[xx_ext.ravel(), yy_ext.ravel()]

    _, _, entropy_ext, _ = model.mc_predict(X_grid_ext, n_samples=30)
    Z = entropy_ext.reshape(xx_ext.shape)
    im = ax.contourf(xx_ext, yy_ext, Z, levels=20, cmap='viridis')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='black', s=30)
    plt.colorbar(im, ax=ax)
    ax.set_title('Uncertainty (Extended View)\n(OOD regions have high uncertainty)')

    plt.suptitle('MC DROPOUT UNCERTAINTY\n'
                 'Dropout at test time → Bayesian uncertainty estimates',
                 fontsize=12)
    plt.tight_layout()
    return fig


def benchmark_uncertainty():
    """Benchmark uncertainty quality."""
    print("\n" + "="*60)
    print("BENCHMARK: Uncertainty Quality")
    print("="*60)

    datasets = get_all_datasets()

    print(f"\n{'Dataset':<15} {'Accuracy':<10} {'ID Entropy':<12} {'OOD Entropy':<12}")
    print("-" * 49)

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential', 'clustered']:
            continue

        model = MCDropoutMLP([X_tr.shape[1], 32, 32, len(np.unique(y_tr))],
                            dropout_rate=0.2)
        model.fit(X_tr, y_tr, epochs=200, verbose=False)

        # In-distribution
        mean_probs, _, entropy_id, _ = model.mc_predict(X_te, n_samples=30)
        acc = accuracy(y_te, np.argmax(mean_probs, axis=1))

        # Out-of-distribution
        X_ood = np.random.randn(100, X_tr.shape[1]) * 10
        _, _, entropy_ood, _ = model.mc_predict(X_ood, n_samples=30)

        print(f"{name:<15} {acc:<10.3f} {np.mean(entropy_id):<12.4f} {np.mean(entropy_ood):<12.4f}")

    print("\n→ OOD entropy should be higher than ID entropy!")


if __name__ == '__main__':
    print("="*60)
    print("MC DROPOUT — Bayesian Uncertainty for Free")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Keep dropout ON at test time.
    Multiple forward passes → uncertainty estimates.

THE KEY INSIGHT:
    Dropout training ≈ Variational inference on Bayesian NN
    Each dropout mask = sample from weight posterior

WHY THIS MATTERS:
    - Standard NNs are OVERCONFIDENT
    - MC Dropout: higher uncertainty for unknown inputs
    - Works with ANY dropout-trained network

EPISTEMIC vs ALEATORIC:
    MC Dropout captures EPISTEMIC uncertainty (model uncertainty)
    - Reducible with more data
    - Higher for out-of-distribution inputs

HOW TO USE:
    1. Train normally with dropout
    2. At test time: keep dropout ON
    3. Run N forward passes
    4. Mean = prediction, Variance = uncertainty
    """)

    ablation_experiments()
    benchmark_uncertainty()

    fig = visualize_uncertainty()
    save_path = '/Users/sid47/ML Algorithms/20_mc_dropout.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Keep dropout ON at inference time
2. Multiple forward passes sample from posterior
3. Variance = epistemic (model) uncertainty
4. Higher uncertainty for OOD and boundary cases
5. ZERO extra training cost - just use your existing model!
6. Well-calibrated: confidence ≈ actual accuracy
    """)
