"""
===============================================================
TRANSFER LEARNING — Paradigm: KNOWLEDGE REUSE
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Use knowledge learned from one task to help with another.

    Source Task (lots of data) → Pre-trained Model
                                        ↓
    Target Task (little data) → Fine-tuned Model

"Don't start from scratch. Stand on the shoulders of giants."

===============================================================
WHY IT WORKS
===============================================================

HIERARCHICAL FEATURE LEARNING:
Neural networks learn hierarchical features:
    Layer 1: Edges, textures (GENERAL)
    Layer 2: Shapes, patterns
    Layer 3: Parts, objects
    Layer N: Task-specific (SPECIFIC)

Early layers learn UNIVERSAL features that transfer!

THE DATA EFFICIENCY ARGUMENT:
- Training from scratch: Need millions of examples
- Fine-tuning: Need hundreds or thousands
- The pre-trained model already knows "how to see"

===============================================================
KEY STRATEGIES
===============================================================

1. FEATURE EXTRACTION
   Freeze pre-trained layers, only train new classifier
   Best when: Little target data, similar domains

2. FINE-TUNING
   Train entire network with small learning rate
   Best when: More target data, related domains

3. GRADUAL UNFREEZING
   Unfreeze layers progressively from top to bottom
   Best when: Moderate data, preventing catastrophic forgetting

4. DOMAIN ADAPTATION
   Explicitly minimize domain shift
   Best when: Different but related domains

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes source and target share low-level features
2. Assumes pre-trained features are useful for target
3. Risk: Negative transfer if domains too different
4. Requires careful learning rate tuning

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Callable


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y]


# ============================================================
# PRE-TRAINED NETWORK (Simulated)
# ============================================================

class PretrainedNetwork:
    """
    Simulates a pre-trained network.

    In practice, this would be loaded from:
    - ImageNet-trained ResNet, VGG, etc.
    - BERT, GPT for NLP
    - Any model trained on large dataset

    Architecture:
        Input → [Feature Extractor (frozen)] → Features → [Classifier] → Output
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: Dimensions of hidden layers (feature extractor)
            num_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Feature extractor layers
        self.feature_weights = []
        self.feature_biases = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            W = he_init(dims[i], dims[i + 1])
            b = np.zeros((1, dims[i + 1]))
            self.feature_weights.append(W)
            self.feature_biases.append(b)

        # Classifier layer
        self.classifier_weight = he_init(hidden_dims[-1], num_classes)
        self.classifier_bias = np.zeros((1, num_classes))

        # Track which layers are frozen
        self.frozen_layers = []

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract features using pre-trained layers."""
        h = x
        for W, b in zip(self.feature_weights, self.feature_biases):
            h = relu(h @ W + b)
        return h

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass."""
        features = self.extract_features(x)
        logits = features @ self.classifier_weight + self.classifier_bias
        probs = softmax(logits)
        return probs, features

    def freeze_feature_extractor(self):
        """Freeze all feature extraction layers."""
        self.frozen_layers = list(range(len(self.feature_weights)))

    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer."""
        if layer_idx in self.frozen_layers:
            self.frozen_layers.remove(layer_idx)

    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.frozen_layers = []

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Return only trainable parameters."""
        params = {'classifier_weight': self.classifier_weight,
                  'classifier_bias': self.classifier_bias}

        for i, (W, b) in enumerate(zip(self.feature_weights, self.feature_biases)):
            if i not in self.frozen_layers:
                params[f'feature_weight_{i}'] = W
                params[f'feature_bias_{i}'] = b

        return params


# ============================================================
# TRANSFER LEARNING STRATEGIES
# ============================================================

class FeatureExtraction:
    """
    Feature Extraction: Freeze pre-trained, train only classifier.

    WHEN TO USE:
    - Very small target dataset
    - Source and target domains are similar
    - Quick training needed

    THE IDEA:
    Pre-trained features are already good → just learn new classifier
    """

    def __init__(self, pretrained_model: PretrainedNetwork, num_target_classes: int):
        self.model = pretrained_model
        self.num_target_classes = num_target_classes

        # Freeze all feature extraction layers
        self.model.freeze_feature_extractor()

        # Replace classifier for new task
        feature_dim = self.model.hidden_dims[-1]
        self.model.classifier_weight = he_init(feature_dim, num_target_classes)
        self.model.classifier_bias = np.zeros((1, num_target_classes))

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """Train only the classifier."""
        # Forward pass
        features = self.model.extract_features(x)  # Features are fixed!
        logits = features @ self.model.classifier_weight + self.model.classifier_bias
        probs = softmax(logits)

        # Compute loss
        y_onehot = one_hot(y, self.num_target_classes)
        loss = cross_entropy_loss(probs, y_onehot)

        # Backward pass (only classifier)
        d_logits = (probs - y_onehot) / len(y)
        d_classifier_weight = features.T @ d_logits
        d_classifier_bias = np.sum(d_logits, axis=0, keepdims=True)

        # Update classifier only
        self.model.classifier_weight -= lr * d_classifier_weight
        self.model.classifier_bias -= lr * d_classifier_bias

        return loss


class FineTuning:
    """
    Fine-Tuning: Train entire network with small learning rate.

    WHEN TO USE:
    - More target data available
    - Want to adapt features to target domain
    - Related but not identical domains

    KEY INSIGHT:
    Use SMALLER learning rate for pre-trained layers!
    They're already good → don't want to destroy them.
    """

    def __init__(self, pretrained_model: PretrainedNetwork, num_target_classes: int,
                 feature_lr_multiplier: float = 0.1):
        """
        Args:
            feature_lr_multiplier: LR for features = base_lr × multiplier
        """
        self.model = pretrained_model
        self.num_target_classes = num_target_classes
        self.feature_lr_multiplier = feature_lr_multiplier

        # Unfreeze all layers
        self.model.unfreeze_all()

        # Replace classifier
        feature_dim = self.model.hidden_dims[-1]
        self.model.classifier_weight = he_init(feature_dim, num_target_classes)
        self.model.classifier_bias = np.zeros((1, num_target_classes))

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """Train entire network with different learning rates."""
        batch_size = len(y)

        # Forward pass (store activations for backprop)
        activations = [x]
        h = x
        for W, b in zip(self.model.feature_weights, self.model.feature_biases):
            z = h @ W + b
            h = relu(z)
            activations.append(h)

        logits = h @ self.model.classifier_weight + self.model.classifier_bias
        probs = softmax(logits)

        # Loss
        y_onehot = one_hot(y, self.num_target_classes)
        loss = cross_entropy_loss(probs, y_onehot)

        # Backward pass
        d_logits = (probs - y_onehot) / batch_size

        # Classifier gradients (full learning rate)
        d_classifier_weight = activations[-1].T @ d_logits
        d_classifier_bias = np.sum(d_logits, axis=0, keepdims=True)

        self.model.classifier_weight -= lr * d_classifier_weight
        self.model.classifier_bias -= lr * d_classifier_bias

        # Backprop through feature layers (reduced learning rate)
        d_h = d_logits @ self.model.classifier_weight.T
        feature_lr = lr * self.feature_lr_multiplier

        for i in range(len(self.model.feature_weights) - 1, -1, -1):
            # ReLU derivative
            d_h = d_h * (activations[i + 1] > 0)

            d_W = activations[i].T @ d_h
            d_b = np.sum(d_h, axis=0, keepdims=True)

            self.model.feature_weights[i] -= feature_lr * d_W
            self.model.feature_biases[i] -= feature_lr * d_b

            if i > 0:
                d_h = d_h @ self.model.feature_weights[i].T

        return loss


class GradualUnfreezing:
    """
    Gradual Unfreezing: Progressively unfreeze layers.

    WHEN TO USE:
    - Moderate target data
    - Want to prevent catastrophic forgetting
    - Careful adaptation needed

    THE STRATEGY:
    Epoch 1: Train only classifier
    Epoch 2: Unfreeze last feature layer, train
    Epoch 3: Unfreeze second-to-last, train
    ...
    Final: All layers trainable

    WHY IT WORKS:
    - Higher layers (task-specific) adapt first
    - Lower layers (general features) adapt slowly
    - Prevents forgetting useful general features
    """

    def __init__(self, pretrained_model: PretrainedNetwork, num_target_classes: int):
        self.model = pretrained_model
        self.num_target_classes = num_target_classes

        # Start fully frozen
        self.model.freeze_feature_extractor()

        # Replace classifier
        feature_dim = self.model.hidden_dims[-1]
        self.model.classifier_weight = he_init(feature_dim, num_target_classes)
        self.model.classifier_bias = np.zeros((1, num_target_classes))

        self.num_layers = len(self.model.feature_weights)
        self.unfrozen_depth = 0

    def unfreeze_next_layer(self):
        """Unfreeze the next layer (from top to bottom)."""
        if self.unfrozen_depth < self.num_layers:
            layer_to_unfreeze = self.num_layers - 1 - self.unfrozen_depth
            self.model.unfreeze_layer(layer_to_unfreeze)
            self.unfrozen_depth += 1
            return True
        return False

    def get_frozen_status(self) -> List[bool]:
        """Return which layers are frozen."""
        return [i in self.model.frozen_layers for i in range(self.num_layers)]


class DomainAdaptation:
    """
    Domain Adaptation: Minimize domain shift.

    WHEN TO USE:
    - Source and target domains are different
    - Want features that work across domains
    - Have unlabeled target data

    KEY TECHNIQUES:
    1. Domain Confusion: Make features domain-invariant
    2. Maximum Mean Discrepancy (MMD): Align feature distributions
    3. Adversarial: Domain discriminator can't tell domains apart
    """

    def __init__(self, pretrained_model: PretrainedNetwork):
        self.model = pretrained_model
        feature_dim = pretrained_model.hidden_dims[-1]

        # Domain discriminator
        self.domain_weight = he_init(feature_dim, 2)  # 2 domains
        self.domain_bias = np.zeros((1, 2))

    def compute_mmd(self, source_features: np.ndarray,
                    target_features: np.ndarray) -> float:
        """
        Maximum Mean Discrepancy: Measure distance between distributions.

        MMD = ||μ_source - μ_target||² in feature space

        If MMD is small, features are domain-invariant.
        """
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)

        mmd = np.sum((source_mean - target_mean) ** 2)
        return mmd

    def compute_coral_loss(self, source_features: np.ndarray,
                           target_features: np.ndarray) -> float:
        """
        CORAL: Correlation Alignment.

        Align second-order statistics (covariance) of source and target.
        """
        # Compute covariance matrices
        source_cov = np.cov(source_features.T)
        target_cov = np.cov(target_features.T)

        # Frobenius norm of difference
        coral = np.sum((source_cov - target_cov) ** 2)
        return coral / (4 * source_features.shape[1] ** 2)


# ============================================================
# SIMULATION: Pre-training on Source Task
# ============================================================

def pretrain_on_source(input_dim: int = 64, hidden_dims: List[int] = [128, 64],
                       num_classes: int = 10, n_samples: int = 5000,
                       epochs: int = 50) -> PretrainedNetwork:
    """
    Simulate pre-training on a large source dataset.
    """
    # Create synthetic source data
    np.random.seed(42)
    X_source = np.random.randn(n_samples, input_dim)
    # Add structure: different clusters for different classes
    for i in range(num_classes):
        mask = np.arange(n_samples) % num_classes == i
        X_source[mask] += np.random.randn(input_dim) * 2

    y_source = np.arange(n_samples) % num_classes

    # Create and train model
    model = PretrainedNetwork(input_dim, hidden_dims, num_classes)

    # Simple training loop
    lr = 0.1
    batch_size = 64

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_source, y_source = X_source[perm], y_source[perm]

        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_source[i:i+batch_size]
            y_batch = y_source[i:i+batch_size]

            # Forward
            probs, _ = model.forward(X_batch)
            y_onehot = one_hot(y_batch, num_classes)
            loss = cross_entropy_loss(probs, y_onehot)
            epoch_loss += loss

            # Simplified backward (just update classifier for demo)
            features = model.extract_features(X_batch)
            d_logits = (probs - y_onehot) / len(y_batch)
            model.classifier_weight -= lr * features.T @ d_logits
            model.classifier_bias -= lr * np.sum(d_logits, axis=0, keepdims=True)

    return model


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_transfer_strategies(pretrained_model: PretrainedNetwork,
                                   target_sizes: List[int] = [50, 100, 500, 1000],
                                   num_target_classes: int = 5) -> dict:
    """
    Compare different transfer learning strategies.

    WHAT TO OBSERVE:
    - Small data: Feature extraction wins
    - Large data: Fine-tuning wins
    - Middle ground: Gradual unfreezing balances both
    """
    print("=" * 60)
    print("EXPERIMENT: Transfer Learning Strategies")
    print("=" * 60)

    results = {'target_sizes': target_sizes,
               'feature_extraction': [],
               'fine_tuning': [],
               'from_scratch': []}

    input_dim = pretrained_model.input_dim

    for n_target in target_sizes:
        print(f"\nTarget dataset size: {n_target}")

        # Create target data (related but different from source)
        X_target = np.random.randn(n_target, input_dim)
        for i in range(num_target_classes):
            mask = np.arange(n_target) % num_target_classes == i
            X_target[mask] += np.random.randn(input_dim) * 1.5  # Slightly different

        y_target = np.arange(n_target) % num_target_classes

        # Split into train/test
        split = int(0.8 * n_target)
        X_train, X_test = X_target[:split], X_target[split:]
        y_train, y_test = y_target[:split], y_target[split:]

        # 1. Feature Extraction
        import copy
        model_fe = copy.deepcopy(pretrained_model)
        fe = FeatureExtraction(model_fe, num_target_classes)
        for _ in range(100):
            fe.train_step(X_train, y_train, lr=0.1)
        probs_fe, _ = fe.model.forward(X_test)
        acc_fe = np.mean(np.argmax(probs_fe, axis=1) == y_test)
        results['feature_extraction'].append(acc_fe)

        # 2. Fine-tuning
        model_ft = copy.deepcopy(pretrained_model)
        ft = FineTuning(model_ft, num_target_classes)
        for _ in range(100):
            ft.train_step(X_train, y_train, lr=0.05)
        probs_ft, _ = ft.model.forward(X_test)
        acc_ft = np.mean(np.argmax(probs_ft, axis=1) == y_test)
        results['fine_tuning'].append(acc_ft)

        # 3. From scratch (no transfer)
        model_scratch = PretrainedNetwork(input_dim, pretrained_model.hidden_dims,
                                          num_target_classes)
        ft_scratch = FineTuning(model_scratch, num_target_classes,
                                feature_lr_multiplier=1.0)  # Full LR
        for _ in range(100):
            ft_scratch.train_step(X_train, y_train, lr=0.05)
        probs_scratch, _ = ft_scratch.model.forward(X_test)
        acc_scratch = np.mean(np.argmax(probs_scratch, axis=1) == y_test)
        results['from_scratch'].append(acc_scratch)

        print(f"  Feature Extraction: {acc_fe:.3f}")
        print(f"  Fine-tuning:        {acc_ft:.3f}")
        print(f"  From scratch:       {acc_scratch:.3f}")

    return results


def experiment_learning_rate_sensitivity(pretrained_model: PretrainedNetwork,
                                         feature_lr_multipliers: List[float] = [0.01, 0.1, 0.5, 1.0],
                                         n_target: int = 500,
                                         num_target_classes: int = 5) -> dict:
    """
    Effect of learning rate on fine-tuning.

    WHAT TO OBSERVE:
    - Too high LR for features: Destroy pre-trained knowledge
    - Too low LR for features: Underfit target task
    - Sweet spot: Small multiplier (0.1-0.3)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Learning Rate for Fine-tuning")
    print("=" * 60)

    import copy
    input_dim = pretrained_model.input_dim

    # Create target data
    X_target = np.random.randn(n_target, input_dim)
    y_target = np.arange(n_target) % num_target_classes

    split = int(0.8 * n_target)
    X_train, X_test = X_target[:split], X_target[split:]
    y_train, y_test = y_target[:split], y_target[split:]

    results = {'multipliers': feature_lr_multipliers, 'accuracies': [], 'losses': []}

    for mult in feature_lr_multipliers:
        model = copy.deepcopy(pretrained_model)
        ft = FineTuning(model, num_target_classes, feature_lr_multiplier=mult)

        losses = []
        for _ in range(100):
            loss = ft.train_step(X_train, y_train, lr=0.05)
            losses.append(loss)

        probs, _ = ft.model.forward(X_test)
        acc = np.mean(np.argmax(probs, axis=1) == y_test)

        results['accuracies'].append(acc)
        results['losses'].append(losses)

        print(f"Feature LR multiplier {mult:.2f}: Accuracy = {acc:.3f}")

    return results


def experiment_domain_shift(pretrained_model: PretrainedNetwork,
                            shift_magnitudes: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
                            n_target: int = 500,
                            num_target_classes: int = 5) -> dict:
    """
    How domain shift affects transfer.

    WHAT TO OBSERVE:
    - Small shift: Transfer helps a lot
    - Large shift: Transfer helps less (negative transfer possible)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Domain Shift Effect")
    print("=" * 60)

    import copy
    input_dim = pretrained_model.input_dim

    results = {'shift_magnitudes': shift_magnitudes,
               'transfer_acc': [],
               'scratch_acc': []}

    for shift in shift_magnitudes:
        # Create target data with domain shift
        X_target = np.random.randn(n_target, input_dim)
        X_target += np.random.randn(input_dim) * shift  # Domain shift!
        y_target = np.arange(n_target) % num_target_classes

        split = int(0.8 * n_target)
        X_train, X_test = X_target[:split], X_target[split:]
        y_train, y_test = y_target[:split], y_target[split:]

        # With transfer
        model_transfer = copy.deepcopy(pretrained_model)
        ft = FineTuning(model_transfer, num_target_classes)
        for _ in range(100):
            ft.train_step(X_train, y_train, lr=0.05)
        probs, _ = ft.model.forward(X_test)
        acc_transfer = np.mean(np.argmax(probs, axis=1) == y_test)

        # Without transfer
        model_scratch = PretrainedNetwork(input_dim, pretrained_model.hidden_dims,
                                          num_target_classes)
        ft_scratch = FineTuning(model_scratch, num_target_classes, feature_lr_multiplier=1.0)
        for _ in range(100):
            ft_scratch.train_step(X_train, y_train, lr=0.05)
        probs, _ = ft_scratch.model.forward(X_test)
        acc_scratch = np.mean(np.argmax(probs, axis=1) == y_test)

        results['transfer_acc'].append(acc_transfer)
        results['scratch_acc'].append(acc_scratch)

        print(f"Domain shift {shift:.1f}: Transfer={acc_transfer:.3f}, Scratch={acc_scratch:.3f}, "
              f"Gain={acc_transfer - acc_scratch:+.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_transfer_learning_concept(save_path: Optional[str] = None):
    """
    Visual explanation of transfer learning.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. The transfer concept
    ax = axes[0]
    ax.set_title('Transfer Learning Concept', fontweight='bold')

    # Source task
    ax.add_patch(plt.Rectangle((0, 2), 2.5, 1.5, facecolor='lightblue',
                               edgecolor='black', linewidth=2))
    ax.annotate('Source Task\n(ImageNet: 1M images)', (1.25, 2.75),
               ha='center', va='center', fontsize=10)

    # Arrow down
    ax.annotate('', xy=(1.25, 1.8), xytext=(1.25, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('Transfer\nKnowledge', (1.8, 1.9), fontsize=9, color='green')

    # Pre-trained model
    ax.add_patch(plt.Rectangle((0.5, 0.8), 1.5, 0.8, facecolor='gold',
                               edgecolor='black', linewidth=2))
    ax.annotate('Pre-trained\nModel', (1.25, 1.2), ha='center', va='center', fontsize=10)

    # Arrow to target
    ax.annotate('', xy=(3, 1.2), xytext=(2.2, 1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    # Target task
    ax.add_patch(plt.Rectangle((3, 0.5), 2.5, 1.4, facecolor='lightcoral',
                               edgecolor='black', linewidth=2))
    ax.annotate('Target Task\n(Your data: 1K images)', (4.25, 1.2),
               ha='center', va='center', fontsize=10)

    ax.set_xlim(-0.5, 6)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # 2. Feature hierarchy
    ax = axes[1]
    ax.set_title('Why Transfer Works: Feature Hierarchy', fontweight='bold')

    layers = ['Input', 'Layer 1\n(Edges)', 'Layer 2\n(Textures)', 'Layer 3\n(Parts)',
              'Layer 4\n(Objects)', 'Output\n(Classes)']
    colors = ['white', 'lightgreen', 'lightgreen', 'yellow', 'orange', 'lightcoral']
    transfer_labels = ['', 'TRANSFER\n(General)', 'TRANSFER\n(General)',
                      'ADAPT', 'ADAPT', 'REPLACE']

    for i, (layer, color, label) in enumerate(zip(layers, colors, transfer_labels)):
        y = 5 - i * 0.9
        rect = plt.Rectangle((0.5, y), 2, 0.7, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.annotate(layer, (1.5, y + 0.35), ha='center', va='center', fontsize=9)
        if label:
            ax.annotate(label, (3.2, y + 0.35), ha='left', va='center',
                       fontsize=8, color='darkgreen' if 'TRANSFER' in label else 'darkorange')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 3. Strategies
    ax = axes[2]
    ax.set_title('Transfer Strategies', fontweight='bold')

    strategies = [
        ('Feature Extraction', 'Freeze all, train classifier only', 'Little data'),
        ('Fine-tuning', 'Train all with small LR', 'More data'),
        ('Gradual Unfreezing', 'Unfreeze top→bottom', 'Moderate data'),
        ('Domain Adaptation', 'Align feature distributions', 'Domain shift'),
    ]

    for i, (name, desc, when) in enumerate(strategies):
        y = 0.85 - i * 0.22
        ax.text(0.05, y, f'• {name}', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.05, y - 0.06, f'  {desc}', fontsize=9,
               transform=ax.transAxes, color='gray')
        ax.text(0.7, y - 0.03, f'When: {when}', fontsize=8,
               transform=ax.transAxes, color='blue')

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_strategy_comparison(results: dict, save_path: Optional[str] = None):
    """
    Compare transfer learning strategies across data sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = results['target_sizes']
    ax.plot(sizes, results['feature_extraction'], 'b-o', label='Feature Extraction',
           linewidth=2, markersize=8)
    ax.plot(sizes, results['fine_tuning'], 'g-s', label='Fine-tuning',
           linewidth=2, markersize=8)
    ax.plot(sizes, results['from_scratch'], 'r-^', label='From Scratch (no transfer)',
           linewidth=2, markersize=8)

    ax.set_xlabel('Target Dataset Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Transfer Learning: Strategy Comparison\nby Target Dataset Size',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate regions
    ax.axvspan(0, 200, alpha=0.1, color='blue', label='_nolegend_')
    ax.axvspan(500, 1200, alpha=0.1, color='green', label='_nolegend_')
    ax.text(100, ax.get_ylim()[1] * 0.95, 'Feature Extract\nwins here',
           ha='center', fontsize=9, color='blue')
    ax.text(800, ax.get_ylim()[1] * 0.95, 'Fine-tuning\nwins here',
           ha='center', fontsize=9, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_domain_shift(results: dict, save_path: Optional[str] = None):
    """
    Visualize effect of domain shift on transfer.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    shifts = results['shift_magnitudes']
    ax.plot(shifts, results['transfer_acc'], 'g-o', label='With Transfer',
           linewidth=2, markersize=8)
    ax.plot(shifts, results['scratch_acc'], 'r-s', label='From Scratch',
           linewidth=2, markersize=8)

    # Plot transfer benefit
    benefit = np.array(results['transfer_acc']) - np.array(results['scratch_acc'])
    ax.fill_between(shifts, results['scratch_acc'], results['transfer_acc'],
                    where=benefit > 0, alpha=0.3, color='green', label='Transfer Benefit')
    ax.fill_between(shifts, results['scratch_acc'], results['transfer_acc'],
                    where=benefit < 0, alpha=0.3, color='red', label='Negative Transfer')

    ax.set_xlabel('Domain Shift Magnitude')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Transfer Learning vs Domain Shift\n'
                '(Green = transfer helps, Red = transfer hurts)',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_gradual_unfreezing(save_path: Optional[str] = None):
    """
    Visualize gradual unfreezing strategy.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    num_layers = 5
    epochs = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4+']

    for ax, epoch in zip(axes, epochs):
        ax.set_title(epoch, fontweight='bold')

        # Determine frozen status
        if epoch == 'Epoch 1':
            frozen = [True, True, True, True, False]  # Only classifier
        elif epoch == 'Epoch 2':
            frozen = [True, True, True, False, False]
        elif epoch == 'Epoch 3':
            frozen = [True, True, False, False, False]
        else:
            frozen = [False, False, False, False, False]

        layer_names = ['Conv1\n(edges)', 'Conv2\n(textures)', 'Conv3\n(parts)',
                      'FC1', 'Classifier']

        for i, (name, is_frozen) in enumerate(zip(layer_names, frozen)):
            color = 'lightblue' if is_frozen else 'lightgreen'
            rect = plt.Rectangle((0.2, 4 - i * 0.9), 1.6, 0.7,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.annotate(name, (1, 4 - i * 0.9 + 0.35), ha='center', va='center', fontsize=9)

            status = '❄️ Frozen' if is_frozen else '🔥 Training'
            ax.annotate(status, (2.1, 4 - i * 0.9 + 0.35), ha='left', fontsize=8)

        ax.set_xlim(0, 3.5)
        ax.set_ylim(-0.5, 5)
        ax.axis('off')

    plt.suptitle('Gradual Unfreezing: Progressively Unfreeze Layers',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFER LEARNING — Paradigm: KNOWLEDGE REUSE")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Don't start from scratch. Use knowledge from related tasks.

    THE PROCESS:
    1. Pre-train on large source dataset (ImageNet, etc.)
    2. Transfer learned features to target task
    3. Fine-tune or freeze based on data availability

    KEY STRATEGIES:
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Strategy            │ When to Use                         │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Feature Extraction  │ Little target data, similar domains │
    │ Fine-tuning         │ More target data, related domains   │
    │ Gradual Unfreezing  │ Moderate data, prevent forgetting   │
    │ Domain Adaptation   │ Different but related domains       │
    └─────────────────────┴─────────────────────────────────────┘
    """)

    # Pre-train a model
    print("\n" + "=" * 70)
    print("PRE-TRAINING ON SOURCE TASK")
    print("=" * 70)
    pretrained = pretrain_on_source()
    print("Pre-training complete!")

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Compare strategies
    strategy_results = experiment_transfer_strategies(pretrained)

    # Experiment 2: Learning rate sensitivity
    lr_results = experiment_learning_rate_sensitivity(pretrained)

    # Experiment 3: Domain shift
    shift_results = experiment_domain_shift(pretrained)

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_transfer_learning_concept('50_transfer_learning_concept.png')
    visualize_strategy_comparison(strategy_results, '50_transfer_learning_strategies.png')
    visualize_domain_shift(shift_results, '50_transfer_learning_domain_shift.png')
    visualize_gradual_unfreezing('50_transfer_learning_unfreezing.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. TRANSFER LEARNING = Knowledge reuse
       - Pre-trained features are often useful
       - Especially early layers (edges, textures)

    2. CHOOSE STRATEGY based on data:
       - Little data → Feature extraction
       - More data → Fine-tuning
       - In between → Gradual unfreezing

    3. LEARNING RATE matters:
       - Too high → Destroy pre-trained knowledge
       - Too low → Underfit target task
       - Use smaller LR for pre-trained layers

    4. DOMAIN SHIFT limits transfer:
       - Similar domains → Transfer helps a lot
       - Very different → May hurt (negative transfer)

    5. IN PRACTICE:
       - Always try transfer first
       - Start with frozen features, then fine-tune
       - Monitor for overfitting
    """)
