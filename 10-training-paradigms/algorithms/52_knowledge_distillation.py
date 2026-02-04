"""
===============================================================
KNOWLEDGE DISTILLATION — Paradigm: TEACHER-STUDENT
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Transfer knowledge from a LARGE model (teacher) to a SMALL model (student).

    Teacher (large, slow, accurate)
                ↓ soft labels
    Student (small, fast, compressed)

"The teacher's mistakes contain valuable information."

===============================================================
THE KEY INSIGHT: SOFT LABELS
===============================================================

Hard labels:  [0, 0, 1, 0, 0]  (one-hot)
Soft labels:  [0.1, 0.2, 0.6, 0.05, 0.05]  (teacher probabilities)

WHY SOFT LABELS ARE BETTER:
1. "2" looks more like "7" than like "4"
2. Soft labels encode this similarity structure!
3. More information per example

TEMPERATURE SCALING:
    softmax(z/T) where T > 1 "softens" the distribution

High temperature → more uniform (more info about similarities)
Low temperature → more peaked (closer to hard labels)

===============================================================
THE DISTILLATION LOSS
===============================================================

L = α × L_hard + (1-α) × L_soft

L_hard = CrossEntropy(student_pred, true_labels)
L_soft = KL(teacher_soft || student_soft)  × T²

The T² factor compensates for reduced gradients at high T.

===============================================================
KEY VARIANTS
===============================================================

1. RESPONSE DISTILLATION: Match output distributions
2. FEATURE DISTILLATION: Match intermediate representations
3. SELF-DISTILLATION: Student and teacher same architecture
4. ONLINE DISTILLATION: Teacher and student train together

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes teacher has learned useful structure
2. Assumes student has capacity to learn this structure
3. Temperature controls information transfer
4. May not work if architectures are too different

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature scaling."""
    return softmax(logits / temperature, axis=-1)

def cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    """Cross entropy loss."""
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(target * np.log(pred), axis=-1))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence: KL(P || Q)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.mean(np.sum(p * np.log(p / q), axis=-1))

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y]


# ============================================================
# TEACHER NETWORK (Large)
# ============================================================

class TeacherNetwork:
    """
    Large teacher network.

    In practice: BERT-large, ResNet-152, GPT-4
    Here: Simple MLP with more capacity
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build layers
        dims = [input_dim] + hidden_dims + [num_classes]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            W = he_init(dims[i], dims[i + 1])
            b = np.zeros((1, dims[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        self.num_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, x: np.ndarray, return_features: bool = False) -> Tuple[np.ndarray, ...]:
        """Forward pass returning logits and optionally intermediate features."""
        features = []
        h = x

        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h = relu(h @ W + b)
            features.append(h)

        # Output layer (no activation)
        logits = h @ self.weights[-1] + self.biases[-1]

        if return_features:
            return logits, features
        return logits

    def predict_proba(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get soft predictions with temperature."""
        logits = self.forward(x)
        return softmax_with_temperature(logits, temperature)


# ============================================================
# STUDENT NETWORK (Small)
# ============================================================

class StudentNetwork:
    """
    Small student network.

    In practice: DistilBERT, MobileNet, TinyLLaMA
    Here: Smaller MLP
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build layers
        dims = [input_dim] + hidden_dims + [num_classes]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            W = he_init(dims[i], dims[i + 1])
            b = np.zeros((1, dims[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        self.num_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, x: np.ndarray, return_features: bool = False) -> Tuple[np.ndarray, ...]:
        """Forward pass."""
        features = []
        h = x

        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h = relu(h @ W + b)
            features.append(h)

        logits = h @ self.weights[-1] + self.biases[-1]

        if return_features:
            return logits, features
        return logits

    def predict_proba(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get soft predictions with temperature."""
        logits = self.forward(x)
        return softmax_with_temperature(logits, temperature)


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

class KnowledgeDistillation:
    """
    Response-based Knowledge Distillation.

    Loss = α × CE(student, hard_labels) + (1-α) × T² × KL(teacher_soft, student_soft)

    The T² factor is important:
    - High T → softer distributions → smaller gradients
    - T² compensates to maintain gradient magnitude
    """

    def __init__(self, teacher: TeacherNetwork, student: StudentNetwork,
                 temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            teacher: Pre-trained teacher network
            student: Student network to train
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for hard label loss (1-alpha for soft)
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute distillation loss.

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Get teacher soft labels (no gradient needed)
        teacher_logits = self.teacher.forward(x)
        teacher_soft = softmax_with_temperature(teacher_logits, self.temperature)

        # Get student predictions
        student_logits = self.student.forward(x)
        student_hard = softmax(student_logits)
        student_soft = softmax_with_temperature(student_logits, self.temperature)

        # Hard label loss
        y_onehot = one_hot(y, self.student.num_classes)
        loss_hard = cross_entropy(student_hard, y_onehot)

        # Soft label loss (KL divergence)
        loss_soft = kl_divergence(teacher_soft, student_soft)

        # Combined loss (T² scaling for soft loss)
        total_loss = self.alpha * loss_hard + (1 - self.alpha) * (self.temperature ** 2) * loss_soft

        return total_loss, {
            'hard_loss': loss_hard,
            'soft_loss': loss_soft,
            'total_loss': total_loss
        }

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> Dict:
        """
        One training step with gradient descent.

        Simplified backprop for demonstration.
        """
        batch_size = len(y)

        # Forward pass
        student_logits, student_features = self.student.forward(x, return_features=True)
        teacher_logits = self.teacher.forward(x)

        # Soft labels
        teacher_soft = softmax_with_temperature(teacher_logits, self.temperature)
        student_soft = softmax_with_temperature(student_logits, self.temperature)
        student_hard = softmax(student_logits)

        # Compute losses
        y_onehot = one_hot(y, self.student.num_classes)
        loss_hard = cross_entropy(student_hard, y_onehot)
        loss_soft = kl_divergence(teacher_soft, student_soft)
        total_loss = self.alpha * loss_hard + (1 - self.alpha) * (self.temperature ** 2) * loss_soft

        # Backward pass (simplified: only update last layer for demo)
        # Gradient from hard loss
        d_logits_hard = (student_hard - y_onehot) / batch_size

        # Gradient from soft loss (scaled by T)
        d_logits_soft = (student_soft - teacher_soft) / self.temperature / batch_size

        # Combined gradient
        d_logits = self.alpha * d_logits_hard + (1 - self.alpha) * self.temperature * d_logits_soft

        # Update last layer
        h = student_features[-1] if student_features else x
        d_W = h.T @ d_logits
        d_b = np.sum(d_logits, axis=0, keepdims=True)

        self.student.weights[-1] -= lr * d_W
        self.student.biases[-1] -= lr * d_b

        return {'hard_loss': loss_hard, 'soft_loss': loss_soft, 'total_loss': total_loss}


# ============================================================
# FEATURE DISTILLATION
# ============================================================

class FeatureDistillation:
    """
    Feature-based Knowledge Distillation.

    Instead of matching outputs, match INTERMEDIATE features.

    Loss = CE(student, labels) + β × ||teacher_features - transform(student_features)||²

    WHY IT HELPS:
    - Teacher's intermediate representations contain rich information
    - Student learns HOW the teacher processes, not just WHAT it outputs
    - Can match multiple layers for deeper transfer
    """

    def __init__(self, teacher: TeacherNetwork, student: StudentNetwork,
                 feature_weight: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.feature_weight = feature_weight

        # Feature transformation (if dimensions differ)
        # Maps student feature to teacher feature dimension
        self.transforms = []
        for t_dim, s_dim in zip(teacher.hidden_dims, student.hidden_dims):
            if t_dim != s_dim:
                transform = he_init(s_dim, t_dim)
            else:
                transform = None
            self.transforms.append(transform)

    def compute_feature_loss(self, teacher_features: List[np.ndarray],
                            student_features: List[np.ndarray]) -> float:
        """Compute MSE loss between teacher and student features."""
        total_loss = 0

        for t_feat, s_feat, transform in zip(teacher_features, student_features, self.transforms):
            # Transform student features if needed
            if transform is not None:
                s_feat = s_feat @ transform

            # MSE loss
            total_loss += np.mean((t_feat - s_feat) ** 2)

        return total_loss / len(teacher_features)

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """Compute combined loss."""
        # Get features from both networks
        teacher_logits, teacher_features = self.teacher.forward(x, return_features=True)
        student_logits, student_features = self.student.forward(x, return_features=True)

        # Classification loss
        student_probs = softmax(student_logits)
        y_onehot = one_hot(y, self.student.num_classes)
        cls_loss = cross_entropy(student_probs, y_onehot)

        # Feature matching loss
        feat_loss = self.compute_feature_loss(teacher_features, student_features)

        # Combined
        total_loss = cls_loss + self.feature_weight * feat_loss

        return total_loss, {
            'cls_loss': cls_loss,
            'feat_loss': feat_loss,
            'total_loss': total_loss
        }


# ============================================================
# SELF-DISTILLATION
# ============================================================

class SelfDistillation:
    """
    Self-Distillation: Use the same architecture as teacher and student.

    Born-Again Networks (BAN):
    1. Train model normally
    2. Use trained model as teacher
    3. Train new model from scratch with distillation
    4. Repeat!

    WHY IT WORKS:
    - The model's own soft predictions provide regularization
    - Each generation can improve slightly
    - Similar to ensemble averaging but with one model
    """

    def __init__(self, model_fn, input_dim: int, hidden_dims: List[int],
                 num_classes: int, temperature: float = 3.0):
        """
        Args:
            model_fn: Function to create new model
        """
        self.model_fn = model_fn
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.temperature = temperature

        # Current generation
        self.current_model = model_fn(input_dim, hidden_dims, num_classes)
        self.generation = 0

    def born_again(self) -> None:
        """Create next generation."""
        # Current model becomes teacher
        self.teacher = self.current_model

        # Create new student
        self.current_model = self.model_fn(self.input_dim, self.hidden_dims, self.num_classes)
        self.generation += 1

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Compute distillation loss using previous generation as teacher."""
        if not hasattr(self, 'teacher'):
            # First generation: just cross-entropy
            logits = self.current_model.forward(x)
            probs = softmax(logits)
            y_onehot = one_hot(y, self.num_classes)
            return {'loss': cross_entropy(probs, y_onehot), 'generation': 0}

        # Distillation loss
        teacher_logits = self.teacher.forward(x)
        teacher_soft = softmax_with_temperature(teacher_logits, self.temperature)

        student_logits = self.current_model.forward(x)
        student_soft = softmax_with_temperature(student_logits, self.temperature)

        loss = kl_divergence(teacher_soft, student_soft) * (self.temperature ** 2)

        return {'loss': loss, 'generation': self.generation}


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_temperature_effect(temperatures: List[float] = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
                                  input_dim: int = 32,
                                  num_classes: int = 10,
                                  n_samples: int = 500) -> dict:
    """
    Effect of temperature on knowledge distillation.

    WHAT TO OBSERVE:
    - T=1: Soft labels ≈ hard labels (minimal transfer)
    - T~3-5: Good balance of information
    - T too high: Too uniform, loses discrimination
    """
    print("=" * 60)
    print("EXPERIMENT: Temperature Effect")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randint(0, num_classes, n_samples)

    # Train teacher
    teacher = TeacherNetwork(input_dim, [128, 64], num_classes)
    # Simulate trained teacher by making it predict based on data
    for i in range(100):
        logits = teacher.forward(X)
        probs = softmax(logits)
        y_onehot = one_hot(y, num_classes)
        d_logits = (probs - y_onehot) / n_samples
        # Simple gradient descent on last layer
        h = relu(X @ teacher.weights[0] + teacher.biases[0])
        for w, b in zip(teacher.weights[1:-1], teacher.biases[1:-1]):
            h = relu(h @ w + b)
        teacher.weights[-1] -= 0.1 * h.T @ d_logits
        teacher.biases[-1] -= 0.1 * np.sum(d_logits, axis=0, keepdims=True)

    results = {'temperatures': temperatures, 'entropies': [], 'student_accs': []}

    for T in temperatures:
        # Get teacher soft labels
        teacher_soft = teacher.predict_proba(X, temperature=T)

        # Entropy of soft labels (higher = more uniform)
        entropy = -np.mean(np.sum(teacher_soft * np.log(teacher_soft + 1e-10), axis=-1))
        results['entropies'].append(entropy)

        # Train student
        student = StudentNetwork(input_dim, [32], num_classes)
        kd = KnowledgeDistillation(teacher, student, temperature=T, alpha=0.3)

        for _ in range(50):
            kd.train_step(X, y, lr=0.1)

        # Evaluate student
        student_pred = np.argmax(student.predict_proba(X, temperature=1.0), axis=1)
        acc = np.mean(student_pred == y)
        results['student_accs'].append(acc)

        print(f"T={T:5.1f}: Soft label entropy = {entropy:.3f}, Student accuracy = {acc:.3f}")

    return results


def experiment_model_size_ratio(teacher_sizes: List[List[int]] = [[512, 256, 128], [256, 128], [128, 64], [64]],
                                student_size: List[int] = [32],
                                input_dim: int = 32,
                                num_classes: int = 10,
                                n_samples: int = 500) -> dict:
    """
    How teacher size affects distillation.

    WHAT TO OBSERVE:
    - Larger teacher doesn't always mean better student
    - Capacity gap can hurt transfer
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Teacher Size Effect")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randint(0, num_classes, n_samples)

    results = {'teacher_params': [], 'teacher_accs': [], 'student_accs': []}

    for t_hidden in teacher_sizes:
        # Create and "train" teacher
        teacher = TeacherNetwork(input_dim, t_hidden, num_classes)

        # Simple training
        for _ in range(100):
            logits = teacher.forward(X)
            probs = softmax(logits)
            y_onehot = one_hot(y, num_classes)
            d_logits = (probs - y_onehot) / n_samples
            # Get hidden activations
            h = relu(X @ teacher.weights[0] + teacher.biases[0])
            for w, b in zip(teacher.weights[1:-1], teacher.biases[1:-1]):
                h = relu(h @ w + b)
            teacher.weights[-1] -= 0.1 * h.T @ d_logits
            teacher.biases[-1] -= 0.1 * np.sum(d_logits, axis=0, keepdims=True)

        teacher_pred = np.argmax(teacher.predict_proba(X), axis=1)
        teacher_acc = np.mean(teacher_pred == y)

        # Train student with distillation
        student = StudentNetwork(input_dim, student_size, num_classes)
        kd = KnowledgeDistillation(teacher, student, temperature=3.0, alpha=0.3)

        for _ in range(50):
            kd.train_step(X, y, lr=0.1)

        student_pred = np.argmax(student.predict_proba(X), axis=1)
        student_acc = np.mean(student_pred == y)

        results['teacher_params'].append(teacher.num_params)
        results['teacher_accs'].append(teacher_acc)
        results['student_accs'].append(student_acc)

        print(f"Teacher {t_hidden}: {teacher.num_params:6d} params, "
              f"acc={teacher_acc:.3f} → Student acc={student_acc:.3f}")

    return results


def experiment_alpha_weight(alphas: List[float] = [0.0, 0.2, 0.5, 0.8, 1.0],
                            input_dim: int = 32,
                            num_classes: int = 10,
                            n_samples: int = 500) -> dict:
    """
    Effect of hard vs soft label balance.

    WHAT TO OBSERVE:
    - α=1: Only hard labels (no distillation)
    - α=0: Only soft labels (pure distillation)
    - Best is usually somewhere in between
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Hard vs Soft Label Balance")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randint(0, num_classes, n_samples)

    # Pre-train teacher
    teacher = TeacherNetwork(input_dim, [128, 64], num_classes)
    for _ in range(100):
        logits = teacher.forward(X)
        probs = softmax(logits)
        y_onehot = one_hot(y, num_classes)
        d_logits = (probs - y_onehot) / n_samples
        h = relu(X @ teacher.weights[0] + teacher.biases[0])
        h = relu(h @ teacher.weights[1] + teacher.biases[1])
        teacher.weights[-1] -= 0.1 * h.T @ d_logits
        teacher.biases[-1] -= 0.1 * np.sum(d_logits, axis=0, keepdims=True)

    results = {'alphas': alphas, 'student_accs': []}

    for alpha in alphas:
        student = StudentNetwork(input_dim, [32], num_classes)
        kd = KnowledgeDistillation(teacher, student, temperature=3.0, alpha=alpha)

        for _ in range(50):
            kd.train_step(X, y, lr=0.1)

        student_pred = np.argmax(student.predict_proba(X), axis=1)
        acc = np.mean(student_pred == y)
        results['student_accs'].append(acc)

        label_type = "Hard only" if alpha == 1.0 else "Soft only" if alpha == 0.0 else "Mixed"
        print(f"α={alpha:.1f} ({label_type:10}): Student accuracy = {acc:.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_distillation_concept(save_path: Optional[str] = None):
    """
    Visual explanation of knowledge distillation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. The core idea
    ax = axes[0]
    ax.set_title('Knowledge Distillation: Core Idea', fontweight='bold')

    # Teacher
    rect_t = plt.Rectangle((0.5, 2), 2, 1.5, facecolor='lightblue',
                           edgecolor='black', linewidth=2)
    ax.add_patch(rect_t)
    ax.annotate('Teacher\n(Large, Accurate)', (1.5, 2.75), ha='center', fontsize=10)
    ax.annotate(f'100M params', (1.5, 2.2), ha='center', fontsize=9, color='gray')

    # Arrow
    ax.annotate('', xy=(1.5, 1.2), xytext=(1.5, 1.9),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('Soft Labels\n(Dark Knowledge)', (2.5, 1.5), fontsize=9, color='green')

    # Student
    rect_s = plt.Rectangle((0.8, 0), 1.4, 1, facecolor='lightcoral',
                           edgecolor='black', linewidth=2)
    ax.add_patch(rect_s)
    ax.annotate('Student\n(Small, Fast)', (1.5, 0.5), ha='center', fontsize=10)
    ax.annotate(f'10M params', (1.5, 0.15), ha='center', fontsize=9, color='gray')

    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')

    # 2. Hard vs Soft labels
    ax = axes[1]
    ax.set_title('Hard vs Soft Labels', fontweight='bold')

    # Hard label
    hard = [0, 0, 1, 0, 0]
    ax.bar(np.arange(5) - 0.2, hard, width=0.35, label='Hard Label', color='blue', alpha=0.7)

    # Soft label
    soft = [0.05, 0.15, 0.6, 0.15, 0.05]
    ax.bar(np.arange(5) + 0.2, soft, width=0.35, label='Soft Label', color='orange', alpha=0.7)

    ax.set_xticks(range(5))
    ax.set_xticklabels(['Cat', 'Dog', 'Bird', 'Fish', 'Frog'])
    ax.set_ylabel('Probability')
    ax.legend()

    ax.text(2, 0.8, '"Bird looks more like\nDog than Fish"',
           ha='center', fontsize=9, style='italic')

    # 3. Temperature effect
    ax = axes[2]
    ax.set_title('Temperature Effect', fontweight='bold')

    logits = np.array([2.0, 1.0, 5.0, 0.5, 0.3])
    temps = [1, 3, 10]
    colors = ['blue', 'green', 'red']

    x = np.arange(5)
    width = 0.25

    for i, (T, color) in enumerate(zip(temps, colors)):
        probs = softmax_with_temperature(logits, T)
        ax.bar(x + i * width - width, probs, width=width, label=f'T={T}',
              color=color, alpha=0.7)

    ax.set_xticks(range(5))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax.set_ylabel('Probability')
    ax.legend()

    ax.text(2, 0.5, 'Higher T → Softer\n(more information)',
           ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_temperature_effect(results: dict, save_path: Optional[str] = None):
    """
    Visualize temperature effect on distillation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    temps = results['temperatures']

    # Entropy of soft labels
    ax = axes[0]
    ax.plot(temps, results['entropies'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Soft Label Entropy')
    ax.set_title('Soft Label Entropy vs Temperature\n(Higher = More Uniform)',
                fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Student accuracy
    ax = axes[1]
    ax.plot(temps, results['student_accs'], 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Student Accuracy')
    ax.set_title('Student Accuracy vs Temperature', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark optimal region
    best_idx = np.argmax(results['student_accs'])
    ax.axvline(temps[best_idx], color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'Best T={temps[best_idx]}', (temps[best_idx], results['student_accs'][best_idx]),
               textcoords='offset points', xytext=(10, -10), fontsize=9, color='red')

    plt.suptitle('Temperature in Knowledge Distillation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_distillation_variants(save_path: Optional[str] = None):
    """
    Visualize different distillation variants.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Response distillation
    ax = axes[0, 0]
    ax.set_title('Response Distillation', fontweight='bold')

    # Teacher
    rect_t = plt.Rectangle((0.5, 1.5), 1.5, 1, facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect_t)
    ax.annotate('Teacher', (1.25, 2), ha='center', fontsize=10)

    # Student
    rect_s = plt.Rectangle((0.5, 0), 1, 0.8, facecolor='lightcoral', edgecolor='black')
    ax.add_patch(rect_s)
    ax.annotate('Student', (1, 0.4), ha='center', fontsize=10)

    # Outputs
    ax.annotate('Soft probs', (2.5, 2), ha='center', fontsize=9)
    ax.annotate('Match!', (2.5, 1), ha='center', fontsize=10, color='green')
    ax.annotate('', xy=(2.5, 0.8), xytext=(2.5, 1.8),
               arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    # 2. Feature distillation
    ax = axes[0, 1]
    ax.set_title('Feature Distillation', fontweight='bold')

    # Teacher layers
    for i in range(3):
        rect = plt.Rectangle((i * 0.8, 1.5), 0.6, 1, facecolor='lightblue',
                             edgecolor='black', alpha=0.7)
        ax.add_patch(rect)

    ax.annotate('Teacher', (1.2, 2.7), ha='center', fontsize=10)

    # Student layers
    for i in range(3):
        rect = plt.Rectangle((i * 0.8, 0), 0.6, 0.8, facecolor='lightcoral',
                             edgecolor='black', alpha=0.7)
        ax.add_patch(rect)

    ax.annotate('Student', (1.2, -0.3), ha='center', fontsize=10)

    # Match intermediate features
    for i in range(3):
        ax.annotate('', xy=(i * 0.8 + 0.3, 1.5), xytext=(i * 0.8 + 0.3, 0.8),
                   arrowprops=dict(arrowstyle='<->', lw=1.5, color='green'))

    ax.annotate('Match features\nat each layer!', (3, 1.2), fontsize=9, color='green')

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    # 3. Self-distillation
    ax = axes[1, 0]
    ax.set_title('Self-Distillation (Born-Again Networks)', fontweight='bold')

    generations = ['Gen 1', 'Gen 2', 'Gen 3']
    for i, gen in enumerate(generations):
        x = i * 1.5
        rect = plt.Rectangle((x, 0.5), 1.2, 1, facecolor='lightgreen',
                             edgecolor='black', alpha=0.7 + i * 0.1)
        ax.add_patch(rect)
        ax.annotate(gen, (x + 0.6, 1), ha='center', fontsize=10)

        if i < 2:
            ax.annotate('', xy=((i+1) * 1.5, 1), xytext=(x + 1.2, 1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            ax.annotate('teach', (x + 1.35, 1.15), fontsize=8, color='blue')

    ax.annotate('Same architecture,\neach generation improves!', (2, 0), ha='center',
               fontsize=9, style='italic')

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 2)
    ax.axis('off')

    # 4. Online distillation
    ax = axes[1, 1]
    ax.set_title('Online Distillation (Co-training)', fontweight='bold')

    # Two models
    rect1 = plt.Rectangle((0.5, 0.5), 1.2, 1, facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect1)
    ax.annotate('Model A', (1.1, 1), ha='center', fontsize=10)

    rect2 = plt.Rectangle((2.5, 0.5), 1.2, 1, facecolor='lightcoral', edgecolor='black')
    ax.add_patch(rect2)
    ax.annotate('Model B', (3.1, 1), ha='center', fontsize=10)

    # Bidirectional arrows
    ax.annotate('', xy=(2.5, 1.2), xytext=(1.7, 1.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(1.7, 0.8), xytext=(2.5, 0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax.annotate('Train together,\nteach each other!', (2.1, 0), ha='center',
               fontsize=9, style='italic')

    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.5, 2)
    ax.axis('off')

    plt.suptitle('Knowledge Distillation Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_practical_examples(save_path: Optional[str] = None):
    """
    Real-world examples of knowledge distillation.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    examples = [
        ('DistilBERT', 'BERT → 40% smaller, 60% faster', '97% performance'),
        ('TinyBERT', 'BERT → 7.5x smaller', '96% performance'),
        ('MobileNet', 'Large CNN → mobile-friendly', 'ImageNet on phones'),
        ('DistilGPT-2', 'GPT-2 → 2x smaller', 'Similar generation quality'),
        ('Whisper (tiny)', 'Whisper large → small', 'Speech recognition on edge'),
    ]

    ax.set_title('Real-World Knowledge Distillation Examples', fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(examples)))

    for i, (name, desc, result) in enumerate(examples):
        y = len(examples) - 1 - i
        rect = plt.Rectangle((0.5, y), 4, 0.8, facecolor=colors[i],
                             edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(name, (0.7, y + 0.4), fontsize=11, fontweight='bold', va='center')
        ax.annotate(desc, (2.5, y + 0.4), fontsize=10, va='center')
        ax.annotate(result, (5.5, y + 0.4), fontsize=10, va='center', color='green')

    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, len(examples) + 0.5)
    ax.axis('off')

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
    print("KNOWLEDGE DISTILLATION — Paradigm: TEACHER-STUDENT")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    A large teacher's soft predictions contain "dark knowledge"
    about similarities between classes that hard labels miss.

    THE LOSS:
    L = α × CE(student, hard_labels) + (1-α) × T² × KL(teacher_soft, student_soft)

    KEY CONCEPTS:
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Concept             │ Description                         │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Soft Labels         │ Teacher's probability distribution  │
    │ Temperature (T)     │ Higher T → softer distribution      │
    │ Dark Knowledge      │ Class similarity info in soft labels│
    │ T² Scaling          │ Compensate for reduced gradients    │
    └─────────────────────┴─────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Temperature effect
    temp_results = experiment_temperature_effect()

    # Experiment 2: Teacher size
    size_results = experiment_model_size_ratio()

    # Experiment 3: Alpha balance
    alpha_results = experiment_alpha_weight()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_distillation_concept('52_knowledge_distillation_concept.png')
    visualize_temperature_effect(temp_results, '52_knowledge_distillation_temperature.png')
    visualize_distillation_variants('52_knowledge_distillation_variants.png')
    visualize_practical_examples('52_knowledge_distillation_examples.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. SOFT LABELS contain more information than hard labels
       - Class similarities encoded in probabilities
       - "2 looks more like 7 than like 4"

    2. TEMPERATURE controls information transfer
       - T=1: Nearly hard labels
       - T=3-5: Good balance (commonly used)
       - T too high: Too uniform, loses discrimination

    3. DISTILLATION VARIANTS:
       - Response: Match output probabilities
       - Feature: Match intermediate representations
       - Self: Same architecture, multiple generations

    4. PRACTICAL BENEFITS:
       - Smaller models (mobile deployment)
       - Faster inference
       - Often maintain 95%+ of teacher performance

    5. REAL-WORLD EXAMPLES:
       - DistilBERT: 40% smaller, 60% faster
       - MobileNet: CNN for mobile devices
       - TinyBERT: 7.5x smaller BERT
    """)
