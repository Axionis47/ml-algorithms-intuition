"""
===============================================================
CURRICULUM LEARNING — Paradigm: EASY TO HARD
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Train on EASIER examples first, then gradually introduce HARDER ones.

    Easy → Medium → Hard

Just like humans learn:
- Addition before calculus
- Letters before sentences
- Simple melodies before symphonies

"Start simple, build complexity."

===============================================================
WHY IT WORKS
===============================================================

1. BETTER OPTIMIZATION LANDSCAPE:
   Easy examples create smooth loss surface
   Harder examples added when model is already in good region

2. AVOID LOCAL MINIMA:
   Random curriculum can trap in bad solutions
   Easy-first guides toward generalizable features

3. FASTER CONVERGENCE:
   Early gradients are meaningful (not random)
   Model builds on solid foundation

===============================================================
KEY QUESTIONS
===============================================================

1. HOW TO MEASURE DIFFICULTY?
   - Loss value (high loss = hard)
   - Human annotation
   - Data properties (length, noise, ambiguity)
   - Teacher model confidence

2. HOW TO SCHEDULE?
   - Linear increase
   - Exponential increase
   - Self-paced (model decides)

===============================================================
KEY VARIANTS
===============================================================

1. PREDEFINED CURRICULUM: Fixed difficulty order
2. SELF-PACED LEARNING: Model selects examples it can learn
3. TEACHER-STUDENT: Teacher guides student's curriculum
4. ANTI-CURRICULUM: Hard examples first (sometimes works!)

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes "easy" correlates with "fundamental"
2. Assumes gradual complexity builds understanding
3. May hurt if easy examples are misleading
4. Requires meaningful difficulty metric

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Callable


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(target * np.log(pred), axis=-1))

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return np.mean((pred - target) ** 2)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)


# ============================================================
# DIFFICULTY METRICS
# ============================================================

class DifficultyMetrics:
    """
    Various ways to measure example difficulty.
    """

    @staticmethod
    def loss_based(model_forward: Callable, X: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
        """
        Difficulty = model's loss on example.

        Higher loss = harder example.
        """
        difficulties = []
        for i in range(len(X)):
            pred = model_forward(X[i:i+1])
            if pred.ndim > 1:
                pred = softmax(pred)
            loss = -np.log(pred[0, y[i]] + 1e-10)
            difficulties.append(loss)
        return np.array(difficulties)

    @staticmethod
    def confidence_based(model_forward: Callable, X: np.ndarray) -> np.ndarray:
        """
        Difficulty = 1 - model confidence.

        Low confidence = hard example.
        """
        difficulties = []
        for i in range(len(X)):
            pred = model_forward(X[i:i+1])
            pred = softmax(pred)
            confidence = np.max(pred)
            difficulties.append(1 - confidence)
        return np.array(difficulties)

    @staticmethod
    def noise_level(X: np.ndarray, clean_X: np.ndarray) -> np.ndarray:
        """
        Difficulty = amount of noise.
        """
        return np.linalg.norm(X - clean_X, axis=1)

    @staticmethod
    def sequence_length(sequences: List[np.ndarray]) -> np.ndarray:
        """
        Difficulty = sequence length.

        Longer sequences are typically harder.
        """
        return np.array([len(seq) for seq in sequences])

    @staticmethod
    def class_frequency(y: np.ndarray) -> np.ndarray:
        """
        Difficulty = 1 / class frequency.

        Rare classes are harder.
        """
        unique, counts = np.unique(y, return_counts=True)
        freq = counts / len(y)
        freq_dict = dict(zip(unique, freq))
        return np.array([1 / freq_dict[label] for label in y])


# ============================================================
# CURRICULUM SCHEDULERS
# ============================================================

class CurriculumScheduler:
    """
    Base class for curriculum schedulers.

    Determines which fraction of data to use at each epoch.
    """

    def get_data_fraction(self, epoch: int) -> float:
        """Return fraction of data to use (0 to 1)."""
        raise NotImplementedError


class LinearScheduler(CurriculumScheduler):
    """
    Linearly increase data fraction.

    Epoch 0: start_fraction
    Epoch T: 1.0 (all data)
    """

    def __init__(self, start_fraction: float = 0.2, warmup_epochs: int = 10):
        self.start_fraction = start_fraction
        self.warmup_epochs = warmup_epochs

    def get_data_fraction(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return 1.0
        progress = epoch / self.warmup_epochs
        return self.start_fraction + (1 - self.start_fraction) * progress


class ExponentialScheduler(CurriculumScheduler):
    """
    Exponentially increase data fraction.

    Starts slow, then quickly includes all data.
    """

    def __init__(self, start_fraction: float = 0.1, growth_rate: float = 1.5):
        self.start_fraction = start_fraction
        self.growth_rate = growth_rate

    def get_data_fraction(self, epoch: int) -> float:
        return min(1.0, self.start_fraction * (self.growth_rate ** epoch))


class StepScheduler(CurriculumScheduler):
    """
    Step-wise curriculum.

    Discrete stages of increasing difficulty.
    """

    def __init__(self, stages: List[Tuple[int, float]]):
        """
        Args:
            stages: List of (epoch, fraction) tuples
        """
        self.stages = sorted(stages, key=lambda x: x[0])

    def get_data_fraction(self, epoch: int) -> float:
        for stage_epoch, fraction in reversed(self.stages):
            if epoch >= stage_epoch:
                return fraction
        return self.stages[0][1]


# ============================================================
# CURRICULUM LEARNING TRAINER
# ============================================================

class CurriculumTrainer:
    """
    Training with curriculum learning.
    """

    def __init__(self, model, scheduler: CurriculumScheduler,
                 difficulty_metric: Callable):
        """
        Args:
            model: Model to train
            scheduler: Curriculum scheduler
            difficulty_metric: Function to compute example difficulties
        """
        self.model = model
        self.scheduler = scheduler
        self.difficulty_metric = difficulty_metric

    def get_curriculum_batch(self, X: np.ndarray, y: np.ndarray,
                            difficulties: np.ndarray, epoch: int,
                            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch according to curriculum.

        Selects easiest examples based on current data fraction.
        """
        # Sort by difficulty (easy first)
        sorted_indices = np.argsort(difficulties)

        # Get data fraction for this epoch
        fraction = self.scheduler.get_data_fraction(epoch)
        num_samples = int(len(X) * fraction)

        # Select easiest examples
        selected_indices = sorted_indices[:num_samples]

        # Random batch from selected
        batch_indices = np.random.choice(selected_indices,
                                        size=min(batch_size, len(selected_indices)),
                                        replace=False)

        return X[batch_indices], y[batch_indices]


# ============================================================
# SELF-PACED LEARNING
# ============================================================

class SelfPacedLearning:
    """
    Self-Paced Learning: Model decides its own curriculum.

    Instead of predefined difficulty, use model's current loss.
    Examples with loss < threshold are "learnable".

    THE OBJECTIVE:
    min_w,v  Σ v_i × L(x_i, y_i; w) - λ Σ v_i

    v_i ∈ {0, 1}: Whether to include example i
    λ: Controls pace (higher λ → more examples)

    SOLUTION:
    v_i = 1 if L(x_i) < λ, else 0
    "Include examples you can learn."
    """

    def __init__(self, model, initial_threshold: float = 1.0,
                 threshold_growth: float = 1.1):
        """
        Args:
            initial_threshold: Starting loss threshold
            threshold_growth: How fast to increase threshold
        """
        self.model = model
        self.threshold = initial_threshold
        self.threshold_growth = threshold_growth

    def compute_weights(self, X: np.ndarray, y: np.ndarray,
                       forward_fn: Callable) -> np.ndarray:
        """
        Compute example weights based on current loss.

        Returns binary weights: 1 if learnable, 0 otherwise.
        """
        weights = np.zeros(len(X))

        for i in range(len(X)):
            pred = forward_fn(X[i:i+1])
            pred = softmax(pred)
            loss = -np.log(pred[0, y[i]] + 1e-10)

            if loss < self.threshold:
                weights[i] = 1.0

        return weights

    def update_threshold(self):
        """Increase threshold to include harder examples."""
        self.threshold *= self.threshold_growth


# ============================================================
# ANTI-CURRICULUM (Hard First)
# ============================================================

class AntiCurriculumTrainer:
    """
    Anti-Curriculum: Start with HARD examples.

    WHY IT MIGHT WORK:
    - Hard examples may be more informative
    - Forces model to learn robust features early
    - Works well when easy examples are too easy

    WHEN TO USE:
    - When easy examples might teach shortcuts
    - When hard examples are rare but important
    """

    def __init__(self, model, scheduler: CurriculumScheduler,
                 difficulty_metric: Callable):
        self.model = model
        self.scheduler = scheduler
        self.difficulty_metric = difficulty_metric

    def get_curriculum_batch(self, X: np.ndarray, y: np.ndarray,
                            difficulties: np.ndarray, epoch: int,
                            batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch with HARDEST examples first.
        """
        # Sort by difficulty (HARD first - reversed!)
        sorted_indices = np.argsort(-difficulties)  # Negative for descending

        fraction = self.scheduler.get_data_fraction(epoch)
        num_samples = int(len(X) * fraction)

        selected_indices = sorted_indices[:num_samples]
        batch_indices = np.random.choice(selected_indices,
                                        size=min(batch_size, len(selected_indices)),
                                        replace=False)

        return X[batch_indices], y[batch_indices]


# ============================================================
# SIMPLE MODEL FOR EXPERIMENTS
# ============================================================

class SimpleClassifier:
    """Simple MLP for curriculum learning experiments."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        self.W1 = he_init(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = he_init(hidden_dim, num_classes)
        self.b2 = np.zeros((1, num_classes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        batch_size = len(y)

        # Forward
        h = relu(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)

        # Loss
        y_onehot = np.eye(probs.shape[1])[y]
        loss = cross_entropy(probs, y_onehot)

        # Backward
        d_logits = (probs - y_onehot) / batch_size
        d_W2 = h.T @ d_logits
        d_b2 = np.sum(d_logits, axis=0, keepdims=True)

        d_h = d_logits @ self.W2.T
        d_h = d_h * (h > 0)
        d_W1 = X.T @ d_h
        d_b1 = np.sum(d_h, axis=0, keepdims=True)

        # Update
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

        return loss


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def create_difficulty_dataset(n_samples: int = 1000, input_dim: int = 20,
                              num_classes: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset with known difficulty levels.

    Easy examples: Clean, well-separated
    Hard examples: Noisy, overlapping classes
    """
    np.random.seed(42)

    X = np.zeros((n_samples, input_dim))
    y = np.zeros(n_samples, dtype=int)
    difficulties = np.zeros(n_samples)

    samples_per_class = n_samples // num_classes

    for c in range(num_classes):
        start_idx = c * samples_per_class
        end_idx = (c + 1) * samples_per_class

        # Class center
        center = np.random.randn(input_dim) * 3

        for i in range(start_idx, end_idx):
            # Difficulty increases with index within class
            difficulty = (i - start_idx) / samples_per_class

            # Add noise based on difficulty
            noise_level = difficulty * 2
            X[i] = center + np.random.randn(input_dim) * noise_level
            y[i] = c
            difficulties[i] = difficulty

    return X, y, difficulties


def experiment_curriculum_vs_random(n_epochs: int = 50,
                                    n_samples: int = 1000) -> dict:
    """
    Compare curriculum learning vs random sampling.

    WHAT TO OBSERVE:
    - Curriculum: Faster early progress, better final accuracy
    - Random: Slower, may get stuck in bad minima
    """
    print("=" * 60)
    print("EXPERIMENT: Curriculum vs Random Training")
    print("=" * 60)

    X, y, difficulties = create_difficulty_dataset(n_samples)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    results = {
        'epochs': list(range(n_epochs)),
        'curriculum_loss': [],
        'random_loss': [],
        'curriculum_acc': [],
        'random_acc': []
    }

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    diff_train = difficulties[:split]

    # Model with curriculum
    model_curr = SimpleClassifier(input_dim, 32, num_classes)
    scheduler = LinearScheduler(start_fraction=0.2, warmup_epochs=n_epochs // 2)
    trainer = CurriculumTrainer(model_curr, scheduler, None)

    # Model with random
    model_rand = SimpleClassifier(input_dim, 32, num_classes)

    for epoch in range(n_epochs):
        # Curriculum training
        X_batch, y_batch = trainer.get_curriculum_batch(
            X_train, y_train, diff_train, epoch, batch_size=64
        )
        loss_curr = model_curr.train_step(X_batch, y_batch, lr=0.1)

        # Random training
        rand_idx = np.random.choice(len(X_train), size=64, replace=False)
        loss_rand = model_rand.train_step(X_train[rand_idx], y_train[rand_idx], lr=0.1)

        # Evaluate
        pred_curr = np.argmax(model_curr.forward(X_test), axis=1)
        acc_curr = np.mean(pred_curr == y_test)

        pred_rand = np.argmax(model_rand.forward(X_test), axis=1)
        acc_rand = np.mean(pred_rand == y_test)

        results['curriculum_loss'].append(loss_curr)
        results['random_loss'].append(loss_rand)
        results['curriculum_acc'].append(acc_curr)
        results['random_acc'].append(acc_rand)

        if epoch % 10 == 0:
            frac = scheduler.get_data_fraction(epoch)
            print(f"Epoch {epoch:3d}: Curriculum (frac={frac:.2f}) acc={acc_curr:.3f}, "
                  f"Random acc={acc_rand:.3f}")

    return results


def experiment_scheduler_comparison(n_epochs: int = 50,
                                    n_samples: int = 1000) -> dict:
    """
    Compare different curriculum schedulers.

    WHAT TO OBSERVE:
    - Linear: Steady progress
    - Exponential: Fast inclusion of all data
    - Step: Discrete jumps in difficulty
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Scheduler Comparison")
    print("=" * 60)

    X, y, difficulties = create_difficulty_dataset(n_samples)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    diff_train = difficulties[:split]

    schedulers = {
        'linear': LinearScheduler(start_fraction=0.2, warmup_epochs=n_epochs // 2),
        'exponential': ExponentialScheduler(start_fraction=0.1, growth_rate=1.1),
        'step': StepScheduler([(0, 0.2), (10, 0.4), (20, 0.6), (30, 0.8), (40, 1.0)])
    }

    results = {'epochs': list(range(n_epochs))}

    for name, scheduler in schedulers.items():
        model = SimpleClassifier(input_dim, 32, num_classes)
        trainer = CurriculumTrainer(model, scheduler, None)

        accs = []
        fracs = []

        for epoch in range(n_epochs):
            X_batch, y_batch = trainer.get_curriculum_batch(
                X_train, y_train, diff_train, epoch, batch_size=64
            )
            model.train_step(X_batch, y_batch, lr=0.1)

            pred = np.argmax(model.forward(X_test), axis=1)
            acc = np.mean(pred == y_test)
            accs.append(acc)
            fracs.append(scheduler.get_data_fraction(epoch))

        results[f'{name}_acc'] = accs
        results[f'{name}_frac'] = fracs

        print(f"{name:12}: Final accuracy = {accs[-1]:.3f}")

    return results


def experiment_self_paced(n_epochs: int = 50, n_samples: int = 1000) -> dict:
    """
    Self-paced learning experiment.

    WHAT TO OBSERVE:
    - Model automatically selects learnable examples
    - Threshold increases over time
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Self-Paced Learning")
    print("=" * 60)

    X, y, difficulties = create_difficulty_dataset(n_samples)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SimpleClassifier(input_dim, 32, num_classes)
    spl = SelfPacedLearning(model, initial_threshold=2.0, threshold_growth=1.05)

    results = {
        'epochs': [],
        'threshold': [],
        'num_selected': [],
        'accuracy': []
    }

    for epoch in range(n_epochs):
        # Get weights (which examples to use)
        weights = spl.compute_weights(X_train, y_train, model.forward)
        selected = np.where(weights > 0)[0]

        if len(selected) > 0:
            # Train on selected examples
            batch_idx = np.random.choice(selected, size=min(64, len(selected)), replace=False)
            model.train_step(X_train[batch_idx], y_train[batch_idx], lr=0.1)

        # Evaluate
        pred = np.argmax(model.forward(X_test), axis=1)
        acc = np.mean(pred == y_test)

        results['epochs'].append(epoch)
        results['threshold'].append(spl.threshold)
        results['num_selected'].append(len(selected))
        results['accuracy'].append(acc)

        # Increase threshold
        spl.update_threshold()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Threshold={spl.threshold:.2f}, "
                  f"Selected={len(selected):4d}/{len(X_train)}, Acc={acc:.3f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_curriculum_concept(save_path: Optional[str] = None):
    """
    Visual explanation of curriculum learning.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. The concept
    ax = axes[0]
    ax.set_title('Curriculum Learning: Core Idea', fontweight='bold')

    stages = [
        ('🅰️', 'Easy\nExamples', 'lightgreen'),
        ('🅱️', 'Medium\nExamples', 'yellow'),
        ('🅲️', 'Hard\nExamples', 'lightcoral')
    ]

    for i, (icon, label, color) in enumerate(stages):
        rect = plt.Rectangle((i * 2, 0.5), 1.5, 1.5, facecolor=color,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.annotate(label, (i * 2 + 0.75, 1.25), ha='center', va='center', fontsize=10)

        if i < 2:
            ax.annotate('→', (i * 2 + 1.7, 1.25), fontsize=20, ha='center')

    ax.annotate('Time / Epochs', (3, 0), ha='center', fontsize=11)

    ax.text(3, 2.5, '"Start with what you can learn,\nthen build complexity"',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    # 2. Data fraction over time
    ax = axes[1]
    ax.set_title('Data Fraction Over Training', fontweight='bold')

    epochs = np.arange(50)

    # Different schedulers
    linear = LinearScheduler(0.2, 25)
    exp = ExponentialScheduler(0.1, 1.1)
    step = StepScheduler([(0, 0.2), (10, 0.4), (25, 0.7), (40, 1.0)])

    ax.plot(epochs, [linear.get_data_fraction(e) for e in epochs],
           'b-', label='Linear', linewidth=2)
    ax.plot(epochs, [exp.get_data_fraction(e) for e in epochs],
           'g-', label='Exponential', linewidth=2)
    ax.plot(epochs, [step.get_data_fraction(e) for e in epochs],
           'r-', label='Step', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fraction of Data Used')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Human analogy
    ax = axes[2]
    ax.set_title('Like Human Learning', fontweight='bold')

    examples = [
        ('Math:', '1+1 → Algebra → Calculus'),
        ('Language:', 'Letters → Words → Sentences'),
        ('Music:', 'Notes → Scales → Symphonies'),
        ('Sports:', 'Walk → Run → Marathon'),
    ]

    for i, (domain, progression) in enumerate(examples):
        y = 0.8 - i * 0.2
        ax.text(0.05, y, domain, fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.25, y, progression, fontsize=10, transform=ax.transAxes)

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_curriculum_results(results: dict, save_path: Optional[str] = None):
    """
    Visualize curriculum vs random training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = results['epochs']

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, results['curriculum_loss'], 'g-', label='Curriculum', linewidth=2)
    ax.plot(epochs, results['random_loss'], 'r-', label='Random', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss: Curriculum vs Random', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, results['curriculum_acc'], 'g-', label='Curriculum', linewidth=2)
    ax.plot(epochs, results['random_acc'], 'r-', label='Random', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy: Curriculum vs Random', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Curriculum Learning Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_self_paced(results: dict, save_path: Optional[str] = None):
    """
    Visualize self-paced learning dynamics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = results['epochs']

    # Threshold over time
    ax = axes[0]
    ax.plot(epochs, results['threshold'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Threshold')
    ax.set_title('Self-Paced: Loss Threshold', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Number of selected examples
    ax = axes[1]
    ax.plot(epochs, results['num_selected'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Examples Selected')
    ax.set_title('Self-Paced: Selected Examples', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[2]
    ax.plot(epochs, results['accuracy'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Self-Paced: Test Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Self-Paced Learning: Model Decides Its Curriculum',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_difficulty_examples(save_path: Optional[str] = None):
    """
    Show what easy vs hard examples look like.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image difficulty
    ax = axes[0]
    ax.set_title('Image Difficulty Examples', fontweight='bold')

    difficulties = ['Easy', 'Medium', 'Hard']
    descriptions = [
        'Clear, centered\nGood lighting',
        'Partial occlusion\nSome noise',
        'Heavy occlusion\nPoor lighting\nAmbiguous'
    ]
    colors = ['lightgreen', 'yellow', 'lightcoral']

    for i, (diff, desc, color) in enumerate(zip(difficulties, descriptions, colors)):
        rect = plt.Rectangle((i * 1.5, 0), 1.2, 2, facecolor=color,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.annotate(diff, (i * 1.5 + 0.6, 1.7), ha='center', fontsize=11,
                   fontweight='bold')
        ax.annotate(desc, (i * 1.5 + 0.6, 0.8), ha='center', fontsize=9)

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    # Text difficulty
    ax = axes[1]
    ax.set_title('Text Difficulty Examples', fontweight='bold')

    examples = [
        ('Easy', '"The cat sat."', 'Short, clear'),
        ('Medium', '"The quick brown fox..."', 'Longer, standard'),
        ('Hard', '"Despite the ostensible..."', 'Complex, rare words'),
    ]

    for i, (diff, text, note) in enumerate(examples):
        y = 2 - i * 0.8
        ax.annotate(f'{diff}:', (0.1, y), fontsize=11, fontweight='bold')
        ax.annotate(text, (0.4, y), fontsize=10)
        ax.annotate(f'({note})', (2.5, y), fontsize=9, color='gray')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.5)
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
    print("CURRICULUM LEARNING — Paradigm: EASY TO HARD")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Train on EASIER examples first, gradually introduce HARDER ones.
    Just like humans learn!

    KEY QUESTIONS:
    1. How to measure difficulty?
       - Model's loss (high = hard)
       - Data properties (noise, length, ambiguity)
       - Human annotation

    2. How to schedule?
       - Linear: Steady increase
       - Exponential: Fast inclusion
       - Self-paced: Model decides

    VARIANTS:
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Variant             │ Description                         │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Predefined          │ Fixed difficulty order              │
    │ Self-Paced          │ Model selects learnable examples    │
    │ Teacher-Student     │ Teacher guides curriculum           │
    │ Anti-Curriculum     │ Hard first (sometimes works!)       │
    └─────────────────────┴─────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Curriculum vs Random
    curr_results = experiment_curriculum_vs_random()

    # Experiment 2: Scheduler comparison
    sched_results = experiment_scheduler_comparison()

    # Experiment 3: Self-paced
    spl_results = experiment_self_paced()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_curriculum_concept('53_curriculum_learning_concept.png')
    visualize_curriculum_results(curr_results, '53_curriculum_learning_results.png')
    visualize_self_paced(spl_results, '53_curriculum_learning_selfpaced.png')
    visualize_difficulty_examples('53_curriculum_learning_difficulty.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. CURRICULUM = Start easy, add difficulty gradually
       - Mimics human learning
       - Better optimization landscape

    2. DIFFICULTY METRICS matter
       - Loss-based: Easy to compute
       - Property-based: Domain knowledge
       - Self-paced: Model decides

    3. SCHEDULER CHOICE affects learning
       - Linear: Steady progress
       - Exponential: Quick to full data
       - Step: Discrete stages

    4. SELF-PACED is adaptive
       - No manual difficulty labels needed
       - Model learns at its own pace
       - Threshold controls progression

    5. ANTI-CURRICULUM sometimes works
       - When easy examples teach shortcuts
       - When hard examples are more informative
    """)
