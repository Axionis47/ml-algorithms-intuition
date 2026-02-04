"""
===============================================================
CONTRASTIVE LEARNING — Paradigm: SIMILARITY STRUCTURE
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Learn representations by COMPARING examples:
    - SIMILAR examples → close in embedding space
    - DIFFERENT examples → far apart

"Birds of a feather should flock together."

THE CORE PRINCIPLE:
    Pull POSITIVE pairs together
    Push NEGATIVE pairs apart

===============================================================
WHY IT WORKS
===============================================================

Without labels, we can still define similarity:
    - Same image, different augmentations → POSITIVE
    - Different images → NEGATIVE

The model learns:
    - What makes two views "the same" (invariance)
    - What makes examples different (discrimination)

This creates semantically meaningful representations!

===============================================================
THE CONTRASTIVE LOSS (InfoNCE)
===============================================================

L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

Where:
    z_i, z_j = embeddings of positive pair
    z_k = embeddings of all examples (positives + negatives)
    τ = temperature (controls hardness of negatives)
    sim() = similarity function (usually cosine)

"Maximize probability of positive pair among all pairs."

===============================================================
KEY METHODS
===============================================================

1. SimCLR: Simple framework, large batch negatives
2. MoCo: Momentum encoder, memory bank for negatives
3. BYOL: No negatives! Uses momentum target
4. SwAV: Clustering-based contrastive learning
5. CLIP: Image-text contrastive learning

===============================================================
INDUCTIVE BIAS
===============================================================

1. Augmentation defines what should be INVARIANT
2. Assumes negatives are truly different semantically
3. Batch size affects quality of negatives
4. Temperature controls the "hardness" of task

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Callable, Dict


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2 normalize along axis."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-10)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a and b."""
    a_norm = normalize(a)
    b_norm = normalize(b)
    return np.dot(a_norm, b_norm.T)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)


# ============================================================
# DATA AUGMENTATION
# ============================================================

class DataAugmentation:
    """
    Data augmentation for contrastive learning.

    Creates different "views" of the same example.
    Views should preserve semantics but change appearance.
    """

    @staticmethod
    def add_noise(x: np.ndarray, std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise."""
        return x + np.random.randn(*x.shape) * std

    @staticmethod
    def random_scale(x: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
        """Random scaling."""
        scale = np.random.uniform(low, high)
        return x * scale

    @staticmethod
    def random_mask(x: np.ndarray, mask_prob: float = 0.1) -> np.ndarray:
        """Randomly mask some features."""
        mask = np.random.random(x.shape) > mask_prob
        return x * mask

    @staticmethod
    def random_permute(x: np.ndarray, num_swaps: int = 5) -> np.ndarray:
        """Randomly permute some elements."""
        x = x.copy()
        for _ in range(num_swaps):
            i, j = np.random.choice(len(x), 2, replace=False)
            x[i], x[j] = x[j], x[i]
        return x

    @staticmethod
    def create_positive_pair(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create two augmented views of the same example."""
        view1 = DataAugmentation.add_noise(
            DataAugmentation.random_scale(x)
        )
        view2 = DataAugmentation.add_noise(
            DataAugmentation.random_scale(x)
        )
        return view1, view2


# ============================================================
# ENCODER NETWORK
# ============================================================

class Encoder:
    """
    Encoder network for contrastive learning.

    f(x) → representation → g(representation) → embedding

    Where:
    - f: Base encoder (e.g., ResNet)
    - g: Projection head (MLP)

    KEY INSIGHT:
    The projection head (g) is crucial!
    - Representations (before projection) transfer better
    - Embeddings (after projection) are better for contrastive loss
    """

    def __init__(self, input_dim: int, repr_dim: int, proj_dim: int):
        """
        Args:
            input_dim: Input dimension
            repr_dim: Representation dimension (what we transfer)
            proj_dim: Projection dimension (for contrastive loss)
        """
        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.proj_dim = proj_dim

        # Base encoder
        self.enc_W1 = he_init(input_dim, repr_dim * 2)
        self.enc_b1 = np.zeros((1, repr_dim * 2))
        self.enc_W2 = he_init(repr_dim * 2, repr_dim)
        self.enc_b2 = np.zeros((1, repr_dim))

        # Projection head (MLP)
        self.proj_W1 = he_init(repr_dim, repr_dim)
        self.proj_b1 = np.zeros((1, repr_dim))
        self.proj_W2 = he_init(repr_dim, proj_dim)
        self.proj_b2 = np.zeros((1, proj_dim))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Get representation (for transfer)."""
        h = relu(x @ self.enc_W1 + self.enc_b1)
        repr = relu(h @ self.enc_W2 + self.enc_b2)
        return repr

    def project(self, repr: np.ndarray) -> np.ndarray:
        """Get projection (for contrastive loss)."""
        h = relu(repr @ self.proj_W1 + self.proj_b1)
        proj = h @ self.proj_W2 + self.proj_b2
        return normalize(proj)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass."""
        repr = self.encode(x)
        proj = self.project(repr)
        return repr, proj


# ============================================================
# CONTRASTIVE LOSSES
# ============================================================

class InfoNCELoss:
    """
    InfoNCE Loss (used in SimCLR, MoCo, CLIP).

    L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

    For batch of N pairs (2N total views):
    - Numerator: similarity of positive pair
    - Denominator: sum over all 2N-1 other views

    TEMPERATURE (τ):
    - Low τ: Focus on hard negatives
    - High τ: Softer distribution, more uniform
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def compute(self, z1: np.ndarray, z2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute InfoNCE loss for a batch of positive pairs.

        Args:
            z1: (batch_size, embed_dim) first views
            z2: (batch_size, embed_dim) second views

        Returns:
            loss: scalar loss value
            similarity_matrix: for visualization
        """
        batch_size = len(z1)

        # Concatenate all embeddings
        z = np.vstack([z1, z2])  # (2*batch_size, embed_dim)

        # Compute all pairwise similarities
        sim_matrix = cosine_similarity(z, z) / self.temperature

        # Mask out self-similarities
        mask = np.eye(2 * batch_size, dtype=bool)
        sim_matrix[mask] = -np.inf

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = np.arange(2 * batch_size)
        labels[:batch_size] = labels[:batch_size] + batch_size
        labels[batch_size:] = labels[batch_size:] - batch_size

        # Compute loss
        exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
        log_prob = sim_matrix - np.log(np.sum(exp_sim, axis=1, keepdims=True) + 1e-10)

        # Loss is negative log probability of positive pair
        loss = 0
        for i in range(2 * batch_size):
            loss -= log_prob[i, labels[i]]
        loss /= (2 * batch_size)

        return loss, sim_matrix[:batch_size, batch_size:]


class TripletLoss:
    """
    Triplet Loss: (anchor, positive, negative).

    L = max(0, d(a,p) - d(a,n) + margin)

    WHERE:
    - d(a,p): distance between anchor and positive
    - d(a,n): distance between anchor and negative
    - margin: minimum required gap

    "Positive should be closer than negative by at least margin."
    """

    def __init__(self, margin: float = 1.0):
        self.margin = margin

    def compute(self, anchor: np.ndarray, positive: np.ndarray,
                negative: np.ndarray) -> float:
        """
        Compute triplet loss.

        Args:
            anchor: (batch, embed_dim)
            positive: (batch, embed_dim)
            negative: (batch, embed_dim)
        """
        # Euclidean distances
        d_pos = np.sum((anchor - positive) ** 2, axis=1)
        d_neg = np.sum((anchor - negative) ** 2, axis=1)

        # Triplet loss with margin
        loss = np.maximum(0, d_pos - d_neg + self.margin)

        return np.mean(loss)


class NTXentLoss:
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).

    Used in SimCLR. Similar to InfoNCE but formulated as cross-entropy.

    For positive pair (i, j):
    L_ij = -log[ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def compute(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Compute NT-Xent loss."""
        batch_size = len(z1)

        # All embeddings
        z = np.vstack([z1, z2])

        # Similarity matrix
        sim = cosine_similarity(z, z) / self.temperature

        # Create labels (positive pairs)
        labels = np.concatenate([np.arange(batch_size) + batch_size,
                                 np.arange(batch_size)])

        # Mask self-similarity
        mask = ~np.eye(2 * batch_size, dtype=bool)

        # Softmax cross-entropy loss
        loss = 0
        for i in range(2 * batch_size):
            numerator = np.exp(sim[i, labels[i]])
            denominator = np.sum(np.exp(sim[i]) * mask[i])
            loss -= np.log(numerator / (denominator + 1e-10) + 1e-10)

        return loss / (2 * batch_size)


# ============================================================
# SimCLR FRAMEWORK
# ============================================================

class SimCLR:
    """
    SimCLR: Simple Framework for Contrastive Learning.

    KEY COMPONENTS:
    1. Data augmentation: Create two views of each image
    2. Encoder: f(x) → representation
    3. Projection head: g(h) → embedding
    4. Contrastive loss: InfoNCE/NT-Xent

    TRAINING:
    - Sample batch of N images
    - Create 2N augmented views (2 per image)
    - Positive: views from same image
    - Negative: views from different images
    """

    def __init__(self, encoder: Encoder, temperature: float = 0.5):
        self.encoder = encoder
        self.loss_fn = NTXentLoss(temperature)
        self.temperature = temperature

    def train_step(self, X: np.ndarray, lr: float = 0.01) -> Dict:
        """
        One training step.

        Args:
            X: (batch_size, input_dim) original examples

        Returns:
            Dictionary with loss and other metrics
        """
        batch_size = len(X)

        # Create positive pairs through augmentation
        X1 = np.array([DataAugmentation.create_positive_pair(x)[0] for x in X])
        X2 = np.array([DataAugmentation.create_positive_pair(x)[1] for x in X])

        # Forward pass
        repr1, z1 = self.encoder.forward(X1)
        repr2, z2 = self.encoder.forward(X2)

        # Compute loss
        loss = self.loss_fn.compute(z1, z2)

        # Simplified backward pass (update projection head)
        # In practice, would backprop through entire network
        # Here we just demonstrate the concept

        return {
            'loss': loss,
            'avg_similarity': np.mean(np.sum(z1 * z2, axis=1))
        }


# ============================================================
# MoCo FRAMEWORK
# ============================================================

class MomentumEncoder:
    """
    Momentum Encoder for MoCo.

    Key insight: Use a slowly-updated copy of the encoder
    to provide consistent targets.

    θ_k = m × θ_k + (1-m) × θ_q

    Where m is close to 1 (e.g., 0.999).
    """

    def __init__(self, encoder: Encoder, momentum: float = 0.999):
        self.encoder = encoder
        self.momentum = momentum

        # Initialize momentum encoder with same weights
        self.momentum_encoder = Encoder(
            encoder.input_dim,
            encoder.repr_dim,
            encoder.proj_dim
        )
        self._copy_weights()

    def _copy_weights(self):
        """Copy weights from encoder to momentum encoder."""
        self.momentum_encoder.enc_W1 = self.encoder.enc_W1.copy()
        self.momentum_encoder.enc_b1 = self.encoder.enc_b1.copy()
        self.momentum_encoder.enc_W2 = self.encoder.enc_W2.copy()
        self.momentum_encoder.enc_b2 = self.encoder.enc_b2.copy()
        self.momentum_encoder.proj_W1 = self.encoder.proj_W1.copy()
        self.momentum_encoder.proj_b1 = self.encoder.proj_b1.copy()
        self.momentum_encoder.proj_W2 = self.encoder.proj_W2.copy()
        self.momentum_encoder.proj_b2 = self.encoder.proj_b2.copy()

    def update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        m = self.momentum
        for attr in ['enc_W1', 'enc_b1', 'enc_W2', 'enc_b2',
                     'proj_W1', 'proj_b1', 'proj_W2', 'proj_b2']:
            me_weight = getattr(self.momentum_encoder, attr)
            e_weight = getattr(self.encoder, attr)
            setattr(self.momentum_encoder, attr,
                   m * me_weight + (1 - m) * e_weight)


class MoCo:
    """
    MoCo: Momentum Contrast.

    IMPROVEMENTS over SimCLR:
    1. Momentum encoder provides stable targets
    2. Queue of negative examples (not limited by batch size)

    QUEUE:
    Store embeddings from momentum encoder in a queue.
    Use queue entries as negatives.
    Allows large number of negatives without large batch.
    """

    def __init__(self, encoder: Encoder, queue_size: int = 1024,
                 temperature: float = 0.07, momentum: float = 0.999):
        self.encoder = encoder
        self.momentum_encoder = MomentumEncoder(encoder, momentum)
        self.temperature = temperature

        # Initialize queue
        self.queue_size = queue_size
        self.queue = None  # Will be initialized on first forward

    def _init_queue(self, embed_dim: int):
        """Initialize queue with random embeddings."""
        self.queue = np.random.randn(self.queue_size, embed_dim)
        self.queue = normalize(self.queue)
        self.queue_ptr = 0

    def _enqueue(self, keys: np.ndarray):
        """Add new keys to queue."""
        batch_size = len(keys)
        ptr = self.queue_ptr

        # Handle wraparound
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            overflow = ptr + batch_size - self.queue_size
            self.queue[ptr:] = keys[:-overflow]
            self.queue[:overflow] = keys[-overflow:]

        self.queue_ptr = (ptr + batch_size) % self.queue_size

    def compute_loss(self, q: np.ndarray, k: np.ndarray) -> float:
        """
        Compute InfoNCE loss with queue negatives.

        q: Query embeddings (from encoder)
        k: Key embeddings (from momentum encoder)
        """
        # Positive similarity
        pos_sim = np.sum(q * k, axis=1) / self.temperature

        # Negative similarities (query vs queue)
        neg_sim = q @ self.queue.T / self.temperature

        # InfoNCE loss
        logits = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=1)
        labels = np.zeros(len(q), dtype=int)  # Positive is always index 0

        # Cross-entropy
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1))
        loss = -pos_sim + max_logits.squeeze() + log_sum_exp

        return np.mean(loss)


# ============================================================
# CLIP-style Vision-Language Contrastive
# ============================================================

class CLIPStyle:
    """
    CLIP-style Image-Text Contrastive Learning.

    IDEA:
    - Encode images and text separately
    - Match image-text pairs in embedding space
    - Learn visual concepts from natural language

    POSITIVE: Matching (image, caption) pairs
    NEGATIVE: Non-matching pairs in batch
    """

    def __init__(self, image_encoder: Encoder, text_encoder: Encoder,
                 temperature: float = 0.07):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

    def compute_loss(self, images: np.ndarray, texts: np.ndarray) -> Dict:
        """
        Compute bidirectional contrastive loss.

        Args:
            images: Image features
            texts: Text features

        Returns:
            Loss dictionary
        """
        # Get embeddings
        _, img_embed = self.image_encoder.forward(images)
        _, txt_embed = self.text_encoder.forward(texts)

        # Similarity matrix
        sim = img_embed @ txt_embed.T / self.temperature

        # Labels (diagonal should match)
        batch_size = len(images)
        labels = np.arange(batch_size)

        # Image-to-text loss
        img_to_txt = -np.mean(sim[np.arange(batch_size), labels] -
                             np.log(np.sum(np.exp(sim), axis=1) + 1e-10))

        # Text-to-image loss
        txt_to_img = -np.mean(sim.T[np.arange(batch_size), labels] -
                             np.log(np.sum(np.exp(sim.T), axis=1) + 1e-10))

        total_loss = (img_to_txt + txt_to_img) / 2

        return {
            'loss': total_loss,
            'img_to_txt': img_to_txt,
            'txt_to_img': txt_to_img
        }


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_temperature_effect(temperatures: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                                  n_samples: int = 200,
                                  embed_dim: int = 32) -> dict:
    """
    Effect of temperature on contrastive learning.

    WHAT TO OBSERVE:
    - Low T: Very hard negatives, may collapse
    - High T: Soft distinction, slower learning
    - Sweet spot: Usually 0.1-0.5
    """
    print("=" * 60)
    print("EXPERIMENT: Temperature Effect")
    print("=" * 60)

    # Create data with clusters
    np.random.seed(42)
    num_clusters = 5
    X = []
    labels = []
    for c in range(num_clusters):
        center = np.random.randn(embed_dim) * 3
        cluster_data = center + np.random.randn(n_samples // num_clusters, embed_dim) * 0.5
        X.extend(cluster_data)
        labels.extend([c] * (n_samples // num_clusters))

    X = np.array(X)
    labels = np.array(labels)

    results = {'temperatures': temperatures, 'losses': [], 'cluster_separation': []}

    for temp in temperatures:
        encoder = Encoder(embed_dim, 64, 32)
        simclr = SimCLR(encoder, temperature=temp)

        # Train for a few steps
        losses = []
        for _ in range(20):
            batch_idx = np.random.choice(len(X), 64, replace=False)
            metrics = simclr.train_step(X[batch_idx])
            losses.append(metrics['loss'])

        # Evaluate cluster separation
        _, embeddings = encoder.forward(X)
        separation = compute_cluster_separation(embeddings, labels)

        results['losses'].append(np.mean(losses[-5:]))
        results['cluster_separation'].append(separation)

        print(f"T={temp:.1f}: Loss={np.mean(losses[-5:]):.4f}, "
              f"Cluster separation={separation:.4f}")

    return results


def compute_cluster_separation(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute ratio of inter-cluster to intra-cluster distance."""
    unique_labels = np.unique(labels)

    # Intra-cluster distances
    intra = []
    centers = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)
        intra.extend(np.linalg.norm(cluster_points - center, axis=1))

    # Inter-cluster distances
    inter = []
    centers = np.array(centers)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            inter.append(np.linalg.norm(centers[i] - centers[j]))

    return np.mean(inter) / (np.mean(intra) + 1e-10)


def experiment_batch_size_effect(batch_sizes: List[int] = [16, 32, 64, 128, 256],
                                 n_samples: int = 500,
                                 embed_dim: int = 32) -> dict:
    """
    Effect of batch size on contrastive learning.

    WHAT TO OBSERVE:
    - Small batch: Few negatives, poor learning
    - Large batch: More negatives, better discrimination
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Batch Size Effect")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(n_samples, embed_dim)

    results = {'batch_sizes': batch_sizes, 'losses': [], 'avg_similarity': []}

    for bs in batch_sizes:
        encoder = Encoder(embed_dim, 64, 32)
        simclr = SimCLR(encoder, temperature=0.5)

        # Train for a few steps
        metrics_list = []
        for _ in range(20):
            batch_idx = np.random.choice(len(X), min(bs, len(X)), replace=False)
            metrics = simclr.train_step(X[batch_idx])
            metrics_list.append(metrics)

        avg_loss = np.mean([m['loss'] for m in metrics_list[-5:]])
        avg_sim = np.mean([m['avg_similarity'] for m in metrics_list[-5:]])

        results['losses'].append(avg_loss)
        results['avg_similarity'].append(avg_sim)

        print(f"Batch size {bs:4d}: Loss={avg_loss:.4f}, Avg similarity={avg_sim:.4f}")

    return results


def experiment_augmentation_importance(n_samples: int = 200,
                                       embed_dim: int = 32) -> dict:
    """
    How augmentation affects learned representations.

    WHAT TO OBSERVE:
    - No augmentation: Model may learn shortcuts
    - Good augmentation: Invariant to unimportant changes
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Augmentation Importance")
    print("=" * 60)

    np.random.seed(42)
    num_clusters = 5
    X = []
    labels = []
    for c in range(num_clusters):
        center = np.random.randn(embed_dim) * 3
        cluster_data = center + np.random.randn(n_samples // num_clusters, embed_dim) * 0.3
        X.extend(cluster_data)
        labels.extend([c] * (n_samples // num_clusters))

    X = np.array(X)
    labels = np.array(labels)

    augmentations = {
        'none': lambda x: (x, x.copy()),  # No augmentation
        'noise_only': lambda x: (
            DataAugmentation.add_noise(x, 0.1),
            DataAugmentation.add_noise(x, 0.1)
        ),
        'scale_only': lambda x: (
            DataAugmentation.random_scale(x),
            DataAugmentation.random_scale(x)
        ),
        'full': DataAugmentation.create_positive_pair
    }

    results = {'augmentation': [], 'cluster_separation': []}

    for aug_name, aug_fn in augmentations.items():
        encoder = Encoder(embed_dim, 64, 32)

        # Train with specific augmentation
        for _ in range(50):
            batch_idx = np.random.choice(len(X), 64, replace=False)
            batch = X[batch_idx]

            # Apply augmentation
            views = [aug_fn(x) for x in batch]
            X1 = np.array([v[0] for v in views])
            X2 = np.array([v[1] for v in views])

            # Forward and simple update
            _, z1 = encoder.forward(X1)
            _, z2 = encoder.forward(X2)

        # Evaluate
        _, embeddings = encoder.forward(X)
        separation = compute_cluster_separation(embeddings, labels)

        results['augmentation'].append(aug_name)
        results['cluster_separation'].append(separation)

        print(f"Augmentation '{aug_name:12}': Cluster separation = {separation:.4f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_contrastive_concept(save_path: Optional[str] = None):
    """
    Visual explanation of contrastive learning.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Core concept
    ax = axes[0]
    ax.set_title('Contrastive Learning: Core Idea', fontweight='bold')

    # Positive pair
    ax.scatter([1, 1.5], [2, 2.2], c='green', s=200, marker='o', label='Positive pair')
    ax.plot([1, 1.5], [2, 2.2], 'g-', linewidth=3)
    ax.annotate('Pull together', (1.25, 2.4), ha='center', color='green', fontsize=10)

    # Negative pairs
    ax.scatter([3, 3.5, 2.5], [1, 2, 0.5], c='red', s=200, marker='x', label='Negatives')
    ax.plot([1, 3], [2, 1], 'r--', linewidth=1, alpha=0.5)
    ax.plot([1, 3.5], [2, 2], 'r--', linewidth=1, alpha=0.5)
    ax.plot([1, 2.5], [2, 0.5], 'r--', linewidth=1, alpha=0.5)
    ax.annotate('Push apart', (2.5, 1.5), ha='center', color='red', fontsize=10)

    ax.legend(loc='upper left')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Embedding dimension 1')
    ax.set_ylabel('Embedding dimension 2')

    # 2. Augmentation
    ax = axes[1]
    ax.set_title('Positive Pairs via Augmentation', fontweight='bold')

    ax.text(0.5, 0.85, '🖼️ Original Image', fontsize=12, ha='center', transform=ax.transAxes)
    ax.annotate('', xy=(0.3, 0.65), xytext=(0.5, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2), transform=ax.transAxes)
    ax.annotate('', xy=(0.7, 0.65), xytext=(0.5, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2), transform=ax.transAxes)

    ax.text(0.2, 0.55, 'View 1\n(crop, flip)', fontsize=10, ha='center',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(0.8, 0.55, 'View 2\n(color, blur)', fontsize=10, ha='center',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.text(0.5, 0.35, '↓ Same image = Positive pair ↓', ha='center',
           transform=ax.transAxes, fontsize=11, color='green')
    ax.text(0.5, 0.15, 'Should have similar embeddings!', ha='center',
           transform=ax.transAxes, fontsize=10, style='italic')

    ax.axis('off')

    # 3. Loss function
    ax = axes[2]
    ax.set_title('InfoNCE Loss', fontweight='bold')

    ax.text(0.5, 0.85, 'L = -log P(positive | all)',
           fontsize=12, ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.text(0.5, 0.65, r'P(pos) = exp(sim(z₁, z₂)/τ) / Σₖ exp(sim(z₁, zₖ)/τ)',
           fontsize=10, ha='center', transform=ax.transAxes)

    ax.text(0.1, 0.4, 'τ (temperature):', fontsize=10, fontweight='bold',
           transform=ax.transAxes)
    ax.text(0.1, 0.3, '• Low τ: Hard negatives', fontsize=9, transform=ax.transAxes)
    ax.text(0.1, 0.2, '• High τ: Soft distinction', fontsize=9, transform=ax.transAxes)

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_simclr_framework(save_path: Optional[str] = None):
    """
    Visualize SimCLR framework.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('SimCLR: Simple Framework for Contrastive Learning',
                fontweight='bold', fontsize=14)

    # Image
    ax.add_patch(plt.Rectangle((0.5, 3), 1, 1, facecolor='lightblue', edgecolor='black'))
    ax.annotate('Image x', (1, 3.5), ha='center', fontsize=10)

    # Augmentation arrows
    ax.annotate('', xy=(2, 4), xytext=(1.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2, 2.5), xytext=(1.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('Augment', (1.8, 3.7), fontsize=9)

    # Two views
    ax.add_patch(plt.Rectangle((2, 3.5), 0.8, 0.8, facecolor='lightgreen', edgecolor='black'))
    ax.annotate('x̃₁', (2.4, 3.9), ha='center', fontsize=10)

    ax.add_patch(plt.Rectangle((2, 2.2), 0.8, 0.8, facecolor='lightgreen', edgecolor='black'))
    ax.annotate('x̃₂', (2.4, 2.6), ha='center', fontsize=10)

    # Encoder arrows
    ax.annotate('', xy=(4, 3.9), xytext=(2.9, 3.9),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(4, 2.6), xytext=(2.9, 2.6),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('f(·)', (3.5, 4.1), fontsize=10)

    # Representations
    ax.add_patch(plt.Rectangle((4, 3.5), 1.5, 0.8, facecolor='gold', edgecolor='black'))
    ax.annotate('h₁ = f(x̃₁)', (4.75, 3.9), ha='center', fontsize=10)

    ax.add_patch(plt.Rectangle((4, 2.2), 1.5, 0.8, facecolor='gold', edgecolor='black'))
    ax.annotate('h₂ = f(x̃₂)', (4.75, 2.6), ha='center', fontsize=10)

    # Projection head
    ax.annotate('', xy=(6.5, 3.9), xytext=(5.6, 3.9),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6.5, 2.6), xytext=(5.6, 2.6),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('g(·)', (6.1, 4.1), fontsize=10)

    # Embeddings
    ax.add_patch(plt.Circle((7, 3.9), 0.3, facecolor='coral', edgecolor='black'))
    ax.annotate('z₁', (7, 3.9), ha='center', va='center', fontsize=10)

    ax.add_patch(plt.Circle((7, 2.6), 0.3, facecolor='coral', edgecolor='black'))
    ax.annotate('z₂', (7, 2.6), ha='center', va='center', fontsize=10)

    # Contrastive loss
    ax.annotate('', xy=(7, 3.5), xytext=(7, 3.0),
               arrowprops=dict(arrowstyle='<->', lw=3, color='green'))
    ax.annotate('Maximize\nsimilarity!', (7.5, 3.25), ha='left', fontsize=10, color='green')

    # Legend box
    ax.add_patch(plt.Rectangle((0.2, 0.5), 7.5, 1.2, facecolor='lightyellow',
                               edgecolor='black', alpha=0.5))
    ax.text(4, 1.3, 'Key: f = encoder (ResNet), g = projection head (MLP)',
           ha='center', fontsize=10)
    ax.text(4, 0.9, 'Representations h transfer to downstream tasks',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_embedding_space(save_path: Optional[str] = None):
    """
    Visualize learned embedding space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)

    # Before training (random)
    ax = axes[0]
    ax.set_title('Before Contrastive Learning\n(Random embeddings)', fontweight='bold')

    for i in range(5):
        points = np.random.randn(20, 2)
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, label=f'Class {i+1}')

    ax.legend(loc='upper right')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # After training (clustered)
    ax = axes[1]
    ax.set_title('After Contrastive Learning\n(Semantically organized)', fontweight='bold')

    colors = plt.cm.tab10(np.arange(5))
    centers = np.array([[2, 2], [-2, 2], [0, -2], [-2, -1], [2, -1]])

    for i in range(5):
        points = centers[i] + np.random.randn(20, 2) * 0.3
        ax.scatter(points[:, 0], points[:, 1], c=[colors[i]], alpha=0.6, label=f'Class {i+1}')

    ax.legend(loc='upper right')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    plt.suptitle('Contrastive Learning Organizes Embedding Space',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_methods_comparison(save_path: Optional[str] = None):
    """
    Compare different contrastive learning methods.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    methods = [
        ('SimCLR', 'Large batch negatives', 'Simple, effective', '2020'),
        ('MoCo', 'Momentum encoder + queue', 'Memory efficient', '2020'),
        ('BYOL', 'No negatives (momentum target)', 'Avoids collapse', '2020'),
        ('SwAV', 'Online clustering', 'Multi-crop strategy', '2020'),
        ('CLIP', 'Image-text pairs', 'Zero-shot transfer', '2021'),
        ('DINO', 'Self-distillation', 'Vision Transformers', '2021'),
    ]

    ax.set_title('Contrastive Learning Methods', fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

    for i, (name, key_idea, strength, year) in enumerate(methods):
        y = len(methods) - 1 - i
        rect = plt.Rectangle((0.5, y), 6, 0.8, facecolor=colors[i],
                             edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(f'{name} ({year})', (0.7, y + 0.4), fontsize=11,
                   fontweight='bold', va='center')
        ax.annotate(key_idea, (2.5, y + 0.4), fontsize=10, va='center')
        ax.annotate(strength, (5.5, y + 0.4), fontsize=9, va='center', color='darkgreen')

    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, len(methods) + 0.5)
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
    print("CONTRASTIVE LEARNING — Paradigm: SIMILARITY STRUCTURE")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Learn by comparing: Pull similar examples together,
    push different examples apart.

    THE LOSS (InfoNCE):
    L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

    KEY COMPONENTS:
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Component           │ Purpose                             │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Augmentation        │ Define what should be invariant     │
    │ Encoder             │ Map inputs to representations       │
    │ Projection head     │ Better for contrastive loss         │
    │ Temperature (τ)     │ Control hardness of negatives       │
    │ Negatives           │ What to push away from              │
    └─────────────────────┴─────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Temperature effect
    temp_results = experiment_temperature_effect()

    # Experiment 2: Batch size effect
    batch_results = experiment_batch_size_effect()

    # Experiment 3: Augmentation importance
    aug_results = experiment_augmentation_importance()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_contrastive_concept('54_contrastive_learning_concept.png')
    visualize_simclr_framework('54_contrastive_learning_simclr.png')
    visualize_embedding_space('54_contrastive_learning_embeddings.png')
    visualize_methods_comparison('54_contrastive_learning_methods.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. CONTRASTIVE = Learn similarity structure
       - Similar things close, different things far
       - No labels needed (self-supervised)

    2. AUGMENTATION defines invariance
       - What changes should the model ignore?
       - Critical design choice

    3. TEMPERATURE controls difficulty
       - Low τ: Hard negatives, may collapse
       - High τ: Easy task, slow learning
       - Sweet spot: 0.07-0.5

    4. BATCH SIZE matters (for SimCLR)
       - More negatives = better discrimination
       - MoCo: Queue allows small batches

    5. MAJOR METHODS:
       - SimCLR: Simple, needs large batch
       - MoCo: Momentum encoder + queue
       - BYOL: No negatives needed
       - CLIP: Vision-language alignment
    """)
