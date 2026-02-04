"""
===============================================================
SELF-SUPERVISED LEARNING — Paradigm: LEARNING FROM STRUCTURE
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Learn representations WITHOUT manual labels.

Create "free" supervision from the DATA ITSELF:
    - Hide part of input, predict the rest
    - Predict relationships between parts
    - Create different views, learn invariance

"The data contains its own supervision signal."

===============================================================
WHY IT MATTERS
===============================================================

LABELING IS EXPENSIVE:
- ImageNet: Years of human labeling effort
- Medical imaging: Expert annotation required
- Most data is UNLABELED

SELF-SUPERVISION IS FREE:
- Unlimited training signal from data structure
- Can leverage massive unlabeled datasets
- Often learns better representations than supervised!

===============================================================
KEY PARADIGMS
===============================================================

1. AUTOENCODING: Reconstruct input from compressed form
   - Autoencoders, VAE, Masked Autoencoders

2. GENERATIVE: Predict next element
   - GPT (next token), Video prediction

3. CONTRASTIVE: Learn similarity structure
   - SimCLR, MoCo, CLIP (separate file)

4. MASKED PREDICTION: Hide and predict
   - BERT (masked tokens), MAE (masked patches)

5. PRETEXT TASKS: Solve "fake" task
   - Rotation prediction, Jigsaw puzzles

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes pretext task requires "understanding" input
2. Assumes learned features transfer to downstream tasks
3. Task design encodes what invariances to learn
4. May learn shortcuts instead of semantics

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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)


# ============================================================
# PRETEXT TASK 1: MASKED PREDICTION (BERT-style)
# ============================================================

class MaskedLanguageModel:
    """
    Masked Language Modeling (BERT-style).

    PRETEXT TASK:
    1. Randomly mask some tokens (replace with [MASK])
    2. Predict the original tokens from context

    WHY IT WORKS:
    To predict masked word, model must understand:
    - Syntax: What word types fit here?
    - Semantics: What makes sense in context?
    - Co-occurrence: What words go together?

    This forces learning of CONTEXTUAL representations.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, mask_prob: float = 0.15):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            mask_prob: Probability of masking each token
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.mask_token_id = vocab_size  # Special [MASK] token

        # Simple encoder (would be Transformer in practice)
        self.embedding = np.random.randn(vocab_size + 1, hidden_dim) * 0.1
        self.encoder_W = he_init(hidden_dim, hidden_dim)
        self.encoder_b = np.zeros((1, hidden_dim))

        # Prediction head
        self.pred_W = he_init(hidden_dim, vocab_size)
        self.pred_b = np.zeros((1, vocab_size))

    def mask_tokens(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply masking to input tokens.

        Returns:
            masked_tokens: Input with some tokens replaced by [MASK]
            mask: Boolean array indicating which positions are masked
            labels: Original token IDs at masked positions (-1 elsewhere)
        """
        mask = np.random.random(tokens.shape) < self.mask_prob
        masked_tokens = tokens.copy()
        masked_tokens[mask] = self.mask_token_id

        labels = np.full_like(tokens, -1)
        labels[mask] = tokens[mask]

        return masked_tokens, mask, labels

    def forward(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            tokens: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) prediction logits
            hidden: (batch, seq_len, hidden_dim) hidden states
        """
        # Embed
        embedded = self.embedding[tokens]  # (batch, seq_len, hidden_dim)

        # Encode (simplified: just one layer)
        hidden = relu(embedded @ self.encoder_W + self.encoder_b)

        # Predict
        logits = hidden @ self.pred_W + self.pred_b

        return logits, hidden

    def compute_loss(self, tokens: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute masked language modeling loss.

        Returns:
            loss: Cross-entropy loss on masked positions only
            hidden: Learned representations
        """
        masked_tokens, mask, labels = self.mask_tokens(tokens)
        logits, hidden = self.forward(masked_tokens)

        # Only compute loss on masked positions
        probs = softmax(logits, axis=-1)

        loss = 0
        count = 0
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                if mask[i, j]:
                    true_token = labels[i, j]
                    loss -= np.log(probs[i, j, true_token] + 1e-10)
                    count += 1

        return loss / max(count, 1), hidden


# ============================================================
# PRETEXT TASK 2: NEXT TOKEN PREDICTION (GPT-style)
# ============================================================

class AutoregressiveModel:
    """
    Autoregressive Language Model (GPT-style).

    PRETEXT TASK:
    Given previous tokens, predict the next token.

    WHY IT WORKS:
    To predict what comes next, model must:
    - Understand context so far
    - Learn patterns and structure
    - Capture long-range dependencies

    Different from BERT: Unidirectional (left-to-right only)
    """

    def __init__(self, vocab_size: int, hidden_dim: int, seq_len: int):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Embeddings
        self.embedding = np.random.randn(vocab_size, hidden_dim) * 0.1

        # Causal self-attention (simplified)
        self.W_qkv = he_init(hidden_dim, hidden_dim * 3)

        # Output projection
        self.out_W = he_init(hidden_dim, vocab_size)
        self.out_b = np.zeros((1, vocab_size))

    def causal_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Causal (masked) self-attention.

        Each position can only attend to previous positions.
        """
        batch, seq_len, d = x.shape

        # Project to Q, K, V
        qkv = x @ self.W_qkv
        Q, K, V = np.split(qkv, 3, axis=-1)

        # Attention scores
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)

        # Causal mask: prevent attending to future
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores[:, mask] = -1e9

        # Softmax and weighted sum
        weights = softmax(scores, axis=-1)
        return weights @ V

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            tokens: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) next token predictions
        """
        embedded = self.embedding[tokens]
        hidden = self.causal_attention(embedded)
        logits = hidden @ self.out_W + self.out_b
        return logits

    def compute_loss(self, tokens: np.ndarray) -> float:
        """
        Compute autoregressive loss.

        Predict token at position t from tokens 0..t-1.
        """
        logits = self.forward(tokens[:, :-1])  # Predict from all but last
        targets = tokens[:, 1:]  # Target is shifted by 1

        probs = softmax(logits, axis=-1)

        # Cross-entropy loss
        batch_size, seq_len = targets.shape
        loss = 0
        for i in range(batch_size):
            for j in range(seq_len):
                loss -= np.log(probs[i, j, targets[i, j]] + 1e-10)

        return loss / (batch_size * seq_len)


# ============================================================
# PRETEXT TASK 3: ROTATION PREDICTION (Vision)
# ============================================================

class RotationPrediction:
    """
    Rotation Prediction for Images.

    PRETEXT TASK:
    1. Rotate image by 0°, 90°, 180°, or 270°
    2. Predict which rotation was applied

    WHY IT WORKS:
    To predict rotation, model must:
    - Recognize object orientation
    - Understand object structure
    - Learn semantic features (not just texture)

    Simple but surprisingly effective!
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Flattened image dimension
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rotations = 4

        # Encoder
        self.enc_W1 = he_init(input_dim, hidden_dim)
        self.enc_b1 = np.zeros((1, hidden_dim))
        self.enc_W2 = he_init(hidden_dim, hidden_dim)
        self.enc_b2 = np.zeros((1, hidden_dim))

        # Rotation classifier
        self.cls_W = he_init(hidden_dim, self.num_rotations)
        self.cls_b = np.zeros((1, self.num_rotations))

    def rotate_image(self, image: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate image by k*90 degrees.

        Args:
            image: (height, width) or (height, width, channels)
            k: Number of 90-degree rotations (0-3)
        """
        return np.rot90(image, k)

    def create_rotation_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training batch with random rotations.

        Args:
            images: (batch, height, width) images

        Returns:
            rotated_images: Flattened rotated images
            labels: Rotation labels (0-3)
        """
        batch_size = len(images)
        labels = np.random.randint(0, 4, batch_size)

        rotated = []
        for img, k in zip(images, labels):
            rot_img = self.rotate_image(img, k)
            rotated.append(rot_img.flatten())

        return np.array(rotated), labels

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Extract features from image."""
        h = relu(x @ self.enc_W1 + self.enc_b1)
        h = relu(h @ self.enc_W2 + self.enc_b2)
        return h

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Returns:
            probs: Rotation prediction probabilities
            features: Learned features
        """
        features = self.encode(x)
        logits = features @ self.cls_W + self.cls_b
        probs = softmax(logits)
        return probs, features


# ============================================================
# PRETEXT TASK 4: JIGSAW PUZZLE
# ============================================================

class JigsawPuzzle:
    """
    Jigsaw Puzzle Solving.

    PRETEXT TASK:
    1. Split image into patches (e.g., 3x3 grid)
    2. Shuffle patches
    3. Predict the correct arrangement

    WHY IT WORKS:
    To solve jigsaw, model must:
    - Understand spatial relationships
    - Recognize object parts
    - Learn compositional structure

    More challenging than rotation → potentially better features.
    """

    def __init__(self, patch_size: int, grid_size: int, hidden_dim: int):
        """
        Args:
            patch_size: Size of each patch
            grid_size: Grid dimension (e.g., 3 for 3x3)
            hidden_dim: Hidden dimension
        """
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.hidden_dim = hidden_dim

        # Number of permutations to consider (subset of all n!)
        self.num_permutations = 100  # Use subset for tractability
        self.permutations = self._generate_permutations()

        # Patch encoder
        patch_dim = patch_size * patch_size
        self.patch_enc = he_init(patch_dim, hidden_dim)

        # Permutation classifier
        self.perm_W = he_init(hidden_dim * self.num_patches, self.num_permutations)
        self.perm_b = np.zeros((1, self.num_permutations))

    def _generate_permutations(self) -> np.ndarray:
        """Generate a fixed set of permutations."""
        perms = [np.arange(self.num_patches)]  # Identity first
        for _ in range(self.num_permutations - 1):
            perm = np.random.permutation(self.num_patches)
            perms.append(perm)
        return np.array(perms)

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract patches from image."""
        h, w = image.shape[:2]
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size

        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                patches.append(patch.flatten())

        return np.array(patches)

    def shuffle_patches(self, patches: np.ndarray, perm_idx: int) -> np.ndarray:
        """Shuffle patches according to permutation."""
        perm = self.permutations[perm_idx]
        return patches[perm]

    def forward(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            patches: (batch, num_patches, patch_dim)

        Returns:
            probs: Permutation prediction probabilities
            features: Learned patch features
        """
        batch_size = patches.shape[0]

        # Encode each patch
        patch_features = relu(patches @ self.patch_enc)  # (batch, num_patches, hidden)

        # Concatenate all patch features
        combined = patch_features.reshape(batch_size, -1)

        # Predict permutation
        logits = combined @ self.perm_W + self.perm_b
        probs = softmax(logits)

        return probs, patch_features


# ============================================================
# PRETEXT TASK 5: CONTEXT PREDICTION
# ============================================================

class ContextPrediction:
    """
    Context Prediction (Word2Vec-style).

    PRETEXT TASK:
    Given a word, predict surrounding words (Skip-gram)
    OR given context, predict center word (CBOW)

    WHY IT WORKS:
    "You shall know a word by the company it keeps" - Firth

    Words in similar contexts → similar embeddings
    This captures semantic relationships!
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int = 2):
        """
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            context_size: Number of context words on each side
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        # Word embeddings (input)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.1

        # Context embeddings (output)
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.1

    def get_context_pairs(self, sequence: np.ndarray) -> List[Tuple[int, int]]:
        """Extract (center, context) pairs from sequence."""
        pairs = []
        for i in range(self.context_size, len(sequence) - self.context_size):
            center = sequence[i]
            for j in range(-self.context_size, self.context_size + 1):
                if j != 0:
                    context = sequence[i + j]
                    pairs.append((center, context))
        return pairs

    def forward_skipgram(self, center_word: int) -> np.ndarray:
        """
        Skip-gram: Predict context from center word.

        Returns probabilities over vocabulary for context words.
        """
        # Get center word embedding
        center_emb = self.W_in[center_word]  # (embedding_dim,)

        # Score all possible context words
        scores = self.W_out @ center_emb  # (vocab_size,)

        return softmax(scores)

    def forward_cbow(self, context_words: List[int]) -> np.ndarray:
        """
        CBOW: Predict center from context words.

        Returns probabilities over vocabulary for center word.
        """
        # Average context embeddings
        context_emb = np.mean(self.W_in[context_words], axis=0)

        # Score all possible center words
        scores = self.W_out @ context_emb

        return softmax(scores)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_masked_prediction(vocab_size: int = 100,
                                 seq_len: int = 20,
                                 mask_probs: List[float] = [0.05, 0.15, 0.30, 0.50],
                                 n_samples: int = 500) -> dict:
    """
    Effect of masking probability.

    WHAT TO OBSERVE:
    - Too low: Not enough learning signal
    - Too high: Context becomes too sparse
    - Sweet spot: ~15% (BERT default)
    """
    print("=" * 60)
    print("EXPERIMENT: Masking Probability Effect")
    print("=" * 60)

    # Create synthetic data with patterns
    np.random.seed(42)
    data = np.random.randint(0, vocab_size, (n_samples, seq_len))

    results = {'mask_probs': mask_probs, 'losses': []}

    for mask_prob in mask_probs:
        model = MaskedLanguageModel(vocab_size, hidden_dim=64, mask_prob=mask_prob)

        # Compute average loss
        total_loss = 0
        for i in range(min(100, n_samples)):
            loss, _ = model.compute_loss(data[i:i+1])
            total_loss += loss

        avg_loss = total_loss / min(100, n_samples)
        results['losses'].append(avg_loss)

        print(f"Mask prob {mask_prob:.2f}: Loss = {avg_loss:.4f}")

    return results


def experiment_pretext_task_comparison(image_size: int = 16,
                                       n_samples: int = 200) -> dict:
    """
    Compare different pretext tasks.

    WHAT TO OBSERVE:
    - Different tasks learn different aspects
    - More challenging tasks may learn better features
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Pretext Task Comparison")
    print("=" * 60)

    # Create synthetic images with simple patterns
    images = np.random.randn(n_samples, image_size, image_size)

    # Add some structure
    for i in range(n_samples):
        # Add horizontal or vertical line
        if i % 2 == 0:
            images[i, image_size // 2, :] = 3  # Horizontal
        else:
            images[i, :, image_size // 2] = 3  # Vertical

    results = {}

    # 1. Rotation prediction
    rotation_model = RotationPrediction(image_size * image_size, hidden_dim=64)
    rotated_images, rotation_labels = rotation_model.create_rotation_batch(images[:100])
    probs, features = rotation_model.forward(rotated_images)
    rotation_acc = np.mean(np.argmax(probs, axis=1) == rotation_labels)
    results['rotation'] = {'accuracy': rotation_acc, 'feature_dim': features.shape[-1]}
    print(f"Rotation Prediction: Accuracy = {rotation_acc:.3f}")

    # 2. Jigsaw (simplified)
    jigsaw_model = JigsawPuzzle(patch_size=4, grid_size=4, hidden_dim=64)
    print(f"Jigsaw Puzzle: {jigsaw_model.num_permutations} permutations to classify")
    results['jigsaw'] = {'num_permutations': jigsaw_model.num_permutations}

    # 3. Context prediction (on synthetic sequence)
    vocab_size = 50
    sequences = np.random.randint(0, vocab_size, (n_samples, 20))
    context_model = ContextPrediction(vocab_size, embedding_dim=32)
    pairs = context_model.get_context_pairs(sequences[0])
    print(f"Context Prediction: {len(pairs)} training pairs from one sequence")
    results['context'] = {'pairs_per_seq': len(pairs)}

    return results


def experiment_representation_quality(vocab_size: int = 100,
                                      hidden_dim: int = 32,
                                      n_samples: int = 500) -> dict:
    """
    Test if self-supervised representations are useful.

    WHAT TO OBSERVE:
    - Self-supervised features should cluster similar items
    - Should transfer to downstream classification
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Representation Quality")
    print("=" * 60)

    # Create data with known clusters
    num_clusters = 5
    seq_len = 10

    # Each cluster has characteristic patterns
    data = []
    labels = []
    for c in range(num_clusters):
        for _ in range(n_samples // num_clusters):
            # Sequences in same cluster share similar patterns
            base_pattern = np.random.randint(c * 20, (c + 1) * 20, seq_len)
            noise = np.random.randint(-2, 3, seq_len)
            seq = np.clip(base_pattern + noise, 0, vocab_size - 1)
            data.append(seq)
            labels.append(c)

    data = np.array(data)
    labels = np.array(labels)

    # Train masked LM
    model = MaskedLanguageModel(vocab_size, hidden_dim, mask_prob=0.15)

    # Extract features
    features = []
    for seq in data:
        _, hidden = model.compute_loss(seq.reshape(1, -1))
        # Average over sequence
        feat = hidden.mean(axis=1).flatten()
        features.append(feat)

    features = np.array(features)

    # Check if clusters are separable
    from collections import defaultdict
    cluster_centers = defaultdict(list)
    for feat, label in zip(features, labels):
        cluster_centers[label].append(feat)

    # Compute inter vs intra cluster distances
    intra_dists = []
    inter_dists = []

    for c in range(num_clusters):
        center = np.mean(cluster_centers[c], axis=0)
        for feat in cluster_centers[c]:
            intra_dists.append(np.linalg.norm(feat - center))

        for c2 in range(c + 1, num_clusters):
            center2 = np.mean(cluster_centers[c2], axis=0)
            inter_dists.append(np.linalg.norm(center - center2))

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)

    print(f"Intra-cluster distance: {avg_intra:.4f}")
    print(f"Inter-cluster distance: {avg_inter:.4f}")
    print(f"Ratio (higher = better separation): {avg_inter / (avg_intra + 1e-10):.4f}")

    return {
        'intra_dist': avg_intra,
        'inter_dist': avg_inter,
        'separation_ratio': avg_inter / (avg_intra + 1e-10)
    }


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_self_supervised_concept(save_path: Optional[str] = None):
    """
    Visual explanation of self-supervised learning.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. The core idea
    ax = axes[0]
    ax.set_title('Self-Supervised Learning: Core Idea', fontweight='bold')

    ax.text(0.5, 0.9, 'Create "free" labels from data itself',
           ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    tasks = [
        ('Masked LM', 'The cat [MASK] on mat → sat'),
        ('Next Token', 'The cat sat → on'),
        ('Rotation', '🖼️ rotated → predict angle'),
        ('Jigsaw', '🧩 shuffled → predict order'),
        ('Contrastive', 'Same image views → similar'),
    ]

    for i, (name, desc) in enumerate(tasks):
        y = 0.7 - i * 0.14
        ax.text(0.1, y, f'• {name}:', fontsize=10, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.4, y, desc, fontsize=9, transform=ax.transAxes)

    ax.axis('off')

    # 2. Masked Language Model
    ax = axes[1]
    ax.set_title('Masked Language Model (BERT)', fontweight='bold')

    sentence = ['The', 'cat', '[MASK]', 'on', 'the', 'mat']
    colors = ['lightblue'] * 6
    colors[2] = 'gold'  # Masked token

    for i, (word, color) in enumerate(zip(sentence, colors)):
        rect = plt.Rectangle((i * 1.2, 2), 1, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.annotate(word, (i * 1.2 + 0.5, 2.4), ha='center', va='center', fontsize=10)

    # Prediction arrow
    ax.annotate('', xy=(2.6, 1.5), xytext=(2.6, 1.9),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('Predict: "sat"', (2.6, 1.2), ha='center', fontsize=11,
               color='green', fontweight='bold')

    ax.text(3.5, 0.5, 'Model learns context\nto predict masked words',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # 3. Why it works
    ax = axes[2]
    ax.set_title('Why Self-Supervision Works', fontweight='bold')

    reasons = [
        ('Unlimited data', 'No manual labeling needed'),
        ('Rich signal', 'Every token/pixel is a label'),
        ('Deep understanding', 'Must learn semantics to solve task'),
        ('Transferable', 'Features useful for downstream'),
    ]

    for i, (title, desc) in enumerate(reasons):
        y = 0.85 - i * 0.2
        ax.text(0.1, y, f'✓ {title}', fontsize=11, fontweight='bold',
               transform=ax.transAxes, color='green')
        ax.text(0.15, y - 0.06, desc, fontsize=9, transform=ax.transAxes)

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_pretext_tasks(save_path: Optional[str] = None):
    """
    Visualize different pretext tasks for vision.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create a simple image
    img = np.zeros((8, 8))
    img[2:6, 3:5] = 1  # Vertical bar
    img[3:5, 2:6] = 1  # Horizontal bar (cross)

    # 1. Original image
    ax = axes[0, 0]
    ax.imshow(img, cmap='Blues')
    ax.set_title('Original Image', fontweight='bold')
    ax.axis('off')

    # 2. Rotation prediction
    ax = axes[0, 1]
    ax.set_title('Rotation Prediction', fontweight='bold')

    rotations = [0, 90, 180, 270]
    for i, rot in enumerate(rotations):
        sub_ax = ax.inset_axes([i * 0.25, 0.2, 0.22, 0.6])
        rotated = np.rot90(img, rot // 90)
        sub_ax.imshow(rotated, cmap='Blues')
        sub_ax.set_title(f'{rot}°', fontsize=9)
        sub_ax.axis('off')

    ax.text(0.5, 0.05, 'Task: Predict rotation angle',
           ha='center', transform=ax.transAxes, fontsize=10)
    ax.axis('off')

    # 3. Jigsaw puzzle
    ax = axes[1, 0]
    ax.set_title('Jigsaw Puzzle', fontweight='bold')

    # Split into 4 patches and shuffle
    patches = [
        img[:4, :4], img[:4, 4:],
        img[4:, :4], img[4:, 4:]
    ]
    shuffled = [patches[2], patches[0], patches[3], patches[1]]  # Random shuffle

    for i, patch in enumerate(shuffled):
        row, col = i // 2, i % 2
        sub_ax = ax.inset_axes([col * 0.4 + 0.1, (1 - row) * 0.4 + 0.1, 0.35, 0.35])
        sub_ax.imshow(patch, cmap='Blues')
        sub_ax.set_title(f'Patch {i+1}', fontsize=8)
        sub_ax.axis('off')

    ax.text(0.5, 0.02, 'Task: Predict correct arrangement',
           ha='center', transform=ax.transAxes, fontsize=10)
    ax.axis('off')

    # 4. Masked autoencoder
    ax = axes[1, 1]
    ax.set_title('Masked Autoencoder', fontweight='bold')

    # Create masked version
    masked = img.copy()
    mask = np.random.random((8, 8)) < 0.5
    masked[mask] = 0.5  # Gray out masked regions

    sub_ax1 = ax.inset_axes([0.1, 0.3, 0.35, 0.5])
    sub_ax1.imshow(masked, cmap='Blues', vmin=0, vmax=1)
    sub_ax1.set_title('Input (masked)', fontsize=9)
    sub_ax1.axis('off')

    ax.annotate('', xy=(0.58, 0.55), xytext=(0.48, 0.55),
               arrowprops=dict(arrowstyle='->', lw=2), xycoords='axes fraction')

    sub_ax2 = ax.inset_axes([0.55, 0.3, 0.35, 0.5])
    sub_ax2.imshow(img, cmap='Blues')
    sub_ax2.set_title('Reconstruct', fontsize=9)
    sub_ax2.axis('off')

    ax.text(0.5, 0.1, 'Task: Reconstruct masked regions',
           ha='center', transform=ax.transAxes, fontsize=10)
    ax.axis('off')

    plt.suptitle('Self-Supervised Pretext Tasks for Vision',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_bert_vs_gpt(save_path: Optional[str] = None):
    """
    Compare BERT (bidirectional) vs GPT (autoregressive).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT
    ax = axes[0]
    ax.set_title('BERT: Masked Language Model\n(Bidirectional)', fontweight='bold')

    words = ['The', 'cat', '[M]', 'on', 'mat']
    for i, word in enumerate(words):
        color = 'gold' if word == '[M]' else 'lightblue'
        rect = plt.Rectangle((i * 1.4, 1.5), 1.2, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.annotate(word, (i * 1.4 + 0.6, 1.9), ha='center', va='center', fontsize=11)

        # Bidirectional arrows for non-masked
        if word != '[M]':
            ax.annotate('', xy=(2.2, 1.5), xytext=(i * 1.4 + 0.6, 1.5),
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5,
                                      connectionstyle='arc3,rad=0.3'))

    ax.annotate('Uses BOTH\nleft & right context', (2.5, 0.8), ha='center',
               fontsize=10, color='blue')
    ax.annotate('↑ Predict "sat"', (2.2, 1.2), ha='center', fontsize=10, color='green')

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # GPT
    ax = axes[1]
    ax.set_title('GPT: Autoregressive\n(Left-to-right)', fontweight='bold')

    words = ['The', 'cat', 'sat', 'on', '?']
    for i, word in enumerate(words):
        color = 'lightgreen' if word == '?' else 'lightblue'
        rect = plt.Rectangle((i * 1.4, 1.5), 1.2, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.annotate(word, (i * 1.4 + 0.6, 1.9), ha='center', va='center', fontsize=11)

        # Left-to-right arrows only
        if i < 4:
            ax.annotate('', xy=((i+1) * 1.4, 1.9), xytext=(i * 1.4 + 1.2, 1.9),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

    ax.annotate('Uses ONLY\nleft context', (2.5, 0.8), ha='center',
               fontsize=10, color='blue')
    ax.annotate('↑ Predict "the"', (5.6, 1.2), ha='center', fontsize=10, color='green')

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 3)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_timeline(save_path: Optional[str] = None):
    """
    Timeline of self-supervised learning milestones.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    milestones = [
        (2013, 'Word2Vec', 'Word embeddings'),
        (2017, 'Transformer', 'Attention is all you need'),
        (2018, 'BERT', 'Bidirectional masked LM'),
        (2018, 'GPT', 'Autoregressive LM'),
        (2020, 'SimCLR', 'Contrastive vision'),
        (2020, 'GPT-3', 'Few-shot learning'),
        (2021, 'CLIP', 'Vision-language'),
        (2022, 'MAE', 'Masked autoencoders'),
        (2023, 'GPT-4', 'Multimodal LLM'),
    ]

    ax.set_title('Self-Supervised Learning: Key Milestones', fontweight='bold', fontsize=14)

    # Draw timeline
    years = [m[0] for m in milestones]
    ax.plot([min(years) - 0.5, max(years) + 0.5], [0, 0], 'k-', lw=2)

    for i, (year, name, desc) in enumerate(milestones):
        # Alternate above/below
        y_offset = 0.5 if i % 2 == 0 else -0.5
        y_text = 1 if i % 2 == 0 else -1

        ax.plot(year, 0, 'ko', markersize=10)
        ax.plot([year, year], [0, y_offset], 'k-', lw=1)

        ax.annotate(f'{name}\n({year})', (year, y_text),
                   ha='center', va='center' if y_offset > 0 else 'center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.annotate(desc, (year, y_text + (0.4 if y_offset > 0 else -0.4)),
                   ha='center', fontsize=8, color='gray')

    ax.set_xlim(2012, 2024)
    ax.set_ylim(-2, 2)
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
    print("SELF-SUPERVISED LEARNING — Paradigm: LEARNING FROM STRUCTURE")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Create supervision from the data itself. No labels needed!

    THE KEY QUESTION:
    What "pretext task" forces the model to learn useful representations?

    MAJOR APPROACHES:
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Approach            │ Examples                            │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Masked Prediction   │ BERT, MAE (masks then predicts)     │
    │ Autoregressive      │ GPT (predict next token)            │
    │ Contrastive         │ SimCLR, CLIP (similarity learning)  │
    │ Reconstruction      │ Autoencoders, VAE                   │
    │ Pretext Tasks       │ Rotation, Jigsaw, Colorization      │
    └─────────────────────┴─────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Masking probability
    mask_results = experiment_masked_prediction()

    # Experiment 2: Task comparison
    task_results = experiment_pretext_task_comparison()

    # Experiment 3: Representation quality
    repr_results = experiment_representation_quality()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_self_supervised_concept('51_self_supervised_concept.png')
    visualize_pretext_tasks('51_self_supervised_pretext.png')
    visualize_bert_vs_gpt('51_self_supervised_bert_gpt.png')
    visualize_timeline('51_self_supervised_timeline.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. SELF-SUPERVISION = Free labels from data structure
       - No manual annotation needed
       - Unlimited training signal

    2. PRETEXT TASK DESIGN is crucial
       - Task must require understanding the data
       - Too easy → trivial features
       - Too hard → no learning

    3. MAJOR PARADIGMS:
       - Language: Masked (BERT) vs Autoregressive (GPT)
       - Vision: Contrastive, Masked, Pretext tasks
       - Both: Learning transferable representations

    4. REPRESENTATIONS TRANSFER well
       - Pre-train on pretext task
       - Fine-tune on downstream task
       - Often better than supervised pre-training!

    5. THE MODERN FOUNDATION:
       - GPT: Autoregressive pre-training → LLMs
       - BERT: Bidirectional understanding
       - CLIP: Vision-language alignment
    """)
