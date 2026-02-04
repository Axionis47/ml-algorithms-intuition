"""
===============================================================
POSITIONAL ENCODING — Paradigm: SEQUENCE AWARENESS
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Transformers and attention mechanisms are PERMUTATION EQUIVARIANT.
They see: {token_a, token_b, token_c} — a SET, not a SEQUENCE.

But order matters! "Dog bites man" ≠ "Man bites dog"

Positional encoding INJECTS position information:
    input_with_position = embedding + position_encoding

"Tell the model WHERE each token is in the sequence."

===============================================================
THE PROBLEM
===============================================================

Attention computes: output_i = Σ_j attention(q_i, k_j) × v_j

This is SYMMETRIC in j — there's no notion of:
- j comes BEFORE i
- j is FAR from i
- j is at the BEGINNING

Without positional info, the model cannot learn position-dependent patterns.

===============================================================
KEY VARIANTS
===============================================================

1. SINUSOIDAL (Fixed) — Original Transformer
2. LEARNED — Trainable embedding per position
3. RELATIVE — Encode distance between positions, not absolute
4. ROTARY (RoPE) — Rotation-based, combines absolute and relative
5. ALiBi — Attention bias based on distance

===============================================================
INDUCTIVE BIAS
===============================================================

1. Sinusoidal: Assumes periodicity is useful; extrapolates to longer sequences
2. Learned: No assumptions; limited to training sequence length
3. Relative: Position relationships matter more than absolute position
4. RoPE: Inner product encodes relative position naturally

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


# ============================================================
# SINUSOIDAL POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding (Original Transformer).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    WHY SINUSOIDS?

    1. RELATIVE POSITIONS as linear functions:
       PE(pos+k) can be represented as a linear function of PE(pos)
       This allows the model to easily learn relative positions.

    2. EXTRAPOLATION:
       Works for any sequence length (unlike learned embeddings)
       Sinusoids continue smoothly beyond training lengths.

    3. UNIQUE ENCODING:
       Each position gets a unique encoding.
       Different frequencies capture different scales of position.

    THE INTUITION:
    Think of it like a binary clock:
    - Fastest changing bit: position 0, 1, 0, 1, 0, 1, ...
    - Slowest changing bit: changes every 2^n positions
    Sinusoids are a smooth, continuous version of this.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of model (embedding size)
            max_len: Maximum sequence length to pre-compute
        """
        self.d_model = d_model
        self.max_len = max_len

        # Pre-compute positional encodings
        self.encoding = self._create_encoding(max_len, d_model)

    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding matrix."""
        # Position indices: [0, 1, 2, ..., max_len-1]
        position = np.arange(max_len)[:, np.newaxis]

        # Dimension indices: [0, 2, 4, ..., d_model-2]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        encoding = np.zeros((max_len, d_model))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional encoding for sequence.

        Args:
            seq_len: Length of sequence

        Returns:
            encoding: (seq_len, d_model)
        """
        return self.encoding[:seq_len]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model) input embeddings

        Returns:
            x + PE: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


# ============================================================
# LEARNED POSITIONAL ENCODING
# ============================================================

class LearnedPositionalEncoding:
    """
    Learned Positional Encoding.

    Each position gets a trainable embedding vector.

    PROS:
    - Flexible: Can learn any position-dependent pattern
    - Task-specific: Adapts to the data

    CONS:
    - Limited to max_len seen during training
    - More parameters
    - May overfit position patterns
    """

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Args:
            d_model: Dimension of model
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len

        # Learnable position embeddings (initialized randomly)
        self.encoding = np.random.randn(max_len, d_model) * 0.02

    def forward(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for sequence."""
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.encoding[:seq_len]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


# ============================================================
# RELATIVE POSITIONAL ENCODING
# ============================================================

class RelativePositionalEncoding:
    """
    Relative Positional Encoding.

    Instead of encoding ABSOLUTE position (0, 1, 2, ...),
    encode RELATIVE distance between positions (i - j).

    WHY RELATIVE?
    - "The word 2 positions ago" is often more meaningful than
      "The word at position 47"
    - Generalizes better to different sequence lengths
    - Captures the LOCAL structure that often matters most

    IMPLEMENTATION:
    For attention scores, add a bias based on relative position:
        score(i, j) = q_i · k_j + bias[i - j]
    """

    def __init__(self, d_model: int, max_relative_position: int = 128):
        """
        Args:
            d_model: Dimension of model
            max_relative_position: Maximum relative distance to encode
        """
        self.d_model = d_model
        self.max_relative_position = max_relative_position

        # Relative position embeddings: -max to +max
        self.num_positions = 2 * max_relative_position + 1
        self.encoding = np.random.randn(self.num_positions, d_model) * 0.02

    def get_relative_positions(self, seq_len: int) -> np.ndarray:
        """
        Compute relative position matrix.

        Returns:
            relative_positions: (seq_len, seq_len) where [i,j] = i - j
        """
        positions = np.arange(seq_len)
        relative = positions[:, np.newaxis] - positions[np.newaxis, :]
        return relative

    def get_relative_embeddings(self, seq_len: int) -> np.ndarray:
        """
        Get relative position embeddings.

        Returns:
            embeddings: (seq_len, seq_len, d_model)
        """
        relative = self.get_relative_positions(seq_len)

        # Clip to max relative position
        relative_clipped = np.clip(relative,
                                   -self.max_relative_position,
                                   self.max_relative_position)

        # Shift to positive indices
        relative_indices = relative_clipped + self.max_relative_position

        return self.encoding[relative_indices]


# ============================================================
# ROTARY POSITIONAL ENCODING (RoPE)
# ============================================================

class RotaryPositionalEncoding:
    """
    Rotary Position Embedding (RoPE).

    Used in: LLaMA, GPT-NeoX, PaLM, and many modern LLMs.

    THE KEY INSIGHT:
    Instead of ADDING position info, ROTATE the embeddings.

    For query q and key k at positions m and n:
        q_m = R(m) @ q
        k_n = R(n) @ k

    Then the dot product becomes:
        q_m · k_n = (R(m) @ q) · (R(n) @ k)
                  = q · R(n-m) @ k

    The relative position (n-m) is naturally encoded!

    WHY ROTATION?
    - Preserves vector magnitudes
    - Relative position emerges from dot product
    - Works for both absolute and relative patterns
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        """
        Args:
            d_model: Dimension of model (must be even)
            base: Base for computing rotation frequencies
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.base = base

        # Compute inverse frequencies
        self.inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))

    def _compute_rotation_matrix(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sin and cos for rotation.

        Args:
            positions: (seq_len,) position indices

        Returns:
            cos, sin: (seq_len, d_model/2) each
        """
        # Outer product: (seq_len,) × (d_model/2,) → (seq_len, d_model/2)
        freqs = np.outer(positions, self.inv_freq)
        return np.cos(freqs), np.sin(freqs)

    def forward(self, x: np.ndarray, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply rotary positional encoding.

        Args:
            x: (batch, seq_len, d_model) input
            positions: Optional custom positions

        Returns:
            rotated: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d = x.shape

        if positions is None:
            positions = np.arange(seq_len)

        cos, sin = self._compute_rotation_matrix(positions)

        # Split x into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation
        # [cos, -sin] [x1]   [x1*cos - x2*sin]
        # [sin,  cos] [x2] = [x1*sin + x2*cos]
        rotated = np.zeros_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos

        return rotated


# ============================================================
# ALiBi (Attention with Linear Biases)
# ============================================================

class ALiBi:
    """
    Attention with Linear Biases.

    Used in: BLOOM, MPT, and other efficient LLMs.

    THE IDEA:
    Don't modify embeddings at all!
    Instead, add a linear BIAS to attention scores:

        attention(i, j) = softmax(q_i · k_j - m × |i - j|)

    Where m is a head-specific slope.

    WHY LINEAR BIAS?
    - Simpler than sinusoidal or learned encodings
    - Naturally penalizes distant positions
    - Excellent length extrapolation
    - No extra parameters to learn
    """

    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads

        # Compute slopes for each head (geometric sequence)
        # Slopes decrease geometrically: 2^(-8/n), 2^(-16/n), ...
        ratio = 2 ** (-8 / num_heads)
        self.slopes = ratio ** np.arange(1, num_heads + 1)

    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Compute ALiBi attention bias.

        Args:
            seq_len: Sequence length

        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        # Relative positions
        positions = np.arange(seq_len)
        relative_positions = positions[:, np.newaxis] - positions[np.newaxis, :]

        # Bias = -slope × |relative_position|
        # Use negative to penalize distant positions
        bias = -self.slopes[:, np.newaxis, np.newaxis] * np.abs(relative_positions)

        return bias

    def apply_to_attention(self, attention_scores: np.ndarray) -> np.ndarray:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: (batch, heads, seq_len, seq_len)

        Returns:
            biased_scores: same shape
        """
        seq_len = attention_scores.shape[-1]
        bias = self.get_bias(seq_len)
        return attention_scores + bias


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_extrapolation(max_train_len: int = 100,
                             test_lens: List[int] = [100, 200, 500, 1000],
                             d_model: int = 64) -> dict:
    """
    Test how well different positional encodings extrapolate.

    WHAT TO OBSERVE:
    - Learned: Fails beyond training length
    - Sinusoidal: Continues smoothly
    - ALiBi: Linear extrapolation
    """
    print("=" * 60)
    print("EXPERIMENT: Length Extrapolation")
    print("=" * 60)

    results = {'test_lens': test_lens,
               'sinusoidal': [],
               'learned': [],
               'alibi': []}

    sinusoidal = SinusoidalPositionalEncoding(d_model, max_len=5000)
    learned = LearnedPositionalEncoding(d_model, max_len=max_train_len)
    alibi = ALiBi(num_heads=4)

    for test_len in test_lens:
        # Sinusoidal: Always works
        try:
            sin_enc = sinusoidal.forward(test_len)
            sin_valid = True
            sin_norm = np.mean(np.linalg.norm(sin_enc, axis=1))
        except:
            sin_valid = False
            sin_norm = np.nan

        # Learned: Fails beyond max_len
        try:
            learn_enc = learned.forward(test_len)
            learn_valid = True
            learn_norm = np.mean(np.linalg.norm(learn_enc, axis=1))
        except:
            learn_valid = False
            learn_norm = np.nan

        # ALiBi: Bias grows linearly
        try:
            alibi_bias = alibi.get_bias(test_len)
            alibi_valid = True
            alibi_max_bias = np.max(np.abs(alibi_bias))
        except:
            alibi_valid = False
            alibi_max_bias = np.nan

        results['sinusoidal'].append((sin_valid, sin_norm))
        results['learned'].append((learn_valid, learn_norm))
        results['alibi'].append((alibi_valid, alibi_max_bias))

        status_sin = "✓" if sin_valid else "✗"
        status_learn = "✓" if learn_valid else "✗"
        status_alibi = "✓" if alibi_valid else "✗"

        print(f"Length {test_len:4d}: Sinusoidal {status_sin}  "
              f"Learned {status_learn}  ALiBi {status_alibi}")

    return results


def experiment_relative_position_encoding(seq_len: int = 20,
                                          d_model: int = 32) -> dict:
    """
    Visualize relative vs absolute positional encoding.

    WHAT TO OBSERVE:
    - Absolute: Position 5 always has same encoding
    - Relative: Same encoding for "2 positions apart"
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Absolute vs Relative Position")
    print("=" * 60)

    # Absolute (sinusoidal)
    sin_enc = SinusoidalPositionalEncoding(d_model)
    absolute = sin_enc.forward(seq_len)

    # Relative
    rel_enc = RelativePositionalEncoding(d_model)
    relative_positions = rel_enc.get_relative_positions(seq_len)

    results = {
        'absolute': absolute,
        'relative_positions': relative_positions
    }

    print(f"\nAbsolute encoding shape: {absolute.shape}")
    print(f"Relative position matrix shape: {relative_positions.shape}")
    print(f"\nRelative position example (first 5 positions):")
    print(relative_positions[:5, :5])

    return results


def experiment_rope_properties(d_model: int = 32, seq_len: int = 20) -> dict:
    """
    Demonstrate RoPE's unique property.

    WHAT TO OBSERVE:
    The dot product of rotated vectors depends only on relative position.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: RoPE Relative Position Property")
    print("=" * 60)

    rope = RotaryPositionalEncoding(d_model)

    # Create random vectors
    q = np.random.randn(1, seq_len, d_model)
    k = np.random.randn(1, seq_len, d_model)

    # Apply RoPE
    q_rotated = rope.forward(q)
    k_rotated = rope.forward(k)

    # Compute attention scores
    scores = np.matmul(q_rotated, k_rotated.transpose(0, 2, 1))[0]

    # Check: scores[i,j] should depend on (i-j), not i and j separately
    # Compute scores for same relative distance
    results = {'same_distance_scores': {}}

    for distance in [1, 2, 3]:
        pairs = [(i, i + distance) for i in range(seq_len - distance)]
        pair_scores = [scores[i, j] for i, j in pairs]
        results['same_distance_scores'][distance] = {
            'scores': pair_scores,
            'std': np.std(pair_scores)
        }
        print(f"Distance {distance}: Score std = {np.std(pair_scores):.4f}")
        print(f"  (Lower std means more consistent relative encoding)")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_sinusoidal_encoding(d_model: int = 64,
                                  seq_len: int = 100,
                                  save_path: Optional[str] = None):
    """
    Visualize sinusoidal positional encoding.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pe = SinusoidalPositionalEncoding(d_model)
    encoding = pe.forward(seq_len)

    # 1. Full encoding heatmap
    ax = axes[0, 0]
    im = ax.imshow(encoding.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Sinusoidal Positional Encoding', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 2. Selected dimensions
    ax = axes[0, 1]
    dims_to_plot = [0, 1, 10, 20, 30, 40]
    for dim in dims_to_plot:
        ax.plot(encoding[:, dim], label=f'dim {dim}', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Different Dimensions = Different Frequencies', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Position similarity
    ax = axes[1, 0]
    # Compute similarity between position 0 and all other positions
    similarity = encoding @ encoding[0]
    ax.plot(similarity)
    ax.set_xlabel('Position')
    ax.set_ylabel('Similarity to Position 0')
    ax.set_title('Position Similarity (dot product)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. The "binary clock" intuition
    ax = axes[1, 1]
    ax.text(0.5, 0.9, 'The Binary Clock Intuition', fontsize=12,
           fontweight='bold', ha='center', transform=ax.transAxes)

    ax.text(0.1, 0.7, 'Position in binary:\n'
           '0: 0000\n1: 0001\n2: 0010\n3: 0011\n4: 0100\n...',
           fontsize=10, transform=ax.transAxes, family='monospace')

    ax.text(0.5, 0.7, '→', fontsize=20, transform=ax.transAxes)

    ax.text(0.6, 0.7, 'Sinusoidal encoding:\n'
           'dim 0,1: Fastest oscillation\n'
           'dim 2,3: Slower oscillation\n'
           '...\n'
           'dim d-2,d-1: Slowest oscillation',
           fontsize=10, transform=ax.transAxes)

    ax.text(0.5, 0.15, 'Each dimension captures position at different "resolutions"',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    ax.axis('off')

    plt.suptitle('Sinusoidal Positional Encoding: How It Works',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_alibi(num_heads: int = 4, seq_len: int = 20,
                   save_path: Optional[str] = None):
    """
    Visualize ALiBi attention biases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    alibi = ALiBi(num_heads)
    bias = alibi.get_bias(seq_len)

    # 1. Bias for each head
    ax = axes[0]
    for head in range(num_heads):
        ax.plot(bias[head, seq_len//2, :], label=f'Head {head+1} (slope={alibi.slopes[head]:.4f})')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Bias (added to attention score)')
    ax.set_title(f'ALiBi Bias from Query Position {seq_len//2}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Full bias matrix for one head
    ax = axes[1]
    im = ax.imshow(bias[0], cmap='RdBu_r', vmin=bias.min(), vmax=0)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('ALiBi Bias Matrix (Head 1)\nDarker = More Penalty (distant positions)',
                fontweight='bold')
    plt.colorbar(im, ax=ax)

    plt.suptitle('ALiBi: Linear Attention Bias Based on Distance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_comparison(d_model: int = 32, seq_len: int = 50,
                        save_path: Optional[str] = None):
    """
    Compare different positional encoding methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Methods
    sinusoidal = SinusoidalPositionalEncoding(d_model)
    learned = LearnedPositionalEncoding(d_model, max_len=seq_len)
    relative = RelativePositionalEncoding(d_model)
    alibi = ALiBi(num_heads=4)

    # 1. Sinusoidal
    ax = axes[0, 0]
    enc = sinusoidal.forward(seq_len)
    ax.imshow(enc.T, aspect='auto', cmap='viridis')
    ax.set_title('Sinusoidal (Fixed)\n+ Extrapolates, - Fixed patterns', fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')

    # 2. Learned
    ax = axes[0, 1]
    enc = learned.forward(seq_len)
    ax.imshow(enc.T, aspect='auto', cmap='viridis')
    ax.set_title('Learned\n+ Flexible, - No extrapolation', fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')

    # 3. Relative positions
    ax = axes[1, 0]
    rel_pos = relative.get_relative_positions(seq_len)
    ax.imshow(rel_pos, aspect='auto', cmap='RdBu')
    ax.set_title('Relative Positions\n+ Translation invariant, - More complex', fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

    # 4. ALiBi
    ax = axes[1, 1]
    bias = alibi.get_bias(seq_len)[0]  # First head
    ax.imshow(bias, aspect='auto', cmap='RdBu_r')
    ax.set_title('ALiBi (Attention Bias)\n+ Simple, extrapolates, - Linear only', fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

    plt.suptitle('Positional Encoding Methods Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_rope_concept(save_path: Optional[str] = None):
    """
    Visualize RoPE concept.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Rotation concept
    ax = axes[0]
    ax.set_title('RoPE: Rotation in 2D', fontweight='bold')

    # Draw original vector
    theta = np.pi / 6
    v = np.array([1, 0.3])
    v_norm = v / np.linalg.norm(v) * 0.8

    ax.arrow(0, 0, v_norm[0], v_norm[1], head_width=0.05, head_length=0.03,
            fc='blue', ec='blue', linewidth=2)
    ax.annotate('Original v', (v_norm[0], v_norm[1] + 0.1), color='blue')

    # Draw rotated vectors for different positions
    for pos, angle, color in [(1, 0.3, 'green'), (2, 0.6, 'orange'), (3, 0.9, 'red')]:
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        v_rot = rot @ v_norm
        ax.arrow(0, 0, v_rot[0], v_rot[1], head_width=0.05, head_length=0.03,
                fc=color, ec=color, linewidth=2, alpha=0.7)
        ax.annotate(f'pos={pos}', (v_rot[0] + 0.05, v_rot[1]), color=color, fontsize=9)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # 2. Key insight
    ax = axes[1]
    ax.set_title('The Key Insight', fontweight='bold')

    ax.text(0.5, 0.85, 'For query q at position m\nand key k at position n:',
           ha='center', fontsize=11, transform=ax.transAxes)

    ax.text(0.5, 0.6, 'q_rotated · k_rotated', ha='center', fontsize=12,
           transform=ax.transAxes, fontweight='bold')
    ax.text(0.5, 0.5, '=', ha='center', fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.4, 'R(m)q · R(n)k', ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.3, '=', ha='center', fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.2, 'q · R(n-m)k', ha='center', fontsize=12,
           transform=ax.transAxes, fontweight='bold', color='green')

    ax.text(0.5, 0.05, 'Relative position (n-m)\nemerges naturally!',
           ha='center', fontsize=10, transform=ax.transAxes, style='italic')

    ax.axis('off')

    # 3. Used in models
    ax = axes[2]
    ax.set_title('Models Using RoPE', fontweight='bold')

    models = [
        ('LLaMA / LLaMA 2', 'Meta'),
        ('GPT-NeoX', 'EleutherAI'),
        ('PaLM', 'Google'),
        ('Falcon', 'TII'),
        ('Mistral', 'Mistral AI'),
        ('Qwen', 'Alibaba'),
    ]

    for i, (model, org) in enumerate(models):
        y = 0.85 - i * 0.14
        ax.text(0.1, y, f'• {model}', fontsize=11, transform=ax.transAxes,
               fontweight='bold')
        ax.text(0.6, y, f'({org})', fontsize=9, transform=ax.transAxes, color='gray')

    ax.axis('off')

    plt.suptitle('Rotary Position Embedding (RoPE)',
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
    print("POSITIONAL ENCODING — Paradigm: SEQUENCE AWARENESS")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Attention is permutation-equivariant — it sees SETS, not SEQUENCES.
    Positional encoding injects position information.

    THE PROBLEM:
    "The cat sat on the mat" and "mat the on sat cat The"
    look identical to pure attention!

    KEY METHODS:
    ┌─────────────────┬────────────────────┬─────────────────────────┐
    │ Method          │ How It Works       │ Properties              │
    ├─────────────────┼────────────────────┼─────────────────────────┤
    │ Sinusoidal      │ Sin/cos at freqs   │ Extrapolates, fixed     │
    │ Learned         │ Trainable vectors  │ Flexible, no extrap.    │
    │ Relative        │ Encode i-j         │ Translation invariant   │
    │ RoPE            │ Rotate embeddings  │ Best of both worlds     │
    │ ALiBi           │ Linear attn bias   │ Simple, extrapolates    │
    └─────────────────┴────────────────────┴─────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Extrapolation
    extrap_results = experiment_extrapolation()

    # Experiment 2: Relative vs Absolute
    rel_results = experiment_relative_position_encoding()

    # Experiment 3: RoPE properties
    rope_results = experiment_rope_properties()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_sinusoidal_encoding(save_path='49_positional_sinusoidal.png')
    visualize_alibi(save_path='49_positional_alibi.png')
    visualize_comparison(save_path='49_positional_comparison.png')
    visualize_rope_concept(save_path='49_positional_rope.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. ATTENTION NEEDS POSITION INFORMATION
       - Pure attention is permutation-equivariant
       - Order matters for language, sequences, etc.

    2. SINUSOIDAL: The classic approach
       - Pro: Works for any length, no parameters
       - Con: Fixed patterns, may not be optimal

    3. LEARNED: Data-driven
       - Pro: Adapts to the task
       - Con: Limited to training length

    4. RELATIVE: Focus on relationships
       - "2 positions apart" vs "position 47"
       - Better for translation-invariant patterns

    5. ROPE: Modern standard
       - Rotation-based encoding
       - Relative position from dot product
       - Used in LLaMA, GPT-NeoX, PaLM, etc.

    6. ALiBi: Simple and effective
       - Just add linear bias to attention
       - Excellent extrapolation
       - Used in BLOOM, MPT
    """)
