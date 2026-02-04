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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


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
# TRAINING INFRASTRUCTURE
# ============================================================

def create_first_last_task(n_samples: int = 3000, seq_len: int = 5,
                           vocab_size: int = 5) -> Tuple:
    """
    Create a "First vs Last" task that REQUIRES positional encoding.

    Task: Given a sequence and a query ("first" or "last"),
          output the first or last token.

    WHY THIS REQUIRES POSITIONAL ENCODING:
    Without PE, attention sees the sequence as a SET, not a list.
    It cannot distinguish [A, B, C, D, E] from [E, D, C, B, A].
    With PE, position 0 and position 4 have different representations.

    Expected results:
    - Without PE: ~50% (random guess - can't distinguish first from last)
    - With PE: ~100% (knows which position is first/last)
    """
    np.random.seed(42)

    # Random sequences where first and last tokens are DIFFERENT
    sequences = []
    for _ in range(n_samples):
        seq = np.random.randint(0, vocab_size, seq_len)
        # Ensure first and last are different for clearer signal
        while seq[0] == seq[-1]:
            seq[-1] = np.random.randint(0, vocab_size)
        sequences.append(seq)
    sequences = np.array(sequences)

    # Query: 0 = "first", 1 = "last" (balanced)
    queries = np.random.randint(0, 2, n_samples)

    # Target: first token if query=0, last token if query=1
    targets = np.where(queries == 0, sequences[:, 0], sequences[:, -1])

    # One-hot encode sequences
    X_seq = np.zeros((n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for j in range(seq_len):
            X_seq[i, j, sequences[i, j]] = 1

    # One-hot encode targets
    y = np.zeros((n_samples, vocab_size))
    y[np.arange(n_samples), targets] = 1

    # Query as scalar (0 or 1)
    X_query = queries.astype(np.float32)

    split = int(0.8 * n_samples)
    return (X_seq[:split], X_query[:split], y[:split],
            X_seq[split:], X_query[split:], y[split:])


def create_position_task(n_samples: int = 1000, seq_len: int = 20,
                         vocab_size: int = 10) -> Tuple:
    """
    Create a position-dependent task: "Output the token at position k"

    Task: Given sequence [a, b, c, ...] and query position k,
          output the token at position k.

    This task is IMPOSSIBLE without positional information because
    pure attention treats the sequence as a SET, not a list.
    """
    np.random.seed(42)

    # Random sequences
    sequences = np.random.randint(0, vocab_size, (n_samples, seq_len))

    # Random query positions
    query_positions = np.random.randint(0, seq_len, n_samples)

    # Target: token at the query position
    targets = sequences[np.arange(n_samples), query_positions]

    # One-hot encode sequences
    X_seq = np.zeros((n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for j in range(seq_len):
            X_seq[i, j, sequences[i, j]] = 1

    # One-hot encode targets
    y = np.zeros((n_samples, vocab_size))
    y[np.arange(n_samples), targets] = 1

    # Query position as one-hot
    X_pos = np.zeros((n_samples, seq_len))
    X_pos[np.arange(n_samples), query_positions] = 1

    split = int(0.8 * n_samples)
    return (X_seq[:split], X_pos[:split], y[:split],
            X_seq[split:], X_pos[split:], y[split:])


class FirstLastModel:
    """
    Model for the First/Last task that TRULY demonstrates PE necessity.

    KEY INSIGHT:
    We use the SAME attention mechanism for both "first" and "last" queries.
    The difference is the QUERY EMBEDDING, which interacts with the sequence.

    WITH PE:
    - Position 0 has encoding PE_0, position 4 has encoding PE_4
    - Query for "first" learns to match PE_0
    - Query for "last" learns to match PE_4
    - Model can distinguish positions!

    WITHOUT PE:
    - All positions of same token have IDENTICAL representation
    - Query cannot distinguish position 0 from position 4
    - Model CANNOT learn the task!

    This is the correct demonstration.
    """

    def __init__(self, seq_len: int, vocab_size: int, d_model: int = 16,
                 use_pe: bool = True):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_pe = use_pe

        # Token embedding
        self.W_embed = np.random.randn(vocab_size, d_model) * 0.1

        # Positional encoding (sinusoidal)
        if use_pe:
            pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len * 2)
            self.pe = pe.forward(seq_len)  # (seq_len, d_model)
        else:
            self.pe = None

        # Query embedding: maps query (0 or 1) to d_model dimension
        # This will learn to create a query that matches position 0 or position seq_len-1
        self.W_query = np.random.randn(2, d_model) * 0.1

        # Key projection: same for all positions (attention needs PE to distinguish)
        self.W_k = np.random.randn(d_model, d_model) * 0.1

        # Output projection
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, X_seq: np.ndarray, X_query: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass using attention mechanism.

        The attention score between query and each position depends on:
        - WITHOUT PE: Only token embedding (same token = same representation)
        - WITH PE: Token embedding + position encoding (position matters!)
        """
        batch_size = X_seq.shape[0]

        # Embed tokens: (batch, seq_len, d_model)
        embedded = np.einsum('bsv,vd->bsd', X_seq, self.W_embed)

        # Add positional encoding (THE KEY DIFFERENCE!)
        if self.use_pe and self.pe is not None:
            embedded = embedded + self.pe

        # Compute keys: (batch, seq_len, d_model)
        keys = embedded @ self.W_k

        # Get query vectors based on query type (0=first, 1=last)
        # Query: (batch, d_model)
        query = self.W_query[X_query.astype(int)]  # (batch, d_model)

        # Attention scores: query dot keys
        # (batch, d_model) @ (batch, d_model, seq_len) -> (batch, seq_len)
        scores = np.einsum('bd,bsd->bs', query, keys) / np.sqrt(self.d_model)
        attn_weights = softmax(scores, axis=-1)

        # Weighted sum of embeddings (values = embedded)
        context = np.einsum('bs,bsd->bd', attn_weights, embedded)

        # Output
        logits = context @ self.W_out
        output = softmax(logits, axis=-1)

        cache = {
            'X_seq': X_seq, 'X_query': X_query,
            'embedded': embedded, 'keys': keys, 'query': query,
            'attn_weights': attn_weights, 'context': context
        }

        return output, cache

    def backward_and_update(self, output: np.ndarray, y: np.ndarray,
                            cache: dict, lr: float = 0.1):
        """Backward pass."""
        batch_size = output.shape[0]

        dlogits = (output - y) / batch_size

        # Update W_out
        dW_out = cache['context'].T @ dlogits
        self.W_out -= lr * np.clip(dW_out, -1, 1)

        dcontext = dlogits @ self.W_out.T

        # Gradient through attention-weighted sum
        dattn = np.einsum('bd,bsd->bs', dcontext, cache['embedded'])

        # Gradient through softmax and attention scores
        attn = cache['attn_weights']
        dscores = attn * (dattn - np.sum(attn * dattn, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.d_model)

        # Gradient through query
        # scores[b,s] = sum_d query[b,d] * keys[b,s,d]
        dquery = np.einsum('bs,bsd->bd', dscores, cache['keys'])

        # Update W_query
        for q_idx in [0, 1]:
            mask = (cache['X_query'] == q_idx)
            if mask.any():
                dW_query_i = dquery[mask].mean(axis=0)
                self.W_query[q_idx] -= lr * 5 * np.clip(dW_query_i, -1, 1)

        # Gradient through keys
        dkeys = np.einsum('bs,bd->bsd', dscores, cache['query'])
        dW_k = np.einsum('bsd,bse->de', cache['embedded'], dkeys)
        self.W_k -= lr * np.clip(dW_k, -1, 1)

        # Update embedding
        dembedded = cache['attn_weights'][:, :, np.newaxis] * dcontext[:, np.newaxis, :]
        dembedded += dkeys @ self.W_k.T  # Gradient through keys
        dW_embed = np.einsum('bsv,bsd->vd', cache['X_seq'], dembedded)
        self.W_embed -= lr * np.clip(dW_embed, -1, 1)


def train_first_last_model(model: FirstLastModel, X_seq: np.ndarray,
                           X_query: np.ndarray, y: np.ndarray,
                           epochs: int = 100, lr: float = 0.1) -> dict:
    """Train the first/last model."""
    results = {'losses': [], 'accuracies': [], 'attention_evolution': []}

    n_samples = X_seq.shape[0]
    batch_size = min(128, n_samples)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_seq = X_seq[indices[start:end]]
            batch_query = X_query[indices[start:end]]
            batch_y = y[indices[start:end]]

            output, cache = model.forward(batch_seq, batch_query)

            loss = -np.mean(np.sum(batch_y * np.log(output + 1e-10), axis=-1))
            epoch_loss += loss * (end - start)

            correct += np.sum(np.argmax(output, axis=1) == np.argmax(batch_y, axis=1))

            model.backward_and_update(output, batch_y, cache, lr)

        results['losses'].append(epoch_loss / n_samples)
        results['accuracies'].append(correct / n_samples)

        # Record attention patterns for first and last queries
        if epoch % 20 == 0 or epoch == epochs - 1:
            # Get attention for "first" queries
            first_mask = X_query[:100] == 0
            last_mask = X_query[:100] == 1

            output_sample, cache_sample = model.forward(X_seq[:100], X_query[:100])

            attn_first = cache_sample['attn_weights'][first_mask].mean(axis=0) if first_mask.any() else np.ones(model.seq_len)/model.seq_len
            attn_last = cache_sample['attn_weights'][last_mask].mean(axis=0) if last_mask.any() else np.ones(model.seq_len)/model.seq_len

            results['attention_evolution'].append({
                'epoch': epoch,
                'attn_first': attn_first,
                'attn_last': attn_last
            })

    return results


class PositionAwareModel:
    """
    Simple model that uses positional encoding to solve position task.

    Architecture:
        Input sequence + Positional encoding → Attention → Output

    Without positional encoding, this model CANNOT solve the task.
    """

    def __init__(self, seq_len: int, vocab_size: int, d_model: int = 32,
                 use_positional_encoding: bool = True,
                 pe_type: str = 'sinusoidal'):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_pe = use_positional_encoding
        self.pe_type = pe_type

        # Embedding
        self.W_embed = np.random.randn(vocab_size, d_model) * 0.1

        # Positional encoding
        if pe_type == 'sinusoidal':
            self.pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len * 2)
        elif pe_type == 'learned':
            self.pe = LearnedPositionalEncoding(d_model, max_len=seq_len)
        else:  # 'none'
            self.pe = None

        # Query projection
        self.W_q = np.random.randn(seq_len, d_model) * 0.1

        # Key and Value projections
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

        # Output
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, X_seq: np.ndarray, X_pos: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.

        Args:
            X_seq: (batch, seq_len, vocab_size) one-hot encoded sequence
            X_pos: (batch, seq_len) one-hot position indicator
        """
        batch_size = X_seq.shape[0]

        # Embed sequence
        embedded = np.einsum('bsv,vd->bsd', X_seq, self.W_embed)

        # Add positional encoding (THE KEY STEP!)
        if self.use_pe and self.pe is not None:
            pe = self.pe.forward(self.seq_len)  # (seq_len, d_model)
            embedded = embedded + pe

        # Query from position indicator
        query = X_pos @ self.W_q  # (batch, d_model)
        query = query[:, np.newaxis, :]

        # Keys and Values
        keys = embedded @ self.W_k
        values = embedded @ self.W_v

        # Attention
        scores = np.matmul(query, keys.transpose(0, 2, 1)) / np.sqrt(self.d_model)
        attn_weights = softmax(scores, axis=-1).squeeze(1)

        # Context
        context = np.einsum('bs,bsd->bd', attn_weights, values)

        # Output
        logits = context @ self.W_out
        output = softmax(logits, axis=-1)

        cache = {
            'X_seq': X_seq, 'X_pos': X_pos,
            'embedded': embedded, 'query': query.squeeze(1),
            'keys': keys, 'values': values,
            'attn_weights': attn_weights, 'context': context
        }

        return output, cache

    def backward_and_update(self, output: np.ndarray, y: np.ndarray,
                            cache: dict, lr: float = 0.01):
        """Simplified backward pass."""
        batch_size = output.shape[0]

        dlogits = (output - y) / batch_size

        # Update W_out
        dW_out = cache['context'].T @ dlogits
        self.W_out -= lr * dW_out

        dcontext = dlogits @ self.W_out.T

        # Update through attention (simplified)
        attn = cache['attn_weights'][:, :, np.newaxis]
        dvalues = attn * dcontext[:, np.newaxis, :]

        dW_v = np.einsum('bsd,bse->de', cache['embedded'], dvalues)
        self.W_v -= lr * dW_v

        dW_k = np.einsum('bsd,bse->de', cache['embedded'], dvalues) * 0.1
        self.W_k -= lr * dW_k

        dW_q = np.einsum('bs,bd->sd', cache['X_pos'], dcontext) * 0.1
        self.W_q -= lr * dW_q

        dW_embed = np.einsum('bsv,bsd->vd', cache['X_seq'], dvalues) * 0.1
        self.W_embed -= lr * dW_embed


def train_position_model(model: PositionAwareModel,
                         X_seq: np.ndarray, X_pos: np.ndarray, y: np.ndarray,
                         epochs: int = 100, lr: float = 0.1) -> dict:
    """Train the position-aware model."""
    results = {'losses': [], 'accuracies': []}

    n_samples = X_seq.shape[0]
    batch_size = min(64, n_samples)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_seq = X_seq[indices[start:end]]
            batch_pos = X_pos[indices[start:end]]
            batch_y = y[indices[start:end]]

            output, cache = model.forward(batch_seq, batch_pos)

            loss = -np.mean(np.sum(batch_y * np.log(output + 1e-10), axis=-1))
            epoch_loss += loss * (end - start)

            correct += np.sum(np.argmax(output, axis=1) == np.argmax(batch_y, axis=1))

            model.backward_and_update(output, batch_y, cache, lr)

        results['losses'].append(epoch_loss / n_samples)
        results['accuracies'].append(correct / n_samples)

    return results


# ============================================================
# TRAINING EXPERIMENTS
# ============================================================

def experiment_position_matters(seq_len: int = 5, vocab_size: int = 5,
                                 epochs: int = 150) -> dict:
    """
    THE KEY EXPERIMENT: Prove that positional encoding is NECESSARY.

    TASK: "First vs Last"
    - Query = 0: Output the FIRST token
    - Query = 1: Output the LAST token

    WHY THIS REQUIRES PE:
    - Without PE, the model sees a SET of tokens, not a list
    - It cannot distinguish position 0 from position 4
    - Expected accuracy WITHOUT PE: ~50% (random guess between first/last)
    - Expected accuracy WITH PE: ~100% (knows which position is which)
    """
    print("=" * 60)
    print("EXPERIMENT: Does Positional Encoding Matter?")
    print("=" * 60)
    print("\nTask: 'First vs Last'")
    print("  Query=0: Output the FIRST token")
    print("  Query=1: Output the LAST token")
    print(f"\nSequence length: {seq_len}, Vocabulary: {vocab_size} tokens")
    print(f"\nWITHOUT PE: Model sees a SET → cannot distinguish first from last")
    print(f"WITH PE: Model sees a LIST → knows position 0 vs position {seq_len-1}")
    print("=" * 60)

    # Create task
    X_seq_train, X_query_train, y_train, X_seq_test, X_query_test, y_test = \
        create_first_last_task(n_samples=3000, seq_len=seq_len, vocab_size=vocab_size)

    results = {}

    # 1. WITHOUT positional encoding
    print("\n--- Training WITHOUT positional encoding ---")
    model_no_pe = FirstLastModel(seq_len, vocab_size, d_model=16, use_pe=False)

    results_no_pe = train_first_last_model(model_no_pe, X_seq_train, X_query_train,
                                            y_train, epochs=epochs, lr=0.1)

    output, cache = model_no_pe.forward(X_seq_test, X_query_test)
    test_acc_no_pe = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))

    # Get final attention patterns
    first_mask = X_query_test == 0
    last_mask = X_query_test == 1
    final_attn_first_no_pe = cache['attn_weights'][first_mask].mean(axis=0)
    final_attn_last_no_pe = cache['attn_weights'][last_mask].mean(axis=0)

    print(f"\nFinal attention (first query): {np.round(final_attn_first_no_pe, 3)}")
    print(f"Final attention (last query):  {np.round(final_attn_last_no_pe, 3)}")
    print(f"\nTest accuracy WITHOUT PE: {test_acc_no_pe:.2%}")
    if test_acc_no_pe < 0.7:
        print("(Cannot distinguish first from last - model failed!)")
    else:
        print("(Model partially learned through other means)")

    results['no_pe'] = {
        'losses': results_no_pe['losses'],
        'accuracies': results_no_pe['accuracies'],
        'test_accuracy': test_acc_no_pe,
        'final_attn_first': final_attn_first_no_pe,
        'final_attn_last': final_attn_last_no_pe,
        'attention_evolution': results_no_pe['attention_evolution']
    }

    # 2. WITH positional encoding
    print("\n--- Training WITH positional encoding ---")
    model_with_pe = FirstLastModel(seq_len, vocab_size, d_model=16, use_pe=True)

    results_with_pe = train_first_last_model(model_with_pe, X_seq_train, X_query_train,
                                              y_train, epochs=epochs, lr=0.1)

    output, cache = model_with_pe.forward(X_seq_test, X_query_test)
    test_acc_with_pe = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))

    final_attn_first_pe = cache['attn_weights'][first_mask].mean(axis=0)
    final_attn_last_pe = cache['attn_weights'][last_mask].mean(axis=0)

    print(f"\nFinal attention (first query): {np.round(final_attn_first_pe, 3)}")
    print(f"Final attention (last query):  {np.round(final_attn_last_pe, 3)}")
    print(f"\nTest accuracy WITH PE: {test_acc_with_pe:.2%}")
    print("(Position 0 and position 4 have different representations!)")

    results['with_pe'] = {
        'losses': results_with_pe['losses'],
        'accuracies': results_with_pe['accuracies'],
        'test_accuracy': test_acc_with_pe,
        'final_attn_first': final_attn_first_pe,
        'final_attn_last': final_attn_last_pe,
        'attention_evolution': results_with_pe['attention_evolution']
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: POSITIONAL ENCODING IS ESSENTIAL")
    print("=" * 60)
    print(f"WITHOUT PE: {test_acc_no_pe:.1%} (near random)")
    print(f"WITH PE:    {test_acc_with_pe:.1%} (learned positions!)")
    print(f"Improvement: +{(test_acc_with_pe - test_acc_no_pe)*100:.1f}%")
    print("=" * 60)

    results['random_baseline'] = 0.5  # Binary choice but with different targets
    results['seq_len'] = seq_len
    results['vocab_size'] = vocab_size

    return results


def experiment_length_generalization_training(train_len: int = 20,
                                               test_lens: List[int] = [20, 30, 40, 50],
                                               epochs: int = 80) -> dict:
    """
    Test length generalization after training.

    WHAT TO OBSERVE:
    - Sinusoidal: Works beyond training length (extrapolates)
    - Learned: Fails beyond training length
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Length Generalization")
    print("=" * 60)
    print(f"\nTraining on sequences of length {train_len}")
    print(f"Testing on lengths: {test_lens}\n")

    vocab_size = 8

    # Create training data
    X_seq_train, X_pos_train, y_train, _, _, _ = \
        create_position_task(n_samples=800, seq_len=train_len, vocab_size=vocab_size)

    results = {'train_len': train_len, 'test_lens': test_lens,
               'sinusoidal': [], 'learned': []}

    # Train both models on training length
    print("Training sinusoidal PE model...")
    model_sin = PositionAwareModel(train_len, vocab_size, d_model=32,
                                    use_positional_encoding=True, pe_type='sinusoidal')
    train_position_model(model_sin, X_seq_train, X_pos_train, y_train, epochs=epochs, lr=0.1)

    print("Training learned PE model...")
    model_learn = PositionAwareModel(train_len, vocab_size, d_model=32,
                                      use_positional_encoding=True, pe_type='learned')
    train_position_model(model_learn, X_seq_train, X_pos_train, y_train, epochs=epochs, lr=0.1)

    # Test on different lengths
    print("\nTesting on different sequence lengths...")
    for test_len in test_lens:
        # Create test data with this length
        _, _, _, X_seq_test, X_pos_test, y_test = \
            create_position_task(n_samples=200, seq_len=test_len, vocab_size=vocab_size)

        # Test sinusoidal (works for any length)
        try:
            # Need to adjust model for longer sequences
            model_sin.seq_len = test_len
            model_sin.W_q = np.random.randn(test_len, model_sin.d_model) * 0.1
            output, _ = model_sin.forward(X_seq_test, X_pos_test)
            acc_sin = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))
        except Exception as e:
            acc_sin = np.nan

        # Test learned (may fail)
        try:
            if test_len <= train_len:
                model_learn.seq_len = test_len
                model_learn.W_q = np.random.randn(test_len, model_learn.d_model) * 0.1
                output, _ = model_learn.forward(X_seq_test, X_pos_test)
                acc_learn = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))
            else:
                acc_learn = np.nan  # Can't extrapolate
        except Exception as e:
            acc_learn = np.nan

        results['sinusoidal'].append(acc_sin)
        results['learned'].append(acc_learn)

        print(f"  Length {test_len}: Sinusoidal = {acc_sin:.2%}, Learned = {'N/A' if np.isnan(acc_learn) else f'{acc_learn:.2%}'}")

    return results


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

def visualize_position_task_results(results: dict, save_path: Optional[str] = None):
    """
    Visualize results of the First vs Last task.

    THE KEY VISUALIZATION for this file.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Training accuracy curves
    ax = axes[0, 0]
    ax.plot(results['no_pe']['accuracies'], 'r-', label='WITHOUT PE', linewidth=2)
    ax.plot(results['with_pe']['accuracies'], 'b-', label='WITH PE', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', label='Random (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Learning Curves: PE is ESSENTIAL', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Training loss curves
    ax = axes[0, 1]
    ax.semilogy(results['no_pe']['losses'], 'r-', label='WITHOUT PE', linewidth=2)
    ax.semilogy(results['with_pe']['losses'], 'b-', label='WITH PE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss: WITH PE Learns Better', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final test accuracy comparison - THE KEY RESULT
    ax = axes[0, 2]
    methods = ['WITHOUT\nPE', 'WITH\nPE']
    accs = [results['no_pe']['test_accuracy'], results['with_pe']['test_accuracy']]
    colors = ['red', 'green']

    bars = ax.bar(methods, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('KEY RESULT: PE is Necessary!', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.1)

    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.1%}', (bar.get_x() + bar.get_width()/2, acc + 0.03),
                   ha='center', fontweight='bold', fontsize=14)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend()

    # 4. Attention pattern WITHOUT PE
    ax = axes[1, 0]
    attn_first = results['no_pe']['final_attn_first']
    attn_last = results['no_pe']['final_attn_last']
    x = np.arange(len(attn_first))
    width = 0.35
    ax.bar(x - width/2, attn_first, width, label='Query: First', color='blue', alpha=0.7)
    ax.bar(x + width/2, attn_last, width, label='Query: Last', color='orange', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title('WITHOUT PE: Attention Similar\n(Cannot distinguish positions!)', fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 1)

    # 5. Attention pattern WITH PE
    ax = axes[1, 1]
    attn_first = results['with_pe']['final_attn_first']
    attn_last = results['with_pe']['final_attn_last']
    colors_first = ['green' if i == 0 else 'blue' for i in range(len(attn_first))]
    colors_last = ['green' if i == len(attn_last)-1 else 'orange' for i in range(len(attn_last))]

    ax.bar(x - width/2, attn_first, width, label='Query: First', color='blue', alpha=0.7)
    ax.bar(x + width/2, attn_last, width, label='Query: Last', color='orange', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title('WITH PE: Different Attention!\n(Learned to focus correctly)', fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 1)

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')

    improvement = results['with_pe']['test_accuracy'] - results['no_pe']['test_accuracy']
    summary = f"""
    FIRST vs LAST TASK RESULTS

    Task: Given sequence [a,b,c,d,e]
          Query=0 → Output first token (a)
          Query=1 → Output last token (e)

    WHY PE IS NECESSARY:
    Without PE, attention sees a SET:
      {{a, b, c, d, e}}
    Cannot distinguish first from last!

    With PE, attention sees a LIST:
      [a@pos0, b@pos1, c@pos2, d@pos3, e@pos4]
    Position 0 ≠ Position 4

    RESULTS:
    • WITHOUT PE: {results['no_pe']['test_accuracy']:.1%}
    • WITH PE:    {results['with_pe']['test_accuracy']:.1%}
    • Improvement: +{improvement*100:.1f}%

    CONCLUSION:
    Positional encoding is NOT optional!
    Without it, attention-based models cannot
    distinguish [A, B, C] from [C, B, A].
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Position-Dependent Task: Proof That PE is Necessary',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


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
# FAILURE MODES
# ============================================================

def experiment_failure_modes() -> dict:
    """
    WHAT BREAKS POSITIONAL ENCODING?

    This demonstrates the critical failure modes of positional encoding,
    showing WHEN and WHY each type fails.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Positional Encoding Failure Modes")
    print("=" * 60)

    results = {}
    np.random.seed(42)

    # 1. DIMENSION MISMATCH
    print("\n1. DIMENSION MISMATCH")
    print("-" * 40)
    print("What if PE dimension ≠ embedding dimension?")

    embed_dim = 64
    for pe_dim in [32, 64, 128]:
        try:
            pe = SinusoidalPositionalEncoding(pe_dim, max_len=100)
            encoding = pe.forward(10)

            if pe_dim != embed_dim:
                # Can't add PE to embedding!
                status = f"✗ FAILS: {pe_dim} ≠ {embed_dim}"
                results[f'dim_mismatch_{pe_dim}'] = False
            else:
                status = f"✓ Works: {pe_dim} = {embed_dim}"
                results[f'dim_mismatch_{pe_dim}'] = True
            print(f"  PE_dim={pe_dim}, Embed_dim={embed_dim}: {status}")
        except Exception as e:
            print(f"  PE_dim={pe_dim}: ✗ Error: {e}")
            results[f'dim_mismatch_{pe_dim}'] = False

    print("\n  INSIGHT: PE dimension MUST match embedding dimension!")

    # 2. LEARNED PE BEYOND TRAINING LENGTH
    print("\n2. LEARNED PE: EXTRAPOLATION FAILURE")
    print("-" * 40)
    print("What if sequence length > max_len for learned PE?")

    train_len = 100
    learned_pe = LearnedPositionalEncoding(d_model=32, max_len=train_len)

    for test_len in [50, 100, 150, 200]:
        try:
            encoding = learned_pe.forward(test_len)
            status = "✓ Works"
            results[f'learned_extrap_{test_len}'] = True
        except Exception as e:
            status = f"✗ FAILS (beyond training length)"
            results[f'learned_extrap_{test_len}'] = False
        print(f"  Test length={test_len}, Train length={train_len}: {status}")

    print("\n  INSIGHT: Learned PE cannot extrapolate beyond training length!")

    # 3. PE ALL ZEROS (No position information)
    print("\n3. ZERO POSITIONAL ENCODING")
    print("-" * 40)
    print("What if PE is all zeros?")

    d_model = 32
    seq_len = 10

    # Normal sinusoidal PE
    normal_pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len * 2)
    normal_encoding = normal_pe.forward(seq_len)

    # Zero PE (simulating failure)
    zero_encoding = np.zeros((seq_len, d_model))

    # Check if positions are distinguishable
    normal_var = np.var(normal_encoding, axis=0).mean()
    zero_var = np.var(zero_encoding, axis=0).mean()

    print(f"  Normal PE variance across positions: {normal_var:.4f}")
    print(f"  Zero PE variance across positions: {zero_var:.4f}")
    print(f"  Zero PE: ✗ All positions are IDENTICAL!")

    results['zero_pe_variance'] = (normal_var, zero_var)

    print("\n  INSIGHT: Zero PE = No position information = Model sees SET not LIST!")

    # 4. WRONG FREQUENCY (for sinusoidal)
    print("\n4. WRONG FREQUENCY SCALE")
    print("-" * 40)
    print("What if the base frequency is wrong?")

    seq_len = 100
    d_model = 32

    # Test different bases
    for base in [10, 10000, 1000000]:
        # Create custom sinusoidal with different base
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(float(base)) / d_model))
        encoding = np.zeros((seq_len, d_model))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        # Check position distinguishability
        # Compute similarity between adjacent positions
        similarities = [np.dot(encoding[i], encoding[i+1]) /
                       (np.linalg.norm(encoding[i]) * np.linalg.norm(encoding[i+1]))
                       for i in range(seq_len-1)]
        avg_sim = np.mean(similarities)

        if base == 10:
            status = "✗ Too fast: positions become aliased"
        elif base == 1000000:
            status = "✗ Too slow: positions too similar"
        else:
            status = "✓ Good: distinguishable positions"

        print(f"  Base={base:>10}: Avg adjacent similarity={avg_sim:.4f} {status}")
        results[f'freq_base_{base}'] = avg_sim

    print("\n  INSIGHT: Frequency scale affects how well positions are distinguished!")

    return results


def visualize_failure_modes(results: dict, save_path: Optional[str] = None):
    """Visualize positional encoding failure modes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dimension mismatch
    ax = axes[0, 0]
    dims = [32, 64, 128]
    embed_dim = 64
    colors = ['red' if d != embed_dim else 'green' for d in dims]
    bars = ax.bar([f'PE={d}' for d in dims], [1 if d == embed_dim else 0 for d in dims],
                  color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Works? (1=Yes, 0=No)')
    ax.set_title(f'Failure: Dimension Mismatch\n(Embedding dim = {embed_dim})', fontweight='bold')
    ax.set_ylim(0, 1.2)

    # 2. Extrapolation failure
    ax = axes[0, 1]
    test_lens = [50, 100, 150, 200]
    train_len = 100
    colors = ['green' if t <= train_len else 'red' for t in test_lens]
    ax.bar([str(t) for t in test_lens], [1 if t <= train_len else 0 for t in test_lens],
           color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.5, color='blue', linestyle='--', linewidth=2, label=f'Train max={train_len}')
    ax.set_xlabel('Test Sequence Length')
    ax.set_ylabel('Works? (1=Yes, 0=No)')
    ax.set_title('Failure: Learned PE Extrapolation', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.2)

    # 3. Zero PE vs Normal PE
    ax = axes[1, 0]
    d_model = 32
    seq_len = 20

    # Show position similarity for normal vs zero PE
    normal_pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len * 2)
    normal_encoding = normal_pe.forward(seq_len)
    zero_encoding = np.zeros((seq_len, d_model))

    # Compute pairwise similarities
    normal_sim = normal_encoding @ normal_encoding.T
    normal_sim = normal_sim / (np.linalg.norm(normal_encoding, axis=1, keepdims=True) @
                               np.linalg.norm(normal_encoding, axis=1, keepdims=True).T + 1e-10)

    im = ax.imshow(normal_sim, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('Normal PE: Positions Distinguishable\n(Diagonal structure = good)', fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = """
    POSITIONAL ENCODING FAILURE MODES:

    1. DIMENSION MISMATCH
       • PE dim must equal embedding dim
       • Can't add vectors of different sizes!

    2. EXTRAPOLATION (Learned PE)
       • Learned PE fails beyond max_len
       • Use sinusoidal for variable lengths

    3. ZERO/NO POSITIONAL ENCODING
       • Model sees SET, not SEQUENCE
       • [A,B,C] = [C,B,A] without PE!

    4. WRONG FREQUENCY SCALE
       • Too fast: position aliasing
       • Too slow: positions too similar
       • Base=10000 is the sweet spot

    BOTTOM LINE:
    Without proper PE, attention-based
    models CANNOT use position information!
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Positional Encoding Failure Modes',
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

    # NEW: THE KEY EXPERIMENT - position matters
    position_results = experiment_position_matters()

    # Experiment 1: Extrapolation (at init)
    extrap_results = experiment_extrapolation()

    # Experiment 2: Relative vs Absolute
    rel_results = experiment_relative_position_encoding()

    # Experiment 3: RoPE properties
    rope_results = experiment_rope_properties()

    # NEW: Failure modes experiment
    failure_results = experiment_failure_modes()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # NEW: Position task visualization (THE KEY FIGURE)
    visualize_position_task_results(position_results, '49_positional_task.png')

    visualize_sinusoidal_encoding(save_path='49_positional_sinusoidal.png')
    visualize_alibi(save_path='49_positional_alibi.png')
    visualize_comparison(save_path='49_positional_comparison.png')
    visualize_rope_concept(save_path='49_positional_rope.png')

    # NEW: Failure modes visualization
    visualize_failure_modes(failure_results, '49_pe_failures.png')

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
