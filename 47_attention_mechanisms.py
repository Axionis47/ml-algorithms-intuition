"""
===============================================================
ATTENTION MECHANISMS — Paradigm: DYNAMIC WEIGHTING
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Attention lets the model CHOOSE what to focus on.

Instead of treating all inputs equally:
    output = f(x_1, x_2, ..., x_n)  (fixed weighting)

Attention computes DATA-DEPENDENT weights:
    output = Σ α_i × v_i    where α_i = softmax(score(q, k_i))

"Look at everything, but PAY ATTENTION to what matters."

===============================================================
THE THREE COMPONENTS: QUERY, KEY, VALUE
===============================================================

Think of it like a search:
    QUERY (Q):  What am I looking for?
    KEY (K):    What does each position offer?
    VALUE (V):  What information does each position have?

    score = Q · K^T          (how well does query match each key?)
    α = softmax(score)       (normalize to get weights)
    output = α × V           (weighted sum of values)

===============================================================
SCALED DOT-PRODUCT ATTENTION
===============================================================

Attention(Q, K, V) = softmax(Q K^T / √d_k) V

WHY SCALE BY √d_k?
- Dot products grow with dimension: E[q·k] = d_k when q,k ~ N(0,1)
- Large dot products → softmax saturates → vanishing gradients
- Scaling keeps variance ≈ 1

===============================================================
KEY VARIANTS
===============================================================

1. SELF-ATTENTION: Q, K, V all from same sequence
2. CROSS-ATTENTION: Q from one sequence, K/V from another
3. MULTI-HEAD: Multiple parallel attention operations
4. MASKED: Prevent attending to future positions (causal)
5. LOCAL/SPARSE: Only attend to nearby positions (efficiency)

===============================================================
INDUCTIVE BIAS
===============================================================

1. All positions can interact (unlike CNN's locality)
2. Permutation equivariant (needs positional encoding)
3. O(n²) complexity in sequence length
4. No inherent notion of distance or order

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


# ============================================================
# BASIC ATTENTION MECHANISMS
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class DotProductAttention:
    """
    Basic dot-product attention.

    score(q, k) = q · k

    INTUITION:
    - High dot product = query and key point in same direction
    - Softmax normalizes scores to sum to 1
    - Output is weighted average of values
    """

    def __init__(self):
        self.attention_weights = None

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            query: (batch, seq_q, d_k)
            key: (batch, seq_k, d_k)
            value: (batch, seq_k, d_v)
            mask: Optional mask (batch, seq_q, seq_k)

        Returns:
            output: (batch, seq_q, d_v)
        """
        # Compute attention scores
        scores = np.matmul(query, key.transpose(0, 2, 1))  # (batch, seq_q, seq_k)

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Softmax over keys
        self.attention_weights = softmax(scores, axis=-1)

        # Weighted sum of values
        output = np.matmul(self.attention_weights, value)

        return output


class ScaledDotProductAttention:
    """
    Scaled dot-product attention (used in Transformers).

    Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    THE SCALING IS CRUCIAL:
    Without scaling, if d_k is large:
    - Dot products have large magnitude
    - Softmax becomes very peaked (one position dominates)
    - Gradients vanish for non-dominant positions
    """

    def __init__(self):
        self.attention_weights = None

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Scaled dot-product attention.

        Args:
            query: (batch, seq_q, d_k) or (batch, heads, seq_q, d_k)
            key: (batch, seq_k, d_k) or (batch, heads, seq_k, d_k)
            value: (batch, seq_k, d_v) or (batch, heads, seq_k, d_v)
            mask: Optional attention mask

        Returns:
            output: Same shape as query (with d_v in last dim)
        """
        d_k = query.shape[-1]

        # Compute scaled scores
        scores = np.matmul(query, np.swapaxes(key, -2, -1)) / np.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Softmax and weighted sum
        self.attention_weights = softmax(scores, axis=-1)
        output = np.matmul(self.attention_weights, value)

        return output


class AdditiveAttention:
    """
    Additive attention (Bahdanau attention).

    score(q, k) = v^T tanh(W_q q + W_k k)

    COMPARISON TO DOT-PRODUCT:
    - More expressive (learned scoring function)
    - More parameters
    - Slower than dot-product
    - Popular in sequence-to-sequence models
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        """
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors
            hidden_dim: Hidden dimension for attention computation
        """
        self.W_q = np.random.randn(query_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(key_dim, hidden_dim) * 0.01
        self.v = np.random.randn(hidden_dim, 1) * 0.01

        self.attention_weights = None

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Additive attention.

        Args:
            query: (batch, seq_q, query_dim)
            key: (batch, seq_k, key_dim)
            value: (batch, seq_k, value_dim)

        Returns:
            output: (batch, seq_q, value_dim)
        """
        batch_size, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        # Transform query and key
        q_transformed = query @ self.W_q  # (batch, seq_q, hidden)
        k_transformed = key @ self.W_k    # (batch, seq_k, hidden)

        # Expand for broadcasting: each query attends to all keys
        q_expanded = q_transformed[:, :, np.newaxis, :]  # (batch, seq_q, 1, hidden)
        k_expanded = k_transformed[:, np.newaxis, :, :]  # (batch, 1, seq_k, hidden)

        # Compute scores
        combined = np.tanh(q_expanded + k_expanded)  # (batch, seq_q, seq_k, hidden)
        scores = (combined @ self.v).squeeze(-1)     # (batch, seq_q, seq_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        self.attention_weights = softmax(scores, axis=-1)
        output = np.matmul(self.attention_weights, value)

        return output


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention:
    """
    Multi-Head Attention (the core of Transformers).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    WHY MULTIPLE HEADS?
    - Different heads can focus on different aspects
    - One head might capture syntax, another semantics
    - Like having multiple "experts" voting

    EFFICIENCY TRICK:
    Instead of h separate attention operations:
    - Project to d_model, reshape to (batch, heads, seq, d_k)
    - Single batched attention operation
    - Reshape and project back
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projection matrices
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.attention = ScaledDotProductAttention()
        self.attention_weights = None

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Multi-head attention forward pass.

        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_q, d_model)
        """
        batch_size = query.shape[0]

        # Linear projections
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        # Reshape to (batch, heads, seq, d_k)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Expand mask for heads if provided
        if mask is not None:
            mask = mask[:, np.newaxis, :, :]  # (batch, 1, seq_q, seq_k)

        # Attention
        attn_output = self.attention.forward(Q, K, V, mask)
        self.attention_weights = self.attention.attention_weights

        # Reshape back: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # Final projection
        output = attn_output @ self.W_o

        return output


# ============================================================
# SELF-ATTENTION AND CROSS-ATTENTION
# ============================================================

class SelfAttention:
    """
    Self-Attention: Query, Key, Value all come from the same sequence.

    Used in:
    - Transformer encoder (bidirectional)
    - Transformer decoder (with causal mask)

    WHAT IT COMPUTES:
    For each position, how much should it attend to every other position?
    This captures dependencies between all pairs of positions.
    """

    def __init__(self, d_model: int, num_heads: int):
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Self-attention: Q = K = V = x

        Args:
            x: (batch, seq, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq, d_model)
        """
        return self.attention.forward(x, x, x, mask)


class CrossAttention:
    """
    Cross-Attention: Query from one sequence, Key/Value from another.

    Used in:
    - Transformer decoder (attending to encoder output)
    - Vision-Language models
    - Encoder-decoder architectures

    WHAT IT COMPUTES:
    For each position in sequence A, how much should it attend
    to each position in sequence B?
    """

    def __init__(self, d_model: int, num_heads: int):
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, query_seq: np.ndarray, context_seq: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cross-attention: Q from query_seq, K/V from context_seq

        Args:
            query_seq: (batch, seq_q, d_model)
            context_seq: (batch, seq_k, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_q, d_model)
        """
        return self.attention.forward(query_seq, context_seq, context_seq, mask)


# ============================================================
# CAUSAL (MASKED) ATTENTION
# ============================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask to prevent attending to future positions.

    Returns upper triangular mask where True = can attend.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


class CausalSelfAttention:
    """
    Causal Self-Attention: Can only attend to past and current positions.

    Used in:
    - GPT-style language models (autoregressive)
    - Transformer decoders

    THE MASK:
    Position i can only attend to positions 0, 1, ..., i
    This prevents "cheating" by looking at future tokens.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.max_seq_len = max_seq_len
        # Pre-compute causal mask
        self.causal_mask = create_causal_mask(max_seq_len)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Causal self-attention.

        Args:
            x: (batch, seq, d_model)

        Returns:
            output: (batch, seq, d_model)
        """
        seq_len = x.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]
        mask = mask[np.newaxis, :, :]  # (1, seq, seq) for broadcasting

        return self.attention.forward(x, x, x, mask)


# ============================================================
# SPARSE ATTENTION PATTERNS
# ============================================================

class LocalAttention:
    """
    Local Attention: Only attend to nearby positions.

    Instead of O(n²) full attention, use O(n × window_size).

    TRADE-OFF:
    - Pro: Much faster for long sequences
    - Con: Can't capture long-range dependencies directly
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int):
        """
        Args:
            window_size: Number of positions to attend to on each side
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.window_size = window_size

    def create_local_mask(self, seq_len: int) -> np.ndarray:
        """Create mask that only allows attending to nearby positions."""
        mask = np.zeros((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True
        return mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Local self-attention."""
        seq_len = x.shape[1]
        mask = self.create_local_mask(seq_len)
        mask = mask[np.newaxis, :, :]

        return self.attention.forward(x, x, x, mask)


# ============================================================
# TRAINING INFRASTRUCTURE
# ============================================================

def create_copy_task(n_samples: int = 1000, seq_len: int = 10,
                     vocab_size: int = 8, copy_pos: int = None) -> Tuple:
    """
    Create a COPY task: output the token at a specific position.

    Task: Given sequence [a, b, c, d, e], and query "position 2",
          output "c".

    This task REQUIRES attention to succeed.
    """
    np.random.seed(42)

    # Random sequences
    X = np.random.randint(0, vocab_size, (n_samples, seq_len))

    # Random position to copy (or fixed)
    if copy_pos is None:
        positions = np.random.randint(0, seq_len, n_samples)
    else:
        positions = np.full(n_samples, copy_pos)

    # Target: the token at that position
    y = X[np.arange(n_samples), positions]

    # One-hot encode
    X_onehot = np.zeros((n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for j in range(seq_len):
            X_onehot[i, j, X[i, j]] = 1

    y_onehot = np.zeros((n_samples, vocab_size))
    y_onehot[np.arange(n_samples), y] = 1

    # Position indicator (which position to copy)
    pos_indicator = np.zeros((n_samples, seq_len))
    pos_indicator[np.arange(n_samples), positions] = 1

    split = int(0.8 * n_samples)
    return (X_onehot[:split], pos_indicator[:split], y_onehot[:split],
            X_onehot[split:], pos_indicator[split:], y_onehot[split:])


def create_simple_copy_task(n_samples: int = 2000, seq_len: int = 5,
                            vocab_size: int = 5) -> Tuple:
    """
    SIMPLIFIED copy task for clear demonstration.

    Task: Always copy the token at position 0 (first position).

    WHY THIS WORKS FOR DEMONSTRATION:
    - Fixed position = model only needs to learn ONE attention pattern
    - Shorter sequence = easier gradient flow
    - Smaller vocab = cleaner signal

    Expected results:
    - Random guess: 20% (1/5 vocab)
    - Trained model: >90% (learns to attend to position 0)
    """
    np.random.seed(42)

    # Random sequences
    X = np.random.randint(0, vocab_size, (n_samples, seq_len))

    # Always copy position 0
    y = X[:, 0]

    # One-hot encode
    X_onehot = np.zeros((n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for j in range(seq_len):
            X_onehot[i, j, X[i, j]] = 1

    y_onehot = np.zeros((n_samples, vocab_size))
    y_onehot[np.arange(n_samples), y] = 1

    # Position indicator - always position 0
    pos_indicator = np.zeros((n_samples, seq_len))
    pos_indicator[:, 0] = 1

    split = int(0.8 * n_samples)
    return (X_onehot[:split], pos_indicator[:split], y_onehot[:split],
            X_onehot[split:], pos_indicator[split:], y_onehot[split:])


class SimpleAttentionModel:
    """
    Ultra-simple attention model that DIRECTLY demonstrates attention learning.

    Architecture:
        1. Sequence → Values (V = X @ W_v)
        2. Learnable attention scores (directly parameterized)
        3. Output = softmax(scores) @ V → logits

    This bypasses Q/K complexity to show attention's core behavior.
    """

    def __init__(self, seq_len: int, vocab_size: int, d_model: int = 16):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Value projection
        self.W_v = np.random.randn(vocab_size, d_model) * 0.1

        # LEARNABLE attention logits (the key parameter!)
        # This will learn to focus on position 0
        self.attention_logits = np.zeros(seq_len)  # Start uniform

        # Output projection
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.
        X: (batch, seq_len, vocab_size) one-hot
        """
        batch_size = X.shape[0]

        # Values: (batch, seq_len, d_model)
        V = np.einsum('bsv,vd->bsd', X, self.W_v)

        # Attention weights from learnable logits
        attn = softmax(self.attention_logits)  # (seq_len,)

        # Weighted sum: (batch, d_model)
        context = np.einsum('s,bsd->bd', attn, V)

        # Output logits
        logits = context @ self.W_out  # (batch, vocab_size)
        output = softmax(logits, axis=-1)

        cache = {'X': X, 'V': V, 'attn': attn, 'context': context}
        return output, cache

    def backward_and_update(self, output: np.ndarray, y: np.ndarray,
                            cache: dict, lr: float = 0.1):
        """Backward pass with gradient computation."""
        batch_size = output.shape[0]

        # Gradient of cross-entropy loss
        dlogits = (output - y) / batch_size

        # Gradient through W_out
        dW_out = cache['context'].T @ dlogits
        self.W_out -= lr * np.clip(dW_out, -1, 1)

        # Gradient to context
        dcontext = dlogits @ self.W_out.T  # (batch, d_model)

        # Gradient through attention (THE KEY!)
        # context = sum_s attn[s] * V[batch, s, :]
        # dL/dattn[s] = sum_batch sum_d dcontext[batch, d] * V[batch, s, d]
        dattn = np.einsum('bd,bsd->s', dcontext, cache['V'])

        # Gradient through softmax
        attn = cache['attn']
        # dsoftmax: dattn_logits = attn * (dattn - sum(attn * dattn))
        dattn_logits = attn * (dattn - np.sum(attn * dattn))

        self.attention_logits -= lr * 10 * dattn_logits  # Higher LR for attention

        # Gradient through V
        dV = np.einsum('s,bd->bsd', attn, dcontext)
        dW_v = np.einsum('bsv,bsd->vd', cache['X'], dV)
        self.W_v -= lr * np.clip(dW_v, -1, 1)


def train_simple_attention(model: SimpleAttentionModel, X_train: np.ndarray,
                           y_train: np.ndarray, epochs: int = 100,
                           lr: float = 0.1) -> dict:
    """Train the simple attention model."""
    results = {
        'losses': [],
        'accuracies': [],
        'attention_evolution': []
    }

    n_samples = X_train.shape[0]
    batch_size = min(128, n_samples)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X_train[indices[start:end]]
            batch_y = y_train[indices[start:end]]

            output, cache = model.forward(batch_X)

            loss = -np.mean(np.sum(batch_y * np.log(output + 1e-10), axis=-1))
            epoch_loss += loss * (end - start)

            correct += np.sum(np.argmax(output, axis=1) == np.argmax(batch_y, axis=1))

            model.backward_and_update(output, batch_y, cache, lr)

        results['losses'].append(epoch_loss / n_samples)
        results['accuracies'].append(correct / n_samples)

        # Record attention evolution
        if epoch % 10 == 0 or epoch == epochs - 1:
            results['attention_evolution'].append({
                'epoch': epoch,
                'attention': softmax(model.attention_logits).copy()
            })

    return results


class AttentionCopyModel:
    """
    Model that uses attention to solve the copy task.

    Architecture:
        Input sequence → Attention (query = position indicator) → Output

    This demonstrates that attention LEARNS to focus on the right position.
    """

    def __init__(self, seq_len: int, vocab_size: int, d_model: int = 32,
                 use_scaling: bool = True):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_scaling = use_scaling

        # Embed tokens
        self.W_embed = np.random.randn(vocab_size, d_model) * 0.1

        # Query projection (from position indicator)
        self.W_q = np.random.randn(seq_len, d_model) * 0.1

        # Key and Value projections
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

        # Output projection
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1

        self.attention_weights = None

    def forward(self, X: np.ndarray, pos_indicator: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.

        Args:
            X: (batch, seq_len, vocab_size) one-hot encoded sequence
            pos_indicator: (batch, seq_len) which position to copy

        Returns:
            output: (batch, vocab_size) predicted token
            cache: dict for backward pass
        """
        batch_size = X.shape[0]

        # Embed sequence
        # X: (batch, seq_len, vocab_size) @ W_embed: (vocab_size, d_model)
        embedded = np.einsum('bsv,vd->bsd', X, self.W_embed)  # (batch, seq_len, d_model)

        # Query from position indicator
        query = pos_indicator @ self.W_q  # (batch, d_model)
        query = query[:, np.newaxis, :]  # (batch, 1, d_model)

        # Keys and Values
        keys = embedded @ self.W_k  # (batch, seq_len, d_model)
        values = embedded @ self.W_v  # (batch, seq_len, d_model)

        # Attention scores
        scores = np.matmul(query, keys.transpose(0, 2, 1))  # (batch, 1, seq_len)

        if self.use_scaling:
            scores = scores / np.sqrt(self.d_model)

        # Softmax
        attn_weights = softmax(scores, axis=-1)  # (batch, 1, seq_len)
        self.attention_weights = attn_weights.squeeze(1)  # (batch, seq_len)

        # Weighted sum of values
        context = np.matmul(attn_weights, values)  # (batch, 1, d_model)
        context = context.squeeze(1)  # (batch, d_model)

        # Output
        logits = context @ self.W_out  # (batch, vocab_size)
        output = softmax(logits, axis=-1)

        cache = {
            'X': X, 'pos_indicator': pos_indicator,
            'embedded': embedded, 'query': query.squeeze(1),
            'keys': keys, 'values': values,
            'attn_weights': self.attention_weights,
            'context': context, 'logits': logits
        }

        return output, cache

    def backward_and_update(self, output: np.ndarray, y: np.ndarray,
                            cache: dict, lr: float = 0.01):
        """Simplified backward pass and update with gradient clipping."""
        batch_size = output.shape[0]

        # Gradient of cross-entropy + softmax
        dlogits = (output - y) / batch_size  # (batch, vocab_size)

        # Gradient through W_out
        dW_out = cache['context'].T @ dlogits
        dW_out = np.clip(dW_out, -1.0, 1.0)  # Clip gradients
        self.W_out -= lr * dW_out

        # Gradient to context
        dcontext = dlogits @ self.W_out.T  # (batch, d_model)

        # Gradient through attention (simplified - just update W_q, W_k, W_v)
        # This is approximate but sufficient for demonstration

        # Gradient through values
        attn = cache['attn_weights'][:, :, np.newaxis]  # (batch, seq_len, 1)
        dvalues = attn * dcontext[:, np.newaxis, :]  # (batch, seq_len, d_model)

        dW_v = np.einsum('bsd,bse->de', cache['embedded'], dvalues)
        dW_v = np.clip(dW_v, -1.0, 1.0)
        self.W_v -= lr * dW_v

        # Gradient through keys - stronger signal now
        dW_k = np.einsum('bsd,bse->de', cache['embedded'], dvalues) * 0.5
        dW_k = np.clip(dW_k, -1.0, 1.0)
        self.W_k -= lr * dW_k

        # Gradient through query - stronger signal now
        dW_q = np.einsum('bs,bd->sd', cache['pos_indicator'], dcontext) * 0.5
        dW_q = np.clip(dW_q, -1.0, 1.0)
        self.W_q -= lr * dW_q

        # Gradient through embedding - stronger signal now
        dW_embed = np.einsum('bsv,bsd->vd', cache['X'], dvalues) * 0.5
        dW_embed = np.clip(dW_embed, -1.0, 1.0)
        self.W_embed -= lr * dW_embed


def train_attention_copy(model: AttentionCopyModel, X_train: np.ndarray,
                         pos_train: np.ndarray, y_train: np.ndarray,
                         epochs: int = 100, lr: float = 0.1) -> dict:
    """Train the attention copy model."""
    results = {
        'losses': [],
        'accuracies': [],
        'attention_patterns': []  # Store attention at key epochs
    }

    n_samples = X_train.shape[0]
    batch_size = min(64, n_samples)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X_train[indices[start:end]]
            batch_pos = pos_train[indices[start:end]]
            batch_y = y_train[indices[start:end]]

            # Forward
            output, cache = model.forward(batch_X, batch_pos)

            # Loss
            loss = -np.mean(np.sum(batch_y * np.log(output + 1e-10), axis=-1))
            epoch_loss += loss * (end - start)

            # Accuracy
            correct += np.sum(np.argmax(output, axis=1) == np.argmax(batch_y, axis=1))

            # Backward and update
            model.backward_and_update(output, batch_y, cache, lr)

        results['losses'].append(epoch_loss / n_samples)
        results['accuracies'].append(correct / n_samples)

        # Store attention pattern at key epochs
        if epoch in [0, 10, 30, epochs-1]:
            # Get attention on first few samples
            output, cache = model.forward(X_train[:5], pos_train[:5])
            results['attention_patterns'].append({
                'epoch': epoch,
                'attention': model.attention_weights.copy(),
                'positions': np.argmax(pos_train[:5], axis=1)
            })

    return results


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_attention_learns_copy(seq_len: int = 5, vocab_size: int = 5,
                                      epochs: int = 150) -> dict:
    """
    THE KEY EXPERIMENT: Train attention to solve the copy task.

    WHAT TO OBSERVE:
    - Initially: Attention is UNIFORM (0.2, 0.2, 0.2, 0.2, 0.2)
    - After training: Attention focuses SHARPLY on position 0 (~0.9, 0.025, ...)
    - This proves attention LEARNS to focus where it matters

    This is the "aha moment" for understanding attention.
    """
    print("=" * 60)
    print("EXPERIMENT: Attention Learns to Copy")
    print("=" * 60)
    print("\nTask: Always output the FIRST token of the sequence.")
    print("The model must learn to attend to position 0.")
    print(f"\nRandom guess baseline: {100/vocab_size:.1f}%")
    print("=" * 60)

    # Create SIMPLE task - always copy position 0
    X_train, _, y_train, X_test, _, y_test = create_simple_copy_task(
        n_samples=3000, seq_len=seq_len, vocab_size=vocab_size
    )

    # Train SIMPLE attention model (directly learnable attention weights)
    print("\n--- Training Simple Attention Model ---")
    print("(Directly learns attention weights over positions)")

    model_simple = SimpleAttentionModel(seq_len, vocab_size, d_model=16)

    # Show initial attention (should be uniform)
    initial_attn = softmax(model_simple.attention_logits)
    print(f"\nINITIAL attention: {np.round(initial_attn, 3)}")
    print("(Uniform - model doesn't know where to look)")

    # Train
    results_simple = train_simple_attention(model_simple, X_train, y_train,
                                            epochs=epochs, lr=0.1)

    # Test
    output, _ = model_simple.forward(X_test)
    test_acc_simple = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))

    # Final attention
    final_attn = softmax(model_simple.attention_logits)
    print(f"\nFINAL attention: {np.round(final_attn, 3)}")
    print("(Should focus heavily on position 0!)")

    print(f"\n  Final training accuracy: {results_simple['accuracies'][-1]:.2%}")
    print(f"  Test accuracy: {test_acc_simple:.2%}")
    print(f"  Improvement over random: {test_acc_simple - 1/vocab_size:.2%}")

    # Compare with a model that has FIXED uniform attention (baseline)
    print("\n--- Baseline: Fixed Uniform Attention ---")
    print("(Cannot learn - always attends equally to all positions)")

    # Compute what uniform attention would give
    # With uniform attention, output is average of all values
    # This can't distinguish position 0 from others
    uniform_acc = 1 / vocab_size  # Random chance
    print(f"  Uniform attention accuracy: ~{uniform_acc:.2%} (random)")

    return {
        'simple': {
            'accuracies': results_simple['accuracies'],
            'losses': results_simple['losses'],
            'attention_evolution': results_simple['attention_evolution']
        },
        'test_acc_simple': test_acc_simple,
        'test_acc_uniform': uniform_acc,
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'random_baseline': 1/vocab_size,
        'initial_attention': initial_attn,
        'final_attention': final_attn
    }


def experiment_scaling_effect(d_ks: List[int] = [8, 32, 64, 128, 256, 512],
                              seq_len: int = 20) -> dict:
    """
    Show why scaling by √d_k is necessary.

    WHAT TO OBSERVE:
    - Without scaling: Attention becomes very peaked as d_k increases
    - With scaling: Attention distribution stays reasonable
    """
    print("=" * 60)
    print("EXPERIMENT: Effect of Scaling in Attention")
    print("=" * 60)

    results = {'d_k': d_ks,
               'unscaled_entropy': [],
               'scaled_entropy': []}

    for d_k in d_ks:
        # Random queries and keys
        Q = np.random.randn(1, seq_len, d_k)
        K = np.random.randn(1, seq_len, d_k)

        # Unscaled attention
        scores_unscaled = np.matmul(Q, K.transpose(0, 2, 1))
        attn_unscaled = softmax(scores_unscaled, axis=-1)

        # Scaled attention
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        attn_scaled = softmax(scores_scaled, axis=-1)

        # Compute entropy (higher = more uniform attention)
        def entropy(p):
            p = np.clip(p, 1e-10, 1)
            return -np.sum(p * np.log(p), axis=-1).mean()

        results['unscaled_entropy'].append(entropy(attn_unscaled))
        results['scaled_entropy'].append(entropy(attn_scaled))

    print("\nAttention entropy (higher = more uniform distribution):")
    print("-" * 50)
    print(f"{'d_k':<10} {'Unscaled':<15} {'Scaled':<15}")
    print("-" * 50)
    for i, d_k in enumerate(d_ks):
        print(f"{d_k:<10} {results['unscaled_entropy'][i]:<15.4f} {results['scaled_entropy'][i]:<15.4f}")

    return results


def experiment_attention_patterns(seq_len: int = 20, d_model: int = 64) -> dict:
    """
    Visualize different attention patterns.

    WHAT TO OBSERVE:
    - Full attention: All positions can attend to all
    - Causal: Lower triangular
    - Local: Band diagonal
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Attention Patterns")
    print("=" * 60)

    # Create sample sequence
    x = np.random.randn(1, seq_len, d_model)

    results = {}

    # Full self-attention
    full_attn = SelfAttention(d_model, num_heads=4)
    _ = full_attn.forward(x)
    results['full'] = full_attn.attention.attention_weights[0, 0]  # First head

    # Causal attention
    causal_attn = CausalSelfAttention(d_model, num_heads=4, max_seq_len=seq_len)
    _ = causal_attn.forward(x)
    results['causal'] = causal_attn.attention.attention_weights[0, 0]

    # Local attention
    local_attn = LocalAttention(d_model, num_heads=4, window_size=3)
    _ = local_attn.forward(x)
    results['local'] = local_attn.attention.attention_weights[0, 0]

    print("\nGenerated attention patterns for visualization.")
    return results


def experiment_multi_head_diversity(num_heads_list: List[int] = [1, 2, 4, 8],
                                    seq_len: int = 20,
                                    d_model: int = 64) -> dict:
    """
    Show how different heads learn different patterns.

    WHAT TO OBSERVE:
    - Single head: One attention pattern
    - Multiple heads: Diverse patterns (hopefully)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Multi-Head Diversity")
    print("=" * 60)

    x = np.random.randn(1, seq_len, d_model)

    results = {}

    for num_heads in num_heads_list:
        mha = MultiHeadAttention(d_model, num_heads)
        _ = mha.forward(x, x, x)

        # Compute diversity: variance of attention patterns across heads
        attn_weights = mha.attention_weights[0]  # (heads, seq, seq)

        # Mean pairwise cosine similarity between heads
        heads_flat = attn_weights.reshape(num_heads, -1)
        similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                sim = np.dot(heads_flat[i], heads_flat[j]) / (
                    np.linalg.norm(heads_flat[i]) * np.linalg.norm(heads_flat[j]) + 1e-10)
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 1.0
        results[num_heads] = {
            'attention_weights': attn_weights,
            'avg_head_similarity': avg_similarity
        }

        print(f"Num heads: {num_heads}, Avg head similarity: {avg_similarity:.4f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_attention_learns(results: dict, save_path: Optional[str] = None):
    """
    Visualize how attention LEARNS to focus on position 0.

    THE KEY VISUALIZATION for this file.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training curves - accuracy
    ax = axes[0, 0]
    ax.plot(results['simple']['accuracies'], 'b-', label='Learnable Attention', linewidth=2)
    ax.axhline(y=results['random_baseline'], color='gray', linestyle=':', label='Random/Uniform', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Learning Curve: Accuracy Improves!', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Training curves - loss
    ax = axes[0, 1]
    ax.semilogy(results['simple']['losses'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss Decreases = Learning', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Final test accuracy comparison - THE KEY RESULT
    ax = axes[0, 2]
    accs = [results['test_acc_simple'], results['random_baseline']]
    labels = ['Learned\nAttention', 'Uniform\n(Random)']
    colors = ['green', 'gray']
    bars = ax.bar(labels, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('KEY RESULT: Attention Learns!', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.1%}', (bar.get_x() + bar.get_width()/2, acc + 0.03),
                   ha='center', fontweight='bold', fontsize=14)
    # Add improvement annotation
    improvement = results['test_acc_simple'] - results['random_baseline']
    ax.annotate(f'+{improvement:.1%}', (0.5, 0.5), fontsize=16, color='green',
               fontweight='bold', ha='center')

    # 4. BEFORE: Initial attention (uniform)
    ax = axes[1, 0]
    attn_init = results['initial_attention']
    positions = range(len(attn_init))
    ax.bar(positions, attn_init, color='red', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Attention Weight')
    ax.set_title('BEFORE Training: Uniform Attention', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1/len(attn_init), color='gray', linestyle='--', alpha=0.7)

    # 5. AFTER: Learned attention (focused on position 0)
    ax = axes[1, 1]
    attn_final = results['final_attention']
    colors_attn = ['green' if i == 0 else 'steelblue' for i in positions]
    bars = ax.bar(positions, attn_final, color=colors_attn, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Attention Weight')
    ax.set_title('AFTER Training: Focused on Position 0!', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_ylim(0, 1.0)
    # Highlight position 0
    if attn_final[0] > 0.5:
        ax.annotate(f'{attn_final[0]:.1%}', (0, attn_final[0] + 0.05),
                   ha='center', fontweight='bold', color='green', fontsize=14)

    # 6. Attention evolution over training
    ax = axes[1, 2]
    evolution = results['simple']['attention_evolution']
    epochs_recorded = [e['epoch'] for e in evolution]
    pos0_weights = [e['attention'][0] for e in evolution]

    ax.plot(epochs_recorded, pos0_weights, 'go-', linewidth=2, markersize=8, label='Position 0')

    # Plot other positions
    for pos in range(1, results['seq_len']):
        other_weights = [e['attention'][pos] for e in evolution]
        ax.plot(epochs_recorded, other_weights, 'o--', alpha=0.5, markersize=4,
               label=f'Position {pos}' if pos == 1 else None)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Evolution: Position 0 Wins!', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.suptitle('ATTENTION LEARNS TO FOCUS\n'
                 f'Task: "Copy the first token" | Random: {results["random_baseline"]:.0%} → Learned: {results["test_acc_simple"]:.0%}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_attention_mechanism(save_path: Optional[str] = None):
    """
    Visual explanation of how attention works.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Query-Key matching
    ax = axes[0]
    ax.set_title('Step 1: Query-Key Matching', fontweight='bold')

    # Draw query and keys
    query_y = 2
    key_ys = [3, 2.5, 2, 1.5, 1]

    ax.annotate('Query', (0.5, query_y), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue'))

    for i, y in enumerate(key_ys):
        ax.annotate(f'Key {i+1}', (3, y), fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow'))
        # Draw arrow with score
        score = np.random.uniform(0.1, 0.9)
        ax.annotate('', xy=(2.5, y), xytext=(1, query_y),
                   arrowprops=dict(arrowstyle='->', color='gray',
                                 alpha=score, lw=2))
        ax.annotate(f'{score:.2f}', (1.75, (y + query_y)/2), fontsize=8)

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # 2. Softmax normalization
    ax = axes[1]
    ax.set_title('Step 2: Softmax → Weights', fontweight='bold')

    scores = np.array([0.8, 0.3, 0.5, 0.2, 0.1])
    weights = softmax(scores)

    bars = ax.barh(range(5), weights, color='steelblue', alpha=0.7)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Key {i+1}' for i in range(5)])
    ax.set_xlabel('Attention Weight')

    for bar, w in zip(bars, weights):
        ax.annotate(f'{w:.2f}', (w + 0.02, bar.get_y() + bar.get_height()/2),
                   va='center', fontsize=10)

    ax.set_xlim(0, 1)

    # 3. Weighted sum
    ax = axes[2]
    ax.set_title('Step 3: Weighted Sum of Values', fontweight='bold')

    ax.text(0.5, 0.9, 'Output = Σ αᵢ × Vᵢ', fontsize=14, ha='center',
           transform=ax.transAxes, fontweight='bold')

    formula = r'= α₁V₁ + α₂V₂ + α₃V₃ + α₄V₄ + α₅V₅'
    ax.text(0.5, 0.7, formula, fontsize=11, ha='center', transform=ax.transAxes)

    # Show the weights
    weighted = ' + '.join([f'{w:.2f}×V{i+1}' for i, w in enumerate(weights)])
    ax.text(0.5, 0.4, f'= {weighted[:40]}...', fontsize=9, ha='center',
           transform=ax.transAxes)

    ax.text(0.5, 0.15, 'Higher weight → more influence on output',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    ax.axis('off')

    plt.suptitle('How Attention Works: Query-Key-Value', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_attention_patterns(results: dict, save_path: Optional[str] = None):
    """
    Visualize different attention pattern types.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    patterns = [
        ('Full Self-Attention', 'full'),
        ('Causal (Masked)', 'causal'),
        ('Local (Window=3)', 'local')
    ]

    for ax, (title, key) in zip(axes, patterns):
        attn = results[key]
        im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=np.max(attn))
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Attention Patterns: What Can Each Position See?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_scaling_importance(results: dict, save_path: Optional[str] = None):
    """
    Visualize why scaling by sqrt(d_k) matters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot entropy vs d_k
    ax = axes[0]
    ax.plot(results['d_k'], results['unscaled_entropy'], 'r-o',
           label='Unscaled', linewidth=2, markersize=8)
    ax.plot(results['d_k'], results['scaled_entropy'], 'b-s',
           label='Scaled (÷√d_k)', linewidth=2, markersize=8)
    ax.set_xlabel('Key Dimension (d_k)')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy vs Dimension\n(Higher = more uniform)',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show example attention distributions
    ax = axes[1]

    # Generate example
    d_k = 256
    seq_len = 10
    Q = np.random.randn(1, 1, d_k)
    K = np.random.randn(1, seq_len, d_k)

    scores = np.matmul(Q, K.transpose(0, 2, 1)).squeeze()
    attn_unscaled = softmax(scores)
    attn_scaled = softmax(scores / np.sqrt(d_k))

    x_pos = np.arange(seq_len)
    width = 0.35

    ax.bar(x_pos - width/2, attn_unscaled, width, label='Unscaled', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, attn_scaled, width, label='Scaled', color='blue', alpha=0.7)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Attention Distribution (d_k={d_k})\nUnscaled → too peaked!',
                fontweight='bold')
    ax.legend()
    ax.set_xticks(x_pos)

    plt.suptitle('Why Scale by √d_k?', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_multi_head_attention(results: dict, save_path: Optional[str] = None):
    """
    Visualize how different heads attend differently.
    """
    # Get 8-head results
    if 8 not in results:
        print("Need 8-head results for this visualization")
        return

    attn = results[8]['attention_weights']  # (8, seq, seq)
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(attn[i], cmap='viridis', vmin=0)
        ax.set_title(f'Head {i+1}', fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    plt.suptitle('Multi-Head Attention: Different Heads, Different Patterns',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_self_vs_cross_attention(save_path: Optional[str] = None):
    """
    Visual comparison of self-attention vs cross-attention.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Self-attention
    ax = axes[0]
    ax.set_title('Self-Attention\n(Q, K, V from same sequence)', fontweight='bold')

    seq_len = 5
    for i in range(seq_len):
        # Draw position
        ax.add_patch(plt.Rectangle((i*1.5, 0), 1, 1, facecolor='lightblue',
                                   edgecolor='black'))
        ax.annotate(f'x{i+1}', (i*1.5 + 0.5, 0.5), ha='center', va='center', fontsize=10)

        # Draw Q, K, V arrows
        ax.annotate('Q,K,V', (i*1.5 + 0.5, 1.2), ha='center', fontsize=8, color='gray')

    # Draw attention arcs
    for i in range(seq_len):
        for j in range(seq_len):
            if i != j:
                ax.annotate('', xy=(j*1.5 + 0.5, 1.5), xytext=(i*1.5 + 0.5, 1.5),
                           arrowprops=dict(arrowstyle='->', color='blue',
                                         alpha=0.3, connectionstyle='arc3,rad=0.3'))

    ax.text(3.5, 2.5, 'Each position attends to ALL positions', ha='center',
           fontsize=11, style='italic')
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    # Cross-attention
    ax = axes[1]
    ax.set_title('Cross-Attention\n(Q from one seq, K/V from another)', fontweight='bold')

    # Query sequence
    for i in range(3):
        ax.add_patch(plt.Rectangle((i*1.5, 2), 1, 1, facecolor='lightblue',
                                   edgecolor='black'))
        ax.annotate(f'q{i+1}', (i*1.5 + 0.5, 2.5), ha='center', va='center', fontsize=10)
        ax.annotate('Q', (i*1.5 + 0.5, 3.2), ha='center', fontsize=8, color='blue')

    # Context sequence
    for i in range(5):
        ax.add_patch(plt.Rectangle((i*1.2 + 0.3, 0), 0.9, 0.8, facecolor='lightyellow',
                                   edgecolor='black'))
        ax.annotate(f'c{i+1}', (i*1.2 + 0.75, 0.4), ha='center', va='center', fontsize=9)

    ax.annotate('K, V', (3, -0.3), ha='center', fontsize=8, color='orange')

    # Draw attention arcs
    for i in range(3):
        for j in range(5):
            ax.annotate('', xy=(j*1.2 + 0.75, 0.9), xytext=(i*1.5 + 0.5, 1.9),
                       arrowprops=dict(arrowstyle='->', color='green',
                                     alpha=0.3))

    ax.text(3, -0.8, 'Query seq attends to context seq', ha='center',
           fontsize=11, style='italic')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-1.5, 4)
    ax.axis('off')

    plt.suptitle('Self-Attention vs Cross-Attention', fontsize=14, fontweight='bold')
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
    WHAT BREAKS ATTENTION?

    This experiment explores edge cases and failure modes.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Attention Failure Modes")
    print("=" * 60)

    results = {}

    # 1. NO SCALING
    print("\n1. NO SCALING (Softmax Saturation)")
    print("-" * 40)
    print("What happens without √d_k scaling?")

    for d_k in [8, 64, 256, 1024]:
        Q = np.random.randn(1, 10, d_k)
        K = np.random.randn(1, 10, d_k)

        # Without scaling
        scores_unscaled = Q @ K.transpose(0, 2, 1)
        attn_unscaled = softmax(scores_unscaled, axis=-1)

        # With scaling
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        attn_scaled = softmax(scores_scaled, axis=-1)

        # Measure entropy (uniform = high, peaked = low)
        def entropy(p):
            p = np.clip(p, 1e-10, 1)
            return -np.sum(p * np.log(p), axis=-1).mean()

        ent_unscaled = entropy(attn_unscaled)
        ent_scaled = entropy(attn_scaled)

        print(f"  d_k={d_k:4}: Unscaled entropy={ent_unscaled:.3f}, Scaled={ent_scaled:.3f}")
        results[f'scale_dk_{d_k}'] = (ent_unscaled, ent_scaled)

    print("\n  INSIGHT: Without scaling, attention becomes too peaked (low entropy)")
    print("           Gradients vanish for non-dominant positions!")

    # 2. SEQUENCE TOO LONG
    print("\n2. SEQUENCE TOO LONG (Memory Explosion)")
    print("-" * 40)
    print("Attention is O(n²) - memory grows quadratically")

    for seq_len in [100, 500, 1000, 2000]:
        memory_mb = (seq_len * seq_len * 4) / (1024 * 1024)  # Float32
        print(f"  seq_len={seq_len:4}: Attention matrix = {memory_mb:.2f} MB")
        results[f'memory_{seq_len}'] = memory_mb

    print("\n  INSIGHT: For seq_len=10000, need ~400MB just for attention!")
    print("           Use sparse attention or linear attention for long sequences.")

    # 3. TEMPERATURE EFFECT
    print("\n3. TEMPERATURE EFFECT")
    print("-" * 40)
    print("What if we use temperature scaling?")

    scores = np.array([[1, 2, 3, 4, 5]])

    for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
        attn = softmax(scores / temp, axis=-1)
        max_weight = attn.max()
        print(f"  temp={temp:<4}: Max attention weight = {max_weight:.4f}")
        results[f'temp_{temp}'] = max_weight

    print("\n  INSIGHT: Low temperature → sharp attention (nearly argmax)")
    print("           High temperature → uniform attention")

    return results


def visualize_failure_modes_attention(results: dict, save_path: Optional[str] = None):
    """Visualize attention failure modes."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Scaling effect
    ax = axes[0]
    d_ks = [8, 64, 256, 1024]
    unscaled = [results.get(f'scale_dk_{d}', (0, 0))[0] for d in d_ks]
    scaled = [results.get(f'scale_dk_{d}', (0, 0))[1] for d in d_ks]

    x_pos = np.arange(len(d_ks))
    ax.bar(x_pos - 0.2, unscaled, 0.4, label='Unscaled', color='red', alpha=0.7)
    ax.bar(x_pos + 0.2, scaled, 0.4, label='Scaled', color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(d) for d in d_ks])
    ax.set_xlabel('Key Dimension (d_k)')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Failure: No √d_k Scaling', fontweight='bold')
    ax.legend()

    # 2. Memory explosion
    ax = axes[1]
    seq_lens = [100, 500, 1000, 2000]
    memory = [results.get(f'memory_{s}', 0) for s in seq_lens]
    ax.bar([str(s) for s in seq_lens], memory, color='coral', alpha=0.7)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Failure: O(n²) Memory', fontweight='bold')

    # 3. Summary
    ax = axes[2]
    ax.axis('off')
    summary = """
    ATTENTION FAILURE MODES:

    1. NO SCALING (√d_k)
       • Softmax saturates
       • Gradients vanish
       • Use: scores / √d_k

    2. SEQUENCE TOO LONG
       • O(n²) memory
       • O(n²) compute
       • Use: sparse/linear attention

    3. WRONG TEMPERATURE
       • Too low: argmax (no gradients)
       • Too high: uniform (no focus)
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

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
    print("ATTENTION MECHANISMS — Paradigm: DYNAMIC WEIGHTING")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Instead of fixed operations, let the model LEARN what to focus on.

    THE FORMULA:
        Attention(Q, K, V) = softmax(Q K^T / √d_k) × V

    COMPONENTS:
        Query (Q):  What am I looking for?
        Key (K):    What does each position offer?
        Value (V):  What information to extract?

    KEY VARIANTS:
    ┌─────────────────┬────────────────────────────────────────┐
    │ Variant         │ Description                            │
    ├─────────────────┼────────────────────────────────────────┤
    │ Self-Attention  │ Q, K, V all from same sequence         │
    │ Cross-Attention │ Q from one seq, K/V from another       │
    │ Multi-Head      │ Multiple parallel attention heads      │
    │ Causal/Masked   │ Can only attend to past positions      │
    │ Local/Sparse    │ Only attend to nearby positions        │
    └─────────────────┴────────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # NEW: THE KEY EXPERIMENT - attention learns to copy
    copy_results = experiment_attention_learns_copy()

    # Experiment 1: Scaling effect
    scaling_results = experiment_scaling_effect()

    # Experiment 2: Attention patterns
    pattern_results = experiment_attention_patterns()

    # Experiment 3: Multi-head diversity
    multihead_results = experiment_multi_head_diversity()

    # NEW: Failure modes
    failure_results = experiment_failure_modes()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # NEW: Learned attention visualization (THE KEY FIGURE)
    visualize_attention_learns(copy_results, '47_attention_learned.png')

    visualize_attention_mechanism('47_attention_mechanism.png')
    visualize_attention_patterns(pattern_results, '47_attention_patterns.png')
    visualize_scaling_importance(scaling_results, '47_attention_scaling.png')
    visualize_multi_head_attention(multihead_results, '47_attention_multihead.png')
    visualize_self_vs_cross_attention('47_attention_self_cross.png')
    visualize_failure_modes_attention(failure_results, '47_attention_failures.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. ATTENTION = Dynamic, data-dependent weighting
       - Not all inputs are equally important
       - Learn what to focus on

    2. QUERY-KEY-VALUE framework:
       - Score = how well query matches each key
       - Output = weighted sum of values

    3. SCALING by √d_k is essential:
       - Without it: Attention becomes too peaked
       - Gradients vanish for non-dominant positions

    4. MULTI-HEAD: Multiple "experts" attending differently
       - Different heads capture different patterns
       - More heads = more diversity (usually)

    5. VARIANTS for different needs:
       - Self-attention: Within-sequence dependencies
       - Cross-attention: Between-sequence dependencies
       - Causal: Autoregressive generation
       - Local: Long sequence efficiency
    """)
