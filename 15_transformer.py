"""
TRANSFORMER — Paradigm: LEARNED FEATURES (Attention is All You Need)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of processing sequences step-by-step (RNN), process ALL positions
in PARALLEL using ATTENTION: each position looks at all other positions
to decide what's relevant.

Self-attention computes:
    Attention(Q, K, V) = softmax(QKᵀ/√d_k) × V

This is a SOFT LOOKUP TABLE:
    - Query Q asks: "What am I looking for?"
    - Keys K answer: "What do I contain?"
    - Values V say: "Here's my information"

===============================================================
THE KEY INSIGHT: ATTENTION = SOFT DICTIONARY LOOKUP
===============================================================

For each position, attention computes:

    attention_weights = softmax(q · K.T / √d)

This gives a probability distribution over all positions.
Then we take a weighted average of values:

    output = attention_weights @ V

High dot product (q · k) = high attention = "this is relevant to me"

===============================================================
WHY TRANSFORMERS BEAT RNNs
===============================================================

1. PARALLELIZATION: All positions processed simultaneously
   (RNNs must go step-by-step)

2. DIRECT CONNECTIONS: Any position can attend to any other
   (RNNs: gradients must flow through all intermediate steps)

3. FLEXIBLE CONTEXT: Attention weights are LEARNED per input
   (RNNs: fixed recurrence pattern)

===============================================================
POSITIONAL ENCODING: TELLING TRANSFORMERS ABOUT ORDER
===============================================================

Self-attention is PERMUTATION EQUIVARIANT:
    shuffling input → shuffled output

But order matters in language/sequences! Solution: ADD position info.

Sinusoidal encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Each position gets a unique pattern. Different frequencies capture
different scales of position information.

===============================================================
MULTI-HEAD ATTENTION
===============================================================

Instead of one attention, use H parallel attention "heads":

    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) @ W_O

Each head can focus on DIFFERENT aspects:
    - Head 1 might attend to syntax
    - Head 2 might attend to semantics
    - Head 3 might attend to nearby positions

This is like having multiple "perspectives" on the same input.

===============================================================
ARCHITECTURE COMPONENTS
===============================================================

1. Multi-head self-attention
2. Position-wise feedforward network (MLP at each position)
3. Layer normalization
4. Residual connections

Stack these blocks = deep transformer.

===============================================================
INDUCTIVE BIAS
===============================================================

1. Permutation equivariant (without positional encoding)
2. All-to-all connectivity (any position can attend to any)
3. Content-based addressing (attention by similarity)
4. Positional encoding injects sequence order bias

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
accuracy = datasets_module.accuracy


def create_transformer_dataset(n_samples=500, seq_len=10, pattern='copy_first'):
    """
    Create sequence-to-class datasets for transformer.

    Patterns:
    - 'copy_first': class = discretized first token value
    - 'copy_last': class = discretized last token value
    - 'max_pos': class = position of maximum value
    - 'sum_sign': class = sign of sum
    - 'pattern_match': class = 1 if sequence contains [+,-,+] pattern
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, seq_len)

    if pattern == 'copy_first':
        # 2-class: first element positive or negative
        y = (X[:, 0] > 0).astype(int)

    elif pattern == 'copy_last':
        y = (X[:, -1] > 0).astype(int)

    elif pattern == 'max_pos':
        # Which half has the max?
        y = (np.argmax(X, axis=1) >= seq_len // 2).astype(int)

    elif pattern == 'sum_sign':
        y = (X.sum(axis=1) > 0).astype(int)

    elif pattern == 'pattern_match':
        # Check if pattern +, -, + exists anywhere
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            for j in range(seq_len - 2):
                if X[i, j] > 0 and X[i, j+1] < 0 and X[i, j+2] > 0:
                    y[i] = 1
                    break

    # Reshape for transformer: (batch, seq_len, d_model=1)
    X = X[:, :, np.newaxis]

    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


class PositionalEncoding:
    """
    Sinusoidal positional encoding.

    Adds position information to embeddings so the transformer
    knows about sequence order.
    """

    def __init__(self, d_model, max_len=100):
        self.d_model = d_model
        self.max_len = max_len

        # Compute positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = np.cos(position * div_term[:d_model//2])

        self.pe = pe

    def forward(self, X):
        """Add positional encoding to input."""
        seq_len = X.shape[1]
        return X + self.pe[:seq_len]


class ScaledDotProductAttention:
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(QKᵀ/√d_k) × V

    The scaling by √d_k prevents dot products from growing
    too large with high dimensions (which would saturate softmax).
    """

    def __init__(self):
        self.cache = None

    def softmax(self, X):
        """Row-wise softmax."""
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V shapes: (batch, seq_len, d_k)
        Output shape: (batch, seq_len, d_k)
        """
        d_k = K.shape[-1]

        # Compute attention scores: QKᵀ/√d_k
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Softmax to get attention weights
        attn_weights = self.softmax(scores)

        # Weighted sum of values
        output = attn_weights @ V

        self.cache = (Q, K, V, attn_weights)
        return output, attn_weights

    def backward(self, dout):
        """Backward pass through attention."""
        Q, K, V, attn_weights = self.cache
        d_k = K.shape[-1]
        batch_size, seq_len, _ = Q.shape

        # Gradient through value multiplication
        dV = attn_weights.transpose(0, 2, 1) @ dout
        dattn = dout @ V.transpose(0, 2, 1)

        # Gradient through softmax
        # dsoftmax[i] = softmax[i] * (dout[i] - sum(softmax * dout))
        sum_dattn = np.sum(dattn * attn_weights, axis=-1, keepdims=True)
        dscores = attn_weights * (dattn - sum_dattn)

        # Gradient through scaling
        dscores = dscores / np.sqrt(d_k)

        # Gradient through QKᵀ
        dQ = dscores @ K
        dK = dscores.transpose(0, 2, 1) @ Q

        return dQ, dK, dV


class MultiHeadAttention:
    """
    Multi-head attention.

    Allows the model to attend to different aspects of the input
    using parallel attention heads.
    """

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        self.attention = ScaledDotProductAttention()
        self.cache = None

    def forward(self, X, mask=None):
        """
        X shape: (batch, seq_len, d_model)
        Output shape: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape

        # Linear projections
        Q = X @ self.W_q  # (batch, seq, d_model)
        K = X @ self.W_k
        V = X @ self.W_v

        # Reshape for multi-head: (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Flatten batch and heads for attention
        Q_flat = Q.reshape(-1, seq_len, self.d_k)
        K_flat = K.reshape(-1, seq_len, self.d_k)
        V_flat = V.reshape(-1, seq_len, self.d_k)

        # Apply attention
        attn_output, attn_weights = self.attention.forward(Q_flat, K_flat, V_flat, mask)

        # Reshape back: (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.reshape(batch_size, self.n_heads, seq_len, self.d_k)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = attn_output @ self.W_o

        self.cache = (X, Q, K, V, attn_output, attn_weights.reshape(batch_size, self.n_heads, seq_len, seq_len))
        return output, attn_weights.reshape(batch_size, self.n_heads, seq_len, seq_len)

    def backward(self, dout, lr=0.01):
        """Backward pass through multi-head attention."""
        X, Q, K, V, attn_output, attn_weights = self.cache
        batch_size, seq_len, _ = X.shape

        # Gradient through output projection
        dattn_output = dout @ self.W_o.T
        dW_o = attn_output.reshape(-1, self.d_model).T @ dout.reshape(-1, self.d_model)

        # Reshape for heads
        dattn_output = dattn_output.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        dattn_output_flat = dattn_output.reshape(-1, seq_len, self.d_k)

        # Gradient through attention
        dQ_flat, dK_flat, dV_flat = self.attention.backward(dattn_output_flat)

        # Reshape gradients
        dQ = dQ_flat.reshape(batch_size, self.n_heads, seq_len, self.d_k).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dK = dK_flat.reshape(batch_size, self.n_heads, seq_len, self.d_k).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dV = dV_flat.reshape(batch_size, self.n_heads, seq_len, self.d_k).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Gradient through linear projections
        dW_q = X.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        dW_k = X.reshape(-1, self.d_model).T @ dK.reshape(-1, self.d_model)
        dW_v = X.reshape(-1, self.d_model).T @ dV.reshape(-1, self.d_model)

        dX = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T

        # Update weights
        self.W_q -= lr * dW_q
        self.W_k -= lr * dW_k
        self.W_v -= lr * dW_v
        self.W_o -= lr * dW_o

        return dX


class FeedForward:
    """
    Position-wise feedforward network.

    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    Applied to each position independently.
    """

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

        self.cache = None

    def forward(self, X):
        """X shape: (batch, seq_len, d_model)"""
        hidden = X @ self.W1 + self.b1
        hidden_relu = np.maximum(0, hidden)
        output = hidden_relu @ self.W2 + self.b2

        self.cache = (X, hidden, hidden_relu)
        return output

    def backward(self, dout, lr=0.01):
        X, hidden, hidden_relu = self.cache

        # Gradient through W2
        dW2 = hidden_relu.reshape(-1, self.d_ff).T @ dout.reshape(-1, self.d_model)
        db2 = np.sum(dout, axis=(0, 1))

        dhidden_relu = dout @ self.W2.T

        # Gradient through ReLU
        dhidden = dhidden_relu * (hidden > 0)

        # Gradient through W1
        dW1 = X.reshape(-1, self.d_model).T @ dhidden.reshape(-1, self.d_ff)
        db1 = np.sum(dhidden, axis=(0, 1))

        dX = dhidden @ self.W1.T

        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return dX


class LayerNorm:
    """Layer normalization."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.cache = None

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + self.eps)
        output = self.gamma * X_norm + self.beta

        self.cache = (X, X_norm, mean, var)
        return output

    def backward(self, dout, lr=0.01):
        X, X_norm, mean, var = self.cache
        d_model = X.shape[-1]

        dgamma = np.sum(dout * X_norm, axis=(0, 1))
        dbeta = np.sum(dout, axis=(0, 1))

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True) + dvar * np.mean(-2 * (X - mean), axis=-1, keepdims=True)

        dX = dX_norm / np.sqrt(var + self.eps) + dvar * 2 * (X - mean) / d_model + dmean / d_model

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dX


class TransformerBlock:
    """
    Single transformer encoder block.

    X → LayerNorm → MultiHeadAttention → + (residual) → LayerNorm → FFN → + (residual)
    """

    def __init__(self, d_model, n_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

        self.cache = None

    def forward(self, X, mask=None):
        # Self-attention with residual
        X_norm1 = self.norm1.forward(X)
        attn_out, attn_weights = self.attention.forward(X_norm1, mask)
        X = X + attn_out

        # FFN with residual
        X_norm2 = self.norm2.forward(X)
        ffn_out = self.ffn.forward(X_norm2)
        X = X + ffn_out

        self.cache = attn_weights
        return X, attn_weights

    def backward(self, dout, lr=0.01):
        # Backward through FFN residual
        dffn_out = dout
        dX_norm2 = self.ffn.backward(dffn_out, lr)
        dout = dout + self.norm2.backward(dX_norm2, lr)

        # Backward through attention residual
        dattn_out = dout
        dX_norm1 = self.attention.backward(dattn_out, lr)
        dout = dout + self.norm1.backward(dX_norm1, lr)

        return dout


class SimpleTransformer:
    """
    Simple Transformer for sequence classification.

    Architecture:
        Input embedding → Positional encoding → Transformer blocks → Mean pool → Classification head
    """

    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, n_classes, max_len=100):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projection
        self.input_proj = np.random.randn(input_dim, d_model) * np.sqrt(2.0 / input_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # Classification head
        self.classifier = np.random.randn(d_model, n_classes) * np.sqrt(2.0 / d_model)
        self.classifier_bias = np.zeros(n_classes)

        self.cache = None

    def forward(self, X):
        """
        X shape: (batch, seq_len, input_dim)
        Output: (batch, n_classes)
        """
        # Project input to d_model
        X_proj = X @ self.input_proj

        # Add positional encoding
        X_enc = self.pos_enc.forward(X_proj)

        # Pass through transformer blocks
        attn_weights_all = []
        for block in self.blocks:
            X_enc, attn_weights = block.forward(X_enc)
            attn_weights_all.append(attn_weights)

        # Mean pooling over sequence
        X_pooled = np.mean(X_enc, axis=1)

        # Classification
        logits = X_pooled @ self.classifier + self.classifier_bias

        self.cache = (X, X_proj, X_enc, X_pooled, attn_weights_all)
        return logits, attn_weights_all

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def backward(self, y, lr=0.01):
        X, X_proj, X_enc, X_pooled, _ = self.cache
        batch_size = len(y)

        # Output gradient
        logits = X_pooled @ self.classifier + self.classifier_bias
        probs = self.softmax(logits)

        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        # Classifier gradients
        dclassifier = X_pooled.T @ dlogits
        dclassifier_bias = np.sum(dlogits, axis=0)

        dX_pooled = dlogits @ self.classifier.T

        # Gradient through mean pooling
        seq_len = X_enc.shape[1]
        dX_enc = np.broadcast_to(dX_pooled[:, np.newaxis, :] / seq_len, X_enc.shape).copy()

        # Backward through transformer blocks (reverse order)
        for block in reversed(self.blocks):
            dX_enc = block.backward(dX_enc, lr)

        # Gradient through input projection
        dinput_proj = X.reshape(-1, X.shape[-1]).T @ dX_enc.reshape(-1, self.d_model)

        # Update weights
        self.classifier -= lr * dclassifier
        self.classifier_bias -= lr * dclassifier_bias
        self.input_proj -= lr * dinput_proj

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        losses = []

        for epoch in range(epochs):
            logits, _ = self.forward(X)
            probs = self.softmax(logits)

            loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10))
            losses.append(loss)

            self.backward(y, lr)

            if verbose and (epoch + 1) % 20 == 0:
                acc = accuracy(y, np.argmax(logits, axis=1))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.3f}")

        return losses

    def predict(self, X):
        logits, _ = self.forward(X)
        return np.argmax(logits, axis=1)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # -------- Experiment 1: Effect of Number of Heads --------
    print("\n1. EFFECT OF NUMBER OF ATTENTION HEADS")
    print("-" * 40)
    X_train, X_test, y_train, y_test = create_transformer_dataset(n_samples=600, seq_len=10, pattern='sum_sign')

    for n_heads in [1, 2, 4]:
        d_model = 16  # Must be divisible by n_heads
        transformer = SimpleTransformer(
            input_dim=1, d_model=d_model, n_heads=n_heads,
            d_ff=32, n_layers=2, n_classes=2
        )
        transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, transformer.predict(X_test))
        print(f"n_heads={n_heads} accuracy={acc:.3f}")
    print("→ Multiple heads can capture different patterns")

    # -------- Experiment 2: Effect of Number of Layers --------
    print("\n2. EFFECT OF NUMBER OF LAYERS")
    print("-" * 40)
    for n_layers in [1, 2, 3, 4]:
        transformer = SimpleTransformer(
            input_dim=1, d_model=16, n_heads=2,
            d_ff=32, n_layers=n_layers, n_classes=2
        )
        transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, transformer.predict(X_test))
        print(f"n_layers={n_layers} accuracy={acc:.3f}")
    print("→ Deeper = more capacity, but diminishing returns")

    # -------- Experiment 3: With vs Without Positional Encoding --------
    print("\n3. EFFECT OF POSITIONAL ENCODING")
    print("-" * 40)
    # Test on a position-sensitive task
    X_train, X_test, y_train, y_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    # With positional encoding
    transformer_pe = SimpleTransformer(
        input_dim=1, d_model=16, n_heads=2,
        d_ff=32, n_layers=2, n_classes=2
    )
    transformer_pe.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    acc_pe = accuracy(y_test, transformer_pe.predict(X_test))

    # Without positional encoding (zero it out)
    transformer_no_pe = SimpleTransformer(
        input_dim=1, d_model=16, n_heads=2,
        d_ff=32, n_layers=2, n_classes=2
    )
    transformer_no_pe.pos_enc.pe = np.zeros_like(transformer_no_pe.pos_enc.pe)  # Disable PE
    transformer_no_pe.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    acc_no_pe = accuracy(y_test, transformer_no_pe.predict(X_test))

    print(f"With positional encoding:    {acc_pe:.3f}")
    print(f"Without positional encoding: {acc_no_pe:.3f}")
    print("→ PE is CRITICAL for position-dependent tasks!")

    # -------- Experiment 4: Different Tasks --------
    print("\n4. PERFORMANCE ON DIFFERENT TASKS")
    print("-" * 40)
    patterns = ['copy_first', 'copy_last', 'max_pos', 'sum_sign', 'pattern_match']

    for pattern in patterns:
        X_train, X_test, y_train, y_test = create_transformer_dataset(
            n_samples=600, seq_len=10, pattern=pattern)
        transformer = SimpleTransformer(
            input_dim=1, d_model=16, n_heads=2,
            d_ff=32, n_layers=2, n_classes=2
        )
        transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, transformer.predict(X_test))
        print(f"pattern={pattern:<15} accuracy={acc:.3f}")

    # -------- Experiment 5: Model Size --------
    print("\n5. EFFECT OF MODEL DIMENSION (d_model)")
    print("-" * 40)
    for d_model in [8, 16, 32, 64]:
        transformer = SimpleTransformer(
            input_dim=1, d_model=d_model, n_heads=2,
            d_ff=d_model*2, n_layers=2, n_classes=2
        )
        n_params = (d_model + d_model**2 * 4 + d_model * d_model * 2 + d_model * 2) * 2
        transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, transformer.predict(X_test))
        print(f"d_model={d_model:<3} ~params={n_params:<6} accuracy={acc:.3f}")

    # -------- Experiment 6: Scaling Comparison --------
    print("\n6. TRANSFORMER vs LSTM COMPARISON")
    print("-" * 40)
    try:
        rnn_module = import_module('14_rnn_lstm')
        LSTM = rnn_module.LSTM

        # Long sequence task
        X_train, X_test, y_train, y_test = create_transformer_dataset(
            n_samples=600, seq_len=20, pattern='copy_first')

        # Reshape for LSTM
        X_train_lstm = X_train
        X_test_lstm = X_test

        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train_lstm, y_train, epochs=100, lr=0.05, verbose=False)
        lstm_acc = accuracy(y_test, lstm.predict(X_test_lstm))

        transformer = SimpleTransformer(
            input_dim=1, d_model=16, n_heads=2,
            d_ff=32, n_layers=2, n_classes=2
        )
        transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        trans_acc = accuracy(y_test, transformer.predict(X_test))

        print(f"LSTM accuracy:        {lstm_acc:.3f}")
        print(f"Transformer accuracy: {trans_acc:.3f}")
        print("→ Transformer can attend directly to first position!")
    except:
        print("LSTM module not available for comparison")


def visualize_attention():
    """Visualize attention weights."""
    print("\n" + "="*60)
    print("ATTENTION VISUALIZATION")
    print("="*60)

    X_train, X_test, y_train, y_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    transformer = SimpleTransformer(
        input_dim=1, d_model=16, n_heads=4,
        d_ff=32, n_layers=2, n_classes=2
    )
    transformer.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)

    # Get attention weights for a test sample
    sample = X_test[0:1]
    _, attn_weights_all = transformer.forward(sample)

    # Plot attention patterns
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    # Layer 1 heads
    for h in range(4):
        ax = axes[0, h]
        attn = attn_weights_all[0][0, h]  # (seq, seq)
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer 1, Head {h+1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)

    # Layer 2 heads
    for h in range(4):
        ax = axes[1, h]
        attn = attn_weights_all[1][0, h]
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer 2, Head {h+1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Transformer Attention Patterns\n'
                 '(Bright = high attention, task: classify by first element)',
                 fontsize=12)
    plt.tight_layout()
    return fig


def visualize_positional_encoding():
    """
    THE KEY VISUALIZATION: Show why positional encoding is essential.

    Without PE, transformers are PERMUTATION EQUIVARIANT:
    - Shuffling input → shuffled output
    - No sense of ORDER

    PE gives each position a UNIQUE FINGERPRINT using sine waves
    at different frequencies.
    """
    print("\n" + "="*60)
    print("POSITIONAL ENCODING VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    # 1. Show the sinusoidal patterns
    ax1 = fig.add_subplot(2, 2, 1)
    d_model = 64
    max_len = 50
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    im = ax1.imshow(pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Position in Sequence', fontsize=11)
    ax1.set_ylabel('Dimension', fontsize=11)
    ax1.set_title('Positional Encoding Matrix\n(Each column = unique position fingerprint)', fontsize=12)
    plt.colorbar(im, ax=ax1)

    # 2. Show individual dimensions (sine waves at different frequencies)
    ax2 = fig.add_subplot(2, 2, 2)
    positions = np.arange(50)
    for dim in [0, 2, 4, 8, 16, 32]:
        freq = 1.0 / (10000 ** (dim / d_model))
        ax2.plot(positions, np.sin(positions * freq), label=f'dim {dim}', linewidth=2)
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('PE value', fontsize=11)
    ax2.set_title('Different Dimensions = Different Frequencies\n(Low dims = fast oscillation, high dims = slow)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 50)
    ax2.grid(True, alpha=0.3)

    # 3. Compare with vs without PE on position-sensitive task
    ax3 = fig.add_subplot(2, 2, 3)
    np.random.seed(42)

    X_train, X_test, y_train, y_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    # With PE
    trans_pe = SimpleTransformer(input_dim=1, d_model=16, n_heads=2, d_ff=32, n_layers=2, n_classes=2)
    losses_pe = trans_pe.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    acc_pe = accuracy(y_test, trans_pe.predict(X_test))

    # Without PE
    np.random.seed(42)
    trans_no_pe = SimpleTransformer(input_dim=1, d_model=16, n_heads=2, d_ff=32, n_layers=2, n_classes=2)
    trans_no_pe.pos_enc.pe = np.zeros_like(trans_no_pe.pos_enc.pe)
    losses_no_pe = trans_no_pe.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    acc_no_pe = accuracy(y_test, trans_no_pe.predict(X_test))

    ax3.plot(losses_pe, 'b-', linewidth=2, label=f'With PE (acc={acc_pe:.2f})')
    ax3.plot(losses_no_pe, 'r--', linewidth=2, label=f'Without PE (acc={acc_no_pe:.2f})')
    ax3.axhline(y=np.log(2), color='gray', linestyle=':', label='Random guess')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Task: "Classify by FIRST element"\n(Without PE, model cannot identify position!)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Show that each position has unique encoding (via distance)
    ax4 = fig.add_subplot(2, 2, 4)
    # Compute cosine similarity between positions
    pe_norm = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-10)
    similarity = pe_norm @ pe_norm.T

    im = ax4.imshow(similarity[:30, :30], cmap='viridis', aspect='auto')
    ax4.set_xlabel('Position', fontsize=11)
    ax4.set_ylabel('Position', fontsize=11)
    ax4.set_title('Position Similarity Matrix\n(Diagonal bright = each position unique)', fontsize=12)
    plt.colorbar(im, ax=ax4)

    plt.suptitle('POSITIONAL ENCODING: How Transformers Know About Order\n'
                 'Without PE, attention is permutation-equivariant (order-blind)!',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_attention_mechanism():
    """
    Show attention as SOFT DICTIONARY LOOKUP step by step.

    The key insight:
        Q·K = similarity scores (how relevant is each key to my query?)
        softmax(Q·K/√d) = attention weights (normalized)
        weights @ V = weighted average of values

    This is a SOFT lookup: instead of picking one value, we blend all values
    weighted by relevance.
    """
    print("\n" + "="*60)
    print("ATTENTION MECHANISM VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 10))

    # Create a simple example
    np.random.seed(42)
    seq_len = 6
    d_k = 4

    # Create example Q, K, V
    Q = np.random.randn(seq_len, d_k) * 0.5
    K = np.random.randn(seq_len, d_k) * 0.5
    V = np.random.randn(seq_len, d_k) * 0.5

    # Step 1: Compute Q·K^T (raw similarity)
    scores = Q @ K.T

    # Step 2: Scale by √d_k
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attn_weights = softmax(scaled_scores)

    # Step 4: Weighted sum of V
    output = attn_weights @ V

    # Plot the steps
    # Row 1: Q, K, V matrices
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(Q, cmap='RdBu', aspect='auto', vmin=-1.5, vmax=1.5)
    ax1.set_title('Query (Q)\n"What am I looking for?"', fontsize=11)
    ax1.set_xlabel('d_k')
    ax1.set_ylabel('Position')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(K, cmap='RdBu', aspect='auto', vmin=-1.5, vmax=1.5)
    ax2.set_title('Key (K)\n"What do I contain?"', fontsize=11)
    ax2.set_xlabel('d_k')
    ax2.set_ylabel('Position')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(V, cmap='RdBu', aspect='auto', vmin=-1.5, vmax=1.5)
    ax3.set_title('Value (V)\n"Here is my info"', fontsize=11)
    ax3.set_xlabel('d_k')
    ax3.set_ylabel('Position')
    plt.colorbar(im3, ax=ax3)

    # Placeholder for equation
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.axis('off')
    ax4.text(0.5, 0.7, 'ATTENTION FORMULA:', fontsize=12, ha='center', fontweight='bold')
    ax4.text(0.5, 0.5, r'Attention(Q,K,V) = softmax($\frac{QK^T}{\sqrt{d_k}}$) × V',
             fontsize=14, ha='center', family='monospace')
    ax4.text(0.5, 0.25, '= "Look up relevant values\n   weighted by similarity"',
             fontsize=11, ha='center', style='italic')

    # Row 2: Step by step computation
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(scaled_scores, cmap='RdBu', aspect='auto')
    ax5.set_title('Step 1: Q·K^T / √d\n(Similarity scores)', fontsize=11)
    ax5.set_xlabel('Key position')
    ax5.set_ylabel('Query position')
    plt.colorbar(im5, ax=ax5)

    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(attn_weights, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Step 2: Softmax\n(Attention weights, sum=1)', fontsize=11)
    ax6.set_xlabel('Key position')
    ax6.set_ylabel('Query position')
    plt.colorbar(im6, ax=ax6)

    ax7 = fig.add_subplot(2, 4, 7)
    im7 = ax7.imshow(output, cmap='RdBu', aspect='auto', vmin=-1.5, vmax=1.5)
    ax7.set_title('Step 3: Weights × V\n(Weighted average output)', fontsize=11)
    ax7.set_xlabel('d_k')
    ax7.set_ylabel('Position')
    plt.colorbar(im7, ax=ax7)

    # Show one position's attention in detail
    ax8 = fig.add_subplot(2, 4, 8)
    query_pos = 2
    bars = ax8.bar(range(seq_len), attn_weights[query_pos], color='steelblue', alpha=0.8)
    bars[query_pos].set_color('coral')  # Highlight self-attention
    ax8.set_xlabel('Key position', fontsize=11)
    ax8.set_ylabel('Attention weight', fontsize=11)
    ax8.set_title(f'Position {query_pos} attends to...\n(Orange = self-attention)', fontsize=11)
    ax8.set_xticks(range(seq_len))
    ax8.set_ylim(0, 1)

    plt.suptitle('ATTENTION = Soft Dictionary Lookup\n'
                 'Each position queries all others and retrieves a weighted blend of values',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_head_specialization():
    """
    Show that different attention heads learn DIFFERENT patterns.

    Key insight: Multi-head attention allows the model to have
    multiple "perspectives" on the same input.

    Common specializations:
    - Some heads attend to nearby positions
    - Some heads attend to specific positions (first, last)
    - Some heads attend based on content similarity
    """
    print("\n" + "="*60)
    print("MULTI-HEAD SPECIALIZATION VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    np.random.seed(42)

    # Train on different tasks to show head specialization
    tasks = [
        ('copy_first', 'Task: Classify by FIRST element'),
        ('copy_last', 'Task: Classify by LAST element'),
        ('sum_sign', 'Task: Sign of SUM'),
    ]

    for task_idx, (pattern, task_name) in enumerate(tasks):
        X_train, X_test, y_train, y_test = create_transformer_dataset(
            n_samples=600, seq_len=10, pattern=pattern)

        np.random.seed(task_idx * 100)
        transformer = SimpleTransformer(
            input_dim=1, d_model=16, n_heads=4, d_ff=32, n_layers=1, n_classes=2
        )
        transformer.fit(X_train, y_train, epochs=150, lr=0.05, verbose=False)

        # Get attention from multiple test samples and average
        _, attn_all = transformer.forward(X_test[:20])
        avg_attn = np.mean(attn_all[0], axis=0)  # Average over batch

        for h in range(4):
            ax = fig.add_subplot(3, 4, task_idx * 4 + h + 1)
            attn = avg_attn[h]
            im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=0.5)

            if h == 0:
                ax.set_ylabel(f'{task_name}', fontsize=10)
            if task_idx == 0:
                ax.set_title(f'Head {h+1}', fontsize=11)
            if task_idx == 2:
                ax.set_xlabel('Key pos', fontsize=10)

            # Annotate what each head seems to be doing
            # Check if head attends to first position
            first_attn = np.mean(attn[:, 0])
            last_attn = np.mean(attn[:, -1])
            diag_attn = np.mean(np.diag(attn))

            if first_attn > 0.3:
                ax.text(0.5, -0.15, 'Attends to FIRST', transform=ax.transAxes,
                       fontsize=8, ha='center', color='red', fontweight='bold')
            elif last_attn > 0.3:
                ax.text(0.5, -0.15, 'Attends to LAST', transform=ax.transAxes,
                       fontsize=8, ha='center', color='blue', fontweight='bold')
            elif diag_attn > 0.25:
                ax.text(0.5, -0.15, 'Self-attention', transform=ax.transAxes,
                       fontsize=8, ha='center', color='green', fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=fig.axes, shrink=0.6, label='Attention weight')

    plt.suptitle('MULTI-HEAD ATTENTION: Different Heads Learn Different Patterns\n'
                 'Each head can specialize in attending to different aspects of the input',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig


def visualize_transformer_vs_rnn():
    """
    Show the KEY advantage of Transformers over RNNs:
    Direct connections to any position vs sequential processing.

    In RNNs: information must flow through all intermediate steps
    In Transformers: ANY position can directly attend to ANY other position
    """
    print("\n" + "="*60)
    print("TRANSFORMER vs RNN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(14, 10))

    # 1. Path length comparison
    ax1 = fig.add_subplot(2, 2, 1)
    seq_lengths = [5, 10, 20, 30, 50]
    rnn_path = seq_lengths  # RNN: path length = sequence length
    trans_path = [1] * len(seq_lengths)  # Transformer: always 1 (direct)

    x = np.arange(len(seq_lengths))
    width = 0.35
    ax1.bar(x - width/2, rnn_path, width, label='RNN path length', color='coral')
    ax1.bar(x + width/2, trans_path, width, label='Transformer path length', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lengths)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Path Length (to reach first position)', fontsize=11)
    ax1.set_title('Information Path Length\n(RNN grows with seq len, Transformer stays O(1))', fontsize=12)
    ax1.legend()
    ax1.set_yscale('log')

    # 2. Accuracy on long-range dependency
    ax2 = fig.add_subplot(2, 2, 2)

    # Test on copy_first task with increasing sequence length
    rnn_accs = []
    trans_accs = []

    try:
        rnn_module = import_module('14_rnn_lstm')
        LSTM = rnn_module.LSTM

        for seq_len in [5, 10, 20, 30]:
            np.random.seed(42)
            X_train, X_test, y_train, y_test = create_transformer_dataset(
                n_samples=500, seq_len=seq_len, pattern='copy_first')

            # LSTM
            lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
            lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
            rnn_accs.append(accuracy(y_test, lstm.predict(X_test)))

            # Transformer
            np.random.seed(42)
            trans = SimpleTransformer(input_dim=1, d_model=16, n_heads=2, d_ff=32, n_layers=2, n_classes=2)
            trans.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
            trans_accs.append(accuracy(y_test, trans.predict(X_test)))

        ax2.plot([5, 10, 20, 30], rnn_accs, 'o-', color='coral', linewidth=2, markersize=8, label='LSTM')
        ax2.plot([5, 10, 20, 30], trans_accs, 's-', color='steelblue', linewidth=2, markersize=8, label='Transformer')
    except:
        ax2.text(0.5, 0.5, 'LSTM module not available', ha='center', va='center', transform=ax2.transAxes)

    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Random guess')
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Long-Range Dependency Task\n(Classify by FIRST element)', fontsize=12)
    ax2.legend()
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, alpha=0.3)

    # 3. Diagram showing RNN vs Transformer connections
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    # Draw RNN
    positions = np.linspace(0.1, 0.9, 5)
    for i, x in enumerate(positions):
        circle = plt.Circle((x, 0.7), 0.05, color='coral', ec='black')
        ax3.add_patch(circle)
        ax3.text(x, 0.55, f't={i}', ha='center', fontsize=10)
        if i > 0:
            ax3.annotate('', xy=(x-0.05, 0.7), xytext=(positions[i-1]+0.05, 0.7),
                        arrowprops=dict(arrowstyle='->', color='coral', lw=2))
    ax3.text(0.5, 0.85, 'RNN: Sequential connections', ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.45, 'Information from t=0 must pass through t=1, t=2, t=3 to reach t=4',
             ha='center', fontsize=10, style='italic')

    # Draw Transformer
    for i, x in enumerate(positions):
        circle = plt.Circle((x, 0.2), 0.05, color='steelblue', ec='black')
        ax3.add_patch(circle)
        ax3.text(x, 0.05, f't={i}', ha='center', fontsize=10)

    # Draw all-to-all connections for position 4
    for i in range(4):
        ax3.annotate('', xy=(positions[4]-0.02, 0.22), xytext=(positions[i]+0.02, 0.18),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5, alpha=0.6))

    ax3.text(0.5, 0.35, 'Transformer: Direct connections (attention)', ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, -0.05, 'Any position can directly attend to any other position',
             ha='center', fontsize=10, style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.1, 0.95)

    # 4. Training speed comparison (conceptual)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.text(0.5, 0.85, 'PARALLELIZATION ADVANTAGE', ha='center', fontsize=14, fontweight='bold')

    ax4.text(0.1, 0.65, 'RNN:', fontsize=12, fontweight='bold', color='coral')
    ax4.text(0.1, 0.5, '• Must process t=0, then t=1, then t=2...\n'
                       '• Sequential bottleneck\n'
                       '• Cannot parallelize over time', fontsize=11)

    ax4.text(0.1, 0.25, 'Transformer:', fontsize=12, fontweight='bold', color='steelblue')
    ax4.text(0.1, 0.1, '• All positions processed IN PARALLEL\n'
                       '• No sequential dependency\n'
                       '• Massive parallelism on GPUs', fontsize=11)

    plt.suptitle('TRANSFORMER vs RNN: Why Attention is All You Need\n'
                 'Direct connections + parallelization = better long-range learning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("TRANSFORMER — Attention is All You Need")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Process sequences in PARALLEL using ATTENTION.
    Each position can attend to ALL other positions.

THE KEY INSIGHT:
    Attention = soft dictionary lookup
    Q asks "what do I need?"
    K answers "what do I have?"
    V gives "here's my content"

WHY TRANSFORMERS WIN:
    1. Parallel processing (no sequential bottleneck)
    2. Direct connections (no vanishing gradients through time)
    3. Flexible attention (learned per input)

POSITIONAL ENCODING:
    Attention is permutation-equivariant!
    PE adds position information so order matters.

MULTI-HEAD ATTENTION:
    Multiple "perspectives" on the same input.
    Different heads learn different patterns.
    """)

    ablation_experiments()

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. THE KEY VISUALIZATION: Positional Encoding
    print("\n1. Generating positional encoding visualization...")
    fig_pe = visualize_positional_encoding()
    save_path = '/Users/sid47/ML Algorithms/15_transformer_positional.png'
    fig_pe.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close(fig_pe)

    # 2. Attention mechanism step-by-step
    print("\n2. Generating attention mechanism visualization...")
    fig_attn = visualize_attention_mechanism()
    save_path = '/Users/sid47/ML Algorithms/15_transformer_attention.png'
    fig_attn.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close(fig_attn)

    # 3. Multi-head specialization
    print("\n3. Generating multi-head specialization visualization...")
    fig_heads = visualize_head_specialization()
    save_path = '/Users/sid47/ML Algorithms/15_transformer_heads.png'
    fig_heads.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close(fig_heads)

    # 4. Transformer vs RNN comparison
    print("\n4. Generating Transformer vs RNN visualization...")
    fig_vs = visualize_transformer_vs_rnn()
    save_path = '/Users/sid47/ML Algorithms/15_transformer_vs_rnn.png'
    fig_vs.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close(fig_vs)

    # 5. Original attention patterns
    print("\n5. Generating attention patterns visualization...")
    fig = visualize_attention()
    save_path = '/Users/sid47/ML Algorithms/15_transformer.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Attention(Q,K,V) = softmax(QKᵀ/√d) × V  (SOFT dictionary lookup)
2. Multi-head = multiple attention perspectives (different patterns)
3. Positional encoding CRITICAL for position-aware tasks
4. All-to-all connectivity = no vanishing gradient problem
5. Parallel processing = faster training than RNNs
6. Inductive bias: content-based addressing

===============================================================
THE KEY INSIGHTS (see visualizations):
===============================================================

1. 15_transformer_positional.png — POSITIONAL ENCODING:
   - Without PE, transformer is ORDER-BLIND (permutation equivariant)
   - PE gives each position a unique "fingerprint" using sine waves
   - Different frequencies encode different scales of position info

2. 15_transformer_attention.png — ATTENTION = SOFT LOOKUP:
   - Q·K = similarity ("how relevant is this key to my query?")
   - softmax = normalize to weights
   - weights @ V = blend values by relevance

3. 15_transformer_heads.png — MULTI-HEAD SPECIALIZATION:
   - Different heads learn different attention patterns
   - Some attend to first, some to last, some to content

4. 15_transformer_vs_rnn.png — WHY TRANSFORMERS WIN:
   - RNN: information must flow through ALL intermediate steps
   - Transformer: ANY position can directly attend to ANY other
   - Path length O(n) vs O(1) → better long-range dependencies
    """)
