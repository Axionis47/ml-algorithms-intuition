"""
PARAMETER-EFFICIENT FINE-TUNING (PEFT) — Paradigm: INTRINSIC DIMENSIONALITY

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Pre-trained models are EXPENSIVE. Fine-tuning ALL parameters is:
    1. Memory-intensive (gradients for billions of params)
    2. Storage-heavy (one full copy per task)
    3. Prone to catastrophic forgetting

THE KEY INSIGHT: Adaptation lives in a LOW-DIMENSIONAL SUBSPACE!

When you fine-tune a model from θ_pre to θ_task:
    Δθ = θ_task - θ_pre

This update Δθ has LOW INTRINSIC DIMENSIONALITY:
    - Most of the "information" is in a small subspace
    - We can parameterize Δθ with far fewer parameters!

===============================================================
THE MATHEMATICS OF INTRINSIC DIMENSIONALITY
===============================================================

For a weight matrix W ∈ ℝ^(d×k), the full update has d×k parameters.

But if the update has low rank:
    ΔW ≈ B × A  where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

Then we only need r(d+k) parameters instead of d×k.

Example: d=1024, k=1024, r=8
    Full: 1,048,576 parameters
    LoRA: 16,384 parameters (64× reduction!)

WHY DOES THIS WORK?

Aghajanyan et al. (2020) showed:
    "Pre-trained models have low intrinsic dimension"

The manifold of good solutions for downstream tasks is
LOW-DIMENSIONAL relative to the full parameter space.

===============================================================
PEFT METHODS OVERVIEW
===============================================================

1. LoRA (Low-Rank Adaptation)
   W' = W + BA where rank(BA) = r << min(d,k)

2. Adapters
   h' = h + Adapter(h) where Adapter has bottleneck

3. Prefix Tuning
   Prepend learnable key-value pairs to attention

4. BitFit
   Only train bias terms (surprisingly effective!)

===============================================================
INDUCTIVE BIAS
===============================================================

1. LOW-RANK ASSUMPTION: The optimal weight update has low effective rank.
   - If true: LoRA captures it with far fewer params
   - If false: LoRA will underfit, missing high-rank components

2. ADDITIVE UPDATES: ΔW is added to W, not replacing it.
   - Preserves pre-trained knowledge
   - Cannot "unlearn" or radically restructure

3. UNIFORM RANK: Same rank r for all layers/projections.
   - In practice, different layers may need different ranks
   - Q and V projections often matter more than K and O

4. LINEAR SUBSPACE: The adaptation lives in a linear subspace.
   - Adapters add nonlinearity (ReLU) for more expressivity
   - LoRA is purely linear (but can be merged at inference)

5. FROZEN BASE: Pre-trained weights don't move.
   - Good for multi-task (one base, many adapters)
   - Bad if pre-training was suboptimal for the domain

WHEN PEFT FAILS:
- Task is very different from pre-training (high-rank update needed)
- Pre-trained representations are poor for the domain
- Fine-tuning data is abundant (full FT may generalize better)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module

# Import Transformer components
transformer_module = import_module('15_transformer')
SimpleTransformer = transformer_module.SimpleTransformer
create_transformer_dataset = transformer_module.create_transformer_dataset
accuracy = transformer_module.accuracy


# ============================================================
# LORA: LOW-RANK ADAPTATION
# ============================================================

class LoRALinear:
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Instead of: y = Wx + b
    We compute: y = Wx + b + (α/r) × B × A × x

    Where:
        W ∈ ℝ^(out×in) is FROZEN (pre-trained weights)
        B ∈ ℝ^(out×r) is trainable
        A ∈ ℝ^(r×in) is trainable
        α is a scaling factor (often α = r, so α/r = 1)
        r is the rank (typically 1-64)

    INITIALIZATION STRATEGY:
        We initialize both A and B with small random values.

        WHY NOT B=0 (as in the original paper)?
        The original LoRA paper initializes B=0, A~N(0,σ²) so BA=0 at start.
        This ensures the model starts exactly at pre-training.

        However, when B=0, the gradient dA = dAx.T @ x where dAx = d_lora @ B = 0.
        So A gets zero gradient initially — a "cold start" problem.

        In practice, optimizers with momentum (Adam) eventually break out,
        but for vanilla SGD (which we use here), we need both A and B nonzero.

        We use small random init for both, accepting a small perturbation
        from the pre-trained state in exchange for immediate gradient flow.

    MATHEMATICAL PERSPECTIVE:
        The update ΔW = BA lives in a rank-r subspace of ℝ^(out×in).
        We're constraining the optimization to this subspace.
        If the true optimal update has low effective rank, we lose nothing!
    """

    def __init__(self, in_features, out_features, rank=4, alpha=None,
                 pretrained_W=None, pretrained_b=None):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank  # Common choice: α = r

        # Frozen pre-trained weights
        if pretrained_W is not None:
            self.W = pretrained_W.copy()  # Frozen!
        else:
            self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)

        if pretrained_b is not None:
            self.b = pretrained_b.copy()
        else:
            self.b = np.zeros(out_features)

        # LoRA matrices (trainable)
        # Small random init for both to ensure gradient flow
        # Scale chosen so initial BA contribution is small relative to W
        self.A = np.random.randn(rank, in_features) * np.sqrt(1.0 / rank)
        self.B = np.random.randn(out_features, rank) * 0.01  # Small so BA starts small

        # Gradients (will be set in backward)
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

        # Cache for backward
        self.cache = None

    def forward(self, x):
        """
        Forward pass: y = Wx + b + (α/r) × BAx

        x shape: (batch, in_features)
        output shape: (batch, out_features)
        """
        # Pre-trained output (frozen)
        y_pretrained = x @ self.W.T + self.b

        # LoRA output
        # First: Ax (batch, rank)
        Ax = x @ self.A.T
        # Then: BAx (batch, out_features)
        BAx = Ax @ self.B.T

        # Scale by α/r
        scaling = self.alpha / self.rank
        y_lora = scaling * BAx

        self.cache = (x, Ax, scaling)
        return y_pretrained + y_lora

    def backward(self, dout):
        """
        Backward pass: compute gradients for A and B only.
        W is frozen, so we don't compute dW.

        Math:
            y = Wx + (α/r) * B @ A @ x

            dy/dB = (α/r) * dout.T @ (A @ x).T = (α/r) * dout.T @ Ax
            dy/dA = (α/r) * B.T @ dout @ x.T ... but we need per-sample

            Let's derive carefully:
            y_lora = scaling * (x @ A.T) @ B.T
                   = scaling * x @ A.T @ B.T

            dL/d(B.T) = dL/dy_lora @ d(y_lora)/d(B.T)
                      = dout @ d(scaling * Ax @ B.T)/d(B.T)
                      = scaling * Ax.T @ dout  (for dB, need transpose)

            dB = scaling * dout.T @ Ax

            dL/dAx = dout @ B (the gradient w.r.t. Ax)
            dA = scaling * (dout @ B).T @ x = scaling * B.T @ dout.T @ x ...

            Actually simpler:
            d_lora = dout * scaling (the gradient at the LoRA output)
            dBAx = d_lora
            dB = dBAx.T @ Ax  (gradient of BAx w.r.t. B)
            dAx = dBAx @ B    (gradient of BAx w.r.t. Ax)
            dA = dAx.T @ x    (gradient of Ax w.r.t. A)
        """
        x, Ax, scaling = self.cache

        # Gradient w.r.t. LoRA output (scaled)
        d_lora = dout * scaling

        # Gradient w.r.t. B: (batch, out) @ (batch, rank) -> sum over batch
        self.dB = d_lora.T @ Ax

        # Gradient w.r.t. Ax: (batch, out) @ (out, rank) -> (batch, rank)
        dAx = d_lora @ self.B

        # Gradient w.r.t. A: (rank, batch) @ (batch, in) -> (rank, in)
        self.dA = dAx.T @ x

        # Gradient w.r.t. input (for backprop through earlier layers)
        # dx = dout @ W + d_lora @ B @ A
        dx = dout @ self.W + dAx @ self.A

        return dx

    def update(self, lr, batch_size):
        """Update only LoRA parameters with gradient descent."""
        # Normalize gradients by batch size
        grad_A = self.dA / batch_size
        grad_B = self.dB / batch_size

        # Gradient clipping to prevent explosion (higher threshold)
        max_norm = 10.0
        grad_A_norm = np.linalg.norm(grad_A)
        grad_B_norm = np.linalg.norm(grad_B)
        if grad_A_norm > max_norm:
            grad_A = grad_A * max_norm / grad_A_norm
        if grad_B_norm > max_norm:
            grad_B = grad_B * max_norm / grad_B_norm

        self.A -= lr * grad_A
        self.B -= lr * grad_B

        # Reset gradients
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

    def get_merged_weights(self):
        """
        Merge LoRA into base weights for inference.

        W' = W + (α/r) × B @ A

        After merging, we can discard A and B and use W' directly.
        This means NO inference overhead from LoRA!
        """
        scaling = self.alpha / self.rank
        return self.W + scaling * (self.B @ self.A)

    def num_trainable_params(self):
        """Count trainable parameters (just A and B)."""
        return self.A.size + self.B.size

    def num_frozen_params(self):
        """Count frozen parameters (W and b)."""
        return self.W.size + self.b.size


# ============================================================
# LORA TRANSFORMER (Full working implementation)
# ============================================================

class LoRATransformer:
    """
    Transformer with LoRA applied to attention projections.

    Architecture:
        - Input projection: FROZEN
        - Positional encoding: Fixed (not learned)
        - For each transformer block:
            - LayerNorm: FROZEN
            - Attention Q, K, V, O projections: LoRA (trainable A, B; frozen W)
            - FFN: FROZEN
        - Classifier: TRAINABLE (always need to adapt the head)

    This lets us adapt the model to new tasks by only training:
        1. LoRA matrices (A, B) for attention
        2. Classification head
    """

    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, n_classes,
                 rank=4, alpha=None, max_len=100, pretrained_weights=None,
                 apply_lora_to='qkvo'):
        """
        Args:
            rank: LoRA rank (r)
            alpha: LoRA scaling factor (default: rank, so α/r = 1)
            pretrained_weights: dict of pretrained weights to use as frozen base
            apply_lora_to: string specifying which projections get LoRA
                          'qkvo' = all, 'qv' = query and value only, etc.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_k = d_model // n_heads
        self.rank = rank
        self.alpha = alpha if alpha else rank
        self.apply_lora_to = apply_lora_to

        # Input projection (FROZEN)
        if pretrained_weights and 'input_proj' in pretrained_weights:
            self.input_proj = pretrained_weights['input_proj'].copy()
        else:
            self.input_proj = np.random.randn(input_dim, d_model) * np.sqrt(2.0 / input_dim)

        # Positional encoding (fixed sinusoidal)
        self.pos_enc = transformer_module.PositionalEncoding(d_model, max_len)

        # LoRA layers for attention projections
        self.lora_layers = []
        scale = np.sqrt(2.0 / d_model)

        for layer_idx in range(n_layers):
            layer_loras = {}
            for proj_name in ['q', 'k', 'v', 'o']:
                # Get pretrained weight if available
                key = f'block_{layer_idx}_W{proj_name}'
                if pretrained_weights and key in pretrained_weights:
                    W_pre = pretrained_weights[key]
                else:
                    W_pre = np.random.randn(d_model, d_model) * scale

                # Only apply LoRA if this projection is in apply_lora_to
                if proj_name in apply_lora_to:
                    layer_loras[proj_name] = {
                        'lora': LoRALinear(d_model, d_model, rank, alpha, W_pre),
                        'use_lora': True
                    }
                else:
                    layer_loras[proj_name] = {
                        'W': W_pre.copy(),
                        'use_lora': False
                    }
            self.lora_layers.append(layer_loras)

        # FFN layers (FROZEN)
        self.ffn_layers = []
        for layer_idx in range(n_layers):
            if pretrained_weights:
                W1 = pretrained_weights.get(f'block_{layer_idx}_W1',
                     np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)).copy()
                b1 = pretrained_weights.get(f'block_{layer_idx}_b1',
                     np.zeros(d_ff)).copy()
                W2 = pretrained_weights.get(f'block_{layer_idx}_W2',
                     np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)).copy()
                b2 = pretrained_weights.get(f'block_{layer_idx}_b2',
                     np.zeros(d_model)).copy()
            else:
                W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
                b1 = np.zeros(d_ff)
                W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
                b2 = np.zeros(d_model)
            self.ffn_layers.append({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

        # Layer norms (FROZEN - we keep gamma=1, beta=0)
        self.layer_norms = []
        for layer_idx in range(n_layers):
            if pretrained_weights:
                g1 = pretrained_weights.get(f'block_{layer_idx}_gamma1', np.ones(d_model)).copy()
                b1 = pretrained_weights.get(f'block_{layer_idx}_beta1', np.zeros(d_model)).copy()
                g2 = pretrained_weights.get(f'block_{layer_idx}_gamma2', np.ones(d_model)).copy()
                b2 = pretrained_weights.get(f'block_{layer_idx}_beta2', np.zeros(d_model)).copy()
            else:
                g1, b1 = np.ones(d_model), np.zeros(d_model)
                g2, b2 = np.ones(d_model), np.zeros(d_model)
            self.layer_norms.append({
                'gamma1': g1, 'beta1': b1,
                'gamma2': g2, 'beta2': b2
            })

        # Classification head (TRAINABLE - always need to adapt this)
        if pretrained_weights and 'classifier' in pretrained_weights:
            self.classifier = pretrained_weights['classifier'].copy()
            self.classifier_bias = pretrained_weights['classifier_bias'].copy()
        else:
            self.classifier = np.random.randn(d_model, n_classes) * np.sqrt(2.0 / d_model)
            self.classifier_bias = np.zeros(n_classes)

        self.eps = 1e-6

    def layer_norm(self, x, gamma, beta):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return gamma * x_norm + beta, (x, mean, var, x_norm)

    def layer_norm_backward(self, dout, cache, gamma):
        """
        Full layer norm backward pass.

        y = gamma * (x - mean) / sqrt(var + eps) + beta

        This is the correct backward, not the simplified version.
        """
        x, mean, var, x_norm = cache
        N = x.shape[-1]  # feature dimension
        std = np.sqrt(var + self.eps)

        # Gradient w.r.t. normalized x
        dx_norm = dout * gamma

        # Gradient w.r.t. variance
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

        # Gradient w.r.t. input
        dx = dx_norm / std + dvar * 2.0 * (x - mean) / N + dmean / N

        return dx

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def attention_forward(self, X, layer_idx):
        """
        Forward pass for one attention layer with LoRA.

        Returns: output, cache for backward
        """
        batch_size, seq_len, _ = X.shape
        loras = self.lora_layers[layer_idx]

        X_flat = X.reshape(-1, self.d_model)

        # Q, K, V projections (with or without LoRA)
        projections = {}
        for name in ['q', 'k', 'v']:
            if loras[name]['use_lora']:
                projections[name] = loras[name]['lora'].forward(X_flat)
            else:
                projections[name] = X_flat @ loras[name]['W'].T

        Q = projections['q'].reshape(batch_size, seq_len, self.d_model)
        K = projections['k'].reshape(batch_size, seq_len, self.d_model)
        V = projections['v'].reshape(batch_size, seq_len, self.d_model)

        # Multi-head reshape: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        attn_weights = self.softmax(scores)
        attn_out = attn_weights @ V

        # Reshape back: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection (with or without LoRA)
        attn_out_flat = attn_out.reshape(-1, self.d_model)
        if loras['o']['use_lora']:
            output = loras['o']['lora'].forward(attn_out_flat)
        else:
            output = attn_out_flat @ loras['o']['W'].T
        output = output.reshape(batch_size, seq_len, self.d_model)

        cache = {
            'X_flat': X_flat,
            'Q': Q, 'K': K, 'V': V,
            'attn_weights': attn_weights,
            'attn_out': attn_out,
            'attn_out_flat': attn_out_flat
        }

        return output, cache

    def attention_backward(self, dout, layer_idx, cache):
        """
        Backward pass for attention layer.

        Computes gradients for LoRA parameters and returns gradient w.r.t. input.
        """
        batch_size = dout.shape[0]
        seq_len = dout.shape[1]
        loras = self.lora_layers[layer_idx]

        X_flat = cache['X_flat']
        Q, K, V = cache['Q'], cache['K'], cache['V']
        attn_weights = cache['attn_weights']
        attn_out = cache['attn_out']
        attn_out_flat = cache['attn_out_flat']

        # Gradient through output projection
        dout_flat = dout.reshape(-1, self.d_model)
        if loras['o']['use_lora']:
            dattn_out_flat = loras['o']['lora'].backward(dout_flat)
        else:
            dattn_out_flat = dout_flat @ loras['o']['W']
        dattn_out = dattn_out_flat.reshape(batch_size, seq_len, self.d_model)

        # Reshape for multi-head: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        dattn_out = dattn_out.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Gradient through attention: attn_out = attn_weights @ V
        dattn_weights = dattn_out @ V.transpose(0, 1, 3, 2)
        dV = attn_weights.transpose(0, 1, 3, 2) @ dattn_out

        # Gradient through softmax
        # d_softmax[i] = softmax[i] * (d[i] - sum(softmax * d))
        sum_dattn = np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True)
        dscores = attn_weights * (dattn_weights - sum_dattn)
        dscores = dscores / np.sqrt(self.d_k)

        # Gradient through Q @ K.T
        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q

        # Reshape back from heads: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Gradient through Q, K, V projections
        dQ_flat = dQ.reshape(-1, self.d_model)
        dK_flat = dK.reshape(-1, self.d_model)
        dV_flat = dV.reshape(-1, self.d_model)

        dX_flat = np.zeros_like(X_flat)

        for name, dproj_flat in [('q', dQ_flat), ('k', dK_flat), ('v', dV_flat)]:
            if loras[name]['use_lora']:
                dX_flat += loras[name]['lora'].backward(dproj_flat)
            else:
                dX_flat += dproj_flat @ loras[name]['W']

        dX = dX_flat.reshape(batch_size, seq_len, self.d_model)
        return dX

    def forward(self, X):
        """Full forward pass."""
        batch_size, seq_len, _ = X.shape

        # Input projection
        X_enc = X @ self.input_proj

        # Add positional encoding
        X_enc = self.pos_enc.forward(X_enc)

        # Store caches for backward
        self.layer_caches = []

        for layer_idx in range(self.n_layers):
            ln = self.layer_norms[layer_idx]
            ffn = self.ffn_layers[layer_idx]

            # Pre-norm architecture
            X_residual = X_enc

            # Layer norm 1
            X_norm, ln1_cache = self.layer_norm(X_enc, ln['gamma1'], ln['beta1'])

            # Attention
            attn_out, attn_cache = self.attention_forward(X_norm, layer_idx)

            # Residual
            X_enc = X_residual + attn_out

            # Layer norm 2
            X_residual2 = X_enc
            X_norm2, ln2_cache = self.layer_norm(X_enc, ln['gamma2'], ln['beta2'])

            # FFN
            hidden = X_norm2 @ ffn['W1'] + ffn['b1']
            hidden_relu = np.maximum(0, hidden)
            ffn_out = hidden_relu @ ffn['W2'] + ffn['b2']

            # Residual
            X_enc = X_residual2 + ffn_out

            self.layer_caches.append({
                'X_residual': X_residual,
                'ln1_cache': ln1_cache,
                'attn_cache': attn_cache,
                'X_residual2': X_residual2,
                'ln2_cache': ln2_cache,
                'hidden': hidden,
                'hidden_relu': hidden_relu
            })

        # Mean pooling
        X_pooled = np.mean(X_enc, axis=1)

        # Classification
        logits = X_pooled @ self.classifier + self.classifier_bias

        self.final_cache = {
            'X_enc': X_enc,
            'X_pooled': X_pooled
        }

        return logits

    def backward(self, y, lr):
        """
        Full backward pass. Updates LoRA parameters and classifier.
        """
        batch_size = len(y)
        seq_len = self.layer_caches[0]['X_residual'].shape[1]

        # Softmax + cross-entropy gradient
        logits = self.final_cache['X_pooled'] @ self.classifier + self.classifier_bias
        probs = self.softmax(logits)

        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        # Classifier gradients
        dclassifier = self.final_cache['X_pooled'].T @ dlogits
        dclassifier_bias = np.sum(dlogits, axis=0)

        # Update classifier
        self.classifier -= lr * dclassifier
        self.classifier_bias -= lr * dclassifier_bias

        # Gradient through mean pooling
        dX_pooled = dlogits @ self.classifier.T
        dX_enc = np.broadcast_to(dX_pooled[:, np.newaxis, :] / seq_len,
                                  (batch_size, seq_len, self.d_model)).copy()

        # Backward through layers (reverse order)
        for layer_idx in reversed(range(self.n_layers)):
            cache = self.layer_caches[layer_idx]
            ln = self.layer_norms[layer_idx]
            ffn = self.ffn_layers[layer_idx]

            # Gradient through FFN residual
            dffn_out = dX_enc.copy()

            # Gradient through FFN (frozen, just pass through)
            dhidden_relu = dffn_out @ ffn['W2'].T
            dhidden = dhidden_relu * (cache['hidden'] > 0)
            dX_norm2 = dhidden @ ffn['W1'].T

            # Gradient through layer norm 2
            dX_ln2 = self.layer_norm_backward(dX_norm2, cache['ln2_cache'], ln['gamma2'])

            # Add residual gradient
            dX_enc = dX_enc + dX_ln2

            # Gradient through attention residual
            dattn_out = dX_enc.copy()

            # Gradient through attention (updates LoRA gradients internally)
            dX_norm = self.attention_backward(dattn_out, layer_idx, cache['attn_cache'])

            # Gradient through layer norm 1
            dX_ln1 = self.layer_norm_backward(dX_norm, cache['ln1_cache'], ln['gamma1'])

            # Add residual gradient
            dX_enc = dX_enc + dX_ln1

        # Update LoRA parameters
        for layer_loras in self.lora_layers:
            for proj_name in ['q', 'k', 'v', 'o']:
                if layer_loras[proj_name]['use_lora']:
                    layer_loras[proj_name]['lora'].update(lr, batch_size)

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        """Train the model."""
        losses = []

        for epoch in range(epochs):
            # Forward
            logits = self.forward(X)
            probs = self.softmax(logits)

            # Loss
            loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10))
            losses.append(loss)

            # Backward
            self.backward(y, lr)

            if verbose and (epoch + 1) % 20 == 0:
                acc = np.mean(np.argmax(logits, axis=1) == y)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.3f}")

        return losses

    def predict(self, X):
        """Predict class labels."""
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def num_trainable_params(self):
        """Count trainable parameters (LoRA + classifier)."""
        total = 0
        for layer_loras in self.lora_layers:
            for proj_name in ['q', 'k', 'v', 'o']:
                if layer_loras[proj_name]['use_lora']:
                    total += layer_loras[proj_name]['lora'].num_trainable_params()
        total += self.classifier.size + self.classifier_bias.size
        return total

    def num_total_params(self):
        """Count total parameters (frozen + trainable)."""
        total = self.input_proj.size
        for layer_loras in self.lora_layers:
            for proj_name in ['q', 'k', 'v', 'o']:
                if layer_loras[proj_name]['use_lora']:
                    total += layer_loras[proj_name]['lora'].num_frozen_params()
                    total += layer_loras[proj_name]['lora'].num_trainable_params()
                else:
                    total += layer_loras[proj_name]['W'].size
        for ffn in self.ffn_layers:
            total += ffn['W1'].size + ffn['b1'].size + ffn['W2'].size + ffn['b2'].size
        total += self.classifier.size + self.classifier_bias.size
        return total


# ============================================================
# HELPER: Extract pretrained weights from SimpleTransformer
# ============================================================

def extract_pretrained_weights(model):
    """Extract all weights from a SimpleTransformer for use with LoRATransformer."""
    weights = {
        'input_proj': model.input_proj.copy(),
        'classifier': model.classifier.copy(),
        'classifier_bias': model.classifier_bias.copy()
    }
    for i, block in enumerate(model.blocks):
        weights[f'block_{i}_Wq'] = block.attention.W_q.copy()
        weights[f'block_{i}_Wk'] = block.attention.W_k.copy()
        weights[f'block_{i}_Wv'] = block.attention.W_v.copy()
        weights[f'block_{i}_Wo'] = block.attention.W_o.copy()
        weights[f'block_{i}_W1'] = block.ffn.W1.copy()
        weights[f'block_{i}_b1'] = block.ffn.b1.copy()
        weights[f'block_{i}_W2'] = block.ffn.W2.copy()
        weights[f'block_{i}_b2'] = block.ffn.b2.copy()
        weights[f'block_{i}_gamma1'] = block.norm1.gamma.copy()
        weights[f'block_{i}_beta1'] = block.norm1.beta.copy()
        weights[f'block_{i}_gamma2'] = block.norm2.gamma.copy()
        weights[f'block_{i}_beta2'] = block.norm2.beta.copy()
    return weights


def copy_weights_to_model(model, weights):
    """Copy weights dict back into a SimpleTransformer."""
    model.input_proj = weights['input_proj'].copy()
    model.classifier = weights['classifier'].copy()
    model.classifier_bias = weights['classifier_bias'].copy()
    for i, block in enumerate(model.blocks):
        block.attention.W_q = weights[f'block_{i}_Wq'].copy()
        block.attention.W_k = weights[f'block_{i}_Wk'].copy()
        block.attention.W_v = weights[f'block_{i}_Wv'].copy()
        block.attention.W_o = weights[f'block_{i}_Wo'].copy()
        block.ffn.W1 = weights[f'block_{i}_W1'].copy()
        block.ffn.b1 = weights[f'block_{i}_b1'].copy()
        block.ffn.W2 = weights[f'block_{i}_W2'].copy()
        block.ffn.b2 = weights[f'block_{i}_b2'].copy()


# ============================================================
# EXPERIMENTS
# ============================================================

def experiment_intrinsic_dimensionality():
    """
    THE KEY EXPERIMENT: Show that fine-tuning updates have low rank.

    We:
    1. Pre-train on task A
    2. Fine-tune on task B (full fine-tuning)
    3. Compute ΔW = W_B - W_A for ALL weight matrices
    4. Analyze singular values of ΔW

    If intrinsic dimensionality is low, most singular values will be tiny!
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Intrinsic Dimensionality of Fine-Tuning")
    print("="*60)

    np.random.seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2

    # Task A: sum_sign
    X_A_train, X_A_test, y_A_train, y_A_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='sum_sign')

    # Task B: copy_first (different task!)
    X_B_train, X_B_test, y_B_train, y_B_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    # Pre-train on task A
    print("\n1. Pre-training on Task A (sum_sign)...")
    model_A = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    model_A.fit(X_A_train, y_A_train, epochs=100, lr=0.05, verbose=False)
    acc_A = accuracy(y_A_test, model_A.predict(X_A_test))
    print(f"   Task A accuracy: {acc_A:.3f}")

    # Save ALL pre-trained weights
    W_A = extract_pretrained_weights(model_A)

    # Create model B and copy ALL weights from A
    print("\n2. Fine-tuning on Task B (copy_first)...")
    model_B = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    copy_weights_to_model(model_B, W_A)

    model_B.fit(X_B_train, y_B_train, epochs=100, lr=0.05, verbose=False)
    acc_B = accuracy(y_B_test, model_B.predict(X_B_test))
    print(f"   Task B accuracy: {acc_B:.3f}")

    # Compute ΔW and analyze singular values for ALL weight matrices
    print("\n3. Analyzing weight updates ΔW = W_B - W_A...")

    all_singular_values = []
    weight_names = []

    # Attention weights
    for i, block in enumerate(model_B.blocks):
        for name, attr in [('Wq', 'W_q'), ('Wk', 'W_k'), ('Wv', 'W_v'), ('Wo', 'W_o')]:
            W_pre = W_A[f'block_{i}_{name}']
            W_post = getattr(block.attention, attr)
            delta_W = W_post - W_pre

            U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
            all_singular_values.append(S)
            weight_names.append(f'L{i}_{name}')

    # FFN weights
    for i, block in enumerate(model_B.blocks):
        for name, attr in [('W1', 'W1'), ('W2', 'W2')]:
            W_pre = W_A[f'block_{i}_{name}']
            W_post = getattr(block.ffn, attr)
            delta_W = W_post - W_pre

            U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
            all_singular_values.append(S)
            weight_names.append(f'L{i}_{name}')

    return all_singular_values, weight_names, acc_A, acc_B


def experiment_lora_rank_ablation():
    """
    Ablation: How does LoRA rank affect performance?

    Hypothesis: Very low rank (r=1,2,4) should capture most of the benefit!
    Higher rank should be equal or better (more expressive).
    """
    print("\n" + "="*60)
    print("EXPERIMENT: LoRA Rank Ablation")
    print("="*60)

    np.random.seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2

    # Task A: Pre-training
    X_A_train, X_A_test, y_A_train, y_A_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='sum_sign')

    # Task B: Fine-tuning target
    X_B_train, X_B_test, y_B_train, y_B_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    # Pre-train
    print("\nPre-training base model on Task A...")
    np.random.seed(42)
    base_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    base_model.fit(X_A_train, y_A_train, epochs=100, lr=0.05, verbose=False)
    acc_A = accuracy(y_A_test, base_model.predict(X_A_test))
    print(f"   Task A accuracy: {acc_A:.3f}")

    pretrained_weights = extract_pretrained_weights(base_model)

    # Test different LoRA ranks
    ranks = [1, 2, 4, 8, 16]
    results = {}

    for rank in ranks:
        print(f"\nLoRA rank r={rank}:")

        # Fresh random seed for LoRA init, but same pretrained base
        np.random.seed(123)  # Different seed for LoRA init
        lora_model = LoRATransformer(
            input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, n_classes=2, rank=rank,
            pretrained_weights=pretrained_weights,
            apply_lora_to='qk'  # Best config from where_to_apply experiment
        )

        # Train with higher learning rate for LoRA (LoRA gradients tend to be smaller)
        lora_model.fit(X_B_train, y_B_train, epochs=300, lr=0.5, verbose=False)
        acc = accuracy(y_B_test, lora_model.predict(X_B_test))

        n_trainable = lora_model.num_trainable_params()
        n_total = lora_model.num_total_params()
        param_ratio = 100.0 * n_trainable / n_total

        results[rank] = {
            'accuracy': acc,
            'trainable_params': n_trainable,
            'total_params': n_total,
            'param_ratio': param_ratio
        }

        print(f"   Task B accuracy: {acc:.3f}")
        print(f"   Trainable params: {n_trainable} ({param_ratio:.1f}% of total)")

    # Full fine-tuning baseline
    print(f"\nFull fine-tuning baseline:")
    np.random.seed(42)
    full_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    copy_weights_to_model(full_model, pretrained_weights)
    full_model.fit(X_B_train, y_B_train, epochs=100, lr=0.05, verbose=False)
    acc_full = accuracy(y_B_test, full_model.predict(X_B_test))
    print(f"   Task B accuracy: {acc_full:.3f}")

    results['full'] = {'accuracy': acc_full, 'param_ratio': 100.0}

    return results


def experiment_where_to_apply_lora():
    """
    Ablation: Which projections benefit most from LoRA?

    The original LoRA paper found that Q and V matter most.
    Let's verify this with our implementation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Where to Apply LoRA")
    print("="*60)

    np.random.seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2
    rank = 4

    # Pre-train
    X_A_train, X_A_test, y_A_train, y_A_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='sum_sign')
    X_B_train, X_B_test, y_B_train, y_B_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    np.random.seed(42)
    base_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    base_model.fit(X_A_train, y_A_train, epochs=100, lr=0.05, verbose=False)
    pretrained_weights = extract_pretrained_weights(base_model)

    # Different configurations
    configs = ['q', 'k', 'v', 'o', 'qv', 'qk', 'kv', 'qkv', 'qkvo']
    results = {}

    for config in configs:
        print(f"\nLoRA applied to: {config.upper()}")

        np.random.seed(123)
        lora_model = LoRATransformer(
            input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, n_classes=2, rank=rank,
            pretrained_weights=pretrained_weights,
            apply_lora_to=config  # THIS IS THE KEY - different configs!
        )

        lora_model.fit(X_B_train, y_B_train, epochs=300, lr=0.5, verbose=False)
        acc = accuracy(y_B_test, lora_model.predict(X_B_test))

        n_trainable = lora_model.num_trainable_params()
        results[config] = {
            'accuracy': acc,
            'trainable_params': n_trainable
        }
        print(f"   Accuracy: {acc:.3f}, Params: {n_trainable}")

    return results


def experiment_peft_comparison():
    """
    Compare: No fine-tuning vs LoRA vs Full fine-tuning

    This shows the value of PEFT - getting most of the benefit
    with a fraction of the parameters.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: PEFT Method Comparison")
    print("="*60)

    np.random.seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2

    # Tasks
    X_A_train, X_A_test, y_A_train, y_A_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='sum_sign')
    X_B_train, X_B_test, y_B_train, y_B_test = create_transformer_dataset(
        n_samples=600, seq_len=10, pattern='copy_first')

    # Pre-train
    print("\nPre-training base model...")
    np.random.seed(42)
    base_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    base_model.fit(X_A_train, y_A_train, epochs=100, lr=0.05, verbose=False)
    pretrained_weights = extract_pretrained_weights(base_model)

    methods = {}

    # 1. No fine-tuning (pretrained only)
    print(f"\nNo Fine-tuning (pretrained only):")
    np.random.seed(42)
    no_ft_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    copy_weights_to_model(no_ft_model, pretrained_weights)
    acc_no_ft = accuracy(y_B_test, no_ft_model.predict(X_B_test))
    methods['No Fine-tuning'] = {'accuracy': acc_no_ft, 'params_pct': 0.0}
    print(f"   Accuracy: {acc_no_ft:.3f}")

    # 2. LoRA with different ranks
    for rank in [2, 4, 8]:
        name = f'LoRA (r={rank})'
        print(f"\n{name}:")

        np.random.seed(123)
        lora_model = LoRATransformer(
            input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, n_classes=2, rank=rank,
            pretrained_weights=pretrained_weights,
            apply_lora_to='qk'  # Best config
        )
        lora_model.fit(X_B_train, y_B_train, epochs=300, lr=0.5, verbose=False)
        acc = accuracy(y_B_test, lora_model.predict(X_B_test))

        n_trainable = lora_model.num_trainable_params()
        n_total = lora_model.num_total_params()
        param_pct = 100.0 * n_trainable / n_total

        methods[name] = {'accuracy': acc, 'params_pct': param_pct}
        print(f"   Accuracy: {acc:.3f}")
        print(f"   Trainable params: {param_pct:.1f}%")

    # 3. Full fine-tuning
    print(f"\nFull Fine-tuning:")
    np.random.seed(42)
    full_model = SimpleTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=n_layers, n_classes=2
    )
    copy_weights_to_model(full_model, pretrained_weights)
    full_model.fit(X_B_train, y_B_train, epochs=100, lr=0.05, verbose=False)
    acc_full = accuracy(y_B_test, full_model.predict(X_B_test))
    methods['Full Fine-tuning'] = {'accuracy': acc_full, 'params_pct': 100.0}
    print(f"   Accuracy: {acc_full:.3f}")

    return methods


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_intrinsic_dimensionality(singular_values, weight_names):
    """
    THE KEY VISUALIZATION: Show that ΔW has low effective rank.
    """
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Singular values for each weight matrix
    ax1 = fig.add_subplot(2, 2, 1)

    for sv, name in zip(singular_values, weight_names):
        sv_normalized = sv / (sv.max() + 1e-10)
        ax1.plot(sv_normalized, label=name, alpha=0.7)

    ax1.set_xlabel('Singular Value Index', fontsize=11)
    ax1.set_ylabel('Normalized Singular Value', fontsize=11)
    ax1.set_title('Singular Values of ΔW (Fine-tuning Update)\n'
                  'Fast decay → Low intrinsic rank!', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=7, ncol=3, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative energy
    ax2 = fig.add_subplot(2, 2, 2)

    for sv, name in zip(singular_values, weight_names):
        energy = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-10)
        ax2.plot(energy, label=name, alpha=0.7)

    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% energy')
    ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% energy')

    ax2.set_xlabel('Number of Singular Values', fontsize=11)
    ax2.set_ylabel('Cumulative Energy', fontsize=11)
    ax2.set_title('Cumulative Energy of ΔW\n'
                  'Few singular values capture most variance!', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right', ncol=2)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effective rank distribution
    ax3 = fig.add_subplot(2, 2, 3)

    effective_ranks_90 = []
    effective_ranks_95 = []

    for sv in singular_values:
        energy = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-10)
        effective_ranks_90.append(np.searchsorted(energy, 0.9) + 1)
        effective_ranks_95.append(np.searchsorted(energy, 0.95) + 1)

    x = np.arange(len(weight_names))
    width = 0.35

    ax3.bar(x - width/2, effective_ranks_90, width, label='90% energy', color='red', alpha=0.7)
    ax3.bar(x + width/2, effective_ranks_95, width, label='95% energy', color='orange', alpha=0.7)

    ax3.axhline(y=4, color='blue', linestyle=':', linewidth=2, label='LoRA r=4')
    ax3.axhline(y=8, color='green', linestyle=':', linewidth=2, label='LoRA r=8')

    ax3.set_xticks(x)
    ax3.set_xticklabels(weight_names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Effective Rank', fontsize=11)
    ax3.set_title('Effective Rank by Energy Threshold\n'
                  'Most updates need rank < 8!', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')

    # Plot 4: Explanation
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    avg_rank_90 = np.mean(effective_ranks_90)
    avg_rank_95 = np.mean(effective_ranks_95)

    explanation = f"""
    WHY INTRINSIC DIMENSIONALITY MATTERS
    ════════════════════════════════════════════

    When we fine-tune from Task A to Task B:
        ΔW = W_B - W_A

    EMPIRICAL FINDING (this experiment):
        Average rank for 90% energy: {avg_rank_90:.1f}
        Average rank for 95% energy: {avg_rank_95:.1f}

    This means:
        • The "direction" of adaptation is low-dimensional
        • LoRA with rank 4-8 captures most of the update
        • We can use ~10% of parameters for ~95% performance

    THE KEY INSIGHT:
    ════════════════
    Pre-trained models have learned a good "basin".
    Fine-tuning only requires small, low-rank adjustments.

    This is NOT a coincidence — it's because:
    1. Pre-training learns general representations
    2. Task-specific adaptation is a small perturbation
    3. The perturbation lives in a low-rank subspace
    """

    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('INTRINSIC DIMENSIONALITY: Why PEFT Works\n'
                 'Fine-tuning updates live in a low-dimensional subspace',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_lora_rank_ablation(results):
    """Visualize LoRA rank ablation results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ranks = [r for r in results.keys() if r != 'full']
    accuracies = [results[r]['accuracy'] for r in ranks]
    param_ratios = [results[r]['param_ratio'] for r in ranks]
    full_acc = results['full']['accuracy']

    # Plot 1: Accuracy vs Rank
    ax1 = axes[0]
    ax1.plot(ranks, accuracies, 'bo-', markersize=10, linewidth=2, label='LoRA')
    ax1.axhline(y=full_acc, color='red', linestyle='--', linewidth=2, label=f'Full FT ({full_acc:.3f})')
    ax1.set_xlabel('LoRA Rank (r)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs LoRA Rank', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ranks)

    # Plot 2: Accuracy vs Parameter %
    ax2 = axes[1]
    ax2.plot(param_ratios, accuracies, 'bo-', markersize=10, linewidth=2)
    ax2.scatter([100], [full_acc], color='red', s=150, zorder=5, label=f'Full FT')
    ax2.axhline(y=full_acc, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Trainable Parameters (%)', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy vs Parameters\n(The efficiency curve)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for r, acc, pct in zip(ranks, accuracies, param_ratios):
        ax2.annotate(f'r={r}', (pct, acc), textcoords="offset points",
                    xytext=(5, 5), ha='left', fontsize=9)

    # Plot 3: Relative performance
    ax3 = axes[2]
    relative_perf = [acc / full_acc * 100 for acc in accuracies]

    bars = ax3.bar(range(len(ranks)), relative_perf, color='steelblue', alpha=0.7)
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Full FT (100%)')
    ax3.axhline(y=95, color='orange', linestyle=':', linewidth=2, label='95% threshold')
    ax3.axhline(y=90, color='green', linestyle=':', linewidth=2, label='90% threshold')

    ax3.set_xticks(range(len(ranks)))
    ax3.set_xticklabels([f'r={r}\n({results[r]["param_ratio"]:.1f}%)' for r in ranks])
    ax3.set_ylabel('% of Full FT Performance', fontsize=11)
    ax3.set_title('Relative Performance by Rank', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')

    # Add value labels on bars
    for bar, perf in zip(bars, relative_perf):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{perf:.1f}%', ha='center', fontsize=9)

    plt.suptitle('LoRA RANK ABLATION: How Much Rank Do You Need?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_where_to_apply(results):
    """Visualize which projections benefit most from LoRA."""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in configs]
    params = [results[c]['trainable_params'] for c in configs]

    # Sort by number of projections (complexity)
    order = sorted(range(len(configs)), key=lambda i: len(configs[i]))
    configs = [configs[i] for i in order]
    accuracies = [accuracies[i] for i in order]
    params = [params[i] for i in order]

    x = np.arange(len(configs))
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c.upper()}\n({p} params)' for c, p in zip(configs, params)],
                       fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Which Projections to Apply LoRA To?\n'
                 '(Q and V tend to matter most)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Q, V, QV
    for i, config in enumerate(configs):
        if config in ['q', 'v', 'qv']:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.9)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def visualize_peft_comparison(methods):
    """Compare all PEFT methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(methods.keys())
    accuracies = [methods[n]['accuracy'] for n in names]
    param_pcts = [methods[n]['params_pct'] for n in names]

    # Plot 1: Accuracy bar chart
    ax1 = axes[0]
    colors = []
    for n in names:
        if 'Full' in n:
            colors.append('red')
        elif 'No' in n:
            colors.append('gray')
        else:
            colors.append('steelblue')

    bars = ax1.bar(range(len(names)), accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha='right')
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=10)

    # Plot 2: Efficiency frontier
    ax2 = axes[1]

    for name, pct, acc, color in zip(names, param_pcts, accuracies, colors):
        if pct == 0:
            pct = 0.5  # Small value for log scale
        ax2.scatter([pct], [acc], s=150, c=color, alpha=0.8, label=name)
        ax2.annotate(name.replace(' Fine-tuning', '').replace('LoRA ', 'r='),
                    (pct, acc), textcoords="offset points",
                    xytext=(10, 0), ha='left', fontsize=9)

    ax2.set_xlabel('Trainable Parameters (%)', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Efficiency Frontier\n(Up and left = more efficient)',
                  fontsize=12, fontweight='bold')
    ax2.set_xscale('symlog', linthresh=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 150)

    plt.suptitle('PEFT vs Full Fine-Tuning: The Efficiency Trade-off',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_lora_mathematics():
    """Visualize the mathematics of LoRA."""
    fig = plt.figure(figsize=(16, 10))

    np.random.seed(42)

    d, k = 32, 32
    rank = 4

    W = np.random.randn(d, k) * 0.1
    B = np.random.randn(d, rank) * 0.1
    A = np.random.randn(rank, k) * 0.1
    delta_W = B @ A
    W_merged = W + delta_W

    vmin, vmax = -0.3, 0.3

    # Plot 1: Original W
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(W, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Frozen W\n({d}×{k} = {d*k} params)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('k')
    ax1.set_ylabel('d')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Plot 2: B matrix
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(B, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f'LoRA B (trainable)\n({d}×{rank} = {d*rank} params)', fontsize=11, fontweight='bold')
    ax2.set_xlabel(f'r={rank}')
    ax2.set_ylabel('d')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Plot 3: A matrix
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(A, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_title(f'LoRA A (trainable)\n({rank}×{k} = {rank*k} params)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('k')
    ax3.set_ylabel(f'r={rank}')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Plot 4: ΔW = BA
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(delta_W, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax4.set_title(f'ΔW = B × A\n(rank-{rank} update)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('k')
    ax4.set_ylabel('d')
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Plot 5: Merged W' = W + BA
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(W_merged, cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax5.set_title("W' = W + BA\n(merged for inference)", fontsize=11, fontweight='bold')
    ax5.set_xlabel('k')
    ax5.set_ylabel('d')
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Plot 6: Explanation
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    full_params = d * k
    lora_params = rank * (d + k)
    savings = 100 * (1 - lora_params / full_params)

    explanation = f"""
    LoRA MATHEMATICS
    ════════════════════════════════

    FORWARD PASS:
        y = Wx + (α/r) × BAx

    TRAINING:
        • W is FROZEN (no gradients)
        • Only update B and A
        • Trainable: {lora_params} params
        • Frozen: {full_params} params
        • Savings: {savings:.1f}%

    INFERENCE:
        W' = W + (α/r) × BA
        y = W'x

        Merge once → NO overhead!

    WHY IT WORKS:
        BA has rank ≤ {rank}
        If optimal ΔW has low rank,
        LoRA captures it exactly!
    """

    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.suptitle('LoRA: Low-Rank Adaptation Mathematics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("PARAMETER-EFFICIENT FINE-TUNING (PEFT)")
    print("Paradigm: INTRINSIC DIMENSIONALITY")
    print("="*70)

    print("""
THE CORE IDEA:
    Pre-trained models already know a lot.
    Fine-tuning to a new task only requires SMALL updates.
    These updates live in a LOW-DIMENSIONAL subspace!

THE MATHEMATICS:
    Weight update: ΔW = W_new - W_old
    If ΔW has low rank: ΔW ≈ BA where rank(BA) = r << dim(W)

    LoRA parameterizes this directly:
        W' = W + (α/r) × BA

    Train only A and B, freeze W!
    """)

    # Run experiments
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)

    # Experiment 1: Intrinsic dimensionality
    sv_data, weight_names, acc_A, acc_B = experiment_intrinsic_dimensionality()

    # Experiment 2: LoRA rank ablation
    rank_results = experiment_lora_rank_ablation()

    # Experiment 3: Where to apply LoRA
    where_results = experiment_where_to_apply_lora()

    # Experiment 4: PEFT comparison
    comparison_results = experiment_peft_comparison()

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    print("\n1. Intrinsic dimensionality visualization...")
    fig1 = visualize_intrinsic_dimensionality(sv_data, weight_names)
    fig1.savefig('/Users/sid47/ML Algorithms/25_peft_intrinsic.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("   Saved: 25_peft_intrinsic.png")

    print("\n2. LoRA rank ablation visualization...")
    fig2 = visualize_lora_rank_ablation(rank_results)
    fig2.savefig('/Users/sid47/ML Algorithms/25_peft_lora_rank.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("   Saved: 25_peft_lora_rank.png")

    print("\n3. Where to apply LoRA visualization...")
    fig3 = visualize_where_to_apply(where_results)
    fig3.savefig('/Users/sid47/ML Algorithms/25_peft_where.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("   Saved: 25_peft_where.png")

    print("\n4. PEFT comparison visualization...")
    fig4 = visualize_peft_comparison(comparison_results)
    fig4.savefig('/Users/sid47/ML Algorithms/25_peft_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("   Saved: 25_peft_comparison.png")

    print("\n5. LoRA mathematics visualization...")
    fig5 = visualize_lora_mathematics()
    fig5.savefig('/Users/sid47/ML Algorithms/25_peft_lora_math.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print("   Saved: 25_peft_lora_math.png")

    # Dynamic summary based on actual results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Calculate summary statistics
    full_acc = rank_results['full']['accuracy']
    best_lora_rank = max([r for r in rank_results if r != 'full'],
                         key=lambda r: rank_results[r]['accuracy'])
    best_lora_acc = rank_results[best_lora_rank]['accuracy']
    best_lora_pct = rank_results[best_lora_rank]['param_ratio']

    avg_effective_rank_90 = np.mean([np.searchsorted(np.cumsum(sv**2) / (np.sum(sv**2) + 1e-10), 0.9) + 1
                                     for sv in sv_data])
    avg_effective_rank_95 = np.mean([np.searchsorted(np.cumsum(sv**2) / (np.sum(sv**2) + 1e-10), 0.95) + 1
                                     for sv in sv_data])

    # Find best projection config
    best_config = max(where_results.keys(), key=lambda c: where_results[c]['accuracy'])

    print(f"""
INTRINSIC DIMENSIONALITY (Experiment 1):
    Average effective rank for 90% energy: {avg_effective_rank_90:.1f}
    Average effective rank for 95% energy: {avg_effective_rank_95:.1f}
    → Fine-tuning updates are indeed LOW RANK!

LoRA RANK ABLATION (Experiment 2):
    Full fine-tuning accuracy: {full_acc:.3f}
    Best LoRA (r={best_lora_rank}): {best_lora_acc:.3f} ({best_lora_pct:.1f}% params)
    Relative performance: {100*best_lora_acc/full_acc:.1f}% of full FT

    Rank progression:""")

    for r in sorted([k for k in rank_results if k != 'full']):
        acc = rank_results[r]['accuracy']
        pct = rank_results[r]['param_ratio']
        rel = 100 * acc / full_acc
        print(f"        r={r:2d}: {acc:.3f} ({pct:5.1f}% params, {rel:5.1f}% of full FT)")

    print(f"""
WHERE TO APPLY LoRA (Experiment 3):
    Best configuration: {best_config.upper()} (accuracy: {where_results[best_config]['accuracy']:.3f})

    Key findings:""")
    for config in ['q', 'v', 'qv', 'qkvo']:
        if config in where_results:
            print(f"        {config.upper():5s}: {where_results[config]['accuracy']:.3f}")

    print(f"""
PEFT COMPARISON (Experiment 4):""")
    for name in comparison_results:
        acc = comparison_results[name]['accuracy']
        pct = comparison_results[name]['params_pct']
        print(f"        {name:20s}: {acc:.3f} ({pct:5.1f}% params)")

    # Compute dynamic takeaway for which projections matter
    q_acc = where_results.get('q', {}).get('accuracy', 0)
    k_acc = where_results.get('k', {}).get('accuracy', 0)
    v_acc = where_results.get('v', {}).get('accuracy', 0)
    projection_ranking = sorted([('Q', q_acc), ('K', k_acc), ('V', v_acc)],
                                 key=lambda x: x[1], reverse=True)
    top_projs = f"{projection_ranking[0][0]} ({projection_ranking[0][1]:.3f}) > " \
                f"{projection_ranking[1][0]} ({projection_ranking[1][1]:.3f}) > " \
                f"{projection_ranking[2][0]} ({projection_ranking[2][1]:.3f})"

    print(f"""
===============================================================
THE KEY TAKEAWAYS
===============================================================

1. INTRINSIC DIMENSIONALITY IS LOW
   Fine-tuning updates ΔW have most energy in top few singular values.
   This is WHY low-rank adaptation works!

2. LoRA CAPTURES MOST OF THE BENEFIT
   Best LoRA (r={best_lora_rank}): {100*best_lora_acc/full_acc:.1f}% of full FT with {best_lora_pct:.1f}% params!
   Higher ranks have diminishing returns.

3. WHICH PROJECTIONS MATTER
   Best config: {best_config.upper()} (accuracy: {where_results[best_config]['accuracy']:.3f})
   Single projections: {top_projs}

4. THE EFFICIENCY FRONTIER
   PEFT methods let you trade parameters for performance.
   Choose rank based on your compute/performance needs.

VISUALIZATIONS SAVED:
    • 25_peft_intrinsic.png  — SVD analysis of ΔW
    • 25_peft_lora_rank.png  — Rank ablation
    • 25_peft_where.png      — Which projections matter
    • 25_peft_comparison.png — Full comparison
    • 25_peft_lora_math.png  — LoRA mathematics
""")
