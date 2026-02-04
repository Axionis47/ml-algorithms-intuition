"""
===============================================================
ENCODER-DECODER ARCHITECTURES — Paradigm: COMPRESS & RECONSTRUCT
===============================================================

WHAT IT IS (THE CORE IDEA)
===============================================================

Two networks working together:
    ENCODER: Input → Compressed Representation (Latent)
    DECODER: Latent → Output

Input → [ENCODER] → Bottleneck → [DECODER] → Output

"First UNDERSTAND the input (encode), then GENERATE the output (decode)."

===============================================================
WHY THIS ARCHITECTURE?
===============================================================

1. COMPRESSION: Force the network to learn essential features
2. TRANSFORMATION: Input and output can have different structures
3. MODULARITY: Encoder and decoder can be designed separately
4. TRANSFER: Pre-trained encoder can be reused

===============================================================
KEY VARIANTS
===============================================================

1. AUTOENCODER: Reconstruct input (output = input)
2. VARIATIONAL AE: Latent is a distribution (generative)
3. SEQ2SEQ: Sequence → Sequence (translation, summarization)
4. U-NET: Skip connections between encoder and decoder (segmentation)
5. TRANSFORMER ENC-DEC: Self-attention + Cross-attention

===============================================================
INDUCTIVE BIAS
===============================================================

1. Assumes useful compression exists (information bottleneck)
2. Assumes decoder can reconstruct from compressed form
3. Bottleneck size controls capacity/generalization tradeoff
4. Architecture of encoder/decoder encodes domain knowledge

Author: ML Algorithms Collection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Callable


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)


# ============================================================
# BASIC AUTOENCODER
# ============================================================

class Autoencoder:
    """
    Basic Autoencoder: Learn to compress and reconstruct.

    Architecture:
        Input (d) → Hidden1 → ... → Latent (k) → ... → Output (d)

    WHAT IT LEARNS:
    - Encoder: Maps high-dim input to low-dim latent
    - Decoder: Reconstructs input from latent
    - Latent: Compressed representation capturing essential features

    THE BOTTLENECK INSIGHT:
    If latent_dim < input_dim, the network MUST learn to compress.
    This forces it to discover the underlying structure.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        """
        Args:
            input_dim: Dimension of input
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of latent space (bottleneck)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder: input → hidden → latent
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder_weights = []
        self.encoder_biases = []
        for i in range(len(encoder_dims) - 1):
            W = he_init(encoder_dims[i], encoder_dims[i + 1])
            b = np.zeros((1, encoder_dims[i + 1]))
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)

        # Build decoder: latent → hidden (reversed) → output
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder_weights = []
        self.decoder_biases = []
        for i in range(len(decoder_dims) - 1):
            W = he_init(decoder_dims[i], decoder_dims[i + 1])
            b = np.zeros((1, decoder_dims[i + 1]))
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        h = x
        for W, b in zip(self.encoder_weights[:-1], self.encoder_biases[:-1]):
            h = relu(h @ W + b)
        # Final encoder layer (no activation for latent)
        h = h @ self.encoder_weights[-1] + self.encoder_biases[-1]
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to output."""
        h = z
        for W, b in zip(self.decoder_weights[:-1], self.decoder_biases[:-1]):
            h = relu(h @ W + b)
        # Final decoder layer (sigmoid for bounded output)
        h = sigmoid(h @ self.decoder_weights[-1] + self.decoder_biases[-1])
        return h

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Compute reconstruction loss (MSE)."""
        x_reconstructed, _ = self.forward(x)
        return np.mean((x - x_reconstructed) ** 2)


# ============================================================
# DENOISING AUTOENCODER
# ============================================================

class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Autoencoder: Learn robust representations.

    Training:
        1. Corrupt input: x_noisy = x + noise
        2. Reconstruct original: x_reconstructed = decode(encode(x_noisy))
        3. Loss = ||x - x_reconstructed||²

    WHY IT WORKS:
    - Learns to extract essential features that survive noise
    - Forces latent to capture structure, not noise
    - More robust representations
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 noise_std: float = 0.1):
        super().__init__(input_dim, hidden_dims, latent_dim)
        self.noise_std = noise_std

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to input."""
        noise = np.random.randn(*x.shape) * self.noise_std
        return x + noise

    def forward_with_noise(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with noisy input."""
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


# ============================================================
# CONVOLUTIONAL ENCODER-DECODER (for images)
# ============================================================

class ConvEncoder:
    """
    Convolutional Encoder for image data.

    Uses strided convolutions for downsampling.

    Architecture:
        Image (H×W×C) → Conv → Pool → Conv → Pool → Flatten → Latent
    """

    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int):
        """
        Args:
            input_shape: (height, width, channels)
            latent_dim: Dimension of latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        H, W, C = input_shape

        # Simple conv layers (for demonstration)
        self.conv1_filters = np.random.randn(3, 3, C, 16) * 0.1
        self.conv2_filters = np.random.randn(3, 3, 16, 32) * 0.1

        # Calculate flattened size after convolutions
        # Assuming stride=2 pooling twice: H/4 × W/4 × 32
        self.flat_dim = (H // 4) * (W // 4) * 32
        self.fc_weight = he_init(self.flat_dim, latent_dim)
        self.fc_bias = np.zeros((1, latent_dim))

    def conv2d(self, x: np.ndarray, filters: np.ndarray, stride: int = 1) -> np.ndarray:
        """Simple 2D convolution (for demonstration)."""
        batch, H, W, C_in = x.shape
        kH, kW, _, C_out = filters.shape
        out_H = (H - kH) // stride + 1
        out_W = (W - kW) // stride + 1

        output = np.zeros((batch, out_H, out_W, C_out))

        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * stride, j * stride
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :]
                for c in range(C_out):
                    output[:, i, j, c] = np.sum(patch * filters[:, :, :, c], axis=(1, 2, 3))

        return output

    def max_pool(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """Max pooling."""
        batch, H, W, C = x.shape
        out_H, out_W = H // pool_size, W // pool_size

        output = np.zeros((batch, out_H, out_W, C))
        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * pool_size, j * pool_size
                patch = x[:, h_start:h_start+pool_size, w_start:w_start+pool_size, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encode image to latent vector."""
        # Conv block 1
        h = relu(self.conv2d(x, self.conv1_filters))
        h = self.max_pool(h)

        # Conv block 2
        h = relu(self.conv2d(h, self.conv2_filters))
        h = self.max_pool(h)

        # Flatten and project to latent
        h_flat = h.reshape(h.shape[0], -1)
        z = h_flat @ self.fc_weight + self.fc_bias

        return z


# ============================================================
# U-NET ARCHITECTURE
# ============================================================

class UNetBlock:
    """
    U-Net style encoder-decoder with skip connections.

    KEY INSIGHT:
    Skip connections preserve spatial information lost during downsampling.

    Encoder: Downsample, extract features
    Decoder: Upsample, combine with skip connections
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Simplified linear transformation (would be conv in real U-Net)
        self.down_weight = he_init(in_channels, out_channels)
        self.up_weight = he_init(out_channels * 2, in_channels)  # *2 for skip connection

    def down(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample and return skip connection."""
        # Downsample (simplified: just linear + pooling simulation)
        h = relu(x @ self.down_weight)
        # Simulate 2x downsampling
        if len(x.shape) > 2:
            h_down = h[:, ::2]  # Subsample
        else:
            h_down = h
        return h_down, x  # Return downsampled and skip

    def up(self, x: np.ndarray, skip: np.ndarray) -> np.ndarray:
        """Upsample and combine with skip connection."""
        # Simulate upsampling
        if len(x.shape) > 2:
            h_up = np.repeat(x, 2, axis=1)[:, :skip.shape[1]]
        else:
            h_up = x

        # Concatenate with skip connection
        h_concat = np.concatenate([h_up, skip], axis=-1)

        # Transform
        out = relu(h_concat @ self.up_weight)
        return out


class SimpleUNet:
    """
    Simplified U-Net for sequence data.

    Architecture:
        Input
          ↓ (encode)
        Level 1 ----skip1---→
          ↓ (encode)          |
        Level 2 ----skip2---→ |
          ↓ (encode)          | |
        Bottleneck            | |
          ↓ (decode + skip2)  | |
        Level 2' ←------------+ |
          ↓ (decode + skip1)    |
        Level 1' ←--------------+
          ↓
        Output

    WHY SKIP CONNECTIONS?
    - Low-level features (edges, textures) are preserved
    - Decoder combines global context with local detail
    - Essential for tasks like segmentation
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: Dimensions at each U-Net level
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_levels = len(hidden_dims)

        # Encoder blocks
        self.encoder_weights = []
        dims = [input_dim] + hidden_dims
        for i in range(self.num_levels):
            W = he_init(dims[i], dims[i + 1])
            self.encoder_weights.append(W)

        # Decoder blocks (with skip connections, so input dim is doubled)
        self.decoder_weights = []
        for i in range(self.num_levels - 1, -1, -1):
            # Input: current level features + skip connection
            in_dim = dims[i + 1] + dims[i] if i < self.num_levels - 1 else dims[i + 1]
            out_dim = dims[i]
            W = he_init(in_dim + dims[i], out_dim)  # Concatenate with skip
            self.decoder_weights.append(W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through U-Net."""
        # Encoder (save activations for skip connections)
        skips = [x]
        h = x
        for W in self.encoder_weights:
            h = relu(h @ W)
            skips.append(h)

        # Decoder (use skip connections)
        for i, W in enumerate(self.decoder_weights):
            skip_idx = self.num_levels - i - 1
            skip = skips[skip_idx]

            # Concatenate with skip connection
            h = np.concatenate([h, skip], axis=-1)
            h = relu(h @ W)

        return h


# ============================================================
# SEQUENCE-TO-SEQUENCE (Simplified)
# ============================================================

class Seq2SeqEncoder:
    """
    Sequence-to-Sequence Encoder (RNN-based).

    Processes input sequence and produces a context vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple RNN weights
        self.W_ih = he_init(input_dim, hidden_dim)
        self.W_hh = he_init(hidden_dim, hidden_dim)
        self.b_h = np.zeros((1, hidden_dim))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode sequence.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            outputs: (batch, seq_len, hidden_dim) - all hidden states
            context: (batch, hidden_dim) - final hidden state
        """
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.hidden_dim))
        outputs = []

        for t in range(seq_len):
            h = tanh(x[:, t] @ self.W_ih + h @ self.W_hh + self.b_h)
            outputs.append(h)

        outputs = np.stack(outputs, axis=1)
        context = h  # Final hidden state

        return outputs, context


class Seq2SeqDecoder:
    """
    Sequence-to-Sequence Decoder.

    Takes context vector and generates output sequence.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # RNN weights
        self.W_hh = he_init(hidden_dim, hidden_dim)
        self.W_out = he_init(hidden_dim, output_dim)
        self.b_h = np.zeros((1, hidden_dim))
        self.b_out = np.zeros((1, output_dim))

    def forward(self, context: np.ndarray, output_len: int) -> np.ndarray:
        """
        Decode from context.

        Args:
            context: (batch, hidden_dim) - context from encoder
            output_len: Length of output sequence

        Returns:
            outputs: (batch, output_len, output_dim)
        """
        batch_size = context.shape[0]
        h = context
        outputs = []

        for _ in range(output_len):
            h = tanh(h @ self.W_hh + self.b_h)
            out = h @ self.W_out + self.b_out
            outputs.append(out)

        return np.stack(outputs, axis=1)


class Seq2Seq:
    """
    Complete Sequence-to-Sequence model.

    Used for:
    - Machine translation
    - Text summarization
    - Question answering
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.encoder = Seq2SeqEncoder(input_dim, hidden_dim)
        self.decoder = Seq2SeqDecoder(hidden_dim, output_dim)

    def forward(self, x: np.ndarray, output_len: int) -> np.ndarray:
        """Encode input sequence, decode to output sequence."""
        _, context = self.encoder.forward(x)
        output = self.decoder.forward(context, output_len)
        return output


# ============================================================
# TRAINING INFRASTRUCTURE
# ============================================================

def create_synthetic_patterns(n_samples: int = 1000, pattern_dim: int = 32) -> Tuple:
    """
    Create synthetic patterns that have STRUCTURE (not random noise).

    Patterns: Sinusoidal mixtures - these have learnable structure.
    """
    np.random.seed(42)

    X = np.zeros((n_samples, pattern_dim))
    t = np.linspace(0, 2 * np.pi, pattern_dim)

    for i in range(n_samples):
        # Random combination of frequencies
        freq1 = np.random.uniform(0.5, 2)
        freq2 = np.random.uniform(2, 4)
        phase = np.random.uniform(0, 2 * np.pi)
        amp1 = np.random.uniform(0.3, 1)
        amp2 = np.random.uniform(0.1, 0.5)

        X[i] = amp1 * np.sin(freq1 * t + phase) + amp2 * np.sin(freq2 * t)

    # Normalize to [0, 1] for sigmoid output
    X = (X - X.min()) / (X.max() - X.min() + 1e-10)

    split = int(0.8 * n_samples)
    return X[:split], X[split:]


class TrainableAutoencoder:
    """
    Autoencoder with actual training capability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b_enc1 = np.zeros((1, hidden_dim))
        self.W_enc2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2/hidden_dim)
        self.b_enc2 = np.zeros((1, latent_dim))

        # Decoder
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2/latent_dim)
        self.b_dec1 = np.zeros((1, hidden_dim))
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2/hidden_dim)
        self.b_dec2 = np.zeros((1, input_dim))

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = relu(x @ self.W_enc1 + self.b_enc1)
        z = h @ self.W_enc2 + self.b_enc2
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = relu(z @ self.W_dec1 + self.b_dec1)
        out = sigmoid(h @ self.W_dec2 + self.b_dec2)
        return out

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Encoder
        h_enc = x @ self.W_enc1 + self.b_enc1
        h_enc_act = relu(h_enc)
        z = h_enc_act @ self.W_enc2 + self.b_enc2

        # Decoder
        h_dec = z @ self.W_dec1 + self.b_dec1
        h_dec_act = relu(h_dec)
        logits = h_dec_act @ self.W_dec2 + self.b_dec2
        out = sigmoid(logits)

        cache = {
            'x': x, 'h_enc': h_enc, 'h_enc_act': h_enc_act,
            'z': z, 'h_dec': h_dec, 'h_dec_act': h_dec_act,
            'logits': logits
        }
        return out, cache

    def backward_and_update(self, x_recon: np.ndarray, x_target: np.ndarray,
                            cache: dict, lr: float = 0.01):
        """Backprop and update weights."""
        batch_size = x_target.shape[0]

        # MSE gradient through sigmoid
        sig_grad = x_recon * (1 - x_recon)
        dlogits = (x_recon - x_target) * sig_grad * 2 / batch_size

        # Decoder layer 2
        dW_dec2 = cache['h_dec_act'].T @ dlogits
        db_dec2 = np.sum(dlogits, axis=0, keepdims=True)
        dh_dec_act = dlogits @ self.W_dec2.T

        # ReLU
        dh_dec = dh_dec_act * (cache['h_dec'] > 0)

        # Decoder layer 1
        dW_dec1 = cache['z'].T @ dh_dec
        db_dec1 = np.sum(dh_dec, axis=0, keepdims=True)
        dz = dh_dec @ self.W_dec1.T

        # Encoder layer 2
        dW_enc2 = cache['h_enc_act'].T @ dz
        db_enc2 = np.sum(dz, axis=0, keepdims=True)
        dh_enc_act = dz @ self.W_enc2.T

        # ReLU
        dh_enc = dh_enc_act * (cache['h_enc'] > 0)

        # Encoder layer 1
        dW_enc1 = cache['x'].T @ dh_enc
        db_enc1 = np.sum(dh_enc, axis=0, keepdims=True)

        # Update weights
        self.W_dec2 -= lr * dW_dec2
        self.b_dec2 -= lr * db_dec2
        self.W_dec1 -= lr * dW_dec1
        self.b_dec1 -= lr * db_dec1
        self.W_enc2 -= lr * dW_enc2
        self.b_enc2 -= lr * db_enc2
        self.W_enc1 -= lr * dW_enc1
        self.b_enc1 -= lr * db_enc1


def train_autoencoder(model: TrainableAutoencoder, X_train: np.ndarray,
                      epochs: int = 100, lr: float = 0.1,
                      batch_size: int = 64) -> dict:
    """Train autoencoder and track metrics."""
    results = {
        'losses': [],
        'reconstructions': []  # Store reconstructions at key epochs
    }

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X_train[indices[start:end]]

            # Forward
            x_recon, cache = model.forward(batch_X)

            # Loss (MSE)
            loss = np.mean((x_recon - batch_X) ** 2)
            epoch_loss += loss * (end - start)

            # Backward and update
            model.backward_and_update(x_recon, batch_X, cache, lr)

        results['losses'].append(epoch_loss / n_samples)

        # Store reconstructions at key epochs
        if epoch in [0, epochs//4, epochs//2, epochs-1]:
            x_recon, _ = model.forward(X_train[:5])
            results['reconstructions'].append({
                'epoch': epoch,
                'original': X_train[:5].copy(),
                'reconstructed': x_recon.copy()
            })

    return results


# ============================================================
# TRAINING EXPERIMENTS
# ============================================================

def experiment_autoencoder_training(input_dim: int = 32,
                                     latent_dims: List[int] = [2, 4, 8, 16, 24],
                                     epochs: int = 150) -> dict:
    """
    THE KEY EXPERIMENT: Train autoencoders with different bottleneck sizes.

    WHAT TO OBSERVE:
    - Tiny bottleneck (dim=2): High loss, captures only coarse structure
    - Medium bottleneck (dim=8): Good balance of compression and quality
    - Large bottleneck (dim=24): Low loss, but less compression benefit

    Shows the TRAINING DYNAMICS, not just final loss.
    """
    print("=" * 60)
    print("EXPERIMENT: Autoencoder Training vs Bottleneck Size")
    print("=" * 60)
    print("\nTraining autoencoders with different latent dimensions...\n")

    # Create dataset
    X_train, X_test = create_synthetic_patterns(n_samples=800, pattern_dim=input_dim)

    results = {
        'latent_dims': latent_dims,
        'training_curves': {},
        'final_losses': [],
        'test_losses': [],
        'reconstructions': {}
    }

    for latent_dim in latent_dims:
        print(f"Training with latent_dim = {latent_dim}...")

        model = TrainableAutoencoder(input_dim, hidden_dim=64, latent_dim=latent_dim)
        train_results = train_autoencoder(model, X_train, epochs=epochs, lr=0.1)

        results['training_curves'][latent_dim] = train_results['losses']
        results['final_losses'].append(train_results['losses'][-1])

        # Test loss
        x_recon, _ = model.forward(X_test)
        test_loss = np.mean((x_recon - X_test) ** 2)
        results['test_losses'].append(test_loss)

        # Store reconstructions
        results['reconstructions'][latent_dim] = train_results['reconstructions']

        # Store latent representations for 2D visualization
        if latent_dim == 2:
            z = model.encode(X_test[:200])
            results['latent_2d'] = z

        print(f"  Final train loss: {train_results['losses'][-1]:.4f}")
        print(f"  Test loss: {test_loss:.4f}")

    return results


def experiment_training_with_skip(input_dim: int = 32, epochs: int = 100) -> dict:
    """
    Train encoder-decoder with and without skip connections.

    WHAT TO OBSERVE:
    - Without skip: Loses fine details, blurry reconstructions
    - With skip: Preserves sharp features and details
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Training With vs Without Skip Connections")
    print("=" * 60)

    X_train, X_test = create_synthetic_patterns(n_samples=800, pattern_dim=input_dim)

    # Add some sharp features (spikes)
    spike_train = X_train.copy()
    for i in range(len(spike_train)):
        spike_pos = np.random.randint(5, input_dim - 5, 2)
        spike_train[i, spike_pos] = 1.0

    results = {'without_skip': {}, 'with_skip': {}}

    # Without skip connection
    print("\nTraining WITHOUT skip connections...")
    model_no_skip = TrainableAutoencoder(input_dim, hidden_dim=32, latent_dim=8)
    no_skip_results = train_autoencoder(model_no_skip, spike_train, epochs=epochs, lr=0.1)
    results['without_skip']['losses'] = no_skip_results['losses']
    results['without_skip']['reconstructions'] = no_skip_results['reconstructions']

    # Test
    x_recon, _ = model_no_skip.forward(spike_train[:10])
    results['without_skip']['test_recon'] = x_recon
    results['without_skip']['test_orig'] = spike_train[:10]

    # "With skip" - simulated by larger latent dim (simplified)
    # In reality, would implement U-Net style architecture
    print("\nTraining WITH more capacity (simulating skip benefit)...")
    model_with_cap = TrainableAutoencoder(input_dim, hidden_dim=64, latent_dim=16)
    skip_results = train_autoencoder(model_with_cap, spike_train, epochs=epochs, lr=0.1)
    results['with_skip']['losses'] = skip_results['losses']
    results['with_skip']['reconstructions'] = skip_results['reconstructions']

    x_recon, _ = model_with_cap.forward(spike_train[:10])
    results['with_skip']['test_recon'] = x_recon
    results['with_skip']['test_orig'] = spike_train[:10]

    print(f"\nWithout skip - Final loss: {no_skip_results['losses'][-1]:.4f}")
    print(f"With more capacity - Final loss: {skip_results['losses'][-1]:.4f}")

    return results


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def experiment_bottleneck_size(input_dim: int = 64,
                               latent_dims: List[int] = [2, 4, 8, 16, 32, 48],
                               n_samples: int = 500) -> dict:
    """
    Effect of bottleneck size on reconstruction.

    WHAT TO OBSERVE:
    - Too small: Can't capture enough information, high loss
    - Too large: No compression, might overfit
    - Sweet spot: Good compression with acceptable loss
    """
    print("=" * 60)
    print("EXPERIMENT: Bottleneck Size Effect")
    print("=" * 60)

    # Generate structured data (not random noise)
    t = np.linspace(0, 4 * np.pi, input_dim)
    X = np.array([np.sin(t + phase) + 0.1 * np.random.randn(input_dim)
                  for phase in np.random.uniform(0, 2*np.pi, n_samples)])

    results = {'latent_dims': latent_dims, 'losses': []}

    for latent_dim in latent_dims:
        ae = Autoencoder(input_dim, hidden_dims=[32], latent_dim=latent_dim)
        loss = ae.reconstruction_loss(X)
        results['losses'].append(loss)
        print(f"Latent dim = {latent_dim:3d}, Reconstruction loss = {loss:.4f}")

    return results


def experiment_skip_connections_unet(input_dim: int = 64, n_samples: int = 200) -> dict:
    """
    Importance of skip connections in U-Net style architecture.

    WHAT TO OBSERVE:
    - Without skip: Decoder struggles to recover fine details
    - With skip: Fine details are preserved
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Skip Connections in U-Net")
    print("=" * 60)

    # Generate data with both global structure and local details
    X = np.random.randn(n_samples, input_dim)
    # Add local structure (spikes)
    for i in range(n_samples):
        spike_pos = np.random.randint(0, input_dim, 3)
        X[i, spike_pos] += 5

    # Simple encoder-decoder without skip
    class NoSkipNet:
        def __init__(self, input_dim, hidden_dim):
            self.enc_W = he_init(input_dim, hidden_dim)
            self.dec_W = he_init(hidden_dim, input_dim)

        def forward(self, x):
            h = relu(x @ self.enc_W)
            return h @ self.dec_W

    # With skip connection
    class SkipNet:
        def __init__(self, input_dim, hidden_dim):
            self.enc_W = he_init(input_dim, hidden_dim)
            self.dec_W = he_init(hidden_dim + input_dim, input_dim)  # +input_dim for skip

        def forward(self, x):
            h = relu(x @ self.enc_W)
            h_with_skip = np.concatenate([h, x], axis=-1)  # Skip connection
            return h_with_skip @ self.dec_W

    hidden_dim = 16  # Small bottleneck

    no_skip = NoSkipNet(input_dim, hidden_dim)
    with_skip = SkipNet(input_dim, hidden_dim)

    no_skip_out = no_skip.forward(X)
    skip_out = with_skip.forward(X)

    no_skip_loss = np.mean((X - no_skip_out) ** 2)
    skip_loss = np.mean((X - skip_out) ** 2)

    print(f"\nWithout skip connections: MSE = {no_skip_loss:.4f}")
    print(f"With skip connections:    MSE = {skip_loss:.4f}")

    # Check preservation of spikes (local details)
    spike_error_no_skip = np.mean(np.abs(X[X > 3] - no_skip_out[X > 3]))
    spike_error_skip = np.mean(np.abs(X[X > 3] - skip_out[X > 3]))

    print(f"\nSpike preservation (lower = better):")
    print(f"Without skip: {spike_error_no_skip:.4f}")
    print(f"With skip:    {spike_error_skip:.4f}")

    return {
        'no_skip_loss': no_skip_loss,
        'skip_loss': skip_loss,
        'spike_error_no_skip': spike_error_no_skip,
        'spike_error_skip': spike_error_skip
    }


def experiment_denoising(noise_levels: List[float] = [0.0, 0.1, 0.3, 0.5],
                        input_dim: int = 32,
                        latent_dim: int = 8,
                        n_samples: int = 500) -> dict:
    """
    Denoising autoencoder vs regular autoencoder.

    WHAT TO OBSERVE:
    - Regular AE: May learn to copy noise
    - Denoising AE: Learns more robust features
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Denoising Autoencoder")
    print("=" * 60)

    # Generate clean data
    X_clean = np.sin(np.linspace(0, 2*np.pi, input_dim)[np.newaxis, :] *
                     np.arange(1, n_samples + 1)[:, np.newaxis] / 100)

    results = {'noise_levels': noise_levels,
               'regular_ae_loss': [],
               'denoising_ae_loss': []}

    for noise in noise_levels:
        # Test data with noise
        X_test = X_clean + np.random.randn(*X_clean.shape) * noise

        # Regular autoencoder (trained on noisy data)
        ae = Autoencoder(input_dim, [16], latent_dim)
        ae_recon, _ = ae.forward(X_test)
        ae_loss = np.mean((X_clean - ae_recon) ** 2)  # Compare to CLEAN

        # Denoising autoencoder
        dae = DenoisingAutoencoder(input_dim, [16], latent_dim, noise_std=noise)
        dae_recon, _ = dae.forward_with_noise(X_clean)  # Input clean, corrupt internally
        dae_loss = np.mean((X_clean - dae_recon) ** 2)

        results['regular_ae_loss'].append(ae_loss)
        results['denoising_ae_loss'].append(dae_loss)

        print(f"Noise = {noise:.1f}: Regular AE loss = {ae_loss:.4f}, "
              f"Denoising AE loss = {dae_loss:.4f}")

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_autoencoder_training(results: dict, save_path: Optional[str] = None):
    """
    Visualize autoencoder training with different bottleneck sizes.

    THE KEY VISUALIZATION for this file.
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Training curves
    ax1 = fig.add_subplot(2, 2, 1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results['latent_dims'])))

    for idx, latent_dim in enumerate(results['latent_dims']):
        losses = results['training_curves'][latent_dim]
        ax1.semilogy(losses, color=colors[idx], label=f'Latent={latent_dim}', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss (log scale)')
    ax1.set_title('Training Curves: Larger Bottleneck = Lower Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final loss vs bottleneck size
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(results['latent_dims'], results['final_losses'], 'b-o', linewidth=2, markersize=10)
    ax2.fill_between(results['latent_dims'], results['final_losses'],
                     alpha=0.3, color='blue')
    ax2.set_xlabel('Latent Dimension (Bottleneck Size)')
    ax2.set_ylabel('Final Reconstruction Loss')
    ax2.set_title('The Bottleneck Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Mark sweet spot
    sweet_spot_idx = len(results['latent_dims']) // 2
    ax2.axvline(results['latent_dims'][sweet_spot_idx], color='green',
                linestyle='--', label='Sweet spot')
    ax2.legend()

    # 3. Reconstructions comparison
    ax3 = fig.add_subplot(2, 2, 3)

    # Get reconstructions for different latent dims at final epoch
    latent_to_show = [results['latent_dims'][0], results['latent_dims'][len(results['latent_dims'])//2],
                      results['latent_dims'][-1]]

    sample_idx = 0
    original = results['reconstructions'][latent_to_show[0]][-1]['original'][sample_idx]
    ax3.plot(original, 'k-', linewidth=2, label='Original')

    for latent_dim in latent_to_show:
        recon = results['reconstructions'][latent_dim][-1]['reconstructed'][sample_idx]
        ax3.plot(recon, '--', linewidth=1.5, label=f'Latent={latent_dim}', alpha=0.8)

    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Value')
    ax3.set_title('Reconstruction Quality vs Bottleneck Size', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Latent space visualization (if 2D available)
    ax4 = fig.add_subplot(2, 2, 4)

    if 'latent_2d' in results:
        z = results['latent_2d']
        ax4.scatter(z[:, 0], z[:, 1], alpha=0.5, s=20)
        ax4.set_xlabel('Latent Dim 1')
        ax4.set_ylabel('Latent Dim 2')
        ax4.set_title('Learned Latent Space (2D)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
        summary = """
        KEY FINDINGS:

        1. BOTTLENECK SIZE MATTERS
           • Too small: Can't capture all structure
           • Too large: No compression benefit
           • Sweet spot: Good compression + quality

        2. TRAINING DYNAMICS
           • Larger bottleneck = faster convergence
           • Smaller bottleneck = higher final loss

        3. THE INFORMATION BOTTLENECK
           • Forces network to learn ESSENTIAL features
           • Ignores noise and unimportant details
           • This is feature learning!

        CONCLUSION:
        The bottleneck is the key to learning
        useful representations.
        """
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Autoencoder Training: The Bottleneck Experiment',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_reconstruction_evolution(results: dict, save_path: Optional[str] = None):
    """
    Show how reconstructions improve during training.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Pick a middle latent dim
    latent_dim = results['latent_dims'][len(results['latent_dims'])//2]
    recons = results['reconstructions'][latent_dim]

    sample_idx = 0

    for idx, recon_data in enumerate(recons[:4]):
        ax = axes[0, idx]
        epoch = recon_data['epoch']
        original = recon_data['original'][sample_idx]
        reconstructed = recon_data['reconstructed'][sample_idx]

        ax.plot(original, 'b-', linewidth=2, label='Original')
        ax.plot(reconstructed, 'r--', linewidth=2, label='Reconstructed')
        ax.set_title(f'Epoch {epoch}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Show error below
        ax2 = axes[1, idx]
        error = np.abs(original - reconstructed)
        ax2.fill_between(range(len(error)), error, alpha=0.7, color='red')
        ax2.set_ylabel('|Error|')
        ax2.set_xlabel('Dimension')
        mse = np.mean((original - reconstructed) ** 2)
        ax2.set_title(f'MSE = {mse:.4f}')

    plt.suptitle(f'Reconstruction Evolution During Training (Latent Dim = {latent_dim})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_encoder_decoder_concept(save_path: Optional[str] = None):
    """
    Visual explanation of encoder-decoder architecture.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Basic concept
    ax = axes[0]
    ax.set_title('Encoder-Decoder Architecture', fontweight='bold')

    # Draw encoder
    encoder_layers = [1.0, 0.7, 0.4]
    for i, width in enumerate(encoder_layers):
        rect = plt.Rectangle((i * 1.2, (1-width)/2), 0.8, width,
                             facecolor='steelblue', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)

    ax.annotate('ENCODER', (1.2, -0.15), ha='center', fontweight='bold', color='steelblue')

    # Bottleneck
    rect = plt.Rectangle((3.6, 0.35), 0.8, 0.3, facecolor='gold',
                         edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.annotate('Latent\n(bottleneck)', (4, 0.5), ha='center', va='center', fontsize=9)

    # Draw decoder
    decoder_layers = [0.4, 0.7, 1.0]
    for i, width in enumerate(decoder_layers):
        rect = plt.Rectangle((4.8 + i * 1.2, (1-width)/2), 0.8, width,
                             facecolor='coral', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)

    ax.annotate('DECODER', (6, -0.15), ha='center', fontweight='bold', color='coral')

    # Arrows
    ax.annotate('', xy=(3.5, 0.5), xytext=(0.9, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(8.3, 0.5), xytext=(4.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.annotate('Input', (-0.2, 0.5), ha='center', va='center', fontsize=11)
    ax.annotate('Output', (8.7, 0.5), ha='center', va='center', fontsize=11)

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.3, 1.2)
    ax.axis('off')

    # 2. What bottleneck does
    ax = axes[1]
    ax.set_title('The Bottleneck Insight', fontweight='bold')

    ax.text(0.5, 0.85, 'Input: 1000 dimensions', ha='center', fontsize=11,
           transform=ax.transAxes)
    ax.text(0.5, 0.7, '↓', ha='center', fontsize=20, transform=ax.transAxes)
    ax.text(0.5, 0.55, 'Latent: 10 dimensions', ha='center', fontsize=11,
           transform=ax.transAxes, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='gold'))
    ax.text(0.5, 0.4, '↓', ha='center', fontsize=20, transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Output: 1000 dimensions', ha='center', fontsize=11,
           transform=ax.transAxes)

    ax.text(0.5, 0.05, '"Only essential information\ncan fit through the bottleneck"',
           ha='center', fontsize=10, style='italic', transform=ax.transAxes)

    ax.axis('off')

    # 3. Applications
    ax = axes[2]
    ax.set_title('Encoder-Decoder Applications', fontweight='bold')

    applications = [
        ('Autoencoder', 'Image → Compressed → Image'),
        ('VAE', 'Image → Distribution → New Image'),
        ('Seq2Seq', 'English → Context → French'),
        ('U-Net', 'Image → Features → Segmentation'),
    ]

    for i, (name, desc) in enumerate(applications):
        y = 0.85 - i * 0.22
        ax.text(0.05, y, name, fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, y - 0.08, desc, fontsize=9, transform=ax.transAxes, color='gray')

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_bottleneck_effect(results: dict, save_path: Optional[str] = None):
    """
    Visualize how bottleneck size affects reconstruction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs latent dimension
    ax = axes[0]
    ax.plot(results['latent_dims'], results['losses'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Latent Dimension (bottleneck size)')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Bottleneck Size vs Reconstruction Quality', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight regions
    ax.axvspan(0, 8, alpha=0.2, color='red', label='Too small (underfitting)')
    ax.axvspan(32, 50, alpha=0.2, color='orange', label='Too large (no compression)')
    ax.legend(loc='upper right')

    # Example reconstructions
    ax = axes[1]
    ax.set_title('Reconstruction Quality vs Bottleneck Size', fontweight='bold')

    # Generate example
    input_dim = 64
    t = np.linspace(0, 4 * np.pi, input_dim)
    x = np.sin(t) + 0.5 * np.sin(2*t)
    x = x.reshape(1, -1)

    ax.plot(t, x.flatten(), 'k-', label='Original', linewidth=2)

    colors = ['red', 'orange', 'green', 'blue']
    for latent_dim, color in zip([2, 8, 16, 32], colors):
        ae = Autoencoder(input_dim, [32], latent_dim)
        recon, _ = ae.forward(x)
        ax.plot(t, recon.flatten(), '--', color=color, label=f'Latent={latent_dim}',
               alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_unet_architecture(save_path: Optional[str] = None):
    """
    Visualize U-Net architecture with skip connections.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('U-Net Architecture: Skip Connections Preserve Details', fontweight='bold')

    # Draw encoder path (left side, going down)
    encoder_widths = [2, 1.5, 1, 0.7]
    encoder_heights = [1, 1.2, 1.4, 1.6]
    encoder_y = [3, 2, 1, 0]

    for i, (w, h, y) in enumerate(zip(encoder_widths, encoder_heights, encoder_y)):
        rect = plt.Rectangle((1 - w/2, y), w, h * 0.8,
                             facecolor='steelblue', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(f'Enc {i+1}', (1, y + h * 0.4), ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

    # Draw decoder path (right side, going up)
    decoder_widths = encoder_widths[::-1]
    decoder_heights = encoder_heights[::-1]
    decoder_y = encoder_y[::-1]

    for i, (w, h, y) in enumerate(zip(decoder_widths, decoder_heights, decoder_y)):
        rect = plt.Rectangle((5 - w/2, y), w, h * 0.8,
                             facecolor='coral', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(f'Dec {i+1}', (5, y + h * 0.4), ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

    # Draw bottleneck
    rect = plt.Rectangle((2.7, -0.5), 0.6, 1,
                         facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.annotate('Bottleneck', (3, 0), ha='center', va='center', fontsize=9)

    # Draw skip connections
    skip_colors = ['green', 'purple', 'orange']
    for i in range(3):
        y_enc = encoder_y[i] + encoder_heights[i] * 0.4
        y_dec = decoder_y[-(i+1)] + decoder_heights[-(i+1)] * 0.4

        # Horizontal skip connection
        ax.annotate('', xy=(4.2, y_dec), xytext=(1.8, y_enc),
                   arrowprops=dict(arrowstyle='->', color=skip_colors[i],
                                 lw=2, connectionstyle='arc3,rad=0.1'))
        ax.annotate(f'skip {i+1}', (3, y_enc + 0.2), ha='center',
                   fontsize=8, color=skip_colors[i])

    # Draw vertical connections (encoder)
    for i in range(3):
        ax.annotate('', xy=(1, encoder_y[i+1] + 0.8), xytext=(1, encoder_y[i]),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Draw vertical connections (decoder)
    for i in range(3):
        ax.annotate('', xy=(5, decoder_y[i] + 0.8), xytext=(5, decoder_y[i+1]),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Bottom connections
    ax.annotate('', xy=(2.7, 0), xytext=(1, encoder_y[-1]),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(5, decoder_y[0]), xytext=(3.3, 0),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Labels
    ax.annotate('Input', (1, 4.2), ha='center', fontsize=11, fontweight='bold')
    ax.annotate('Output', (5, 4.2), ha='center', fontsize=11, fontweight='bold')

    ax.text(3, -1.2, 'Skip connections: Preserve spatial information\n'
           'for pixel-wise predictions (segmentation, etc.)',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, 4.5)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_seq2seq(save_path: Optional[str] = None):
    """
    Visualize sequence-to-sequence architecture.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Sequence-to-Sequence: Encoder-Decoder for Sequences',
                fontweight='bold')

    # Input sequence (encoder)
    input_words = ['The', 'cat', 'sat', '<EOS>']
    for i, word in enumerate(input_words):
        # RNN cell
        rect = plt.Rectangle((i * 1.5, 2), 1, 1,
                             facecolor='steelblue', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(word, (i * 1.5 + 0.5, 1.5), ha='center', fontsize=10)
        ax.annotate('h', (i * 1.5 + 0.5, 2.5), ha='center', fontsize=9, color='white')

        # Horizontal arrow
        if i < len(input_words) - 1:
            ax.annotate('', xy=((i+1) * 1.5, 2.5), xytext=(i * 1.5 + 1, 2.5),
                       arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('ENCODER', (2.25, 3.5), ha='center', fontsize=12,
               fontweight='bold', color='steelblue')

    # Context vector
    circle = plt.Circle((6, 2.5), 0.4, facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.annotate('c', (6, 2.5), ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('Context', (6, 1.8), ha='center', fontsize=9)

    # Arrow from encoder to context
    ax.annotate('', xy=(5.6, 2.5), xytext=(5, 2.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Output sequence (decoder)
    output_words = ['Le', 'chat', 's\'assit', '<EOS>']
    for i, word in enumerate(output_words):
        x = 7.5 + i * 1.5
        rect = plt.Rectangle((x, 2), 1, 1,
                             facecolor='coral', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(word, (x + 0.5, 1.5), ha='center', fontsize=10)
        ax.annotate('h', (x + 0.5, 2.5), ha='center', fontsize=9, color='white')

        if i < len(output_words) - 1:
            ax.annotate('', xy=(x + 1.5, 2.5), xytext=(x + 1, 2.5),
                       arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('DECODER', (10, 3.5), ha='center', fontsize=12,
               fontweight='bold', color='coral')

    # Arrow from context to decoder
    ax.annotate('', xy=(7.5, 2.5), xytext=(6.4, 2.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Explanation
    ax.text(6.5, 0.5, 'The encoder compresses the input sequence into a single context vector.\n'
           'The decoder generates the output sequence from this context.',
           ha='center', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 14)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# FAILURE MODES
# ============================================================

def experiment_failure_modes() -> dict:
    """
    WHAT BREAKS ENCODER-DECODER?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Encoder-Decoder Failure Modes")
    print("=" * 60)

    results = {}
    np.random.seed(42)

    # 1. BOTTLENECK = INPUT DIM (No compression)
    print("\n1. BOTTLENECK = INPUT DIM (No Compression)")
    print("-" * 40)

    input_dim = 64
    for latent_dim in [4, 16, 32, 64, 128]:
        # Simple autoencoder test
        W_enc = np.random.randn(input_dim, latent_dim) * 0.1
        W_dec = np.random.randn(latent_dim, input_dim) * 0.1

        x = np.random.randn(100, input_dim)
        z = x @ W_enc
        x_recon = z @ W_dec

        recon_loss = np.mean((x - x_recon) ** 2)
        compression = latent_dim / input_dim

        status = "✗ No compress" if compression >= 1 else "✓"
        print(f"  Latent={latent_dim:3}, Compress={compression:.2f}x: Loss={recon_loss:.4f} {status}")
        results[f'bottleneck_{latent_dim}'] = (compression, recon_loss)

    print("\n  INSIGHT: Bottleneck ≥ input → no compression benefit!")

    # 2. BOTTLENECK TOO SMALL
    print("\n2. BOTTLENECK TOO SMALL")
    print("-" * 40)

    for latent_dim in [1, 2, 4, 8, 16]:
        compression = latent_dim / input_dim
        theoretical_loss = 1.0 / (latent_dim + 0.1)  # Simplified estimate

        status = "✗ Too small" if latent_dim < 4 else "✓"
        print(f"  Latent={latent_dim}: {compression:.2%} of info preserved {status}")
        results[f'small_bottleneck_{latent_dim}'] = latent_dim

    print("\n  INSIGHT: Too small bottleneck → information loss!")

    # 3. ENCODER-DECODER MISMATCH
    print("\n3. CAPACITY MISMATCH")
    print("-" * 40)
    print("What if encoder >> decoder or vice versa?")

    for enc_layers, dec_layers in [(5, 1), (1, 5), (3, 3)]:
        status = "✗ Asymmetric" if abs(enc_layers - dec_layers) > 2 else "✓ Balanced"
        print(f"  Encoder={enc_layers} layers, Decoder={dec_layers} layers: {status}")
        results[f'capacity_{enc_layers}_{dec_layers}'] = (enc_layers, dec_layers)

    print("\n  INSIGHT: Balanced capacity usually works best!")

    return results


def visualize_failure_modes(results: dict, save_path: Optional[str] = None):
    """Visualize encoder-decoder failure modes."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Bottleneck size effect
    ax = axes[0]
    latent_dims = [4, 16, 32, 64, 128]
    compressions = [results.get(f'bottleneck_{d}', (0, 0))[0] for d in latent_dims]
    colors = ['green' if c < 1 else 'red' for c in compressions]
    ax.bar([str(d) for d in latent_dims], compressions, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='No compression')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Failure: No Compression', fontweight='bold')
    ax.legend()

    # 2. Information loss
    ax = axes[1]
    small_dims = [1, 2, 4, 8, 16]
    info_preserved = [d / 64 * 100 for d in small_dims]
    colors = ['red' if p < 10 else 'green' for p in info_preserved]
    ax.bar([str(d) for d in small_dims], info_preserved, color=colors, alpha=0.7)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Info Preserved (%)')
    ax.set_title('Failure: Too Much Compression', fontweight='bold')

    # 3. Summary
    ax = axes[2]
    ax.axis('off')
    summary = """
    ENCODER-DECODER FAILURE MODES:

    1. NO COMPRESSION
       • Bottleneck ≥ input dim
       • No feature learning
       • Just memorization

    2. TOO MUCH COMPRESSION
       • Bottleneck too small
       • Information loss
       • Poor reconstruction

    3. CAPACITY MISMATCH
       • Encoder >> Decoder or vice versa
       • Bottleneck becomes bottleneck
       • Balance capacity!
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
    print("ENCODER-DECODER — Paradigm: COMPRESS & RECONSTRUCT")
    print("=" * 70)

    print("""
    CORE INSIGHT:
    Two networks working together:
        ENCODER: Compress input → Latent representation
        DECODER: Reconstruct output from Latent

    THE BOTTLENECK:
    Forces the network to learn ESSENTIAL features.
    Only information that fits through can be preserved.

    KEY VARIANTS:
    ┌─────────────────┬────────────────────────────────────────┐
    │ Variant         │ Use Case                               │
    ├─────────────────┼────────────────────────────────────────┤
    │ Autoencoder     │ Compression, feature learning          │
    │ Denoising AE    │ Robust features, denoising             │
    │ VAE             │ Generative modeling                    │
    │ U-Net           │ Image segmentation (skip connections)  │
    │ Seq2Seq         │ Translation, summarization             │
    │ Transformer E-D │ Modern NLP tasks                       │
    └─────────────────┴────────────────────────────────────────┘
    """)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    # NEW: THE KEY EXPERIMENT - autoencoder training
    ae_training_results = experiment_autoencoder_training()

    # NEW: Skip connection benefit in training
    skip_training_results = experiment_training_with_skip()

    # Experiment 1: Bottleneck size (at initialization)
    bottleneck_results = experiment_bottleneck_size()

    # Experiment 2: Skip connections
    skip_results = experiment_skip_connections_unet()

    # Experiment 3: Denoising
    denoise_results = experiment_denoising()

    # NEW: Failure modes experiment
    failure_results = experiment_failure_modes()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # NEW: Training visualization (THE KEY FIGURE)
    visualize_autoencoder_training(ae_training_results, '48_encoder_decoder_training.png')

    # NEW: Reconstruction evolution
    visualize_reconstruction_evolution(ae_training_results, '48_encoder_decoder_evolution.png')

    visualize_encoder_decoder_concept('48_encoder_decoder_concept.png')
    visualize_bottleneck_effect(bottleneck_results, '48_encoder_decoder_bottleneck.png')
    visualize_unet_architecture('48_encoder_decoder_unet.png')
    visualize_seq2seq('48_encoder_decoder_seq2seq.png')

    # NEW: Failure modes visualization
    visualize_failure_modes(failure_results, '48_encoder_failures.png')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    KEY TAKEAWAYS:

    1. ENCODER-DECODER = Compress then Reconstruct
       - Encoder: Extract essential features
       - Decoder: Generate output from features

    2. BOTTLENECK is the key
       - Too small: Loses information
       - Too large: No compression benefit
       - Right size: Captures structure, ignores noise

    3. SKIP CONNECTIONS (U-Net) preserve details
       - Essential for pixel-wise predictions
       - Combine global context with local detail

    4. APPLICATIONS span many domains:
       - Images: Compression, segmentation, generation
       - Sequences: Translation, summarization
       - General: Feature learning, denoising
    """)
