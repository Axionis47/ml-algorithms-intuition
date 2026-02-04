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

    # Experiment 1: Bottleneck size
    bottleneck_results = experiment_bottleneck_size()

    # Experiment 2: Skip connections
    skip_results = experiment_skip_connections_unet()

    # Experiment 3: Denoising
    denoise_results = experiment_denoising()

    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualize_encoder_decoder_concept('48_encoder_decoder_concept.png')
    visualize_bottleneck_effect(bottleneck_results, '48_encoder_decoder_bottleneck.png')
    visualize_unet_architecture('48_encoder_decoder_unet.png')
    visualize_seq2seq('48_encoder_decoder_seq2seq.png')

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
