"""
CONVOLUTIONAL NEURAL NETWORK — Paradigm: LEARNED FEATURES (Local + Shared)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of fully connected layers where every input connects to every neuron,
use LOCAL CONNECTIONS with SHARED WEIGHTS (convolution).

The SAME small filter slides across the input, detecting the SAME pattern
anywhere in the image → TRANSLATION EQUIVARIANCE.

===============================================================
THE KEY INSIGHT: WEIGHT SHARING = INDUCTIVE BIAS
===============================================================

An MLP treating a 28×28 image has 784 × hidden_size weights in layer 1.
A CNN with 3×3 filters has only 9 × n_filters weights!

This isn't just efficiency. It's a STRUCTURAL PRIOR:
    "Patterns that matter are LOCAL and can appear ANYWHERE."

This is why CNNs dominate vision:
    - Edges are local
    - Textures are local
    - A cat's ear is a cat's ear regardless of position

===============================================================
CONVOLUTION = TEMPLATE MATCHING
===============================================================

Convolution with filter W at position (i,j):

    output[i,j] = Σₘ Σₙ input[i+m, j+n] × W[m,n]

This is a DOT PRODUCT between the filter and the local patch.
High activation = patch LOOKS LIKE the filter.

So convolution is TEMPLATE MATCHING across all positions!

===============================================================
HIERARCHICAL FEATURE LEARNING
===============================================================

Layer 1: Learns edges, gradients (like Gabor filters)
Layer 2: Learns textures, corners (combinations of edges)
Layer 3: Learns parts (combinations of textures)
Layer 4+: Learns objects (combinations of parts)

Each layer's receptive field GROWS:
    3×3 filter → 3×3 receptive field
    Stack two 3×3 → 5×5 effective receptive field
    Stack three 3×3 → 7×7 effective receptive field

===============================================================
POOLING = TRANSLATION INVARIANCE
===============================================================

Max pooling: take max over local region
    - Provides SLIGHT translation invariance
    - Reduces spatial dimensions
    - Keeps strongest activations

Stride: skip positions when convolving
    - Similar effect to pooling
    - Modern architectures often use strided convolutions instead

===============================================================
INDUCTIVE BIAS
===============================================================

1. Locality: patterns are local
2. Translation equivariance: same pattern, same response (anywhere)
3. Hierarchical composition: simple → complex features
4. Spatial structure: 2D grid matters

This is why CNNs fail on permuted images!
If you shuffle pixels, the LOCAL patterns are destroyed.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
accuracy = datasets_module.accuracy


def create_simple_image_dataset(n_samples=500):
    """
    Create a simple 8×8 image dataset for binary classification.

    Class 0: Horizontal stripe pattern
    Class 1: Vertical stripe pattern

    This tests whether CNN can learn orientation-specific filters.
    """
    np.random.seed(42)

    images = []
    labels = []

    for i in range(n_samples):
        img = np.random.randn(8, 8) * 0.1  # noise

        if i % 2 == 0:
            # Horizontal stripes
            img[2:4, :] += 1.0
            img[5:7, :] += 1.0
            labels.append(0)
        else:
            # Vertical stripes
            img[:, 2:4] += 1.0
            img[:, 5:7] += 1.0
            labels.append(1)

        images.append(img)

    X = np.array(images)
    y = np.array(labels)

    # Split
    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


def create_digit_like_dataset(n_samples=1000):
    """
    Create simple 8×8 'digit-like' patterns.

    Class 0: 'L' shape
    Class 1: 'T' shape
    Class 2: '+' shape

    This tests pattern recognition beyond just orientation.
    """
    np.random.seed(42)

    images = []
    labels = []

    for i in range(n_samples):
        img = np.random.randn(8, 8) * 0.1

        label = i % 3

        if label == 0:  # L shape
            img[1:6, 2] += 1.0  # vertical bar
            img[5, 2:6] += 1.0   # horizontal bar
        elif label == 1:  # T shape
            img[1, 2:6] += 1.0   # top horizontal
            img[1:6, 3:5] += 1.0  # vertical
        else:  # + shape
            img[3:5, 1:7] += 1.0  # horizontal
            img[1:7, 3:5] += 1.0  # vertical

        images.append(img)
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


def im2col(X, kernel_size, stride, padding):
    """
    Transform image to column matrix for efficient convolution.

    This converts convolution into matrix multiplication!
    Instead of sliding a filter, we extract all patches as columns.

    X shape: (N, C, H, W)
    Output shape: (N * H_out * W_out, C * kernel_size * kernel_size)
    """
    N, C, H, W = X.shape

    # Apply padding
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (0, 0),
                              (padding, padding),
                              (padding, padding)), mode='constant')
    else:
        X_padded = X

    H_padded, W_padded = X_padded.shape[2], X_padded.shape[3]
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    # Extract patches using stride tricks for efficiency
    shape = (N, C, H_out, W_out, kernel_size, kernel_size)
    strides = (X_padded.strides[0], X_padded.strides[1],
               X_padded.strides[2] * stride, X_padded.strides[3] * stride,
               X_padded.strides[2], X_padded.strides[3])

    patches = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    # Reshape to (N * H_out * W_out, C * K * K)
    cols = patches.reshape(N * H_out * W_out, -1)

    return cols, H_out, W_out, X_padded


def col2im(dcols, X_shape, kernel_size, stride, padding, H_out, W_out):
    """
    Reverse of im2col - accumulate gradients back to image format.
    """
    N, C, H, W = X_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    dX_padded = np.zeros((N, C, H_padded, W_padded))

    # Reshape dcols to patches
    dcols_reshaped = dcols.reshape(N, H_out, W_out, C, kernel_size, kernel_size)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            dX_padded[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size] += \
                dcols_reshaped[:, i, j, :, :, :]

    # Remove padding
    if padding > 0:
        return dX_padded[:, :, padding:-padding, padding:-padding]
    return dX_padded


class Conv2D:
    """
    2D Convolution Layer (VECTORIZED with im2col).

    This is where the magic happens: local receptive fields + weight sharing.
    Uses im2col transformation to convert convolution to matrix multiplication.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        Parameters:
        -----------
        in_channels : Input channels (1 for grayscale, 3 for RGB)
        out_channels : Number of filters (output feature maps)
        kernel_size : Size of each filter (assumes square)
        stride : Step size when sliding filter
        padding : Zero-padding around input
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights: shape (out_channels, in_channels, kernel_size, kernel_size)
        # Kaiming initialization for ReLU
        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)

        # Gradients
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        """
        Forward pass: convolve filters across input using im2col.

        X shape: (batch_size, in_channels, height, width)
        Output shape: (batch_size, out_channels, out_height, out_width)
        """
        N, C, H, W = X.shape

        # im2col transformation
        cols, H_out, W_out, X_padded = im2col(X, self.kernel_size, self.stride, self.padding)

        # Reshape weights to (out_channels, in_channels * K * K)
        W_flat = self.W.reshape(self.out_channels, -1)

        # Convolution as matrix multiplication!
        # cols: (N * H_out * W_out, C * K * K)
        # W_flat.T: (C * K * K, out_channels)
        # Result: (N * H_out * W_out, out_channels)
        out = cols @ W_flat.T + self.b

        # Reshape to (N, out_channels, H_out, W_out)
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

        self.cache = (X, cols, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass: compute gradients using im2col.
        """
        X, cols, H_out, W_out = self.cache
        N, C, H, W = X.shape

        # Reshape dout to (N * H_out * W_out, out_channels)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Gradient w.r.t. weights
        # dW = cols.T @ dout_flat, reshaped
        self.dW = (cols.T @ dout_flat).T.reshape(self.W.shape)

        # Gradient w.r.t. bias
        self.db = np.sum(dout_flat, axis=0)

        # Gradient w.r.t. input
        W_flat = self.W.reshape(self.out_channels, -1)
        dcols = dout_flat @ W_flat  # (N * H_out * W_out, C * K * K)

        # col2im to get dX
        dX = col2im(dcols, X.shape, self.kernel_size, self.stride, self.padding, H_out, W_out)

        return dX


class MaxPool2D:
    """
    Max Pooling Layer (VECTORIZED).

    Takes the maximum value in each local region.
    Provides slight translation invariance and reduces dimensions.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        """
        Forward pass: max over each pool_size × pool_size region.
        Vectorized using reshape and stride tricks.
        """
        N, C, H, W = X.shape
        p = self.pool_size
        s = self.stride

        H_out = (H - p) // s + 1
        W_out = (W - p) // s + 1

        # Use stride tricks to extract pooling regions
        shape = (N, C, H_out, W_out, p, p)
        strides = (X.strides[0], X.strides[1],
                   X.strides[2] * s, X.strides[3] * s,
                   X.strides[2], X.strides[3])

        X_strided = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        # Max over the pooling windows
        out = X_strided.max(axis=(4, 5))

        # Store for backward
        self.cache = (X, X_strided, out)
        return out

    def backward(self, dout):
        """
        Backward pass: gradient flows only to the max element.
        """
        X, X_strided, out = self.cache
        N, C, H, W = X.shape
        p = self.pool_size
        s = self.stride
        _, _, H_out, W_out = dout.shape

        dX = np.zeros_like(X)

        # Create mask where max occurred
        # out has shape (N, C, H_out, W_out)
        # X_strided has shape (N, C, H_out, W_out, p, p)
        mask = (X_strided == out[:, :, :, :, None, None])

        # Distribute gradients
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s
                dX[:, :, h_start:h_start+p, w_start:w_start+p] += \
                    mask[:, :, i, j, :, :] * dout[:, :, i:i+1, j:j+1]

        return dX


class Flatten:
    """Flatten spatial dimensions for dense layer."""

    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)


class ReLU:
    """ReLU activation."""

    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        X = self.cache
        return dout * (X > 0)


class Dense:
    """Fully connected layer."""

    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dout):
        X = self.cache
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T


class SimpleCNN:
    """
    Simple CNN for image classification.

    Architecture:
        Conv(3×3) → ReLU → Pool(2×2) → Conv(3×3) → ReLU → Pool(2×2) → Dense → Output
    """

    def __init__(self, input_shape=(1, 8, 8), n_classes=2, n_filters=8):
        """
        Parameters:
        -----------
        input_shape : (channels, height, width)
        n_classes : Number of output classes
        n_filters : Number of filters in conv layers
        """
        C, H, W = input_shape

        self.layers = []

        # Layer 1: Conv + ReLU
        self.layers.append(Conv2D(C, n_filters, kernel_size=3, padding=1))
        self.layers.append(ReLU())

        # Layer 2: Pool
        self.layers.append(MaxPool2D(pool_size=2, stride=2))

        # Calculate size after pool: H/2, W/2
        H1, W1 = H // 2, W // 2

        # Layer 3: Conv + ReLU
        self.layers.append(Conv2D(n_filters, n_filters * 2, kernel_size=3, padding=1))
        self.layers.append(ReLU())

        # Layer 4: Pool
        self.layers.append(MaxPool2D(pool_size=2, stride=2))

        # Calculate size after second pool: H/4, W/4
        H2, W2 = H1 // 2, W1 // 2
        flat_size = n_filters * 2 * H2 * W2

        # Layer 5: Flatten
        self.layers.append(Flatten())

        # Layer 6: Dense → output
        self.layers.append(Dense(flat_size, n_classes))

        self.n_classes = n_classes

    def forward(self, X):
        """Forward pass through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def softmax(self, X):
        """Numerically stable softmax."""
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def cross_entropy_loss(self, logits, y):
        """Cross-entropy loss with softmax."""
        probs = self.softmax(logits)
        n = len(y)

        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # Loss: -log(prob of correct class)
        loss = -np.sum(np.log(probs[np.arange(n), y])) / n

        # Gradient: softmax - one_hot
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n

        return loss, dlogits

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32, verbose=True):
        """Train the CNN."""
        n_samples = len(y)

        # Reshape X if needed: (N, H, W) → (N, 1, H, W)
        if len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]

        losses = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                logits = self.forward(X_batch)

                # Loss
                loss, dlogits = self.cross_entropy_loss(logits, y_batch)
                epoch_loss += loss * len(y_batch)

                # Backward
                self.backward(dlogits)

                # Update weights (SGD)
                for layer in self.layers:
                    if hasattr(layer, 'W'):
                        layer.W -= lr * layer.dW
                        layer.b -= lr * layer.db

            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        return losses

    def predict(self, X):
        """Predict class labels."""
        if len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]
        logits = self.forward(X)
        return self.softmax(logits)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X_train, X_test, y_train, y_test = create_simple_image_dataset()

    # -------- Experiment 1: Number of Filters --------
    print("\n1. EFFECT OF NUMBER OF FILTERS")
    print("-" * 40)
    for n_filters in [2, 4, 8, 16, 32]:
        cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=n_filters)
        cnn.fit(X_train, y_train, epochs=50, lr=0.01, verbose=False)
        train_acc = accuracy(y_train, cnn.predict(X_train))
        test_acc = accuracy(y_test, cnn.predict(X_test))
        n_params = sum(l.W.size + l.b.size for l in cnn.layers if hasattr(l, 'W'))
        print(f"n_filters={n_filters:<3} params={n_params:<6} train={train_acc:.3f} test={test_acc:.3f}")
    print("→ More filters = more capacity, but diminishing returns")

    # -------- Experiment 2: CNN vs MLP --------
    print("\n2. CNN vs MLP (Same Parameter Budget)")
    print("-" * 40)

    # Train CNN
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_acc = accuracy(y_test, cnn.predict(X_test))
    cnn_params = sum(l.W.size + l.b.size for l in cnn.layers if hasattr(l, 'W'))

    # Train MLP with similar params
    from importlib import import_module
    try:
        mlp_module = import_module('12_mlp')
        MLP = mlp_module.MLP

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        mlp = MLP(layer_sizes=[64, 32, 16, 2], activation='relu')
        mlp.fit(X_train_flat, y_train, epochs=100, lr=0.01, verbose=False)
        mlp_acc = accuracy(y_test, mlp.predict(X_test_flat))
        mlp_params = sum(l.W.size + l.b.size for l in mlp.layers if hasattr(l, 'W'))

        print(f"CNN: params={cnn_params:<6} accuracy={cnn_acc:.3f}")
        print(f"MLP: params={mlp_params:<6} accuracy={mlp_acc:.3f}")
        print("→ CNN achieves same/better with fewer params (weight sharing!)")
    except:
        print(f"CNN: params={cnn_params:<6} accuracy={cnn_acc:.3f}")
        print("MLP module not available for comparison")

    # -------- Experiment 3: Effect of Pooling --------
    print("\n3. EFFECT OF POOLING (Translation Invariance)")
    print("-" * 40)

    # Test with shifted images
    def shift_images(X, shift=1):
        """Shift images by 'shift' pixels."""
        X_shifted = np.zeros_like(X)
        X_shifted[:, shift:, shift:] = X[:, :-shift, :-shift]
        return X_shifted

    X_test_shifted = shift_images(X_test, shift=1)

    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)

    acc_original = accuracy(y_test, cnn.predict(X_test))
    acc_shifted = accuracy(y_test, cnn.predict(X_test_shifted))

    print(f"Original test accuracy:    {acc_original:.3f}")
    print(f"Shifted test accuracy:     {acc_shifted:.3f}")
    print(f"Drop due to shift:         {acc_original - acc_shifted:.3f}")
    print("→ Pooling provides SOME shift invariance, but not complete")

    # -------- Experiment 4: Pixel Shuffle Test --------
    print("\n4. CNN vs MLP on SHUFFLED PIXELS")
    print("-" * 40)
    print("This tests the spatial inductive bias!")

    # Create a fixed random permutation
    perm = np.random.permutation(64)

    X_train_shuffled = X_train.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)
    X_test_shuffled = X_test.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)

    # CNN on shuffled
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train_shuffled, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_shuffled_acc = accuracy(y_test, cnn.predict(X_test_shuffled))

    print(f"CNN on normal images:     {acc_original:.3f}")
    print(f"CNN on shuffled pixels:   {cnn_shuffled_acc:.3f}")
    print("→ CNN FAILS on shuffled pixels! Local structure destroyed.")

    # -------- Experiment 5: Visualize Learned Filters --------
    print("\n5. LEARNED FILTERS (First Conv Layer)")
    print("-" * 40)

    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)

    # Get first conv layer filters
    conv1 = cnn.layers[0]
    filters = conv1.W  # Shape: (n_filters, in_channels, H, W)

    print(f"Filter shape: {filters.shape}")
    print(f"Filter 0 (horizontal/vertical detector?):")
    print(filters[0, 0].round(2))
    print("→ Filters learn edge/pattern detectors automatically!")

    # -------- Experiment 6: Multi-class Classification --------
    print("\n6. MULTI-CLASS CLASSIFICATION (L, T, + shapes)")
    print("-" * 40)

    X_train_mc, X_test_mc, y_train_mc, y_test_mc = create_digit_like_dataset()

    for n_filters in [4, 8, 16]:
        cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=3, n_filters=n_filters)
        cnn.fit(X_train_mc, y_train_mc, epochs=100, lr=0.01, verbose=False)
        acc = accuracy(y_test_mc, cnn.predict(X_test_mc))
        print(f"n_filters={n_filters:<3} 3-class accuracy={acc:.3f}")


def visualize_cnn_story():
    """
    Create comprehensive CNN visualization that tells the complete story:
    1. Learned filters (what CNN sees)
    2. Feature maps (how activations propagate)
    3. CNN vs MLP comparison (weight sharing advantage)
    4. Shuffle test (spatial inductive bias proof)
    5. Loss curves (training dynamics)
    6. Parameter efficiency
    """
    print("\n" + "="*60)
    print("CNN VISUALIZATION — The Complete Story")
    print("="*60)

    np.random.seed(42)
    X_train, X_test, y_train, y_test = create_simple_image_dataset(n_samples=600)

    # Train CNN
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    losses_cnn = cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_acc = accuracy(y_test, cnn.predict(X_test))
    cnn_params = sum(l.W.size + l.b.size for l in cnn.layers if hasattr(l, 'W'))

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(15, 10))

    # ============ Plot 1: Sample Images & Learned Filters ============
    ax1 = fig.add_subplot(2, 3, 1)

    # Show 4 sample images (2 per class) and corresponding filter responses
    conv1 = cnn.layers[0]
    n_filters_show = 4

    # Create a grid showing input → filters → activations
    combined = np.zeros((8*2 + 2, 8*n_filters_show + n_filters_show - 1))

    # Top row: sample images
    sample_h = X_test[0]  # Horizontal stripe
    sample_v = X_test[1]  # Vertical stripe
    combined[:8, :8] = sample_h
    combined[:8, 9:17] = sample_v

    # Show filters
    for i in range(min(n_filters_show, conv1.W.shape[0])):
        # Normalize filter for display
        filt = conv1.W[i, 0]
        filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        # Place in bottom row
        col_start = i * 9
        combined[10:13, col_start:col_start+3] = filt_norm

    ax1.imshow(combined, cmap='RdBu_r')
    ax1.set_title('Input Images (top) & Learned Filters (bottom)\nFilters learn edge/orientation detectors')
    ax1.axis('off')

    # ============ Plot 2: Feature Maps After Conv ============
    ax2 = fig.add_subplot(2, 3, 2)

    # Get activations after first conv+relu
    test_img = X_test[0:1, np.newaxis, :, :]  # Add channel dim
    conv_out = cnn.layers[0].forward(test_img)
    relu_out = cnn.layers[1].forward(conv_out)

    # Show feature maps
    n_show = min(8, relu_out.shape[1])
    rows = 2
    cols = 4

    for i in range(n_show):
        ax_sub = fig.add_axes([0.35 + (i % cols) * 0.04, 0.55 + (1 - i // cols) * 0.08, 0.035, 0.07])
        ax_sub.imshow(relu_out[0, i], cmap='viridis')
        ax_sub.axis('off')
        ax_sub.set_title(f'F{i}', fontsize=6)

    ax2.text(0.5, 0.5, 'Feature Maps\n(After Conv+ReLU)\n\nEach filter responds\nto different patterns\n\n→ See small plots →',
             ha='center', va='center', fontsize=10, transform=ax2.transAxes)
    ax2.axis('off')

    # ============ Plot 3: CNN vs MLP ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Train MLP
    try:
        mlp_module = import_module('12_mlp')
        MLP = mlp_module.MLP

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        mlp = MLP(layer_sizes=[64, 32, 16, 2], activation='relu', n_epochs=100, lr=0.01, random_state=42)
        mlp.fit(X_train_flat, y_train)
        mlp_acc = accuracy(y_test, mlp.predict(X_test_flat))
        mlp_params = sum(w.size for w in mlp.weights) + sum(b.size for b in mlp.biases)

        # Bar chart
        models = ['CNN', 'MLP']
        accs = [cnn_acc, mlp_acc]
        params = [cnn_params, mlp_params]

        x = np.arange(2)
        width = 0.35

        bars1 = ax3.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
        ax3.set_ylabel('Accuracy', color='steelblue')
        ax3.set_ylim(0, 1.1)

        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, params, width, label='Parameters', color='coral', alpha=0.7)
        ax3_twin.set_ylabel('Parameters', color='coral')

        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.set_title(f'CNN vs MLP: Weight Sharing Wins!\nCNN: {cnn_params} params, MLP: {mlp_params} params')

        # Add value labels
        for bar, val in zip(bars1, accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                    ha='center', fontsize=9)
        for bar, val in zip(bars2, params):
            ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val}',
                         ha='center', fontsize=9)
    except:
        ax3.text(0.5, 0.5, 'MLP comparison\nnot available', ha='center', va='center', transform=ax3.transAxes)

    # ============ Plot 4: Shuffle Test (Key Insight!) ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Create fixed permutation
    perm = np.random.permutation(64)

    # Train CNN on shuffled data
    X_train_shuffled = X_train.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)
    X_test_shuffled = X_test.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)

    cnn_shuffled = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn_shuffled.fit(X_train_shuffled, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_shuffled_acc = accuracy(y_test, cnn_shuffled.predict(X_test_shuffled))

    # MLP on shuffled (should still work!)
    try:
        mlp_shuffled = MLP(layer_sizes=[64, 32, 16, 2], activation='relu', n_epochs=100, lr=0.01, random_state=42)
        X_train_shuf_flat = X_train_shuffled.reshape(-1, 64)
        X_test_shuf_flat = X_test_shuffled.reshape(-1, 64)
        mlp_shuffled.fit(X_train_shuf_flat, y_train)
        mlp_shuffled_acc = accuracy(y_test, mlp_shuffled.predict(X_test_shuf_flat))

        # Grouped bar chart
        conditions = ['Original', 'Shuffled Pixels']
        cnn_accs = [cnn_acc, cnn_shuffled_acc]
        mlp_accs = [mlp_acc, mlp_shuffled_acc]

        x = np.arange(2)
        width = 0.35

        ax4.bar(x - width/2, cnn_accs, width, label='CNN', color='steelblue')
        ax4.bar(x + width/2, mlp_accs, width, label='MLP', color='coral')

        ax4.set_xticks(x)
        ax4.set_xticklabels(conditions)
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.set_title('THE KEY TEST: Shuffled Pixels\nCNN FAILS (needs spatial structure)\nMLP works (no spatial bias)')

        # Add annotations
        ax4.annotate('', xy=(0.5, cnn_shuffled_acc), xytext=(0.5, cnn_acc),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax4.text(0.65, (cnn_acc + cnn_shuffled_acc)/2, 'CNN\nbreaks!', fontsize=9, color='red')
    except:
        ax4.text(0.5, 0.5, 'Shuffle test\nrequires MLP', ha='center', va='center', transform=ax4.transAxes)

    # ============ Plot 5: Loss Curve ============
    ax5 = fig.add_subplot(2, 3, 5)

    ax5.plot(losses_cnn, 'b-', linewidth=2, label='Training Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Cross-Entropy Loss')
    ax5.set_title('CNN Training Dynamics\nSmooth convergence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Visual Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = """
    CNN INDUCTIVE BIAS
    ══════════════════

    ✓ LOCAL: Small receptive fields
       → Captures edges, textures

    ✓ SHARED: Same filter everywhere
       → Translation equivariance
       → Far fewer parameters

    ✓ HIERARCHICAL: Stack layers
       → Simple → Complex features

    ✗ BREAKS when structure destroyed
       → Shuffled pixels = random noise
       → Proves spatial bias is essential

    KEY INSIGHT:
    CNN encodes "patterns are local
    and can appear anywhere"
    """

    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('CONVOLUTIONAL NEURAL NETWORK — The Complete Story\n'
                 'Local patterns + Weight sharing + Hierarchical composition',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_filters_and_activations():
    """Wrapper for backward compatibility."""
    return visualize_cnn_story()


def visualize_feature_hierarchy():
    """
    THE KEY CNN VISUALIZATION: Show how features build hierarchically.

    Layer 1: Detects edges/gradients (like Gabor filters)
    Layer 2: Detects textures/corners (combinations of edges)

    Shows:
    1. Input image
    2. All Layer 1 feature maps (edge detectors)
    3. All Layer 2 feature maps (texture detectors)
    4. The learned filters themselves
    """
    np.random.seed(42)
    X_train, X_test, y_train, y_test = create_simple_image_dataset(n_samples=600)

    # Train CNN
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)

    fig = plt.figure(figsize=(16, 12))

    # Get sample images - one horizontal, one vertical
    idx_h = np.where(y_test == 0)[0][0]  # Horizontal stripe
    idx_v = np.where(y_test == 1)[0][0]  # Vertical stripe

    sample_h = X_test[idx_h]
    sample_v = X_test[idx_v]

    # ============ Row 1: Horizontal stripe through the network ============
    # Input
    ax1 = fig.add_subplot(4, 6, 1)
    ax1.imshow(sample_h, cmap='gray')
    ax1.set_title('INPUT\n(Horizontal)', fontsize=9, fontweight='bold')
    ax1.axis('off')

    # Forward through layers
    x = sample_h[np.newaxis, np.newaxis, :, :]  # (1, 1, 8, 8)

    # Conv1 + ReLU
    conv1_out = cnn.layers[0].forward(x)
    relu1_out = cnn.layers[1].forward(conv1_out)

    # Show first 4 feature maps from layer 1
    for i in range(min(4, relu1_out.shape[1])):
        ax = fig.add_subplot(4, 6, 2 + i)
        ax.imshow(relu1_out[0, i], cmap='viridis')
        ax.set_title(f'Conv1-F{i}', fontsize=8)
        ax.axis('off')

    # Pool1
    pool1_out = cnn.layers[2].forward(relu1_out)

    # Conv2 + ReLU
    conv2_out = cnn.layers[3].forward(pool1_out)
    relu2_out = cnn.layers[4].forward(conv2_out)

    # Show feature map from layer 2
    ax6 = fig.add_subplot(4, 6, 6)
    # Average across channels for visualization
    ax6.imshow(relu2_out[0].mean(axis=0), cmap='viridis')
    ax6.set_title('Conv2 (avg)', fontsize=8)
    ax6.axis('off')

    # ============ Row 2: Vertical stripe through the network ============
    ax7 = fig.add_subplot(4, 6, 7)
    ax7.imshow(sample_v, cmap='gray')
    ax7.set_title('INPUT\n(Vertical)', fontsize=9, fontweight='bold')
    ax7.axis('off')

    # Forward through layers
    x_v = sample_v[np.newaxis, np.newaxis, :, :]
    conv1_out_v = cnn.layers[0].forward(x_v)
    relu1_out_v = cnn.layers[1].forward(conv1_out_v)

    for i in range(min(4, relu1_out_v.shape[1])):
        ax = fig.add_subplot(4, 6, 8 + i)
        ax.imshow(relu1_out_v[0, i], cmap='viridis')
        ax.set_title(f'Conv1-F{i}', fontsize=8)
        ax.axis('off')

    pool1_out_v = cnn.layers[2].forward(relu1_out_v)
    conv2_out_v = cnn.layers[3].forward(pool1_out_v)
    relu2_out_v = cnn.layers[4].forward(conv2_out_v)

    ax12 = fig.add_subplot(4, 6, 12)
    ax12.imshow(relu2_out_v[0].mean(axis=0), cmap='viridis')
    ax12.set_title('Conv2 (avg)', fontsize=8)
    ax12.axis('off')

    # ============ Row 3: The learned filters ============
    conv1 = cnn.layers[0]
    conv2 = cnn.layers[3]

    # Show Conv1 filters (3x3)
    ax13 = fig.add_subplot(4, 6, 13)
    ax13.text(0.5, 0.5, 'LAYER 1\nFILTERS\n(3×3)', ha='center', va='center',
              fontsize=10, fontweight='bold', transform=ax13.transAxes)
    ax13.axis('off')

    for i in range(min(4, conv1.W.shape[0])):
        ax = fig.add_subplot(4, 6, 14 + i)
        filt = conv1.W[i, 0]
        ax.imshow(filt, cmap='RdBu_r', vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
        ax.set_title(f'F{i}', fontsize=8)
        ax.axis('off')

    ax18 = fig.add_subplot(4, 6, 18)
    ax18.text(0.5, 0.5, 'Edge\ndetectors', ha='center', va='center',
              fontsize=9, transform=ax18.transAxes)
    ax18.axis('off')

    # ============ Row 4: Explanation ============
    ax19 = fig.add_subplot(4, 6, 19)
    ax19.axis('off')

    ax20 = fig.add_subplot(4, 6, (20, 24))
    ax20.axis('off')
    explanation = """
    HIERARCHICAL FEATURE LEARNING
    ═══════════════════════════════

    Layer 1 (Conv1): Detects EDGES
    • Each filter responds to a specific edge orientation
    • Same filter fires wherever that edge appears (translation equivariance)
    • Different filters detect different orientations

    Layer 2 (Conv2): Detects TEXTURES
    • Combines edges from Layer 1
    • Horizontal stripes = horizontal edges repeated
    • Vertical stripes = vertical edges repeated

    KEY INSIGHT: The network learns a HIERARCHY
    Simple patterns → Complex patterns → Objects
    """
    ax20.text(0.05, 0.95, explanation, transform=ax20.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('CNN FEATURE HIERARCHY: How Convolution Builds Complex Features\n'
                 'Same filter responds to same pattern anywhere (weight sharing)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_spatial_inductive_bias():
    """
    THE CRITICAL TEST: Prove CNN's spatial inductive bias.

    If we SHUFFLE the pixels (destroy spatial structure):
    - CNN should FAIL (it relies on local patterns)
    - MLP should still WORK (it has no spatial assumption)

    This proves CNN encodes "patterns are local and can appear anywhere"
    """
    np.random.seed(42)
    X_train, X_test, y_train, y_test = create_simple_image_dataset(n_samples=600)

    fig = plt.figure(figsize=(16, 10))

    # Create fixed permutation
    perm = np.random.permutation(64)
    inv_perm = np.argsort(perm)  # To unshuffle

    # Shuffle the pixels
    X_train_shuffled = X_train.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)
    X_test_shuffled = X_test.reshape(-1, 64)[:, perm].reshape(-1, 8, 8)

    # ============ Row 1: Show what shuffling does ============
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(X_test[0], cmap='gray')
    ax1.set_title('Original Image\n(Horizontal stripe)', fontsize=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(X_test_shuffled[0], cmap='gray')
    ax2.set_title('SHUFFLED Pixels\n(Same info, destroyed structure)', fontsize=10)
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(X_test[1], cmap='gray')
    ax3.set_title('Original Image\n(Vertical stripe)', fontsize=10)
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(X_test_shuffled[1], cmap='gray')
    ax4.set_title('SHUFFLED Pixels\n(Same info, destroyed structure)', fontsize=10)
    ax4.axis('off')

    # ============ Row 2: Train and compare ============

    # Train CNN on original
    cnn_orig = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn_orig.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_orig_acc = accuracy(y_test, cnn_orig.predict(X_test))

    # Train CNN on shuffled
    cnn_shuf = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn_shuf.fit(X_train_shuffled, y_train, epochs=100, lr=0.01, verbose=False)
    cnn_shuf_acc = accuracy(y_test, cnn_shuf.predict(X_test_shuffled))

    # Train MLP on original
    try:
        mlp_module = import_module('12_mlp')
        MLP = mlp_module.MLP

        X_train_flat = X_train.reshape(-1, 64)
        X_test_flat = X_test.reshape(-1, 64)
        X_train_shuf_flat = X_train_shuffled.reshape(-1, 64)
        X_test_shuf_flat = X_test_shuffled.reshape(-1, 64)

        mlp_orig = MLP(layer_sizes=[64, 32, 16, 2], activation='relu',
                       n_epochs=100, lr=0.1, random_state=42)
        mlp_orig.fit(X_train_flat, y_train)
        mlp_orig_acc = accuracy(y_test, mlp_orig.predict(X_test_flat))

        mlp_shuf = MLP(layer_sizes=[64, 32, 16, 2], activation='relu',
                       n_epochs=100, lr=0.1, random_state=42)
        mlp_shuf.fit(X_train_shuf_flat, y_train)
        mlp_shuf_acc = accuracy(y_test, mlp_shuf.predict(X_test_shuf_flat))

        mlp_available = True
    except:
        mlp_available = False
        mlp_orig_acc = 0
        mlp_shuf_acc = 0

    # Bar chart comparison
    ax5 = fig.add_subplot(2, 4, 5)

    if mlp_available:
        x = np.arange(2)
        width = 0.35

        ax5.bar(x - width/2, [cnn_orig_acc, cnn_shuf_acc], width, label='CNN', color='steelblue')
        ax5.bar(x + width/2, [mlp_orig_acc, mlp_shuf_acc], width, label='MLP', color='coral')

        ax5.set_xticks(x)
        ax5.set_xticklabels(['Original', 'Shuffled'])
        ax5.set_ylabel('Accuracy')
        ax5.set_ylim(0, 1.1)
        ax5.legend()
        ax5.set_title('THE KEY TEST\nCNN needs structure, MLP does not', fontsize=10, fontweight='bold')

        # Add value labels
        for i, (c, m) in enumerate(zip([cnn_orig_acc, cnn_shuf_acc], [mlp_orig_acc, mlp_shuf_acc])):
            ax5.text(i - width/2, c + 0.02, f'{c:.2f}', ha='center', fontsize=9)
            ax5.text(i + width/2, m + 0.02, f'{m:.2f}', ha='center', fontsize=9)

        # Arrow showing CNN drop
        ax5.annotate('', xy=(0.5 - width/2, cnn_shuf_acc),
                    xytext=(0.5 - width/2, cnn_orig_acc - 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax5.text(0.3, (cnn_orig_acc + cnn_shuf_acc)/2, 'CNN\nBREAKS!',
                fontsize=10, color='red', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'MLP not available', ha='center', va='center', transform=ax5.transAxes)

    # Accuracy drop analysis
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.axis('off')

    if mlp_available:
        cnn_drop = cnn_orig_acc - cnn_shuf_acc
        mlp_drop = mlp_orig_acc - mlp_shuf_acc

        analysis = f"""
    ACCURACY DROP ANALYSIS
    ═══════════════════════

    CNN:
      Original:  {cnn_orig_acc:.2f}
      Shuffled:  {cnn_shuf_acc:.2f}
      Drop:      {cnn_drop:.2f} {'← BIG DROP!' if cnn_drop > 0.1 else ''}

    MLP:
      Original:  {mlp_orig_acc:.2f}
      Shuffled:  {mlp_shuf_acc:.2f}
      Drop:      {mlp_drop:.2f} {'← Still works!' if mlp_drop < 0.1 else ''}

    CONCLUSION:
    CNN encodes spatial structure
    MLP treats pixels independently
        """
        ax6.text(0.1, 0.95, analysis, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # Why CNN fails
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')

    why_text = """
    WHY CNN FAILS ON SHUFFLED:
    ══════════════════════════

    CNN uses 3×3 filters that detect
    LOCAL patterns (edges, corners).

    When you shuffle pixels:
    • Adjacent pixels no longer related
    • Local patterns destroyed
    • Filters find nothing meaningful

    This PROVES the inductive bias:
    "Patterns are LOCAL and can
     appear ANYWHERE"
    """
    ax7.text(0.1, 0.95, why_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Why MLP works
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    why_mlp = """
    WHY MLP STILL WORKS:
    ════════════════════

    MLP treats each pixel as an
    independent input feature.

    Shuffling is just a relabeling:
    • pixel[0] → feature[perm[0]]
    • Same information, different order
    • MLP learns new mapping easily

    MLP has NO spatial assumption:
    "All input features are
     equally connected"
    """
    ax8.text(0.1, 0.95, why_mlp, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle('SPATIAL INDUCTIVE BIAS: The Critical Proof\n'
                 'Shuffle pixels → CNN fails, MLP works → CNN needs local structure!',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_translation_equivariance():
    """
    Show translation equivariance: shift input → shift output.

    Same pattern at different positions gives same filter response,
    just at different positions.
    """
    np.random.seed(42)
    X_train, X_test, y_train, y_test = create_simple_image_dataset(n_samples=600)

    # Train CNN
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Create a simple pattern and shift it
    pattern = np.zeros((8, 8))
    pattern[2:5, 2:5] = 1.0  # 3x3 square

    shifts = [(0, 0), (0, 2), (2, 0), (2, 2)]

    # Row 1: Shifted inputs
    axes[0, 0].text(0.5, 0.5, 'INPUT\nIMAGES', ha='center', va='center',
                    fontsize=12, fontweight='bold', transform=axes[0, 0].transAxes)
    axes[0, 0].axis('off')

    shifted_images = []
    for idx, (dy, dx) in enumerate(shifts):
        shifted = np.zeros((8, 8))
        y_start, y_end = dy, min(dy + 3, 8)
        x_start, x_end = dx, min(dx + 3, 8)
        shifted[y_start:y_end, x_start:x_end] = 1.0
        shifted_images.append(shifted)

        axes[0, idx + 1].imshow(shifted, cmap='gray')
        axes[0, idx + 1].set_title(f'Shift ({dy},{dx})', fontsize=9)
        axes[0, idx + 1].axis('off')

    # Row 2: Feature maps from Conv1 (one specific filter)
    axes[1, 0].text(0.5, 0.5, 'CONV1\nOUTPUT\n(Filter 0)', ha='center', va='center',
                    fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    for idx, img in enumerate(shifted_images):
        x = img[np.newaxis, np.newaxis, :, :]
        conv_out = cnn.layers[0].forward(x)
        relu_out = cnn.layers[1].forward(conv_out)

        axes[1, idx + 1].imshow(relu_out[0, 0], cmap='viridis')
        axes[1, idx + 1].set_title(f'Response shifts too!', fontsize=9)
        axes[1, idx + 1].axis('off')

    # Row 3: Explanation
    for i in range(5):
        axes[2, i].axis('off')

    axes[2, 2].text(0.5, 0.5,
        'TRANSLATION EQUIVARIANCE\n'
        '═══════════════════════════\n\n'
        'When input shifts by (dy, dx),\n'
        'the feature map shifts by (dy, dx) too!\n\n'
        'This is because the SAME filter\n'
        'is applied at EVERY position.\n\n'
        'Weight sharing → Translation equivariance\n'
        '(Same pattern → Same response, anywhere)',
        ha='center', va='center', fontsize=11, fontfamily='monospace',
        transform=axes[2, 2].transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('TRANSLATION EQUIVARIANCE: Shift Input → Shift Output\n'
                 'Same filter at every position = same response pattern shifts with input',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def benchmark_on_shapes():
    """Benchmark CNN on shape recognition."""
    print("\n" + "="*60)
    print("BENCHMARK: CNN on Shape Recognition")
    print("="*60)

    # Binary (stripes)
    X_train, X_test, y_train, y_test = create_simple_image_dataset(n_samples=800)
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=2, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    acc_binary = accuracy(y_test, cnn.predict(X_test))
    print(f"Binary (H vs V stripes): {acc_binary:.3f}")

    # Multi-class (shapes)
    X_train, X_test, y_train, y_test = create_digit_like_dataset(n_samples=1200)
    cnn = SimpleCNN(input_shape=(1, 8, 8), n_classes=3, n_filters=8)
    cnn.fit(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    acc_multi = accuracy(y_test, cnn.predict(X_test))
    print(f"Multi-class (L, T, +):   {acc_multi:.3f}")

    return {'binary': acc_binary, 'multiclass': acc_multi}


if __name__ == '__main__':
    print("="*60)
    print("CNN — Convolutional Neural Network")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    LOCAL connections + SHARED weights across positions.
    The SAME filter detects the SAME pattern ANYWHERE.

THE KEY INSIGHT:
    Convolution = template matching
    Weight sharing = translation equivariance
    This is an INDUCTIVE BIAS for spatial data!

HIERARCHICAL FEATURES:
    Layer 1: Edges
    Layer 2: Textures (combinations of edges)
    Layer 3+: Parts, objects (higher abstractions)

WHY IT WORKS FOR IMAGES:
    - Patterns are LOCAL (edges, textures)
    - Patterns can appear ANYWHERE
    - Composition: simple → complex

WHY IT FAILS ON SHUFFLED PIXELS:
    Shuffling destroys LOCAL structure!
    MLP doesn't care (no spatial bias), CNN breaks.
    """)

    ablation_experiments()
    results = benchmark_on_shapes()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Feature hierarchy (THE KEY INSIGHT)
    print("1. Generating feature hierarchy visualization...")
    fig1 = visualize_feature_hierarchy()
    save_path1 = '/Users/sid47/ML Algorithms/13_cnn_hierarchy.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {save_path1}")
    plt.close(fig1)

    # 2. Spatial inductive bias proof (shuffle test)
    print("2. Generating spatial inductive bias visualization...")
    fig2 = visualize_spatial_inductive_bias()
    save_path2 = '/Users/sid47/ML Algorithms/13_cnn_shuffle_test.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {save_path2}")
    plt.close(fig2)

    # 3. Translation equivariance
    print("3. Generating translation equivariance visualization...")
    fig3 = visualize_translation_equivariance()
    save_path3 = '/Users/sid47/ML Algorithms/13_cnn_equivariance.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {save_path3}")
    plt.close(fig3)

    # 4. Complete story (existing, for backward compatibility)
    print("4. Generating complete story visualization...")
    fig4 = visualize_cnn_story()
    save_path4 = '/Users/sid47/ML Algorithms/13_cnn.png'
    fig4.savefig(save_path4, dpi=100, bbox_inches='tight')
    print(f"   Saved to: {save_path4}")
    plt.close(fig4)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Convolution = LOCAL receptive field + WEIGHT SHARING
2. Translation equivariance: same pattern → same response anywhere
3. Pooling adds SOME translation invariance
4. Hierarchical: edges → textures → parts → objects
5. FAILS on shuffled pixels (spatial inductive bias)
6. Fewer parameters than MLP with same/better performance

===============================================================
THE KEY INSIGHTS (see visualizations):
===============================================================

    13_cnn_hierarchy.png    — Layer 1 edges → Layer 2 textures
    13_cnn_shuffle_test.png — CNN FAILS on shuffled, MLP works!
    13_cnn_equivariance.png — Shift input → Shift output
    13_cnn.png              — Complete story overview

CNN's INDUCTIVE BIAS:
    "Patterns are LOCAL and can appear ANYWHERE"
    This is why it works for images but breaks on shuffled pixels.

NEXT: RNN/LSTM — exploit sequential structure through recurrence
    """)
