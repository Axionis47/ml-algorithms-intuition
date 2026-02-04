"""
RNN & LSTM — Paradigm: LEARNED FEATURES (Sequential Memory)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Process sequences by maintaining a HIDDEN STATE that gets updated
at each time step. The hidden state is a COMPRESSED MEMORY of the past.

RNN:
    h_t = tanh(W_xh × x_t + W_hh × h_{t-1} + b)

The SAME weights are used at every time step (weight sharing in TIME).
This is like CNNs share weights across SPACE.

===============================================================
THE KEY INSIGHT: HIDDEN STATE = COMPRESSED HISTORY
===============================================================

At time t, the hidden state h_t contains information about:
    x_1, x_2, ..., x_t

But compressed into a fixed-size vector!

This is both powerful (arbitrary length sequences) and limiting
(information must fit in the bottleneck).

===============================================================
THE VANISHING GRADIENT PROBLEM
===============================================================

Backprop through time (BPTT) chains derivatives:
    ∂h_t/∂h_1 = ∂h_t/∂h_{t-1} × ∂h_{t-1}/∂h_{t-2} × ... × ∂h_2/∂h_1

Each term is bounded by max eigenvalue of W_hh.
If |λ_max| < 1: gradient → 0 (vanishes)
If |λ_max| > 1: gradient → ∞ (explodes)

Long sequences = many multiplications = extreme gradients.

RNNs struggle to learn long-range dependencies!

===============================================================
LSTM: LEARNED FORGETTING
===============================================================

LSTM solves vanishing gradients with GATES:

    f_t = σ(W_f × [h_{t-1}, x_t] + b_f)   # Forget gate
    i_t = σ(W_i × [h_{t-1}, x_t] + b_i)   # Input gate
    o_t = σ(W_o × [h_{t-1}, x_t] + b_o)   # Output gate

    c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)  # Candidate cell state

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        # Cell state update
    h_t = o_t ⊙ tanh(c_t)                   # Hidden state

THE KEY: The cell state c_t can flow unchanged (when f_t ≈ 1, i_t ≈ 0).
This creates a "gradient highway" through time.

Gates LEARN when to:
    - Forget old information (f_t → 0)
    - Remember new information (i_t → 1)
    - Output the hidden state (o_t → 1)

===============================================================
INDUCTIVE BIAS
===============================================================

1. Sequential processing: order matters
2. Markov-ish: future depends on (compressed) past
3. Weight sharing in time: same dynamics at each step
4. Fixed memory bottleneck: h_t is finite-dimensional

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
accuracy = datasets_module.accuracy


def create_sequence_dataset(n_samples=500, seq_len=10, pattern='sum'):
    """
    Create sequence classification datasets.

    Patterns:
    - 'sum': class = 1 if sum of sequence > 0
    - 'first_last': class = 1 if first and last have same sign
    - 'xor_first_last': class = 1 if xor of (first>0, last>0)
    - 'majority': class = 1 if more positives than negatives
    - 'delayed': class = 1 if first element > 0 (tests long-range memory)
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, seq_len, 1)  # (samples, timesteps, features)
    y = np.zeros(n_samples, dtype=int)

    if pattern == 'sum':
        y = (X.sum(axis=(1, 2)) > 0).astype(int)

    elif pattern == 'first_last':
        first = X[:, 0, 0]
        last = X[:, -1, 0]
        y = ((first > 0) == (last > 0)).astype(int)

    elif pattern == 'xor_first_last':
        first = X[:, 0, 0] > 0
        last = X[:, -1, 0] > 0
        y = (first ^ last).astype(int)

    elif pattern == 'majority':
        y = (np.sum(X > 0, axis=(1, 2)) > seq_len / 2).astype(int)

    elif pattern == 'delayed':
        # Decision based ONLY on first element
        # This tests long-range dependency learning
        y = (X[:, 0, 0] > 0).astype(int)

    # Split
    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


class SimpleRNN:
    """
    Vanilla RNN for sequence classification.

    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

    This will STRUGGLE with long sequences (vanishing gradients).
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Parameters:
        -----------
        input_size : Dimension of input at each timestep
        hidden_size : Dimension of hidden state
        output_size : Number of output classes
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights (Xavier initialization)
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + output_size))

        self.W_xh = np.random.randn(input_size, hidden_size) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_h = np.zeros(hidden_size)

        self.W_hy = np.random.randn(hidden_size, output_size) * scale_hy
        self.b_y = np.zeros(output_size)

    def forward(self, X):
        """
        Forward pass through time.

        X shape: (batch_size, seq_len, input_size)
        Returns: final hidden states and all hidden states
        """
        batch_size, seq_len, _ = X.shape

        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))

        # Store all hidden states for backprop
        h_states = [h]

        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, input_size)

            # RNN update: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            h_states.append(h)

        self.h_states = h_states
        self.X = X

        return h, h_states

    def output(self, h):
        """Compute output from final hidden state."""
        return h @ self.W_hy + self.b_y

    def softmax(self, logits):
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def backward(self, y, lr=0.01, clip_value=5.0):
        """
        Backpropagation through time (BPTT).

        This is where vanishing gradients happen!
        """
        batch_size = y.shape[0]
        seq_len = self.X.shape[1]
        h_states = self.h_states

        # Output layer gradient
        h_final = h_states[-1]
        logits = self.output(h_final)
        probs = self.softmax(logits)

        # Cross-entropy gradient
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        # Gradients for output layer
        dW_hy = h_final.T @ dlogits
        db_y = np.sum(dlogits, axis=0)

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        # Gradient flowing back through time
        dh_next = dlogits @ self.W_hy.T

        # BPTT: go backwards through time
        for t in reversed(range(seq_len)):
            h_t = h_states[t + 1]
            h_prev = h_states[t]
            x_t = self.X[:, t, :]

            # Gradient through tanh: d/dx tanh(x) = 1 - tanh²(x)
            dtanh = dh_next * (1 - h_t ** 2)

            # Accumulate gradients
            dW_xh += x_t.T @ dtanh
            dW_hh += h_prev.T @ dtanh
            db_h += np.sum(dtanh, axis=0)

            # Gradient to previous hidden state
            dh_next = dtanh @ self.W_hh.T

        # Gradient clipping (prevent explosion)
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Update weights
        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.b_h -= lr * db_h
        self.W_hy -= lr * dW_hy
        self.b_y -= lr * db_y

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        """Train the RNN."""
        losses = []

        for epoch in range(epochs):
            # Forward
            h_final, _ = self.forward(X)
            logits = self.output(h_final)
            probs = self.softmax(logits)

            # Loss
            loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10))
            losses.append(loss)

            # Backward
            self.backward(y, lr=lr)

            if verbose and (epoch + 1) % 20 == 0:
                acc = accuracy(y, np.argmax(logits, axis=1))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.3f}")

        return losses

    def predict(self, X):
        """Predict class labels."""
        h_final, _ = self.forward(X)
        logits = self.output(h_final)
        return np.argmax(logits, axis=1)


class LSTM:
    """
    Long Short-Term Memory network.

    Solves the vanishing gradient problem with GATES.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Combined weight matrices for efficiency
        # W_x: input to all gates (forget, input, output, cell)
        # W_h: hidden to all gates

        combined_size = 4 * hidden_size  # f, i, o, c

        scale_x = np.sqrt(2.0 / (input_size + hidden_size))
        scale_h = np.sqrt(2.0 / (hidden_size + hidden_size))

        self.W_x = np.random.randn(input_size, combined_size) * scale_x
        self.W_h = np.random.randn(hidden_size, combined_size) * scale_h
        self.b = np.zeros(combined_size)

        # Initialize forget gate bias to 1 (helps gradient flow initially)
        self.b[:hidden_size] = 1.0

        # Output layer
        self.W_hy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_y = np.zeros(output_size)

    def sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, X):
        """
        Forward pass with LSTM cells.

        X shape: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = X.shape
        hs = self.hidden_size

        # Initialize states
        h = np.zeros((batch_size, hs))
        c = np.zeros((batch_size, hs))

        # Store for backprop
        self.cache = {
            'X': X,
            'h_states': [h],
            'c_states': [c],
            'gates': []  # (f, i, o, c_tilde) at each step
        }

        for t in range(seq_len):
            x_t = X[:, t, :]

            # Compute all gates at once
            gates = x_t @ self.W_x + h @ self.W_h + self.b

            # Split into individual gates
            f = self.sigmoid(gates[:, :hs])           # Forget gate
            i = self.sigmoid(gates[:, hs:2*hs])       # Input gate
            o = self.sigmoid(gates[:, 2*hs:3*hs])     # Output gate
            c_tilde = np.tanh(gates[:, 3*hs:])        # Candidate cell

            # Update cell state: c_t = f ⊙ c_{t-1} + i ⊙ c̃_t
            c = f * c + i * c_tilde

            # Update hidden state: h_t = o ⊙ tanh(c_t)
            h = o * np.tanh(c)

            self.cache['h_states'].append(h)
            self.cache['c_states'].append(c)
            self.cache['gates'].append((f, i, o, c_tilde))

        return h

    def output(self, h):
        """Output layer."""
        return h @ self.W_hy + self.b_y

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def backward(self, y, lr=0.01, clip_value=5.0):
        """BPTT for LSTM."""
        batch_size = y.shape[0]
        X = self.cache['X']
        seq_len = X.shape[1]
        hs = self.hidden_size

        h_states = self.cache['h_states']
        c_states = self.cache['c_states']
        gates = self.cache['gates']

        # Output gradient
        h_final = h_states[-1]
        logits = self.output(h_final)
        probs = self.softmax(logits)

        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        dW_hy = h_final.T @ dlogits
        db_y = np.sum(dlogits, axis=0)

        # Initialize gradients
        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        db = np.zeros_like(self.b)

        # Backprop through time
        dh_next = dlogits @ self.W_hy.T
        dc_next = np.zeros((batch_size, hs))

        for t in reversed(range(seq_len)):
            x_t = X[:, t, :]
            h_prev = h_states[t]
            c_prev = c_states[t]
            c_t = c_states[t + 1]
            f, i, o, c_tilde = gates[t]

            # Gradient to output gate: dL/do = dL/dh × tanh(c)
            tanh_c = np.tanh(c_t)
            do = dh_next * tanh_c

            # Gradient to cell state through h
            dc = dh_next * o * (1 - tanh_c ** 2) + dc_next

            # Gradient to forget gate: dL/df = dL/dc × c_{t-1}
            df = dc * c_prev

            # Gradient to input gate: dL/di = dL/dc × c̃
            di = dc * c_tilde

            # Gradient to candidate: dL/dc̃ = dL/dc × i
            dc_tilde = dc * i

            # Gradient to previous cell state: dL/dc_{t-1} = dL/dc × f
            dc_next = dc * f

            # Gradients through gate activations
            df_raw = df * f * (1 - f)  # sigmoid derivative
            di_raw = di * i * (1 - i)
            do_raw = do * o * (1 - o)
            dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)  # tanh derivative

            # Combine gates
            dgates = np.concatenate([df_raw, di_raw, do_raw, dc_tilde_raw], axis=1)

            # Accumulate weight gradients
            dW_x += x_t.T @ dgates
            dW_h += h_prev.T @ dgates
            db += np.sum(dgates, axis=0)

            # Gradient to previous hidden state
            dh_next = dgates @ self.W_h.T

        # Clip gradients
        for grad in [dW_x, dW_h, db, dW_hy, db_y]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Update weights
        self.W_x -= lr * dW_x
        self.W_h -= lr * dW_h
        self.b -= lr * db
        self.W_hy -= lr * dW_hy
        self.b_y -= lr * db_y

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        losses = []

        for epoch in range(epochs):
            h_final = self.forward(X)
            logits = self.output(h_final)
            probs = self.softmax(logits)

            loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-10))
            losses.append(loss)

            self.backward(y, lr=lr)

            if verbose and (epoch + 1) % 20 == 0:
                acc = accuracy(y, np.argmax(logits, axis=1))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.3f}")

        return losses

    def predict(self, X):
        h_final = self.forward(X)
        logits = self.output(h_final)
        return np.argmax(logits, axis=1)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # -------- Experiment 1: RNN vs LSTM on Short Sequences --------
    print("\n1. RNN vs LSTM on SHORT SEQUENCES (len=5)")
    print("-" * 40)
    X_train, X_test, y_train, y_test = create_sequence_dataset(n_samples=800, seq_len=5, pattern='sum')

    rnn = SimpleRNN(input_size=1, hidden_size=16, output_size=2)
    rnn.fit(X_train, y_train, epochs=100, lr=0.1, verbose=False)
    rnn_acc = accuracy(y_test, rnn.predict(X_test))

    lstm = LSTM(input_size=1, hidden_size=16, output_size=2)
    lstm.fit(X_train, y_train, epochs=100, lr=0.1, verbose=False)
    lstm_acc = accuracy(y_test, lstm.predict(X_test))

    print(f"RNN accuracy:  {rnn_acc:.3f}")
    print(f"LSTM accuracy: {lstm_acc:.3f}")
    print("→ Both work well on short sequences")

    # -------- Experiment 2: RNN vs LSTM on LONG Sequences --------
    print("\n2. RNN vs LSTM on LONG SEQUENCES (len=50)")
    print("-" * 40)
    print("Testing long-range dependency (delayed pattern)")
    X_train, X_test, y_train, y_test = create_sequence_dataset(n_samples=800, seq_len=50, pattern='delayed')

    rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
    rnn.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    rnn_acc = accuracy(y_test, rnn.predict(X_test))

    lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
    lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
    lstm_acc = accuracy(y_test, lstm.predict(X_test))

    print(f"RNN accuracy:  {rnn_acc:.3f}")
    print(f"LSTM accuracy: {lstm_acc:.3f}")
    print("→ RNN FAILS on long-range dependency!")
    print("  (Vanishing gradients prevent learning)")

    # -------- Experiment 3: Sequence Length Sweep --------
    print("\n3. SEQUENCE LENGTH SWEEP (Delayed Pattern)")
    print("-" * 40)
    print("How far back can each model remember?")

    lengths = [5, 10, 20, 30, 50]
    rnn_accs = []
    lstm_accs = []

    for seq_len in lengths:
        X_train, X_test, y_train, y_test = create_sequence_dataset(
            n_samples=800, seq_len=seq_len, pattern='delayed')

        rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
        rnn.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        rnn_acc = accuracy(y_test, rnn.predict(X_test))
        rnn_accs.append(rnn_acc)

        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        lstm_acc = accuracy(y_test, lstm.predict(X_test))
        lstm_accs.append(lstm_acc)

        print(f"len={seq_len:<3} RNN={rnn_acc:.3f} LSTM={lstm_acc:.3f}")

    print("→ RNN degrades rapidly with length")
    print("→ LSTM maintains performance (gradient highway)")

    # -------- Experiment 4: Hidden Size Effect --------
    print("\n4. EFFECT OF HIDDEN SIZE")
    print("-" * 40)
    X_train, X_test, y_train, y_test = create_sequence_dataset(n_samples=800, seq_len=20, pattern='sum')

    for hs in [4, 8, 16, 32, 64]:
        lstm = LSTM(input_size=1, hidden_size=hs, output_size=2)
        lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, lstm.predict(X_test))
        n_params = lstm.W_x.size + lstm.W_h.size + lstm.b.size + lstm.W_hy.size + lstm.b_y.size
        print(f"hidden_size={hs:<3} params={n_params:<6} accuracy={acc:.3f}")
    print("→ Larger hidden = more capacity, but diminishing returns")

    # -------- Experiment 5: Different Patterns --------
    print("\n5. DIFFERENT SEQUENCE PATTERNS")
    print("-" * 40)

    patterns = ['sum', 'first_last', 'xor_first_last', 'majority', 'delayed']

    for pattern in patterns:
        X_train, X_test, y_train, y_test = create_sequence_dataset(
            n_samples=800, seq_len=20, pattern=pattern)

        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        acc = accuracy(y_test, lstm.predict(X_test))
        print(f"pattern={pattern:<15} accuracy={acc:.3f}")

    print("→ 'delayed' is hardest (requires remembering first element)")

    # -------- Experiment 6: Gradient Magnitude Through Time --------
    print("\n6. GRADIENT FLOW ANALYSIS")
    print("-" * 40)
    print("Comparing gradient magnitudes through time")

    X_train, X_test, y_train, y_test = create_sequence_dataset(n_samples=100, seq_len=20, pattern='sum')

    # We'll measure by looking at weight gradient magnitudes
    rnn = SimpleRNN(input_size=1, hidden_size=16, output_size=2)
    lstm = LSTM(input_size=1, hidden_size=16, output_size=2)

    # Train both
    rnn.fit(X_train, y_train, epochs=50, lr=0.05, verbose=False)
    lstm.fit(X_train, y_train, epochs=50, lr=0.05, verbose=False)

    print(f"RNN W_hh max:  {np.max(np.abs(rnn.W_hh)):.4f}")
    print(f"LSTM W_h max:  {np.max(np.abs(lstm.W_h)):.4f}")
    print("→ LSTM weights remain more stable")


def visualize_gate_activations():
    """Visualize LSTM gates over a sequence."""
    print("\n" + "="*60)
    print("LSTM GATE VISUALIZATION")
    print("="*60)

    # Create a simple sequence
    X_train, X_test, y_train, y_test = create_sequence_dataset(n_samples=800, seq_len=20, pattern='delayed')

    lstm = LSTM(input_size=1, hidden_size=16, output_size=2)
    lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)

    # Get gate activations for a test sample
    sample = X_test[0:1]  # Shape (1, seq_len, 1)
    _ = lstm.forward(sample)

    gates = lstm.cache['gates']

    # Extract gate values over time
    seq_len = sample.shape[1]
    forget_gates = np.array([g[0][0, 0] for g in gates])  # First hidden unit
    input_gates = np.array([g[1][0, 0] for g in gates])
    output_gates = np.array([g[2][0, 0] for g in gates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot input sequence
    axes[0, 0].plot(sample[0, :, 0], 'b-o', markersize=4)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Input Sequence')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Value')

    # Plot gates
    axes[0, 1].plot(forget_gates, 'r-o', label='Forget', markersize=4)
    axes[0, 1].plot(input_gates, 'g-o', label='Input', markersize=4)
    axes[0, 1].plot(output_gates, 'b-o', label='Output', markersize=4)
    axes[0, 1].set_title('LSTM Gates (Hidden Unit 0)')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Activation')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    # Cell state evolution
    c_states = np.array([c[0, 0] for c in lstm.cache['c_states'][1:]])
    axes[1, 0].plot(c_states, 'm-o', markersize=4)
    axes[1, 0].set_title('Cell State (Hidden Unit 0)')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Cell value')

    # Hidden state evolution
    h_states = np.array([h[0, 0] for h in lstm.cache['h_states'][1:]])
    axes[1, 1].plot(h_states, 'c-o', markersize=4)
    axes[1, 1].set_title('Hidden State (Hidden Unit 0)')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Hidden value')

    plt.suptitle('LSTM Internal Dynamics\n'
                 'Forget gate ≈1 → preserve memory, Input gate ≈1 → add new info',
                 fontsize=12)
    plt.tight_layout()
    return fig


def visualize_vanishing_gradient():
    """
    THE KEY VISUALIZATION: Show vanishing gradients in RNN vs LSTM.

    This demonstrates WHY RNN fails on long sequences:
    - Gradients must flow backward through EVERY timestep
    - In RNN: gradient shrinks exponentially
    - In LSTM: gradient highway keeps it flowing
    """
    np.random.seed(42)

    fig = plt.figure(figsize=(16, 12))

    # ============ Part 1: RNN vs LSTM Accuracy by Sequence Length ============
    ax1 = fig.add_subplot(2, 2, 1)

    lengths = [5, 10, 15, 20, 30, 40, 50]
    rnn_accs = []
    lstm_accs = []

    print("Computing RNN vs LSTM accuracy by sequence length...")
    for seq_len in lengths:
        X_train, X_test, y_train, y_test = create_sequence_dataset(
            n_samples=600, seq_len=seq_len, pattern='delayed')

        # RNN
        rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
        rnn.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        rnn_acc = accuracy(y_test, rnn.predict(X_test))
        rnn_accs.append(rnn_acc)

        # LSTM
        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        lstm_acc = accuracy(y_test, lstm.predict(X_test))
        lstm_accs.append(lstm_acc)

    ax1.plot(lengths, rnn_accs, 'r-o', label='RNN', linewidth=2, markersize=8)
    ax1.plot(lengths, lstm_accs, 'b-o', label='LSTM', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('THE KEY TEST: Long-Range Dependency\n(Classify by FIRST element only)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.annotate('RNN collapses\nto random!', xy=(40, rnn_accs[-2]),
                xytext=(30, 0.65), fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate('LSTM maintains\nperformance', xy=(40, lstm_accs[-2]),
                xytext=(25, 0.95), fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # ============ Part 2: Gradient Flow Illustration ============
    ax2 = fig.add_subplot(2, 2, 2)

    # Theoretical gradient decay
    timesteps = np.arange(1, 51)
    # RNN: gradient decays as λ^t where λ < 1 typically
    lambda_rnn = 0.9  # Typical eigenvalue magnitude
    rnn_gradient = lambda_rnn ** timesteps

    # LSTM: gradient can stay constant (when forget gate ≈ 1)
    lstm_gradient = np.ones_like(timesteps) * 0.8  # Simplified - stays roughly constant

    ax2.semilogy(timesteps, rnn_gradient, 'r-', linewidth=2, label='RNN gradient')
    ax2.semilogy(timesteps, lstm_gradient, 'b-', linewidth=2, label='LSTM gradient (cell state path)')
    ax2.fill_between(timesteps, rnn_gradient, alpha=0.3, color='red')
    ax2.fill_between(timesteps, lstm_gradient, alpha=0.3, color='blue')
    ax2.set_xlabel('Timesteps Back', fontsize=11)
    ax2.set_ylabel('Gradient Magnitude (log scale)', fontsize=11)
    ax2.set_title('WHY: Gradient Vanishes in RNN\n(∂h_t/∂h_1 = λ^t where λ<1)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ============ Part 3: Memory Retention Test ============
    ax3 = fig.add_subplot(2, 2, 3)

    # Train on delayed pattern with different delays
    delays = [2, 5, 10, 15, 20, 25]
    rnn_memory = []
    lstm_memory = []

    print("Computing memory retention...")
    for delay in delays:
        # Create dataset where class depends on element at position 0
        # but we pad with `delay` more timesteps after
        X_train, X_test, y_train, y_test = create_sequence_dataset(
            n_samples=600, seq_len=delay + 5, pattern='delayed')

        rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
        rnn.fit(X_train, y_train, epochs=80, lr=0.05, verbose=False)
        rnn_memory.append(accuracy(y_test, rnn.predict(X_test)))

        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train, y_train, epochs=80, lr=0.05, verbose=False)
        lstm_memory.append(accuracy(y_test, lstm.predict(X_test)))

    ax3.bar(np.array(delays) - 1.5, rnn_memory, width=3, label='RNN', color='red', alpha=0.7)
    ax3.bar(np.array(delays) + 1.5, lstm_memory, width=3, label='LSTM', color='blue', alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Delay (timesteps to remember)', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('MEMORY TEST: How Far Back Can It Remember?', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.set_ylim(0.4, 1.05)

    # ============ Part 4: Explanation ============
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    explanation = """
    THE VANISHING GRADIENT PROBLEM
    ══════════════════════════════════════

    RNN backpropagation through time:

        ∂L     ∂L    ∂h_T   ∂h_{T-1}       ∂h_2
       ─── = ─── × ──── × ─────── × ... × ────
       ∂h_1   ∂h_T  ∂h_{T-1} ∂h_{T-2}       ∂h_1

    Each term ∂h_t/∂h_{t-1} involves W_hh.
    If |eigenvalue| < 1 → gradient → 0
    If |eigenvalue| > 1 → gradient → ∞

    LSTM SOLUTION: The Cell State Highway
    ═════════════════════════════════════

        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

    When forget gate f ≈ 1 and input gate i ≈ 0:
        c_t ≈ c_{t-1}  (information preserved!)

    Gradient through cell state:
        ∂c_t/∂c_{t-1} = f_t  (can be ≈ 1)

    → Gradient flows unchanged through time!
    → This is the "gradient highway"
    """

    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('VANISHING GRADIENTS: Why RNN Fails and LSTM Succeeds\n'
                 'RNN gradients decay exponentially, LSTM provides a gradient highway',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_lstm_gates_detailed():
    """
    Detailed LSTM gate visualization showing how gates control information flow.
    """
    np.random.seed(42)

    # Train LSTM on delayed pattern
    X_train, X_test, y_train, y_test = create_sequence_dataset(
        n_samples=600, seq_len=20, pattern='delayed')

    lstm = LSTM(input_size=1, hidden_size=16, output_size=2)
    lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)

    # Get samples from each class
    idx_class0 = np.where(y_test == 0)[0][0]
    idx_class1 = np.where(y_test == 1)[0][0]

    fig = plt.figure(figsize=(16, 10))

    for plot_idx, (idx, class_label) in enumerate([(idx_class0, 'Class 0 (first<0)'),
                                                    (idx_class1, 'Class 1 (first>0)')]):
        sample = X_test[idx:idx+1]
        _ = lstm.forward(sample)

        gates = lstm.cache['gates']
        seq_len = sample.shape[1]

        # Average gates across all hidden units
        forget_gates = np.array([g[0][0].mean() for g in gates])
        input_gates = np.array([g[1][0].mean() for g in gates])
        output_gates = np.array([g[2][0].mean() for g in gates])

        c_states = np.array([c[0].mean() for c in lstm.cache['c_states'][1:]])
        h_states = np.array([h[0].mean() for h in lstm.cache['h_states'][1:]])

        # Plot input
        ax1 = fig.add_subplot(2, 4, plot_idx * 4 + 1)
        ax1.plot(sample[0, :, 0], 'k-o', markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.scatter([0], [sample[0, 0, 0]], color='red' if class_label == 'Class 0 (first<0)' else 'green',
                   s=100, zorder=5, label='First element (decision)')
        ax1.set_title(f'{class_label}\nInput Sequence', fontsize=10)
        ax1.set_xlabel('Time')
        ax1.legend(fontsize=8)

        # Plot gates
        ax2 = fig.add_subplot(2, 4, plot_idx * 4 + 2)
        ax2.plot(forget_gates, 'r-', label='Forget', linewidth=2)
        ax2.plot(input_gates, 'g-', label='Input', linewidth=2)
        ax2.plot(output_gates, 'b-', label='Output', linewidth=2)
        ax2.set_title('Gate Activations (avg)', fontsize=10)
        ax2.set_xlabel('Time')
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # Plot cell state
        ax3 = fig.add_subplot(2, 4, plot_idx * 4 + 3)
        ax3.plot(c_states, 'm-', linewidth=2)
        ax3.set_title('Cell State (memory)', fontsize=10)
        ax3.set_xlabel('Time')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        # Plot hidden state
        ax4 = fig.add_subplot(2, 4, plot_idx * 4 + 4)
        ax4.plot(h_states, 'c-', linewidth=2)
        ax4.set_title('Hidden State (output)', fontsize=10)
        ax4.set_xlabel('Time')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle('LSTM Gate Dynamics: How Memory is Controlled\n'
                 'Red dashed line = first timestep (where the decision info is)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_rnn_vs_lstm_training():
    """
    Show training dynamics: RNN vs LSTM learning curves on long sequences.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Long sequence delayed pattern
    X_train, X_test, y_train, y_test = create_sequence_dataset(
        n_samples=600, seq_len=30, pattern='delayed')

    # Train RNN
    rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
    rnn_losses = rnn.fit(X_train, y_train, epochs=150, lr=0.05, verbose=False)

    # Train LSTM
    lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
    lstm_losses = lstm.fit(X_train, y_train, epochs=150, lr=0.05, verbose=False)

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(rnn_losses, 'r-', label='RNN', linewidth=2)
    ax1.plot(lstm_losses, 'b-', label='LSTM', linewidth=2)
    ax1.axhline(y=np.log(2), color='gray', linestyle='--', alpha=0.5, label='Random (log(2))')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss (seq_len=30, delayed pattern)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training accuracy over time
    ax2 = axes[1]

    # Recompute with accuracy tracking
    rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
    lstm = LSTM(input_size=1, hidden_size=32, output_size=2)

    rnn_accs = []
    lstm_accs = []

    for epoch in range(150):
        # RNN
        h_final, _ = rnn.forward(X_train)
        logits = rnn.output(h_final)
        rnn.backward(y_train, lr=0.05)
        rnn_accs.append(accuracy(y_test, rnn.predict(X_test)))

        # LSTM
        h_final = lstm.forward(X_train)
        logits = lstm.output(h_final)
        lstm.backward(y_train, lr=0.05)
        lstm_accs.append(accuracy(y_test, lstm.predict(X_test)))

    ax2.plot(rnn_accs, 'r-', label='RNN', linewidth=2)
    ax2.plot(lstm_accs, 'b-', label='LSTM', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title('Learning Dynamics: RNN Stuck, LSTM Learns', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.05)

    # Plot 3: Final comparison
    ax3 = axes[2]

    rnn_final = rnn_accs[-1]
    lstm_final = lstm_accs[-1]

    bars = ax3.bar(['RNN', 'LSTM'], [rnn_final, lstm_final],
                   color=['red', 'blue'], alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Final Test Accuracy', fontsize=11)
    ax3.set_title('Final Performance\n(Long-range dependency)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, [rnn_final, lstm_final]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

    # Add annotation
    if rnn_final < 0.6:
        ax3.text(0, rnn_final + 0.1, 'FAILS!', ha='center', color='red', fontsize=11, fontweight='bold')
    if lstm_final > 0.8:
        ax3.text(1, lstm_final + 0.05, 'SUCCESS!', ha='center', color='blue', fontsize=11, fontweight='bold')

    plt.suptitle('RNN vs LSTM: Training on Long-Range Dependencies\n'
                 'Task: Classify sequence by its FIRST element (must remember 30 steps)',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig


def benchmark_patterns():
    """Benchmark on different sequence patterns."""
    print("\n" + "="*60)
    print("BENCHMARK: RNN vs LSTM")
    print("="*60)

    patterns = ['sum', 'first_last', 'xor_first_last', 'majority', 'delayed']
    results = {'RNN': {}, 'LSTM': {}}

    print(f"\n{'Pattern':<18} {'RNN':<10} {'LSTM':<10}")
    print("-" * 38)

    for pattern in patterns:
        X_train, X_test, y_train, y_test = create_sequence_dataset(
            n_samples=800, seq_len=20, pattern=pattern)

        rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=2)
        rnn.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        rnn_acc = accuracy(y_test, rnn.predict(X_test))
        results['RNN'][pattern] = rnn_acc

        lstm = LSTM(input_size=1, hidden_size=32, output_size=2)
        lstm.fit(X_train, y_train, epochs=100, lr=0.05, verbose=False)
        lstm_acc = accuracy(y_test, lstm.predict(X_test))
        results['LSTM'][pattern] = lstm_acc

        print(f"{pattern:<18} {rnn_acc:<10.3f} {lstm_acc:<10.3f}")

    return results


if __name__ == '__main__':
    print("="*60)
    print("RNN & LSTM — Sequential Memory")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Process sequences with a HIDDEN STATE that accumulates information.
    h_t = f(h_{t-1}, x_t)  — current state depends on previous + input

THE VANISHING GRADIENT PROBLEM:
    ∂h_t/∂h_1 = product of many terms
    If terms < 1 → gradient vanishes
    If terms > 1 → gradient explodes
    RNNs struggle with LONG sequences!

LSTM SOLUTION: GATES
    Forget gate: what to forget from cell state
    Input gate: what new information to add
    Output gate: what to output

    Cell state can flow unchanged → "gradient highway"

WEIGHT SHARING:
    Same weights at every time step (like CNN in space)
    This is the sequential inductive bias.
    """)

    ablation_experiments()
    results = benchmark_patterns()

    fig = visualize_gate_activations()
    save_path = '/Users/sid47/ML Algorithms/14_rnn_lstm.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. RNN: h_t = tanh(W_xh x_t + W_hh h_{t-1})
2. Vanishing gradients: RNN fails on long sequences
3. LSTM: gates control information flow
4. Forget gate f ≈ 1 → preserve memory
5. Cell state = gradient highway through time
6. Weight sharing in TIME (like CNN in space)
    """)
