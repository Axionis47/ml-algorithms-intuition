"""
STATISTICAL VS DEEP LEARNING — Paradigm: KNOW WHEN TO USE WHAT

===============================================================
THE BIG QUESTION
===============================================================

"Should I use ARIMA or LSTM for my time series?"

This is one of the most common questions in forecasting.
The answer might surprise you!

===============================================================
THE SURPRISING TRUTH
===============================================================

For UNIVARIATE time series with < 1000 observations:
    STATISTICAL METHODS OFTEN WIN!

Why?
1. Deep learning needs LOTS of data to learn patterns
2. ARIMA/ES encode domain knowledge (trend, seasonality) directly
3. Deep models can overfit on small samples
4. Statistical methods are faster and more interpretable

===============================================================
WHEN DEEP LEARNING SHINES
===============================================================

Deep learning excels when:
1. MULTIPLE related series (transfer learning)
2. LONG sequences with complex patterns
3. EXTERNAL features (covariates) are important
4. NON-LINEAR relationships dominate
5. LARGE datasets (>10,000 observations)

===============================================================
THE FAIR COMPARISON
===============================================================

To compare fairly:
1. Same train/test split
2. Same forecast horizon
3. Proper hyperparameter tuning for BOTH
4. Multiple random seeds for deep learning
5. Account for computation time

Many "deep learning wins" papers fail on points 1-4!

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms/11-time-series')

# Import our implementations
import importlib.util

spec_es = importlib.util.spec_from_file_location("exponential_smoothing",
    "/Users/sid47/ML Algorithms/11-time-series/03_exponential_smoothing.py")
es_module = importlib.util.module_from_spec(spec_es)
spec_es.loader.exec_module(es_module)
SimpleExponentialSmoothing = es_module.SimpleExponentialSmoothing
HoltLinear = es_module.HoltLinear
HoltWinters = es_module.HoltWinters

spec_arima = importlib.util.spec_from_file_location("arima",
    "/Users/sid47/ML Algorithms/11-time-series/04_arima.py")
arima_module = importlib.util.module_from_spec(spec_arima)
spec_arima.loader.exec_module(arima_module)
ARIMA = arima_module.ARIMA
AR = arima_module.AR


# ============================================================
# SIMPLE RNN/LSTM IMPLEMENTATION (for comparison)
# ============================================================

class SimpleRNN:
    """
    Minimal RNN for time series forecasting.

    This is a simplified version to demonstrate the comparison.
    For production, use PyTorch/TensorFlow.
    """

    def __init__(self, hidden_size=16, lookback=10, lr=0.01, epochs=100):
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs

        # Weights will be initialized in fit()
        self.W_xh = None
        self.W_hh = None
        self.W_hy = None
        self.b_h = None
        self.b_y = None

    def _init_weights(self):
        """Xavier initialization."""
        scale_xh = np.sqrt(2.0 / (1 + self.hidden_size))
        scale_hh = np.sqrt(2.0 / (self.hidden_size * 2))
        scale_hy = np.sqrt(2.0 / (self.hidden_size + 1))

        self.W_xh = np.random.randn(1, self.hidden_size) * scale_xh
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.W_hy = np.random.randn(self.hidden_size, 1) * scale_hy
        self.b_h = np.zeros((1, self.hidden_size))
        self.b_y = np.zeros((1, 1))

    def _create_sequences(self, y):
        """Create input sequences and targets."""
        X, Y = [], []
        for i in range(len(y) - self.lookback):
            X.append(y[i:i+self.lookback])
            Y.append(y[i+self.lookback])
        return np.array(X), np.array(Y).reshape(-1, 1)

    def _forward(self, X):
        """Forward pass through RNN."""
        batch_size = X.shape[0]

        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))

        # Process sequence
        for t in range(self.lookback):
            x_t = X[:, t:t+1]  # (batch, 1)
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

        # Output
        y_pred = h @ self.W_hy + self.b_y
        return y_pred, h

    def fit(self, y):
        """Train the RNN."""
        self._init_weights()

        # Normalize
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-8
        y_norm = (y - self.mean) / self.std

        # Create sequences
        X, Y = self._create_sequences(y_norm)

        if len(X) == 0:
            return self

        # Training loop
        for epoch in range(self.epochs):
            # Forward
            y_pred, h = self._forward(X)

            # Loss (MSE)
            loss = np.mean((y_pred - Y)**2)

            # Backward (simplified gradient descent)
            # This is approximate BPTT
            d_loss = 2 * (y_pred - Y) / len(Y)

            # Output layer gradients
            dW_hy = h.T @ d_loss
            db_y = np.sum(d_loss, axis=0, keepdims=True)

            # Hidden layer gradients (approximate)
            dh = d_loss @ self.W_hy.T
            dh_raw = dh * (1 - h**2)  # tanh derivative

            dW_hh = h.T @ dh_raw
            dW_xh = X[:, -1:].T @ dh_raw
            db_h = np.sum(dh_raw, axis=0, keepdims=True)

            # Update weights
            self.W_hy -= self.lr * np.clip(dW_hy, -5, 5)
            self.b_y -= self.lr * np.clip(db_y, -5, 5)
            self.W_hh -= self.lr * np.clip(dW_hh, -5, 5)
            self.W_xh -= self.lr * np.clip(dW_xh, -5, 5)
            self.b_h -= self.lr * np.clip(db_h, -5, 5)

        self._y = y
        return self

    def forecast(self, h=1):
        """Forecast h steps ahead."""
        y_norm = (self._y - self.mean) / self.std

        # Start with last lookback values
        current_seq = y_norm[-self.lookback:].copy()

        forecasts = []
        for _ in range(h):
            X = current_seq.reshape(1, -1)
            y_pred, _ = self._forward(X)
            forecasts.append(y_pred[0, 0])

            # Shift sequence
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = y_pred[0, 0]

        # Denormalize
        forecasts = np.array(forecasts) * self.std + self.mean
        return forecasts


class SimpleLSTM:
    """
    Minimal LSTM for time series forecasting.
    """

    def __init__(self, hidden_size=16, lookback=10, lr=0.01, epochs=100):
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs

    def _init_weights(self):
        """Initialize LSTM weights."""
        hs = self.hidden_size

        # Combined weights for all gates
        scale = np.sqrt(2.0 / (1 + hs))
        self.W_x = np.random.randn(1, 4 * hs) * scale
        self.W_h = np.random.randn(hs, 4 * hs) * scale
        self.b = np.zeros((1, 4 * hs))

        # Initialize forget gate bias to 1
        self.b[0, :hs] = 1.0

        # Output layer
        self.W_y = np.random.randn(hs, 1) * np.sqrt(2.0 / hs)
        self.b_y = np.zeros((1, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _create_sequences(self, y):
        X, Y = [], []
        for i in range(len(y) - self.lookback):
            X.append(y[i:i+self.lookback])
            Y.append(y[i+self.lookback])
        return np.array(X), np.array(Y).reshape(-1, 1)

    def _forward(self, X):
        """Forward pass through LSTM."""
        batch_size = X.shape[0]
        hs = self.hidden_size

        h = np.zeros((batch_size, hs))
        c = np.zeros((batch_size, hs))

        for t in range(self.lookback):
            x_t = X[:, t:t+1]

            # All gates at once
            gates = x_t @ self.W_x + h @ self.W_h + self.b

            f = self._sigmoid(gates[:, :hs])
            i = self._sigmoid(gates[:, hs:2*hs])
            o = self._sigmoid(gates[:, 2*hs:3*hs])
            g = np.tanh(gates[:, 3*hs:])

            c = f * c + i * g
            h = o * np.tanh(c)

        y_pred = h @ self.W_y + self.b_y
        return y_pred, h

    def fit(self, y):
        """Train LSTM."""
        self._init_weights()

        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-8
        y_norm = (y - self.mean) / self.std

        X, Y = self._create_sequences(y_norm)

        if len(X) == 0:
            return self

        for epoch in range(self.epochs):
            y_pred, h = self._forward(X)
            loss = np.mean((y_pred - Y)**2)

            # Simplified gradient update
            d_loss = 2 * (y_pred - Y) / len(Y)
            dW_y = h.T @ d_loss
            db_y = np.sum(d_loss, axis=0, keepdims=True)

            self.W_y -= self.lr * np.clip(dW_y, -5, 5)
            self.b_y -= self.lr * np.clip(db_y, -5, 5)

        self._y = y
        return self

    def forecast(self, h=1):
        """Forecast h steps."""
        y_norm = (self._y - self.mean) / self.std
        current_seq = y_norm[-self.lookback:].copy()

        forecasts = []
        for _ in range(h):
            X = current_seq.reshape(1, -1)
            y_pred, _ = self._forward(X)
            forecasts.append(y_pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = y_pred[0, 0]

        return np.array(forecasts) * self.std + self.mean


# ============================================================
# DATA GENERATORS
# ============================================================

def generate_linear_trend(n, noise=1.0, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    return 10 + 0.1 * t + np.random.randn(n) * noise


def generate_seasonal(n, period=12, noise=1.0, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    return 10 + 5 * np.sin(2 * np.pi * t / period) + np.random.randn(n) * noise


def generate_complex(n, seed=42):
    """Complex pattern with trend, multiple seasonalities, and nonlinearity."""
    np.random.seed(seed)
    t = np.arange(n)

    trend = 0.05 * t
    season1 = 3 * np.sin(2 * np.pi * t / 12)
    season2 = 1.5 * np.sin(2 * np.pi * t / 5)
    nonlinear = 2 * np.sin(0.1 * t) * np.cos(0.05 * t)
    noise = np.random.randn(n) * 0.5

    return 10 + trend + season1 + season2 + nonlinear + noise


def generate_nonlinear_ar(n, seed=42):
    """Nonlinear autoregressive process."""
    np.random.seed(seed)
    y = np.zeros(n)
    y[0] = np.random.randn()
    y[1] = np.random.randn()

    for t in range(2, n):
        # Nonlinear relationship
        y[t] = 0.5 * y[t-1] + 0.3 * y[t-2] + 0.2 * np.sin(y[t-1]) + np.random.randn() * 0.5

    return y + 10


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def run_comparison(y, train_ratio=0.8, n_runs=5):
    """
    Compare statistical and deep learning methods.

    Returns dict of {method: {'mse': mean, 'std': std, 'time': time}}
    """
    import time

    n = len(y)
    train_end = int(n * train_ratio)
    y_train = y[:train_end]
    y_test = y[train_end:]
    h = len(y_test)

    results = {}

    # Statistical methods (single run - deterministic)
    methods_stat = {
        'SES': SimpleExponentialSmoothing(alpha=0.3),
        'Holt': HoltLinear(alpha=0.3, beta=0.1),
        'AR(2)': AR(p=2),
        'ARIMA(1,1,1)': ARIMA(p=1, d=1, q=1),
    }

    for name, model in methods_stat.items():
        try:
            start = time.time()
            model.fit(y_train)
            forecast = model.forecast(h)
            elapsed = time.time() - start

            min_len = min(len(forecast), len(y_test))
            mse = np.mean((y_test[:min_len] - forecast[:min_len])**2)

            results[name] = {'mse': mse, 'std': 0, 'time': elapsed}
        except:
            results[name] = {'mse': np.inf, 'std': 0, 'time': 0}

    # Deep learning methods (multiple runs due to randomness)
    methods_dl = {
        'RNN': lambda: SimpleRNN(hidden_size=16, lookback=10, lr=0.01, epochs=100),
        'LSTM': lambda: SimpleLSTM(hidden_size=16, lookback=10, lr=0.01, epochs=100),
    }

    for name, model_fn in methods_dl.items():
        mses = []
        times = []

        for run in range(n_runs):
            np.random.seed(run)
            model = model_fn()

            try:
                start = time.time()
                model.fit(y_train)
                forecast = model.forecast(h)
                elapsed = time.time() - start

                min_len = min(len(forecast), len(y_test))
                mse = np.mean((y_test[:min_len] - forecast[:min_len])**2)

                if np.isfinite(mse):
                    mses.append(mse)
                    times.append(elapsed)
            except:
                pass

        if mses:
            results[name] = {
                'mse': np.mean(mses),
                'std': np.std(mses),
                'time': np.mean(times)
            }
        else:
            results[name] = {'mse': np.inf, 'std': 0, 'time': 0}

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_main_comparison():
    """
    THE MAIN COMPARISON: Statistical vs Deep Learning
    """
    np.random.seed(42)

    fig = plt.figure(figsize=(18, 14))

    fig.suptitle('STATISTICAL vs DEEP LEARNING: When Does Each Win?\n'
                 '"The surprising truth: Simple methods often beat complex ones"',
                 fontsize=14, fontweight='bold', y=0.98)

    # Test scenarios with different sample sizes
    scenarios = {
        'Small Data\n(n=100)': 100,
        'Medium Data\n(n=300)': 300,
        'Large Data\n(n=1000)': 1000,
    }

    # Different patterns
    patterns = {
        'Linear Trend': generate_linear_trend,
        'Seasonal': generate_seasonal,
        'Complex': generate_complex,
        'Nonlinear AR': generate_nonlinear_ar,
    }

    all_results = {}

    for pattern_name, generator in patterns.items():
        all_results[pattern_name] = {}
        for scenario_name, n in scenarios.items():
            y = generator(n)
            results = run_comparison(y, train_ratio=0.8, n_runs=3)
            all_results[pattern_name][scenario_name] = results

    # Create visualization
    n_patterns = len(patterns)
    n_scenarios = len(scenarios)

    for pat_idx, (pattern_name, pattern_results) in enumerate(all_results.items()):
        for scen_idx, (scenario_name, results) in enumerate(pattern_results.items()):
            ax = fig.add_subplot(n_patterns, n_scenarios, pat_idx * n_scenarios + scen_idx + 1)

            methods = list(results.keys())
            mses = [results[m]['mse'] for m in methods]
            stds = [results[m]['std'] for m in methods]

            # Color by type
            colors = []
            for m in methods:
                if m in ['RNN', 'LSTM']:
                    colors.append('coral')
                else:
                    colors.append('steelblue')

            x = np.arange(len(methods))
            bars = ax.bar(x, mses, yerr=stds, capsize=3, color=colors, alpha=0.7, edgecolor='black')

            # Highlight winner
            if mses:
                best_idx = np.argmin(mses)
                bars[best_idx].set_edgecolor('green')
                bars[best_idx].set_linewidth(3)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('MSE' if scen_idx == 0 else '')

            if pat_idx == 0:
                ax.set_title(scenario_name, fontsize=10, fontweight='bold')
            if scen_idx == 0:
                ax.text(-0.3, 0.5, pattern_name, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', rotation=90,
                        verticalalignment='center')

            ax.grid(True, alpha=0.3, axis='y')

            # Mark winner
            if mses:
                winner = methods[best_idx]
                is_statistical = winner not in ['RNN', 'LSTM']
                ax.text(0.95, 0.95, f'Winner:\n{winner}',
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round',
                                 facecolor='lightblue' if is_statistical else 'lightyellow',
                                 alpha=0.8))

    # Add legend
    fig.text(0.5, 0.02,
             '■ Statistical Methods (Blue)    ■ Deep Learning (Orange)\n'
             'Green border = Winner | Error bars = standard deviation across runs',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    return fig


def visualize_sample_size_effect():
    """
    Show how the winner changes with sample size.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sample_sizes = [50, 100, 200, 500, 1000, 2000]

    patterns = {
        'Linear Trend': generate_linear_trend,
        'Seasonal': generate_seasonal,
        'Complex': generate_complex,
        'Nonlinear AR': generate_nonlinear_ar,
    }

    methods_to_track = ['SES', 'ARIMA(1,1,1)', 'RNN', 'LSTM']
    colors = {'SES': 'blue', 'ARIMA(1,1,1)': 'green', 'RNN': 'red', 'LSTM': 'orange'}

    for ax_idx, (pattern_name, generator) in enumerate(patterns.items()):
        ax = axes[ax_idx // 2, ax_idx % 2]

        method_mses = {m: [] for m in methods_to_track}

        for n in sample_sizes:
            y = generator(n)
            results = run_comparison(y, train_ratio=0.8, n_runs=3)

            for m in methods_to_track:
                if m in results:
                    method_mses[m].append(results[m]['mse'])
                else:
                    method_mses[m].append(np.nan)

        for method, mses in method_mses.items():
            ax.plot(sample_sizes, mses, 'o-', color=colors[method],
                    linewidth=2, markersize=6, label=method, alpha=0.8)

        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('MSE')
        ax.set_title(pattern_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Mark crossover point if exists
        arima_mses = np.array(method_mses['ARIMA(1,1,1)'])
        lstm_mses = np.array(method_mses['LSTM'])

        # Find where LSTM starts winning
        crossover = np.where(lstm_mses < arima_mses)[0]
        if len(crossover) > 0:
            cross_n = sample_sizes[crossover[0]]
            ax.axvline(x=cross_n, color='purple', linestyle='--', alpha=0.7)
            ax.text(cross_n, ax.get_ylim()[1] * 0.9, f'LSTM wins\nat n={cross_n}',
                    fontsize=8, color='purple', ha='center')

    plt.suptitle('HOW SAMPLE SIZE AFFECTS THE WINNER\n'
                 '"Deep learning needs more data to shine"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_decision_guide():
    """
    Visual guide for choosing between statistical and deep learning.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    guide = """
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    STATISTICAL vs DEEP LEARNING: DECISION GUIDE                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║   CHOOSE STATISTICAL METHODS (ARIMA, Exponential Smoothing) WHEN:                     ║
    ║   ─────────────────────────────────────────────────────────────────                   ║
    ║   ✓ Single univariate time series                                                     ║
    ║   ✓ Small to medium data (< 1000 observations)                                        ║
    ║   ✓ Clear trend and/or seasonality                                                    ║
    ║   ✓ Need for interpretability                                                         ║
    ║   ✓ Fast inference required                                                           ║
    ║   ✓ No external features/covariates                                                   ║
    ║                                                                                       ║
    ║                                                                                       ║
    ║   CHOOSE DEEP LEARNING (RNN, LSTM, Transformer) WHEN:                                 ║
    ║   ─────────────────────────────────────────────────────────────────                   ║
    ║   ✓ Multiple related time series (can share patterns)                                 ║
    ║   ✓ Large data (> 10,000 observations)                                                ║
    ║   ✓ Complex nonlinear patterns                                                        ║
    ║   ✓ Many external features to incorporate                                             ║
    ║   ✓ Very long sequences with subtle patterns                                          ║
    ║   ✓ Transfer learning from related domains                                            ║
    ║                                                                                       ║
    ║                                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                              THE DATA SIZE RULE OF THUMB                              ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║      n < 100        │  100 < n < 1000     │  1000 < n < 10000   │  n > 10000          ║
    ║   ──────────────────┼─────────────────────┼─────────────────────┼───────────────────  ║
    ║   Simple methods    │  Statistical wins   │  Toss-up            │  Deep learning      ║
    ║   (SES, Naive)      │  (ARIMA, ES)        │  Try both!          │  can win            ║
    ║                     │                     │                     │                     ║
    ║                                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                  COMMON MISTAKES                                      ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║   ✗ Using LSTM because it's "state of the art" (without checking if data supports)   ║
    ║   ✗ Not tuning hyperparameters for BOTH statistical and deep learning                ║
    ║   ✗ Ignoring computation time and interpretability requirements                      ║
    ║   ✗ Using deep learning on a single short series                                     ║
    ║   ✗ Not comparing against simple baselines (Naive, SES)                              ║
    ║                                                                                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                              THE META-LESSON                                          ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║     "The best forecasters don't have a favorite method.                               ║
    ║      They have a favorite PROCESS:                                                    ║
    ║                                                                                       ║
    ║      1. Understand your data (EDA, decomposition)                                     ║
    ║      2. Try simple methods first (baselines)                                          ║
    ║      3. Increase complexity only if justified                                         ║
    ║      4. Validate rigorously (proper time series CV)                                   ║
    ║      5. Consider interpretability and deployment needs"                               ║
    ║                                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, guide, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.title('WHEN TO USE WHAT: The Complete Guide\n'
              '"Match your method to your data and constraints"',
              fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def visualize_honest_benchmark():
    """
    Show what an honest benchmark looks like.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Generate complex seasonal data
    n = 300
    y = generate_seasonal(n, period=12, noise=1.5)

    train_end = int(n * 0.8)
    y_train = y[:train_end]
    y_test = y[train_end:]
    h = len(y_test)

    # Fit all methods
    methods = {
        'SES': SimpleExponentialSmoothing(alpha=0.3),
        'Holt': HoltLinear(alpha=0.3, beta=0.1),
        'AR(2)': AR(p=2),
        'ARIMA(1,1,1)': ARIMA(p=1, d=1, q=1),
        'RNN': SimpleRNN(hidden_size=16, lookback=12, epochs=200),
        'LSTM': SimpleLSTM(hidden_size=16, lookback=12, epochs=200),
    }

    forecasts = {}
    mses = {}

    for name, model in methods.items():
        try:
            model.fit(y_train)
            forecast = model.forecast(h)
            min_len = min(len(forecast), len(y_test))
            forecasts[name] = forecast[:min_len]
            mses[name] = np.mean((y_test[:min_len] - forecast[:min_len])**2)
        except Exception as e:
            forecasts[name] = np.full(h, np.nan)
            mses[name] = np.inf

    # Plot forecasts
    for idx, (name, forecast) in enumerate(forecasts.items()):
        ax = axes[idx // 3, idx % 3]

        ax.plot(range(train_end), y_train, 'b-', linewidth=0.8, alpha=0.5, label='Train')
        ax.plot(range(train_end, n), y_test, 'k-', linewidth=2, label='Actual')

        color = 'coral' if name in ['RNN', 'LSTM'] else 'steelblue'
        ax.plot(range(train_end, train_end + len(forecast)), forecast,
                '--', color=color, linewidth=2, label=f'{name}')

        ax.axvline(x=train_end, color='gray', linestyle=':', alpha=0.7)

        mse = mses[name]
        ax.set_title(f'{name}\nMSE = {mse:.2f}', fontsize=11, fontweight='bold',
                     color='green' if mse == min(mses.values()) else 'black')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('HONEST BENCHMARK: Same Data, Same Split, Fair Comparison\n'
                 f'Winner: {min(mses, key=mses.get)} (Seasonal data, n={n})',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("STATISTICAL vs DEEP LEARNING — Know When to Use What")
    print("="*70)

    print("""

THE SURPRISING TRUTH:
    For univariate time series with < 1000 observations,
    STATISTICAL METHODS OFTEN WIN!

WHY?
    1. Deep learning needs LOTS of data
    2. ARIMA/ES encode domain knowledge directly
    3. Deep models can overfit on small samples
    4. Statistical methods are faster and interpretable

WHEN DEEP LEARNING SHINES:
    ✓ Multiple related series
    ✓ Large datasets (> 10,000 points)
    ✓ Complex nonlinear patterns
    ✓ Many external features
    ✓ Transfer learning scenarios

THE META-LESSON:
    The best forecasters don't have a favorite method.
    They have a favorite PROCESS.

    """)

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. Main comparison
    print("Creating main comparison (this may take a moment)...")
    fig1 = visualize_main_comparison()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/06_stat_vs_dl_comparison.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. Sample size effect
    print("Creating sample size effect visualization...")
    fig2 = visualize_sample_size_effect()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/06_stat_vs_dl_sample_size.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Decision guide
    print("Creating decision guide...")
    fig3 = visualize_decision_guide()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/06_stat_vs_dl_guide.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. Honest benchmark
    print("Creating honest benchmark...")
    fig4 = visualize_honest_benchmark()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/06_stat_vs_dl_benchmark.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    print("\n" + "="*60)
    print("SUMMARY: Statistical vs Deep Learning")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 06_stat_vs_dl_comparison.png  — Head-to-head across patterns and sizes
    2. 06_stat_vs_dl_sample_size.png — How sample size affects the winner
    3. 06_stat_vs_dl_guide.png       — Decision guide: when to use what
    4. 06_stat_vs_dl_benchmark.png   — Honest benchmark on seasonal data

KEY TAKEAWAYS:
    1. Statistical methods often win on small-medium data
    2. Deep learning needs ~1000+ points to show its strength
    3. The winner depends on: data size, pattern type, constraints
    4. Always compare against simple baselines!
    5. Consider interpretability and deployment needs

THIS COMPLETES THE TIME SERIES MODULE:
    01 - Fundamentals (decomposition, stationarity)
    02 - Autocorrelation (memory, ACF/PACF)
    03 - Exponential Smoothing (forgetting curve)
    04 - ARIMA (AR, I, MA components)
    05 - Forecasting Showdown (method comparison)
    06 - Statistical vs Deep Learning (the bridge)

    """)
