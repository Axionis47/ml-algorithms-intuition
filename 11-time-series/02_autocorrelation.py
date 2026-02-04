"""
AUTOCORRELATION — Paradigm: MEMORY IN TIME

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Autocorrelation measures how much a time series is correlated with
ITSELF at different time lags.

    ACF(k) = Corr(y_t, y_{t-k})

"How much does knowing y at time t-k tell us about y at time t?"

===============================================================
THE MEMORY PROBLEM
===============================================================

This is THE fundamental question in time series forecasting:

    HOW FAR BACK SHOULD YOU LOOK?

- Look back 1 step: Maybe not enough information
- Look back 100 steps: Maybe too noisy, computationally expensive
- Look back ∞ steps: Impossible

ACF ANSWERS THIS QUESTION:
    - High ACF at lag k → information at t-k is useful
    - ACF ≈ 0 at lag k → information at t-k is useless
    - ACF shows periodic peaks → seasonality exists

===============================================================
ACF vs PACF
===============================================================

ACF (Autocorrelation Function):
    Total correlation between y_t and y_{t-k}
    INCLUDES indirect effects through intermediate lags

PACF (Partial Autocorrelation Function):
    DIRECT correlation between y_t and y_{t-k}
    REMOVES the effect of intermediate lags

EXAMPLE:
    If y_t correlates with y_{t-1}, and y_{t-1} correlates with y_{t-2},
    then y_t will correlate with y_{t-2} (through y_{t-1}).

    ACF(2) captures this total correlation.
    PACF(2) captures only the DIRECT link, controlling for y_{t-1}.

===============================================================
INTERPRETING ACF/PACF FOR MODEL SELECTION
===============================================================

AR(p) process: y_t = φ₁y_{t-1} + ... + φₚy_{t-p} + ε
    - ACF: Decays exponentially (or oscillates and decays)
    - PACF: Cuts off after lag p (spikes at lags 1,...,p, then ≈0)

MA(q) process: y_t = ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}
    - ACF: Cuts off after lag q
    - PACF: Decays exponentially

ARMA(p,q): Both decay (neither cuts off sharply)

===============================================================
THE SIGNIFICANCE BANDS
===============================================================

Under the null hypothesis of white noise:
    ACF(k) ~ N(0, 1/n) approximately

So 95% confidence bands are ± 1.96/√n

If ACF exceeds these bands → significant autocorrelation exists
If all ACF within bands → series is effectively white noise

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms/11-time-series')


# ============================================================
# AUTOCORRELATION FUNCTIONS
# ============================================================

def acf(y, max_lag=None):
    """
    Compute the Autocorrelation Function.

    ACF(k) = Cov(y_t, y_{t-k}) / Var(y)

    This is the sample autocorrelation.
    """
    n = len(y)
    if max_lag is None:
        max_lag = min(n // 4, 40)

    y_centered = y - np.mean(y)
    var_y = np.var(y)

    acf_values = np.zeros(max_lag + 1)
    acf_values[0] = 1.0  # ACF at lag 0 is always 1

    for k in range(1, max_lag + 1):
        acf_values[k] = np.mean(y_centered[k:] * y_centered[:-k]) / var_y

    return acf_values


def pacf(y, max_lag=None):
    """
    Compute the Partial Autocorrelation Function using Durbin-Levinson recursion.

    PACF(k) = Corr(y_t, y_{t-k} | y_{t-1}, ..., y_{t-k+1})

    This gives the DIRECT correlation, removing intermediate effects.
    """
    n = len(y)
    if max_lag is None:
        max_lag = min(n // 4, 40)

    # First compute ACF
    r = acf(y, max_lag)

    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0
    pacf_values[1] = r[1]

    # Durbin-Levinson recursion
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = r[1]

    for k in range(2, max_lag + 1):
        # Compute phi[k,k]
        numerator = r[k] - np.sum(phi[k-1, 1:k] * r[1:k][::-1])
        denominator = 1 - np.sum(phi[k-1, 1:k] * r[1:k])

        if abs(denominator) < 1e-10:
            pacf_values[k] = 0
        else:
            phi[k, k] = numerator / denominator
            pacf_values[k] = phi[k, k]

        # Update other coefficients
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

    return pacf_values


def significance_bands(n, alpha=0.05):
    """
    Compute significance bands for ACF/PACF.

    Under H0 (white noise), ACF(k) ~ N(0, 1/n) for large n.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return z / np.sqrt(n)


# ============================================================
# SYNTHETIC PROCESSES FOR DEMONSTRATION
# ============================================================

def generate_ar_process(n, phi, sigma=1.0, seed=42):
    """
    Generate AR(p) process.

    y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t

    phi: list of AR coefficients [φ₁, φ₂, ...]
    """
    np.random.seed(seed)
    p = len(phi)
    y = np.zeros(n)
    y[:p] = np.random.randn(p) * sigma

    for t in range(p, n):
        y[t] = np.dot(phi, y[t-p:t][::-1]) + np.random.randn() * sigma

    return y


def generate_ma_process(n, theta, sigma=1.0, seed=42):
    """
    Generate MA(q) process.

    y_t = ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θqε_{t-q}

    theta: list of MA coefficients [θ₁, θ₂, ...]
    """
    np.random.seed(seed)
    q = len(theta)
    eps = np.random.randn(n + q) * sigma

    y = np.zeros(n)
    for t in range(n):
        y[t] = eps[t + q] + np.dot(theta, eps[t:t+q][::-1])

    return y


def generate_arma_process(n, phi, theta, sigma=1.0, seed=42):
    """
    Generate ARMA(p,q) process.

    y_t = φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}
    """
    np.random.seed(seed)
    p = len(phi) if phi else 0
    q = len(theta) if theta else 0
    max_pq = max(p, q)

    eps = np.random.randn(n + max_pq) * sigma
    y = np.zeros(n + max_pq)

    for t in range(max_pq, n + max_pq):
        ar_part = np.dot(phi, y[t-p:t][::-1]) if p > 0 else 0
        ma_part = np.dot(theta, eps[t-q:t][::-1]) if q > 0 else 0
        y[t] = ar_part + eps[t] + ma_part

    return y[max_pq:]


def generate_seasonal_process(n, period=12, ar_coef=0.7, seasonal_ar_coef=0.5, sigma=1.0, seed=42):
    """
    Generate seasonal AR process.

    y_t = φ₁y_{t-1} + Φ₁y_{t-s} + ε_t

    where s is the seasonal period.
    """
    np.random.seed(seed)
    y = np.zeros(n)
    y[:period] = np.random.randn(period) * sigma

    for t in range(period, n):
        y[t] = (ar_coef * y[t-1] +
                seasonal_ar_coef * y[t-period] +
                np.random.randn() * sigma)

    return y


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_memory_problem():
    """
    THE MEMORY PROBLEM VISUALIZATION

    Shows how looking back different amounts changes prediction quality.
    This is the key intuition for why autocorrelation matters.
    """
    np.random.seed(42)

    # Generate an AR(2) process with clear structure
    n = 300
    y = generate_ar_process(n, phi=[0.7, 0.2], sigma=1.0)

    fig = plt.figure(figsize=(16, 12))

    fig.suptitle('THE MEMORY PROBLEM: How Far Back Should You Look?\n'
                 '"Autocorrelation tells you which lags carry information"',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: Scatter plots of y_t vs y_{t-k} for different k
    lags_to_show = [1, 2, 5, 10, 20]

    for idx, lag in enumerate(lags_to_show):
        ax = fig.add_subplot(3, 5, idx + 1)

        y_current = y[lag:]
        y_lagged = y[:-lag]

        # Compute correlation
        corr = np.corrcoef(y_current, y_lagged)[0, 1]

        ax.scatter(y_lagged, y_current, alpha=0.3, s=20, c='blue')

        # Add regression line
        z = np.polyfit(y_lagged, y_current, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_lagged.min(), y_lagged.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'R={corr:.2f}')

        ax.set_xlabel(f'$y_{{t-{lag}}}$')
        ax.set_ylabel('$y_t$')
        ax.set_title(f'Lag {lag}\nCorr = {corr:.3f}', fontsize=10,
                     fontweight='bold' if abs(corr) > 0.2 else 'normal',
                     color='green' if abs(corr) > 0.2 else 'gray')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: The ACF - answering the memory question
    ax_acf = fig.add_subplot(3, 2, 3)

    acf_values = acf(y, max_lag=30)
    lags = np.arange(len(acf_values))
    band = significance_bands(n)

    ax_acf.bar(lags, acf_values, color='steelblue', alpha=0.7, width=0.8)
    ax_acf.axhline(y=band, color='red', linestyle='--', linewidth=1.5, label='95% significance')
    ax_acf.axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
    ax_acf.axhline(y=0, color='black', linewidth=0.5)

    # Highlight significant lags
    significant_lags = np.where(np.abs(acf_values[1:]) > band)[0] + 1
    for lag in significant_lags[:5]:  # First 5 significant
        ax_acf.bar(lag, acf_values[lag], color='green', alpha=0.8, width=0.8)

    ax_acf.set_xlabel('Lag k', fontsize=11)
    ax_acf.set_ylabel('ACF(k)', fontsize=11)
    ax_acf.set_title('AUTOCORRELATION FUNCTION (ACF)\n'
                     '"Which lags carry information about the future?"',
                     fontsize=11, fontweight='bold')
    ax_acf.legend(fontsize=9)
    ax_acf.set_xlim(-0.5, 30.5)

    # Add insight
    ax_acf.text(0.98, 0.95,
                f'Significant lags: {list(significant_lags[:5])}\n'
                f'These lags help prediction!',
                transform=ax_acf.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Row 2, right: PACF
    ax_pacf = fig.add_subplot(3, 2, 4)

    pacf_values = pacf(y, max_lag=30)

    ax_pacf.bar(lags, pacf_values, color='coral', alpha=0.7, width=0.8)
    ax_pacf.axhline(y=band, color='red', linestyle='--', linewidth=1.5)
    ax_pacf.axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
    ax_pacf.axhline(y=0, color='black', linewidth=0.5)

    # Highlight significant lags
    significant_pacf = np.where(np.abs(pacf_values[1:]) > band)[0] + 1
    for lag in significant_pacf[:5]:
        ax_pacf.bar(lag, pacf_values[lag], color='darkred', alpha=0.8, width=0.8)

    ax_pacf.set_xlabel('Lag k', fontsize=11)
    ax_pacf.set_ylabel('PACF(k)', fontsize=11)
    ax_pacf.set_title('PARTIAL AUTOCORRELATION FUNCTION (PACF)\n'
                      '"Direct effect of each lag (removing intermediate lags)"',
                      fontsize=11, fontweight='bold')
    ax_pacf.set_xlim(-0.5, 30.5)

    # Add interpretation
    ax_pacf.text(0.98, 0.95,
                 f'PACF cuts off after lag ~2\n'
                 f'→ This is likely an AR(2) process!',
                 transform=ax_pacf.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Row 3: Prediction quality vs lag depth
    ax_pred = fig.add_subplot(3, 1, 3)

    # Train simple AR models with different lag depths
    train_size = 200
    test_size = n - train_size

    lag_depths = list(range(1, 21))
    mse_scores = []

    for p in lag_depths:
        # Simple AR(p) using least squares
        try:
            X_train = np.column_stack([y[p-i-1:train_size-i-1] for i in range(p)])
            y_target = y[p:train_size]

            # Ensure dimensions match
            min_len = min(len(X_train), len(y_target))
            X_train = X_train[:min_len]
            y_target = y_target[:min_len]

            coeffs = np.linalg.lstsq(X_train, y_target, rcond=None)[0]

            # Predict on test
            predictions = []
            for t in range(train_size, n):
                x_t = y[t-p:t][::-1]
                pred = np.dot(coeffs, x_t)
                predictions.append(pred)

            mse = np.mean((y[train_size:] - np.array(predictions))**2)
            mse_scores.append(mse)
        except Exception as e:
            mse_scores.append(np.inf)

    ax_pred.plot(lag_depths, mse_scores, 'bo-', linewidth=2, markersize=8)
    # Find best lag (handle inf values)
    mse_array = np.array(mse_scores)
    mse_array[~np.isfinite(mse_array)] = np.inf
    best_idx = np.argmin(mse_array)
    best_lag = lag_depths[best_idx]
    ax_pred.axvline(x=best_lag, color='green', linestyle='--', linewidth=2,
                    label=f'Best lag depth: {best_lag}')
    ax_pred.scatter([best_lag], [np.nanmin(mse_scores)], color='green', s=200,
                    zorder=5, marker='*')

    ax_pred.set_xlabel('Number of Lags Used (Model Complexity)', fontsize=11)
    ax_pred.set_ylabel('Mean Squared Error (Test Set)', fontsize=11)
    ax_pred.set_title('PREDICTION ERROR vs LAG DEPTH\n'
                      '"Using too few or too many lags hurts prediction"',
                      fontsize=11, fontweight='bold')
    ax_pred.legend(fontsize=10)
    ax_pred.grid(True, alpha=0.3)

    # Add insight box
    ax_pred.text(0.98, 0.95,
                 'THE KEY INSIGHT:\n'
                 '• Too few lags → Missing information\n'
                 '• Too many lags → Overfitting to noise\n'
                 '• ACF/PACF guide the right lag depth',
                 transform=ax_pred.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_acf_pacf_patterns():
    """
    Show how ACF/PACF patterns reveal the underlying process.
    This is the key to model identification.
    """
    np.random.seed(42)
    n = 500

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Different process types
    processes = [
        ('AR(1): φ=0.8', generate_ar_process(n, [0.8]), 'AR(1)'),
        ('AR(2): φ=[0.5, 0.3]', generate_ar_process(n, [0.5, 0.3]), 'AR(2)'),
        ('MA(1): θ=0.8', generate_ma_process(n, [0.8]), 'MA(1)'),
        ('MA(2): θ=[0.5, 0.3]', generate_ma_process(n, [0.5, 0.3]), 'MA(2)'),
    ]

    max_lag = 20
    band = significance_bands(n)

    for row, (title, y, process_type) in enumerate(processes):
        # Column 1: Time series
        axes[row, 0].plot(y[:100], 'b-', linewidth=0.8)
        axes[row, 0].set_title(title, fontsize=10, fontweight='bold')
        axes[row, 0].set_xlabel('Time')
        axes[row, 0].set_ylabel('Value')
        axes[row, 0].grid(True, alpha=0.3)

        # Column 2: ACF
        acf_vals = acf(y, max_lag)
        lags = np.arange(len(acf_vals))

        colors = ['green' if abs(v) > band else 'steelblue' for v in acf_vals]
        axes[row, 1].bar(lags, acf_vals, color=colors, alpha=0.7, width=0.8)
        axes[row, 1].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[row, 1].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[row, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[row, 1].set_title('ACF', fontsize=10)
        axes[row, 1].set_xlabel('Lag')
        axes[row, 1].set_xlim(-0.5, max_lag + 0.5)

        # Column 3: PACF
        pacf_vals = pacf(y, max_lag)

        colors = ['darkred' if abs(v) > band else 'coral' for v in pacf_vals]
        axes[row, 2].bar(lags, pacf_vals, color=colors, alpha=0.7, width=0.8)
        axes[row, 2].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[row, 2].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[row, 2].axhline(y=0, color='black', linewidth=0.5)
        axes[row, 2].set_title('PACF', fontsize=10)
        axes[row, 2].set_xlabel('Lag')
        axes[row, 2].set_xlim(-0.5, max_lag + 0.5)

        # Column 4: Interpretation
        axes[row, 3].axis('off')

        if 'AR' in process_type and 'MA' not in title:
            interpretation = (
                f"PATTERN: {process_type}\n\n"
                f"ACF: Decays exponentially\n"
                f"PACF: Cuts off after lag {process_type[-2]}\n\n"
                f"RULE:\n"
                f"• ACF decays → AR component\n"
                f"• PACF cutoff → AR order"
            )
            color = 'lightblue'
        else:
            q = int(process_type[-2])
            interpretation = (
                f"PATTERN: {process_type}\n\n"
                f"ACF: Cuts off after lag {q}\n"
                f"PACF: Decays exponentially\n\n"
                f"RULE:\n"
                f"• ACF cutoff → MA order\n"
                f"• PACF decays → MA component"
            )
            color = 'lightyellow'

        axes[row, 3].text(0.1, 0.5, interpretation,
                         transform=axes[row, 3].transAxes, fontsize=10,
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                         family='monospace')

    plt.suptitle('ACF/PACF PATTERNS FOR MODEL IDENTIFICATION\n'
                 '"The signature of each process type is in its autocorrelation structure"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_seasonal_acf():
    """
    Show how seasonality appears in ACF.
    """
    np.random.seed(42)
    n = 300

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Generate series with different seasonal periods
    periods = [7, 12, 24]  # Weekly, Monthly, Daily cycles
    period_names = ['Weekly (period=7)', 'Monthly (period=12)', 'Daily (period=24)']

    for col, (period, name) in enumerate(zip(periods, period_names)):
        # Generate seasonal process
        y = generate_seasonal_process(n, period=period, ar_coef=0.3,
                                       seasonal_ar_coef=0.6, seed=42+col)

        # Time series
        axes[0, col].plot(y[:100], 'b-', linewidth=0.8)
        axes[0, col].set_title(name, fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel('Time')
        if col == 0:
            axes[0, col].set_ylabel('Value')
        axes[0, col].grid(True, alpha=0.3)

        # Mark seasonal periods
        for i in range(0, 100, period):
            axes[0, col].axvline(x=i, color='red', alpha=0.2, linewidth=1)

        # ACF
        max_lag = 50
        acf_vals = acf(y, max_lag)
        lags = np.arange(len(acf_vals))
        band = significance_bands(n)

        axes[1, col].bar(lags, acf_vals, color='steelblue', alpha=0.7, width=0.8)
        axes[1, col].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[1, col].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[1, col].axhline(y=0, color='black', linewidth=0.5)
        axes[1, col].set_xlabel('Lag')
        if col == 0:
            axes[1, col].set_ylabel('ACF')
        axes[1, col].set_title('Autocorrelation', fontsize=10)

        # Highlight seasonal lags
        for i in range(period, max_lag + 1, period):
            axes[1, col].bar(i, acf_vals[i], color='green', alpha=0.9, width=0.8)
            axes[1, col].annotate(f'Lag {i}', (i, acf_vals[i]),
                                   textcoords="offset points", xytext=(0, 10),
                                   ha='center', fontsize=8, color='green')

        # PACF
        pacf_vals = pacf(y, max_lag)

        axes[2, col].bar(lags, pacf_vals, color='coral', alpha=0.7, width=0.8)
        axes[2, col].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[2, col].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[2, col].axhline(y=0, color='black', linewidth=0.5)
        axes[2, col].set_xlabel('Lag')
        if col == 0:
            axes[2, col].set_ylabel('PACF')
        axes[2, col].set_title('Partial Autocorrelation', fontsize=10)

        # Highlight seasonal lags in PACF
        for i in range(period, max_lag + 1, period):
            if i < len(pacf_vals):
                axes[2, col].bar(i, pacf_vals[i], color='darkred', alpha=0.9, width=0.8)

    plt.suptitle('SEASONAL PATTERNS IN ACF/PACF\n'
                 '"Periodic spikes in ACF reveal the seasonal period"',
                 fontsize=13, fontweight='bold', y=0.98)

    # Add insight box
    fig.text(0.5, 0.02,
             'KEY INSIGHT: Peaks at regular intervals (lag s, 2s, 3s, ...) indicate seasonality with period s.\n'
             'The seasonal period is the most important lag for forecasting!',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def visualize_white_noise_vs_signal():
    """
    Compare ACF of white noise vs structured signal.
    Drives home what "no autocorrelation" looks like.
    """
    np.random.seed(42)
    n = 500

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # White noise
    white_noise = np.random.randn(n)

    # AR(1) with different strengths
    ar_weak = generate_ar_process(n, [0.3])
    ar_strong = generate_ar_process(n, [0.9])

    # Random walk
    random_walk = np.cumsum(np.random.randn(n))

    series_list = [
        (white_noise, 'WHITE NOISE\n(No Structure)', 'No memory'),
        (ar_weak, 'WEAK AR(1)\nφ=0.3', 'Short memory'),
        (ar_strong, 'STRONG AR(1)\nφ=0.9', 'Long memory'),
        (random_walk, 'RANDOM WALK\n(Non-stationary)', 'Infinite memory')
    ]

    max_lag = 30
    band = significance_bands(n)

    for col, (y, title, memory_type) in enumerate(series_list):
        # Time series
        axes[0, col].plot(y[:150], 'b-', linewidth=0.8)
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')
        axes[0, col].set_xlabel('Time')
        if col == 0:
            axes[0, col].set_ylabel('Value')
        axes[0, col].grid(True, alpha=0.3)

        # ACF
        acf_vals = acf(y, max_lag)
        lags = np.arange(len(acf_vals))

        # Color by significance
        colors = ['green' if abs(v) > band else 'gray' for v in acf_vals]
        axes[1, col].bar(lags, acf_vals, color=colors, alpha=0.7, width=0.8)
        axes[1, col].axhline(y=band, color='red', linestyle='--', linewidth=1.5,
                             label='95% band')
        axes[1, col].axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
        axes[1, col].axhline(y=0, color='black', linewidth=0.5)
        axes[1, col].set_xlabel('Lag')
        if col == 0:
            axes[1, col].set_ylabel('ACF')
            axes[1, col].legend(fontsize=8)
        axes[1, col].set_xlim(-0.5, max_lag + 0.5)
        axes[1, col].set_ylim(-0.3, 1.1)

        # Count significant lags
        n_significant = np.sum(np.abs(acf_vals[1:]) > band)
        axes[1, col].text(0.95, 0.95, f'{memory_type}\n{n_significant} sig. lags',
                          transform=axes[1, col].transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('FROM NO MEMORY TO INFINITE MEMORY\n'
                 '"ACF reveals how much the past matters for prediction"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_acf_for_model_selection():
    """
    Practical guide: Using ACF/PACF to select ARIMA parameters.
    """
    np.random.seed(42)
    n = 400

    fig = plt.figure(figsize=(16, 14))

    # Generate ARMA(2,1) process as example
    y = generate_arma_process(n, phi=[0.5, 0.3], theta=[0.4])

    # Add main title
    fig.suptitle('USING ACF/PACF FOR MODEL SELECTION\n'
                 'A step-by-step guide to identifying ARIMA orders',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Time series
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(y[:200], 'b-', linewidth=0.8)
    ax1.set_title('STEP 1: Examine the Time Series\n'
                  'Is it stationary? (constant mean/variance)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Add rolling mean
    window = 30
    rolling_mean = np.convolve(y[:200], np.ones(window)/window, mode='same')
    ax1.plot(rolling_mean, 'r-', linewidth=2, alpha=0.7, label='Rolling mean')
    ax1.legend()

    # Panel 2: ACF
    ax2 = fig.add_subplot(3, 2, 2)
    max_lag = 25
    acf_vals = acf(y, max_lag)
    lags = np.arange(len(acf_vals))
    band = significance_bands(n)

    ax2.bar(lags, acf_vals, color='steelblue', alpha=0.7, width=0.8)
    ax2.axhline(y=band, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('STEP 2: Examine ACF\n'
                  'Does it decay or cut off?', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')
    ax2.set_xlim(-0.5, max_lag + 0.5)

    # Add annotation
    ax2.annotate('Decays (not cuts off)\n→ AR component likely',
                 xy=(10, acf_vals[10]), xytext=(15, 0.5),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9, color='green')

    # Panel 3: PACF
    ax3 = fig.add_subplot(3, 2, 3)
    pacf_vals = pacf(y, max_lag)

    ax3.bar(lags, pacf_vals, color='coral', alpha=0.7, width=0.8)
    ax3.axhline(y=band, color='red', linestyle='--', linewidth=1.5)
    ax3.axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('STEP 3: Examine PACF\n'
                  'Does it cut off or decay?', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('PACF')
    ax3.set_xlim(-0.5, max_lag + 0.5)

    # Highlight significant lags
    for i in range(1, max_lag + 1):
        if abs(pacf_vals[i]) > band:
            ax3.bar(i, pacf_vals[i], color='darkred', alpha=0.9, width=0.8)

    ax3.annotate('Significant at lags 1, 2\n→ AR(2) suggested',
                 xy=(2, pacf_vals[2]), xytext=(8, 0.4),
                 arrowprops=dict(arrowstyle='->', color='darkred'),
                 fontsize=9, color='darkred')

    # Panel 4: Decision rules
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.axis('off')

    rules_text = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           MODEL IDENTIFICATION RULES                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  PATTERN               │  MODEL    │  PARAMETERS          ║
    ║  ──────────────────────┼───────────┼────────────────────  ║
    ║  ACF cuts off at q     │  MA(q)    │  q = cutoff lag      ║
    ║  PACF decays           │           │                      ║
    ║  ──────────────────────┼───────────┼────────────────────  ║
    ║  ACF decays            │  AR(p)    │  p = PACF cutoff     ║
    ║  PACF cuts off at p    │           │                      ║
    ║  ──────────────────────┼───────────┼────────────────────  ║
    ║  Both decay            │  ARMA(p,q)│  Use information     ║
    ║                        │           │  criteria (AIC/BIC)  ║
    ║  ──────────────────────┼───────────┼────────────────────  ║
    ║  ACF decays very slow  │  Need     │  Difference first    ║
    ║  (non-stationary)      │  d > 0    │  then re-examine     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    ax4.text(0.05, 0.95, rules_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Panel 5: Residual check
    ax5 = fig.add_subplot(3, 2, 5)

    # Fit AR(2) and check residuals
    p = 2
    X = np.column_stack([y[i:n-p+i] for i in range(p)])
    y_target = y[p:]
    coeffs = np.linalg.lstsq(X, y_target, rcond=None)[0]
    predictions = X @ coeffs
    residuals = y_target - predictions

    residual_acf = acf(residuals, max_lag)
    ax5.bar(lags, residual_acf, color='gray', alpha=0.7, width=0.8)
    ax5.axhline(y=band, color='red', linestyle='--', linewidth=1.5)
    ax5.axhline(y=-band, color='red', linestyle='--', linewidth=1.5)
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_title('STEP 4: Check Residual ACF\n'
                  'Should be white noise (no significant lags)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('Residual ACF')
    ax5.set_xlim(-0.5, max_lag + 0.5)

    # Count significant
    n_sig = np.sum(np.abs(residual_acf[1:]) > band)
    verdict = '✓ Good fit!' if n_sig <= 2 else '✗ Structure remaining'
    ax5.text(0.95, 0.95, f'{n_sig} significant lags\n{verdict}',
             transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round',
                      facecolor='lightgreen' if n_sig <= 2 else 'lightyellow',
                      alpha=0.9))

    # Panel 6: Summary
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    summary = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                    ANALYSIS SUMMARY                       ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  1. Series appears stationary (no differencing needed)    ║
    ║     → d = 0                                               ║
    ║                                                           ║
    ║  2. ACF decays gradually                                  ║
    ║     → AR component present                                ║
    ║                                                           ║
    ║  3. PACF significant at lags 1-2, then cuts off           ║
    ║     → AR(2) suggested, p = 2                              ║
    ║                                                           ║
    ║  4. Some residual structure remains                       ║
    ║     → Consider adding MA term, q = 1                      ║
    ║                                                           ║
    ║  ─────────────────────────────────────────────────────    ║
    ║  RECOMMENDATION: ARIMA(2, 0, 1) or AR(2)                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_sample_size():
    """
    How does sample size affect ACF estimation?
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of Sample Size on ACF Estimation")
    print("="*60)

    # True AR(1) with phi=0.7
    true_phi = 0.7

    sample_sizes = [50, 100, 200, 500, 1000, 2000]

    print(f"\nTrue AR(1) coefficient: φ = {true_phi}")
    print("Estimated ACF(1) for different sample sizes:")
    print("-" * 50)

    for n in sample_sizes:
        # Generate multiple samples
        estimates = []
        for seed in range(20):
            y = generate_ar_process(n, [true_phi], seed=seed)
            acf_val = acf(y, max_lag=1)[1]
            estimates.append(acf_val)

        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        bias = mean_est - true_phi

        print(f"n={n:5d}: ACF(1) = {mean_est:.3f} ± {std_est:.3f}, bias = {bias:+.3f}")

    print("\n→ Larger samples give more accurate, less variable estimates")


def ablation_ar_strength():
    """
    How does AR coefficient affect the range of significant lags?
    """
    print("\n" + "="*60)
    print("ABLATION: AR Coefficient vs Significant Lag Range")
    print("="*60)

    n = 500
    ar_coeffs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    print("\nNumber of significant ACF lags for different AR(1) coefficients:")
    print("-" * 50)

    band = significance_bands(n)

    for phi in ar_coeffs:
        y = generate_ar_process(n, [phi], seed=42)
        acf_vals = acf(y, max_lag=50)

        # Count significant lags
        n_significant = np.sum(np.abs(acf_vals[1:]) > band)

        # Theoretical decay: ACF(k) = phi^k
        # Find k where phi^k = band
        if phi > 0:
            theoretical_range = int(np.log(band) / np.log(phi)) if phi < 1 else float('inf')
        else:
            theoretical_range = 1

        print(f"φ = {phi:.2f}: {n_significant:2d} significant lags "
              f"(theory: ~{min(theoretical_range, 50)})")

    print("\n→ Stronger AR = longer memory = more significant lags")
    print("→ φ close to 1 = near non-stationary = very long memory")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("AUTOCORRELATION — Paradigm: MEMORY IN TIME")
    print("="*70)

    print("""

THE MEMORY PROBLEM:
    How far back should you look to predict the future?

    This is THE fundamental question in time series forecasting.
    Autocorrelation provides the answer.

AUTOCORRELATION (ACF):
    ACF(k) = Corr(y_t, y_{t-k})

    Measures total correlation between present and past.
    High ACF(k) → lag k is useful for prediction.

PARTIAL AUTOCORRELATION (PACF):
    PACF(k) = Corr(y_t, y_{t-k} | y_{t-1}, ..., y_{t-k+1})

    Measures DIRECT effect of lag k, removing intermediate effects.
    Crucial for identifying AR order.

MODEL IDENTIFICATION:
    - ACF decays, PACF cuts off → AR(p), where p = PACF cutoff
    - ACF cuts off, PACF decays → MA(q), where q = ACF cutoff
    - Both decay → ARMA(p,q), use information criteria

    """)

    # Run ablations
    ablation_sample_size()
    ablation_ar_strength()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. Memory problem
    fig1 = visualize_memory_problem()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/02_acf_memory_problem.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. ACF/PACF patterns
    fig2 = visualize_acf_pacf_patterns()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/02_acf_pacf_patterns.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Seasonal ACF
    fig3 = visualize_seasonal_acf()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/02_acf_seasonal.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. White noise vs signal
    fig4 = visualize_white_noise_vs_signal()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/02_acf_memory_spectrum.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    # 5. Model selection guide
    fig5 = visualize_acf_for_model_selection()
    save_path5 = '/Users/sid47/ML Algorithms/11-time-series/02_acf_model_selection.png'
    fig5.savefig(save_path5, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path5}")
    plt.close(fig5)

    print("\n" + "="*60)
    print("SUMMARY: Autocorrelation")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 02_acf_memory_problem.png   — How far back to look?
    2. 02_acf_pacf_patterns.png    — Signatures of AR/MA processes
    3. 02_acf_seasonal.png         — Detecting seasonality via ACF
    4. 02_acf_memory_spectrum.png  — From no memory to infinite memory
    5. 02_acf_model_selection.png  — Step-by-step model identification

KEY TAKEAWAYS:
    1. ACF answers "how much does the past tell us about the future?"
    2. PACF reveals DIRECT effects, crucial for AR order selection
    3. Seasonal patterns show as periodic spikes in ACF
    4. White noise has no significant autocorrelation
    5. Model selection: ACF/PACF patterns → AR/MA orders

NEXT: Exponential Smoothing — "How fast should you forget the past?"
    """)
