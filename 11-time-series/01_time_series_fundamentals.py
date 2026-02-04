"""
TIME SERIES FUNDAMENTALS — Paradigm: TEMPORAL STRUCTURE

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A time series is a sequence of observations ordered in time.
    y_1, y_2, y_3, ..., y_t

The FUNDAMENTAL ASSUMPTION:
    "The past contains information about the future"

If this isn't true, prediction is impossible.
If this IS true, the question becomes: HOW MUCH past? WHAT structure?

===============================================================
THE DECOMPOSITION VIEW
===============================================================

Any time series can be decomposed into:

    y_t = Trend_t + Seasonality_t + Residual_t

TREND: Long-term direction (up, down, flat)
    - The "drift" of the series
    - Usually modeled as polynomial or moving average

SEASONALITY: Repeating patterns at fixed intervals
    - Daily, weekly, monthly, yearly cycles
    - The "heartbeat" of the series

RESIDUAL: What's left after removing trend and seasonality
    - Ideally: random noise (unpredictable)
    - If structured: your model is missing something!

===============================================================
THE KEY INSIGHT
===============================================================

    Time Series = PREDICTABLE + UNPREDICTABLE

Your model's job: Maximize what you can predict.
The residual shows what you CAN'T predict.

If residuals have structure → your model is leaving money on the table.
If residuals are white noise → you've extracted all the signal.

===============================================================
STATIONARITY: THE CRITICAL ASSUMPTION
===============================================================

A series is STATIONARY if its statistical properties don't change over time:
    - Constant mean: E[y_t] = μ for all t
    - Constant variance: Var(y_t) = σ² for all t
    - Autocovariance depends only on lag, not time

WHY IT MATTERS:
    Most time series methods ASSUME stationarity.
    Non-stationary data must be TRANSFORMED first.

COMMON TRANSFORMATIONS:
    - Differencing: y'_t = y_t - y_{t-1} (removes trend)
    - Log transform: log(y_t) (stabilizes variance)
    - Seasonal differencing: y'_t = y_t - y_{t-s} (removes seasonality)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d


# ============================================================
# SYNTHETIC TIME SERIES GENERATORS
# ============================================================

def generate_trend(n, trend_type='linear', strength=1.0):
    """
    Generate trend component.

    Types:
    - 'linear': constant slope
    - 'quadratic': accelerating/decelerating
    - 'exponential': compound growth
    - 'logistic': S-curve (saturation)
    """
    t = np.arange(n)

    if trend_type == 'linear':
        return strength * t / n
    elif trend_type == 'quadratic':
        return strength * (t / n) ** 2
    elif trend_type == 'exponential':
        return strength * (np.exp(t / n) - 1)
    elif trend_type == 'logistic':
        return strength / (1 + np.exp(-0.02 * (t - n/2)))
    else:
        return np.zeros(n)


def generate_seasonality(n, period=12, amplitude=1.0, pattern='sine'):
    """
    Generate seasonal component.

    Patterns:
    - 'sine': smooth sinusoidal
    - 'sawtooth': linear increase then drop
    - 'square': binary high/low
    - 'complex': multiple harmonics
    """
    t = np.arange(n)

    if pattern == 'sine':
        return amplitude * np.sin(2 * np.pi * t / period)
    elif pattern == 'sawtooth':
        return amplitude * (2 * (t % period) / period - 1)
    elif pattern == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * t / period))
    elif pattern == 'complex':
        # Multiple harmonics
        return (amplitude * np.sin(2 * np.pi * t / period) +
                0.5 * amplitude * np.sin(4 * np.pi * t / period) +
                0.25 * amplitude * np.sin(6 * np.pi * t / period))
    else:
        return np.zeros(n)


def generate_noise(n, noise_type='gaussian', scale=1.0):
    """
    Generate noise component.

    Types:
    - 'gaussian': white noise (most common assumption)
    - 'uniform': bounded noise
    - 'heteroscedastic': variance changes over time
    - 'autocorrelated': noise depends on previous noise (AR(1) noise)
    """
    if noise_type == 'gaussian':
        return scale * np.random.randn(n)
    elif noise_type == 'uniform':
        return scale * (np.random.rand(n) - 0.5) * 2
    elif noise_type == 'heteroscedastic':
        # Variance increases over time
        variance = 1 + np.arange(n) / n
        return scale * np.random.randn(n) * np.sqrt(variance)
    elif noise_type == 'autocorrelated':
        # AR(1) noise with coefficient 0.7
        noise = np.zeros(n)
        noise[0] = np.random.randn()
        for i in range(1, n):
            noise[i] = 0.7 * noise[i-1] + scale * np.random.randn()
        return noise
    else:
        return np.zeros(n)


def generate_time_series(n=200, trend='linear', trend_strength=2.0,
                         period=20, seasonal_amplitude=1.0, seasonal_pattern='sine',
                         noise_scale=0.3, noise_type='gaussian', seed=42):
    """
    Generate a complete time series with all components.

    Returns: y, trend, seasonality, noise
    """
    np.random.seed(seed)

    trend_component = generate_trend(n, trend, trend_strength)
    seasonal_component = generate_seasonality(n, period, seasonal_amplitude, seasonal_pattern)
    noise_component = generate_noise(n, noise_type, noise_scale)

    y = trend_component + seasonal_component + noise_component

    return y, trend_component, seasonal_component, noise_component


# ============================================================
# DECOMPOSITION METHODS
# ============================================================

def moving_average(y, window):
    """Simple moving average for trend estimation."""
    return uniform_filter1d(y, size=window, mode='nearest')


def decompose_additive(y, period):
    """
    Classical additive decomposition.

    y_t = T_t + S_t + R_t

    Steps:
    1. Estimate trend with moving average (window = period)
    2. Detrend: y - T
    3. Estimate seasonality: average detrended values for each season
    4. Residual: y - T - S
    """
    n = len(y)

    # Step 1: Trend via centered moving average
    trend = moving_average(y, period)

    # Step 2: Detrend
    detrended = y - trend

    # Step 3: Seasonal component (average for each position in cycle)
    seasonal = np.zeros(n)
    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal_mean = np.mean(detrended[indices])
        seasonal[indices] = seasonal_mean

    # Normalize seasonal component to sum to zero
    seasonal = seasonal - np.mean(seasonal)

    # Step 4: Residual
    residual = y - trend - seasonal

    return trend, seasonal, residual


def decompose_stl(y, period, trend_window=None):
    """
    STL-like decomposition (Seasonal-Trend decomposition using LOESS).

    This is a simplified version that uses moving averages.
    Real STL uses LOESS regression.
    """
    if trend_window is None:
        trend_window = period + 1 if period % 2 == 0 else period

    n = len(y)

    # Initial trend estimate
    trend = moving_average(y, trend_window)

    # Iterate to refine
    for _ in range(3):
        # Detrend
        detrended = y - trend

        # Estimate seasonal
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal[indices] = np.mean(detrended[indices])

        # Deseasonal
        deseasoned = y - seasonal

        # Re-estimate trend
        trend = moving_average(deseasoned, trend_window)

    residual = y - trend - seasonal

    return trend, seasonal, residual


# ============================================================
# STATIONARITY TESTS
# ============================================================

def adf_test_simple(y, max_lag=None):
    """
    Simplified Augmented Dickey-Fuller test for stationarity.

    Tests H0: series has a unit root (non-stationary)
    vs H1: series is stationary

    Returns: test statistic (more negative = more likely stationary)

    NOTE: This is a simplified version. Use statsmodels.tsa.stattools.adfuller
    for production use.
    """
    n = len(y)
    if max_lag is None:
        max_lag = int(np.floor(np.power(n - 1, 1/3)))

    # First difference
    dy = np.diff(y)

    # Lagged level
    y_lag = y[:-1]

    # Simple regression: dy_t = alpha + beta * y_{t-1} + error
    # Under H0: beta = 0 (unit root)
    X = np.column_stack([np.ones(len(y_lag)), y_lag])

    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        residuals = dy - X @ beta

        # Standard error of beta[1]
        mse = np.sum(residuals**2) / (len(dy) - 2)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(var_beta[1, 1])

        # Test statistic
        t_stat = beta[1] / se_beta

        return t_stat
    except:
        return np.nan


def check_stationarity(y, window=50):
    """
    Visual check for stationarity by computing rolling statistics.

    Returns: rolling_mean, rolling_std
    """
    n = len(y)
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)

    half_window = window // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window)
        rolling_mean[i] = np.mean(y[start:end])
        rolling_std[i] = np.std(y[start:end])

    return rolling_mean, rolling_std


# ============================================================
# DIFFERENCING OPERATIONS
# ============================================================

def difference(y, d=1):
    """
    Apply differencing d times.

    d=1: y'_t = y_t - y_{t-1}  (removes linear trend)
    d=2: y''_t = y'_t - y'_{t-1}  (removes quadratic trend)
    """
    result = y.copy()
    for _ in range(d):
        result = np.diff(result)
    return result


def seasonal_difference(y, period):
    """
    Seasonal differencing.

    y'_t = y_t - y_{t-period}

    Removes seasonality with the given period.
    """
    return y[period:] - y[:-period]


def integrate(dy, y0=0, d=1):
    """
    Inverse of differencing (cumulative sum).
    """
    result = dy.copy()
    for _ in range(d):
        result = np.cumsum(np.concatenate([[y0], result]))
    return result


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_anatomy():
    """
    THE ANATOMY VISUALIZATION

    Shows how a time series is composed of trend + seasonality + noise.
    This is the foundational intuition for all time series analysis.
    """
    np.random.seed(42)

    # Generate series with clear components
    n = 200
    y, trend, seasonal, noise = generate_time_series(
        n=n,
        trend='linear', trend_strength=3.0,
        period=20, seasonal_amplitude=1.5, seasonal_pattern='sine',
        noise_scale=0.4, noise_type='gaussian'
    )

    fig = plt.figure(figsize=(16, 12))

    # Main title
    fig.suptitle("ANATOMY OF A TIME SERIES: What's Hiding in Your Data?\n"
                 '"Every time series = Predictable Structure + Unpredictable Noise"',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: The decomposition story
    t = np.arange(n)

    # Panel 1: Raw signal
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.plot(t, y, 'b-', linewidth=1, alpha=0.8)
    ax1.set_title('RAW SIGNAL\n"What you observe"', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Trend
    ax2 = fig.add_subplot(2, 5, 2)
    ax2.plot(t, trend, 'r-', linewidth=2)
    ax2.set_title('TREND\n"Where is it going?"', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.1, '↗ Long-term drift', transform=ax2.transAxes,
             ha='center', fontsize=9, color='red')

    # Panel 3: Seasonality
    ax3 = fig.add_subplot(2, 5, 3)
    ax3.plot(t, seasonal, 'g-', linewidth=1.5)
    ax3.set_title('SEASONALITY\n"What repeats?"', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.text(0.5, 0.1, '∿ Cycles at period=20', transform=ax3.transAxes,
             ha='center', fontsize=9, color='green')

    # Panel 4: Noise
    ax4 = fig.add_subplot(2, 5, 4)
    ax4.plot(t, noise, 'gray', linewidth=0.8, alpha=0.7)
    ax4.set_title('RESIDUAL (NOISE)\n"What we can\'t predict"', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.text(0.5, 0.1, '··· Random fluctuations', transform=ax4.transAxes,
             ha='center', fontsize=9, color='gray')

    # Panel 5: Reconstruction
    ax5 = fig.add_subplot(2, 5, 5)
    reconstruction = trend + seasonal
    ax5.plot(t, y, 'b-', linewidth=1, alpha=0.4, label='Original')
    ax5.plot(t, reconstruction, 'purple', linewidth=2, label='Trend + Seasonal')
    ax5.set_title('RECONSTRUCTION\n"Trend + Seasonal ≈ Signal"', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Row 2: The key insight with real decomposition
    ax6 = fig.add_subplot(2, 1, 2)

    # Decompose using our method (simulating what we'd do without knowing truth)
    est_trend, est_seasonal, est_residual = decompose_additive(y, period=20)

    # Plot all components stacked
    ax6.fill_between(t, 0, trend, alpha=0.3, color='red', label='Trend (predictable)')
    ax6.fill_between(t, trend, trend + seasonal, alpha=0.3, color='green', label='Seasonality (predictable)')
    ax6.plot(t, y, 'b-', linewidth=1.5, label='Observed', alpha=0.8)
    ax6.plot(t, trend + seasonal, 'purple', linewidth=2, linestyle='--', label='Predictable part')

    ax6.set_xlabel('Time', fontsize=11)
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('THE KEY INSIGHT: Separate What You CAN Predict From What You CANNOT',
                  fontsize=11, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Add insight box
    textstr = ('PREDICTABLE = Trend + Seasonality\n'
               'UNPREDICTABLE = Residual (noise)\n\n'
               'If residuals show patterns → Model is missing something!\n'
               'If residuals are random → You\'ve extracted all the signal.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax6.text(0.98, 0.98, textstr, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_decomposition_comparison():
    """
    Compare different decomposition methods on the same series.
    Shows what happens with different seasonal patterns.
    """
    np.random.seed(42)
    n = 200
    period = 20

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    patterns = ['sine', 'sawtooth', 'square', 'complex']
    pattern_names = ['Sinusoidal', 'Sawtooth', 'Square Wave', 'Complex Harmonics']

    for col, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        # Generate series
        y, true_trend, true_seasonal, true_noise = generate_time_series(
            n=n, trend='linear', trend_strength=2.0,
            period=period, seasonal_amplitude=1.5, seasonal_pattern=pattern,
            noise_scale=0.3
        )

        # Decompose
        est_trend, est_seasonal, est_residual = decompose_additive(y, period=period)

        t = np.arange(n)

        # Row 1: Original
        axes[0, col].plot(t, y, 'b-', linewidth=0.8)
        axes[0, col].set_title(f'{name}\nPattern', fontsize=10, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel('Original', fontsize=10)
        axes[0, col].grid(True, alpha=0.3)

        # Row 2: True vs Estimated Trend
        axes[1, col].plot(t, true_trend, 'r-', linewidth=2, label='True', alpha=0.7)
        axes[1, col].plot(t, est_trend, 'r--', linewidth=2, label='Estimated')
        if col == 0:
            axes[1, col].set_ylabel('Trend', fontsize=10)
            axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        # Row 3: True vs Estimated Seasonal
        axes[2, col].plot(t, true_seasonal, 'g-', linewidth=1.5, label='True', alpha=0.7)
        axes[2, col].plot(t, est_seasonal, 'g--', linewidth=1.5, label='Estimated')
        if col == 0:
            axes[2, col].set_ylabel('Seasonal', fontsize=10)
            axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

        # Row 4: Residuals
        axes[3, col].plot(t, true_noise, 'gray', linewidth=0.8, alpha=0.5, label='True noise')
        axes[3, col].plot(t, est_residual, 'k-', linewidth=0.8, label='Residual')
        axes[3, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        if col == 0:
            axes[3, col].set_ylabel('Residual', fontsize=10)
            axes[3, col].legend(fontsize=8)
        axes[3, col].set_xlabel('Time')
        axes[3, col].grid(True, alpha=0.3)

        # Check if residuals look random
        residual_autocorr = np.corrcoef(est_residual[:-1], est_residual[1:])[0, 1]
        status = '✓ Good' if abs(residual_autocorr) < 0.3 else '✗ Structured'
        axes[3, col].text(0.95, 0.95, f'Autocorr: {residual_autocorr:.2f}\n{status}',
                         transform=axes[3, col].transAxes, fontsize=8,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('DECOMPOSITION ACROSS SEASONAL PATTERNS\n'
                 'Classical decomposition works well for smooth patterns, struggles with sharp edges',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_stationarity():
    """
    Visualize what stationarity means and how to detect non-stationarity.
    """
    np.random.seed(42)
    n = 300
    t = np.arange(n)

    fig = plt.figure(figsize=(16, 12))

    # Generate different types of series
    # 1. Stationary (white noise)
    stationary = np.random.randn(n)

    # 2. Non-stationary: trend
    trend_nonstat = np.random.randn(n) + 0.02 * t

    # 3. Non-stationary: changing variance
    hetero = np.random.randn(n) * (1 + 0.01 * t)

    # 4. Non-stationary: random walk
    random_walk = np.cumsum(np.random.randn(n))

    series_list = [
        (stationary, 'STATIONARY\n(White Noise)', True),
        (trend_nonstat, 'NON-STATIONARY\n(Trend)', False),
        (hetero, 'NON-STATIONARY\n(Changing Variance)', False),
        (random_walk, 'NON-STATIONARY\n(Random Walk)', False)
    ]

    for idx, (series, title, is_stationary) in enumerate(series_list):
        # Time series plot
        ax1 = fig.add_subplot(4, 3, idx * 3 + 1)
        ax1.plot(t, series, 'b-', linewidth=0.8)
        ax1.set_title(title, fontsize=10, fontweight='bold',
                      color='green' if is_stationary else 'red')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)

        # Rolling statistics
        ax2 = fig.add_subplot(4, 3, idx * 3 + 2)
        roll_mean, roll_std = check_stationarity(series, window=30)
        ax2.plot(t, roll_mean, 'r-', linewidth=2, label='Rolling Mean')
        ax2.fill_between(t, roll_mean - roll_std, roll_mean + roll_std,
                         alpha=0.3, color='red', label='±1 Std')
        ax2.axhline(y=np.mean(series), color='blue', linestyle='--',
                    alpha=0.7, label='Global Mean')
        ax2.set_title('Rolling Statistics\n(Should be flat for stationary)', fontsize=9)
        ax2.set_xlabel('Time')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Histogram at different times
        ax3 = fig.add_subplot(4, 3, idx * 3 + 3)
        # Split into thirds
        third = n // 3
        ax3.hist(series[:third], bins=20, alpha=0.5, label='First third', density=True)
        ax3.hist(series[2*third:], bins=20, alpha=0.5, label='Last third', density=True)
        ax3.set_title('Distribution Over Time\n(Should overlap for stationary)', fontsize=9)
        ax3.legend(fontsize=7)
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')

        # Add ADF test result
        adf_stat = adf_test_simple(series)
        verdict = "Likely Stationary" if adf_stat < -2.9 else "Likely Non-Stationary"
        ax3.text(0.95, 0.95, f'ADF stat: {adf_stat:.2f}\n{verdict}',
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round',
                          facecolor='lightgreen' if adf_stat < -2.9 else 'lightyellow',
                          alpha=0.8))

    plt.suptitle('STATIONARITY: The Critical Assumption\n'
                 '"Most time series methods assume the statistical properties don\'t change over time"',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def visualize_differencing():
    """
    Show how differencing transforms non-stationary series to stationary.
    """
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    # Generate different non-stationary series
    # 1. Linear trend
    linear = 0.05 * t + np.random.randn(n) * 0.5

    # 2. Quadratic trend
    quadratic = 0.001 * t**2 + np.random.randn(n) * 2

    # 3. Random walk
    random_walk = np.cumsum(np.random.randn(n))

    # 4. Trend + Seasonality
    trend_seasonal = 0.03 * t + 2 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 0.3

    series_list = [
        (linear, 'Linear Trend', 1),
        (quadratic, 'Quadratic Trend', 2),
        (random_walk, 'Random Walk', 1),
        (trend_seasonal, 'Trend + Seasonal', 1)
    ]

    for col, (series, title, d_needed) in enumerate(series_list):
        # Original
        axes[0, col].plot(series, 'b-', linewidth=0.8)
        axes[0, col].set_title(f'{title}\n(Non-stationary)', fontsize=10, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel('Original', fontsize=10)
        axes[0, col].grid(True, alpha=0.3)

        # First difference
        diff1 = difference(series, d=1)
        axes[1, col].plot(diff1, 'g-', linewidth=0.8)
        adf1 = adf_test_simple(diff1)
        status1 = '✓' if adf1 < -2.9 else '✗'
        axes[1, col].set_title(f'd=1: ADF={adf1:.1f} {status1}', fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel('First Difference\n$y_t - y_{t-1}$', fontsize=10)
        axes[1, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, col].grid(True, alpha=0.3)

        # Second difference
        diff2 = difference(series, d=2)
        axes[2, col].plot(diff2, 'purple', linewidth=0.8)
        adf2 = adf_test_simple(diff2)
        status2 = '✓' if adf2 < -2.9 else '✗'
        axes[2, col].set_title(f'd=2: ADF={adf2:.1f} {status2}', fontsize=9)
        if col == 0:
            axes[2, col].set_ylabel('Second Difference\n$\\Delta^2 y_t$', fontsize=10)
        axes[2, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2, col].set_xlabel('Time')
        axes[2, col].grid(True, alpha=0.3)

        # Highlight the row that achieves stationarity
        if d_needed == 1:
            axes[1, col].patch.set_facecolor('lightgreen')
            axes[1, col].patch.set_alpha(0.3)
        elif d_needed == 2:
            axes[2, col].patch.set_facecolor('lightgreen')
            axes[2, col].patch.set_alpha(0.3)

    plt.suptitle('DIFFERENCING: The Cure for Non-Stationarity\n'
                 '"Remove trends by computing changes instead of levels"',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_predictability_spectrum():
    """
    Show the spectrum from completely predictable to completely random.
    This drives home the core insight about what forecasting can achieve.
    """
    np.random.seed(42)
    n = 150
    t = np.arange(n)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Generate series with different signal-to-noise ratios
    trend = 0.03 * t
    seasonal = np.sin(2 * np.pi * t / 20)

    snr_levels = [
        (0.0, 'PURE SIGNAL\n(100% Predictable)'),
        (0.3, 'HIGH SNR\n(~90% Predictable)'),
        (1.0, 'MODERATE SNR\n(~50% Predictable)'),
        (3.0, 'PURE NOISE\n(~0% Predictable)')
    ]

    for col, (noise_level, title) in enumerate(snr_levels):
        signal = trend + seasonal
        noise = noise_level * np.random.randn(n)
        y = signal + noise

        # Calculate actual predictability (R² of signal in y)
        if noise_level > 0:
            r2 = 1 - np.var(noise) / np.var(y)
        else:
            r2 = 1.0
        r2 = max(0, r2)

        # Top row: Time series
        axes[0, col].plot(t, y, 'b-', linewidth=0.8, alpha=0.8, label='Observed')
        axes[0, col].plot(t, signal, 'r-', linewidth=2, alpha=0.8, label='True Signal')
        axes[0, col].set_title(f'{title}\nR² = {r2:.1%}', fontsize=10, fontweight='bold')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Value')

        # Bottom row: What forecasting can achieve
        # Simple prediction: use signal estimate
        train_end = 100

        # Fit trend on training data
        train_trend = np.polyfit(t[:train_end], y[:train_end], 1)
        trend_pred = np.polyval(train_trend, t)

        axes[1, col].plot(t[:train_end], y[:train_end], 'b-', linewidth=0.8, alpha=0.5)
        axes[1, col].plot(t[train_end:], y[train_end:], 'b-', linewidth=1.5, label='Actual')
        axes[1, col].plot(t[train_end:], trend_pred[train_end:], 'r--', linewidth=2, label='Forecast')
        axes[1, col].axvline(x=train_end, color='gray', linestyle=':', alpha=0.7)
        axes[1, col].fill_between(t[train_end:],
                                   trend_pred[train_end:] - 1.96 * noise_level,
                                   trend_pred[train_end:] + 1.96 * noise_level,
                                   alpha=0.2, color='red', label='95% CI')
        axes[1, col].legend(fontsize=7)
        axes[1, col].set_xlabel('Time')
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel('Forecast')

        # Color code by predictability
        if r2 > 0.8:
            color = 'green'
        elif r2 > 0.4:
            color = 'orange'
        else:
            color = 'red'
        axes[0, col].title.set_color(color)

    plt.suptitle('THE PREDICTABILITY SPECTRUM\n'
                 '"No model can predict noise. Your job is finding the signal."',
                 fontsize=13, fontweight='bold', y=0.98)

    # Add insight box
    fig.text(0.5, 0.02,
             '← More Predictable                                                    More Random →\n'
             'If a series is 50% noise, the BEST possible model can only explain 50% of variance.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_decomposition_period():
    """
    What happens when you use the wrong period for decomposition?
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of Wrong Period in Decomposition")
    print("="*60)

    np.random.seed(42)

    # True period is 20
    true_period = 20
    y, trend, seasonal, noise = generate_time_series(
        n=200, period=true_period, seasonal_amplitude=2.0, noise_scale=0.3
    )

    print(f"\nTrue seasonal period: {true_period}")
    print("\nTesting different assumed periods:")
    print("-" * 40)

    for test_period in [10, 15, 20, 25, 30, 40]:
        est_trend, est_seasonal, est_residual = decompose_additive(y, period=test_period)

        # Measure quality: residual should be uncorrelated if decomposition is good
        residual_var = np.var(est_residual)
        residual_autocorr = np.abs(np.corrcoef(est_residual[:-1], est_residual[1:])[0, 1])

        # Compare to true noise
        noise_recovery = 1 - np.var(est_residual - noise) / np.var(noise)

        status = "✓ CORRECT" if test_period == true_period else ""
        print(f"Period={test_period:3d}: Residual Var={residual_var:.3f}, "
              f"Autocorr={residual_autocorr:.3f} {status}")

    print("\n→ Wrong period leaves structure in residuals (high autocorrelation)")
    print("→ Correct period produces near-white-noise residuals")


def ablation_noise_level():
    """
    How does noise level affect our ability to recover components?
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of Noise Level on Decomposition Quality")
    print("="*60)

    np.random.seed(42)

    print("\nRecovery quality (R² for trend and seasonal estimation):")
    print("-" * 50)

    for noise_scale in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        y, true_trend, true_seasonal, true_noise = generate_time_series(
            n=200, period=20, seasonal_amplitude=1.5, noise_scale=noise_scale
        )

        est_trend, est_seasonal, est_residual = decompose_additive(y, period=20)

        # R² for trend
        ss_tot_trend = np.sum((true_trend - np.mean(true_trend))**2)
        ss_res_trend = np.sum((true_trend - est_trend)**2)
        r2_trend = 1 - ss_res_trend / ss_tot_trend if ss_tot_trend > 0 else 0

        # R² for seasonal
        ss_tot_seas = np.sum((true_seasonal - np.mean(true_seasonal))**2)
        ss_res_seas = np.sum((true_seasonal - est_seasonal)**2)
        r2_seas = 1 - ss_res_seas / ss_tot_seas if ss_tot_seas > 0 else 0

        # Signal-to-noise ratio
        snr = np.var(true_trend + true_seasonal) / np.var(true_noise)

        print(f"Noise={noise_scale:.1f} (SNR={snr:.2f}): "
              f"Trend R²={max(0,r2_trend):.3f}, Seasonal R²={max(0,r2_seas):.3f}")

    print("\n→ Higher noise makes component recovery harder")
    print("→ Trend is usually more robust than seasonality")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("TIME SERIES FUNDAMENTALS — Paradigm: TEMPORAL STRUCTURE")
    print("="*70)

    print("""

THE CORE IDEA:
    A time series is a sequence ordered in time: y_1, y_2, ..., y_t

    The fundamental assumption: "The past contains information about the future"

DECOMPOSITION:
    y_t = Trend + Seasonality + Residual

    TREND: Where is it going? (long-term drift)
    SEASONALITY: What repeats? (cycles at fixed intervals)
    RESIDUAL: What we can't predict (hopefully just noise)

THE KEY INSIGHT:
    Time Series = PREDICTABLE + UNPREDICTABLE

    Your model's job: Maximize the predictable part.
    If residuals have structure → your model is missing something.
    If residuals are white noise → you've extracted all the signal.

STATIONARITY:
    Most methods assume statistical properties don't change over time.
    Non-stationary data must be TRANSFORMED first (differencing, log, etc.)

    """)

    # Run ablations
    ablation_decomposition_period()
    ablation_noise_level()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. Anatomy visualization
    fig1 = visualize_anatomy()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/01_ts_anatomy.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. Decomposition comparison
    fig2 = visualize_decomposition_comparison()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/01_ts_decomposition.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Stationarity visualization
    fig3 = visualize_stationarity()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/01_ts_stationarity.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. Differencing visualization
    fig4 = visualize_differencing()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/01_ts_differencing.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    # 5. Predictability spectrum
    fig5 = visualize_predictability_spectrum()
    save_path5 = '/Users/sid47/ML Algorithms/11-time-series/01_ts_predictability.png'
    fig5.savefig(save_path5, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path5}")
    plt.close(fig5)

    print("\n" + "="*60)
    print("SUMMARY: Time Series Fundamentals")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 01_ts_anatomy.png        — Decomposition into trend + seasonal + noise
    2. 01_ts_decomposition.png  — How decomposition handles different patterns
    3. 01_ts_stationarity.png   — What stationarity means and how to detect it
    4. 01_ts_differencing.png   — How differencing achieves stationarity
    5. 01_ts_predictability.png — The spectrum from signal to noise

KEY TAKEAWAYS:
    1. Every time series = Predictable part + Unpredictable part
    2. Decomposition reveals: trend (drift), seasonality (cycles), residual (noise)
    3. Stationarity is critical: mean and variance shouldn't change over time
    4. Differencing transforms non-stationary → stationary
    5. No model can predict pure noise — know your limits!

NEXT: Autocorrelation — "How much does the past tell us about the future?"
    """)
