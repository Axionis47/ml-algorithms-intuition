"""
EXPONENTIAL SMOOTHING — Paradigm: WEIGHTED FORGETTING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Forecast by taking a WEIGHTED AVERAGE of past observations,
where recent observations get MORE weight than distant ones.

The weights decay EXPONENTIALLY:
    weight(t-k) = α × (1-α)^k

    α = smoothing parameter (0 < α < 1)

===============================================================
THE FORGETTING CURVE
===============================================================

This is THE key intuition:

    α controls HOW FAST you forget the past.

    α → 1: "Short memory" — mostly use recent data, react fast
    α → 0: "Long memory" — use distant data, smooth but slow

The EFFECTIVE MEMORY LENGTH is approximately 1/α:
    α = 0.1 → ~10 periods of memory
    α = 0.5 → ~2 periods of memory
    α = 0.9 → ~1 period of memory

===============================================================
SIMPLE EXPONENTIAL SMOOTHING (SES)
===============================================================

For series with NO trend and NO seasonality:

    ŷ_{t+1} = α × y_t + (1-α) × ŷ_t

Equivalent recursive form:
    ŷ_{t+1} = ŷ_t + α × (y_t - ŷ_t)

    "New forecast = Old forecast + α × (Error)"

This is the simplest adaptive filter!

===============================================================
HOLT'S METHOD (Double Exponential Smoothing)
===============================================================

For series WITH trend but NO seasonality:

    Level:  l_t = α × y_t + (1-α) × (l_{t-1} + b_{t-1})
    Trend:  b_t = β × (l_t - l_{t-1}) + (1-β) × b_{t-1}

    Forecast: ŷ_{t+h} = l_t + h × b_t

Two smoothing parameters:
    α: level smoothing (how fast to adapt level)
    β: trend smoothing (how fast to adapt trend)

===============================================================
HOLT-WINTERS (Triple Exponential Smoothing)
===============================================================

For series WITH trend AND seasonality:

ADDITIVE seasonality:
    Level:    l_t = α(y_t - s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})
    Trend:    b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}
    Seasonal: s_t = γ(y_t - l_t) + (1-γ)s_{t-m}

    Forecast: ŷ_{t+h} = l_t + h×b_t + s_{t+h-m}

MULTIPLICATIVE seasonality:
    Level:    l_t = α(y_t / s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})
    Trend:    b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}
    Seasonal: s_t = γ(y_t / l_t) + (1-γ)s_{t-m}

    Forecast: ŷ_{t+h} = (l_t + h×b_t) × s_{t+h-m}

Three smoothing parameters:
    α: level smoothing
    β: trend smoothing
    γ: seasonal smoothing

===============================================================
INDUCTIVE BIAS
===============================================================

1. Recent observations are more relevant than distant ones
2. The future will be "like" the recent past
3. Changes happen gradually (smooth transitions)

WHAT IT CAN DO:
    ✓ Adapt to changing levels
    ✓ Track trends
    ✓ Capture seasonal patterns
    ✓ Work with limited data

WHAT IT CAN'T DO:
    ✗ Capture complex nonlinear patterns
    ✗ Handle sudden structural breaks
    ✗ Use external predictors (covariates)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms/11-time-series')


# ============================================================
# EXPONENTIAL SMOOTHING IMPLEMENTATIONS
# ============================================================

class SimpleExponentialSmoothing:
    """
    Simple Exponential Smoothing (SES).

    For series with NO trend and NO seasonality.

    ŷ_{t+1} = α × y_t + (1-α) × ŷ_t
    """

    def __init__(self, alpha=0.3):
        """
        Parameters:
        -----------
        alpha : float (0 < alpha < 1)
            Smoothing parameter. Higher = more weight on recent observations.
        """
        self.alpha = alpha
        self.fitted_values = None
        self.level = None

    def fit(self, y):
        """Fit the model to time series y."""
        n = len(y)
        self.fitted_values = np.zeros(n)

        # Initialize level with first observation
        self.level = y[0]
        self.fitted_values[0] = self.level

        # Recursively compute smoothed values
        for t in range(1, n):
            self.level = self.alpha * y[t-1] + (1 - self.alpha) * self.level
            self.fitted_values[t] = self.level

        # Update final level
        self.level = self.alpha * y[-1] + (1 - self.alpha) * self.level

        return self

    def forecast(self, h=1):
        """Forecast h steps ahead."""
        # SES forecast is flat (constant at current level)
        return np.full(h, self.level)

    def get_weights(self, n_weights=20):
        """
        Return the weights given to past observations.

        weight(k) = α × (1-α)^k for k = 0, 1, 2, ...
        """
        k = np.arange(n_weights)
        weights = self.alpha * (1 - self.alpha) ** k
        return weights


class HoltLinear:
    """
    Holt's Linear Method (Double Exponential Smoothing).

    For series WITH trend but NO seasonality.
    """

    def __init__(self, alpha=0.3, beta=0.1):
        """
        Parameters:
        -----------
        alpha : float (0 < alpha < 1)
            Level smoothing parameter.
        beta : float (0 < beta < 1)
            Trend smoothing parameter.
        """
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        self.fitted_values = None

    def fit(self, y):
        """Fit the model to time series y."""
        n = len(y)
        self.fitted_values = np.zeros(n)

        # Initialize level and trend
        self.level = y[0]
        self.trend = y[1] - y[0] if n > 1 else 0

        self.fitted_values[0] = self.level

        for t in range(1, n):
            # Store previous level for trend update
            level_prev = self.level

            # Update level
            self.level = self.alpha * y[t] + (1 - self.alpha) * (self.level + self.trend)

            # Update trend
            self.trend = self.beta * (self.level - level_prev) + (1 - self.beta) * self.trend

            self.fitted_values[t] = self.level

        return self

    def forecast(self, h=1):
        """Forecast h steps ahead."""
        return np.array([self.level + i * self.trend for i in range(1, h + 1)])


class HoltWinters:
    """
    Holt-Winters Method (Triple Exponential Smoothing).

    For series WITH trend AND seasonality.
    """

    def __init__(self, alpha=0.3, beta=0.1, gamma=0.1, period=12, seasonal='additive'):
        """
        Parameters:
        -----------
        alpha : float
            Level smoothing parameter.
        beta : float
            Trend smoothing parameter.
        gamma : float
            Seasonal smoothing parameter.
        period : int
            Seasonal period (e.g., 12 for monthly data with yearly seasonality).
        seasonal : str
            'additive' or 'multiplicative'.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.period = period
        self.seasonal = seasonal

        self.level = None
        self.trend = None
        self.seasonals = None
        self.fitted_values = None

    def fit(self, y):
        """Fit the model to time series y."""
        n = len(y)
        m = self.period

        if n < 2 * m:
            raise ValueError(f"Need at least 2 full periods ({2*m} observations)")

        self.fitted_values = np.zeros(n)

        # Initialize level: average of first period
        self.level = np.mean(y[:m])

        # Initialize trend: average slope between first two periods
        self.trend = np.mean([(y[i + m] - y[i]) / m for i in range(m)])

        # Initialize seasonal factors
        if self.seasonal == 'additive':
            self.seasonals = [y[i] - self.level for i in range(m)]
        else:  # multiplicative
            self.seasonals = [y[i] / self.level for i in range(m)]

        # Start fitting from period m
        for t in range(m, n):
            season_idx = t % m
            prev_level = self.level

            if self.seasonal == 'additive':
                # Level update
                self.level = (self.alpha * (y[t] - self.seasonals[season_idx]) +
                              (1 - self.alpha) * (self.level + self.trend))
                # Trend update
                self.trend = (self.beta * (self.level - prev_level) +
                              (1 - self.beta) * self.trend)
                # Seasonal update
                self.seasonals[season_idx] = (self.gamma * (y[t] - self.level) +
                                               (1 - self.gamma) * self.seasonals[season_idx])
                # Fitted value
                self.fitted_values[t] = self.level + self.trend + self.seasonals[season_idx]
            else:  # multiplicative
                # Level update
                self.level = (self.alpha * (y[t] / self.seasonals[season_idx]) +
                              (1 - self.alpha) * (self.level + self.trend))
                # Trend update
                self.trend = (self.beta * (self.level - prev_level) +
                              (1 - self.beta) * self.trend)
                # Seasonal update
                self.seasonals[season_idx] = (self.gamma * (y[t] / self.level) +
                                               (1 - self.gamma) * self.seasonals[season_idx])
                # Fitted value
                self.fitted_values[t] = (self.level + self.trend) * self.seasonals[season_idx]

        # Fill in first m fitted values
        for t in range(m):
            if self.seasonal == 'additive':
                self.fitted_values[t] = np.mean(y[:m]) + self.seasonals[t]
            else:
                self.fitted_values[t] = np.mean(y[:m]) * self.seasonals[t]

        return self

    def forecast(self, h=1):
        """Forecast h steps ahead."""
        forecasts = np.zeros(h)
        m = self.period

        for i in range(h):
            season_idx = i % m

            if self.seasonal == 'additive':
                forecasts[i] = self.level + (i + 1) * self.trend + self.seasonals[season_idx]
            else:
                forecasts[i] = (self.level + (i + 1) * self.trend) * self.seasonals[season_idx]

        return forecasts


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_level_series(n=100, level=10, noise=1.0, seed=42):
    """Generate series with constant level (no trend, no seasonality)."""
    np.random.seed(seed)
    return level + np.random.randn(n) * noise


def generate_trend_series(n=100, level=10, slope=0.1, noise=1.0, seed=42):
    """Generate series with linear trend."""
    np.random.seed(seed)
    t = np.arange(n)
    return level + slope * t + np.random.randn(n) * noise


def generate_seasonal_series(n=100, level=10, amplitude=5, period=12, noise=1.0, seed=42):
    """Generate series with seasonality."""
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    return level + seasonal + np.random.randn(n) * noise


def generate_full_series(n=200, level=10, slope=0.05, amplitude=5, period=12, noise=1.0, seed=42):
    """Generate series with trend AND seasonality."""
    np.random.seed(seed)
    t = np.arange(n)
    trend = level + slope * t
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    return trend + seasonal + np.random.randn(n) * noise


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_forgetting_curve():
    """
    THE FORGETTING CURVE VISUALIZATION

    This is THE key intuition for exponential smoothing:
    α controls how fast you forget the past.
    """
    fig = plt.figure(figsize=(16, 12))

    fig.suptitle('THE FORGETTING CURVE: How Fast Should You Forget?\n'
                 '"α controls the effective memory length of your forecast"',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Weight decay curves for different α
    ax1 = fig.add_subplot(2, 2, 1)

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(alphas)))
    n_weights = 25

    for alpha, color in zip(alphas, colors):
        ses = SimpleExponentialSmoothing(alpha=alpha)
        weights = ses.get_weights(n_weights)

        ax1.plot(range(n_weights), weights, 'o-', color=color, linewidth=2,
                 markersize=6, label=f'α = {alpha}', alpha=0.8)

        # Mark effective memory (where weight drops to 1/e)
        effective_memory = int(1 / alpha) if alpha > 0 else n_weights
        if effective_memory < n_weights:
            ax1.axvline(x=effective_memory, color=color, linestyle=':', alpha=0.5)

    ax1.set_xlabel('Lag (how far back)', fontsize=11)
    ax1.set_ylabel('Weight', fontsize=11)
    ax1.set_title('WEIGHT DECAY: How Much Does Each Past Observation Matter?',
                  fontsize=11, fontweight='bold')
    ax1.legend(title='Smoothing α', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, n_weights - 0.5)

    # Add annotation
    ax1.annotate('High α = Fast decay\n(short memory)',
                 xy=(3, 0.6), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.annotate('Low α = Slow decay\n(long memory)',
                 xy=(15, 0.08), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Panel 2: Cumulative weights
    ax2 = fig.add_subplot(2, 2, 2)

    for alpha, color in zip(alphas, colors):
        ses = SimpleExponentialSmoothing(alpha=alpha)
        weights = ses.get_weights(n_weights)
        cumulative = np.cumsum(weights)

        ax2.plot(range(n_weights), cumulative, '-', color=color, linewidth=2,
                 label=f'α = {alpha}')

        # Mark 90% cumulative weight
        idx_90 = np.argmax(cumulative >= 0.9)
        ax2.scatter([idx_90], [cumulative[idx_90]], color=color, s=100, zorder=5)

    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='90% weight')
    ax2.axhline(y=0.95, color='red', linestyle=':', linewidth=1, label='95% weight')

    ax2.set_xlabel('Number of Past Observations Used', fontsize=11)
    ax2.set_ylabel('Cumulative Weight', fontsize=11)
    ax2.set_title('EFFECTIVE MEMORY: How Many Past Points Really Matter?',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, n_weights - 0.5)
    ax2.set_ylim(0, 1.05)

    # Panel 3: Effect on smoothing (time series comparison)
    ax3 = fig.add_subplot(2, 2, 3)

    np.random.seed(42)
    y = generate_level_series(n=80, level=10, noise=2.0)

    ax3.plot(y, 'k-', linewidth=0.8, alpha=0.5, label='Original', zorder=1)

    for alpha, color in zip([0.1, 0.5, 0.9], [colors[0], colors[2], colors[4]]):
        ses = SimpleExponentialSmoothing(alpha=alpha)
        ses.fit(y)
        ax3.plot(ses.fitted_values, '-', color=color, linewidth=2,
                 label=f'α = {alpha}', zorder=2)

    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('SMOOTHING EFFECT: Low α = Smoother, High α = More Reactive',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Effective memory length
    ax4 = fig.add_subplot(2, 2, 4)

    alpha_range = np.linspace(0.05, 0.95, 50)
    effective_memory = 1 / alpha_range

    ax4.plot(alpha_range, effective_memory, 'b-', linewidth=3)
    ax4.fill_between(alpha_range, 0, effective_memory, alpha=0.2)

    # Mark some key points
    key_alphas = [0.1, 0.2, 0.3, 0.5, 0.9]
    for alpha in key_alphas:
        mem = 1 / alpha
        ax4.scatter([alpha], [mem], color='red', s=100, zorder=5)
        ax4.annotate(f'{mem:.1f} periods', xy=(alpha, mem),
                     xytext=(alpha + 0.05, mem + 1),
                     fontsize=9, ha='left')

    ax4.set_xlabel('Smoothing Parameter α', fontsize=11)
    ax4.set_ylabel('Effective Memory (periods)', fontsize=11)
    ax4.set_title('THE KEY RELATIONSHIP: Memory ≈ 1/α',
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 25)

    # Add insight box
    ax4.text(0.95, 0.95,
             'RULE OF THUMB:\n'
             'α = 0.1 → 10 periods memory\n'
             'α = 0.2 → 5 periods memory\n'
             'α = 0.5 → 2 periods memory\n'
             'α = 0.9 → 1 period memory',
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_alpha_tradeoff():
    """
    Show the bias-variance tradeoff controlled by α.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Generate a series with a level shift
    n = 100
    y = np.concatenate([
        generate_level_series(50, level=10, noise=1.0, seed=42),
        generate_level_series(50, level=15, noise=1.0, seed=43)
    ])

    alphas = [0.1, 0.3, 0.9]
    alpha_names = ['LOW α = 0.1\n(Long memory, smooth)',
                   'MEDIUM α = 0.3\n(Balanced)',
                   'HIGH α = 0.9\n(Short memory, reactive)']

    for col, (alpha, name) in enumerate(zip(alphas, alpha_names)):
        # Fit SES
        ses = SimpleExponentialSmoothing(alpha=alpha)
        ses.fit(y)

        # Top row: Fitted values
        axes[0, col].plot(y, 'k-', linewidth=0.8, alpha=0.4, label='Original')
        axes[0, col].plot(ses.fitted_values, 'b-', linewidth=2, label=f'Smoothed (α={alpha})')
        axes[0, col].axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Level shift')
        axes[0, col].set_title(name, fontsize=11, fontweight='bold')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Value')

        # Calculate error metrics
        residuals = y[1:] - ses.fitted_values[1:]
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))

        # Bottom row: Residuals
        axes[1, col].plot(residuals, 'g-', linewidth=0.8)
        axes[1, col].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        axes[1, col].axvline(x=49, color='red', linestyle='--', alpha=0.7)
        axes[1, col].set_xlabel('Time')
        if col == 0:
            axes[1, col].set_ylabel('Residual (Error)')
        axes[1, col].grid(True, alpha=0.3)

        # Add error metrics
        axes[1, col].text(0.95, 0.95, f'MSE = {mse:.2f}\nMAE = {mae:.2f}',
                          transform=axes[1, col].transAxes, fontsize=10,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Highlight adaptation speed after level shift
        if col == 0:
            axes[1, col].annotate('Slow to adapt\n(high lag error)',
                                  xy=(60, residuals[59]), xytext=(70, -2),
                                  arrowprops=dict(arrowstyle='->', color='red'),
                                  fontsize=9, color='red')
        elif col == 2:
            axes[1, col].annotate('Fast adaptation\nbut noisy',
                                  xy=(55, residuals[54]), xytext=(65, 3),
                                  arrowprops=dict(arrowstyle='->', color='orange'),
                                  fontsize=9, color='orange')

    plt.suptitle('THE BIAS-VARIANCE TRADEOFF: Smooth vs Reactive\n'
                 '"Low α → smooth but slow to adapt | High α → reactive but noisy"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_ses_vs_holt_vs_hw():
    """
    Compare the three exponential smoothing methods.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Column 1: Level only (SES appropriate)
    y_level = generate_level_series(n=100, level=10, noise=1.5)

    # Column 2: Level + Trend (Holt appropriate)
    y_trend = generate_trend_series(n=100, level=10, slope=0.15, noise=1.5)

    # Column 3: Level + Trend + Seasonality (HW appropriate)
    y_seasonal = generate_full_series(n=120, level=10, slope=0.1, amplitude=4,
                                       period=12, noise=1.0)

    series_list = [
        (y_level, 'LEVEL ONLY\n(No trend, no seasonality)', 'Simple ES wins'),
        (y_trend, 'LEVEL + TREND\n(No seasonality)', 'Holt wins'),
        (y_seasonal, 'LEVEL + TREND + SEASONAL\n(Full complexity)', 'Holt-Winters wins')
    ]

    for col, (y, title, winner) in enumerate(series_list):
        n = len(y)
        train_end = int(n * 0.8)
        y_train = y[:train_end]
        y_test = y[train_end:]
        h = len(y_test)

        # Row 1: Data
        axes[0, col].plot(range(train_end), y_train, 'b-', linewidth=1, label='Train')
        axes[0, col].plot(range(train_end, n), y_test, 'g-', linewidth=1.5, label='Test')
        axes[0, col].axvline(x=train_end, color='gray', linestyle='--', alpha=0.7)
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Data')

        # Row 2: Forecasts
        # SES
        ses = SimpleExponentialSmoothing(alpha=0.3)
        ses.fit(y_train)
        ses_forecast = ses.forecast(h)

        # Holt
        holt = HoltLinear(alpha=0.3, beta=0.1)
        holt.fit(y_train)
        holt_forecast = holt.forecast(h)

        # Holt-Winters (only for seasonal)
        if col == 2:
            hw = HoltWinters(alpha=0.3, beta=0.1, gamma=0.1, period=12)
            hw.fit(y_train)
            hw_forecast = hw.forecast(h)
        else:
            hw_forecast = None

        axes[1, col].plot(range(train_end, n), y_test, 'k-', linewidth=2,
                          label='Actual', alpha=0.7)
        axes[1, col].plot(range(train_end, n), ses_forecast, 'r--', linewidth=2,
                          label='SES', alpha=0.8)
        axes[1, col].plot(range(train_end, n), holt_forecast, 'b--', linewidth=2,
                          label='Holt', alpha=0.8)
        if hw_forecast is not None:
            axes[1, col].plot(range(train_end, n), hw_forecast, 'g--', linewidth=2,
                              label='Holt-Winters', alpha=0.8)
        axes[1, col].axvline(x=train_end, color='gray', linestyle='--', alpha=0.7)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel('Forecast')

        # Row 3: Errors
        ses_mse = np.mean((y_test - ses_forecast)**2)
        holt_mse = np.mean((y_test - holt_forecast)**2)
        hw_mse = np.mean((y_test - hw_forecast)**2) if hw_forecast is not None else np.inf

        methods = ['SES', 'Holt', 'H-W'] if hw_forecast is not None else ['SES', 'Holt']
        mses = [ses_mse, holt_mse, hw_mse] if hw_forecast is not None else [ses_mse, holt_mse]
        colors = ['red', 'blue', 'green'] if hw_forecast is not None else ['red', 'blue']

        # Highlight winner
        bar_colors = ['lightgreen' if mse == min(mses) else c for c, mse in zip(colors, mses)]

        axes[2, col].bar(methods, mses, color=bar_colors, alpha=0.7, edgecolor='black')
        axes[2, col].set_ylabel('MSE (Test)')
        axes[2, col].set_title(f'Winner: {winner}', fontsize=10, color='green')
        axes[2, col].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, (method, mse) in enumerate(zip(methods, mses)):
            axes[2, col].text(i, mse + 0.1, f'{mse:.2f}', ha='center', fontsize=9)

    plt.suptitle('CHOOSING THE RIGHT METHOD: Match Complexity to Data Structure\n'
                 '"Simple models for simple patterns, complex models only when needed"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_parameter_sensitivity():
    """
    Show how changing parameters affects forecasts.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Generate data with trend and seasonality
    n = 120
    y = generate_full_series(n=n, level=10, slope=0.1, amplitude=4, period=12, noise=0.8)

    train_end = 100
    y_train = y[:train_end]
    y_test = y[train_end:]
    h = len(y_test)

    # Row 1: Varying α (level smoothing)
    alphas = [0.1, 0.3, 0.7]
    for col, alpha in enumerate(alphas):
        hw = HoltWinters(alpha=alpha, beta=0.1, gamma=0.1, period=12)
        hw.fit(y_train)
        forecast = hw.forecast(h)

        axes[0, col].plot(range(train_end), y_train, 'b-', linewidth=0.8, alpha=0.5)
        axes[0, col].plot(range(train_end, n), y_test, 'k-', linewidth=2, label='Actual')
        axes[0, col].plot(range(train_end, n), forecast, 'r--', linewidth=2, label='Forecast')
        axes[0, col].axvline(x=train_end, color='gray', linestyle='--', alpha=0.5)

        mse = np.mean((y_test - forecast)**2)
        axes[0, col].set_title(f'α = {alpha} (Level)\nMSE = {mse:.2f}', fontsize=11, fontweight='bold')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Value')

    # Row 2: Varying γ (seasonal smoothing)
    gammas = [0.05, 0.2, 0.5]
    for col, gamma in enumerate(gammas):
        hw = HoltWinters(alpha=0.3, beta=0.1, gamma=gamma, period=12)
        hw.fit(y_train)
        forecast = hw.forecast(h)

        axes[1, col].plot(range(train_end), y_train, 'b-', linewidth=0.8, alpha=0.5)
        axes[1, col].plot(range(train_end, n), y_test, 'k-', linewidth=2, label='Actual')
        axes[1, col].plot(range(train_end, n), forecast, 'g--', linewidth=2, label='Forecast')
        axes[1, col].axvline(x=train_end, color='gray', linestyle='--', alpha=0.5)

        mse = np.mean((y_test - forecast)**2)
        axes[1, col].set_title(f'γ = {gamma} (Seasonal)\nMSE = {mse:.2f}', fontsize=11, fontweight='bold')
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_xlabel('Time')
        if col == 0:
            axes[1, col].set_ylabel('Value')

    plt.suptitle('PARAMETER SENSITIVITY: How Each Parameter Affects Forecasts\n'
                 '"α controls level adaptation, β controls trend adaptation, γ controls seasonal adaptation"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_optimal_alpha_selection():
    """
    Show how to select optimal α using cross-validation.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate test series
    y = generate_level_series(n=150, level=10, noise=2.0)

    # Split: train / validation / test
    train_end = 100
    val_end = 130

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    # Panel 1: Try different alphas
    ax1 = axes[0, 0]
    alphas = np.linspace(0.05, 0.95, 30)
    val_mses = []

    for alpha in alphas:
        ses = SimpleExponentialSmoothing(alpha=alpha)
        ses.fit(y_train)
        forecast = ses.forecast(len(y_val))
        mse = np.mean((y_val - forecast)**2)
        val_mses.append(mse)

    ax1.plot(alphas, val_mses, 'b-', linewidth=2)
    best_alpha = alphas[np.argmin(val_mses)]
    ax1.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
                label=f'Best α = {best_alpha:.2f}')
    ax1.scatter([best_alpha], [min(val_mses)], color='green', s=150, zorder=5)

    ax1.set_xlabel('α', fontsize=11)
    ax1.set_ylabel('Validation MSE', fontsize=11)
    ax1.set_title('STEP 1: Find Optimal α on Validation Set', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Compare forecasts
    ax2 = axes[0, 1]

    # Suboptimal alphas
    alpha_low = 0.1
    alpha_high = 0.9

    ses_best = SimpleExponentialSmoothing(alpha=best_alpha)
    ses_best.fit(y[:val_end])
    forecast_best = ses_best.forecast(len(y_test))

    ses_low = SimpleExponentialSmoothing(alpha=alpha_low)
    ses_low.fit(y[:val_end])
    forecast_low = ses_low.forecast(len(y_test))

    ses_high = SimpleExponentialSmoothing(alpha=alpha_high)
    ses_high.fit(y[:val_end])
    forecast_high = ses_high.forecast(len(y_test))

    ax2.plot(range(val_end, len(y)), y_test, 'k-', linewidth=2, label='Actual')
    ax2.plot(range(val_end, len(y)), forecast_low, 'r--', linewidth=2,
             label=f'α={alpha_low} (too smooth)')
    ax2.plot(range(val_end, len(y)), forecast_best, 'g-', linewidth=2,
             label=f'α={best_alpha:.2f} (optimal)')
    ax2.plot(range(val_end, len(y)), forecast_high, 'b--', linewidth=2,
             label=f'α={alpha_high} (too reactive)')

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('STEP 2: Compare Forecasts on Test Set', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Error decomposition
    ax3 = axes[1, 0]

    test_mse_low = np.mean((y_test - forecast_low)**2)
    test_mse_best = np.mean((y_test - forecast_best)**2)
    test_mse_high = np.mean((y_test - forecast_high)**2)

    # Compute bias and variance components
    methods = [f'α={alpha_low}', f'α={best_alpha:.2f}', f'α={alpha_high}']
    forecasts = [forecast_low, forecast_best, forecast_high]
    mses = [test_mse_low, test_mse_best, test_mse_high]

    x = np.arange(len(methods))
    colors = ['red', 'green', 'blue']

    ax3.bar(x, mses, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel('Test MSE', fontsize=11)
    ax3.set_title('RESULT: Optimal α Minimizes Error', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for i, mse in enumerate(mses):
        ax3.text(i, mse + 0.1, f'{mse:.2f}', ha='center', fontsize=10)

    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = """
    ╔═══════════════════════════════════════════════════════════╗
    ║         SELECTING THE OPTIMAL SMOOTHING PARAMETER         ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  METHOD: Time Series Cross-Validation                     ║
    ║                                                           ║
    ║  1. Split data: Train | Validation | Test                 ║
    ║                                                           ║
    ║  2. For each candidate α:                                 ║
    ║     • Fit on training data                                ║
    ║     • Forecast validation period                          ║
    ║     • Compute validation error                            ║
    ║                                                           ║
    ║  3. Select α with lowest validation error                 ║
    ║                                                           ║
    ║  4. Retrain on train+validation, evaluate on test         ║
    ║                                                           ║
    ║  ─────────────────────────────────────────────────────    ║
    ║                                                           ║
    ║  IMPORTANT: Never tune on test data!                      ║
    ║  That would give you an optimistic (biased) error         ║
    ║  estimate that won't generalize to new data.              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('HOW TO SELECT OPTIMAL α: Cross-Validation Approach\n'
                 '"Let the data tell you how fast to forget"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_alpha_range():
    """
    Systematic study of α across its full range.
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of α Across Full Range")
    print("="*60)

    np.random.seed(42)
    y = generate_level_series(n=200, level=10, noise=2.0)

    train_end = 150
    y_train = y[:train_end]
    y_test = y[train_end:]

    print(f"\nData: {len(y_train)} train, {len(y_test)} test observations")
    print("-" * 50)

    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]

    for alpha in alphas:
        ses = SimpleExponentialSmoothing(alpha=alpha)
        ses.fit(y_train)
        forecast = ses.forecast(len(y_test))

        mse = np.mean((y_test - forecast)**2)
        mae = np.mean(np.abs(y_test - forecast))
        effective_memory = 1 / alpha

        print(f"α = {alpha:.2f}: MSE = {mse:.2f}, MAE = {mae:.2f}, "
              f"Effective memory ≈ {effective_memory:.1f} periods")

    print("\n→ Extreme α values (too low or too high) perform worse")
    print("→ Optimal α depends on the signal-to-noise ratio in the data")


def ablation_trend_detection():
    """
    When does Holt outperform SES?
    """
    print("\n" + "="*60)
    print("ABLATION: SES vs Holt on Different Trend Strengths")
    print("="*60)

    np.random.seed(42)

    slopes = [0.0, 0.05, 0.1, 0.2, 0.3]

    print("\nComparing SES vs Holt on series with different trend strengths:")
    print("-" * 60)

    for slope in slopes:
        y = generate_trend_series(n=150, level=10, slope=slope, noise=1.5)

        train_end = 100
        y_train = y[:train_end]
        y_test = y[train_end:]

        # SES
        ses = SimpleExponentialSmoothing(alpha=0.3)
        ses.fit(y_train)
        ses_forecast = ses.forecast(len(y_test))
        ses_mse = np.mean((y_test - ses_forecast)**2)

        # Holt
        holt = HoltLinear(alpha=0.3, beta=0.1)
        holt.fit(y_train)
        holt_forecast = holt.forecast(len(y_test))
        holt_mse = np.mean((y_test - holt_forecast)**2)

        winner = "Holt" if holt_mse < ses_mse else "SES"
        improvement = (ses_mse - holt_mse) / ses_mse * 100 if ses_mse > 0 else 0

        print(f"Slope = {slope:.2f}: SES MSE = {ses_mse:.2f}, Holt MSE = {holt_mse:.2f}, "
              f"Winner: {winner} ({improvement:+.1f}%)")

    print("\n→ Holt wins when there's a strong trend")
    print("→ SES is adequate when trend is weak or absent")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("EXPONENTIAL SMOOTHING — Paradigm: WEIGHTED FORGETTING")
    print("="*70)

    print("""

THE CORE IDEA:
    Recent observations are more relevant than distant ones.
    Weight decays exponentially: weight(k) = α × (1-α)^k

THE FORGETTING CURVE:
    α controls how fast you forget the past.

    α → 1: Short memory, reactive, noisy
    α → 0: Long memory, smooth, slow to adapt

    Effective memory ≈ 1/α periods

THE THREE METHODS:
    1. Simple ES (SES): Level only — ŷ_{t+1} = α×y_t + (1-α)×ŷ_t
    2. Holt: Level + Trend — tracks linear trends
    3. Holt-Winters: Level + Trend + Seasonal — full model

CHOOSING A METHOD:
    • No trend, no seasonality → SES
    • Trend, no seasonality → Holt
    • Trend + Seasonality → Holt-Winters

    """)

    # Run ablations
    ablation_alpha_range()
    ablation_trend_detection()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. Forgetting curve
    fig1 = visualize_forgetting_curve()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/03_es_forgetting_curve.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. Alpha tradeoff
    fig2 = visualize_alpha_tradeoff()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/03_es_alpha_tradeoff.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Method comparison
    fig3 = visualize_ses_vs_holt_vs_hw()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/03_es_method_comparison.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. Parameter sensitivity
    fig4 = visualize_parameter_sensitivity()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/03_es_parameter_sensitivity.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    # 5. Optimal alpha selection
    fig5 = visualize_optimal_alpha_selection()
    save_path5 = '/Users/sid47/ML Algorithms/11-time-series/03_es_optimal_alpha.png'
    fig5.savefig(save_path5, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path5}")
    plt.close(fig5)

    print("\n" + "="*60)
    print("SUMMARY: Exponential Smoothing")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 03_es_forgetting_curve.png      — THE key insight: α controls memory
    2. 03_es_alpha_tradeoff.png        — Bias-variance: smooth vs reactive
    3. 03_es_method_comparison.png     — SES vs Holt vs Holt-Winters
    4. 03_es_parameter_sensitivity.png — How each parameter affects forecasts
    5. 03_es_optimal_alpha.png         — Cross-validation for parameter selection

KEY TAKEAWAYS:
    1. α controls effective memory length: memory ≈ 1/α
    2. Low α = smooth but slow; High α = reactive but noisy
    3. Match method complexity to data complexity
    4. Use cross-validation to select optimal parameters
    5. Simple methods often work surprisingly well!

NEXT: ARIMA — "Predict from your past, fix your mistakes"
    """)
