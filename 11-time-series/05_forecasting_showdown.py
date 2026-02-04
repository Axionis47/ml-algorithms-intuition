"""
FORECASTING SHOWDOWN — Paradigm: KNOW YOUR WEAPONS

===============================================================
THE BIG QUESTION
===============================================================

"Which forecasting method should I use?"

ANSWER: It depends on your data!

This module puts all methods head-to-head on different scenarios
to build intuition about when each method excels and fails.

===============================================================
THE CONTENDERS
===============================================================

1. NAIVE METHODS (Baselines)
   - Last value: ŷ_{t+1} = y_t
   - Seasonal naive: ŷ_{t+1} = y_{t-period}
   - Mean: ŷ_{t+1} = mean(y)

2. EXPONENTIAL SMOOTHING
   - Simple ES: weighted average of past
   - Holt: adds trend tracking
   - Holt-Winters: adds seasonality

3. ARIMA
   - AR: predict from past values
   - MA: correct from past errors
   - ARIMA: full model with differencing

4. (Future) DEEP LEARNING
   - RNN/LSTM: neural sequence models
   - For comparison with traditional methods

===============================================================
THE SCENARIOS
===============================================================

1. STATIONARY (no trend, no seasonality)
   → Simple ES or ARIMA should win

2. TRENDING (linear or exponential)
   → Holt or ARIMA with d>0 should win

3. SEASONAL (repeating patterns)
   → Holt-Winters or SARIMA should win

4. TRENDING + SEASONAL
   → Holt-Winters or SARIMA should win

5. NOISY (high noise-to-signal ratio)
   → Simpler models often win (less overfitting)

6. STRUCTURAL BREAK (sudden change)
   → All methods struggle! Important to know limits.

===============================================================
THE KEY INSIGHT
===============================================================

THERE IS NO UNIVERSALLY BEST METHOD!

- Complex models can OVERFIT on simple/noisy data
- Simple models UNDERFIT on complex data
- Match model complexity to data complexity

The best forecasters know WHEN to use each tool.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms/11-time-series')

# Import our implementations
try:
    from _03_exponential_smoothing import (
        SimpleExponentialSmoothing, HoltLinear, HoltWinters
    )
    from _04_arima import ARIMA, AR
except ImportError:
    # Direct import when running as main module
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
# BASELINE METHODS
# ============================================================

class NaiveForecaster:
    """
    Naive forecast: ŷ_{t+h} = y_t (last observed value)

    This is THE baseline. Any good method must beat this!
    """
    def __init__(self):
        self.last_value = None

    def fit(self, y):
        self.last_value = y[-1]
        return self

    def forecast(self, h=1):
        return np.full(h, self.last_value)


class SeasonalNaiveForecaster:
    """
    Seasonal naive: ŷ_{t+h} = y_{t+h-period}

    Useful baseline for seasonal data.
    """
    def __init__(self, period=12):
        self.period = period
        self.seasonal_values = None

    def fit(self, y):
        self.seasonal_values = y[-self.period:]
        return self

    def forecast(self, h=1):
        forecasts = []
        for i in range(h):
            idx = i % self.period
            forecasts.append(self.seasonal_values[idx])
        return np.array(forecasts)


class MeanForecaster:
    """
    Mean forecast: ŷ_{t+h} = mean(y)

    Simple baseline assuming mean reversion.
    """
    def __init__(self):
        self.mean_value = None

    def fit(self, y):
        self.mean_value = np.mean(y)
        return self

    def forecast(self, h=1):
        return np.full(h, self.mean_value)


class DriftForecaster:
    """
    Drift forecast: ŷ_{t+h} = y_t + h × (y_t - y_1) / (t - 1)

    Extrapolates the average trend.
    """
    def __init__(self):
        self.last_value = None
        self.drift = None

    def fit(self, y):
        n = len(y)
        self.last_value = y[-1]
        self.drift = (y[-1] - y[0]) / (n - 1) if n > 1 else 0
        return self

    def forecast(self, h=1):
        return np.array([self.last_value + i * self.drift for i in range(1, h + 1)])


# ============================================================
# DATA GENERATORS FOR DIFFERENT SCENARIOS
# ============================================================

def generate_stationary(n=200, level=10, noise=2.0, seed=42):
    """Stationary series (no trend, no seasonality)."""
    np.random.seed(seed)
    return level + np.random.randn(n) * noise


def generate_trending(n=200, level=10, slope=0.1, noise=1.5, seed=42):
    """Series with linear trend."""
    np.random.seed(seed)
    t = np.arange(n)
    return level + slope * t + np.random.randn(n) * noise


def generate_seasonal(n=200, level=10, amplitude=5, period=12, noise=1.0, seed=42):
    """Series with seasonality (no trend)."""
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    return level + seasonal + np.random.randn(n) * noise


def generate_trending_seasonal(n=200, level=10, slope=0.05, amplitude=4,
                                period=12, noise=1.0, seed=42):
    """Series with both trend and seasonality."""
    np.random.seed(seed)
    t = np.arange(n)
    trend = level + slope * t
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    return trend + seasonal + np.random.randn(n) * noise


def generate_noisy(n=200, level=10, noise=5.0, seed=42):
    """High-noise stationary series."""
    np.random.seed(seed)
    # Add some weak AR structure
    y = np.zeros(n)
    y[0] = level + np.random.randn() * noise
    for t in range(1, n):
        y[t] = 0.3 * (y[t-1] - level) + level + np.random.randn() * noise
    return y


def generate_structural_break(n=200, level1=10, level2=20, break_point=100,
                              noise=1.5, seed=42):
    """Series with a structural break (sudden level shift)."""
    np.random.seed(seed)
    y = np.zeros(n)
    y[:break_point] = level1 + np.random.randn(break_point) * noise
    y[break_point:] = level2 + np.random.randn(n - break_point) * noise
    return y


def generate_nonlinear(n=200, seed=42):
    """Nonlinear series (exponential growth with noise)."""
    np.random.seed(seed)
    t = np.arange(n)
    trend = np.exp(0.02 * t)
    noise = np.random.randn(n) * 0.5 * (1 + t / n)  # Heteroscedastic noise
    return trend + noise


# ============================================================
# EVALUATION METRICS
# ============================================================

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


# ============================================================
# SHOWDOWN FRAMEWORK
# ============================================================

def run_showdown(y, train_ratio=0.8, period=12, scenario_name=""):
    """
    Run all methods on a time series and compare performance.

    Returns dict of {method_name: {'mse': ..., 'mae': ..., 'forecast': ...}}
    """
    n = len(y)
    train_end = int(n * train_ratio)
    y_train = y[:train_end]
    y_test = y[train_end:]
    h = len(y_test)

    results = {}

    # 1. Baselines
    methods = {
        'Naive': NaiveForecaster(),
        'Mean': MeanForecaster(),
        'Drift': DriftForecaster(),
        'Seasonal Naive': SeasonalNaiveForecaster(period=period),
    }

    # 2. Exponential Smoothing
    methods['SES (α=0.3)'] = SimpleExponentialSmoothing(alpha=0.3)
    methods['Holt'] = HoltLinear(alpha=0.3, beta=0.1)

    # Try Holt-Winters only if we have enough data
    if train_end >= 2 * period:
        methods['Holt-Winters'] = HoltWinters(alpha=0.3, beta=0.1, gamma=0.1, period=period)

    # 3. ARIMA family
    methods['AR(1)'] = AR(p=1)
    methods['AR(2)'] = AR(p=2)
    methods['ARIMA(1,1,0)'] = ARIMA(p=1, d=1, q=0)
    methods['ARIMA(1,1,1)'] = ARIMA(p=1, d=1, q=1)

    # Run each method
    for name, model in methods.items():
        try:
            model.fit(y_train)
            forecast = model.forecast(h)

            # Ensure forecast length matches
            if len(forecast) != h:
                forecast = forecast[:h] if len(forecast) > h else np.pad(forecast, (0, h - len(forecast)), constant_values=forecast[-1])

            results[name] = {
                'mse': mse(y_test, forecast),
                'mae': mae(y_test, forecast),
                'rmse': rmse(y_test, forecast),
                'forecast': forecast
            }
        except Exception as e:
            results[name] = {
                'mse': np.inf,
                'mae': np.inf,
                'rmse': np.inf,
                'forecast': np.full(h, np.nan),
                'error': str(e)
            }

    return results, y_train, y_test


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_scenario_showdown():
    """
    The main showdown visualization across all scenarios.
    """
    np.random.seed(42)

    scenarios = {
        'STATIONARY\n(No trend, no season)': generate_stationary(200, noise=2.0),
        'TRENDING\n(Linear trend)': generate_trending(200, slope=0.15),
        'SEASONAL\n(Period=12)': generate_seasonal(200, period=12, amplitude=5),
        'TREND + SEASON\n(Both)': generate_trending_seasonal(200, slope=0.08, period=12),
        'HIGH NOISE\n(SNR low)': generate_noisy(200, noise=5.0),
        'STRUCTURAL BREAK\n(Level shift)': generate_structural_break(200, level1=10, level2=20),
    }

    fig = plt.figure(figsize=(18, 16))

    fig.suptitle('THE FORECASTING SHOWDOWN: Which Method Wins When?\n'
                 '"There is no universally best method — match complexity to data"',
                 fontsize=14, fontweight='bold', y=0.98)

    n_scenarios = len(scenarios)
    rows = 3
    cols = 2

    for idx, (scenario_name, y) in enumerate(scenarios.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)

        # Run showdown
        results, y_train, y_test = run_showdown(y, train_ratio=0.8, period=12)

        # Find best method
        mses = {name: res['mse'] for name, res in results.items() if res['mse'] < np.inf}
        if mses:
            best_method = min(mses, key=mses.get)
            best_mse = mses[best_method]
        else:
            best_method = "None"
            best_mse = np.inf

        # Plot data
        train_end = len(y_train)
        n = len(y)
        ax.plot(range(train_end), y_train, 'b-', linewidth=1, alpha=0.6, label='Train')
        ax.plot(range(train_end, n), y_test, 'k-', linewidth=2, label='Actual')

        # Plot top 3 forecasts
        sorted_methods = sorted(mses.items(), key=lambda x: x[1])[:3]
        colors = ['green', 'orange', 'red']

        for (method_name, method_mse), color in zip(sorted_methods, colors):
            forecast = results[method_name]['forecast']
            linestyle = '-' if method_name == best_method else '--'
            linewidth = 2.5 if method_name == best_method else 1.5
            ax.plot(range(train_end, n), forecast, linestyle, color=color,
                    linewidth=linewidth, alpha=0.8,
                    label=f'{method_name} (MSE={method_mse:.1f})')

        ax.axvline(x=train_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(f'{scenario_name}\nWinner: {best_method}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

        if idx % 2 == 0:
            ax.set_ylabel('Value')
        if idx >= 4:
            ax.set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_method_ranking():
    """
    Bar chart showing which method wins in each scenario.
    """
    np.random.seed(42)

    scenarios = {
        'Stationary': generate_stationary,
        'Trending': generate_trending,
        'Seasonal': generate_seasonal,
        'Trend+Season': generate_trending_seasonal,
        'High Noise': generate_noisy,
        'Struct. Break': generate_structural_break,
    }

    # Run multiple simulations
    n_sims = 10
    method_wins = {}
    method_avg_rank = {}

    all_methods = ['Naive', 'Mean', 'Drift', 'Seasonal Naive', 'SES (α=0.3)', 'Holt',
                   'Holt-Winters', 'AR(1)', 'AR(2)', 'ARIMA(1,1,0)', 'ARIMA(1,1,1)']

    for method in all_methods:
        method_wins[method] = {s: 0 for s in scenarios.keys()}
        method_avg_rank[method] = {s: [] for s in scenarios.keys()}

    for scenario_name, generator in scenarios.items():
        for sim in range(n_sims):
            if scenario_name == 'Struct. Break':
                y = generator(200, seed=42 + sim)
            else:
                y = generator(200, seed=42 + sim)

            results, _, _ = run_showdown(y, train_ratio=0.8, period=12)

            # Rank methods
            mses = [(name, res['mse']) for name, res in results.items() if res['mse'] < np.inf]
            mses_sorted = sorted(mses, key=lambda x: x[1])

            if mses_sorted:
                winner = mses_sorted[0][0]
                method_wins[winner][scenario_name] += 1

            # Track ranks
            for rank, (method, _) in enumerate(mses_sorted):
                method_avg_rank[method][scenario_name].append(rank + 1)

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: Win counts
    ax1 = axes[0]

    scenario_list = list(scenarios.keys())
    x = np.arange(len(scenario_list))
    width = 0.08

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods)))

    for i, method in enumerate(all_methods):
        wins = [method_wins[method][s] for s in scenario_list]
        offset = (i - len(all_methods) / 2) * width
        bars = ax1.bar(x + offset, wins, width, label=method, color=colors[i], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_list, fontsize=10)
    ax1.set_ylabel('Number of Wins (out of 10 simulations)', fontsize=11)
    ax1.set_title('WHICH METHOD WINS MOST OFTEN?\n'
                  '(Across 10 random simulations per scenario)',
                  fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Average rank
    ax2 = axes[1]

    # Compute average rank per scenario
    avg_ranks = np.zeros((len(all_methods), len(scenario_list)))
    for i, method in enumerate(all_methods):
        for j, scenario in enumerate(scenario_list):
            ranks = method_avg_rank[method][scenario]
            avg_ranks[i, j] = np.mean(ranks) if ranks else len(all_methods)

    # Heatmap
    im = ax2.imshow(avg_ranks, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(all_methods))

    ax2.set_xticks(np.arange(len(scenario_list)))
    ax2.set_yticks(np.arange(len(all_methods)))
    ax2.set_xticklabels(scenario_list, fontsize=10)
    ax2.set_yticklabels(all_methods, fontsize=9)

    # Add text annotations
    for i in range(len(all_methods)):
        for j in range(len(scenario_list)):
            text = ax2.text(j, i, f'{avg_ranks[i, j]:.1f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if avg_ranks[i, j] > 5 else 'black')

    ax2.set_title('AVERAGE RANK BY SCENARIO\n'
                  '(1 = best, green | 10 = worst, red)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Average Rank')

    plt.suptitle('METHOD PERFORMANCE ACROSS SCENARIOS\n'
                 '"Know which tool to use for which job"',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def visualize_failure_modes():
    """
    Show where each method fails spectacularly.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Failure 1: SES on trending data
    y_trend = generate_trending(150, slope=0.2, noise=1.0)
    train_end = 100

    ses = SimpleExponentialSmoothing(alpha=0.3)
    ses.fit(y_trend[:train_end])
    ses_forecast = ses.forecast(50)

    axes[0, 0].plot(range(train_end), y_trend[:train_end], 'b-', linewidth=1)
    axes[0, 0].plot(range(train_end, 150), y_trend[train_end:], 'k-', linewidth=2, label='Actual')
    axes[0, 0].plot(range(train_end, 150), ses_forecast, 'r--', linewidth=2, label='SES Forecast')
    axes[0, 0].axvline(x=train_end, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].set_title('FAILURE: SES on Trending Data\n"SES assumes no trend!"', fontsize=10, fontweight='bold', color='red')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Failure 2: ARIMA without differencing on trend
    arima_no_d = ARIMA(p=2, d=0, q=0)
    arima_no_d.fit(y_trend[:train_end])
    arima_forecast = arima_no_d.forecast(50)

    axes[0, 1].plot(range(train_end), y_trend[:train_end], 'b-', linewidth=1)
    axes[0, 1].plot(range(train_end, 150), y_trend[train_end:], 'k-', linewidth=2, label='Actual')
    axes[0, 1].plot(range(train_end, 150), arima_forecast, 'r--', linewidth=2, label='ARIMA(2,0,0)')
    axes[0, 1].axvline(x=train_end, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('FAILURE: ARIMA(p,0,q) on Trend\n"Need d>0 for trends!"', fontsize=10, fontweight='bold', color='red')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Failure 3: All methods on structural break
    y_break = generate_structural_break(150, level1=10, level2=25, break_point=100)

    methods = {
        'SES': SimpleExponentialSmoothing(alpha=0.3),
        'Holt': HoltLinear(alpha=0.3, beta=0.1),
        'ARIMA': ARIMA(p=1, d=1, q=0),
    }

    axes[0, 2].plot(range(100), y_break[:100], 'b-', linewidth=1)
    axes[0, 2].plot(range(100, 150), y_break[100:], 'k-', linewidth=2, label='Actual')

    colors = ['red', 'orange', 'purple']
    for (name, model), color in zip(methods.items(), colors):
        model.fit(y_break[:100])
        forecast = model.forecast(50)
        axes[0, 2].plot(range(100, 150), forecast, '--', color=color, linewidth=2, label=name)

    axes[0, 2].axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    axes[0, 2].set_title('FAILURE: All Methods on Structural Break\n"Past doesn\'t predict regime changes!"',
                         fontsize=10, fontweight='bold', color='red')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Failure 4: Complex model on simple data (overfitting)
    y_simple = generate_stationary(150, noise=3.0)

    # Compare simple vs complex
    naive = NaiveForecaster()
    naive.fit(y_simple[:100])
    naive_forecast = naive.forecast(50)

    arima_complex = ARIMA(p=3, d=1, q=2)
    arima_complex.fit(y_simple[:100])
    complex_forecast = arima_complex.forecast(50)

    axes[1, 0].plot(range(100), y_simple[:100], 'b-', linewidth=1)
    axes[1, 0].plot(range(100, 150), y_simple[100:], 'k-', linewidth=2, label='Actual')
    axes[1, 0].plot(range(100, 150), naive_forecast, 'g--', linewidth=2, label=f'Naive (MSE={mse(y_simple[100:], naive_forecast):.1f})')
    axes[1, 0].plot(range(100, 150), complex_forecast, 'r--', linewidth=2, label=f'ARIMA(3,1,2) (MSE={mse(y_simple[100:], complex_forecast):.1f})')
    axes[1, 0].axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].set_title('FAILURE: Complex Model on Simple Data\n"Overfitting to noise!"',
                         fontsize=10, fontweight='bold', color='red')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Failure 5: Non-seasonal method on seasonal data
    y_seasonal = generate_seasonal(150, period=12, amplitude=5, noise=0.5)

    arima_no_season = ARIMA(p=1, d=0, q=1)
    arima_no_season.fit(y_seasonal[:100])
    no_season_forecast = arima_no_season.forecast(50)

    hw = HoltWinters(alpha=0.3, beta=0.1, gamma=0.3, period=12)
    hw.fit(y_seasonal[:100])
    hw_forecast = hw.forecast(50)

    axes[1, 1].plot(range(100), y_seasonal[:100], 'b-', linewidth=1)
    axes[1, 1].plot(range(100, 150), y_seasonal[100:], 'k-', linewidth=2, label='Actual')
    axes[1, 1].plot(range(100, 150), no_season_forecast, 'r--', linewidth=2,
                    label=f'ARIMA(1,0,1) (MSE={mse(y_seasonal[100:], no_season_forecast):.1f})')
    axes[1, 1].plot(range(100, 150), hw_forecast, 'g--', linewidth=2,
                    label=f'Holt-Winters (MSE={mse(y_seasonal[100:], hw_forecast):.1f})')
    axes[1, 1].axvline(x=100, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_title('FAILURE: Ignoring Seasonality\n"Must model the cycles!"',
                         fontsize=10, fontweight='bold', color='red')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: Summary of lessons
    axes[1, 2].axis('off')

    lessons = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           LESSONS FROM FAILURE MODES                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  1. SES FAILS ON TRENDS                                   ║
    ║     → Use Holt or ARIMA with d>0                          ║
    ║                                                           ║
    ║  2. ARIMA(p,0,q) FAILS ON TRENDS                          ║
    ║     → Set d=1 (or d=2 for quadratic)                      ║
    ║                                                           ║
    ║  3. ALL METHODS FAIL ON STRUCTURAL BREAKS                 ║
    ║     → No statistical method predicts regime changes       ║
    ║     → Need domain knowledge or change detection           ║
    ║                                                           ║
    ║  4. COMPLEX MODELS OVERFIT ON SIMPLE DATA                 ║
    ║     → Simpler is often better when SNR is low             ║
    ║     → Use cross-validation to select complexity           ║
    ║                                                           ║
    ║  5. NON-SEASONAL MODELS MISS CYCLES                       ║
    ║     → Always check for seasonality first                  ║
    ║     → Use ACF to detect seasonal periods                  ║
    ║                                                           ║
    ║  ─────────────────────────────────────────────────────    ║
    ║  THE META-LESSON: Know your data BEFORE choosing model!   ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    axes[1, 2].text(0.05, 0.95, lessons, transform=axes[1, 2].transAxes, fontsize=9,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('FAILURE MODES: Learning from What Goes Wrong\n'
                 '"Every method has its Achilles heel"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_decision_tree():
    """
    Visual decision tree for method selection.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    # Create a text-based decision tree
    decision_tree = """

                            ╔═══════════════════════════════════════╗
                            ║     FORECASTING METHOD SELECTION      ║
                            ║           DECISION TREE               ║
                            ╚═══════════════════════════════════════╝

                                        START HERE
                                            │
                            ┌───────────────┴───────────────┐
                            │     Is there a TREND?          │
                            └───────────────┬───────────────┘
                                    ┌───────┴───────┐
                                   YES              NO
                                    │               │
                    ┌───────────────┴───┐     ┌────┴────────────────┐
                    │ Is there SEASONALITY? │     │ Is there SEASONALITY? │
                    └───────────────┬───┘     └────┬────────────────┘
                            ┌───────┴───────┐        ┌───┴───┐
                           YES              NO       YES      NO
                            │               │        │        │
                    ┌───────┴───────┐ ┌─────┴─────┐ │  ┌─────┴─────┐
                    │               │ │           │ │  │           │
              HOLT-WINTERS    HOLT or    Seasonal    SES or
              or SARIMA      ARIMA(p,d,0) ARIMA    ARIMA(p,0,q)


    ════════════════════════════════════════════════════════════════════════════

                                QUICK REFERENCE

    ┌────────────────────┬────────────────────┬────────────────────────────────┐
    │     DATA TYPE      │   RECOMMENDED      │           WHY                  │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Stationary         │ SES, AR, ARIMA     │ No need for trend/season terms │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Trend only         │ Holt, ARIMA(p,1,q) │ Need trend component           │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Seasonal only      │ Seasonal Naive,    │ Must capture periodic pattern  │
    │                    │ HW, SARIMA         │                                │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Trend + Seasonal   │ Holt-Winters,      │ Need both components           │
    │                    │ SARIMA             │                                │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Very noisy         │ Simple methods!    │ Complex models overfit         │
    ├────────────────────┼────────────────────┼────────────────────────────────┤
    │ Structural break   │ NONE work well     │ Need regime detection first    │
    └────────────────────┴────────────────────┴────────────────────────────────┘


    ════════════════════════════════════════════════════════════════════════════

                            RULES OF THUMB

    1. ALWAYS start with simple baselines (Naive, Mean) to calibrate expectations

    2. CHECK FOR STATIONARITY first — most methods assume it!
       • Plot the series: does it look stable?
       • Check rolling mean/variance: are they constant?
       • If non-stationary: difference first (d=1 or d=2)

    3. USE ACF/PACF to identify orders:
       • PACF cuts off at lag p → AR(p)
       • ACF cuts off at lag q → MA(q)
       • Spikes at seasonal lags → seasonal component needed

    4. PREFER SIMPLER MODELS when:
       • Data is noisy (low signal-to-noise ratio)
       • Sample size is small
       • You need interpretability

    5. USE CROSS-VALIDATION to compare methods on YOUR data
       Don't trust general benchmarks — every dataset is different!

    """

    ax.text(0.5, 0.5, decision_tree, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title('METHOD SELECTION GUIDE\n'
              '"The right tool for the right job"',
              fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("FORECASTING SHOWDOWN — Paradigm: KNOW YOUR WEAPONS")
    print("="*70)

    print("""

THE BIG QUESTION: Which method should I use?

ANSWER: It depends on your data!

THE CONTENDERS:
    • Baselines: Naive, Mean, Drift, Seasonal Naive
    • Exponential Smoothing: SES, Holt, Holt-Winters
    • ARIMA: AR, MA, ARIMA variants

THE SCENARIOS:
    • Stationary → SES or ARIMA
    • Trending → Holt or ARIMA(p,d>0,q)
    • Seasonal → Holt-Winters or SARIMA
    • Noisy → Simpler models (less overfitting)
    • Structural break → All methods struggle!

THE KEY INSIGHT:
    THERE IS NO UNIVERSALLY BEST METHOD!

    Match model complexity to data complexity.
    The best forecasters know WHEN to use each tool.

    """)

    # Run a quick showdown
    print("\n" + "="*60)
    print("QUICK SHOWDOWN: Comparing Methods on Different Scenarios")
    print("="*60)

    scenarios = {
        'Stationary': generate_stationary(200),
        'Trending': generate_trending(200, slope=0.15),
        'Seasonal': generate_seasonal(200, period=12),
        'Trend+Season': generate_trending_seasonal(200),
    }

    for scenario_name, y in scenarios.items():
        print(f"\n{scenario_name}:")
        print("-" * 40)
        results, _, _ = run_showdown(y, train_ratio=0.8)

        # Sort by MSE
        sorted_results = sorted(
            [(name, res['mse']) for name, res in results.items() if res['mse'] < np.inf],
            key=lambda x: x[1]
        )

        for i, (name, mse_val) in enumerate(sorted_results[:5]):
            marker = "★" if i == 0 else " "
            print(f"  {marker} {name:<20} MSE = {mse_val:.2f}")

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. Scenario showdown
    fig1 = visualize_scenario_showdown()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/05_showdown_scenarios.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. Method ranking
    fig2 = visualize_method_ranking()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/05_showdown_ranking.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Failure modes
    fig3 = visualize_failure_modes()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/05_showdown_failures.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. Decision tree
    fig4 = visualize_decision_tree()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/05_showdown_decision_tree.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    print("\n" + "="*60)
    print("SUMMARY: Forecasting Showdown")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 05_showdown_scenarios.png     — Head-to-head on 6 scenarios
    2. 05_showdown_ranking.png       — Win rates and average ranks
    3. 05_showdown_failures.png      — Where each method fails
    4. 05_showdown_decision_tree.png — How to choose your method

KEY TAKEAWAYS:
    1. No method wins everywhere — know the tradeoffs!
    2. Simple baselines are surprisingly hard to beat
    3. Match model complexity to data complexity
    4. Structural breaks defeat all statistical methods
    5. Use cross-validation on YOUR data
    6. When in doubt, try multiple methods and compare

THE META-LESSON:
    The best forecaster isn't the one who knows the fanciest method.
    It's the one who knows WHEN to use each method.

    """)
