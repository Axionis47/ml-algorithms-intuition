"""
ARIMA — Paradigm: AUTOREGRESSIVE INTEGRATED MOVING AVERAGE

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

ARIMA combines THREE simple ideas:

    AR (AutoRegressive): Predict from your own past
        y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t

    I (Integrated): Difference to remove trend
        y'_t = y_t - y_{t-1}  (makes non-stationary → stationary)

    MA (Moving Average): Learn from your past mistakes
        y_t = ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θqε_{t-q}

ARIMA(p, d, q):
    p = AR order (how many past values to use)
    d = Differencing order (how many times to difference)
    q = MA order (how many past errors to use)

===============================================================
THE AR COMPONENT: "The Past Predicts the Future"
===============================================================

AR(1): y_t = φ × y_{t-1} + ε_t
    "Today ≈ φ × Yesterday + noise"

AR(p): y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
    "Today ≈ weighted combination of last p days + noise"

KEY INSIGHT: AR models capture MOMENTUM / PERSISTENCE
    |φ| close to 1 → strong persistence (what goes up stays up)
    |φ| close to 0 → weak persistence (quickly reverts to mean)

STATIONARITY CONSTRAINT:
    For AR(1): |φ| < 1 (otherwise series explodes or oscillates wildly)
    For AR(p): roots of characteristic polynomial outside unit circle

===============================================================
THE I COMPONENT: "Remove the Trend First"
===============================================================

Many real series are non-stationary (trending up/down).
ARIMA can't model non-stationary data directly.

SOLUTION: Differencing
    d=1: y'_t = y_t - y_{t-1}  (removes linear trend)
    d=2: y''_t = y'_t - y'_{t-1} (removes quadratic trend)

After differencing, fit ARMA to the differenced series.
Then INTEGRATE back to get forecasts in original scale.

===============================================================
THE MA COMPONENT: "Learn from Your Mistakes"
===============================================================

MA(1): y_t = ε_t + θ × ε_{t-1}
    "Today = noise + θ × yesterday's noise"

MA(q): y_t = ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}
    "Today = noise + corrections based on recent errors"

KEY INSIGHT: MA models capture SHOCKS / INNOVATIONS
    If yesterday's prediction error was positive,
    the MA term adjusts today's prediction accordingly.

INVERTIBILITY CONSTRAINT:
    For MA(1): |θ| < 1

===============================================================
WHY ARIMA WORKS
===============================================================

1. PARSIMONY: Few parameters capture complex patterns
   AR captures persistence, MA captures shocks

2. FLEXIBILITY: Can model many real-world patterns
   Trends (via I), cycles (via AR), irregular shocks (via MA)

3. INTERPRETABILITY: Each component has clear meaning
   Unlike black-box models

4. PROVEN: 50+ years of theory and practice

===============================================================
MODEL SELECTION: THE BOX-JENKINS METHOD
===============================================================

1. IDENTIFICATION: Look at ACF/PACF to guess (p, d, q)
   - PACF cuts off at p → AR(p)
   - ACF cuts off at q → MA(q)
   - ACF decays slowly → need differencing (d > 0)

2. ESTIMATION: Fit parameters using maximum likelihood

3. DIAGNOSTIC: Check residuals
   - Should be white noise (no remaining autocorrelation)
   - If patterns remain, revise model

4. FORECASTING: Generate predictions with confidence intervals

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms/11-time-series')
try:
    from _02_autocorrelation import acf, pacf, significance_bands
except ImportError:
    # Direct import when running as main module
    import importlib.util
    spec = importlib.util.spec_from_file_location("autocorrelation",
        "/Users/sid47/ML Algorithms/11-time-series/02_autocorrelation.py")
    autocorr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(autocorr_module)
    acf = autocorr_module.acf
    pacf = autocorr_module.pacf
    significance_bands = autocorr_module.significance_bands


# ============================================================
# AR, MA, ARIMA IMPLEMENTATIONS
# ============================================================

class AR:
    """
    AutoRegressive model of order p.

    y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t

    Fitted using ordinary least squares.
    """

    def __init__(self, p=1):
        """
        Parameters:
        -----------
        p : int
            AR order (number of lags).
        """
        self.p = p
        self.phi = None  # AR coefficients
        self.const = None  # Constant term
        self.sigma = None  # Residual standard deviation
        self.fitted_values = None
        self.residuals = None

    def fit(self, y):
        """Fit AR(p) model using OLS."""
        n = len(y)
        p = self.p

        if n <= p:
            raise ValueError(f"Need more than {p} observations for AR({p})")

        # Construct design matrix
        # X[t] = [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
        X = np.column_stack([
            np.ones(n - p),
            *[y[p-i-1:n-i-1] for i in range(p)]
        ])
        y_target = y[p:]

        # OLS: β = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X, y_target, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X) @ y_target

        self.const = beta[0]
        self.phi = beta[1:]

        # Compute fitted values and residuals
        self.fitted_values = X @ beta
        self.residuals = y_target - self.fitted_values
        self.sigma = np.std(self.residuals)

        # Store original series for forecasting
        self._y = y

        return self

    def forecast(self, h=1, return_conf_int=False, alpha=0.05):
        """
        Forecast h steps ahead.

        Returns point forecasts and optionally confidence intervals.
        """
        y = self._y.copy()
        forecasts = []

        for _ in range(h):
            # y_{t+1} = c + φ₁y_t + φ₂y_{t-1} + ... + φₚy_{t-p+1}
            y_recent = y[-self.p:][::-1]  # [y_t, y_{t-1}, ..., y_{t-p+1}]
            y_next = self.const + np.dot(self.phi, y_recent)
            forecasts.append(y_next)
            y = np.append(y, y_next)

        forecasts = np.array(forecasts)

        if return_conf_int:
            # Approximate confidence intervals (assumes forecast errors grow with sqrt(h))
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
            se = self.sigma * np.sqrt(np.arange(1, h + 1))
            lower = forecasts - z * se
            upper = forecasts + z * se
            return forecasts, lower, upper

        return forecasts

    def get_roots(self):
        """
        Get roots of characteristic polynomial.

        For stationarity, all roots should be outside unit circle.
        """
        # Characteristic polynomial: 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0
        # Equivalent to: zᵖ - φ₁z^{p-1} - ... - φₚ = 0
        coeffs = np.concatenate([[1], -self.phi])
        roots = np.roots(coeffs)
        return roots


class MA:
    """
    Moving Average model of order q.

    y_t = c + ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}

    Fitted using conditional least squares (approximate).
    """

    def __init__(self, q=1):
        """
        Parameters:
        -----------
        q : int
            MA order (number of lagged errors).
        """
        self.q = q
        self.theta = None
        self.const = None
        self.sigma = None
        self.residuals = None

    def fit(self, y, n_iter=100, tol=1e-6):
        """
        Fit MA(q) using iterative conditional least squares.

        This is a simple approximation. For production use,
        maximum likelihood estimation is preferred.
        """
        n = len(y)
        q = self.q

        # Initialize
        self.const = np.mean(y)
        self.theta = np.zeros(q)
        eps = np.zeros(n)

        for iteration in range(n_iter):
            theta_old = self.theta.copy()

            # E-step: estimate residuals given current parameters
            for t in range(n):
                ma_part = 0
                for j in range(1, min(q + 1, t + 1)):
                    ma_part += self.theta[j-1] * eps[t-j]
                eps[t] = y[t] - self.const - ma_part

            # M-step: estimate parameters given residuals
            # This is approximate - proper MA estimation uses MLE
            for j in range(q):
                numerator = np.sum(eps[j+1:] * eps[:n-j-1])
                denominator = np.sum(eps[:n-j-1]**2)
                if abs(denominator) > 1e-10:
                    self.theta[j] = numerator / denominator

            # Constrain for invertibility
            self.theta = np.clip(self.theta, -0.99, 0.99)

            # Check convergence
            if np.max(np.abs(self.theta - theta_old)) < tol:
                break

        self.residuals = eps
        self.sigma = np.std(eps)
        self._y = y

        return self

    def forecast(self, h=1):
        """Forecast h steps ahead."""
        # For MA, forecast beyond q steps is just the constant
        forecasts = np.full(h, self.const)

        # First q forecasts use past residuals
        for i in range(min(h, self.q)):
            ma_part = 0
            for j in range(i + 1, self.q + 1):
                if j <= len(self.residuals):
                    ma_part += self.theta[j-1] * self.residuals[-j+i]
            forecasts[i] += ma_part

        return forecasts


class ARIMA:
    """
    ARIMA(p, d, q) model.

    Combines AR(p), differencing of order d, and MA(q).
    """

    def __init__(self, p=1, d=1, q=0):
        """
        Parameters:
        -----------
        p : int
            AR order.
        d : int
            Differencing order.
        q : int
            MA order.
        """
        self.p = p
        self.d = d
        self.q = q

        self.phi = None  # AR coefficients
        self.theta = None  # MA coefficients
        self.const = None
        self.sigma = None

        self._y_orig = None
        self._y_diff = None
        self.fitted_values = None
        self.residuals = None

    def _difference(self, y, d):
        """Apply differencing d times."""
        result = y.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _integrate(self, y_diff, y_orig, d):
        """
        Reverse differencing to get back to original scale.

        For forecasting, we need to add back the differences.
        """
        if d == 0:
            return y_diff

        # For d=1: y_t = y_{t-1} + y'_t
        result = y_diff.copy()
        for _ in range(d):
            result = np.cumsum(np.concatenate([[y_orig[-1]], result]))

        return result[1:]  # Remove the initial value

    def fit(self, y, n_iter=100):
        """Fit ARIMA model."""
        self._y_orig = y.copy()

        # Step 1: Difference
        y_diff = self._difference(y, self.d)
        self._y_diff = y_diff

        n = len(y_diff)
        p, q = self.p, self.q

        # Step 2: Fit ARMA to differenced series
        # Using conditional least squares (simplified)

        # Initialize
        self.const = np.mean(y_diff) if q == 0 else 0
        self.phi = np.zeros(p) if p > 0 else np.array([])
        self.theta = np.zeros(q) if q > 0 else np.array([])

        eps = np.zeros(n)

        for iteration in range(n_iter):
            # Estimate residuals
            for t in range(max(p, q), n):
                ar_part = self.const
                for i in range(p):
                    ar_part += self.phi[i] * y_diff[t-i-1]

                ma_part = 0
                for j in range(q):
                    if t - j - 1 >= 0:
                        ma_part += self.theta[j] * eps[t-j-1]

                eps[t] = y_diff[t] - ar_part - ma_part

            # Update AR coefficients using OLS
            if p > 0:
                X_ar = np.column_stack([y_diff[p-i-1:n-i-1] for i in range(p)])
                y_ar = y_diff[p:] - eps[p:]
                if q > 0:
                    ma_contrib = np.zeros(n - p)
                    for t in range(p, n):
                        for j in range(q):
                            if t - j - 1 >= 0:
                                ma_contrib[t-p] += self.theta[j] * eps[t-j-1]
                    y_ar = y_diff[p:] - ma_contrib

                try:
                    self.phi = np.linalg.lstsq(X_ar, y_ar - self.const, rcond=None)[0]
                except:
                    pass

            # Update MA coefficients
            if q > 0:
                for j in range(q):
                    numerator = np.sum(eps[j+1:] * eps[:n-j-1])
                    denominator = np.sum(eps[:n-j-1]**2) + 1e-10
                    self.theta[j] = 0.5 * self.theta[j] + 0.5 * numerator / denominator

                self.theta = np.clip(self.theta, -0.95, 0.95)

        self.residuals = eps
        self.sigma = np.std(eps[max(p, q):])

        # Compute fitted values in original scale
        fitted_diff = np.zeros(n)
        for t in range(max(p, q), n):
            ar_part = self.const
            for i in range(p):
                ar_part += self.phi[i] * y_diff[t-i-1]
            ma_part = 0
            for j in range(q):
                if t - j - 1 >= 0:
                    ma_part += self.theta[j] * eps[t-j-1]
            fitted_diff[t] = ar_part + ma_part

        self.fitted_values = self._integrate(fitted_diff, y[:self.d], self.d)

        return self

    def forecast(self, h=1, return_conf_int=False, alpha=0.05):
        """Forecast h steps ahead."""
        y_diff = self._y_diff.copy()
        eps = self.residuals.copy()

        forecasts_diff = []
        p, q = self.p, self.q

        for step in range(h):
            ar_part = self.const
            for i in range(p):
                if len(y_diff) > i:
                    ar_part += self.phi[i] * y_diff[-i-1]

            ma_part = 0
            for j in range(q):
                if step == 0 and len(eps) > j:
                    ma_part += self.theta[j] * eps[-j-1]
                # For multi-step forecasts, assume future errors are 0

            forecast_diff = ar_part + ma_part
            forecasts_diff.append(forecast_diff)

            y_diff = np.append(y_diff, forecast_diff)
            eps = np.append(eps, 0)  # Future errors unknown

        # Integrate back to original scale
        forecasts_diff = np.array(forecasts_diff)
        forecasts = self._integrate(forecasts_diff, self._y_orig, self.d)

        if return_conf_int:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
            # Approximate SE (grows with horizon)
            se = self.sigma * np.sqrt(np.arange(1, h + 1))
            lower = forecasts - z * se
            upper = forecasts + z * se
            return forecasts, lower, upper

        return forecasts


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_ar_process(n, phi, const=0, sigma=1.0, seed=42):
    """Generate AR(p) process."""
    np.random.seed(seed)
    p = len(phi)
    y = np.zeros(n)
    y[:p] = np.random.randn(p) * sigma

    for t in range(p, n):
        ar_part = const + np.dot(phi, y[t-p:t][::-1])
        y[t] = ar_part + np.random.randn() * sigma

    return y


def generate_ma_process(n, theta, const=0, sigma=1.0, seed=42):
    """Generate MA(q) process."""
    np.random.seed(seed)
    q = len(theta)
    eps = np.random.randn(n + q) * sigma
    y = np.zeros(n)

    for t in range(n):
        y[t] = const + eps[t + q] + np.dot(theta, eps[t:t+q][::-1])

    return y


def generate_arima_process(n, phi, theta, d=1, const=0, sigma=1.0, seed=42):
    """Generate ARIMA(p, d, q) process."""
    np.random.seed(seed)
    p = len(phi) if phi is not None else 0
    q = len(theta) if theta is not None else 0

    # Generate ARMA
    n_arma = n + d * 10  # Extra for differencing warmup
    eps = np.random.randn(n_arma + q) * sigma
    y_arma = np.zeros(n_arma)

    for t in range(max(p, q), n_arma):
        ar_part = const
        if p > 0:
            ar_part += np.dot(phi, y_arma[t-p:t][::-1])
        ma_part = 0
        if q > 0:
            ma_part = np.dot(theta, eps[t:t+q][::-1])
        y_arma[t] = ar_part + ma_part + eps[t + q]

    # Integrate d times
    y = y_arma
    for _ in range(d):
        y = np.cumsum(y)

    return y[-n:]


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_ar_intuition():
    """
    Visualize the AR component: "The past predicts the future"
    """
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Different AR(1) coefficients
    phis = [-0.8, 0.0, 0.3, 0.7, 0.95, 0.99]
    phi_labels = ['φ = -0.8\n(Oscillating)', 'φ = 0.0\n(White noise)',
                  'φ = 0.3\n(Weak persistence)', 'φ = 0.7\n(Moderate persistence)',
                  'φ = 0.95\n(Strong persistence)', 'φ = 0.99\n(Near unit root)']

    for idx, (phi, label) in enumerate(zip(phis, phi_labels)):
        ax = axes[idx // 3, idx % 3]

        y = generate_ar_process(n, [phi], sigma=1.0)

        ax.plot(y, 'b-', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)

        # Add variance info
        ax.text(0.95, 0.95, f'Var = {np.var(y):.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0, 0].set_ylabel('Value')
    axes[1, 0].set_ylabel('Value')

    plt.suptitle('THE AR COMPONENT: How φ Controls Persistence\n'
                 '"Higher |φ| = stronger momentum = slower mean reversion"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_ma_intuition():
    """
    Visualize the MA component: "Learn from your mistakes"
    """
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Different MA(1) coefficients
    thetas = [-0.8, 0.0, 0.3, 0.7, 0.9, 0.99]
    theta_labels = ['θ = -0.8\n(Negative correction)', 'θ = 0.0\n(White noise)',
                    'θ = 0.3\n(Weak correction)', 'θ = 0.7\n(Moderate correction)',
                    'θ = 0.9\n(Strong correction)', 'θ = 0.99\n(Very strong)']

    for idx, (theta, label) in enumerate(zip(thetas, theta_labels)):
        ax = axes[idx // 3, idx % 3]

        y = generate_ma_process(n, [theta], sigma=1.0)

        ax.plot(y, 'g-', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)

        # Compute and show ACF(1)
        acf_1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        ax.text(0.95, 0.95, f'ACF(1) = {acf_1:.2f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0, 0].set_ylabel('Value')
    axes[1, 0].set_ylabel('Value')

    plt.suptitle('THE MA COMPONENT: How θ Controls Shock Persistence\n'
                 '"MA captures how past shocks (errors) affect current values"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_differencing():
    """
    Visualize the I component: "Remove the trend first"
    """
    np.random.seed(42)
    n = 150

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Generate trending series
    t = np.arange(n)
    trend_linear = 0.1 * t
    trend_quadratic = 0.002 * t**2
    random_walk = np.cumsum(np.random.randn(n) * 0.5)

    noise = np.random.randn(n) * 2

    series_list = [
        (trend_linear + noise, 'LINEAR TREND', 1),
        (trend_quadratic + noise, 'QUADRATIC TREND', 2),
        (random_walk, 'RANDOM WALK', 1)
    ]

    for col, (y, title, d_needed) in enumerate(series_list):
        # Original series
        axes[0, col].plot(y, 'b-', linewidth=1)
        axes[0, col].set_title(f'{title}\nOriginal (Non-stationary)', fontsize=10, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel('Original')
        axes[0, col].grid(True, alpha=0.3)

        # First difference
        y_diff1 = np.diff(y)
        axes[1, col].plot(y_diff1, 'g-', linewidth=0.8)
        axes[1, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Check stationarity (simple variance check)
        var_first_half = np.var(y_diff1[:len(y_diff1)//2])
        var_second_half = np.var(y_diff1[len(y_diff1)//2:])
        is_stationary = abs(var_first_half - var_second_half) / max(var_first_half, var_second_half) < 0.5

        status = '✓ Stationary' if is_stationary else '✗ Still trending'
        color = 'green' if is_stationary else 'orange'
        axes[1, col].set_title(f'd=1: First Difference\n{status}', fontsize=10, color=color)
        if col == 0:
            axes[1, col].set_ylabel('First Diff')
        axes[1, col].grid(True, alpha=0.3)

        # Second difference
        y_diff2 = np.diff(y_diff1)
        axes[2, col].plot(y_diff2, 'purple', linewidth=0.8)
        axes[2, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)

        var_first_half2 = np.var(y_diff2[:len(y_diff2)//2])
        var_second_half2 = np.var(y_diff2[len(y_diff2)//2:])
        is_stationary2 = abs(var_first_half2 - var_second_half2) / max(var_first_half2, var_second_half2) < 0.5

        status2 = '✓ Stationary' if is_stationary2 else '✗ Still non-stationary'
        axes[2, col].set_title(f'd=2: Second Difference\n{status2}', fontsize=10,
                               color='green' if is_stationary2 else 'red')
        if col == 0:
            axes[2, col].set_ylabel('Second Diff')
        axes[2, col].set_xlabel('Time')
        axes[2, col].grid(True, alpha=0.3)

        # Highlight the row that achieves stationarity
        if d_needed == 1:
            axes[1, col].set_facecolor('honeydew')
        elif d_needed == 2:
            axes[2, col].set_facecolor('honeydew')

    plt.suptitle('THE I (INTEGRATED) COMPONENT: Differencing to Achieve Stationarity\n'
                 '"Remove the trend by computing changes instead of levels"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_arima_components():
    """
    Show how AR, I, MA combine in ARIMA.
    """
    np.random.seed(42)
    n = 200

    fig = plt.figure(figsize=(16, 14))

    # Generate ARIMA(2,1,1) process
    phi = [0.7, -0.2]
    theta = [0.4]
    y = generate_arima_process(n, phi, theta, d=1, sigma=1.0)

    # Main title
    fig.suptitle('ARIMA DISSECTED: Understanding Each Component\n'
                 'ARIMA(p,d,q) = AR(p) + Differencing(d) + MA(q)',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: The full series
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(y, 'b-', linewidth=1)
    ax1.set_title('ORIGINAL SERIES (Non-stationary)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Panel 2: After differencing
    ax2 = fig.add_subplot(3, 2, 2)
    y_diff = np.diff(y)
    ax2.plot(y_diff, 'g-', linewidth=0.8)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('AFTER DIFFERENCING (d=1)\nNow stationary!', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Δy')
    ax2.grid(True, alpha=0.3)

    # Panel 3: ACF of differenced series
    ax3 = fig.add_subplot(3, 2, 3)
    acf_vals = acf(y_diff, max_lag=20)
    band = significance_bands(len(y_diff))
    lags = np.arange(len(acf_vals))

    colors = ['green' if abs(v) > band else 'steelblue' for v in acf_vals]
    ax3.bar(lags, acf_vals, color=colors, alpha=0.7, width=0.8)
    ax3.axhline(y=band, color='red', linestyle='--', linewidth=1)
    ax3.axhline(y=-band, color='red', linestyle='--', linewidth=1)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('ACF of Differenced Series\n(Decays → suggests AR component)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')

    # Panel 4: PACF of differenced series
    ax4 = fig.add_subplot(3, 2, 4)
    pacf_vals = pacf(y_diff, max_lag=20)

    colors = ['darkred' if abs(v) > band else 'coral' for v in pacf_vals]
    ax4.bar(lags, pacf_vals, color=colors, alpha=0.7, width=0.8)
    ax4.axhline(y=band, color='red', linestyle='--', linewidth=1)
    ax4.axhline(y=-band, color='red', linestyle='--', linewidth=1)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('PACF of Differenced Series\n(Significant at lags 1,2 → AR(2))', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('PACF')

    # Panel 5: Fit ARIMA and show forecast
    ax5 = fig.add_subplot(3, 2, 5)

    train_end = 150
    y_train = y[:train_end]
    y_test = y[train_end:]

    model = ARIMA(p=2, d=1, q=1)
    model.fit(y_train)
    forecasts, lower, upper = model.forecast(len(y_test), return_conf_int=True)

    ax5.plot(range(train_end), y_train, 'b-', linewidth=1, label='Training')
    ax5.plot(range(train_end, n), y_test, 'k-', linewidth=2, label='Actual')
    ax5.plot(range(train_end, n), forecasts, 'r--', linewidth=2, label='Forecast')
    ax5.fill_between(range(train_end, n), lower, upper, alpha=0.2, color='red', label='95% CI')
    ax5.axvline(x=train_end, color='gray', linestyle='--', alpha=0.7)
    ax5.set_title('ARIMA(2,1,1) FORECAST\nwith Confidence Intervals', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Value')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Residual analysis
    ax6 = fig.add_subplot(3, 2, 6)
    residual_acf = acf(model.residuals[10:], max_lag=15)  # Skip initial values
    lags_r = np.arange(len(residual_acf))
    band_r = significance_bands(len(model.residuals) - 10)

    colors = ['green' if abs(v) > band_r else 'gray' for v in residual_acf]
    ax6.bar(lags_r, residual_acf, color=colors, alpha=0.7, width=0.8)
    ax6.axhline(y=band_r, color='red', linestyle='--', linewidth=1)
    ax6.axhline(y=-band_r, color='red', linestyle='--', linewidth=1)
    ax6.axhline(y=0, color='black', linewidth=0.5)

    n_significant = np.sum(np.abs(residual_acf[1:]) > band_r)
    verdict = '✓ White noise (good fit!)' if n_significant <= 1 else '✗ Structure remains'
    ax6.set_title(f'RESIDUAL ACF\n{verdict}', fontsize=11, fontweight='bold',
                  color='green' if n_significant <= 1 else 'red')
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('Residual ACF')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_model_selection():
    """
    Show how ACF/PACF patterns indicate model type.
    """
    np.random.seed(42)
    n = 400

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Different ARIMA models
    models = [
        (([0.8], None, 0), 'AR(1)', 'PACF cuts off at 1\nACF decays'),
        (([0.5, 0.3], None, 0), 'AR(2)', 'PACF cuts off at 2\nACF decays'),
        ((None, [0.7], 0), 'MA(1)', 'ACF cuts off at 1\nPACF decays'),
        (([0.6], [0.4], 0), 'ARMA(1,1)', 'Both decay\n(harder to identify)'),
    ]

    max_lag = 20
    band = significance_bands(n)

    for row, ((phi, theta, d), name, pattern) in enumerate(models):
        # Generate process
        if phi is not None and theta is not None:
            y = generate_arima_process(n, phi, theta, d=0, sigma=1.0)
        elif phi is not None:
            y = generate_ar_process(n, phi, sigma=1.0)
        else:
            y = generate_ma_process(n, theta, sigma=1.0)

        # Column 1: Time series
        axes[row, 0].plot(y[:150], 'b-', linewidth=0.8)
        axes[row, 0].set_title(f'{name}', fontsize=11, fontweight='bold')
        if row == 3:
            axes[row, 0].set_xlabel('Time')
        if row == 0:
            axes[row, 0].set_title(f'{name}\nTime Series', fontsize=11, fontweight='bold')
        axes[row, 0].grid(True, alpha=0.3)

        # Column 2: ACF
        acf_vals = acf(y, max_lag)
        lags = np.arange(len(acf_vals))
        colors = ['green' if abs(v) > band else 'steelblue' for v in acf_vals]
        axes[row, 1].bar(lags, acf_vals, color=colors, alpha=0.7, width=0.8)
        axes[row, 1].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[row, 1].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[row, 1].axhline(y=0, color='black', linewidth=0.5)
        if row == 0:
            axes[row, 1].set_title('ACF', fontsize=11, fontweight='bold')
        if row == 3:
            axes[row, 1].set_xlabel('Lag')
        axes[row, 1].set_xlim(-0.5, max_lag + 0.5)

        # Column 3: PACF
        pacf_vals = pacf(y, max_lag)
        colors = ['darkred' if abs(v) > band else 'coral' for v in pacf_vals]
        axes[row, 2].bar(lags, pacf_vals, color=colors, alpha=0.7, width=0.8)
        axes[row, 2].axhline(y=band, color='red', linestyle='--', linewidth=1)
        axes[row, 2].axhline(y=-band, color='red', linestyle='--', linewidth=1)
        axes[row, 2].axhline(y=0, color='black', linewidth=0.5)
        if row == 0:
            axes[row, 2].set_title('PACF', fontsize=11, fontweight='bold')
        if row == 3:
            axes[row, 2].set_xlabel('Lag')
        axes[row, 2].set_xlim(-0.5, max_lag + 0.5)

        # Column 4: Interpretation
        axes[row, 3].axis('off')
        axes[row, 3].text(0.1, 0.5, pattern,
                          transform=axes[row, 3].transAxes, fontsize=11,
                          verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        if row == 0:
            axes[row, 3].set_title('Pattern', fontsize=11, fontweight='bold')

    plt.suptitle('MODEL IDENTIFICATION FROM ACF/PACF PATTERNS\n'
                 '"The signature of each process is in its autocorrelation structure"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def visualize_forecast_uncertainty():
    """
    Show how forecast uncertainty grows with horizon.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate and fit ARIMA
    n = 150
    y = generate_arima_process(n, [0.7], [0.3], d=1, sigma=1.0)

    train_end = 100
    y_train = y[:train_end]

    model = ARIMA(p=1, d=1, q=1)
    model.fit(y_train)

    # Panel 1: Forecast with growing uncertainty
    ax1 = axes[0, 0]
    h = 50
    forecasts, lower, upper = model.forecast(h, return_conf_int=True)

    ax1.plot(range(train_end), y_train, 'b-', linewidth=1, label='Observed')
    ax1.plot(range(train_end, train_end + h), forecasts, 'r-', linewidth=2, label='Forecast')
    ax1.fill_between(range(train_end, train_end + h), lower, upper,
                     alpha=0.3, color='red', label='95% CI')
    ax1.axvline(x=train_end, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('THE CONE OF UNCERTAINTY\n"Prediction intervals always widen"',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: CI width vs horizon
    ax2 = axes[0, 1]
    ci_width = upper - lower
    ax2.plot(range(1, h + 1), ci_width, 'b-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Forecast Horizon (steps ahead)')
    ax2.set_ylabel('95% Confidence Interval Width')
    ax2.set_title('UNCERTAINTY GROWS WITH HORIZON\n"The further you look, the less you know"',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add sqrt reference line
    horizon = np.arange(1, h + 1)
    ax2.plot(horizon, ci_width[0] * np.sqrt(horizon), 'r--', linewidth=1.5,
             label='~√h growth (theoretical)')
    ax2.legend(fontsize=9)

    # Panel 3: Multi-step forecast accuracy
    ax3 = axes[1, 0]

    # Generate multiple forecasts and track error
    n_sims = 50
    horizons = [1, 5, 10, 20, 30]
    mse_by_horizon = {h: [] for h in horizons}

    for sim in range(n_sims):
        y_sim = generate_arima_process(150, [0.7], [0.3], d=1, sigma=1.0, seed=sim)
        y_train_sim = y_sim[:100]
        y_test_sim = y_sim[100:]

        model_sim = ARIMA(p=1, d=1, q=1)
        model_sim.fit(y_train_sim)

        for h in horizons:
            if h <= len(y_test_sim):
                forecast = model_sim.forecast(h)
                mse = (y_test_sim[h-1] - forecast[-1])**2
                mse_by_horizon[h].append(mse)

    mean_mse = [np.mean(mse_by_horizon[h]) for h in horizons]
    std_mse = [np.std(mse_by_horizon[h]) for h in horizons]

    ax3.bar(range(len(horizons)), mean_mse, yerr=std_mse, capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(horizons)))
    ax3.set_xticklabels([f'h={h}' for h in horizons])
    ax3.set_xlabel('Forecast Horizon')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('FORECAST ERROR INCREASES WITH HORIZON\n(Average over 50 simulations)',
                  fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Summary insight
    ax4 = axes[1, 1]
    ax4.axis('off')

    insight = """
    ╔═══════════════════════════════════════════════════════════╗
    ║       WHY UNCERTAINTY GROWS WITH FORECAST HORIZON         ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  1-STEP FORECAST:                                         ║
    ║     ŷ_{t+1} = f(y_t, y_{t-1}, ...) + ε_{t+1}             ║
    ║     Uncertainty = σ (one error term)                      ║
    ║                                                           ║
    ║  2-STEP FORECAST:                                         ║
    ║     ŷ_{t+2} = f(ŷ_{t+1}, y_t, ...) + ε_{t+2}             ║
    ║     Uncertainty = σ√2 (errors compound!)                  ║
    ║                                                           ║
    ║  h-STEP FORECAST:                                         ║
    ║     Uncertainty ≈ σ√h                                     ║
    ║                                                           ║
    ║  ─────────────────────────────────────────────────────    ║
    ║                                                           ║
    ║  KEY INSIGHT: This is FUNDAMENTAL, not a model flaw.      ║
    ║  No model can predict far into the future precisely.      ║
    ║                                                           ║
    ║  PRACTICAL IMPLICATION:                                   ║
    ║  • Short-term forecasts: narrow CI, trust them            ║
    ║  • Long-term forecasts: wide CI, use with caution         ║
    ║  • Very long-term: often just mean reversion              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    ax4.text(0.05, 0.95, insight, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('FORECAST UNCERTAINTY: The Unavoidable Truth\n'
                 '"The further you predict, the less certain you can be"',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_ar_order():
    """
    Effect of AR order on fit and forecast.
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of AR Order on Forecasting")
    print("="*60)

    np.random.seed(42)

    # True AR(2) process
    true_phi = [0.6, 0.2]
    y = generate_ar_process(200, true_phi, sigma=1.0)

    train_end = 150
    y_train = y[:train_end]
    y_test = y[train_end:]

    print(f"\nTrue process: AR(2) with φ = {true_phi}")
    print("-" * 50)

    for p in [1, 2, 3, 4, 5]:
        model = AR(p=p)
        model.fit(y_train)
        forecasts = model.forecast(len(y_test))

        mse = np.mean((y_test - forecasts)**2)
        aic = len(y_train) * np.log(model.sigma**2) + 2 * (p + 1)  # Approximate AIC

        print(f"AR({p}): MSE = {mse:.3f}, σ = {model.sigma:.3f}, AIC ≈ {aic:.1f}")

    print("\n→ AR(2) should have lowest error (true model)")
    print("→ Higher orders overfit, lower orders underfit")


def ablation_differencing_order():
    """
    Effect of differencing order.
    """
    print("\n" + "="*60)
    print("ABLATION: Effect of Differencing Order")
    print("="*60)

    np.random.seed(42)

    # Series with linear trend
    n = 200
    t = np.arange(n)
    y = 0.1 * t + np.random.randn(n) * 2  # Linear trend + noise

    train_end = 150
    y_train = y[:train_end]
    y_test = y[train_end:]

    print("\nData: Linear trend + noise")
    print("-" * 50)

    for d in [0, 1, 2]:
        model = ARIMA(p=1, d=d, q=0)
        model.fit(y_train)
        forecasts = model.forecast(len(y_test))

        # Ensure same length
        min_len = min(len(y_test), len(forecasts))
        mse = np.mean((y_test[:min_len] - forecasts[:min_len])**2)
        print(f"ARIMA(1,{d},0): MSE = {mse:.3f}")

    print("\n→ d=1 should work best for linear trend")
    print("→ d=0 can't capture trend, d=2 overdifferences")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("ARIMA — Paradigm: AUTOREGRESSIVE INTEGRATED MOVING AVERAGE")
    print("="*70)

    print("""

THE THREE COMPONENTS:

    AR (AutoRegressive): Predict from your own past
        y_t = φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t
        "Today ≈ weighted sum of recent past + noise"

    I (Integrated): Difference to remove trend
        y'_t = y_t - y_{t-1}
        "Model changes, not levels"

    MA (Moving Average): Learn from past errors
        y_t = ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}
        "Correct based on recent mistakes"

MODEL SELECTION (Box-Jenkins):
    1. Plot ACF/PACF
    2. PACF cuts off at p → AR(p)
    3. ACF cuts off at q → MA(q)
    4. Slow ACF decay → need differencing (d > 0)
    5. Check residuals for white noise

    """)

    # Run ablations
    ablation_ar_order()
    ablation_differencing_order()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # 1. AR intuition
    fig1 = visualize_ar_intuition()
    save_path1 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_ar_intuition.png'
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path1}")
    plt.close(fig1)

    # 2. MA intuition
    fig2 = visualize_ma_intuition()
    save_path2 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_ma_intuition.png'
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close(fig2)

    # 3. Differencing
    fig3 = visualize_differencing()
    save_path3 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_differencing.png'
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path3}")
    plt.close(fig3)

    # 4. ARIMA components
    fig4 = visualize_arima_components()
    save_path4 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_components.png'
    fig4.savefig(save_path4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path4}")
    plt.close(fig4)

    # 5. Model selection
    fig5 = visualize_model_selection()
    save_path5 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_model_selection.png'
    fig5.savefig(save_path5, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path5}")
    plt.close(fig5)

    # 6. Forecast uncertainty
    fig6 = visualize_forecast_uncertainty()
    save_path6 = '/Users/sid47/ML Algorithms/11-time-series/04_arima_uncertainty.png'
    fig6.savefig(save_path6, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path6}")
    plt.close(fig6)

    print("\n" + "="*60)
    print("SUMMARY: ARIMA")
    print("="*60)
    print("""
VISUALIZATIONS GENERATED:
    1. 04_arima_ar_intuition.png    — How φ controls persistence
    2. 04_arima_ma_intuition.png    — How θ controls shock response
    3. 04_arima_differencing.png    — The I component: removing trends
    4. 04_arima_components.png      — Full ARIMA dissection
    5. 04_arima_model_selection.png — ACF/PACF patterns for identification
    6. 04_arima_uncertainty.png     — The cone of uncertainty

KEY TAKEAWAYS:
    1. AR captures persistence/momentum in the series
    2. MA captures response to shocks/innovations
    3. I (differencing) makes non-stationary series stationary
    4. ACF/PACF patterns reveal the appropriate model
    5. Forecast uncertainty ALWAYS grows with horizon
    6. Residuals should be white noise if model is adequate

NEXT: Forecasting Showdown — "When does each method win?"
    """)
