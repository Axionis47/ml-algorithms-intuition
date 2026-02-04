"""
LINEAR REGRESSION — Paradigm: PROJECTION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

You believe: y ≈ Xw + b + noise

Linear regression finds the w and b that minimize squared error:
    w* = argmin_w ||y - Xw||²

GEOMETRICALLY: You're projecting y onto the column space of X.
The prediction ŷ = Xw is the point in span(X) closest to y.
The residual (y - ŷ) is orthogonal to every column of X.

That's it. That's the whole model.

===============================================================
THE ONE EQUATION (closed form)
===============================================================

    w = (X'X)⁻¹ X'y

This falls out from setting the gradient to zero:
    dL/dw = -2X'(y - Xw) = 0
    X'Xw = X'y
    w = (X'X)⁻¹ X'y

===============================================================
INDUCTIVE BIAS (what the model assumes)
===============================================================

1. LINEAR relationship — the decision boundary is a hyperplane
2. GAUSSIAN noise — squared error is MLE under Gaussian
3. HOMOSCEDASTICITY — error variance is constant
4. NO MULTICOLLINEARITY — X'X must be invertible

WHAT IT CAN'T SEE:
- Curves, circles, spirals — anything nonlinear
- Feature interactions (unless you manually add them)
- Heteroscedastic data (varying noise levels)

===============================================================
WHEN TO USE / NOT USE
===============================================================

USE when: You believe the relationship is actually linear, or as a
          baseline to see how much nonlinearity helps.

DON'T USE when: The relationship is clearly nonlinear (circles, XOR).
                The model will average across the nonlinearity and fail.

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
plot_decision_boundary = datasets_module.plot_decision_boundary
accuracy = datasets_module.accuracy


class LinearRegression:
    """
    Linear Regression with two solvers.

    WHY TWO SOLVERS?
        Normal equation: O(d³) — fast when d is small
        Gradient descent: O(nd × iterations) — scales to large d

    Use normal equation when features < ~10k.
    Use GD when features are huge or data doesn't fit in memory.
    """

    def __init__(self, method='normal', lr=0.01, n_iters=1000, regularization=0.0):
        """
        Parameters:
        -----------
        method : 'normal' or 'gd'
        lr : learning rate for gradient descent
        n_iters : iterations for gradient descent
        regularization : L2 penalty (Ridge). Set > 0 if X'X is near-singular.
        """
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Fit the model to training data."""
        if self.method == 'normal':
            self._fit_normal(X, y)
        else:
            self._fit_gd(X, y)
        return self

    def _fit_normal(self, X, y):
        """
        CLOSED FORM SOLUTION

        w = (X'X + λI)⁻¹ X'y

        The λI term is Ridge regularization — adds λ to the diagonal,
        making the matrix invertible even when X'X is singular.

        GEOMETRIC INTERPRETATION:
            Without regularization: project y onto span(X)
            With regularization: shrink toward origin (prefer smaller weights)
        """
        n_samples, n_features = X.shape

        # Add bias column (column of 1s)
        # This makes bias just another weight, absorbed into w
        X_b = np.c_[np.ones(n_samples), X]

        # Regularization matrix (don't regularize the bias term)
        reg_matrix = self.regularization * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0  # Don't penalize bias

        # w = (X'X + λI)⁻¹ X'y
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        w = np.linalg.solve(XtX + reg_matrix, Xty)

        self.bias = w[0]
        self.weights = w[1:]

    def _fit_gd(self, X, y):
        """
        GRADIENT DESCENT

        The loss surface is a BOWL (quadratic, convex).
        There's exactly one minimum. GD will find it.

        Update rule:
            w := w - lr * dL/dw
            dL/dw = -(2/n) X'(y - Xw) + 2λw

        The gradient points uphill. We go opposite direction (downhill).

        WHY /n?
            Makes learning rate independent of batch size.
            Without it: larger dataset = larger gradient = need smaller lr.

        CONVERGENCE:
            - Too high lr: overshoot, diverge
            - Too low lr: painfully slow
            - Just right: smooth exponential decay toward minimum
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iters):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Residuals: (what we want) - (what we got)
            residuals = y - y_pred

            # Gradients
            # dL/dw = -(2/n) X'(y - Xw) + 2λw
            dw = -(2 / n_samples) * (X.T @ residuals) + 2 * self.regularization * self.weights
            db = -(2 / n_samples) * np.sum(residuals)

            # Update (go opposite of gradient = downhill)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss
            mse = np.mean(residuals ** 2)
            reg_loss = self.regularization * np.sum(self.weights ** 2)
            self.loss_history.append(mse + reg_loss)

        return self

    def predict(self, X):
        """Predict y for new X."""
        return X @ self.weights + self.bias

    def predict_class(self, X, threshold=0.5):
        """For classification: threshold the continuous prediction."""
        return (self.predict(X) > threshold).astype(int)


# ============================================================
# FOR CLASSIFICATION (using Linear Regression)
# ============================================================

class LinearRegressionClassifier:
    """
    Classification via Linear Regression.

    WHY THIS EXISTS:
        Shows what happens when you use regression for classification.
        Spoiler: it works on linear problems but has issues:
        1. Predictions can be outside [0,1] — not probabilities
        2. Outliers in one class drag the line toward them
        3. Decision boundary is still linear (hyperplane)

    This is a BASELINE. Logistic regression fixes these issues.
    """

    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_class(X)

    def predict_proba(self, X):
        """Raw predictions (not true probabilities!)"""
        return self.model.predict(X)


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    ABLATION: What happens when you change each component?

    These experiments build intuition about what each part does.
    """
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # Generate data: y = 3x + 7 + noise
    X = 2 * np.random.rand(200, 1)
    y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5
    y_noisy = y.copy()

    # Add some outliers
    y_with_outliers = y.copy()
    y_with_outliers[:5] = 50  # Extreme outliers

    # -------- Experiment 1: Normal vs GD --------
    print("\n1. NORMAL EQUATION vs GRADIENT DESCENT")
    print("-" * 40)
    model_normal = LinearRegression(method='normal')
    model_normal.fit(X, y)
    print(f"Normal:  w={model_normal.weights[0]:.4f}, b={model_normal.bias:.4f}")

    model_gd = LinearRegression(method='gd', lr=0.1, n_iters=500)
    model_gd.fit(X, y)
    print(f"GD:      w={model_gd.weights[0]:.4f}, b={model_gd.bias:.4f}")
    print("→ Both converge to same solution (as they should).")

    # -------- Experiment 2: Learning Rate --------
    print("\n2. LEARNING RATE SWEEP")
    print("-" * 40)
    lrs = [0.001, 0.01, 0.1, 1.0, 2.0]
    for lr in lrs:
        model = LinearRegression(method='gd', lr=lr, n_iters=100)
        model.fit(X, y)
        final_loss = model.loss_history[-1] if model.loss_history else float('inf')
        converged = final_loss < 1.0
        status = "✓ converged" if converged else "✗ not converged" if final_loss < 100 else "✗ DIVERGED"
        print(f"lr={lr:<5} final_loss={final_loss:>10.4f}  {status}")
    print("→ Too low: slow convergence. Too high: divergence.")

    # -------- Experiment 3: Outlier Sensitivity --------
    print("\n3. OUTLIER SENSITIVITY")
    print("-" * 40)
    model_clean = LinearRegression(method='normal')
    model_clean.fit(X, y)
    print(f"Clean data:   w={model_clean.weights[0]:.4f}, b={model_clean.bias:.4f}")

    model_outlier = LinearRegression(method='normal')
    model_outlier.fit(X, y_with_outliers)
    print(f"With outliers: w={model_outlier.weights[0]:.4f}, b={model_outlier.bias:.4f}")
    print("→ Outliers DRAG the line. Squared loss heavily penalizes them.")

    # -------- Experiment 4: Regularization --------
    print("\n4. REGULARIZATION (Ridge)")
    print("-" * 40)
    # Create ill-conditioned data (multicollinear features)
    X_collinear = np.column_stack([X, X + np.random.randn(200, 1) * 0.01])

    for reg in [0, 0.01, 0.1, 1.0, 10.0]:
        model = LinearRegression(method='normal', regularization=reg)
        model.fit(X_collinear, y)
        weight_norm = np.linalg.norm(model.weights)
        print(f"λ={reg:<5} ||w||={weight_norm:.4f}  w={model.weights}")
    print("→ Higher regularization = smaller weights = simpler model.")

    # -------- Experiment 5: Feature Scaling --------
    print("\n5. FEATURE SCALING EFFECT ON GD")
    print("-" * 40)
    # Create features with very different scales
    X_scaled = np.column_stack([X, X * 1000])  # Second feature is 1000x larger
    y_multi = 3 * X.squeeze() + 0.003 * (X * 1000).squeeze() + 7 + np.random.randn(200) * 0.5

    model_unscaled = LinearRegression(method='gd', lr=0.0001, n_iters=1000)
    model_unscaled.fit(X_scaled, y_multi)
    print(f"Unscaled (lr=0.0001): final_loss={model_unscaled.loss_history[-1]:.4f}")

    # Scale features
    X_normalized = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    model_scaled = LinearRegression(method='gd', lr=0.1, n_iters=1000)
    model_scaled.fit(X_normalized, y_multi)
    print(f"Scaled (lr=0.1):      final_loss={model_scaled.loss_history[-1]:.4f}")
    print("→ Scaling allows much higher learning rate and faster convergence.")

    # -------- Experiment 6: Beyond Linear --------
    print("\n6. FAILURE ON NONLINEAR DATA")
    print("-" * 40)
    # Quadratic relationship
    X_quad = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_quad = X_quad.squeeze() ** 2 + np.random.randn(200) * 0.5

    model_linear = LinearRegression(method='normal')
    model_linear.fit(X_quad, y_quad)
    y_pred = model_linear.predict(X_quad)
    mse = np.mean((y_quad - y_pred) ** 2)
    print(f"Linear model on quadratic data: MSE={mse:.4f}")
    print("→ It fits a line through a parabola — misses the structure entirely.")

    return model_gd  # Return for plotting


# ============================================================
# BENCHMARK ON CHALLENGE DATASETS
# ============================================================

def benchmark_on_datasets():
    """Evaluate Linear Regression Classifier on all datasets."""
    print("\n" + "="*60)
    print("BENCHMARK: Linear Regression on Challenge Datasets")
    print("="*60)

    datasets = get_all_datasets()
    model = LinearRegressionClassifier(method='normal')
    results = {}

    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        if name in ['sequential']:  # Skip non-standard
            continue
        if name == 'clustered':  # Multi-class, convert to binary
            y_tr = (y_tr > 2).astype(int)
            y_te = (y_te > 2).astype(int)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy(y_te, y_pred)
        results[name] = acc
        print(f"{name:<15} accuracy: {acc:.3f}")

    return results


def visualize_decision_boundaries():
    """Visualize decision boundaries on 2D datasets."""
    print("\n" + "="*60)
    print("VISUALIZING DECISION BOUNDARIES")
    print("="*60)

    datasets = get_2d_datasets()
    # Only plot datasets where 2D visualization makes sense
    plot_datasets = {k: v for k, v in datasets.items()
                     if k not in ['clustered']}  # clustered is multiclass

    n = len(plot_datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    model = LinearRegressionClassifier(method='normal')

    for i, (name, (X_tr, X_te, y_tr, y_te)) in enumerate(plot_datasets.items()):
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        model.fit(X_tr, y_tr)
        acc = accuracy(y_te, model.predict(X_te))

        plot_decision_boundary(model.predict, X, y, ax=axes[i],
                              title=f'{name} (acc={acc:.2f})')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('LINEAR REGRESSION: Decision Boundaries\n'
                 '(Can only draw straight lines — fails on nonlinear data)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("LINEAR REGRESSION — Paradigm: PROJECTION")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Find the hyperplane that minimizes squared distance to data.
    Geometrically: project y onto the column space of X.

INDUCTIVE BIAS:
    - Assumes LINEAR relationship
    - Sensitive to outliers (squared loss)
    - Can't capture curves, circles, or feature interactions

EXPECT IT TO FAIL ON:
    - circles (need curved boundary)
    - xor (need feature interaction)
    - spiral (highly nonlinear)
    """)

    # Run ablations
    ablation_experiments()

    # Benchmark
    results = benchmark_on_datasets()

    # Visualize
    fig = visualize_decision_boundaries()
    save_path = '/Users/sid47/ML Algorithms/01_linear_regression.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved decision boundaries to: {save_path}")
    plt.close(fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Linear Regression Reveals")
    print("="*60)
    print("""
1. Linear models can ONLY draw hyperplanes
2. On nonlinear data (circles, XOR, spiral), they fail catastrophically
3. Outliers drag the fit (squared loss penalizes heavily)
4. Regularization shrinks weights, helps with multicollinearity
5. Feature scaling is critical for gradient descent

NEXT: Logistic Regression adds a sigmoid to squash outputs to [0,1]
      but still has the same LINEAR decision boundary limitation.
    """)
