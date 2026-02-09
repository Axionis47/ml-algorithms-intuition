"""
FEATURE ENGINEERING & PREPROCESSING — Paradigm: DATA TRANSFORMATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Before any algorithm sees your data, you must PREPARE it.
Feature engineering is 80% of applied ML, yet rarely taught.

THE FUNDAMENTAL QUESTIONS:
1. Should I SCALE my features? (Yes, if distance-based)
2. What do I do with MISSING values? (Impute, don't just drop)
3. How do I handle CATEGORIES? (Encode them as numbers)
4. Should I CREATE new features? (Polynomial, interaction terms)
5. What about OUTLIERS? (Detect, then decide)

THE CORE INSIGHT:
    Algorithms see NUMBERS, not meaning.
    If your numbers don't represent the right thing,
    no algorithm can save you.

===============================================================
THE MATHEMATICS
===============================================================

STANDARDIZATION (z-score):
    z = (x - mean) / std
    Result: mean=0, std=1
    When: Always for distance-based methods (KNN, SVM, PCA)

MIN-MAX SCALING:
    z = (x - min) / (max - min)
    Result: values in [0, 1]
    When: Neural networks (bounded activations), image pixels

ROBUST SCALING:
    z = (x - median) / IQR
    Result: median=0, IQR=1
    When: Data has outliers (median/IQR are outlier-resistant)

POLYNOMIAL FEATURES:
    [x1, x2] → [x1, x2, x1^2, x1*x2, x2^2]  (degree 2)
    Allows linear models to fit nonlinear boundaries.
    WARNING: d features, degree k → O(d^k) features (combinatorial explosion)

ONE-HOT ENCODING:
    color=red → [1, 0, 0]
    color=green → [0, 1, 0]
    color=blue → [0, 0, 1]
    When: Categorical features with NO ordering

ORDINAL ENCODING:
    small=0, medium=1, large=2
    When: Categorical features WITH natural ordering

===============================================================
INDUCTIVE BIAS — What Preprocessing Assumes
===============================================================

1. SCALING assumes features should contribute equally
   - KNN with unscaled: feature with range [0, 1000] dominates
   - Decision trees don't care about scale (splits are threshold-based)

2. MEAN IMPUTATION assumes missing values are "average"
   - Biased if data is not missing at random (MNAR)
   - Indicator columns can help model learn missingness patterns

3. POLYNOMIAL FEATURES assume interactions matter
   - High degree → overfitting (more parameters than data points)
   - Regularization is essential with polynomial features

4. ONE-HOT assumes no ordering
   - Using ordinal for unordered categories → model assumes red < green < blue

5. OUTLIER REMOVAL assumes extreme values are errors
   - In some domains, outliers ARE the signal (fraud detection)

===============================================================
WHERE IT SHOWS UP IN THIS REPO
===============================================================

- 03_knn.py: Distance-sensitive → scaling critical
- 07_decision_tree.py: Scale-invariant → scaling doesn't matter
- 01_linear_regression.py: Outlier-sensitive → robust scaling helps
- 59_pca.py: Ablation 4 shows scaling effect on PCA
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SCALERS
# ============================================================

class StandardScaler:
    """
    Standardization: z = (x - mean) / std

    Centers features to mean=0, scales to std=1.
    Most common preprocessing for ML algorithms.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return Z * self.std_ + self.mean_


class MinMaxScaler:
    """
    Min-Max Scaling: z = (x - min) / (max - min)

    Scales features to [0, 1] range.
    Good for neural networks with bounded activations.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        X_std = (X - self.min_) / scale
        lo, hi = self.feature_range
        return X_std * (hi - lo) + lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """
    Robust Scaling: z = (x - median) / IQR

    Uses median and interquartile range instead of mean/std.
    Resistant to outliers (median doesn't move much).
    """

    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        self.median_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ============================================================
# FEATURE CREATION
# ============================================================

def polynomial_features(X, degree=2):
    """
    Generate polynomial and interaction features.

    [x1, x2] with degree=2 → [x1, x2, x1^2, x1*x2, x2^2]

    Args:
        X: Input features (n_samples, n_features)
        degree: Maximum polynomial degree

    Returns:
        X_poly: Expanded feature matrix
    """
    n_samples, n_features = X.shape
    features = [X]  # Start with original features

    for d in range(2, degree + 1):
        # Generate all combinations of features for this degree
        # For degree d: all monomials x_i1 * x_i2 * ... * x_id
        # We use a recursive approach to generate indices
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(range(n_features), d):
            col = np.ones(n_samples)
            for idx in combo:
                col *= X[:, idx]
            features.append(col.reshape(-1, 1))

    return np.hstack(features)


def one_hot_encode(x):
    """
    One-hot encode integer labels.

    Args:
        x: 1D array of integer labels (e.g., [0, 2, 1, 0])

    Returns:
        encoded: (n_samples, n_classes) binary matrix
    """
    x = np.asarray(x, dtype=int)
    n_classes = x.max() + 1
    encoded = np.zeros((len(x), n_classes))
    encoded[np.arange(len(x)), x] = 1.0
    return encoded


# ============================================================
# MISSING DATA HANDLING
# ============================================================

def impute_mean(X):
    """
    Replace NaN with column mean.

    Simple, fast, but biased if data is not missing at random.
    """
    X_out = X.copy()
    for j in range(X.shape[1]):
        col = X_out[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmean(col)
    return X_out


def impute_median(X):
    """
    Replace NaN with column median.

    More robust to outliers than mean imputation.
    """
    X_out = X.copy()
    for j in range(X.shape[1]):
        col = X_out[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return X_out


def impute_indicator(X):
    """
    Mean imputation + binary indicator columns.

    For each column with missing values, adds a new column
    that is 1 where value was missing, 0 otherwise.
    This lets the model LEARN from the missingness pattern.
    """
    X_imputed = impute_mean(X)
    indicators = []

    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            indicators.append(mask.astype(float).reshape(-1, 1))

    if indicators:
        return np.hstack([X_imputed] + indicators)
    return X_imputed


# ============================================================
# OUTLIER DETECTION
# ============================================================

def detect_outliers_zscore(X, threshold=3.0):
    """
    Z-score outlier detection: |z| > threshold.

    Assumes data is roughly Gaussian.
    Returns boolean mask (True = outlier).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    z_scores = np.abs((X - mean) / std)
    return np.any(z_scores > threshold, axis=1)


def detect_outliers_iqr(X, factor=1.5):
    """
    IQR outlier detection: below Q1 - factor*IQR or above Q3 + factor*IQR.

    More robust than z-score (doesn't assume Gaussian).
    factor=1.5 is the standard "Tukey fence".
    """
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    lower = q25 - factor * iqr
    upper = q75 + factor * iqr
    return np.any((X < lower) | (X > upper), axis=1)


# ============================================================
# SIMPLE MODELS FOR DEMONSTRATION
# ============================================================

class SimpleKNN:
    """
    K-Nearest Neighbors — DISTANCE-SENSITIVE.
    Scaling matters because distance is computed directly.
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            idx = np.argsort(dists)[:self.k]
            neighbors = self.y_train[idx]
            # Majority vote
            values, counts = np.unique(neighbors, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)


class SimpleDecisionTree:
    """
    Decision Tree — SCALE-INVARIANT.
    Only cares about ordering, not magnitude.
    """

    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.mean(y == c)
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        n = len(y)
        parent_gini = self._gini(y)

        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, j], np.linspace(10, 90, 15))
            for t in thresholds:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) < 2 or len(right) < 2:
                    continue
                gain = parent_gini - (len(left)/n * self._gini(left) +
                                      len(right)/n * self._gini(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = t
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': values[np.argmax(counts)]}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None:
            values, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': values[np.argmax(counts)]}

        left_mask = X[:, feature] <= threshold
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class SimpleLinearRegression:
    """
    Linear Regression — OUTLIER-SENSITIVE.
    Minimizes squared error → outliers have O(error^2) influence.
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add bias column
        X_b = np.column_stack([np.ones(len(X)), X])
        # Normal equation: w = (X^T X)^{-1} X^T y
        try:
            self.weights = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# ============================================================
# DATASET GENERATORS
# ============================================================

def make_classification_data(n_samples=300, random_state=42):
    """
    Generate 2D classification data where features have VERY different scales.
    Feature 0: salary in [20000, 200000]
    Feature 1: age in [18, 70]
    """
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    # Class 0: low salary, young
    X0 = np.column_stack([
        rng.normal(50000, 15000, n_per_class),  # salary
        rng.normal(30, 5, n_per_class),           # age
    ])

    # Class 1: high salary, older
    X1 = np.column_stack([
        rng.normal(120000, 20000, n_per_class),
        rng.normal(50, 8, n_per_class),
    ])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def make_regression_data(n_samples=100, random_state=42, n_outliers=5):
    """
    Generate 1D regression data with outliers.
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0, 10, n_samples).reshape(-1, 1)
    y = 2.5 * X.ravel() + 3.0 + rng.normal(0, 1.5, n_samples)

    # Add outliers
    if n_outliers > 0:
        outlier_idx = rng.choice(n_samples, n_outliers, replace=False)
        y[outlier_idx] += rng.choice([-1, 1], n_outliers) * rng.uniform(15, 25, n_outliers)

    return X, y


def make_missing_data(X, missing_frac=0.2, random_state=42):
    """Introduce missing values (NaN) randomly into X."""
    rng = np.random.RandomState(random_state)
    X_missing = X.copy().astype(float)
    mask = rng.rand(*X.shape) < missing_frac
    X_missing[mask] = np.nan
    return X_missing


def make_nonlinear_data(n_samples=200, random_state=42):
    """
    Generate 1D nonlinear regression data: y = sin(x) + noise.
    Polynomial features can capture this.
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.2 * rng.randn(n_samples)
    return X, y


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """
    5 ablations showing when and why preprocessing matters.
    """
    np.random.seed(42)
    print("=" * 70)
    print("FEATURE ENGINEERING — ABLATION EXPERIMENTS")
    print("=" * 70)

    # --------------------------------------------------------
    # ABLATION 1: Scaling Effect on KNN vs Tree
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 1: Scaling Effect on KNN vs Decision Tree")
    print("=" * 60)
    print("\nKNN uses distance → scaling matters!")
    print("Decision tree uses thresholds → scaling irrelevant.\n")

    X, y = make_classification_data(300)

    # Train/test split
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Scalers
    scalers = {
        'None (raw)': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
    }

    print(f"  Feature ranges: salary [{X[:,0].min():.0f}, {X[:,0].max():.0f}], "
          f"age [{X[:,1].min():.0f}, {X[:,1].max():.0f}]")
    print(f"  Scale ratio: {(X[:,0].max()-X[:,0].min()) / (X[:,1].max()-X[:,1].min()):.0f}x\n")

    print(f"  {'Scaler':20s}  {'KNN Accuracy':>14s}  {'Tree Accuracy':>14s}")
    print(f"  {'-'*55}")

    for name, scaler in scalers.items():
        if scaler is not None:
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        knn = SimpleKNN(k=5).fit(X_tr, y_train)
        tree = SimpleDecisionTree(max_depth=5).fit(X_tr, y_train)

        knn_acc = np.mean(knn.predict(X_te) == y_test)
        tree_acc = np.mean(tree.predict(X_te) == y_test)

        print(f"  {name:20s}  {knn_acc:14.3f}  {tree_acc:14.3f}")

    print("\n  KEY INSIGHT: KNN accuracy jumps significantly with scaling.")
    print("  Tree accuracy is unchanged (scale-invariant).")
    print("  Without scaling, salary (range ~180K) dominates age (range ~52).")

    # --------------------------------------------------------
    # ABLATION 2: Missing Data Imputation
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 2: Missing Data — Mean vs Median vs Indicator")
    print("=" * 60)

    X, y = make_classification_data(300)
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Standardize first
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    missing_fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"\n  {'Missing %':>10s}  {'Mean':>8s}  {'Median':>8s}  {'Indicator':>10s}")
    print(f"  {'-'*45}")

    for frac in missing_fracs:
        if frac == 0:
            knn = SimpleKNN(k=5).fit(X_train_s, y_train)
            acc = np.mean(knn.predict(X_test_s) == y_test)
            print(f"  {frac*100:9.0f}%  {acc:8.3f}  {acc:8.3f}  {acc:10.3f}")
            continue

        accs = {'mean': [], 'median': [], 'indicator': []}
        for trial in range(5):
            X_tr_miss = make_missing_data(X_train_s, frac, random_state=trial)
            X_te_miss = make_missing_data(X_test_s, frac, random_state=trial + 100)

            for method, imputer in [('mean', impute_mean), ('median', impute_median),
                                     ('indicator', impute_indicator)]:
                X_tr_imp = imputer(X_tr_miss)
                X_te_imp = imputer(X_te_miss)

                # Match dimensions for indicator
                if X_te_imp.shape[1] != X_tr_imp.shape[1]:
                    # Pad or truncate test to match train columns
                    n_cols = X_tr_imp.shape[1]
                    if X_te_imp.shape[1] < n_cols:
                        pad = np.zeros((X_te_imp.shape[0], n_cols - X_te_imp.shape[1]))
                        X_te_imp = np.hstack([X_te_imp, pad])
                    else:
                        X_te_imp = X_te_imp[:, :n_cols]

                knn = SimpleKNN(k=5).fit(X_tr_imp, y_train)
                acc = np.mean(knn.predict(X_te_imp) == y_test)
                accs[method].append(acc)

        print(f"  {frac*100:9.0f}%  {np.mean(accs['mean']):8.3f}  "
              f"{np.mean(accs['median']):8.3f}  {np.mean(accs['indicator']):10.3f}")

    print("\n  KEY INSIGHT: All imputation methods degrade gracefully.")
    print("  Indicator imputation can help when missingness is informative.")

    # --------------------------------------------------------
    # ABLATION 3: One-Hot vs Ordinal Encoding
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 3: One-Hot vs Ordinal Encoding")
    print("=" * 60)
    print("\nWhen categories have NO ordering, one-hot is correct.")
    print("Ordinal encoding imposes a FALSE ordering.\n")

    rng = np.random.RandomState(42)
    n = 300

    # Simulate: 3 unordered categories (color) + numeric feature
    # True rule: red → class 0, green → class 1, blue → class 0
    categories = rng.choice(3, n)  # 0=red, 1=green, 2=blue
    numeric = rng.randn(n)
    y = np.zeros(n)
    y[categories == 1] = 1  # green → class 1
    y[(categories == 0) & (numeric > 0)] = 1  # some reds too

    # Ordinal encoding: just use 0, 1, 2
    X_ordinal = np.column_stack([categories.astype(float), numeric])

    # One-hot encoding
    cat_onehot = one_hot_encode(categories)
    X_onehot = np.column_stack([cat_onehot, numeric])

    n_train = 200
    results = {}
    for enc_name, X_enc in [('Ordinal', X_ordinal), ('One-Hot', X_onehot)]:
        X_tr, X_te = X_enc[:n_train], X_enc[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        tree = SimpleDecisionTree(max_depth=5).fit(X_tr, y_tr)
        tree_acc = np.mean(tree.predict(X_te) == y_te)

        knn = SimpleKNN(k=7).fit(X_tr, y_tr)
        knn_acc = np.mean(knn.predict(X_te) == y_te)

        results[enc_name] = {'tree': tree_acc, 'knn': knn_acc}
        print(f"  {enc_name:10s}:  Tree acc = {tree_acc:.3f},  KNN acc = {knn_acc:.3f}")

    print("\n  KEY INSIGHT: Decision trees handle ordinal encoding well (can split")
    print("  at any threshold). KNN suffers because ordinal imposes distances")
    print("  (dist(red=0, blue=2) > dist(red=0, green=1), which is meaningless).")

    # --------------------------------------------------------
    # ABLATION 4: Polynomial Degree Sweep
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 4: Polynomial Degree Sweep (Underfitting → Overfitting)")
    print("=" * 60)

    X, y = make_nonlinear_data(200)
    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"\n  Data: y = sin(x) + noise, 150 train / 50 test")
    print(f"  {'Degree':>8s}  {'# Features':>10s}  {'Train MSE':>10s}  {'Test MSE':>10s}  {'Status':>15s}")
    print(f"  {'-'*60}")

    for degree in [1, 2, 3, 5, 7, 10, 15]:
        X_tr_poly = polynomial_features(X_train, degree)
        X_te_poly = polynomial_features(X_test, degree)

        # Standardize polynomial features
        scaler = StandardScaler().fit(X_tr_poly)
        X_tr_s = scaler.transform(X_tr_poly)
        X_te_s = scaler.transform(X_te_poly)

        model = SimpleLinearRegression().fit(X_tr_s, y_train)
        train_mse = np.mean((model.predict(X_tr_s) - y_train) ** 2)
        test_mse = np.mean((model.predict(X_te_s) - y_test) ** 2)

        # Clamp test MSE for display
        test_mse_disp = min(test_mse, 999.99)

        if degree <= 2:
            status = "UNDERFITTING"
        elif degree <= 7:
            status = "GOOD"
        else:
            status = "OVERFITTING"

        print(f"  {degree:8d}  {X_tr_poly.shape[1]:10d}  "
              f"{train_mse:10.4f}  {test_mse_disp:10.4f}  {status:>15s}")

    print("\n  KEY INSIGHT: Degree 3-5 fits sin(x) well (sweet spot).")
    print("  Too low → can't capture the curve (underfitting).")
    print("  Too high → fits noise instead of signal (overfitting).")

    # --------------------------------------------------------
    # ABLATION 5: Outlier Effect on Linear Regression vs Tree
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 5: Outlier Effect — Linear Regression vs Tree")
    print("=" * 60)

    print(f"\n  {'# Outliers':>12s}  {'Linear MSE':>12s}  {'Tree MSE':>10s}  "
          f"{'Linear Slope':>14s}")
    print(f"  {'-'*55}")

    for n_out in [0, 2, 5, 10, 15]:
        X, y = make_regression_data(100, random_state=42, n_outliers=n_out)

        lr = SimpleLinearRegression().fit(X, y)
        tree = SimpleDecisionTree(max_depth=4).fit(X, y)

        lr_pred = lr.predict(X)
        tree_pred = tree.predict(X)

        lr_mse = np.mean((lr_pred - y) ** 2)
        tree_mse = np.mean((tree_pred - y) ** 2)

        print(f"  {n_out:12d}  {lr_mse:12.2f}  {tree_mse:10.2f}  "
              f"{lr.weights[0]:14.3f} (true=2.5)")

    print("\n  KEY INSIGHT: Linear regression slope shifts toward outliers.")
    print("  Decision tree is robust (splits are threshold-based).")
    print("  With 15 outliers, linear slope is pulled significantly from true value 2.5.")

    # Outlier detection comparison
    print("\n  Outlier detection methods:")
    X, y = make_regression_data(100, random_state=42, n_outliers=10)
    data = np.column_stack([X.ravel(), y])
    zscore_mask = detect_outliers_zscore(data, threshold=2.5)
    iqr_mask = detect_outliers_iqr(data, factor=1.5)
    print(f"    Z-score (|z|>2.5): detected {zscore_mask.sum()} outliers")
    print(f"    IQR (1.5x): detected {iqr_mask.sum()} outliers")

    return True


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualize_scaling_comparison():
    """
    Show before/after scaling and its effect on KNN decision boundaries.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Scaling: Effect on Data & KNN Decision Boundaries',
                 fontsize=14, fontweight='bold')

    X, y = make_classification_data(300)
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scalers = [
        ('No Scaling', None),
        ('StandardScaler', StandardScaler()),
        ('RobustScaler', RobustScaler()),
    ]

    for col, (name, scaler) in enumerate(scalers):
        if scaler is not None:
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train.copy(), X_test.copy()

        # Top row: scatter plot of training data
        ax = axes[0, col]
        for c, color, label in [(0, '#2196F3', 'Class 0'), (1, '#F44336', 'Class 1')]:
            mask = y_train == c
            ax.scatter(X_tr[mask, 0], X_tr[mask, 1], c=color, alpha=0.5,
                      s=20, label=label)
        ax.set_title(f'{name}', fontsize=12)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.legend(fontsize=8)

        # Bottom row: KNN decision boundary
        ax = axes[1, col]
        knn = SimpleKNN(k=5).fit(X_tr, y_train)
        acc = np.mean(knn.predict(X_te) == y_test)

        # Create mesh
        x_min, x_max = X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5
        y_min, y_max = X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5
        # Use coarse grid to keep it fast
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50)
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        Z = knn.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5],
                   colors=['#2196F3', '#F44336'])
        for c, color in [(0, '#2196F3'), (1, '#F44336')]:
            mask = y_train == c
            ax.scatter(X_tr[mask, 0], X_tr[mask, 1], c=color, alpha=0.5,
                      s=20, edgecolors='black', linewidth=0.3)
        ax.set_title(f'KNN Boundary (acc={acc:.3f})', fontsize=11)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/69_scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 69_scaling_comparison.png")
    return fig


def visualize_missing_data():
    """
    Show imputation strategies and their accuracy impact.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Missing Data: Imputation Strategies',
                 fontsize=14, fontweight='bold')

    X, y = make_classification_data(300)
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    n_train = 200

    # Plot 1: Visualize missing data pattern
    ax = axes[0]
    frac = 0.3
    X_miss = make_missing_data(X_s, frac, random_state=42)
    nan_mask = np.isnan(X_miss)

    ax.imshow(nan_mask[:50].astype(float), aspect='auto', cmap='Reds',
              interpolation='nearest')
    ax.set_title(f'Missing Pattern ({frac*100:.0f}% missing)\nFirst 50 samples', fontsize=11)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Sample')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Salary', 'Age'])

    # Plot 2: Imputed values comparison
    ax = axes[1]
    col_idx = 0
    mask_col = np.isnan(X_miss[:, col_idx])
    if mask_col.any():
        true_vals = X_s[mask_col, col_idx]
        mean_vals = np.full_like(true_vals, np.nanmean(X_miss[:, col_idx]))
        median_vals = np.full_like(true_vals, np.nanmedian(X_miss[:, col_idx]))

        n_show = min(20, len(true_vals))
        x_pos = np.arange(n_show)
        ax.scatter(x_pos, true_vals[:n_show], c='green', s=40, zorder=3, label='True value')
        ax.scatter(x_pos, mean_vals[:n_show], c='blue', s=30, marker='^', label='Mean impute')
        ax.scatter(x_pos, median_vals[:n_show], c='red', s=30, marker='s', label='Median impute')
        ax.legend(fontsize=8)
    ax.set_title('Imputed vs True Values\n(Feature 0, missing entries)', fontsize=11)
    ax.set_xlabel('Missing sample index')
    ax.set_ylabel('Value')

    # Plot 3: Accuracy vs missing fraction
    ax = axes[2]
    fracs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    X_train_s, X_test_s = X_s[:n_train], X_s[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    for method_name, imputer, color in [('Mean', impute_mean, '#2196F3'),
                                          ('Median', impute_median, '#F44336'),
                                          ('Indicator', impute_indicator, '#4CAF50')]:
        accs = []
        for frac in fracs:
            trial_accs = []
            for trial in range(3):
                if frac == 0:
                    X_tr_imp = X_train_s.copy()
                    X_te_imp = X_test_s.copy()
                else:
                    X_tr_miss = make_missing_data(X_train_s, frac, random_state=trial)
                    X_te_miss = make_missing_data(X_test_s, frac, random_state=trial + 100)
                    X_tr_imp = imputer(X_tr_miss)
                    X_te_imp = imputer(X_te_miss)
                    # Match dimensions
                    if X_te_imp.shape[1] != X_tr_imp.shape[1]:
                        n_cols = X_tr_imp.shape[1]
                        if X_te_imp.shape[1] < n_cols:
                            pad = np.zeros((X_te_imp.shape[0], n_cols - X_te_imp.shape[1]))
                            X_te_imp = np.hstack([X_te_imp, pad])
                        else:
                            X_te_imp = X_te_imp[:, :n_cols]

                knn = SimpleKNN(k=5).fit(X_tr_imp, y_train)
                trial_accs.append(np.mean(knn.predict(X_te_imp) == y_test))
            accs.append(np.mean(trial_accs))
        ax.plot([f*100 for f in fracs], accs, 'o-', color=color, label=method_name)

    ax.set_title('KNN Accuracy vs Missing %', fontsize=11)
    ax.set_xlabel('Missing fraction (%)')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/69_missing_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 69_missing_data.png")
    return fig


def visualize_polynomial_features():
    """
    Show polynomial degree sweep: underfitting → just right → overfitting.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Polynomial Features: Degree Sweep\n'
                 'Linear model + polynomial features = nonlinear fit',
                 fontsize=14, fontweight='bold')

    X, y = make_nonlinear_data(200)
    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # X values for smooth prediction curve
    X_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)

    degrees = [1, 2, 3, 5, 10, 15]

    for idx, degree in enumerate(degrees):
        ax = axes[idx // 3, idx % 3]

        X_tr_poly = polynomial_features(X_train, degree)
        X_te_poly = polynomial_features(X_test, degree)
        X_pl_poly = polynomial_features(X_plot, degree)

        scaler = StandardScaler().fit(X_tr_poly)
        X_tr_s = scaler.transform(X_tr_poly)
        X_te_s = scaler.transform(X_te_poly)
        X_pl_s = scaler.transform(X_pl_poly)

        model = SimpleLinearRegression().fit(X_tr_s, y_train)

        train_mse = np.mean((model.predict(X_tr_s) - y_train) ** 2)
        test_mse = np.mean((model.predict(X_te_s) - y_test) ** 2)
        test_mse = min(test_mse, 50.0)  # Clamp for display

        y_plot = model.predict(X_pl_s)
        # Clamp predictions for visualization
        y_plot = np.clip(y_plot, y.min() - 2, y.max() + 2)

        ax.scatter(X_train, y_train, c='#2196F3', alpha=0.4, s=15, label='Train')
        ax.scatter(X_test, y_test, c='#F44336', alpha=0.4, s=15, label='Test')
        ax.plot(X_plot, y_plot, 'k-', linewidth=2, label='Fit')

        # True function
        ax.plot(X_plot, np.sin(X_plot), '--', color='green', linewidth=1, alpha=0.5,
                label='True sin(x)')

        if degree <= 2:
            status = "UNDERFITTING"
            color = '#FF9800'
        elif degree <= 7:
            status = "GOOD FIT"
            color = '#4CAF50'
        else:
            status = "OVERFITTING"
            color = '#F44336'

        ax.set_title(f'Degree {degree} ({X_tr_poly.shape[1]} features)\n'
                     f'Train MSE={train_mse:.3f}, Test MSE={test_mse:.3f}',
                     fontsize=10, color=color)
        ax.set_ylim(y.min() - 1.5, y.max() + 1.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('/Users/sid47/ML Algorithms/69_polynomial_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 69_polynomial_features.png")
    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  FEATURE ENGINEERING — Paradigm: DATA TRANSFORMATION")
    print("  Scaling, Imputation, Encoding, Polynomial Features, Outliers")
    print("=" * 70)

    # Run ablation experiments
    ablation_experiments()

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    visualize_scaling_comparison()
    visualize_missing_data()
    visualize_polynomial_features()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Feature Engineering Decisions")
    print("=" * 70)
    print("""
    PREPROCESSING     | WHEN TO USE                    | WATCH OUT FOR
    ------------------+--------------------------------+---------------------------
    StandardScaler    | Default for most algorithms    | Assumes Gaussian-like data
    MinMaxScaler      | Neural networks, [0,1] needed  | Sensitive to outliers
    RobustScaler      | Data has outliers              | Less common, IQR-based
    Polynomial feats  | Linear model + nonlinear data  | Combinatorial explosion
    Mean imputation   | Quick and dirty                | Biases toward mean
    Median imputation | Outliers in missing columns    | Loses variance
    Indicator columns | Missingness is informative     | Adds dimensions
    One-hot encoding  | Unordered categories           | Adds d-1 columns per cat
    Ordinal encoding  | Ordered categories             | False ordering for KNN

    RULES OF THUMB:
    1. ALWAYS scale for: KNN, SVM, PCA, neural networks
    2. NEVER scale for: decision trees, random forests (waste of time)
    3. Polynomial degree 2-3 usually sufficient; regularize above that
    4. Mean imputation is OK for <10% missing; beyond that, be careful
    5. One-hot for KNN/linear; ordinal is fine for trees
    6. Remove outliers for linear models; keep them for trees
    """)
