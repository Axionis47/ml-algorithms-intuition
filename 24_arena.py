"""
ARENA — Master Comparison of All ML Paradigms

===============================================================
THE ULTIMATE SHOWDOWN
===============================================================

Every model. Every dataset. One comparison grid.

This arena reveals:
    - Which paradigms win where
    - The price of different inductive biases
    - Trade-offs between simplicity and power

===============================================================
THE PARADIGMS
===============================================================

P1: PROJECTION (Linear models)
    - Linear Regression, Logistic Regression

P2: MEMORY (Instance-based)
    - KNN

P3: PROBABILISTIC (Model distributions)
    - Naive Bayes, Gaussian Process, Bayesian Linear Reg

P4: PARTITIONING (Split the space)
    - Decision Tree

P5: MARGIN (Find the gap)
    - SVM

P6: COMMITTEE (Combine learners)
    - Random Forest, Gradient Boosting, AdaBoost

P7: LEARNED FEATURES (Neural networks)
    - MLP, CNN, RNN/LSTM, Transformer

P8: GENERATIVE (Model the data)
    - GMM, HMM, VAE

P9: UNCERTAINTY (Know what you don't know)
    - Conformal Prediction, MC Dropout, Calibration

P10: ONLINE & META (Adapt on the fly)
    - Online Learning, MAML

===============================================================
THE DATASETS
===============================================================

- linear: Can you find a hyperplane?
- circles: Can you bend?
- xor: Do you understand interactions?
- spiral: Can you handle complex manifolds?
- moons: Robust to noise?
- high_dim: Curse of dimensionality?
- imbalanced: Handle class imbalance?
- dist_shift: Robust to distribution shift?

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')
from importlib import import_module

# Import datasets
datasets_module = import_module('00_datasets')
get_all_datasets = datasets_module.get_all_datasets
get_2d_datasets = datasets_module.get_2d_datasets
accuracy = datasets_module.accuracy


def load_models():
    """Load all available models."""
    models = {}

    # P1: Projection
    try:
        mod = import_module('02_logistic_regression')
        models['Logistic'] = lambda: mod.LogisticRegression(learning_rate=0.1, n_iterations=200)
    except:
        pass

    # P2: Memory
    try:
        mod = import_module('03_knn')
        models['KNN'] = lambda: mod.KNN(k=5)
    except:
        pass

    # P3: Probabilistic
    try:
        mod = import_module('04_naive_bayes')
        models['NaiveBayes'] = lambda: mod.NaiveBayes()
    except:
        pass

    try:
        mod = import_module('05_gaussian_process')
        models['GP'] = lambda: mod.GaussianProcessClassifier(kernel='rbf', length_scale=1.0)
    except:
        pass

    # P4: Partitioning
    try:
        mod = import_module('07_decision_tree')
        models['DecisionTree'] = lambda: mod.DecisionTreeClassifier(max_depth=10)
    except:
        pass

    # P5: Margin
    try:
        mod = import_module('08_svm')
        models['SVM'] = lambda: mod.SVM(kernel='rbf', C=1.0)
    except:
        pass

    # P6: Committee
    try:
        mod = import_module('09_random_forest')
        models['RandomForest'] = lambda: mod.RandomForestClassifier(n_estimators=50, max_depth=10)
    except:
        pass

    try:
        mod = import_module('10_gradient_boosting')
        models['GradBoost'] = lambda: mod.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    except:
        pass

    try:
        mod = import_module('11_adaboost')
        models['AdaBoost'] = lambda: mod.AdaBoostClassifier(n_estimators=50)
    except:
        pass

    # P7: Learned Features
    try:
        mod = import_module('12_mlp')
        models['MLP'] = lambda: mod.MLP(layer_sizes=[2, 32, 32, 2], activation='relu')
    except:
        pass

    return models


def run_arena():
    """Run the arena comparison."""
    print("\n" + "="*70)
    print(" " * 25 + "THE ARENA")
    print("="*70)

    np.random.seed(42)

    models = load_models()
    datasets = get_all_datasets()

    # Filter to 2D datasets for fair comparison
    dataset_names = ['linear', 'circles', 'xor', 'spiral', 'moons', 'imbalanced', 'dist_shift']

    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    print(f"Testing on {len(dataset_names)} datasets: {dataset_names}")

    # Results matrix
    results = {}
    for name in models:
        results[name] = {}

    # Run comparisons
    print("\n" + "-"*70)
    header = f"{'Model':<15}"
    for ds in dataset_names:
        header += f"{ds[:8]:<10}"
    print(header)
    print("-"*70)

    for model_name, model_fn in models.items():
        row = f"{model_name:<15}"

        for ds_name in dataset_names:
            try:
                X_tr, X_te, y_tr, y_te = datasets[ds_name]

                # Handle multi-class datasets
                if ds_name == 'clustered':
                    y_tr = (y_tr > 2).astype(int)
                    y_te = (y_te > 2).astype(int)

                # Handle high-dim differently
                if ds_name == 'high_dim' and model_name in ['GP']:
                    results[model_name][ds_name] = -1
                    row += f"{'skip':<10}"
                    continue

                model = model_fn()

                # Adjust input size for models that need it
                if model_name == 'MLP' and ds_name == 'high_dim':
                    mod = import_module('12_mlp')
                    model = mod.MLP(layer_sizes=[X_tr.shape[1], 32, 32, 2], activation='relu')

                model.fit(X_tr, y_tr)

                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_te)
                else:
                    y_pred = np.argmax(model.predict_proba(X_te), axis=1)

                acc = accuracy(y_te, y_pred)
                results[model_name][ds_name] = acc
                row += f"{acc:<10.3f}"

            except Exception as e:
                results[model_name][ds_name] = -1
                row += f"{'err':<10}"

        print(row)

    print("-"*70)

    return results, dataset_names


def analyze_results(results, dataset_names):
    """Analyze arena results."""
    print("\n" + "="*70)
    print(" " * 20 + "ANALYSIS")
    print("="*70)

    # Best model per dataset
    print("\nBEST MODEL PER DATASET:")
    print("-"*40)
    for ds in dataset_names:
        best_model = None
        best_acc = -1
        for model_name in results:
            if ds in results[model_name] and results[model_name][ds] > best_acc:
                best_acc = results[model_name][ds]
                best_model = model_name
        if best_model:
            print(f"{ds:<15} → {best_model:<15} ({best_acc:.3f})")

    # Average accuracy per model
    print("\nAVERAGE ACCURACY PER MODEL:")
    print("-"*40)
    avg_accs = []
    for model_name in results:
        accs = [v for v in results[model_name].values() if v > 0]
        if accs:
            avg = np.mean(accs)
            avg_accs.append((model_name, avg))

    for model_name, avg in sorted(avg_accs, key=lambda x: -x[1]):
        print(f"{model_name:<15} {avg:.3f}")

    # Paradigm insights
    print("\n" + "="*70)
    print(" " * 15 + "PARADIGM INSIGHTS")
    print("="*70)

    insights = """
WHAT THE ARENA REVEALS:

1. LINEAR (linear) → Most models succeed
   - Even simple models work on linearly separable data

2. NONLINEAR (circles, xor) → Trees and NNs shine
   - Linear models fail, need feature space transformation
   - XOR specifically tests interaction detection

3. COMPLEX (spiral) → Deep learners dominate
   - Requires learning complex decision boundaries
   - Shallow methods struggle

4. NOISE (moons) → Regularization matters
   - Models that overfit struggle
   - Ensemble methods robust

5. HIGH DIMENSIONALITY → Feature selection wins
   - KNN fails (curse of dimensionality)
   - Naive Bayes surprisingly good (independence assumption)

6. IMBALANCE → Class-aware methods needed
   - Simple accuracy misleading
   - Need weighted or probabilistic approaches

7. DISTRIBUTION SHIFT → Robustness varies
   - Tests generalization to different distributions
   - Reveals overfitting to training distribution

KEY TAKEAWAYS:
- No single model wins everywhere
- Match the inductive bias to the problem
- Ensembles are strong general-purpose
- NNs powerful but need more data/tuning
- Simple models often surprisingly competitive
"""
    print(insights)


def create_decision_boundary_grid():
    """Create decision boundary visualization for all models."""
    print("\n" + "="*70)
    print(" " * 15 + "DECISION BOUNDARY GRID")
    print("="*70)

    models = load_models()
    datasets = get_2d_datasets()

    # Use subset for visualization
    model_names = ['Logistic', 'KNN', 'NaiveBayes', 'DecisionTree', 'SVM', 'RandomForest', 'MLP']
    dataset_names = ['linear', 'circles', 'xor', 'moons']

    # Filter to available models
    model_names = [m for m in model_names if m in models]

    n_models = len(model_names)
    n_datasets = len(dataset_names)

    fig, axes = plt.subplots(n_datasets, n_models, figsize=(3*n_models, 3*n_datasets))

    for i, ds_name in enumerate(dataset_names):
        X_tr, X_te, y_tr, y_te = datasets[ds_name]
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        X_grid = np.c_[xx.ravel(), yy.ravel()]

        for j, model_name in enumerate(model_names):
            ax = axes[i, j] if n_datasets > 1 else axes[j]

            try:
                model = models[model_name]()
                model.fit(X_tr, y_tr)

                # Get predictions on grid
                if hasattr(model, 'predict'):
                    Z = model.predict(X_grid)
                else:
                    Z = np.argmax(model.predict_proba(X_grid), axis=1)

                Z = Z.reshape(xx.shape)

                # Plot decision boundary
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu',
                          edgecolors='black', s=20, alpha=0.7)

                acc = accuracy(y_te, model.predict(X_te) if hasattr(model, 'predict')
                              else np.argmax(model.predict_proba(X_te), axis=1))

                if i == 0:
                    ax.set_title(f'{model_name}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{ds_name}', fontsize=10)

                ax.text(0.05, 0.95, f'{acc:.2f}', transform=ax.transAxes,
                       fontsize=8, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            except Exception as e:
                ax.text(0.5, 0.5, 'Error', transform=ax.transAxes,
                       ha='center', va='center')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('DECISION BOUNDARY ARENA\n'
                 'Same data, different inductive biases → different boundaries',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def paradigm_summary():
    """Print paradigm summary."""
    summary = """
╔══════════════════════════════════════════════════════════════════════╗
║                    ML PARADIGM CHEAT SHEET                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PROJECTION (Linear)                                                 ║
║    What: Find hyperplane                                             ║
║    Good: Fast, interpretable, convex                                 ║
║    Bad: Can't bend                                                   ║
║                                                                      ║
║  MEMORY (KNN)                                                        ║
║    What: Store and compare                                           ║
║    Good: No training, simple                                         ║
║    Bad: Curse of dimensionality, slow                                ║
║                                                                      ║
║  PROBABILISTIC (Bayes, GP)                                           ║
║    What: Model distributions                                         ║
║    Good: Uncertainty, missing data                                   ║
║    Bad: Strong assumptions                                           ║
║                                                                      ║
║  PARTITIONING (Trees)                                                ║
║    What: Recursive splits                                            ║
║    Good: Interpretable, handles interactions                         ║
║    Bad: Overfits, axis-aligned only                                  ║
║                                                                      ║
║  MARGIN (SVM)                                                        ║
║    What: Maximum margin                                              ║
║    Good: Kernel trick, few support vectors                           ║
║    Bad: Slow for large N, kernel choice                              ║
║                                                                      ║
║  COMMITTEE (Ensemble)                                                ║
║    What: Combine weak learners                                       ║
║    Good: Robust, reduces variance/bias                               ║
║    Bad: Less interpretable, more compute                             ║
║                                                                      ║
║  LEARNED FEATURES (Neural Nets)                                      ║
║    What: Differentiable function composition                         ║
║    Good: Learns representations, universal                           ║
║    Bad: Needs data, black box, hyperparameters                       ║
║                                                                      ║
║  GENERATIVE (GMM, VAE)                                               ║
║    What: Model data generation                                       ║
║    Good: Can generate, understand data                               ║
║    Bad: Harder to train, model assumptions                           ║
║                                                                      ║
║  UNCERTAINTY (Conformal, MC Dropout)                                 ║
║    What: Know what you don't know                                    ║
║    Good: Calibrated, safety-critical                                 ║
║    Bad: Larger outputs (sets), more compute                          ║
║                                                                      ║
║  META-LEARNING (MAML)                                                ║
║    What: Learn to learn                                              ║
║    Good: Few-shot adaptation                                         ║
║    Bad: Needs task distribution, complex training                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(summary)


if __name__ == '__main__':
    print("="*70)
    print(" " * 20 + "ML ARENA")
    print(" " * 10 + "All Models × All Datasets")
    print("="*70)

    # Run arena
    results, dataset_names = run_arena()

    # Analyze
    analyze_results(results, dataset_names)

    # Paradigm summary
    paradigm_summary()

    # Create visualization
    try:
        fig = create_decision_boundary_grid()
        save_path = '/Users/sid47/ML Algorithms/24_arena.png'
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\n" + "="*70)
    print(" " * 15 + "ARENA COMPLETE")
    print("="*70)
    print("""
You now have implementations of every major ML paradigm:

  ✓ Linear models (01, 02)
  ✓ Instance-based (03)
  ✓ Probabilistic (04, 05, 06)
  ✓ Tree-based (07)
  ✓ Kernel methods (08)
  ✓ Ensembles (09, 10, 11)
  ✓ Neural networks (12, 13, 14, 15)
  ✓ Generative models (16, 17, 18)
  ✓ Uncertainty quantification (19, 20, 21)
  ✓ Online & Meta (22, 23)

Each with:
  - Core intuition in docstrings
  - From-scratch implementation
  - Ablation experiments showing what breaks
  - Visualizations of decision boundaries/behavior

Run any file: python "XX_name.py"

The key insight: MATCH THE INDUCTIVE BIAS TO YOUR PROBLEM.
No model is universally best. Understanding WHY each works
helps you design better solutions for new problems.
    """)
