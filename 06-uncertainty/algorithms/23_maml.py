"""
MAML — Paradigm: META-LEARNING (Learning to Learn)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Model-Agnostic Meta-Learning (MAML) learns an INITIALIZATION θ
such that a FEW gradient steps on a NEW task produces good performance.

Traditional learning: Find θ that works for this task
Meta-learning: Find θ that ADAPTS QUICKLY to ANY task

===============================================================
THE KEY INSIGHT: LEARN A GOOD STARTING POINT
===============================================================

For each task T_i:
    1. Start from θ
    2. Take K gradient steps on T_i's training data → θ'_i
    3. Evaluate on T_i's test data

Meta-objective: Find θ that minimizes test loss AFTER adaptation

    θ* = argmin_θ Σ_i L_test(θ'_i)

where θ'_i = θ - α ∇L_train(θ) (adapted parameters)

This is OPTIMIZATION THROUGH OPTIMIZATION:
    Outer loop optimizes the initialization
    Inner loop adapts to each task

===============================================================
WHY THIS MATTERS: FEW-SHOT LEARNING
===============================================================

Classic ML needs thousands of examples per class.
Humans learn new concepts from ~1-5 examples.

MAML enables:
    - 1-shot learning: one example per class
    - 5-shot learning: five examples per class
    - Rapid adaptation to new tasks

===============================================================
THE ALGORITHM
===============================================================

1. Sample batch of tasks {T_1, ..., T_B}
2. For each task T_i:
   a. Sample K examples (support set)
   b. Compute adapted params: θ'_i = θ - α ∇L_support(θ)
   c. Sample different examples (query set)
   d. Compute query loss: L_query(θ'_i)
3. Meta-update: θ ← θ - β ∇_θ Σ_i L_query(θ'_i)

The key: gradients flow THROUGH the adaptation step!
This requires computing gradients of gradients (second-order).

===============================================================
FIRST-ORDER MAML (FOMAML)
===============================================================

Full MAML needs second derivatives (expensive).
FOMAML approximates by ignoring second-order terms:

    ∇_θ L(θ'_i) ≈ ∇_θ' L(θ'_i)

Much cheaper, often works nearly as well!

===============================================================
TASK DISTRIBUTION MATTERS
===============================================================

MAML learns to adapt from θ to any task in the distribution.

If tasks are too similar: just learning θ is enough
If tasks are too different: no good θ exists
Sweet spot: related but distinct tasks

===============================================================
INDUCTIVE BIAS
===============================================================

1. Good initialization exists for task family
2. Few gradient steps can reach good task-specific params
3. Tasks share common structure (transferable)
4. Inner learning rate α matters

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


def create_sinusoid_tasks(n_tasks, n_support=5, n_query=10, amplitude_range=(0.1, 5.0), phase_range=(0, np.pi)):
    """
    Create sinusoid regression tasks for meta-learning.

    Each task: y = A × sin(x + φ)
    with different amplitude A and phase φ.

    This is the classic MAML benchmark.
    """
    tasks = []

    for _ in range(n_tasks):
        # Random amplitude and phase
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(*phase_range)

        # Sample x values
        x_support = np.random.uniform(-5, 5, n_support)
        x_query = np.random.uniform(-5, 5, n_query)

        # Compute y values
        y_support = amplitude * np.sin(x_support + phase)
        y_query = amplitude * np.sin(x_query + phase)

        tasks.append({
            'x_support': x_support.reshape(-1, 1),
            'y_support': y_support.reshape(-1, 1),
            'x_query': x_query.reshape(-1, 1),
            'y_query': y_query.reshape(-1, 1),
            'amplitude': amplitude,
            'phase': phase
        })

    return tasks


def create_classification_tasks(n_tasks, n_support=5, n_query=10):
    """
    Create N-way K-shot classification tasks.

    Each task has different class boundaries.
    """
    tasks = []

    for _ in range(n_tasks):
        # Random rotation of decision boundary
        angle = np.random.uniform(0, 2 * np.pi)
        w = np.array([np.cos(angle), np.sin(angle)])

        # Generate data
        n_total = n_support + n_query
        X = np.random.randn(n_total * 2, 2)
        y = (X @ w > 0).astype(int)

        # Split by class
        class_0 = X[y == 0]
        class_1 = X[y == 1]

        # Take support and query from each class
        x_support = np.vstack([class_0[:n_support//2 + 1], class_1[:n_support//2 + 1]])
        y_support = np.array([0] * (n_support//2 + 1) + [1] * (n_support//2 + 1))

        x_query = np.vstack([class_0[n_support//2+1:n_support//2+1+n_query//2],
                            class_1[n_support//2+1:n_support//2+1+n_query//2]])
        y_query = np.array([0] * (n_query//2) + [1] * (n_query//2))

        # Shuffle
        perm_s = np.random.permutation(len(y_support))
        perm_q = np.random.permutation(len(y_query))

        tasks.append({
            'x_support': x_support[perm_s],
            'y_support': y_support[perm_s],
            'x_query': x_query[perm_q],
            'y_query': y_query[perm_q],
            'w': w
        })

    return tasks


class SimpleNN:
    """Simple neural network for MAML experiments."""

    def __init__(self, layer_sizes, output_type='regression'):
        self.layer_sizes = layer_sizes
        self.output_type = output_type
        self.n_layers = len(layer_sizes) - 1

        # Initialize weights
        self.params = {}
        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            self.params[f'b{i}'] = np.zeros(layer_sizes[i+1])

    def forward(self, X, params=None):
        """Forward pass with optional external params (for MAML)."""
        if params is None:
            params = self.params

        current = X
        for i in range(self.n_layers - 1):
            current = np.maximum(0, current @ params[f'W{i}'] + params[f'b{i}'])

        # Output layer (no activation for regression)
        output = current @ params[f'W{self.n_layers-1}'] + params[f'b{self.n_layers-1}']

        if self.output_type == 'classification':
            # Sigmoid for binary
            output = 1 / (1 + np.exp(-np.clip(output, -500, 500)))

        return output

    def compute_loss(self, X, y, params=None):
        """Compute loss and gradients."""
        if params is None:
            params = self.params

        pred = self.forward(X, params)

        if self.output_type == 'regression':
            # MSE loss
            loss = np.mean((pred - y) ** 2)
        else:
            # Binary cross-entropy
            pred = np.clip(pred, 1e-10, 1 - 1e-10)
            loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

        return loss, pred

    def compute_gradients(self, X, y, params=None):
        """Compute gradients (simple finite differences for clarity)."""
        if params is None:
            params = self.params

        grads = {}
        eps = 1e-5

        for key in params:
            grad = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index

                # f(x + eps)
                params[key][idx] += eps
                loss_plus, _ = self.compute_loss(X, y, params)
                params[key][idx] -= eps

                # f(x - eps)
                params[key][idx] -= eps
                loss_minus, _ = self.compute_loss(X, y, params)
                params[key][idx] += eps

                # Gradient
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()

            grads[key] = grad

        return grads


class MAML:
    """
    Model-Agnostic Meta-Learning.

    Learns an initialization that can adapt quickly to new tasks.
    """

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=1):
        """
        Parameters:
        -----------
        model : Base neural network
        inner_lr : Learning rate for task adaptation (α)
        meta_lr : Learning rate for meta-update (β)
        inner_steps : Number of gradient steps for adaptation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

    def adapt(self, x_support, y_support, params=None):
        """
        Adapt parameters to a specific task.

        Takes inner_steps gradient steps on support set.
        """
        if params is None:
            params = {k: v.copy() for k, v in self.model.params.items()}
        else:
            params = {k: v.copy() for k, v in params.items()}

        for _ in range(self.inner_steps):
            grads = self.model.compute_gradients(x_support, y_support, params)
            for key in params:
                params[key] -= self.inner_lr * grads[key]

        return params

    def meta_train_step(self, tasks):
        """
        One meta-training step on a batch of tasks.

        For each task:
        1. Adapt on support set
        2. Compute query loss with adapted params
        3. Accumulate meta-gradient
        """
        meta_grads = {k: np.zeros_like(v) for k, v in self.model.params.items()}
        total_loss = 0

        for task in tasks:
            # Adapt to this task
            adapted_params = self.adapt(task['x_support'], task['y_support'])

            # Compute query loss
            loss, _ = self.model.compute_loss(task['x_query'], task['y_query'], adapted_params)
            total_loss += loss

            # FOMAML: use gradients at adapted params directly
            # (Full MAML would require second-order gradients)
            grads = self.model.compute_gradients(task['x_query'], task['y_query'], adapted_params)

            for key in meta_grads:
                meta_grads[key] += grads[key]

        # Average gradients
        for key in meta_grads:
            meta_grads[key] /= len(tasks)

        # Meta-update
        for key in self.model.params:
            self.model.params[key] -= self.meta_lr * meta_grads[key]

        return total_loss / len(tasks)

    def train(self, task_generator, n_iterations=1000, tasks_per_iter=4, verbose=True):
        """Train MAML on task distribution."""
        losses = []

        for i in range(n_iterations):
            # Sample batch of tasks
            tasks = task_generator(tasks_per_iter)

            # Meta-update
            loss = self.meta_train_step(tasks)
            losses.append(loss)

            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Meta-loss: {loss:.4f}")

        return losses

    def evaluate(self, task, n_adaptation_steps=None):
        """Evaluate on a new task."""
        if n_adaptation_steps is None:
            n_adaptation_steps = self.inner_steps

        # Adapt
        params = {k: v.copy() for k, v in self.model.params.items()}
        for _ in range(n_adaptation_steps):
            grads = self.model.compute_gradients(task['x_support'], task['y_support'], params)
            for key in params:
                params[key] -= self.inner_lr * grads[key]

        # Evaluate
        loss, pred = self.model.compute_loss(task['x_query'], task['y_query'], params)
        return loss, pred, params


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # -------- Experiment 1: MAML vs Random Init --------
    print("\n1. MAML vs RANDOM INITIALIZATION")
    print("-" * 40)
    print("5-shot sinusoid regression")

    # Task generator
    def task_gen(n):
        return create_sinusoid_tasks(n, n_support=5, n_query=10)

    # Train MAML
    maml_model = SimpleNN([1, 40, 40, 1], output_type='regression')
    maml = MAML(maml_model, inner_lr=0.01, meta_lr=0.001, inner_steps=1)
    maml.train(task_gen, n_iterations=500, tasks_per_iter=4, verbose=False)

    # Compare on new tasks
    test_tasks = task_gen(10)

    maml_losses = []
    random_losses = []

    for task in test_tasks:
        # MAML
        loss_maml, _, _ = maml.evaluate(task, n_adaptation_steps=5)
        maml_losses.append(loss_maml)

        # Random init
        random_model = SimpleNN([1, 40, 40, 1], output_type='regression')
        random_maml = MAML(random_model, inner_lr=0.01, meta_lr=0.001, inner_steps=1)
        loss_random, _, _ = random_maml.evaluate(task, n_adaptation_steps=5)
        random_losses.append(loss_random)

    print(f"MAML (after 5 steps):   {np.mean(maml_losses):.4f} ± {np.std(maml_losses):.4f}")
    print(f"Random init (5 steps):  {np.mean(random_losses):.4f} ± {np.std(random_losses):.4f}")
    print("→ MAML learns initialization that adapts MUCH faster!")

    # -------- Experiment 2: Number of Inner Steps --------
    print("\n2. EFFECT OF INNER STEPS (Adaptation Steps)")
    print("-" * 40)

    for inner_steps in [1, 3, 5, 10]:
        losses = []
        for task in test_tasks:
            loss, _, _ = maml.evaluate(task, n_adaptation_steps=inner_steps)
            losses.append(loss)
        print(f"inner_steps={inner_steps:<3} loss={np.mean(losses):.4f}")
    print("→ More steps = better adaptation (up to a point)")

    # -------- Experiment 3: Support Set Size (K-shot) --------
    print("\n3. EFFECT OF SUPPORT SET SIZE (K-shot)")
    print("-" * 40)

    for k in [1, 3, 5, 10, 20]:
        # Train MAML for this K
        def task_gen_k(n):
            return create_sinusoid_tasks(n, n_support=k, n_query=10)

        model_k = SimpleNN([1, 40, 40, 1], output_type='regression')
        maml_k = MAML(model_k, inner_lr=0.01, meta_lr=0.001, inner_steps=1)
        maml_k.train(task_gen_k, n_iterations=300, tasks_per_iter=4, verbose=False)

        # Evaluate
        test_tasks_k = task_gen_k(10)
        losses = [maml_k.evaluate(t, n_adaptation_steps=5)[0] for t in test_tasks_k]
        print(f"K={k:<3} shot: loss={np.mean(losses):.4f}")
    print("→ More examples = better adaptation")

    # -------- Experiment 4: Inner Learning Rate --------
    print("\n4. EFFECT OF INNER LEARNING RATE")
    print("-" * 40)

    for inner_lr in [0.001, 0.01, 0.05, 0.1]:
        model = SimpleNN([1, 40, 40, 1], output_type='regression')
        maml_lr = MAML(model, inner_lr=inner_lr, meta_lr=0.001, inner_steps=1)
        maml_lr.train(task_gen, n_iterations=300, tasks_per_iter=4, verbose=False)

        test_tasks = task_gen(10)
        losses = [maml_lr.evaluate(t, n_adaptation_steps=5)[0] for t in test_tasks]
        print(f"inner_lr={inner_lr:<5} loss={np.mean(losses):.4f}")
    print("→ Inner LR affects adaptation speed and stability")

    # -------- Experiment 5: Task Similarity --------
    print("\n5. EFFECT OF TASK SIMILARITY")
    print("-" * 40)
    print("Narrower amplitude range = more similar tasks")

    for amp_range in [(0.5, 1.5), (0.1, 3.0), (0.1, 5.0)]:
        def task_gen_amp(n):
            return create_sinusoid_tasks(n, n_support=5, n_query=10,
                                        amplitude_range=amp_range)

        model = SimpleNN([1, 40, 40, 1], output_type='regression')
        maml_amp = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=1)
        maml_amp.train(task_gen_amp, n_iterations=300, tasks_per_iter=4, verbose=False)

        test_tasks = task_gen_amp(10)
        losses = [maml_amp.evaluate(t, n_adaptation_steps=5)[0] for t in test_tasks]
        print(f"amp_range={amp_range}: loss={np.mean(losses):.4f}")
    print("→ Similar tasks → easier meta-learning")
    print("→ Too diverse tasks → harder to find good init")

    # -------- Experiment 6: Classification Tasks --------
    print("\n6. MAML ON CLASSIFICATION")
    print("-" * 40)

    def clf_task_gen(n):
        return create_classification_tasks(n, n_support=5, n_query=10)

    clf_model = SimpleNN([2, 32, 32, 1], output_type='classification')
    maml_clf = MAML(clf_model, inner_lr=0.1, meta_lr=0.01, inner_steps=1)
    maml_clf.train(clf_task_gen, n_iterations=300, tasks_per_iter=4, verbose=False)

    test_clf = clf_task_gen(10)
    accs = []
    for task in test_clf:
        _, pred, _ = maml_clf.evaluate(task, n_adaptation_steps=5)
        acc = np.mean((pred > 0.5).astype(int).flatten() == task['y_query'])
        accs.append(acc)
    print(f"5-shot classification accuracy: {np.mean(accs):.3f}")


def visualize_maml():
    """Visualize MAML learning and adaptation."""
    print("\n" + "="*60)
    print("MAML VISUALIZATION")
    print("="*60)

    np.random.seed(42)

    def task_gen(n):
        return create_sinusoid_tasks(n, n_support=5, n_query=10)

    # Train MAML
    model = SimpleNN([1, 40, 40, 1], output_type='regression')
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=1)
    losses = maml.train(task_gen, n_iterations=500, tasks_per_iter=4, verbose=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Plot 1: Meta-training loss
    ax = axes[0, 0]
    ax.plot(losses, alpha=0.7)
    ax.set_xlabel('Meta-iteration')
    ax.set_ylabel('Meta-loss')
    ax.set_title('MAML Training\n(Meta-loss decreases)')

    # Plot 2: Adaptation on a single task
    ax = axes[0, 1]
    task = task_gen(1)[0]
    x_plot = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_true = task['amplitude'] * np.sin(x_plot + task['phase'])

    # Before adaptation
    y_before = model.forward(x_plot)
    ax.plot(x_plot, y_before, 'b--', label='Before adaptation', alpha=0.7)

    # After adaptation
    _, y_after, _ = maml.evaluate(task, n_adaptation_steps=5)
    adapted_params = maml.adapt(task['x_support'], task['y_support'])
    y_adapted = model.forward(x_plot, adapted_params)

    ax.plot(x_plot, y_true, 'g-', label='True function', linewidth=2)
    ax.plot(x_plot, y_adapted, 'r-', label='After 5 steps', linewidth=2)
    ax.scatter(task['x_support'], task['y_support'], c='black', s=100, zorder=5, label='Support')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Task Adaptation\n(5-shot learning)')
    ax.legend()

    # Plot 3: Multiple tasks
    ax = axes[0, 2]
    tasks = task_gen(3)
    colors = ['r', 'b', 'g']  # Use single-char codes for format strings
    color_names = ['red', 'blue', 'green']

    for i, task in enumerate(tasks):
        x_plot = np.linspace(-5, 5, 100).reshape(-1, 1)
        y_true = task['amplitude'] * np.sin(x_plot + task['phase'])

        adapted_params = maml.adapt(task['x_support'], task['y_support'])
        y_pred = model.forward(x_plot, adapted_params)

        ax.plot(x_plot, y_true, f'{colors[i]}--', alpha=0.5)
        ax.plot(x_plot, y_pred, colors[i], alpha=0.7, label=f'Task {i+1}')
        ax.scatter(task['x_support'], task['y_support'], c=color_names[i], s=50)

    ax.set_title('MAML on Multiple Tasks\n(Same init adapts to all)')
    ax.legend()

    # Plot 4: Steps of adaptation
    ax = axes[1, 0]
    task = task_gen(1)[0]
    x_plot = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_true = task['amplitude'] * np.sin(x_plot + task['phase'])

    params = {k: v.copy() for k, v in model.params.items()}
    for step in [0, 1, 3, 5, 10]:
        y_pred = model.forward(x_plot, params)
        ax.plot(x_plot, y_pred, alpha=0.7, label=f'Step {step}')
        if step < 10:
            for _ in range(1 if step == 0 else (step - len([0, 1, 3, 5, 10][:list([0, 1, 3, 5, 10]).index(step)]))):
                pass
            # Take gradient step
            grads = model.compute_gradients(task['x_support'], task['y_support'], params)
            for key in params:
                params[key] -= 0.01 * grads[key]

    ax.plot(x_plot, y_true, 'k--', linewidth=2, label='True')
    ax.scatter(task['x_support'], task['y_support'], c='black', s=100, zorder=5)
    ax.set_title('Adaptation Steps\n(Converges to task)')
    ax.legend(loc='upper right')

    # Plot 5: Random init comparison
    ax = axes[1, 1]
    task = task_gen(1)[0]
    x_plot = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_true = task['amplitude'] * np.sin(x_plot + task['phase'])

    # MAML adaptation
    adapted_maml = maml.adapt(task['x_support'], task['y_support'])
    for _ in range(4):
        grads = model.compute_gradients(task['x_support'], task['y_support'], adapted_maml)
        for key in adapted_maml:
            adapted_maml[key] -= 0.01 * grads[key]
    y_maml = model.forward(x_plot, adapted_maml)

    # Random init adaptation
    random_model = SimpleNN([1, 40, 40, 1], output_type='regression')
    adapted_random = {k: v.copy() for k, v in random_model.params.items()}
    for _ in range(5):
        grads = random_model.compute_gradients(task['x_support'], task['y_support'], adapted_random)
        for key in adapted_random:
            adapted_random[key] -= 0.01 * grads[key]
    y_random = random_model.forward(x_plot, adapted_random)

    ax.plot(x_plot, y_true, 'k--', label='True', linewidth=2)
    ax.plot(x_plot, y_maml, 'g-', label='MAML (5 steps)', linewidth=2)
    ax.plot(x_plot, y_random, 'r-', label='Random (5 steps)', linewidth=2, alpha=0.7)
    ax.scatter(task['x_support'], task['y_support'], c='black', s=100, zorder=5)
    ax.set_title('MAML vs Random Init\n(MAML adapts faster)')
    ax.legend()

    # Plot 6: K-shot performance
    ax = axes[1, 2]
    k_values = [1, 2, 3, 5, 10]
    maml_losses = []
    random_losses = []

    for k in k_values:
        tasks_k = create_sinusoid_tasks(5, n_support=k, n_query=10)

        maml_k_losses = []
        random_k_losses = []

        for task in tasks_k:
            # MAML
            loss_maml, _, _ = maml.evaluate(task, n_adaptation_steps=5)
            maml_k_losses.append(loss_maml)

            # Random
            random_model = SimpleNN([1, 40, 40, 1], output_type='regression')
            random_maml = MAML(random_model, inner_lr=0.01, inner_steps=5)
            loss_random, _, _ = random_maml.evaluate(task, n_adaptation_steps=5)
            random_k_losses.append(loss_random)

        maml_losses.append(np.mean(maml_k_losses))
        random_losses.append(np.mean(random_k_losses))

    ax.plot(k_values, maml_losses, 'g-o', label='MAML', markersize=8)
    ax.plot(k_values, random_losses, 'r-o', label='Random init', markersize=8)
    ax.set_xlabel('K (support set size)')
    ax.set_ylabel('Loss after adaptation')
    ax.set_title('K-shot Performance\n(MAML wins across all K)')
    ax.legend()

    plt.suptitle('MAML — Model-Agnostic Meta-Learning\n'
                 'Learning to learn: finding initialization that adapts quickly',
                 fontsize=12)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("MAML — Model-Agnostic Meta-Learning")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    Learn an INITIALIZATION θ that adapts quickly to new tasks.

THE KEY INSIGHT:
    Meta-objective: find θ such that after K gradient steps,
    performance on NEW task is good.

    θ* = argmin_θ Σ_tasks L_query(θ - α∇L_support(θ))

WHY IT MATTERS:
    - Few-shot learning (1-5 examples)
    - Rapid task adaptation
    - Transfer across related tasks

THE ALGORITHM:
    For each task:
        1. Adapt: θ' = θ - α∇L_support
        2. Evaluate: L_query(θ')
    Meta-update: θ ← θ - β∇_θ Σ L_query(θ')

FOMAML:
    Ignores second-order terms
    Much faster, often works as well
    """)

    ablation_experiments()

    fig = visualize_maml()
    save_path = '/Users/sid47/ML Algorithms/23_maml.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. MAML learns initialization, not final weights
2. Few gradient steps → good task-specific performance
3. Works across regression and classification
4. Task similarity affects meta-learning success
5. Inner LR and steps are key hyperparameters
6. FOMAML: fast approximation that works well
    """)
