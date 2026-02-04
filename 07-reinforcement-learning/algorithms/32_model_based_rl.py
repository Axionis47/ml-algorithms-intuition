"""
Model-Based RL — Learning World Models
======================================

Paradigm: LEARN THE ENVIRONMENT, THEN PLAN

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of learning directly from environment interactions,
LEARN A MODEL of the environment, then use it to PLAN.

MODEL-FREE:
    Act → Observe → Update value/policy directly
    "Learn from experience"

MODEL-BASED:
    Act → Observe → Update MODEL → PLAN with model → Act
    "Learn the world, then imagine"

THE WORLD MODEL:
    f_θ(s, a) → s', r

    Given (state, action), predict:
    - Next state s'
    - Reward r

===============================================================
WHY MODEL-BASED?
===============================================================

BENEFITS:
1. SAMPLE EFFICIENCY: Learn from imagined experience
2. PLANNING: Can look ahead without real interaction
3. TRANSFER: Model can generalize to new tasks
4. INTERPRETABILITY: Can inspect what model learned

CHALLENGES:
1. MODEL ERROR: Errors compound over time
2. COMPLEXITY: Learning accurate models is hard
3. PLANNING COST: Search over possible futures

===============================================================
DYNA-Q: SIMPLE MODEL-BASED RL
===============================================================

Combines model-free learning with model-based planning.

ALGORITHM:
1. Act in real environment, observe (s, a, r, s')
2. Update model: f(s, a) → (r, s')
3. Update Q with REAL experience: Q(s,a) ← ...
4. PLANNING: For k steps:
   - Sample past (s, a) from memory
   - Simulate: r, s' = f(s, a)
   - Update Q with SIMULATED experience

More planning = faster learning (if model is good)

===============================================================
MODEL PREDICTIVE CONTROL (MPC)
===============================================================

At each step:
1. Sample many random action sequences
2. Simulate each with learned model
3. Pick action sequence with highest reward
4. Execute FIRST action only
5. Repeat

No explicit policy! Pure planning with learned model.

===============================================================
THE MODEL ERROR PROBLEM
===============================================================

Model errors COMPOUND over long horizons:

True trajectory:   s0 → s1 → s2 → s3 → s4
Model trajectory:  s0 → s1' → s2'' → s3''' → s4''''
                         ↑ small error amplifies!

SOLUTIONS:
1. SHORT HORIZONS: Don't plan too far ahead
2. ENSEMBLES: Multiple models, use uncertainty
3. DYNA: Mix real and simulated experience
4. MODEL AWARENESS: Adjust trust based on accuracy

===============================================================
INDUCTIVE BIAS
===============================================================

1. WORLD IS LEARNABLE: Assumes environment has structure
2. MARKOV: State is sufficient for prediction
3. STATIONARY: Dynamics don't change
4. MODEL FAMILY: Neural net may not capture all dynamics

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class TabularEnvironmentModel:
    """
    Tabular model of environment dynamics.

    Learns: P(s', r | s, a) from experience.
    """

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        # Counts for estimating transition probabilities
        self.counts = np.zeros((n_states, n_actions, n_states))
        self.reward_sum = np.zeros((n_states, n_actions))
        self.reward_count = np.zeros((n_states, n_actions))

        # Store observed (s, a) pairs for sampling
        self.observed_pairs = set()

    def update(self, state, action, reward, next_state):
        """Update model with observed transition."""
        self.counts[state, action, next_state] += 1
        self.reward_sum[state, action] += reward
        self.reward_count[state, action] += 1
        self.observed_pairs.add((state, action))

    def predict(self, state, action):
        """Predict next state and reward."""
        # Expected reward
        if self.reward_count[state, action] > 0:
            reward = self.reward_sum[state, action] / self.reward_count[state, action]
        else:
            reward = 0

        # Most likely next state
        counts = self.counts[state, action]
        if np.sum(counts) > 0:
            probs = counts / np.sum(counts)
            next_state = np.argmax(probs)  # Deterministic prediction
            # Or sample: next_state = np.random.choice(self.n_states, p=probs)
        else:
            next_state = state  # Default: stay in place

        return next_state, reward

    def sample_observed(self):
        """Sample a previously observed (state, action) pair."""
        if not self.observed_pairs:
            return None, None
        pair = list(self.observed_pairs)[np.random.randint(len(self.observed_pairs))]
        return pair


class NeuralEnvironmentModel:
    """
    Neural network model of environment dynamics.

    f_θ(s, a) → s', r
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Network for predicting (Δs, r) given (s, a)
        input_dim = state_dim + action_dim

        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)

        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)

        # Output: [Δs (state change), r (reward)]
        self.W_state = np.random.randn(hidden_dim, state_dim) * 0.1
        self.b_state = np.zeros(state_dim)

        self.W_reward = np.random.randn(hidden_dim, 1) * 0.1
        self.b_reward = np.zeros(1)

        # Experience buffer for training
        self.buffer = []
        self.buffer_size = 10000

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state, action):
        """Predict state change and reward."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        x = np.concatenate([state, action], axis=-1)

        h = self.relu(x @ self.W1 + self.b1)
        h = self.relu(h @ self.W2 + self.b2)

        delta_state = h @ self.W_state + self.b_state
        reward = h @ self.W_reward + self.b_reward

        return delta_state, reward.squeeze()

    def predict(self, state, action):
        """Predict next state and reward."""
        delta_state, reward = self.forward(state, action)
        next_state = state + delta_state
        return next_state.squeeze(), reward.item() if hasattr(reward, 'item') else float(reward)

    def update(self, state, action, reward, next_state, lr=0.001):
        """Update model with single transition."""
        # Compute prediction
        delta_pred, reward_pred = self.forward(state.reshape(1, -1),
                                                action.reshape(1, -1))

        # Targets
        delta_true = next_state - state
        reward_true = reward

        # Simple gradient step (MSE loss)
        delta_error = delta_pred - delta_true.reshape(1, -1)
        reward_error = reward_pred - reward_true

        # Backprop through last layer (simplified)
        self.W_state -= lr * delta_error.T @ delta_error
        self.W_reward -= lr * reward_error * reward_error

    def store(self, state, action, reward, next_state):
        """Store transition in buffer."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))


class GridWorldEnv:
    """
    Simple gridworld environment for testing.

    Goal: Navigate from start to goal.
    """

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Down, Left, Right
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        """Reset to start position."""
        self.pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        """Convert position to state index."""
        return self.pos[0] * self.size + self.pos[1]

    def _get_pos(self, state):
        """Convert state index to position."""
        return (state // self.size, state % self.size)

    def step(self, action):
        """Take action, return (state, reward, done)."""
        row, col = self.pos

        # Move
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)

        self.pos = (row, col)

        # Reward
        if self.pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False

        return self._get_state(), reward, done


class DynaQ:
    """
    Dyna-Q: Combines Q-learning with model-based planning.

    1. Learn from real experience (model-free)
    2. Learn environment model
    3. Plan with simulated experience (model-based)
    """

    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.1,
                 epsilon=0.1, n_planning=10):
        """
        Parameters:
        - n_states: Number of states
        - n_actions: Number of actions
        - gamma: Discount factor
        - alpha: Learning rate
        - epsilon: Exploration rate
        - n_planning: Number of planning steps per real step
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_planning = n_planning

        # Q-table
        self.Q = np.zeros((n_states, n_actions))

        # Environment model
        self.model = TabularEnvironmentModel(n_states, n_actions)

        # History
        self.history = {'episode_rewards': [], 'episode_lengths': []}

    def select_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update_q(self, state, action, reward, next_state, done):
        """Q-learning update."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def planning_step(self):
        """Single planning step using learned model."""
        # Sample previously observed (s, a)
        state, action = self.model.sample_observed()
        if state is None:
            return

        # Simulate with model
        next_state, reward = self.model.predict(state, action)

        # Update Q with simulated experience
        self.update_q(state, action, reward, next_state, done=False)

    def train_episode(self, env):
        """Train for one episode."""
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Select action
            action = self.select_action(state)

            # Execute action
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            # Update model
            self.model.update(state, action, reward, next_state)

            # Q-learning update (real experience)
            self.update_q(state, action, reward, next_state, done)

            # Planning steps (simulated experience)
            for _ in range(self.n_planning):
                self.planning_step()

            state = next_state

            if done or steps > 200:
                break

        self.history['episode_rewards'].append(total_reward)
        self.history['episode_lengths'].append(steps)

        return total_reward


class MPC:
    """
    Model Predictive Control.

    At each step:
    1. Sample action sequences
    2. Evaluate with learned model
    3. Execute best first action
    """

    def __init__(self, env_model, horizon=5, n_samples=100, gamma=0.99):
        """
        Parameters:
        - env_model: Learned environment model
        - horizon: Planning horizon
        - n_samples: Number of action sequences to sample
        """
        self.model = env_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.gamma = gamma

    def select_action(self, state, n_actions):
        """Select action using random shooting MPC."""
        best_return = -np.inf
        best_action = 0

        for _ in range(self.n_samples):
            # Sample random action sequence
            actions = np.random.randint(0, n_actions, size=self.horizon)

            # Simulate trajectory
            total_return = 0
            current_state = state

            for t, action in enumerate(actions):
                next_state, reward = self.model.predict(current_state, action)
                total_return += (self.gamma ** t) * reward
                current_state = next_state

            # Track best
            if total_return > best_return:
                best_return = total_return
                best_action = actions[0]

        return best_action


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_model_based_rl():
    """
    Comprehensive model-based RL visualization:
    1. Dyna-Q vs Q-learning
    2. Planning steps effect
    3. Model accuracy
    4. MPC planning
    5. Sample efficiency
    6. Summary
    """
    print("\n" + "="*60)
    print("MODEL-BASED RL VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: Dyna-Q vs Q-Learning ============
    ax1 = fig.add_subplot(2, 3, 1)

    # Q-learning (no planning)
    env = GridWorldEnv(size=5)
    q_agent = DynaQ(env.n_states, env.n_actions, n_planning=0)

    q_rewards = []
    for episode in range(100):
        r = q_agent.train_episode(env)
        q_rewards.append(r)

    # Dyna-Q (with planning)
    env = GridWorldEnv(size=5)
    dyna_agent = DynaQ(env.n_states, env.n_actions, n_planning=10)

    dyna_rewards = []
    for episode in range(100):
        r = dyna_agent.train_episode(env)
        dyna_rewards.append(r)

    # Smooth and plot
    window = 10
    q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    dyna_smooth = np.convolve(dyna_rewards, np.ones(window)/window, mode='valid')

    ax1.plot(q_smooth, 'b-', linewidth=2, label='Q-learning (no planning)')
    ax1.plot(dyna_smooth, 'r-', linewidth=2, label='Dyna-Q (n=10)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Dyna-Q vs Q-Learning\nPlanning speeds up learning!')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Effect of Planning Steps ============
    ax2 = fig.add_subplot(2, 3, 2)

    planning_values = [0, 5, 10, 20, 50]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(planning_values)))

    for n_plan, color in zip(planning_values, colors):
        env = GridWorldEnv(size=5)
        agent = DynaQ(env.n_states, env.n_actions, n_planning=n_plan)

        rewards = []
        for episode in range(50):
            r = agent.train_episode(env)
            rewards.append(r)

        smooth = np.convolve(rewards, np.ones(5)/5, mode='valid')
        ax2.plot(smooth, color=color, linewidth=2, label=f'n={n_plan}')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Effect of Planning Steps\nMore planning = faster learning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Model Accuracy Over Time ============
    ax3 = fig.add_subplot(2, 3, 3)

    env = GridWorldEnv(size=5)
    agent = DynaQ(env.n_states, env.n_actions, n_planning=10)

    # Track model accuracy
    model_errors = []

    for episode in range(50):
        state = env.reset()
        episode_errors = []

        for step in range(50):
            action = agent.select_action(state)
            true_next_state, true_reward, done = env.step(action)

            # Model prediction
            pred_next_state, pred_reward = agent.model.predict(state, action)

            # Error (if model has data for this (s,a))
            if (state, action) in agent.model.observed_pairs:
                error = abs(pred_next_state - true_next_state) + abs(pred_reward - true_reward)
                episode_errors.append(error)

            agent.model.update(state, action, true_reward, true_next_state)
            agent.update_q(state, action, true_reward, true_next_state, done)

            state = true_next_state
            if done:
                break

        if episode_errors:
            model_errors.append(np.mean(episode_errors))

    ax3.plot(model_errors, 'g-', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Model Error')
    ax3.set_title('Model Accuracy Improves\nAs more data is collected')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Learned Q-values ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Show Q-values as heatmap
    q_max = np.max(dyna_agent.Q, axis=1).reshape(5, 5)

    im = ax4.imshow(q_max, cmap='RdYlGn', aspect='equal')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    ax4.set_title('Learned Value Function\nBrighter = higher value')
    plt.colorbar(im, ax=ax4, fraction=0.046)

    # Mark start and goal
    ax4.plot(0, 0, 'bs', markersize=15, label='Start')
    ax4.plot(4, 4, 'r*', markersize=15, label='Goal')
    ax4.legend(loc='upper right')

    # ============ Plot 5: Sample Efficiency ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Compare real environment steps needed
    target_reward = 5.0  # Threshold for "solved"

    results = {}

    for n_plan in [0, 10, 50]:
        env = GridWorldEnv(size=5)
        agent = DynaQ(env.n_states, env.n_actions, n_planning=n_plan)

        total_steps = 0
        rewards = []

        for episode in range(100):
            state = env.reset()
            episode_steps = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)

                agent.model.update(state, action, reward, next_state)
                agent.update_q(state, action, reward, next_state, done)

                for _ in range(n_plan):
                    agent.planning_step()

                state = next_state
                episode_steps += 1
                total_steps += 1

                if done or episode_steps > 100:
                    break

            rewards.append(agent.history['episode_rewards'][-1] if agent.history['episode_rewards'] else 0)

        results[n_plan] = total_steps

    x = np.arange(len(results))
    ax5.bar(x, list(results.values()), color=['gray', 'steelblue', 'coral'])
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'n={k}' for k in results.keys()])
    ax5.set_ylabel('Real Environment Steps')
    ax5.set_title('Sample Efficiency\nPlanning reduces real interactions')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Model-Based RL
    ══════════════════════════════

    THE KEY IDEA:
    Learn the world, then plan!

    f_θ(s, a) → s', r  (World Model)

    DYNA-Q ALGORITHM:
    ┌────────────────────────────┐
    │ 1. Act in real world       │
    │ 2. Update Q (real exp)     │
    │ 3. Update model            │
    │ 4. Planning:               │
    │    - Sample (s,a)          │
    │    - Simulate with model   │
    │    - Update Q (simulated)  │
    └────────────────────────────┘

    MODEL PREDICTIVE CONTROL:
    Sample action sequences
    Evaluate with model
    Execute best first action

    BENEFITS:
    ✓ Sample efficient
    ✓ Can plan ahead
    ✓ Model is reusable

    CHALLENGES:
    • Model errors compound
    • Learning good models is hard
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Model-Based RL — Learning World Models\n'
                 'Plan in imagination, act in reality',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for model-based RL."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Planning steps
    print("\n1. EFFECT OF PLANNING STEPS")
    print("-" * 40)

    for n_plan in [0, 5, 10, 20, 50]:
        env = GridWorldEnv(size=5)
        agent = DynaQ(env.n_states, env.n_actions, n_planning=n_plan)

        final_rewards = []
        for episode in range(50):
            agent.train_episode(env)
            if episode >= 40:
                final_rewards.append(agent.history['episode_rewards'][-1])

        print(f"n_planning={n_plan:<3}  avg_reward={np.mean(final_rewards):.2f}")

    print("→ More planning = better, but diminishing returns")

    # 2. Grid size (complexity)
    print("\n2. EFFECT OF ENVIRONMENT COMPLEXITY")
    print("-" * 40)

    for size in [3, 5, 7, 10]:
        env = GridWorldEnv(size=size)
        agent = DynaQ(env.n_states, env.n_actions, n_planning=10)

        final_rewards = []
        for episode in range(100):
            agent.train_episode(env)
            if episode >= 80:
                final_rewards.append(agent.history['episode_rewards'][-1])

        print(f"grid_size={size}  n_states={size*size:<4}  avg_reward={np.mean(final_rewards):.2f}")

    print("→ Larger state space = harder to learn model")

    # 3. Learning rate
    print("\n3. EFFECT OF LEARNING RATE")
    print("-" * 40)

    for alpha in [0.01, 0.1, 0.3, 0.5]:
        env = GridWorldEnv(size=5)
        agent = DynaQ(env.n_states, env.n_actions, alpha=alpha, n_planning=10)

        final_rewards = []
        for episode in range(50):
            agent.train_episode(env)
            if episode >= 40:
                final_rewards.append(agent.history['episode_rewards'][-1])

        print(f"α={alpha:.2f}  avg_reward={np.mean(final_rewards):.2f}")

    print("→ α=0.1-0.3 often works well")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Model-Based RL — Learning World Models")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_model_based_rl()
    save_path = '/Users/sid47/ML Algorithms/32_model_based_rl.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Model-Based RL: Learn a model, then plan with it
2. World Model: f(s,a) → s', r
3. Dyna-Q: Combine Q-learning + model-based planning
4. Planning with simulated experience speeds learning
5. More sample efficient than model-free
6. Challenge: Model errors compound over horizon
7. MPC: Plan at each step with learned model
    """)
