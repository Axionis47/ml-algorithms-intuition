"""
POLICY GRADIENT (REINFORCE) — Direct Policy Optimization

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of learning a value function and deriving a policy,
DIRECTLY optimize the policy!

THE POLICY GRADIENT THEOREM:
    grad J(theta) = E[ grad log pi(a|s) * R ]

WHAT THIS MEANS:
    - pi(a|s): probability of taking action a in state s
    - R: the return (cumulative reward)
    - grad log pi(a|s): direction to increase action probability

INTUITION:
    "Increase probability of actions that led to HIGH reward"
    "Decrease probability of actions that led to LOW reward"

===============================================================
WHY POLICY GRADIENT?
===============================================================

Value-based (Q-learning):
    - Learn Q(s,a), then derive policy: pi(s) = argmax Q(s,a)
    - Policy is always deterministic (except for exploration)
    - Can't handle continuous actions easily

Policy-based:
    - Learn pi(a|s) directly
    - Can be stochastic (natural exploration!)
    - Works with continuous actions
    - Can learn policies that value-based can't represent

===============================================================
THE VARIANCE PROBLEM
===============================================================

Policy gradients have HIGH VARIANCE:
    - Same trajectory can have wildly different returns
    - Gradient estimates are noisy
    - Training is unstable

SOLUTIONS:
1. BASELINE: Subtract baseline b(s) from return
   grad J = E[ grad log pi(a|s) * (R - b(s)) ]
   Doesn't change expectation, but reduces variance!

2. REWARD-TO-GO: Only use future rewards
   Instead of full return R, use G_t = sum_{t'>=t} r_{t'}
   Actions shouldn't be credited for past rewards

3. ADVANTAGE: Use A(s,a) = Q(s,a) - V(s)
   "How much better is this action than average?"
   This is Actor-Critic (next file)

===============================================================
INDUCTIVE BIAS
===============================================================

1. ON-POLICY: Must use current policy's trajectories
2. STOCHASTIC: Policy outputs probabilities (explores naturally)
3. PARAMETRIC: Policy is a function (e.g., neural network)
4. EPISODIC: Typically needs complete episodes for returns

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict


# =============================================================
# ENVIRONMENTS
# =============================================================

class CartPole:
    """
    Simplified CartPole environment.

    State: [cart_position, cart_velocity, pole_angle, pole_velocity]
    Actions: 0 = push left, 1 = push right

    Goal: Keep pole balanced (angle small) and cart centered.
    """

    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step

        # Thresholds for failure
        self.x_threshold = 2.4
        self.theta_threshold = 12 * np.pi / 180  # 12 degrees

        self.n_actions = 2
        self.state_dim = 4

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset to random initial state."""
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action, return (next_state, reward, done)."""
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        # Physics equations
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0/3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # Check termination
        done = (
            x < -self.x_threshold or
            x > self.x_threshold or
            theta < -self.theta_threshold or
            theta > self.theta_threshold or
            self.steps >= 200
        )

        # Reward: +1 for each step survived
        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done


class GridWorld:
    """Simple gridworld for policy gradient."""

    def __init__(self, rows: int = 4, cols: int = 4,
                 start: Tuple[int, int] = (3, 0),
                 goal: Tuple[int, int] = (0, 3)):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal

        self.n_actions = 4
        self.state_dim = 2  # (row, col)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.reset()

    def reset(self) -> np.ndarray:
        self.state = np.array(self.start, dtype=float)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        dr, dc = self.actions[action]
        r, c = int(self.state[0]), int(self.state[1])

        new_r = max(0, min(self.rows - 1, r + dr))
        new_c = max(0, min(self.cols - 1, c + dc))

        self.state = np.array([new_r, new_c], dtype=float)

        if (new_r, new_c) == self.goal:
            return self.state.copy(), 1.0, True

        return self.state.copy(), -0.01, False


# =============================================================
# POLICY NETWORKS
# =============================================================

class SoftmaxPolicy:
    """
    Softmax policy with linear features.

    pi(a|s) = exp(theta[a] @ phi(s)) / sum_b exp(theta[b] @ phi(s))

    For simple environments, we use state directly as features.
    """

    def __init__(self, state_dim: int, n_actions: int, feature_dim: int = None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim or state_dim

        # Policy parameters
        self.theta = np.random.randn(n_actions, self.feature_dim) * 0.01

    def features(self, state: np.ndarray) -> np.ndarray:
        """Extract features from state."""
        return state

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute pi(a|s) for all actions."""
        phi = self.features(state)
        logits = self.theta @ phi

        # Softmax with numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def sample_action(self, state: np.ndarray) -> int:
        """Sample action from policy."""
        probs = self.action_probs(state)
        return np.random.choice(self.n_actions, p=probs)

    def log_prob(self, state: np.ndarray, action: int) -> float:
        """Compute log pi(a|s)."""
        probs = self.action_probs(state)
        return np.log(probs[action] + 1e-10)

    def grad_log_prob(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute grad_theta log pi(a|s).

        For softmax: grad log pi(a|s) = phi(s) * (I[a=a'] - pi(a'|s))

        Returns gradient with same shape as theta.
        """
        phi = self.features(state)
        probs = self.action_probs(state)

        # One-hot for action
        one_hot = np.zeros(self.n_actions)
        one_hot[action] = 1.0

        # grad log pi(a|s) w.r.t. theta[a', :]
        grad = np.outer(one_hot - probs, phi)

        return grad


class NeuralPolicy:
    """
    Neural network policy for continuous state spaces.

    Single hidden layer MLP with softmax output.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 32):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.W1 = np.random.randn(hidden_dim, state_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(n_actions, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions)

        # For storing forward pass activations
        self.h = None
        self.probs = None

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass, return action probabilities."""
        # Hidden layer with ReLU
        self.z1 = self.W1 @ state + self.b1
        self.h = np.maximum(0, self.z1)

        # Output layer with softmax
        logits = self.W2 @ self.h + self.b2
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        self.probs = exp_logits / np.sum(exp_logits)

        return self.probs

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        return self.forward(state)

    def sample_action(self, state: np.ndarray) -> int:
        probs = self.forward(state)
        return np.random.choice(self.n_actions, p=probs)

    def log_prob(self, state: np.ndarray, action: int) -> float:
        probs = self.forward(state)
        return np.log(probs[action] + 1e-10)

    def backward(self, state: np.ndarray, action: int) -> Dict:
        """
        Compute gradients of log pi(a|s) w.r.t. all parameters.

        Uses stored activations from forward pass.
        """
        # Ensure forward pass was done
        if self.probs is None:
            self.forward(state)

        # Gradient of log softmax w.r.t. logits
        one_hot = np.zeros(self.n_actions)
        one_hot[action] = 1.0
        d_logits = one_hot - self.probs  # shape: (n_actions,)

        # Gradient w.r.t. W2, b2
        grad_W2 = np.outer(d_logits, self.h)
        grad_b2 = d_logits

        # Backprop through hidden layer
        d_h = self.W2.T @ d_logits  # shape: (hidden_dim,)
        d_z1 = d_h * (self.z1 > 0)  # ReLU gradient

        # Gradient w.r.t. W1, b1
        grad_W1 = np.outer(d_z1, state)
        grad_b1 = d_z1

        return {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2
        }

    def update(self, grads: Dict, lr: float):
        """Update parameters with gradients."""
        self.W1 += lr * grads['W1']
        self.b1 += lr * grads['b1']
        self.W2 += lr * grads['W2']
        self.b2 += lr * grads['b2']


# =============================================================
# REINFORCE ALGORITHM
# =============================================================

def reinforce(env, policy, n_episodes: int = 1000,
              lr: float = 0.01, gamma: float = 0.99,
              use_baseline: bool = False,
              baseline_lr: float = 0.01) -> Dict:
    """
    REINFORCE algorithm (Monte Carlo Policy Gradient).

    THE ALGORITHM:
        1. Collect trajectory using current policy
        2. Compute returns G_t for each step
        3. Update: theta += lr * grad log pi(a_t|s_t) * G_t

    WITH BASELINE:
        theta += lr * grad log pi(a_t|s_t) * (G_t - b(s_t))

    The baseline b(s) is typically learned as a value function.
    """
    episode_rewards = []
    episode_lengths = []
    policy_entropy = []

    # Baseline (simple linear)
    if use_baseline:
        baseline_w = np.zeros(env.state_dim)

    for ep in range(n_episodes):
        # Collect trajectory
        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Compute returns (reward-to-go)
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # Normalize returns (helps stability)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update policy
        if isinstance(policy, SoftmaxPolicy):
            # Accumulate gradients
            grad_accum = np.zeros_like(policy.theta)

            for t in range(T):
                # Baseline subtraction
                if use_baseline:
                    baseline_value = baseline_w @ states[t]
                    advantage = returns[t] - baseline_value

                    # Update baseline
                    baseline_w += baseline_lr * (returns[t] - baseline_value) * states[t]
                else:
                    advantage = returns[t]

                grad = policy.grad_log_prob(states[t], actions[t])
                grad_accum += grad * advantage

            # Apply update
            policy.theta += lr * grad_accum / T

        else:  # NeuralPolicy
            for t in range(T):
                if use_baseline:
                    baseline_value = baseline_w @ states[t]
                    advantage = returns[t] - baseline_value
                    baseline_w += baseline_lr * (returns[t] - baseline_value) * states[t]
                else:
                    advantage = returns[t]

                policy.forward(states[t])
                grads = policy.backward(states[t], actions[t])

                # Scale by advantage
                scaled_grads = {k: v * advantage for k, v in grads.items()}
                policy.update(scaled_grads, lr / T)

        # Track metrics
        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))

        # Track entropy
        state = env.reset()
        probs = policy.action_probs(state)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        policy_entropy.append(entropy)

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'entropy': policy_entropy
    }


# =============================================================
# VISUALIZATIONS
# =============================================================

def visualize_reinforce_cartpole():
    """REINFORCE on CartPole."""
    print("\n" + "="*60)
    print("REINFORCE ON CARTPOLE")
    print("="*60)

    env = CartPole()
    policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)

    results = reinforce(env, policy, n_episodes=500, lr=0.001, gamma=0.99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Smooth rewards
    window = 20
    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    axes[0].plot(smooth(results['rewards'], window), linewidth=2)
    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved threshold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('REINFORCE on CartPole\n(Solved = 195+ average)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(results['lengths'], window), linewidth=2, color='orange')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(smooth(results['entropy'], window), linewidth=2, color='green')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Policy Entropy')
    axes[2].set_title('Policy Entropy\n(Decreases as policy becomes deterministic)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    final_reward = np.mean(results['rewards'][-50:])
    print(f"Final average reward: {final_reward:.1f}")
    print("(Solved threshold: 195)")

    return fig


def visualize_baseline_effect():
    """Compare REINFORCE with and without baseline."""
    print("\n" + "="*60)
    print("ABLATION: BASELINE EFFECT")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_runs = 5
    n_episodes = 300

    all_no_baseline = []
    all_with_baseline = []

    for run in range(n_runs):
        env = CartPole()

        # Without baseline
        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=0.001,
                           use_baseline=False)
        all_no_baseline.append(results['rewards'])

        # With baseline
        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=0.001,
                           use_baseline=True, baseline_lr=0.01)
        all_with_baseline.append(results['rewards'])

    # Average and smooth
    no_baseline_mean = np.mean(all_no_baseline, axis=0)
    with_baseline_mean = np.mean(all_with_baseline, axis=0)
    no_baseline_std = np.std(all_no_baseline, axis=0)
    with_baseline_std = np.std(all_with_baseline, axis=0)

    window = 20
    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    # Learning curves
    x = np.arange(len(smooth(no_baseline_mean, window)))
    axes[0].plot(smooth(no_baseline_mean, window), label='Without Baseline',
                linewidth=2, color='red')
    axes[0].fill_between(x,
                        smooth(no_baseline_mean - no_baseline_std, window),
                        smooth(no_baseline_mean + no_baseline_std, window),
                        alpha=0.2, color='red')

    axes[0].plot(smooth(with_baseline_mean, window), label='With Baseline',
                linewidth=2, color='blue')
    axes[0].fill_between(x,
                        smooth(with_baseline_mean - with_baseline_std, window),
                        smooth(with_baseline_mean + with_baseline_std, window),
                        alpha=0.2, color='blue')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title(f'Effect of Baseline (avg over {n_runs} runs)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Variance comparison
    variance_no = np.var(all_no_baseline, axis=0)
    variance_with = np.var(all_with_baseline, axis=0)

    axes[1].plot(smooth(variance_no, window), label='Without Baseline',
                linewidth=2, color='red')
    axes[1].plot(smooth(variance_with, window), label='With Baseline',
                linewidth=2, color='blue')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Variance of Returns')
    axes[1].set_title('Gradient Variance\n(Baseline reduces variance)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nKey insight:")
    print("  - Baseline subtracts expected return from each update")
    print("  - Doesn't change expected gradient (still unbiased)")
    print("  - But REDUCES VARIANCE -> more stable learning")

    return fig


def visualize_learning_rate():
    """Effect of learning rate on REINFORCE."""
    print("\n" + "="*60)
    print("ABLATION: LEARNING RATE")
    print("="*60)

    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    n_episodes = 300

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))

    for lr, color in zip(learning_rates, colors):
        env = CartPole()
        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=lr)

        window = 20
        smoothed = np.convolve(results['rewards'], np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'lr={lr}', color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Effect of Learning Rate\n(Too small: slow, too large: unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nInsight:")
    print("  - lr too small: slow learning")
    print("  - lr too large: policy changes too fast, unstable")
    print("  - Need to tune carefully for each problem")

    return fig


def visualize_discount_factor():
    """Effect of discount factor on REINFORCE."""
    print("\n" + "="*60)
    print("ABLATION: DISCOUNT FACTOR")
    print("="*60)

    gammas = [0.9, 0.95, 0.99, 1.0]
    n_episodes = 300

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.plasma(np.linspace(0, 1, len(gammas)))

    for gamma, color in zip(gammas, colors):
        env = CartPole()
        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=0.001, gamma=gamma)

        window = 20
        smoothed = np.convolve(results['rewards'], np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'gamma={gamma}', color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Effect of Discount Factor gamma')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nInsight:")
    print("  - Lower gamma: focuses on immediate rewards")
    print("  - Higher gamma: considers longer-term consequences")
    print("  - gamma=1: no discounting (full returns)")

    return fig


def visualize_policy_evolution():
    """Visualize how policy changes during training."""
    print("\n" + "="*60)
    print("POLICY EVOLUTION")
    print("="*60)

    env = GridWorld(rows=4, cols=4)
    policy = SoftmaxPolicy(env.state_dim, env.n_actions)

    # Collect policy snapshots
    snapshots = {}
    checkpoints = [0, 50, 200, 500]

    for ep in range(501):
        if ep in checkpoints:
            # Save policy for all states
            policy_grid = np.zeros((env.rows, env.cols, env.n_actions))
            for r in range(env.rows):
                for c in range(env.cols):
                    state = np.array([r, c], dtype=float)
                    policy_grid[r, c] = policy.action_probs(state)
            snapshots[ep] = policy_grid

        if ep >= 500:
            break

        # Run episode and update
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False

        while not done and len(states) < 100:
            action = policy.sample_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Compute returns
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + 0.99 * G
            returns[t] = G

        if T > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update
        for t in range(T):
            grad = policy.grad_log_prob(states[t], actions[t])
            policy.theta += 0.1 * grad * returns[t] / T

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    action_symbols = ['^', '>', 'v', '<']  # up, right, down, left

    for ax, ep in zip(axes, checkpoints):
        policy_grid = snapshots[ep]

        # Background
        ax.imshow(np.zeros((env.rows, env.cols)), cmap='Greys', vmin=0, vmax=1)

        # Draw policy arrows
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) == env.goal:
                    ax.scatter([c], [r], marker='*', s=300, c='gold',
                              edgecolors='black', zorder=5)
                    continue

                probs = policy_grid[r, c]
                best_action = np.argmax(probs)

                # Draw arrow with opacity based on probability
                for a in range(4):
                    if probs[a] > 0.1:  # Only show significant actions
                        dx_list = [0, 0.3, 0, -0.3]
                        dy_list = [-0.3, 0, 0.3, 0]
                        ax.arrow(c, r, dx_list[a] * probs[a],
                                dy_list[a] * probs[a],
                                head_width=0.1, head_length=0.05,
                                fc='blue', ec='blue', alpha=probs[a])

        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(env.rows - 0.5, -0.5)
        ax.set_title(f'Episode {ep}')
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.grid(True, alpha=0.3)

    plt.suptitle('Policy Evolution During Training\n'
                 '(Arrow size = action probability)', fontsize=12)
    plt.tight_layout()

    print("\nKey insight: Policy starts random, becomes deterministic")
    print("  - Early: uniform action probabilities (high entropy)")
    print("  - Later: peaked on best actions (low entropy)")

    return fig


def visualize_variance_problem():
    """Demonstrate the high variance problem of policy gradients."""
    print("\n" + "="*60)
    print("THE VARIANCE PROBLEM")
    print("="*60)

    # Run same policy multiple times, show variance in gradient estimates
    env = CartPole()
    policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)

    # Collect multiple trajectories and their returns
    n_trajectories = 100
    trajectory_returns = []

    for _ in range(n_trajectories):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy.sample_action(state)
            state, reward, done = env.step(action)
            total_reward += reward

        trajectory_returns.append(total_reward)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution of returns
    axes[0].hist(trajectory_returns, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=np.mean(trajectory_returns), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(trajectory_returns):.1f}')
    axes[0].set_xlabel('Trajectory Return')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Distribution of Returns\n(Std: {np.std(trajectory_returns):.1f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Compare variance with/without baseline over training
    n_runs = 10
    n_episodes = 200

    no_baseline_rewards = []
    with_baseline_rewards = []

    for _ in range(n_runs):
        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=0.001,
                           use_baseline=False)
        no_baseline_rewards.append(results['rewards'])

        policy = NeuralPolicy(env.state_dim, env.n_actions, hidden_dim=32)
        results = reinforce(env, policy, n_episodes=n_episodes, lr=0.001,
                           use_baseline=True)
        with_baseline_rewards.append(results['rewards'])

    # Plot variance across runs
    no_baseline_var = np.var(no_baseline_rewards, axis=0)
    with_baseline_var = np.var(with_baseline_rewards, axis=0)

    window = 10
    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    axes[1].plot(smooth(no_baseline_var, window), label='Without Baseline',
                linewidth=2, color='red')
    axes[1].plot(smooth(with_baseline_var, window), label='With Baseline',
                linewidth=2, color='blue')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Variance Across Runs')
    axes[1].set_title('Training Variance\n(High variance = unreliable learning)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nThe Variance Problem:")
    print("  - Same policy can get wildly different returns")
    print("  - Gradient estimates are noisy")
    print("  - Baseline helps but doesn't eliminate variance")
    print("  - Actor-Critic (next file) helps more")

    return fig


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print("="*70)
    print(" " * 15 + "POLICY GRADIENT (REINFORCE)")
    print(" " * 12 + "Direct Policy Optimization")
    print("="*70)

    print("""
THE POLICY GRADIENT THEOREM:

    grad J(theta) = E[ grad log pi(a|s) * R ]

WHAT THIS SAYS:
    "Increase probability of actions that led to high reward"

REINFORCE ALGORITHM:
    1. Collect trajectory (s0, a0, r0, s1, a1, r1, ...)
    2. Compute returns G_t = sum of future rewards
    3. Update: theta += lr * grad log pi(a_t|s_t) * G_t

THE VARIANCE PROBLEM:
    - Same trajectory can have different returns (stochastic)
    - Gradient estimates are noisy
    - Solution: subtract baseline b(s) that doesn't change expectation

WITH BASELINE:
    theta += lr * grad log pi(a|s) * (G - b(s))

    Common baselines:
    - Constant (average return)
    - Learned value function V(s) -> Actor-Critic
    """)

    figs = []

    # Main experiments
    figs.append(('cartpole', visualize_reinforce_cartpole()))
    figs.append(('baseline', visualize_baseline_effect()))
    figs.append(('lr', visualize_learning_rate()))
    figs.append(('gamma', visualize_discount_factor()))
    figs.append(('evolution', visualize_policy_evolution()))
    figs.append(('variance', visualize_variance_problem()))

    # Save figures
    for name, fig in figs:
        save_path = f'/Users/sid47/ML Algorithms/27_policy_gradient_{name}.png'
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # Summary
    print("\n" + "="*70)
    print(" " * 20 + "KEY TAKEAWAYS")
    print("="*70)
    print("""
1. POLICY GRADIENT = DIRECT OPTIMIZATION
   Optimize pi(a|s) directly, not through Q-values
   Natural for stochastic policies and continuous actions

2. THE POLICY GRADIENT THEOREM
   grad J = E[grad log pi(a|s) * R]
   "Credit good actions, blame bad actions"

3. HIGH VARIANCE IS THE ENEMY
   Same policy -> different returns -> noisy gradients
   Baseline reduces variance without adding bias

4. ON-POLICY LIMITATION
   Must use current policy's data
   Can't reuse old experience (unlike Q-learning)

5. NEXT: ACTOR-CRITIC
   Use learned V(s) as baseline
   Reduces variance further
   Enables bootstrapping (TD learning)
    """)
