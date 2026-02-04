"""
Q-LEARNING — Value Bootstrapping

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Learn Q(s,a) = expected return from taking action a in state s.

THE Q-LEARNING UPDATE (Bellman backup):
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
                               |________TD target_________|
                                         ^
                              "bootstrapped" estimate

WHAT THIS MEANS:
    - r: immediate reward you got
    - gamma * max Q(s',a'): best future value (discounted)
    - Together: estimate of true Q(s,a)
    - Update Q toward this estimate

This is TEMPORAL DIFFERENCE learning: use your own estimates
to improve your estimates (bootstrap).

===============================================================
THE KEY INSIGHT: BELLMAN EQUATION
===============================================================

The optimal Q* satisfies:
    Q*(s,a) = E[r + gamma * max_a' Q*(s',a')]

"The value of (s,a) equals immediate reward plus
 the discounted value of the best next action."

Q-learning iterates toward this fixed point.

===============================================================
OFF-POLICY vs ON-POLICY
===============================================================

Q-LEARNING is OFF-POLICY:
    - Learns Q* (optimal policy's values)
    - Can learn from ANY data (random exploration, demos, etc.)
    - Uses max_a' Q(s',a') regardless of what you actually do next

SARSA is ON-POLICY:
    - Learns Q^pi (current policy's values)
    - Update uses actual next action: r + gamma * Q(s',a')
    - Must use epsilon-greedy for both behavior AND learning

===============================================================
INDUCTIVE BIAS
===============================================================

1. MARKOV: State is sufficient (no memory needed)
2. STATIONARY: Environment doesn't change
3. TABULAR: Can enumerate all (s,a) pairs
4. INFINITE DATA: Will visit every (s,a) infinitely often

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from collections import defaultdict


# =============================================================
# GRIDWORLD ENVIRONMENT
# =============================================================

class GridWorld:
    """
    Simple gridworld for demonstrating value learning.

    Actions: 0=up, 1=right, 2=down, 3=left
    """

    def __init__(self, rows: int = 4, cols: int = 4,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (0, 3),
                 walls: List[Tuple[int, int]] = None,
                 step_reward: float = -0.04):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = walls or [(1, 1)]
        self.step_reward = step_reward

        # Actions
        self.n_actions = 4
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        self.action_names = ['up', 'right', 'down', 'left']

        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Reset to start state."""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take action, return (next_state, reward, done)."""
        if self.state == self.goal:
            return self.state, 0, True

        # Compute next position
        dr, dc = self.actions[action]
        next_r = self.state[0] + dr
        next_c = self.state[1] + dc

        # Check bounds and walls
        if (0 <= next_r < self.rows and
            0 <= next_c < self.cols and
            (next_r, next_c) not in self.walls):
            self.state = (next_r, next_c)

        # Check goal
        if self.state == self.goal:
            return self.state, 1.0, True  # Goal reward

        return self.state, self.step_reward, False

    def get_all_states(self) -> List[Tuple[int, int]]:
        """Return all valid states."""
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    states.append((r, c))
        return states


class CliffWalk:
    """
    Cliff walking environment.

    S = Start, G = Goal, C = Cliff (fall = -100, reset to start)

    This shows the difference between Q-learning and SARSA:
    - Q-learning finds optimal (risky) path along cliff
    - SARSA finds safe path (accounts for exploration mistakes)
    """

    def __init__(self, rows: int = 4, cols: int = 12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)

        # Cliff is bottom row except start and goal
        self.cliff = [(rows - 1, c) for c in range(1, cols - 1)]

        self.n_actions = 4
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['up', 'right', 'down', 'left']

        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        if self.state == self.goal:
            return self.state, 0, True

        dr, dc = self.actions[action]
        next_r = max(0, min(self.rows - 1, self.state[0] + dr))
        next_c = max(0, min(self.cols - 1, self.state[1] + dc))

        self.state = (next_r, next_c)

        # Check cliff
        if self.state in self.cliff:
            self.state = self.start  # Reset to start
            return self.state, -100, False

        # Check goal
        if self.state == self.goal:
            return self.state, 0, True

        return self.state, -1, False

    def get_all_states(self) -> List[Tuple[int, int]]:
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                states.append((r, c))
        return states


# =============================================================
# Q-LEARNING AGENT
# =============================================================

class QLearningAgent:
    """
    Tabular Q-learning agent.

    THE Q-LEARNING UPDATE:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    OFF-POLICY: Uses max over next actions (greedy w.r.t. Q)
    regardless of what action we actually take.
    """

    def __init__(self, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action: int, reward: float,
               next_state, done: bool):
        """Q-learning update."""
        # TD target: r + gamma * max_a' Q(s',a')
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = td_target - self.Q[state][action]

        # Update
        self.Q[state][action] += self.alpha * td_error

        return td_error


class SARSAAgent:
    """
    SARSA: State-Action-Reward-State-Action (On-policy TD).

    THE SARSA UPDATE:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
                                          ^
                              actual next action (not max!)

    ON-POLICY: Uses the action we actually take next.
    Learns the value of the CURRENT policy (including exploration).
    """

    def __init__(self, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action: int, reward: float,
               next_state, next_action: int, done: bool):
        """SARSA update - uses actual next action."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state][next_action]

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        return td_error


# =============================================================
# TRAINING FUNCTIONS
# =============================================================

def train_q_learning(env, agent: QLearningAgent, n_episodes: int,
                     max_steps: int = 100) -> Dict:
    """Train Q-learning agent."""
    episode_rewards = []
    episode_lengths = []
    q_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Save Q-table snapshot
        if ep % (n_episodes // 10) == 0:
            q_history.append({s: agent.Q[s].copy() for s in agent.Q})

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'q_history': q_history
    }


def train_sarsa(env, agent: SARSAAgent, n_episodes: int,
                max_steps: int = 100) -> Dict:
    """Train SARSA agent."""
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            total_reward += reward
            steps += 1
            state = next_state
            action = next_action

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths
    }


# =============================================================
# VISUALIZATION FUNCTIONS
# =============================================================

def visualize_q_values(env, agent, title: str = "Q-Values"):
    """Visualize Q-values and derived policy on gridworld."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get value function V(s) = max_a Q(s,a)
    V = np.zeros((env.rows, env.cols))
    policy = np.zeros((env.rows, env.cols), dtype=int)

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if hasattr(env, 'walls') and state in env.walls:
                V[r, c] = np.nan
            elif hasattr(env, 'cliff') and state in env.cliff:
                V[r, c] = -100
            else:
                V[r, c] = np.max(agent.Q[state])
                policy[r, c] = np.argmax(agent.Q[state])

    # Plot value function
    im = axes[0].imshow(V, cmap='RdYlGn', origin='upper')
    axes[0].set_title(f'{title}\nValue Function V(s) = max_a Q(s,a)')
    plt.colorbar(im, ax=axes[0])

    # Add value text
    for r in range(env.rows):
        for c in range(env.cols):
            if not np.isnan(V[r, c]):
                axes[0].text(c, r, f'{V[r, c]:.2f}', ha='center', va='center',
                           fontsize=8, color='black')

    # Mark special states
    if hasattr(env, 'goal'):
        axes[0].scatter([env.goal[1]], [env.goal[0]], marker='*',
                       s=300, c='gold', edgecolors='black', zorder=5)
    if hasattr(env, 'start'):
        axes[0].scatter([env.start[1]], [env.start[0]], marker='s',
                       s=200, c='blue', edgecolors='black', zorder=5)

    # Plot policy
    axes[1].imshow(V, cmap='RdYlGn', origin='upper', alpha=0.3)
    axes[1].set_title(f'{title}\nPolicy pi(s) = argmax_a Q(s,a)')

    # Draw arrows for policy
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if hasattr(env, 'walls') and state in env.walls:
                continue
            if hasattr(env, 'cliff') and state in env.cliff:
                axes[1].text(c, r, 'C', ha='center', va='center',
                           fontsize=10, color='red', fontweight='bold')
                continue
            if state == env.goal:
                continue

            # Arrow directions
            a = policy[r, c]
            arrows = [(0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0)]  # up, right, down, left
            dx, dy = arrows[a]
            axes[1].arrow(c, r, dx, dy, head_width=0.15, head_length=0.1,
                         fc='black', ec='black')

    # Mark special states
    if hasattr(env, 'goal'):
        axes[1].scatter([env.goal[1]], [env.goal[0]], marker='*',
                       s=300, c='gold', edgecolors='black', zorder=5)
    if hasattr(env, 'start'):
        axes[1].scatter([env.start[1]], [env.start[0]], marker='s',
                       s=200, c='blue', edgecolors='black', zorder=5)

    for ax in axes:
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_value_propagation(env, n_episodes: int = 500):
    """Show Q-value propagation from goal."""
    print("\n" + "="*60)
    print("VALUE PROPAGATION VISUALIZATION")
    print("="*60)

    agent = QLearningAgent(env.n_actions, alpha=0.5, gamma=0.95, epsilon=0.2)

    # Collect Q-values at different training stages
    checkpoints = [0, 10, 50, 100, 200, 500]
    q_snapshots = {}

    for ep in range(n_episodes + 1):
        if ep in checkpoints:
            V = np.zeros((env.rows, env.cols))
            for r in range(env.rows):
                for c in range(env.cols):
                    state = (r, c)
                    if hasattr(env, 'walls') and state in env.walls:
                        V[r, c] = np.nan
                    else:
                        V[r, c] = np.max(agent.Q[state])
            q_snapshots[ep] = V.copy()

        if ep < n_episodes:
            # Run one episode
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

    # Plot snapshots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, ep in zip(axes, checkpoints):
        V = q_snapshots[ep]
        im = ax.imshow(V, cmap='RdYlGn', origin='upper', vmin=-1, vmax=1)
        ax.set_title(f'Episode {ep}')

        # Add values
        for r in range(env.rows):
            for c in range(env.cols):
                if not np.isnan(V[r, c]):
                    ax.text(c, r, f'{V[r, c]:.2f}', ha='center', va='center',
                           fontsize=7, color='black')

        # Mark goal and start
        ax.scatter([env.goal[1]], [env.goal[0]], marker='*',
                  s=200, c='gold', edgecolors='black', zorder=5)
        ax.scatter([env.start[1]], [env.start[0]], marker='s',
                  s=150, c='blue', edgecolors='black', zorder=5)

        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))

    plt.suptitle('Q-Learning: Value Propagation Over Training\n'
                 '(Values propagate backwards from goal)', fontsize=12)
    plt.tight_layout()

    print("\nKey insight: Values propagate BACKWARDS from goal")
    print("  - Initially all Q(s,a) = 0")
    print("  - First, states adjacent to goal get positive values")
    print("  - Then values propagate one step back each episode")
    print("  - This is Bellman backup in action!")

    return fig


def visualize_learning_curves(n_episodes: int = 500):
    """Compare learning curves of Q-learning and SARSA."""
    print("\n" + "="*60)
    print("LEARNING CURVES")
    print("="*60)

    env = GridWorld(rows=4, cols=4, walls=[(1, 1)])

    q_agent = QLearningAgent(env.n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    sarsa_agent = SARSAAgent(env.n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)

    q_results = train_q_learning(env, q_agent, n_episodes)

    env.reset()
    sarsa_results = train_sarsa(env, sarsa_agent, n_episodes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Smooth rewards
    window = 20

    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    axes[0].plot(smooth(q_results['rewards'], window), label='Q-learning', linewidth=2)
    axes[0].plot(smooth(sarsa_results['rewards'], window), label='SARSA', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward (smoothed)')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(q_results['lengths'], window), label='Q-learning', linewidth=2)
    axes[1].plot(smooth(sarsa_results['lengths'], window), label='SARSA', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (smoothed)')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Q-Learning vs SARSA on GridWorld', fontsize=12)
    plt.tight_layout()

    return fig


def visualize_cliff_comparison(n_episodes: int = 500, n_runs: int = 20):
    """Compare Q-learning and SARSA on CliffWalk."""
    print("\n" + "="*60)
    print("CLIFF WALK: Q-LEARNING vs SARSA")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_q_rewards = []
    all_sarsa_rewards = []

    # Multiple runs for statistical significance
    for _ in range(n_runs):
        env = CliffWalk()
        q_agent = QLearningAgent(env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        q_results = train_q_learning(env, q_agent, n_episodes, max_steps=200)
        all_q_rewards.append(q_results['rewards'])

        env = CliffWalk()
        sarsa_agent = SARSAAgent(env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
        sarsa_results = train_sarsa(env, sarsa_agent, n_episodes, max_steps=200)
        all_sarsa_rewards.append(sarsa_results['rewards'])

    # Average over runs
    q_mean = np.mean(all_q_rewards, axis=0)
    sarsa_mean = np.mean(all_sarsa_rewards, axis=0)

    # Smooth
    window = 20
    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    axes[0, 0].plot(smooth(q_mean, window), label='Q-learning', linewidth=2, color='blue')
    axes[0, 0].plot(smooth(sarsa_mean, window), label='SARSA', linewidth=2, color='orange')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward per Episode')
    axes[0, 0].set_title(f'CliffWalk Learning Curves (avg over {n_runs} runs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-150, 0)

    # Final performance comparison
    final_q = np.mean([np.mean(r[-50:]) for r in all_q_rewards])
    final_sarsa = np.mean([np.mean(r[-50:]) for r in all_sarsa_rewards])

    axes[0, 1].bar(['Q-learning', 'SARSA'], [final_q, final_sarsa],
                  color=['blue', 'orange'])
    axes[0, 1].set_ylabel('Average Reward (last 50 episodes)')
    axes[0, 1].set_title('Final Performance')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Visualize learned policies
    env = CliffWalk()
    q_agent = QLearningAgent(env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
    train_q_learning(env, q_agent, n_episodes, max_steps=200)

    env = CliffWalk()
    sarsa_agent = SARSAAgent(env.n_actions, alpha=0.5, gamma=1.0, epsilon=0.1)
    train_sarsa(env, sarsa_agent, n_episodes, max_steps=200)

    # Plot Q-learning path
    ax = axes[1, 0]
    env = CliffWalk()

    # Background
    grid = np.zeros((env.rows, env.cols))
    for r, c in env.cliff:
        grid[r, c] = -1
    ax.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1, origin='upper')

    # Draw policy arrows
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.cliff or state == env.goal:
                continue
            a = np.argmax(q_agent.Q[state])
            arrows = [(0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0)]
            dx, dy = arrows[a]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.1,
                    fc='blue', ec='blue')

    ax.scatter([env.start[1]], [env.start[0]], marker='s', s=200, c='blue',
              edgecolors='black', zorder=5, label='Start')
    ax.scatter([env.goal[1]], [env.goal[0]], marker='*', s=300, c='gold',
              edgecolors='black', zorder=5, label='Goal')
    ax.set_title('Q-learning Policy\n(Optimal but risky path along cliff)')
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))

    # Plot SARSA path
    ax = axes[1, 1]
    ax.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1, origin='upper')

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.cliff or state == env.goal:
                continue
            a = np.argmax(sarsa_agent.Q[state])
            arrows = [(0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0)]
            dx, dy = arrows[a]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.1,
                    fc='orange', ec='orange')

    ax.scatter([env.start[1]], [env.start[0]], marker='s', s=200, c='blue',
              edgecolors='black', zorder=5)
    ax.scatter([env.goal[1]], [env.goal[0]], marker='*', s=300, c='gold',
              edgecolors='black', zorder=5)
    ax.set_title('SARSA Policy\n(Safe path away from cliff)')
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))

    plt.suptitle('CliffWalk: Off-Policy (Q-learning) vs On-Policy (SARSA)', fontsize=12)
    plt.tight_layout()

    print("\nKey insight: Q-learning vs SARSA on cliff walk")
    print("  - Q-learning: learns OPTIMAL policy (shortest path)")
    print("    -> But path is along cliff edge (risky during exploration)")
    print("    -> During training with epsilon-greedy, falls off often")
    print("  - SARSA: learns to AVOID exploration mistakes")
    print("    -> Takes longer safe path away from cliff")
    print("    -> Better training performance (fewer falls)")

    return fig


# =============================================================
# ABLATION EXPERIMENTS
# =============================================================

def ablation_learning_rate():
    """Ablation: Effect of learning rate alpha."""
    print("\n" + "="*60)
    print("ABLATION: LEARNING RATE alpha")
    print("="*60)

    env = GridWorld(rows=4, cols=4, walls=[(1, 1)])
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
    n_episodes = 300
    n_runs = 10

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for alpha, color in zip(alphas, colors):
        all_rewards = []
        for _ in range(n_runs):
            agent = QLearningAgent(env.n_actions, alpha=alpha, gamma=0.99, epsilon=0.1)
            results = train_q_learning(env, agent, n_episodes)
            all_rewards.append(results['rewards'])

        mean_rewards = np.mean(all_rewards, axis=0)
        window = 10
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'alpha={alpha}', color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Effect of Learning Rate alpha\n(alpha too low: slow, alpha too high: unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nInsight:")
    print("  - alpha too low: slow learning, takes many episodes")
    print("  - alpha too high: unstable, values oscillate")
    print("  - Sweet spot depends on problem")

    return fig


def ablation_discount_factor():
    """Ablation: Effect of discount factor gamma."""
    print("\n" + "="*60)
    print("ABLATION: DISCOUNT FACTOR gamma")
    print("="*60)

    # Use larger gridworld
    env = GridWorld(rows=5, cols=5, walls=[(1, 1), (2, 2), (3, 3)],
                   goal=(4, 4), start=(0, 0))

    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
    n_episodes = 500

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, gamma in enumerate(gammas):
        agent = QLearningAgent(env.n_actions, alpha=0.3, gamma=gamma, epsilon=0.1)
        train_q_learning(env, agent, n_episodes)

        # Plot value function
        V = np.zeros((env.rows, env.cols))
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                if state in env.walls:
                    V[r, c] = np.nan
                else:
                    V[r, c] = np.max(agent.Q[state])

        ax = axes[i]
        im = ax.imshow(V, cmap='RdYlGn', origin='upper')
        ax.set_title(f'gamma = {gamma}')
        plt.colorbar(im, ax=ax)

        # Draw policy arrows
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                if state in env.walls or state == env.goal:
                    continue
                a = np.argmax(agent.Q[state])
                arrows = [(0, -0.25), (0.25, 0), (0, 0.25), (-0.25, 0)]
                dx, dy = arrows[a]
                ax.arrow(c, r, dx, dy, head_width=0.1, head_length=0.08,
                        fc='black', ec='black')

        ax.scatter([env.goal[1]], [env.goal[0]], marker='*', s=200, c='gold',
                  edgecolors='black', zorder=5)

        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))

    # Analysis in last subplot
    axes[5].axis('off')
    axes[5].text(0.1, 0.8, "DISCOUNT FACTOR gamma:", fontsize=12, fontweight='bold',
                transform=axes[5].transAxes)
    axes[5].text(0.1, 0.6, "gamma = 0: Only immediate reward (myopic)",
                transform=axes[5].transAxes)
    axes[5].text(0.1, 0.5, "  -> Doesn't plan ahead, random-looking policy",
                transform=axes[5].transAxes)
    axes[5].text(0.1, 0.35, "gamma = 0.9-0.99: Future matters (farsighted)",
                transform=axes[5].transAxes)
    axes[5].text(0.1, 0.25, "  -> Plans path to goal, clear value gradient",
                transform=axes[5].transAxes)
    axes[5].text(0.1, 0.1, "gamma = 1: Infinite horizon (can be unstable)",
                transform=axes[5].transAxes)

    plt.suptitle('Effect of Discount Factor gamma on Value Function & Policy', fontsize=12)
    plt.tight_layout()

    print("\nInsight:")
    print("  - gamma=0: Only cares about immediate reward (myopic)")
    print("    -> Can't learn paths longer than 1 step!")
    print("  - gamma->1: Cares equally about all future rewards")
    print("    -> Can plan long sequences, but slower convergence")
    print("  - Sweet spot: gamma=0.9-0.99 for most problems")

    return fig


def ablation_exploration():
    """Ablation: Effect of exploration rate epsilon."""
    print("\n" + "="*60)
    print("ABLATION: EXPLORATION RATE epsilon")
    print("="*60)

    env = GridWorld(rows=4, cols=4, walls=[(1, 1)])
    epsilons = [0.0, 0.01, 0.1, 0.3, 0.5]
    n_episodes = 300
    n_runs = 10

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(epsilons)))

    final_performance = []

    for epsilon, color in zip(epsilons, colors):
        all_rewards = []
        for _ in range(n_runs):
            agent = QLearningAgent(env.n_actions, alpha=0.1, gamma=0.99, epsilon=epsilon)
            results = train_q_learning(env, agent, n_episodes)
            all_rewards.append(results['rewards'])

        mean_rewards = np.mean(all_rewards, axis=0)
        window = 10
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=f'epsilon={epsilon}', color=color, linewidth=2)

        # Final performance (with greedy policy)
        final_rewards = []
        for _ in range(20):
            agent = QLearningAgent(env.n_actions, alpha=0.1, gamma=0.99, epsilon=epsilon)
            train_q_learning(env, agent, n_episodes)

            # Test with greedy policy
            state = env.reset()
            total_reward = 0
            for _ in range(50):
                action = agent.select_action(state, greedy=True)
                state, reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            final_rewards.append(total_reward)
        final_performance.append(np.mean(final_rewards))

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Training Reward (smoothed)')
    axes[0].set_title('Training Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(epsilons)), final_performance, color=colors)
    axes[1].set_xticks(range(len(epsilons)))
    axes[1].set_xticklabels([f'eps={e}' for e in epsilons])
    axes[1].set_ylabel('Test Reward (greedy policy)')
    axes[1].set_title('Final Policy Quality')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Effect of Exploration Rate epsilon', fontsize=12)
    plt.tight_layout()

    print("\nInsight:")
    print("  - epsilon=0: No exploration -> may never find goal!")
    print("  - epsilon too high: Wanders randomly, slow learning")
    print("  - Training with exploration, TEST with greedy")

    return fig


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print("="*70)
    print(" " * 20 + "Q-LEARNING")
    print(" " * 15 + "Value Bootstrapping")
    print("="*70)

    print("""
THE Q-LEARNING UPDATE:

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
                               |_______TD target_______|

WHAT THIS SAYS:
    "Adjust Q toward observed reward + best future value"

THE BELLMAN INSIGHT:
    Q*(s,a) = r + gamma * max Q*(s',a')

    "Value of (state, action) = immediate reward
     + discounted value of best next action"

    Q-learning iterates toward this fixed point.

KEY PROPERTIES:
    - OFF-POLICY: Learns optimal Q regardless of behavior
    - BOOTSTRAPS: Uses its own estimates as targets
    - TABULAR: Needs to visit every (s,a) pair
    """)

    figs = []

    # GridWorld visualization
    print("\n" + "="*60)
    print("GRIDWORLD TRAINING")
    print("="*60)

    env = GridWorld(rows=4, cols=4, walls=[(1, 1)])
    agent = QLearningAgent(env.n_actions, alpha=0.3, gamma=0.99, epsilon=0.1)

    print("Training Q-learning agent...")
    results = train_q_learning(env, agent, n_episodes=500)
    print(f"Final average reward: {np.mean(results['rewards'][-50:]):.2f}")

    figs.append(('q_values', visualize_q_values(env, agent, "Q-Learning on GridWorld")))
    figs.append(('value_prop', visualize_value_propagation(GridWorld(rows=4, cols=4, walls=[(1, 1)]))))
    figs.append(('learning', visualize_learning_curves()))
    figs.append(('cliff', visualize_cliff_comparison()))

    # Ablations
    print("\n" + "="*70)
    print(" " * 20 + "ABLATION EXPERIMENTS")
    print("="*70)

    figs.append(('abl_alpha', ablation_learning_rate()))
    figs.append(('abl_gamma', ablation_discount_factor()))
    figs.append(('abl_epsilon', ablation_exploration()))

    # Save figures
    for name, fig in figs:
        save_path = f'/Users/sid47/ML Algorithms/26_q_learning_{name}.png'
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # Summary
    print("\n" + "="*70)
    print(" " * 20 + "KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Q-LEARNING = BELLMAN BACKUP
   Update Q toward r + gamma * max Q(s',a')
   Iterates toward optimal Q*

2. OFF-POLICY vs ON-POLICY
   Q-learning: learns Q* (optimal)
   SARSA: learns Q^pi (current policy)
   Cliff example shows the difference!

3. EXPLORATION IS STILL ESSENTIAL
   Need epsilon-greedy or similar
   Balance learning optimal Q vs safe behavior

4. DISCOUNT gamma = TIME HORIZON
   gamma=0: myopic (immediate reward only)
   gamma->1: farsighted (cares about future)

5. LIMITATIONS: TABULAR
   Must visit every (s,a) pair
   Doesn't scale to large state spaces
   -> Next: DQN (neural network Q-function)
    """)
