"""
DQN — Deep Q-Network
=====================

Paradigm: VALUE FUNCTION APPROXIMATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Replace Q-table with a NEURAL NETWORK: Q_θ(s,a)

WHY?
- Tabular Q-learning: must visit EVERY (s,a) pair
- Can't handle continuous/large state spaces (images!)
- No generalization to similar states

DQN: Learn to GENERALIZE across states!

THE LOSS FUNCTION:
    L(θ) = E[(r + γ × max_a' Q_θ'(s',a') - Q_θ(s,a))²]
                  \_____target network____/

This is just MSE between prediction and TD target.

===============================================================
THE THREE DQN TRICKS (all essential!)
===============================================================

1. EXPERIENCE REPLAY
   - Store transitions (s, a, r, s', done) in buffer
   - Sample RANDOM batches for training
   - WHY: Breaks correlation between consecutive samples
          Reuses data (sample efficient)
          Stabilizes learning

2. TARGET NETWORK
   - Separate network θ' for computing TD target
   - Update θ' slowly (copy from θ periodically)
   - WHY: Moving target problem!
          Without this: chasing a moving target → unstable

3. FRAME STACKING (for Atari)
   - Stack last k frames as state
   - WHY: Single frame is not Markov (velocity hidden)
          Need temporal context

===============================================================
THE MOVING TARGET PROBLEM
===============================================================

In supervised learning:
    Target y is FIXED
    Network learns to match y

In Q-learning without target network:
    Target = r + γ max Q_θ(s', a')
    Target CHANGES as θ changes!
    Like chasing your own shadow

Solution: Target network θ' updates slowly
    Target = r + γ max Q_θ'(s', a')
    θ' is frozen for many steps
    Gives stable target to learn toward

===============================================================
DOUBLE DQN
===============================================================

Problem: Q-learning OVERESTIMATES values
    max_a Q(s,a) is biased upward (max of noisy estimates)

Double DQN solution:
    - Use θ to SELECT action: a* = argmax_a Q_θ(s',a)
    - Use θ' to EVALUATE: Q_θ'(s', a*)

Decouples selection from evaluation → less overestimation

===============================================================
INDUCTIVE BIAS
===============================================================

1. FUNCTION APPROXIMATION
   - Assumes similar states have similar values
   - Network architecture encodes assumptions (CNN for images)

2. TEMPORAL DIFFERENCE
   - Bootstrap from own estimates
   - More sample efficient than Monte Carlo

3. OFF-POLICY
   - Can learn from any experience
   - Enables experience replay

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
rl_fund = import_module('25_rl_fundamentals')
GridWorld = rl_fund.GridWorld


class ReplayBuffer:
    """
    EXPERIENCE REPLAY BUFFER

    THE KEY INSIGHT:
    Store transitions, sample randomly for training.

    Benefits:
    1. Breaks correlation (i.i.d. samples)
    2. Reuses data (sample efficient)
    3. Smooths learning (averages over many experiences)
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """
    Simple neural network for Q-function approximation.

    Q_θ(s) → [Q(s,a_1), Q(s,a_2), ..., Q(s,a_n)]

    Architecture: state → hidden → hidden → Q-values
    """

    def __init__(self, state_dim, n_actions, hidden_dims=[64, 64]):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Build layers
        dims = [state_dim] + hidden_dims + [n_actions]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

        self.cache = {}

    def forward(self, states):
        """
        Forward pass.

        states: (batch_size, state_dim)
        returns: Q-values (batch_size, n_actions)
        """
        h = states
        activations = [h]

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                h = np.maximum(0, z)
            else:  # No activation for output
                h = z
            activations.append(h)

        self.cache['activations'] = activations
        return h

    def backward(self, dQ, learning_rate=0.001):
        """
        Backward pass with gradient descent update.

        dQ: Gradient w.r.t. Q-values (batch_size, n_actions)
        """
        activations = self.cache['activations']
        dh = dQ

        for i in range(len(self.weights) - 1, -1, -1):
            h_prev = activations[i]
            z = activations[i+1]

            # Gradient through linear
            dW = h_prev.T @ dh
            db = np.sum(dh, axis=0)
            dh_prev = dh @ self.weights[i].T

            # Gradient through ReLU (except output layer)
            if i > 0:
                dh_prev = dh_prev * (activations[i] > 0)

            # Update weights
            self.weights[i] -= learning_rate * dW / len(dQ)
            self.biases[i] -= learning_rate * db / len(dQ)

            dh = dh_prev

    def copy_from(self, other):
        """Copy weights from another network (for target network)."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()

    def get_action(self, state, epsilon=0.0):
        """ε-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.forward(state.reshape(1, -1))
            return np.argmax(q_values[0])


class DQN:
    """
    Deep Q-Network Agent

    THE ALGORITHM:
    1. Initialize Q-network θ and target network θ'
    2. Initialize replay buffer D
    3. For each episode:
       a. Observe state s
       b. Select action (ε-greedy from Q_θ)
       c. Execute action, observe r, s'
       d. Store (s, a, r, s', done) in D
       e. Sample minibatch from D
       f. Compute target: y = r + γ max_a' Q_θ'(s', a')
       g. Update θ by minimizing (y - Q_θ(s,a))²
       h. Periodically copy θ → θ'
    """

    def __init__(self, state_dim, n_actions, hidden_dims=[64, 64],
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000,
                 batch_size=32, target_update_freq=100,
                 learning_rate=0.001, double_dqn=False):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.double_dqn = double_dqn

        # Q-network and target network
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dims)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        self.total_steps = 0

    def select_action(self, state):
        """Select action using ε-greedy policy."""
        return self.q_network.get_action(state, self.epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        THE DQN UPDATE

        1. Sample minibatch from buffer
        2. Compute TD target using target network
        3. Update Q-network to minimize TD error
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q-values
        q_values = self.q_network.forward(states)
        current_q = q_values[np.arange(len(actions)), actions]

        # Target Q-values
        if self.double_dqn:
            # DOUBLE DQN: Use Q-network to select, target network to evaluate
            next_q_values = self.q_network.forward(next_states)
            best_actions = np.argmax(next_q_values, axis=1)
            next_q_target = self.target_network.forward(next_states)
            max_next_q = next_q_target[np.arange(len(best_actions)), best_actions]
        else:
            # Standard DQN: Use target network for both
            next_q_target = self.target_network.forward(next_states)
            max_next_q = np.max(next_q_target, axis=1)

        # TD target: r + γ max Q(s', a') for non-terminal, r for terminal
        target = rewards + self.gamma * max_next_q * (1 - dones)

        # TD error
        td_error = current_q - target

        # Gradient: d/dQ [(Q - target)²] = 2(Q - target)
        dQ = np.zeros_like(q_values)
        dQ[np.arange(len(actions)), actions] = 2 * td_error

        # Backprop
        self.q_network.backward(dQ, self.learning_rate)

        # Update target network periodically
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return np.mean(td_error ** 2)


def state_to_vector(state, grid_size):
    """Convert grid position to one-hot vector."""
    vec = np.zeros(grid_size * grid_size)
    idx = state[0] * grid_size + state[1]
    vec[idx] = 1.0
    return vec


def train_dqn(env, agent, n_episodes=500, max_steps=100):
    """Train DQN agent."""
    episode_rewards = []
    episode_lengths = []
    losses = []

    for episode in range(n_episodes):
        state = env.reset()
        state_vec = state_to_vector(state, env.size)
        total_reward = 0
        ep_loss = []

        for step in range(max_steps):
            # Select and execute action
            action = agent.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)

            # Store and update
            agent.store_transition(state_vec, action, reward, next_state_vec, done)
            loss = agent.update()
            if loss > 0:
                ep_loss.append(loss)

            total_reward += reward
            state = next_state
            state_vec = next_state_vec

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        if ep_loss:
            losses.append(np.mean(ep_loss))

    return episode_rewards, episode_lengths, losses


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_dqn():
    """
    Create comprehensive DQN visualization:
    1. Learning curve comparison (with/without replay)
    2. Target network effect
    3. Double DQN vs DQN
    4. Replay buffer importance
    5. Q-value evolution
    6. Summary
    """
    print("\n" + "="*60)
    print("DQN VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: DQN Learning Curve ============
    ax1 = fig.add_subplot(2, 3, 1)

    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                gamma=0.99, buffer_size=5000, batch_size=32,
                target_update_freq=50, learning_rate=0.001)

    rewards, lengths, losses = train_dqn(env, agent, n_episodes=300)

    # Smooth
    window = 20
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, 'b-', linewidth=2, label='DQN')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Learning Curve\nLearns to reach goal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ============ Plot 2: With vs Without Experience Replay ============
    ax2 = fig.add_subplot(2, 3, 2)

    n_runs = 3
    n_episodes = 200

    # With replay
    rewards_with_replay = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                    buffer_size=5000, batch_size=32)
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_with_replay.append(r)

    # Without replay (batch_size = 1, immediate update)
    rewards_no_replay = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                    buffer_size=100, batch_size=1)  # Minimal replay
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_no_replay.append(r)

    # Average and smooth
    with_replay = np.mean(rewards_with_replay, axis=0)
    no_replay = np.mean(rewards_no_replay, axis=0)

    window = 15
    ax2.plot(np.convolve(with_replay, np.ones(window)/window, mode='valid'),
             'b-', linewidth=2, label='With Replay')
    ax2.plot(np.convolve(no_replay, np.ones(window)/window, mode='valid'),
             'r--', linewidth=2, label='Minimal Replay')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('EXPERIENCE REPLAY Effect\nReplay stabilizes learning!')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Target Network Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Fast target update (unstable)
    rewards_fast = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                    target_update_freq=1)  # Update every step!
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_fast.append(r)

    # Slow target update (stable)
    rewards_slow = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                    target_update_freq=100)  # Update every 100 steps
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_slow.append(r)

    fast_mean = np.mean(rewards_fast, axis=0)
    slow_mean = np.mean(rewards_slow, axis=0)

    ax3.plot(np.convolve(fast_mean, np.ones(window)/window, mode='valid'),
             'r--', linewidth=2, label='Fast update (every step)')
    ax3.plot(np.convolve(slow_mean, np.ones(window)/window, mode='valid'),
             'b-', linewidth=2, label='Slow update (every 100)')

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('TARGET NETWORK Update Frequency\nSlow updates → stable!')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Double DQN ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Standard DQN
    rewards_dqn = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32], double_dqn=False)
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_dqn.append(r)

    # Double DQN
    rewards_ddqn = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = DQN(state_dim, n_actions, hidden_dims=[32, 32], double_dqn=True)
        r, _, _ = train_dqn(env, agent, n_episodes=n_episodes)
        rewards_ddqn.append(r)

    dqn_mean = np.mean(rewards_dqn, axis=0)
    ddqn_mean = np.mean(rewards_ddqn, axis=0)

    ax4.plot(np.convolve(dqn_mean, np.ones(window)/window, mode='valid'),
             'b-', linewidth=2, label='DQN')
    ax4.plot(np.convolve(ddqn_mean, np.ones(window)/window, mode='valid'),
             'g-', linewidth=2, label='Double DQN')

    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('DQN vs Double DQN\nDouble reduces overestimation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: TD Loss During Training ============
    ax5 = fig.add_subplot(2, 3, 5)

    env = GridWorld(size=5)
    agent = DQN(state_dim, n_actions, hidden_dims=[32, 32])
    _, _, losses = train_dqn(env, agent, n_episodes=300)

    if losses:
        window = 10
        smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax5.plot(smoothed_loss, 'b-', linewidth=1)
        ax5.set_xlabel('Update Step')
        ax5.set_ylabel('TD Loss')
        ax5.set_title('TD Loss During Training\nDecreases as Q converges')
        ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    DQN — Deep Q-Network
    ═════════════════════

    CORE IDEA:
    Q_θ(s,a) ≈ Q*(s,a)
    Neural network approximates Q-table

    THE THREE TRICKS:
    ┌─────────────────────────────┐
    │ 1. EXPERIENCE REPLAY        │
    │    Break correlations       │
    │    Reuse data               │
    ├─────────────────────────────┤
    │ 2. TARGET NETWORK           │
    │    Stable learning target   │
    │    Copy θ → θ' periodically │
    ├─────────────────────────────┤
    │ 3. DOUBLE DQN               │
    │    Reduce overestimation    │
    │    Decouple select/evaluate │
    └─────────────────────────────┘

    LOSS:
    L = (r + γ max Q_θ'(s',a') - Q_θ(s,a))²

    WHY IT WORKS:
    • Generalizes across similar states
    • Sample efficient (replay)
    • Stable (target network)
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('DQN — Deep Q-Network\n'
                 'Neural network + Experience Replay + Target Network',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for DQN."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # 1. Buffer size
    print("\n1. EFFECT OF REPLAY BUFFER SIZE")
    print("-" * 40)

    for buffer_size in [100, 500, 1000, 5000, 10000]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                        buffer_size=buffer_size)
            rewards, _, _ = train_dqn(env, agent, n_episodes=150)
            rewards_list.append(np.mean(rewards[-30:]))
        print(f"buffer_size={buffer_size:>5}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Larger buffer = more diverse samples = better learning")

    # 2. Batch size
    print("\n2. EFFECT OF BATCH SIZE")
    print("-" * 40)

    for batch_size in [1, 8, 32, 64, 128]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                        batch_size=batch_size)
            rewards, _, _ = train_dqn(env, agent, n_episodes=150)
            rewards_list.append(np.mean(rewards[-30:]))
        print(f"batch_size={batch_size:>3}  final_reward={np.mean(rewards_list):.3f}")

    print("→ batch_size=32-64 typically good balance")

    # 3. Target update frequency
    print("\n3. EFFECT OF TARGET UPDATE FREQUENCY")
    print("-" * 40)

    for freq in [1, 10, 50, 100, 500]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                        target_update_freq=freq)
            rewards, _, _ = train_dqn(env, agent, n_episodes=150)
            rewards_list.append(np.mean(rewards[-30:]))
        print(f"target_update_freq={freq:>3}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Update every 50-100 steps usually stable")

    # 4. Network architecture
    print("\n4. EFFECT OF NETWORK SIZE")
    print("-" * 40)

    for hidden_dims in [[16], [32, 32], [64, 64], [128, 64, 32]]:
        rewards_list = []
        n_params = 0
        for _ in range(3):
            env = GridWorld(size=5)
            agent = DQN(state_dim, n_actions, hidden_dims=hidden_dims)
            rewards, _, _ = train_dqn(env, agent, n_episodes=150)
            rewards_list.append(np.mean(rewards[-30:]))
            n_params = sum(w.size for w in agent.q_network.weights)
        print(f"hidden={str(hidden_dims):<15}  params={n_params:<5}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Moderate network sufficient for simple tasks")

    # 5. DQN vs Double DQN
    print("\n5. DQN vs DOUBLE DQN")
    print("-" * 40)

    for double in [False, True]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = DQN(state_dim, n_actions, hidden_dims=[32, 32],
                        double_dqn=double)
            rewards, _, _ = train_dqn(env, agent, n_episodes=200)
            rewards_list.append(np.mean(rewards[-30:]))
        name = "Double DQN" if double else "Standard DQN"
        print(f"{name:<15}  final_reward={np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")

    print("→ Double DQN reduces overestimation, often more stable")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("DQN — Deep Q-Network")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_dqn()
    save_path = '/Users/sid47/ML Algorithms/27_dqn.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. DQN: Neural network approximates Q-function
2. Experience Replay: Break correlations, reuse data
3. Target Network: Stable learning target
4. Double DQN: Reduce overestimation bias
5. Key insight: Generalize across similar states!
6. Enables RL on high-dimensional state spaces
    """)
