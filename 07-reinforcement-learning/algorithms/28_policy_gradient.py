"""
POLICY GRADIENT — Direct Policy Optimization
=============================================

Paradigm: LEARN THE POLICY DIRECTLY

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Instead of learning Q-values, DIRECTLY optimize the policy!

THE POLICY GRADIENT THEOREM:
    ∇J(θ) = E[∇log π_θ(a|s) × R]

    "Increase probability of actions that led to HIGH reward"
    "Decrease probability of actions that led to LOW reward"

INTUITION:
- π_θ(a|s): probability of taking action a in state s
- ∇log π_θ: direction to increase that probability
- Multiply by R: weight by how good the outcome was

===============================================================
REINFORCE ALGORITHM
===============================================================

The simplest policy gradient: Monte Carlo Policy Gradient

ALGORITHM:
1. Sample trajectory τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...)
2. Compute return G_t = Σ_{k=0}^∞ γ^k r_{t+k}
3. Update: θ ← θ + α × Σ_t ∇log π_θ(a_t|s_t) × G_t

KEY INSIGHT:
- No value function needed!
- Learn directly from complete episodes
- Naturally handles continuous actions

===============================================================
THE VARIANCE PROBLEM
===============================================================

Policy gradients have HIGH VARIANCE:
- Same action can have different returns (stochasticity)
- Long episodes = more variance
- Slow learning, unstable

SOLUTIONS:

1. BASELINE SUBTRACTION
   ∇J ≈ E[∇log π(a|s) × (R - b(s))]

   Subtract baseline b(s) to reduce variance.
   Common choice: b(s) = V(s) (average return from s)

   WHY IT WORKS:
   - Doesn't change expected gradient (unbiased)
   - But reduces variance significantly!

2. ADVANTAGE FUNCTION
   A(s,a) = Q(s,a) - V(s)
          = "How much BETTER is this action than average?"

   Use advantage instead of return:
   ∇J ≈ E[∇log π(a|s) × A(s,a)]

3. REWARD NORMALIZATION
   Normalize returns to have mean 0, std 1
   Stabilizes learning

===============================================================
POLICY GRADIENT vs VALUE-BASED
===============================================================

VALUE-BASED (Q-learning, DQN):
+ Sample efficient (off-policy)
+ Stable learning
- Can't easily handle continuous actions
- Only deterministic policies (after training)

POLICY GRADIENT:
+ Natural for continuous actions
+ Can learn stochastic policies
+ Directly optimizes what we care about
- High variance
- Sample inefficient (on-policy)
- Local optima

===============================================================
INDUCTIVE BIAS
===============================================================

1. STOCHASTIC POLICIES
   - Naturally explores
   - Can represent mixed strategies
   - Smooth optimization landscape

2. ON-POLICY LEARNING
   - Must use current policy's data
   - Can't reuse old experience directly

3. DIRECT OPTIMIZATION
   - No intermediate value function
   - Optimize end objective directly

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
rl_fund = import_module('25_rl_fundamentals')
GridWorld = rl_fund.GridWorld


class PolicyNetwork:
    """
    Neural network that outputs action probabilities.

    π_θ(a|s) = softmax(f_θ(s))

    Architecture: state → hidden → softmax(logits) → probabilities
    """

    def __init__(self, state_dim, n_actions, hidden_dim=32):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + n_actions))

        self.W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * scale2
        self.b2 = np.zeros(n_actions)

        self.cache = {}

    def forward(self, state):
        """
        Forward pass: state → action probabilities.

        Returns: probabilities π(a|s) for all actions
        """
        # Hidden layer with tanh
        z1 = state @ self.W1 + self.b1
        h = np.tanh(z1)

        # Output logits
        logits = h @ self.W2 + self.b2

        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cache for backprop
        self.cache = {'state': state, 'z1': z1, 'h': h, 'logits': logits, 'probs': probs}

        return probs

    def sample_action(self, state):
        """Sample action from policy distribution."""
        probs = self.forward(state.reshape(1, -1))[0]
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def update(self, states, actions, advantages, learning_rate=0.01):
        """
        POLICY GRADIENT UPDATE

        ∇J = Σ ∇log π(a|s) × A(s,a)

        For softmax policy:
        ∇log π(a|s) = e_a - π(·|s)
        where e_a is one-hot for action a
        """
        batch_size = len(states)

        # Forward pass
        probs = self.forward(states)

        # Gradient of log π(a|s) w.r.t. logits
        # d/d_logits log π(a) = e_a - π for softmax
        one_hot = np.zeros((batch_size, self.n_actions))
        one_hot[np.arange(batch_size), actions] = 1

        # Gradient weighted by advantage
        # d_logits = (one_hot - probs) × advantage
        d_logits = (one_hot - probs) * advantages.reshape(-1, 1)

        # Backprop through network
        h = self.cache['h']
        z1 = self.cache['z1']

        # Gradients
        dW2 = h.T @ d_logits
        db2 = np.sum(d_logits, axis=0)

        dh = d_logits @ self.W2.T
        dz1 = dh * (1 - np.tanh(z1)**2)  # tanh gradient

        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update (gradient ASCENT for maximizing reward)
        self.W2 += learning_rate * dW2 / batch_size
        self.b2 += learning_rate * db2 / batch_size
        self.W1 += learning_rate * dW1 / batch_size
        self.b1 += learning_rate * db1 / batch_size


class ValueNetwork:
    """
    Value function approximator for baseline.

    V_φ(s) ≈ E[G_t | S_t = s]
    """

    def __init__(self, state_dim, hidden_dim=32):
        self.state_dim = state_dim

        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + 1))

        self.W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)

        self.cache = {}

    def forward(self, states):
        """Predict value for states."""
        z1 = states @ self.W1 + self.b1
        h = np.tanh(z1)
        v = h @ self.W2 + self.b2
        self.cache = {'states': states, 'z1': z1, 'h': h}
        return v.flatten()

    def update(self, states, returns, learning_rate=0.01):
        """Update value function to minimize MSE with returns."""
        v_pred = self.forward(states)
        td_error = returns - v_pred

        # Backprop
        h = self.cache['h']
        z1 = self.cache['z1']

        dv = -2 * td_error.reshape(-1, 1) / len(states)

        dW2 = h.T @ dv
        db2 = np.sum(dv, axis=0)

        dh = dv @ self.W2.T
        dz1 = dh * (1 - np.tanh(z1)**2)

        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return np.mean(td_error**2)


class REINFORCE:
    """
    REINFORCE: Monte Carlo Policy Gradient

    THE ALGORITHM:
    1. Sample episode using current policy
    2. Compute returns G_t for each step
    3. Optionally compute advantages A_t = G_t - V(s_t)
    4. Update policy: θ ← θ + α ∇log π_θ(a|s) × A
    5. Update baseline: minimize (V(s) - G)²
    """

    def __init__(self, state_dim, n_actions, hidden_dim=32,
                 gamma=0.99, policy_lr=0.01, value_lr=0.01,
                 use_baseline=True, normalize_returns=True):

        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.use_baseline = use_baseline
        self.normalize_returns = normalize_returns

        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dim)

        if use_baseline:
            self.value_fn = ValueNetwork(state_dim, hidden_dim)
        else:
            self.value_fn = None

    def select_action(self, state):
        """Sample action from policy."""
        action, _ = self.policy.sample_action(state)
        return action

    def compute_returns(self, rewards):
        """
        Compute discounted returns G_t = Σ γ^k r_{t+k}

        Go BACKWARD for efficiency:
        G_t = r_t + γ G_{t+1}
        """
        returns = np.zeros(len(rewards))
        G = 0
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update(self, states, actions, rewards):
        """
        Update policy (and baseline) from episode.
        """
        states = np.array(states)
        actions = np.array(actions)

        # Compute returns
        returns = self.compute_returns(rewards)

        # Compute advantages
        if self.use_baseline:
            values = self.value_fn.forward(states)
            advantages = returns - values

            # Update value function
            self.value_fn.update(states, returns, self.value_lr)
        else:
            advantages = returns

        # Normalize advantages (reduces variance)
        if self.normalize_returns and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update policy
        self.policy.update(states, actions, advantages, self.policy_lr)

        return np.sum(rewards)


def state_to_vector(state, grid_size):
    """Convert grid position to one-hot vector."""
    vec = np.zeros(grid_size * grid_size)
    idx = state[0] * grid_size + state[1]
    vec[idx] = 1.0
    return vec


def train_reinforce(env, agent, n_episodes=500, max_steps=100):
    """Train REINFORCE agent."""
    episode_rewards = []

    for episode in range(n_episodes):
        states, actions, rewards = [], [], []

        state = env.reset()
        state_vec = state_to_vector(state, env.size)

        for step in range(max_steps):
            action = agent.select_action(state_vec)

            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)

            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            state_vec = next_state_vec

            if done:
                break

        # Update at end of episode
        total_reward = agent.update(states, actions, rewards)
        episode_rewards.append(total_reward)

    return episode_rewards


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_policy_gradient():
    """
    Create comprehensive Policy Gradient visualization:
    1. Learning curve
    2. With vs without baseline
    3. Effect of learning rate
    4. Policy evolution
    5. Variance analysis
    6. Summary
    """
    print("\n" + "="*60)
    print("POLICY GRADIENT VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # ============ Plot 1: REINFORCE Learning Curve ============
    ax1 = fig.add_subplot(2, 3, 1)

    agent = REINFORCE(state_dim, n_actions, hidden_dim=32,
                      use_baseline=True, normalize_returns=True)
    rewards = train_reinforce(env, agent, n_episodes=500)

    window = 30
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('REINFORCE Learning Curve\nLearns to reach goal')
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Baseline Effect ============
    ax2 = fig.add_subplot(2, 3, 2)

    n_runs = 5
    n_episodes = 300

    # With baseline
    rewards_baseline = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = REINFORCE(state_dim, n_actions, use_baseline=True)
        r = train_reinforce(env, agent, n_episodes=n_episodes)
        rewards_baseline.append(r)

    # Without baseline
    rewards_no_baseline = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = REINFORCE(state_dim, n_actions, use_baseline=False)
        r = train_reinforce(env, agent, n_episodes=n_episodes)
        rewards_no_baseline.append(r)

    with_bl = np.mean(rewards_baseline, axis=0)
    no_bl = np.mean(rewards_no_baseline, axis=0)

    window = 20
    ax2.plot(np.convolve(with_bl, np.ones(window)/window, mode='valid'),
             'b-', linewidth=2, label='With Baseline')
    ax2.plot(np.convolve(no_bl, np.ones(window)/window, mode='valid'),
             'r--', linewidth=2, label='No Baseline')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('BASELINE EFFECT\nBaseline reduces variance!')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Learning Rate Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    lrs = [0.001, 0.01, 0.05, 0.1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(lrs)))

    for lr, color in zip(lrs, colors):
        env = GridWorld(size=5)
        agent = REINFORCE(state_dim, n_actions, policy_lr=lr)
        r = train_reinforce(env, agent, n_episodes=300)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax3.plot(smoothed, color=color, label=f'lr={lr}', linewidth=1.5)

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Effect of Learning Rate\nToo high: unstable, Too low: slow')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Policy Evolution ============
    ax4 = fig.add_subplot(2, 3, 4)

    env = GridWorld(size=5)
    agent = REINFORCE(state_dim, n_actions, hidden_dim=32)

    # Track policy entropy over training
    entropies = []
    checkpoints = [0, 50, 150, 300]
    policies_at_checkpoint = []

    for episode in range(301):
        states, actions, rewards = [], [], []
        state = env.reset()
        state_vec = state_to_vector(state, env.size)

        for step in range(100):
            action = agent.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)

            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            state_vec = next_state_vec
            if done:
                break

        agent.update(states, actions, rewards)

        # Compute average entropy
        start_state = state_to_vector(env.start, env.size)
        probs = agent.policy.forward(start_state.reshape(1, -1))[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

        if episode in checkpoints:
            policies_at_checkpoint.append(probs.copy())

    ax4.plot(entropies, 'b-', linewidth=1)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Policy Entropy at Start State')
    ax4.set_title('Policy Evolution\nEntropy decreases (becomes more certain)')
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Variance Analysis ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Compute variance of gradient estimates
    env = GridWorld(size=5)
    agent_bl = REINFORCE(state_dim, n_actions, use_baseline=True)
    agent_no_bl = REINFORCE(state_dim, n_actions, use_baseline=False)

    var_with_bl = []
    var_no_bl = []

    for episode in range(200):
        # Collect episode
        states, actions, rewards = [], [], []
        state = env.reset()
        state_vec = state_to_vector(state, env.size)

        for _ in range(100):
            action = agent_bl.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)
            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)
            state, state_vec = next_state, next_state_vec
            if done:
                break

        # Compute returns
        returns = agent_bl.compute_returns(rewards)
        states_arr = np.array(states)

        # Variance with baseline
        if agent_bl.value_fn:
            values = agent_bl.value_fn.forward(states_arr)
            advantages_bl = returns - values
        else:
            advantages_bl = returns
        var_with_bl.append(np.var(advantages_bl))

        # Variance without baseline
        var_no_bl.append(np.var(returns))

        # Update both agents
        agent_bl.update(states, actions, rewards)

    window = 10
    ax5.semilogy(np.convolve(var_no_bl, np.ones(window)/window, mode='valid'),
                 'r--', label='No Baseline', linewidth=2)
    ax5.semilogy(np.convolve(var_with_bl, np.ones(window)/window, mode='valid'),
                 'b-', label='With Baseline', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Variance (log scale)')
    ax5.set_title('VARIANCE REDUCTION\nBaseline dramatically reduces variance!')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    POLICY GRADIENT — Direct Policy Learning
    ═════════════════════════════════════════

    THE GRADIENT:
    ∇J = E[∇log π(a|s) × R]

    "Increase prob of good actions,
     decrease prob of bad actions"

    REINFORCE:
    1. Sample episode
    2. Compute returns G_t
    3. Update: θ += α ∇log π × G

    THE VARIANCE PROBLEM:
    ┌─────────────────────────┐
    │ High variance           │
    │ → Slow learning         │
    │ → Unstable              │
    └─────────────────────────┘

    SOLUTIONS:
    ✓ Baseline: A = G - V(s)
    ✓ Normalize returns
    ✓ Larger batches

    VS VALUE-BASED:
    + Continuous actions
    + Stochastic policies
    - Sample inefficient
    - High variance
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('POLICY GRADIENT — REINFORCE\n'
                 'Directly optimize the policy with gradient ascent',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for Policy Gradient."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # 1. Baseline effect
    print("\n1. EFFECT OF BASELINE")
    print("-" * 40)

    for use_baseline in [False, True]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = REINFORCE(state_dim, n_actions, use_baseline=use_baseline)
            r = train_reinforce(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        name = "With baseline" if use_baseline else "No baseline"
        print(f"{name:<15}  final_reward={np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")

    print("→ Baseline reduces variance significantly!")

    # 2. Return normalization
    print("\n2. EFFECT OF RETURN NORMALIZATION")
    print("-" * 40)

    for normalize in [False, True]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = REINFORCE(state_dim, n_actions, normalize_returns=normalize)
            r = train_reinforce(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        name = "Normalized" if normalize else "Raw returns"
        print(f"{name:<15}  final_reward={np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")

    print("→ Normalization stabilizes learning")

    # 3. Learning rate
    print("\n3. EFFECT OF POLICY LEARNING RATE")
    print("-" * 40)

    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = REINFORCE(state_dim, n_actions, policy_lr=lr)
            r = train_reinforce(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"lr={lr:.3f}  final_reward={np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")

    print("→ lr=0.01-0.05 typically good")

    # 4. Discount factor
    print("\n4. EFFECT OF DISCOUNT FACTOR")
    print("-" * 40)

    for gamma in [0.9, 0.95, 0.99, 1.0]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = REINFORCE(state_dim, n_actions, gamma=gamma)
            r = train_reinforce(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"γ={gamma:.2f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ γ≈0.99 usually good for episodic tasks")

    # 5. Network size
    print("\n5. EFFECT OF NETWORK SIZE")
    print("-" * 40)

    for hidden_dim in [8, 16, 32, 64, 128]:
        rewards_list = []
        for _ in range(5):
            env = GridWorld(size=5)
            agent = REINFORCE(state_dim, n_actions, hidden_dim=hidden_dim)
            r = train_reinforce(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"hidden={hidden_dim:<3}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Moderate size sufficient for simple tasks")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("POLICY GRADIENT — REINFORCE")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_policy_gradient()
    save_path = '/Users/sid47/ML Algorithms/28_policy_gradient.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Policy Gradient: Directly optimize π_θ(a|s)
2. Gradient: ∇J = E[∇log π(a|s) × R]
3. REINFORCE: Monte Carlo policy gradient
4. High variance problem → Use baseline!
5. Baseline: A = G - V(s) reduces variance
6. Naturally handles continuous actions
    """)
