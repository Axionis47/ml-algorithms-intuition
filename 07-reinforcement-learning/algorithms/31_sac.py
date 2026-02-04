"""
SAC — Soft Actor-Critic
========================

Paradigm: MAXIMUM ENTROPY REINFORCEMENT LEARNING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Maximize reward AND entropy (randomness)!

STANDARD RL OBJECTIVE:
    J(π) = Σ E[r(s,a)]
    "Get high reward"

MAXIMUM ENTROPY OBJECTIVE:
    J(π) = Σ E[r(s,a) + α × H(π(·|s))]
                       ↑
               Entropy bonus!
    "Get high reward, but ALSO stay random"

WHY ENTROPY?
1. EXPLORATION: Random = try different things
2. ROBUSTNESS: Multiple ways to succeed → more robust
3. DIVERSE BEHAVIORS: Captures multiple solutions
4. SMOOTHER OPTIMIZATION: Less sensitive to local optima

===============================================================
THE SAC COMPONENTS
===============================================================

1. SOFT Q-FUNCTION Q(s,a)
   Includes future entropy:
   Q(s,a) = r + γ E[Q(s',a') - α log π(a'|s')]
                            ↑
                    Entropy term!

2. SOFT VALUE FUNCTION V(s)
   V(s) = E_a[Q(s,a) - α log π(a|s)]
        = E_a[Q(s,a)] + α H(π(·|s))
        "Expected Q plus entropy bonus"

3. POLICY π(a|s)
   Gaussian policy: μ(s), σ(s)
   Outputs mean and std for continuous actions

4. TEMPERATURE α
   Controls exploration-exploitation:
   - High α: More entropy, more exploration
   - Low α: Less entropy, more exploitation
   Can be learned automatically!

===============================================================
THE REPARAMETERIZATION TRICK
===============================================================

Instead of sampling: a ~ π(a|s)
Use: a = μ(s) + σ(s) × ε,  where ε ~ N(0,1)

WHY?
- Sampling blocks gradients
- Reparameterization lets gradients flow!
- Can backprop through the sample

===============================================================
SAC UPDATE EQUATIONS
===============================================================

Q-FUNCTION UPDATE (Bellman):
    L_Q = E[(Q(s,a) - (r + γ(Q(s',a') - α log π(a'|s'))))²]

    Minimize difference between:
    - Current Q estimate
    - Bootstrap target with entropy

POLICY UPDATE:
    L_π = E[α log π(a|s) - Q(s,a)]

    Maximize Q(s,a) - α log π(a|s)
    = Maximize reward + entropy

TEMPERATURE UPDATE (optional):
    L_α = E[-α × (log π(a|s) + target_entropy)]

    Adjust α so that entropy ≈ target_entropy

===============================================================
SAC vs OTHER ALGORITHMS
===============================================================

vs DQN:
+ Better exploration (entropy)
+ Continuous actions (naturally)
+ More stable (soft updates)

vs PPO:
+ Off-policy (more sample efficient)
+ Continuous actions (naturally)
- More complex

vs DDPG:
+ Better exploration (stochastic policy)
+ More stable (entropy regularization)
- Slower (two Q-functions)

===============================================================
INDUCTIVE BIAS
===============================================================

1. STOCHASTIC POLICY: Always maintains randomness
2. CONTINUOUS ACTIONS: Designed for continuous control
3. OFF-POLICY: Can learn from any data
4. ENTROPY: Prefers diverse solutions

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy:
    """
    Gaussian Policy for continuous actions.

    π(a|s) = N(μ(s), σ(s)²)

    Uses tanh squashing for bounded actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 log_std_min=-20, log_std_max=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Network: state → [mean, log_std]
        scale = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)

        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)

        # Output: mean and log_std
        self.W_mean = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b_mean = np.zeros(action_dim)

        self.W_log_std = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b_log_std = np.zeros(action_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state):
        """Get mean and log_std."""
        h = self.relu(state @ self.W1 + self.b1)
        h = self.relu(h @ self.W2 + self.b2)

        mean = h @ self.W_mean + self.b_mean
        log_std = h @ self.W_log_std + self.b_log_std
        log_std = np.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, deterministic=False):
        """
        Sample action using reparameterization trick.

        Returns: action, log_prob
        """
        mean, log_std = self.forward(state)
        std = np.exp(log_std)

        if deterministic:
            # For evaluation
            action = np.tanh(mean)
            log_prob = 0.0
        else:
            # Reparameterization: a = μ + σ × ε
            epsilon = np.random.randn(*mean.shape)
            action_pre_tanh = mean + std * epsilon
            action = np.tanh(action_pre_tanh)

            # Log probability with tanh squashing correction
            # log π(a|s) = log N(z|μ,σ) - Σ log(1 - tanh²(z))
            log_prob = -0.5 * (((action_pre_tanh - mean) / (std + 1e-8))**2
                              + 2 * log_std + np.log(2 * np.pi))
            log_prob = np.sum(log_prob, axis=-1)

            # Tanh squashing correction
            log_prob -= np.sum(np.log(1 - action**2 + 1e-6), axis=-1)

        return action, log_prob

    def get_params(self):
        """Get all parameters."""
        return [self.W1, self.b1, self.W2, self.b2,
                self.W_mean, self.b_mean, self.W_log_std, self.b_log_std]

    def set_params(self, params):
        """Set all parameters."""
        (self.W1, self.b1, self.W2, self.b2,
         self.W_mean, self.b_mean, self.W_log_std, self.b_log_std) = params


class QNetwork:
    """
    Soft Q-Network Q(s, a).

    Takes state and action, outputs Q-value.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        input_dim = state_dim + action_dim

        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)

        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(hidden_dim, 1) * 0.1
        self.b3 = np.zeros(1)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state, action):
        """Compute Q(s, a)."""
        # Ensure 2D
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        x = np.concatenate([state, action], axis=-1)
        h = self.relu(x @ self.W1 + self.b1)
        h = self.relu(h @ self.W2 + self.b2)
        q = h @ self.W3 + self.b3
        return q.squeeze()

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params


class SAC:
    """
    Soft Actor-Critic Agent.

    Features:
    - Two Q-networks (clipped double Q for stability)
    - Gaussian policy with tanh squashing
    - Automatic temperature adjustment
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 gamma=0.99, tau=0.005, alpha=0.2, lr=0.001,
                 auto_alpha=True, target_entropy=None):
        """
        Parameters:
        - state_dim: State space dimension
        - action_dim: Action space dimension
        - gamma: Discount factor
        - tau: Soft target update rate
        - alpha: Initial temperature
        - auto_alpha: Whether to learn alpha
        - target_entropy: Target entropy (default: -action_dim)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.auto_alpha = auto_alpha

        # Temperature
        self.log_alpha = np.log(alpha)
        self.alpha = alpha
        self.target_entropy = target_entropy or -action_dim

        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

        # Target Q networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)

        # Initialize targets
        self._copy_params(self.q1, self.q1_target)
        self._copy_params(self.q2, self.q2_target)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training history
        self.history = {
            'q_loss': [], 'policy_loss': [], 'alpha_loss': [],
            'alpha': [], 'episode_rewards': []
        }

    def _copy_params(self, source, target):
        """Copy parameters from source to target."""
        target.set_params([p.copy() for p in source.get_params()])

    def _soft_update(self, source, target):
        """Soft update: target = τ*source + (1-τ)*target."""
        for p_source, p_target in zip(source.get_params(), target.get_params()):
            p_target[:] = self.tau * p_source + (1 - self.tau) * p_target

    def select_action(self, state, deterministic=False):
        """Select action from policy."""
        state = np.array(state).reshape(1, -1)
        action, _ = self.policy.sample(state, deterministic)
        return action.squeeze()

    def update(self, batch_size=64):
        """Update networks using one batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # ========== Update Q-functions ==========
        # Sample next actions from current policy
        next_actions, next_log_probs = self.policy.sample(next_states)

        # Compute target Q values
        q1_next = self.q1_target.forward(next_states, next_actions)
        q2_next = self.q2_target.forward(next_states, next_actions)
        q_next = np.minimum(q1_next, q2_next)  # Clipped double Q

        # Target: r + γ(Q_next - α log π)
        targets = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

        # Current Q values
        q1_current = self.q1.forward(states, actions)
        q2_current = self.q2.forward(states, actions)

        # Q loss (MSE)
        q1_loss = np.mean((q1_current - targets)**2)
        q2_loss = np.mean((q2_current - targets)**2)

        # Simplified gradient update for Q networks
        # (Full backprop through neural network is complex)
        self._update_q_network(self.q1, states, actions, targets)
        self._update_q_network(self.q2, states, actions, targets)

        # ========== Update Policy ==========
        # Sample actions from current policy
        new_actions, log_probs = self.policy.sample(states)

        # Q values for new actions
        q1_new = self.q1.forward(states, new_actions)
        q2_new = self.q2.forward(states, new_actions)
        q_new = np.minimum(q1_new, q2_new)

        # Policy loss: α log π - Q
        policy_loss = np.mean(self.alpha * log_probs - q_new)

        # Simplified policy gradient update
        self._update_policy(states)

        # ========== Update Temperature α ==========
        if self.auto_alpha:
            # α loss: -α (log π + target_entropy)
            alpha_loss = -np.mean(self.alpha * (log_probs + self.target_entropy))

            # Update log_alpha
            self.log_alpha -= self.lr * alpha_loss
            self.alpha = np.exp(self.log_alpha)
            self.alpha = np.clip(self.alpha, 0.01, 10.0)  # Stability

            self.history['alpha_loss'].append(alpha_loss)
            self.history['alpha'].append(self.alpha)

        # ========== Soft Update Targets ==========
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        # Record history
        self.history['q_loss'].append(0.5 * (q1_loss + q2_loss))
        self.history['policy_loss'].append(policy_loss)

    def _update_q_network(self, q_net, states, actions, targets):
        """Simplified Q network update."""
        # Compute current Q and gradient direction
        q_current = q_net.forward(states, actions)
        error = q_current - targets

        # Simplified gradient: just update last layer
        # (Full backprop would be complex)
        q_net.b3 -= self.lr * np.mean(error)

    def _update_policy(self, states):
        """Simplified policy update."""
        # Compute policy gradient direction
        # Approximate: increase probability of high-Q actions
        actions, log_probs = self.policy.sample(states)
        q_values = self.q1.forward(states, actions)

        # Advantage-like term
        baseline = np.mean(q_values)
        advantages = q_values - baseline

        # Simple policy gradient update
        mean, _ = self.policy.forward(states)

        # Update toward actions with positive advantage
        gradient = np.mean(advantages.reshape(-1, 1) * (actions - mean), axis=0)
        self.policy.b_mean += self.lr * gradient


class ContinuousEnvironment:
    """
    Simple continuous control environment for testing SAC.

    Task: Move to target position
    State: [position, velocity]
    Action: [-1, 1] force
    """

    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.dt = 0.1
        self.target = 1.0
        self.reset()

    def reset(self):
        """Reset to random initial state."""
        self.position = np.random.uniform(-0.5, 0.5)
        self.velocity = np.random.uniform(-0.1, 0.1)
        return self._get_state()

    def _get_state(self):
        return np.array([self.position, self.velocity])

    def step(self, action):
        """Take action, return (state, reward, done)."""
        action = np.clip(action, -1, 1).item() if hasattr(action, 'item') else float(action)

        # Physics update
        self.velocity += action * self.dt
        self.velocity = np.clip(self.velocity, -1, 1)
        self.position += self.velocity * self.dt

        # Reward: negative distance to target
        reward = -abs(self.position - self.target) - 0.1 * abs(action)

        # Done if reached target
        done = abs(self.position - self.target) < 0.1

        return self._get_state(), reward, done


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_sac():
    """
    Comprehensive SAC visualization:
    1. Training progress
    2. Temperature adaptation
    3. Policy evolution
    4. Entropy effect
    5. Double Q comparison
    6. Summary
    """
    print("\n" + "="*60)
    print("SAC VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: SAC Training ============
    ax1 = fig.add_subplot(2, 3, 1)

    env = ContinuousEnvironment()
    agent = SAC(state_dim=2, action_dim=1, hidden_dim=32,
                auto_alpha=True, alpha=0.2)

    episode_rewards = []

    for episode in range(100):
        state = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.replay_buffer.push(state, [action], reward, next_state, done)
            agent.update(batch_size=32)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

    # Smoothed rewards
    window = 10
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(range(window-1, len(episode_rewards)), smoothed,
             color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('SAC Training\nContinuous control task')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Temperature Adaptation ============
    ax2 = fig.add_subplot(2, 3, 2)

    if agent.history['alpha']:
        ax2.plot(agent.history['alpha'], 'r-', linewidth=2)
        ax2.axhline(y=0.2, color='gray', linestyle='--', label='Initial α')
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Temperature α')
        ax2.set_title('Automatic Temperature Tuning\nα adjusts to maintain target entropy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Q-function Loss ============
    ax3 = fig.add_subplot(2, 3, 3)

    if agent.history['q_loss']:
        ax3.plot(agent.history['q_loss'], 'b-', alpha=0.7)
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Q Loss')
        ax3.set_title('Q-function Learning\nSoft Bellman backup')
        ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Entropy Effect Comparison ============
    ax4 = fig.add_subplot(2, 3, 4)

    # Compare different fixed temperatures
    alphas = [0.01, 0.1, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alphas)))

    for alpha, color in zip(alphas, colors):
        env = ContinuousEnvironment()
        agent = SAC(state_dim=2, action_dim=1, hidden_dim=32,
                    auto_alpha=False, alpha=alpha)

        rewards = []
        for episode in range(50):
            state = env.reset()
            total_reward = 0

            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.push(state, [action], reward, next_state, done)
                agent.update(batch_size=32)
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)

        smoothed = np.convolve(rewards, np.ones(5)/5, mode='valid')
        ax4.plot(smoothed, color=color, linewidth=2, label=f'α={alpha}')

    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.set_title('Temperature Effect\nHigher α = more exploration')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Policy Stochasticity ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Show policy action distribution at different training stages
    env = ContinuousEnvironment()

    # Fresh policy
    agent_fresh = SAC(state_dim=2, action_dim=1, auto_alpha=False, alpha=0.5)

    test_state = np.array([[0.5, 0.0]])  # Fixed state

    # Sample actions from fresh policy
    actions_fresh = [agent_fresh.policy.sample(test_state)[0].item() for _ in range(200)]

    # Train for a bit
    for _ in range(50):
        state = env.reset()
        for _ in range(30):
            action = agent_fresh.select_action(state)
            next_state, reward, done = env.step(action)
            agent_fresh.replay_buffer.push(state, [action], reward, next_state, done)
            agent_fresh.update(batch_size=32)
            state = next_state
            if done:
                break

    # Sample actions from trained policy
    actions_trained = [agent_fresh.policy.sample(test_state)[0].item() for _ in range(200)]

    ax5.hist(actions_fresh, bins=20, alpha=0.5, label='Initial (random)', density=True)
    ax5.hist(actions_trained, bins=20, alpha=0.5, label='After training', density=True)
    ax5.set_xlabel('Action')
    ax5.set_ylabel('Density')
    ax5.set_title('Policy Distribution Evolution\nStays stochastic but becomes directed')
    ax5.legend()

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    SAC — Soft Actor-Critic
    ══════════════════════════════

    THE KEY IDEA:
    Maximize reward + entropy!

    J(π) = Σ E[r + α × H(π)]

    "High reward, but stay random"

    COMPONENTS:
    ┌────────────────────────────┐
    │ Soft Q: Q(s,a)             │
    │   Includes entropy term    │
    ├────────────────────────────┤
    │ Gaussian Policy:           │
    │   μ(s), σ(s) → action     │
    ├────────────────────────────┤
    │ Temperature α:             │
    │   Exploration-exploitation │
    └────────────────────────────┘

    BENEFITS:
    ✓ Better exploration (entropy)
    ✓ Off-policy (sample efficient)
    ✓ Stable (soft updates)
    ✓ Auto temperature tuning

    STATE-OF-THE-ART for:
    Continuous control tasks!
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    plt.suptitle('SAC — Soft Actor-Critic\n'
                 'Maximum Entropy Reinforcement Learning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for SAC."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Temperature α effect
    print("\n1. EFFECT OF TEMPERATURE α")
    print("-" * 40)

    for alpha in [0.01, 0.1, 0.2, 0.5, 1.0]:
        env = ContinuousEnvironment()
        agent = SAC(state_dim=2, action_dim=1, auto_alpha=False, alpha=alpha)

        total_rewards = []
        for episode in range(30):
            state = env.reset()
            episode_reward = 0
            for _ in range(50):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.push(state, [action], reward, next_state, done)
                agent.update(batch_size=32)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)

        print(f"α={alpha:.2f}  final_reward={np.mean(total_rewards[-10:]):.3f}")

    print("→ Moderate α balances exploration and exploitation")

    # 2. Auto vs Fixed Temperature
    print("\n2. AUTO vs FIXED TEMPERATURE")
    print("-" * 40)

    for auto_alpha in [False, True]:
        env = ContinuousEnvironment()
        agent = SAC(state_dim=2, action_dim=1, auto_alpha=auto_alpha, alpha=0.2)

        total_rewards = []
        for episode in range(30):
            state = env.reset()
            episode_reward = 0
            for _ in range(50):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.push(state, [action], reward, next_state, done)
                agent.update(batch_size=32)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)

        label = "Auto" if auto_alpha else "Fixed"
        print(f"{label:<6}  final_reward={np.mean(total_rewards[-10:]):.3f}")

    print("→ Auto-tuning often helps")

    # 3. Soft Update Rate τ
    print("\n3. EFFECT OF SOFT UPDATE RATE τ")
    print("-" * 40)

    for tau in [0.001, 0.005, 0.01, 0.05, 0.1]:
        env = ContinuousEnvironment()
        agent = SAC(state_dim=2, action_dim=1, tau=tau)

        total_rewards = []
        for episode in range(30):
            state = env.reset()
            episode_reward = 0
            for _ in range(50):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.replay_buffer.push(state, [action], reward, next_state, done)
                agent.update(batch_size=32)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)

        print(f"τ={tau:.3f}  final_reward={np.mean(total_rewards[-10:]):.3f}")

    print("→ τ≈0.005 is standard choice")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("SAC — Soft Actor-Critic")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_sac()
    save_path = '/Users/sid47/ML Algorithms/31_sac.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. SAC: Maximum entropy RL - maximize reward + entropy
2. J(π) = Σ E[r + α H(π)]
3. Entropy encourages exploration and diverse solutions
4. Gaussian policy with tanh squashing for continuous actions
5. Reparameterization trick for gradient through sampling
6. Two Q-networks (clipped double Q) for stability
7. Automatic temperature tuning
8. State-of-the-art for continuous control
    """)
