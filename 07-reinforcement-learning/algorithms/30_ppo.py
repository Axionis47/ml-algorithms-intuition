"""
PPO — Proximal Policy Optimization
===================================

Paradigm: STABLE POLICY OPTIMIZATION

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Policy gradient but CLIPPED to prevent too-large updates!

THE PPO-CLIP OBJECTIVE:
    L(θ) = E[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]

Where r_t(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)

"Move toward better actions, but NOT TOO FAR from old policy"

===============================================================
THE PROBLEM PPO SOLVES
===============================================================

Vanilla policy gradient issues:
1. Large updates can destroy the policy
2. Hard to pick right step size
3. One bad update = catastrophic

TRPO solution: Constrain KL divergence
   But complex! Requires conjugate gradient, line search...

PPO solution: CLIP the ratio!
   Simple! Just clamp values → automatic constraint

===============================================================
HOW CLIPPING WORKS
===============================================================

r_t(θ) = π_θ(a|s) / π_old(a|s)

If r > 1: New policy more likely to take action a
If r < 1: New policy less likely to take action a

THE CLIPPING LOGIC:

Case 1: Advantage A > 0 (good action)
   - Want to INCREASE probability (r > 1)
   - But clip at 1+ε prevents going too far
   - No incentive to push beyond clip

Case 2: Advantage A < 0 (bad action)
   - Want to DECREASE probability (r < 1)
   - But clip at 1-ε prevents going too far
   - No incentive to reduce beyond clip

RESULT: Automatic TRUST REGION without complex math!

===============================================================
PPO IMPLEMENTATION TRICKS
===============================================================

1. MULTIPLE EPOCHS ON SAME BATCH
   Unlike vanilla PG which uses data once,
   PPO can reuse data for several epochs.
   (Clipping makes this safe!)

2. VALUE FUNCTION CLIPPING (optional)
   Also clip value function updates:
   L_V = max((V - target)², (clip(V, V_old-ε, V_old+ε) - target)²)

3. ADVANTAGE NORMALIZATION
   Normalize advantages to mean=0, std=1.
   Stabilizes learning across different reward scales.

4. ENTROPY BONUS
   Add entropy term to encourage exploration.
   Prevents premature convergence.

===============================================================
PPO vs OTHER ALGORITHMS
===============================================================

vs REINFORCE:
   + More stable (clipping)
   + More sample efficient (multiple epochs)

vs TRPO:
   + Simpler (no conjugate gradient)
   + Faster per iteration
   ≈ Similar performance

vs DQN:
   + On-policy (learns current policy)
   + Natural for continuous actions
   - Less sample efficient

===============================================================
HYPERPARAMETERS
===============================================================

ε (clip ratio): 0.1 - 0.3
   Controls how far policy can move
   Smaller = more conservative

K (epochs per batch): 3 - 10
   How many times to reuse collected data
   More = more sample efficient, but risk overfitting

GAE λ: 0.95 - 0.99
   Bias-variance trade-off in advantage estimation

Batch size: 64 - 2048
   Larger = more stable, slower

===============================================================
INDUCTIVE BIAS
===============================================================

1. TRUST REGION (implicit)
   - Policy changes are bounded
   - Prevents catastrophic updates

2. ON-POLICY
   - Must use current policy's data
   - Multiple epochs partially mitigates

3. CLIPPING SYMMETRY
   - Same constraint for increasing/decreasing
   - Balanced exploration

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
rl_fund = import_module('25_rl_fundamentals')
GridWorld = rl_fund.GridWorld


class PPOAgent:
    """
    PPO: Proximal Policy Optimization

    THE ALGORITHM:
    1. Collect batch of trajectories with current policy π_old
    2. Compute advantages using GAE
    3. For K epochs:
       a. Compute ratio r = π_θ / π_old
       b. Compute clipped surrogate objective
       c. Update policy θ
       d. Update value function φ
    """

    def __init__(self, state_dim, n_actions, hidden_dim=32,
                 gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, policy_lr=0.01, value_lr=0.01,
                 epochs_per_update=4, entropy_coef=0.01,
                 value_clip=None):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.epochs_per_update = epochs_per_update
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip

        # Policy network
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + n_actions))

        self.policy_W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.policy_b1 = np.zeros(hidden_dim)
        self.policy_W2 = np.random.randn(hidden_dim, n_actions) * scale2
        self.policy_b2 = np.zeros(n_actions)

        # Value network
        scale_v = np.sqrt(2.0 / (hidden_dim + 1))
        self.value_W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.value_b1 = np.zeros(hidden_dim)
        self.value_W2 = np.random.randn(hidden_dim, 1) * scale_v
        self.value_b2 = np.zeros(1)

    def get_policy(self, states):
        """Compute action probabilities."""
        z1 = states @ self.policy_W1 + self.policy_b1
        h = np.tanh(z1)
        logits = h @ self.policy_W2 + self.policy_b2

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    def get_value(self, states):
        """Compute state values."""
        z1 = states @ self.value_W1 + self.value_b1
        h = np.tanh(z1)
        v = h @ self.value_W2 + self.value_b2
        return v.flatten()

    def select_action(self, state):
        """Sample action from policy."""
        probs = self.get_policy(state.reshape(1, -1))[0]
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0

        for t in range(T - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_probs, advantages, returns, old_values):
        """
        PPO UPDATE

        For K epochs:
        1. Compute ratio r = π_new / π_old
        2. Compute clipped objective
        3. Update networks
        """
        states = np.array(states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        advantages = np.array(advantages)
        returns = np.array(returns)
        old_values = np.array(old_values)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        policy_losses = []
        value_losses = []
        clip_fractions = []

        for epoch in range(self.epochs_per_update):
            # Get current policy probabilities
            current_probs = self.get_policy(states)
            current_action_probs = current_probs[np.arange(len(actions)), actions]

            # Compute ratio
            ratio = current_action_probs / (old_probs + 1e-10)

            # Clipped surrogate objective
            unclipped = ratio * advantages
            clipped = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # PPO objective: min of clipped and unclipped
            policy_loss = -np.mean(np.minimum(unclipped, clipped))

            # Track clipping
            clip_fraction = np.mean(np.abs(ratio - 1) > self.clip_ratio)
            clip_fractions.append(clip_fraction)

            # Entropy bonus
            entropy = -np.sum(current_probs * np.log(current_probs + 1e-10), axis=1)
            entropy_loss = -np.mean(entropy)

            # Update policy
            self._update_policy(states, actions, advantages, old_probs)

            # Update value function
            current_values = self.get_value(states)

            if self.value_clip:
                # Clipped value loss
                value_clipped = old_values + np.clip(
                    current_values - old_values, -self.value_clip, self.value_clip
                )
                v_loss1 = (current_values - returns) ** 2
                v_loss2 = (value_clipped - returns) ** 2
                value_loss = np.mean(np.maximum(v_loss1, v_loss2))
            else:
                value_loss = np.mean((current_values - returns) ** 2)

            self._update_value(states, returns)

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        return np.mean(policy_losses), np.mean(value_losses), np.mean(clip_fractions)

    def _update_policy(self, states, actions, advantages, old_probs):
        """Policy gradient update with clipping."""
        batch_size = len(states)

        # Forward pass
        z1 = states @ self.policy_W1 + self.policy_b1
        h = np.tanh(z1)
        logits = h @ self.policy_W2 + self.policy_b2

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Compute ratio
        action_probs = probs[np.arange(batch_size), actions]
        ratio = action_probs / (old_probs + 1e-10)

        # Gradient of clipped objective
        # Simplified: just use policy gradient weighted by clipped ratio
        clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Use minimum of clipped and unclipped
        use_clipped = (clipped_ratio * advantages < ratio * advantages)
        effective_ratio = np.where(use_clipped, clipped_ratio, ratio)

        # Policy gradient
        one_hot = np.zeros((batch_size, self.n_actions))
        one_hot[np.arange(batch_size), actions] = 1

        d_logits = (one_hot - probs) * (effective_ratio * advantages).reshape(-1, 1)

        # Add entropy gradient
        d_logits += self.entropy_coef * (1.0 / self.n_actions - probs)

        # Backprop
        dW2 = h.T @ d_logits
        db2 = np.sum(d_logits, axis=0)

        dh = d_logits @ self.policy_W2.T
        dz1 = dh * (1 - np.tanh(z1)**2)

        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update (gradient ascent)
        self.policy_W2 += self.policy_lr * dW2 / batch_size
        self.policy_b2 += self.policy_lr * db2 / batch_size
        self.policy_W1 += self.policy_lr * dW1 / batch_size
        self.policy_b1 += self.policy_lr * db1 / batch_size

    def _update_value(self, states, returns):
        """Value function update."""
        batch_size = len(states)

        # Forward
        z1 = states @ self.value_W1 + self.value_b1
        h = np.tanh(z1)
        v = (h @ self.value_W2 + self.value_b2).flatten()

        # Gradient
        dv = 2 * (v - returns).reshape(-1, 1)

        dW2 = h.T @ dv
        db2 = np.sum(dv, axis=0)

        dh = dv @ self.value_W2.T
        dz1 = dh * (1 - np.tanh(z1)**2)

        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update (gradient descent)
        self.value_W2 -= self.value_lr * dW2 / batch_size
        self.value_b2 -= self.value_lr * db2 / batch_size
        self.value_W1 -= self.value_lr * dW1 / batch_size
        self.value_b1 -= self.value_lr * db1 / batch_size


def state_to_vector(state, grid_size):
    """Convert grid position to one-hot vector."""
    vec = np.zeros(grid_size * grid_size)
    idx = state[0] * grid_size + state[1]
    vec[idx] = 1.0
    return vec


def train_ppo(env, agent, n_iterations=100, steps_per_iteration=128):
    """Train PPO agent."""
    episode_rewards = []
    policy_losses = []
    value_losses = []
    clip_fractions = []

    total_reward = 0
    episode_reward = 0
    state = env.reset()
    state_vec = state_to_vector(state, env.size)

    for iteration in range(n_iterations):
        # Collect batch of experience
        states, actions, rewards, next_states, dones = [], [], [], [], []
        old_probs = []
        old_values = []

        for step in range(steps_per_iteration):
            action, prob = agent.select_action(state_vec)
            value = agent.get_value(state_vec.reshape(1, -1))[0]

            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)

            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state_vec)
            dones.append(done)
            old_probs.append(prob)
            old_values.append(value)

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()
                state_vec = state_to_vector(state, env.size)
            else:
                state = next_state
                state_vec = next_state_vec

        # Compute advantages and returns
        states_arr = np.array(states)
        next_states_arr = np.array(next_states)
        rewards_arr = np.array(rewards)
        dones_arr = np.array(dones, dtype=float)

        values = agent.get_value(states_arr)
        next_values = agent.get_value(next_states_arr)

        advantages, returns = agent.compute_gae(rewards_arr, values, next_values, dones_arr)

        # PPO update
        p_loss, v_loss, clip_frac = agent.update(
            states, actions, old_probs, advantages, returns, values
        )

        policy_losses.append(p_loss)
        value_losses.append(v_loss)
        clip_fractions.append(clip_frac)

    return episode_rewards, policy_losses, value_losses, clip_fractions


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_ppo():
    """Create comprehensive PPO visualization."""
    print("\n" + "="*60)
    print("PPO VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # ============ Plot 1: Learning Curve ============
    ax1 = fig.add_subplot(2, 3, 1)

    agent = PPOAgent(state_dim, n_actions, hidden_dim=32, clip_ratio=0.2)
    rewards, _, _, _ = train_ppo(env, agent, n_iterations=200)

    if rewards:
        window = min(20, len(rewards)//2) if len(rewards) > 40 else 5
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, 'b-', linewidth=2)
        else:
            ax1.plot(rewards, 'b-', linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Learning Curve')
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Clip Ratio Effect ============
    ax2 = fig.add_subplot(2, 3, 2)

    clip_ratios = [0.1, 0.2, 0.3, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(clip_ratios)))

    for clip_ratio, color in zip(clip_ratios, colors):
        env = GridWorld(size=5)
        agent = PPOAgent(state_dim, n_actions, clip_ratio=clip_ratio)
        rewards, _, _, _ = train_ppo(env, agent, n_iterations=100)

        if rewards and len(rewards) > 10:
            window = min(10, len(rewards)//2)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, color=color, label=f'ε={clip_ratio}', linewidth=1.5)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Effect of Clip Ratio ε\nSmaller = more conservative')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Epochs per Update ============
    ax3 = fig.add_subplot(2, 3, 3)

    epochs_list = [1, 4, 10]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(epochs_list)))

    for epochs, color in zip(epochs_list, colors):
        env = GridWorld(size=5)
        agent = PPOAgent(state_dim, n_actions, epochs_per_update=epochs)
        rewards, _, _, _ = train_ppo(env, agent, n_iterations=100)

        if rewards and len(rewards) > 10:
            window = min(10, len(rewards)//2)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax3.plot(smoothed, color=color, label=f'K={epochs}', linewidth=1.5)

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Effect of Update Epochs K\nMore epochs = more sample efficient')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Clip Fraction Over Training ============
    ax4 = fig.add_subplot(2, 3, 4)

    env = GridWorld(size=5)
    agent = PPOAgent(state_dim, n_actions, clip_ratio=0.2)
    _, _, _, clip_fractions = train_ppo(env, agent, n_iterations=150)

    if clip_fractions:
        ax4.plot(clip_fractions, 'g-', linewidth=1, alpha=0.7)
        window = min(10, len(clip_fractions)//2) if len(clip_fractions) > 20 else 3
        if window > 1:
            smoothed = np.convolve(clip_fractions, np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(clip_fractions)), smoothed, 'b-', linewidth=2)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Clip Fraction')
    ax4.set_title('Clipping During Training\nFraction of ratios clipped')
    ax4.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Target ~0.2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Policy and Value Loss ============
    ax5 = fig.add_subplot(2, 3, 5)

    env = GridWorld(size=5)
    agent = PPOAgent(state_dim, n_actions)
    _, policy_losses, value_losses, _ = train_ppo(env, agent, n_iterations=150)

    if policy_losses and value_losses:
        ax5.plot(policy_losses, 'b-', alpha=0.7, label='Policy Loss')
        ax5.plot(value_losses, 'r-', alpha=0.7, label='Value Loss')

    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Loss')
    ax5.set_title('PPO Losses')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    PPO — Proximal Policy Optimization
    ════════════════════════════════════

    THE KEY IDEA:
    Clip the policy ratio to prevent large updates!

    L = min(r×A, clip(r, 1-ε, 1+ε)×A)

    r = π_new(a|s) / π_old(a|s)

    CLIPPING LOGIC:
    ┌────────────────────────────────┐
    │ A > 0: Want r > 1 (good action)│
    │        But clip at 1+ε        │
    ├────────────────────────────────┤
    │ A < 0: Want r < 1 (bad action) │
    │        But clip at 1-ε        │
    └────────────────────────────────┘

    HYPERPARAMETERS:
    ε (clip): 0.1-0.3 (how far to move)
    K (epochs): 3-10 (reuse data)
    λ (GAE): 0.95-0.99 (advantage)

    vs TRPO:
    Simpler (no conjugate gradient)
    Similar performance
    Industry standard!
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('PPO — Proximal Policy Optimization\n'
                 'Stable policy updates via clipping',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for PPO."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # 1. Clip ratio
    print("\n1. EFFECT OF CLIP RATIO ε")
    print("-" * 40)

    for clip_ratio in [0.1, 0.2, 0.3, 0.5]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = PPOAgent(state_dim, n_actions, clip_ratio=clip_ratio)
            rewards, _, _, _ = train_ppo(env, agent, n_iterations=80)
            if rewards:
                rewards_list.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))

        if rewards_list:
            print(f"ε={clip_ratio:.1f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ ε=0.2 is standard choice")

    # 2. Epochs per update
    print("\n2. EFFECT OF UPDATE EPOCHS K")
    print("-" * 40)

    for epochs in [1, 4, 10, 20]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = PPOAgent(state_dim, n_actions, epochs_per_update=epochs)
            rewards, _, _, _ = train_ppo(env, agent, n_iterations=80)
            if rewards:
                rewards_list.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))

        if rewards_list:
            print(f"K={epochs:<3}  final_reward={np.mean(rewards_list):.3f}")

    print("→ K=4-10 typically good balance")

    # 3. GAE lambda
    print("\n3. EFFECT OF GAE LAMBDA")
    print("-" * 40)

    for gae_lambda in [0.9, 0.95, 0.99, 1.0]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = PPOAgent(state_dim, n_actions, gae_lambda=gae_lambda)
            rewards, _, _, _ = train_ppo(env, agent, n_iterations=80)
            if rewards:
                rewards_list.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))

        if rewards_list:
            print(f"λ={gae_lambda:.2f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ λ=0.95 usually works well")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("PPO — Proximal Policy Optimization")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_ppo()
    save_path = '/Users/sid47/ML Algorithms/30_ppo.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. PPO: Clip policy ratio to prevent large updates
2. L = min(r×A, clip(r, 1-ε, 1+ε)×A)
3. Simpler than TRPO, similar performance
4. Can reuse data for multiple epochs (sample efficient)
5. Key hyperparams: ε (clip), K (epochs), λ (GAE)
6. Industry standard for many RL applications!
    """)
