"""
ACTOR-CRITIC — Combining Value and Policy Learning
===================================================

Paradigm: VALUE + POLICY (The Best of Both Worlds)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Combine the ACTOR (policy) and CRITIC (value function)!

ACTOR: Policy network π_θ(a|s)
   "What action should I take?"

CRITIC: Value network V_φ(s)
   "How good is this state?"

THE KEY INSIGHT:
Use the critic to reduce variance of policy gradient!

REINFORCE gradient:
   ∇J = E[∇log π(a|s) × G_t]     (high variance!)

Actor-Critic gradient:
   ∇J = E[∇log π(a|s) × A(s,a)]  (lower variance!)

Where A(s,a) = ADVANTAGE = "How much better than average?"

===============================================================
THE ADVANTAGE FUNCTION
===============================================================

A(s,a) = Q(s,a) - V(s)
       = "Value of taking action a" - "Average value from s"
       = "How much BETTER is this action than average?"

TD ESTIMATE OF ADVANTAGE:
   A(s,a) ≈ r + γV(s') - V(s)
          = TD error δ

This is the ONE-STEP advantage estimate.
No need to wait for episode end!

===============================================================
A2C: ADVANTAGE ACTOR-CRITIC
===============================================================

The synchronous version of A3C (Asynchronous).

UPDATE RULES:

ACTOR (policy):
   θ ← θ + α_π × ∇log π_θ(a|s) × A(s,a)
   "Increase probability of better-than-average actions"

CRITIC (value):
   φ ← φ - α_v × ∇(r + γV(s') - V_φ(s))²
   "Make value predictions more accurate"

===============================================================
N-STEP RETURNS
===============================================================

One-step TD: High bias, low variance
   A = r + γV(s') - V(s)

Monte Carlo: Low bias, high variance
   A = G_t - V(s)

N-STEP: Trade-off!
   A^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n}) - V(s_t)

Typical n = 5 to 20 steps.

===============================================================
GENERALIZED ADVANTAGE ESTIMATION (GAE)
===============================================================

Exponentially-weighted average of n-step advantages:

   A^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

Where:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
- λ ∈ [0,1] controls bias-variance trade-off

λ = 0: One-step TD (high bias)
λ = 1: Monte Carlo (high variance)
λ ≈ 0.95: Good balance (used in practice)

===============================================================
ENTROPY REGULARIZATION
===============================================================

Add entropy bonus to encourage exploration:

   L = L_policy + c_1 × L_value - c_2 × H(π)

H(π) = -Σ π(a|s) log π(a|s)

High entropy = more random = more exploration
Prevents premature convergence to deterministic policy.

===============================================================
INDUCTIVE BIAS
===============================================================

1. BOOTSTRAPPING
   - Uses value estimates (not full returns)
   - More sample efficient, introduces bias

2. ON-POLICY
   - Must use current policy's data
   - Can't directly reuse old experience

3. FUNCTION APPROXIMATION
   - Assumes similar states have similar values/policies

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')

from importlib import import_module
rl_fund = import_module('25_rl_fundamentals')
GridWorld = rl_fund.GridWorld


class ActorNetwork:
    """
    Actor: Policy network π_θ(a|s)

    Outputs action probabilities.
    """

    def __init__(self, state_dim, n_actions, hidden_dim=32):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Initialize weights
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + n_actions))

        self.W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * scale2
        self.b2 = np.zeros(n_actions)

        self.cache = {}

    def forward(self, state):
        """Compute action probabilities."""
        z1 = state @ self.W1 + self.b1
        h = np.tanh(z1)
        logits = h @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        self.cache = {'state': state, 'h': h, 'probs': probs}
        return probs

    def sample_action(self, state):
        """Sample action from policy."""
        probs = self.forward(state.reshape(1, -1))[0]
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def get_entropy(self, probs):
        """Compute entropy of policy."""
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    def update(self, states, actions, advantages, learning_rate=0.01,
               entropy_coef=0.01):
        """
        Update actor using policy gradient with entropy bonus.
        """
        batch_size = len(states)
        probs = self.forward(states)

        # Policy gradient
        one_hot = np.zeros((batch_size, self.n_actions))
        one_hot[np.arange(batch_size), actions] = 1

        # d_logits = (one_hot - probs) × advantage + entropy_grad
        d_logits = (one_hot - probs) * advantages.reshape(-1, 1)

        # Add entropy gradient (encourages exploration)
        # d/d_logits H(π) = -log(π) - 1, but simplified: just add noise toward uniform
        entropy_grad = entropy_coef * (1.0 / self.n_actions - probs)
        d_logits += entropy_grad

        # Backprop
        h = self.cache['h']
        z1 = states @ self.W1 + self.b1

        dW2 = h.T @ d_logits
        db2 = np.sum(d_logits, axis=0)

        dh = d_logits @ self.W2.T
        dz1 = dh * (1 - np.tanh(z1)**2)

        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update (gradient ascent)
        self.W2 += learning_rate * dW2 / batch_size
        self.b2 += learning_rate * db2 / batch_size
        self.W1 += learning_rate * dW1 / batch_size
        self.b1 += learning_rate * db1 / batch_size

        return np.mean(self.get_entropy(probs))


class CriticNetwork:
    """
    Critic: Value network V_φ(s)

    Estimates state value.
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
        """Predict state values."""
        z1 = states @ self.W1 + self.b1
        h = np.tanh(z1)
        v = h @ self.W2 + self.b2
        self.cache = {'states': states, 'z1': z1, 'h': h}
        return v.flatten()

    def update(self, states, targets, learning_rate=0.01):
        """
        Update critic to minimize MSE with targets.

        Target = r + γV(s') for TD
        """
        v_pred = self.forward(states)
        td_error = targets - v_pred

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


class ActorCritic:
    """
    A2C: Advantage Actor-Critic

    THE ALGORITHM:
    1. Collect trajectory using current policy
    2. Compute advantages A = r + γV(s') - V(s)
    3. Update critic: minimize (target - V(s))²
    4. Update actor: maximize E[log π(a|s) × A]
    """

    def __init__(self, state_dim, n_actions, hidden_dim=32,
                 gamma=0.99, actor_lr=0.01, critic_lr=0.01,
                 entropy_coef=0.01, n_steps=5, use_gae=False, gae_lambda=0.95):

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(state_dim, n_actions, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

    def select_action(self, state):
        """Sample action from policy."""
        action, _ = self.actor.sample_action(state)
        return action

    def compute_advantages_td(self, rewards, values, next_values, dones):
        """
        Compute TD advantages: A = r + γV(s') - V(s)
        """
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        return advantages

    def compute_advantages_gae(self, rewards, values, next_values, dones):
        """
        Compute GAE advantages.

        A^GAE_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0

        for t in range(T - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update actor and critic from batch of transitions.
        """
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=float)

        # Get value estimates
        values = self.critic.forward(states)
        next_values = self.critic.forward(next_states)

        # Compute advantages
        if self.use_gae:
            advantages = self.compute_advantages_gae(rewards, values, next_values, dones)
        else:
            advantages = self.compute_advantages_td(rewards, values, next_values, dones)

        # Compute targets for critic
        targets = rewards + self.gamma * next_values * (1 - dones)

        # Update critic
        critic_loss = self.critic.update(states, targets, self.critic_lr)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update actor
        entropy = self.actor.update(states, actions, advantages,
                                    self.actor_lr, self.entropy_coef)

        return critic_loss, entropy


def state_to_vector(state, grid_size):
    """Convert grid position to one-hot vector."""
    vec = np.zeros(grid_size * grid_size)
    idx = state[0] * grid_size + state[1]
    vec[idx] = 1.0
    return vec


def train_actor_critic(env, agent, n_episodes=500, max_steps=100):
    """Train Actor-Critic agent."""
    episode_rewards = []
    critic_losses = []
    entropies = []

    for episode in range(n_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        state = env.reset()
        state_vec = state_to_vector(state, env.size)
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = state_to_vector(next_state, env.size)

            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state_vec)
            dones.append(done)

            total_reward += reward
            state = next_state
            state_vec = next_state_vec

            # Update every n_steps or at episode end
            if len(states) >= agent.n_steps or done:
                c_loss, ent = agent.update(states, actions, rewards, next_states, dones)
                critic_losses.append(c_loss)
                entropies.append(ent)
                states, actions, rewards, next_states, dones = [], [], [], [], []

            if done:
                break

        episode_rewards.append(total_reward)

    return episode_rewards, critic_losses, entropies


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_actor_critic():
    """
    Create comprehensive Actor-Critic visualization.
    """
    print("\n" + "="*60)
    print("ACTOR-CRITIC VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # ============ Plot 1: Learning Curve ============
    ax1 = fig.add_subplot(2, 3, 1)

    agent = ActorCritic(state_dim, n_actions, hidden_dim=32,
                        gamma=0.99, n_steps=5, use_gae=True)
    rewards, _, _ = train_actor_critic(env, agent, n_episodes=500)

    window = 30
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('A2C Learning Curve\nActor-Critic with GAE')
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: TD vs GAE ============
    ax2 = fig.add_subplot(2, 3, 2)

    n_runs = 3
    n_episodes = 300

    # TD advantage
    rewards_td = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = ActorCritic(state_dim, n_actions, use_gae=False, n_steps=1)
        r, _, _ = train_actor_critic(env, agent, n_episodes=n_episodes)
        rewards_td.append(r)

    # GAE advantage
    rewards_gae = []
    for _ in range(n_runs):
        env = GridWorld(size=5)
        agent = ActorCritic(state_dim, n_actions, use_gae=True, gae_lambda=0.95)
        r, _, _ = train_actor_critic(env, agent, n_episodes=n_episodes)
        rewards_gae.append(r)

    td_mean = np.mean(rewards_td, axis=0)
    gae_mean = np.mean(rewards_gae, axis=0)

    window = 20
    ax2.plot(np.convolve(td_mean, np.ones(window)/window, mode='valid'),
             'r--', label='TD (1-step)', linewidth=2)
    ax2.plot(np.convolve(gae_mean, np.ones(window)/window, mode='valid'),
             'b-', label='GAE (λ=0.95)', linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('TD vs GAE Advantage\nGAE balances bias-variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: N-step Effect ============
    ax3 = fig.add_subplot(2, 3, 3)

    n_steps_list = [1, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_steps_list)))

    for n_steps, color in zip(n_steps_list, colors):
        env = GridWorld(size=5)
        agent = ActorCritic(state_dim, n_actions, n_steps=n_steps, use_gae=False)
        r, _, _ = train_actor_critic(env, agent, n_episodes=300)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax3.plot(smoothed, color=color, label=f'n={n_steps}', linewidth=1.5)

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Effect of N-step Returns\nn=5-10 often good')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Entropy Over Training ============
    ax4 = fig.add_subplot(2, 3, 4)

    env = GridWorld(size=5)
    agent = ActorCritic(state_dim, n_actions, entropy_coef=0.01)
    _, _, entropies = train_actor_critic(env, agent, n_episodes=400)

    if entropies:
        window = 50
        smoothed_ent = np.convolve(entropies, np.ones(window)/window, mode='valid')
        ax4.plot(smoothed_ent, 'g-', linewidth=2)
    ax4.set_xlabel('Update Step')
    ax4.set_ylabel('Policy Entropy')
    ax4.set_title('Entropy During Training\nDecreases as policy becomes certain')
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Critic Loss ============
    ax5 = fig.add_subplot(2, 3, 5)

    env = GridWorld(size=5)
    agent = ActorCritic(state_dim, n_actions)
    _, critic_losses, _ = train_actor_critic(env, agent, n_episodes=400)

    if critic_losses:
        window = 50
        smoothed_loss = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        ax5.plot(smoothed_loss, 'b-', linewidth=1)
    ax5.set_xlabel('Update Step')
    ax5.set_ylabel('Critic Loss (MSE)')
    ax5.set_title('Critic Loss During Training\nValue estimates improve')
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    ACTOR-CRITIC — Value + Policy
    ══════════════════════════════

    ACTOR (Policy): π_θ(a|s)
    CRITIC (Value): V_φ(s)

    ADVANTAGE:
    A(s,a) = r + γV(s') - V(s)
    "How much better than average?"

    UPDATES:
    ┌─────────────────────────────┐
    │ Actor:  θ += α∇logπ × A    │
    │ Critic: φ -= α∇(V - target)²│
    └─────────────────────────────┘

    GAE (λ):
    A^GAE = Σ (γλ)^l δ_{t+l}

    λ=0: TD (high bias)
    λ=1: MC (high variance)
    λ≈0.95: Good balance

    ENTROPY BONUS:
    Encourages exploration
    Prevents premature convergence
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('ACTOR-CRITIC — A2C\n'
                 'Combine policy gradient with value function baseline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for Actor-Critic."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    env = GridWorld(size=5)
    state_dim = env.size * env.size
    n_actions = env.n_actions

    # 1. GAE lambda
    print("\n1. EFFECT OF GAE LAMBDA")
    print("-" * 40)

    for gae_lambda in [0.0, 0.5, 0.9, 0.95, 1.0]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = ActorCritic(state_dim, n_actions, use_gae=True,
                               gae_lambda=gae_lambda)
            r, _, _ = train_actor_critic(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"λ={gae_lambda:.2f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ λ≈0.95 often works best")

    # 2. N-steps
    print("\n2. EFFECT OF N-STEPS")
    print("-" * 40)

    for n_steps in [1, 5, 10, 20, 50]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = ActorCritic(state_dim, n_actions, n_steps=n_steps, use_gae=False)
            r, _, _ = train_actor_critic(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"n_steps={n_steps:<3}  final_reward={np.mean(rewards_list):.3f}")

    print("→ n=5-10 typically good")

    # 3. Entropy coefficient
    print("\n3. EFFECT OF ENTROPY COEFFICIENT")
    print("-" * 40)

    for ent_coef in [0.0, 0.001, 0.01, 0.1]:
        rewards_list = []
        for _ in range(3):
            env = GridWorld(size=5)
            agent = ActorCritic(state_dim, n_actions, entropy_coef=ent_coef)
            r, _, _ = train_actor_critic(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"entropy_coef={ent_coef:.3f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Small entropy bonus helps exploration")

    # 4. Actor vs Critic learning rate ratio
    print("\n4. EFFECT OF ACTOR/CRITIC LR RATIO")
    print("-" * 40)

    for ratio in [0.1, 0.5, 1.0, 2.0]:
        rewards_list = []
        critic_lr = 0.01
        actor_lr = critic_lr * ratio
        for _ in range(3):
            env = GridWorld(size=5)
            agent = ActorCritic(state_dim, n_actions,
                               actor_lr=actor_lr, critic_lr=critic_lr)
            r, _, _ = train_actor_critic(env, agent, n_episodes=200)
            rewards_list.append(np.mean(r[-30:]))
        print(f"actor_lr/critic_lr={ratio:.1f}  final_reward={np.mean(rewards_list):.3f}")

    print("→ Critic often needs higher LR (faster value learning)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("ACTOR-CRITIC — A2C")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_actor_critic()
    save_path = '/Users/sid47/ML Algorithms/29_actor_critic.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Actor-Critic: Combine policy (actor) + value (critic)
2. Advantage: A = r + γV(s') - V(s)
3. Critic reduces variance of policy gradient
4. GAE: Exponentially-weighted n-step advantages
5. Entropy bonus encourages exploration
6. Key hyperparams: n-steps, GAE λ, LR ratio
    """)
