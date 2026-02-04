"""
Inverse RL — Learning Rewards from Behavior
============================================

Paradigm: INFER REWARD FUNCTION FROM EXPERT DEMONSTRATIONS

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Standard RL: Given reward function, learn optimal policy
    R → π*

Inverse RL: Given expert behavior, learn the reward function
    π* → R

WHY?
- Reward engineering is HARD
- Experts demonstrate WHAT to do, not HOW to reward
- Sometimes easier to show than to specify

EXAMPLE:
    Instead of defining a reward for "drive safely",
    observe expert drivers and learn what they optimize for.

===============================================================
THE AMBIGUITY PROBLEM
===============================================================

Many reward functions explain the same behavior!

- R(s) = 0 for all s → Any policy is optimal
- R(s) = c (constant) → Any policy is optimal

We need additional constraints or assumptions to find
the "right" reward function.

===============================================================
MAXIMUM ENTROPY IRL
===============================================================

PRINCIPLE:
Expert is optimal w.r.t. some reward,
but also maximizes ENTROPY (random tie-breaking).

P(trajectory τ) ∝ exp(Σ_t r(s_t, a_t))

"Expert trajectory probability is proportional to
 exponential of cumulative reward"

This gives us a principled way to learn R!

ALGORITHM:
1. Parameterize reward: r_θ(s, a)
2. Maximize likelihood of expert demonstrations:
   θ* = argmax Σ log P(τ_expert | r_θ)

===============================================================
BEHAVIORAL CLONING
===============================================================

Simplest approach: Just imitate the actions!

    π(a|s) = supervised learning on (s, a) pairs

PROBLEM: Covariate shift
    - Expert only visits good states
    - Learner might visit bad states
    - No training data for bad states → mistakes compound

===============================================================
GAIL — Generative Adversarial Imitation Learning
===============================================================

GAN for imitation learning!

GENERATOR: Policy π_θ (tries to match expert)
DISCRIMINATOR: D_φ (distinguishes expert vs policy)

    max_φ E_π[log D(s,a)] + E_expert[log(1-D(s,a))]
    min_θ E_π[log D(s,a)]

Policy learns to fool the discriminator.
No explicit reward recovery — directly imitates!

INTUITION:
- Discriminator learns what makes expert behavior special
- Generator (policy) learns to produce expert-like behavior
- Converges when discriminator can't tell them apart

===============================================================
INDUCTIVE BIAS
===============================================================

1. EXPERT OPTIMALITY: Expert behavior is (approximately) optimal
2. REWARD STRUCTURE: Reward is function of state (or state-action)
3. RATIONALITY: Expert's actions are purposeful
4. TRANSFERABILITY: Learned reward generalizes

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnv:
    """Simple gridworld for IRL experiments."""

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Down, Left, Right
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        return self.pos[0] * self.size + self.pos[1]

    def _get_pos(self, state):
        return (state // self.size, state % self.size)

    def step(self, action):
        row, col = self.pos

        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)

        self.pos = (row, col)
        done = self.pos == self.goal
        reward = 10.0 if done else -0.1

        return self._get_state(), reward, done


def get_expert_policy(env, gamma=0.99):
    """
    Compute optimal policy using value iteration.
    This serves as our "expert".
    """
    n_states = env.n_states
    n_actions = env.n_actions

    # Transition dynamics (deterministic)
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            env.pos = env._get_pos(s)
            ns, r, _ = env.step(a)
            P[s, a, ns] = 1.0
            R[s, a] = r
            env.pos = env._get_pos(s)  # Reset

    # Value iteration
    V = np.zeros(n_states)
    for _ in range(100):
        V_new = np.max(R + gamma * P @ V, axis=1)
        if np.max(np.abs(V - V_new)) < 1e-6:
            break
        V = V_new

    # Extract policy
    Q = R + gamma * P @ V
    policy = np.argmax(Q, axis=1)

    return policy, V, Q


def generate_expert_demonstrations(env, policy, n_episodes=50, max_steps=100):
    """Generate expert trajectories."""
    demonstrations = []

    for _ in range(n_episodes):
        trajectory = []
        state = env.reset()

        for _ in range(max_steps):
            action = policy[state]
            next_state, reward, done = env.step(action)

            trajectory.append((state, action, reward, next_state))
            state = next_state

            if done:
                break

        demonstrations.append(trajectory)

    return demonstrations


class BehavioralCloning:
    """
    Behavioral Cloning: Supervised learning on expert actions.
    """

    def __init__(self, n_states, n_actions, hidden_dim=32):
        self.n_states = n_states
        self.n_actions = n_actions

        # Simple neural network: state → action probabilities
        scale = np.sqrt(2.0 / n_states)
        self.W1 = np.random.randn(n_states, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * 0.1
        self.b2 = np.zeros(n_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def one_hot(self, state):
        x = np.zeros(self.n_states)
        x[state] = 1.0
        return x

    def forward(self, state):
        """Get action probabilities."""
        x = self.one_hot(state)
        h = self.relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def train(self, demonstrations, epochs=100, lr=0.01):
        """Train on expert demonstrations."""
        # Extract state-action pairs
        states = []
        actions = []
        for traj in demonstrations:
            for s, a, _, _ in traj:
                states.append(s)
                actions.append(a)

        losses = []

        for epoch in range(epochs):
            total_loss = 0

            for s, a in zip(states, actions):
                # Forward pass
                probs = self.forward(s)

                # Cross-entropy loss
                loss = -np.log(probs[a] + 1e-10)
                total_loss += loss

                # Backward pass (simplified)
                x = self.one_hot(s)
                h = self.relu(x @ self.W1 + self.b1)

                # Gradient at output
                d_logits = probs.copy()
                d_logits[a] -= 1  # Softmax gradient

                # Update output layer
                self.W2 -= lr * np.outer(h, d_logits)
                self.b2 -= lr * d_logits

            losses.append(total_loss / len(states))

        return losses

    def get_policy(self):
        """Get deterministic policy."""
        policy = []
        for s in range(self.n_states):
            probs = self.forward(s)
            policy.append(np.argmax(probs))
        return np.array(policy)


class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning.

    Learn reward function from expert demonstrations.
    """

    def __init__(self, n_states, n_actions, n_features=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_features = n_features or n_states

        # Linear reward: r(s) = θ^T φ(s)
        self.theta = np.zeros(self.n_features)

    def get_features(self, state):
        """Get feature vector for state."""
        # Simple: one-hot encoding
        phi = np.zeros(self.n_features)
        if self.n_features == self.n_states:
            phi[state] = 1.0
        else:
            # Or some learned features
            phi = np.random.randn(self.n_features)
        return phi

    def get_reward(self, state):
        """Get reward for state."""
        return np.dot(self.theta, self.get_features(state))

    def compute_feature_expectations(self, demonstrations):
        """Compute expected feature counts from demonstrations."""
        mu = np.zeros(self.n_features)

        total = 0
        for traj in demonstrations:
            for s, _, _, _ in traj:
                mu += self.get_features(s)
                total += 1

        return mu / total

    def train(self, env, demonstrations, epochs=50, lr=0.1, gamma=0.99):
        """
        Train using gradient ascent on log-likelihood.
        """
        # Expert feature expectations
        mu_expert = self.compute_feature_expectations(demonstrations)

        losses = []

        for epoch in range(epochs):
            # Compute policy for current reward
            R = np.array([self.get_reward(s) for s in range(self.n_states)])

            # Solve for optimal policy (simplified: use soft value iteration)
            V = np.zeros(self.n_states)
            for _ in range(50):
                V_new = np.zeros(self.n_states)
                for s in range(self.n_states):
                    q_values = []
                    for a in range(self.n_actions):
                        # Compute next state (deterministic)
                        env.pos = env._get_pos(s)
                        ns, _, _ = env.step(a)
                        env.pos = env._get_pos(s)
                        q = R[s] + gamma * V[ns]
                        q_values.append(q)
                    V_new[s] = np.max(q_values)  # Or soft max
                V = V_new

            # Compute state visitation under current policy
            mu_policy = np.zeros(self.n_features)
            n_samples = 100
            for _ in range(n_samples):
                state = env.reset()
                for _ in range(50):
                    # Greedy action
                    q_values = []
                    for a in range(self.n_actions):
                        env.pos = env._get_pos(state)
                        ns, _, _ = env.step(a)
                        env.pos = env._get_pos(state)
                        q_values.append(R[state] + gamma * V[ns])
                    action = np.argmax(q_values)

                    mu_policy += self.get_features(state)
                    next_state, _, done = env.step(action)
                    state = next_state
                    if done:
                        break

            mu_policy /= n_samples

            # Gradient: ∇L = μ_expert - μ_policy
            gradient = mu_expert - mu_policy

            # Update
            self.theta += lr * gradient

            # Loss (negative log likelihood proxy)
            loss = np.linalg.norm(gradient)
            losses.append(loss)

        return losses


class GAIL:
    """
    Simplified GAIL (Generative Adversarial Imitation Learning).

    Discriminator learns to distinguish expert from learner.
    Policy learns to fool discriminator.
    """

    def __init__(self, n_states, n_actions, hidden_dim=32):
        self.n_states = n_states
        self.n_actions = n_actions

        # Discriminator: (s, a) → [0, 1]
        input_dim = n_states + n_actions
        scale = np.sqrt(2.0 / input_dim)
        self.D_W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.D_b2 = np.zeros(1)

        # Policy: s → action probabilities
        scale = np.sqrt(2.0 / n_states)
        self.P_W1 = np.random.randn(n_states, hidden_dim) * scale
        self.P_b1 = np.zeros(hidden_dim)
        self.P_W2 = np.random.randn(hidden_dim, n_actions) * 0.1
        self.P_b2 = np.zeros(n_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def one_hot_state(self, state):
        x = np.zeros(self.n_states)
        x[state] = 1.0
        return x

    def one_hot_action(self, action):
        x = np.zeros(self.n_actions)
        x[action] = 1.0
        return x

    def discriminator(self, state, action):
        """D(s, a) → probability of being expert."""
        x = np.concatenate([self.one_hot_state(state), self.one_hot_action(action)])
        h = self.relu(x @ self.D_W1 + self.D_b1)
        return self.sigmoid(h @ self.D_W2 + self.D_b2)[0]

    def policy(self, state):
        """π(a|s) → action probabilities."""
        x = self.one_hot_state(state)
        h = self.relu(x @ self.P_W1 + self.P_b1)
        logits = h @ self.P_W2 + self.P_b2
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def train(self, env, demonstrations, epochs=100, lr_d=0.01, lr_p=0.01):
        """Train GAIL."""
        # Extract expert data
        expert_data = []
        for traj in demonstrations:
            for s, a, _, _ in traj:
                expert_data.append((s, a))

        d_losses = []
        p_rewards = []

        for epoch in range(epochs):
            # Generate policy data
            policy_data = []
            state = env.reset()
            for _ in range(len(expert_data)):
                probs = self.policy(state)
                action = np.random.choice(self.n_actions, p=probs)
                policy_data.append((state, action))
                next_state, _, done = env.step(action)
                state = next_state
                if done:
                    state = env.reset()

            # Update discriminator
            d_loss = 0
            for (s_e, a_e), (s_p, a_p) in zip(expert_data[:50], policy_data[:50]):
                # Expert: D should output high
                D_e = self.discriminator(s_e, a_e)
                loss_e = -np.log(D_e + 1e-10)

                # Policy: D should output low
                D_p = self.discriminator(s_p, a_p)
                loss_p = -np.log(1 - D_p + 1e-10)

                d_loss += loss_e + loss_p

                # Simple gradient update
                self.D_b2 += lr_d * (1 - D_e)
                self.D_b2 -= lr_d * D_p

            d_losses.append(d_loss / 100)

            # Update policy (maximize D)
            total_reward = 0
            for s, a in policy_data[:50]:
                reward = np.log(self.discriminator(s, a) + 1e-10)
                total_reward += reward

                # Policy gradient (simplified)
                probs = self.policy(s)
                grad = self.one_hot_action(a) - probs
                self.P_b2 += lr_p * reward * grad

            p_rewards.append(total_reward / 50)

        return d_losses, p_rewards

    def get_policy(self):
        """Get deterministic policy."""
        policy = []
        for s in range(self.n_states):
            probs = self.policy(s)
            policy.append(np.argmax(probs))
        return np.array(policy)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_inverse_rl():
    """Comprehensive Inverse RL visualization."""
    print("\n" + "="*60)
    print("INVERSE RL VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    env = GridWorldEnv(size=5)

    # Get expert policy
    expert_policy, expert_V, expert_Q = get_expert_policy(env)
    demonstrations = generate_expert_demonstrations(env, expert_policy, n_episodes=50)

    # ============ Plot 1: Expert Value Function ============
    ax1 = fig.add_subplot(2, 3, 1)

    V_grid = expert_V.reshape(5, 5)
    im = ax1.imshow(V_grid, cmap='RdYlGn', aspect='equal')
    ax1.set_title('Expert Value Function\nThis is what we want to recover')
    plt.colorbar(im, ax=ax1, label='Value')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # ============ Plot 2: Behavioral Cloning ============
    ax2 = fig.add_subplot(2, 3, 2)

    bc = BehavioralCloning(env.n_states, env.n_actions)
    bc_losses = bc.train(demonstrations, epochs=100)
    bc_policy = bc.get_policy()

    # Compare with expert
    accuracy = np.mean(bc_policy == expert_policy)

    ax2.plot(bc_losses, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Behavioral Cloning\nPolicy accuracy: {accuracy:.0%}')
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Max Entropy IRL ============
    ax3 = fig.add_subplot(2, 3, 3)

    irl = MaxEntIRL(env.n_states, env.n_actions)
    irl_losses = irl.train(env, demonstrations, epochs=30)

    ax3.plot(irl_losses, 'r-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Feature Expectation Diff')
    ax3.set_title('Max Entropy IRL\nLearning reward from demos')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Learned vs True Reward ============
    ax4 = fig.add_subplot(2, 3, 4)

    # True reward (sparse)
    true_R = np.zeros(env.n_states)
    true_R[env.size * env.size - 1] = 10  # Goal state

    # Learned reward
    learned_R = np.array([irl.get_reward(s) for s in range(env.n_states)])

    x = np.arange(env.n_states)
    ax4.bar(x - 0.2, true_R / np.max(np.abs(true_R) + 1e-10), 0.4,
           label='True (normalized)', alpha=0.7)
    ax4.bar(x + 0.2, learned_R / np.max(np.abs(learned_R) + 1e-10), 0.4,
           label='Learned (normalized)', alpha=0.7)
    ax4.set_xlabel('State')
    ax4.set_ylabel('Normalized Reward')
    ax4.set_title('Learned vs True Reward\nShould highlight goal state')
    ax4.legend()

    # ============ Plot 5: GAIL Training ============
    ax5 = fig.add_subplot(2, 3, 5)

    gail = GAIL(env.n_states, env.n_actions)
    d_losses, p_rewards = gail.train(env, demonstrations, epochs=50)

    ax5_twin = ax5.twinx()
    ax5.plot(d_losses, 'b-', linewidth=2, label='D loss')
    ax5_twin.plot(p_rewards, 'r-', linewidth=2, label='Policy reward')

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Discriminator Loss', color='blue')
    ax5_twin.set_ylabel('Policy Reward', color='red')
    ax5.set_title('GAIL Training\nAdversarial imitation learning')

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Inverse RL
    ══════════════════════════════

    THE KEY IDEA:
    Learn REWARD from expert behavior!

    Standard RL: R → π*
    Inverse RL:  π* → R

    APPROACHES:
    ┌────────────────────────────┐
    │ Behavioral Cloning         │
    │   Supervised: (s,a) → π    │
    │   Problem: Covariate shift │
    ├────────────────────────────┤
    │ Max Entropy IRL            │
    │   P(τ) ∝ exp(Σ r(s))       │
    │   Learns reward function   │
    ├────────────────────────────┤
    │ GAIL                       │
    │   GAN-style imitation      │
    │   D: expert vs policy      │
    │   No explicit reward       │
    └────────────────────────────┘

    WHY IRL?
    • Reward engineering is hard
    • Experts show, not tell
    • Transfer to new settings
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Inverse RL — Learning Rewards from Behavior\n'
                 'Observe experts, infer what they optimize',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for Inverse RL."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    env = GridWorldEnv(size=5)
    expert_policy, _, _ = get_expert_policy(env)

    # 1. Number of demonstrations
    print("\n1. EFFECT OF NUMBER OF DEMONSTRATIONS")
    print("-" * 40)

    for n_demos in [5, 10, 25, 50, 100]:
        demos = generate_expert_demonstrations(env, expert_policy, n_episodes=n_demos)

        bc = BehavioralCloning(env.n_states, env.n_actions)
        bc.train(demos, epochs=100)
        bc_policy = bc.get_policy()
        accuracy = np.mean(bc_policy == expert_policy)

        print(f"n_demos={n_demos:<4}  policy_accuracy={accuracy:.2%}")

    print("→ More demonstrations = better imitation")

    # 2. Behavioral Cloning vs Max Entropy IRL
    print("\n2. BC vs MaxEnt IRL")
    print("-" * 40)

    demos = generate_expert_demonstrations(env, expert_policy, n_episodes=50)

    # BC
    bc = BehavioralCloning(env.n_states, env.n_actions)
    bc.train(demos, epochs=100)
    bc_accuracy = np.mean(bc.get_policy() == expert_policy)

    # IRL (evaluate by rolling out learned reward)
    irl = MaxEntIRL(env.n_states, env.n_actions)
    irl.train(env, demos, epochs=30)

    print(f"Behavioral Cloning  accuracy={bc_accuracy:.2%}")
    print(f"Max Entropy IRL     (learns reward, not direct policy)")

    # 3. Expert optimality
    print("\n3. EFFECT OF EXPERT OPTIMALITY")
    print("-" * 40)

    for noise in [0.0, 0.1, 0.2, 0.3]:
        # Create noisy expert
        noisy_policy = expert_policy.copy()
        n_flip = int(len(noisy_policy) * noise)
        flip_idx = np.random.choice(len(noisy_policy), n_flip, replace=False)
        for idx in flip_idx:
            noisy_policy[idx] = np.random.randint(env.n_actions)

        demos = generate_expert_demonstrations(env, noisy_policy, n_episodes=50)

        bc = BehavioralCloning(env.n_states, env.n_actions)
        bc.train(demos, epochs=100)
        bc_accuracy = np.mean(bc.get_policy() == expert_policy)

        print(f"expert_noise={noise:.1f}  imitation_accuracy={bc_accuracy:.2%}")

    print("→ Noisy expert = harder to learn from")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Inverse RL — Learning Rewards from Behavior")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_inverse_rl()
    save_path = '/Users/sid47/ML Algorithms/34_inverse_rl.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Inverse RL: Learn reward function from expert behavior
2. Standard RL: R → π*, Inverse RL: π* → R
3. Behavioral Cloning: Supervised (s,a) learning
4. Max Entropy IRL: P(trajectory) ∝ exp(cumulative reward)
5. GAIL: GAN-style imitation (no explicit reward)
6. Challenge: Reward ambiguity (many R explain same behavior)
7. Applications: Learning from demonstrations, apprenticeship
    """)
