"""
Multi-Agent RL — Learning in Multi-Player Settings
===================================================

Paradigm: GAME THEORY + REINFORCEMENT LEARNING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

Multiple agents learning SIMULTANEOUSLY in shared environment.
Each agent's optimal policy DEPENDS on what others do!

SINGLE-AGENT RL:
    Agent ↔ Environment
    Learn: π(a|s) to maximize R

MULTI-AGENT RL:
    Agent₁ ↔ Environment ↔ Agent₂
    Learn: π₁(a|s), π₂(a|s) where each affects the other

THE GAME-THEORETIC VIEW:
    Each agent is a player in a game.
    Solution: Nash Equilibrium (no one wants to deviate)

===============================================================
SETTINGS
===============================================================

1. COOPERATIVE
   - Agents share a common goal
   - Same reward function for all
   - Example: Robot team coordination

2. COMPETITIVE (Zero-Sum)
   - One agent's gain = another's loss
   - Σ rewards = 0
   - Example: Chess, Go, Poker

3. MIXED (General-Sum)
   - Both cooperation and competition
   - Most realistic setting
   - Example: Negotiation, traffic

===============================================================
THE NON-STATIONARITY PROBLEM
===============================================================

From Agent A's perspective:
- Environment includes Agent B
- Agent B is LEARNING (changing policy)
- Environment appears NON-STATIONARY!

    Agent A learns → Agent B adapts → Agent A's policy now suboptimal

This violates the basic RL assumption of stationary MDP!

===============================================================
APPROACHES
===============================================================

1. INDEPENDENT LEARNERS
   Each agent ignores others, treats them as environment.
   Simple but unstable (non-stationarity).

2. CENTRALIZED TRAINING, DECENTRALIZED EXECUTION (CTDE)
   Train: Use global information (all observations/actions)
   Execute: Each agent uses only local observation
   Examples: MADDPG, QMIX

3. SELF-PLAY
   Agent plays against copies of itself.
   How AlphaGo, AlphaZero were trained.

4. OPPONENT MODELING
   Explicitly model other agents' policies.
   Adapt to predicted opponent behavior.

===============================================================
KEY CONCEPTS
===============================================================

NASH EQUILIBRIUM:
    No agent can improve by changing their policy alone.
    π₁*, π₂* are Nash if:
    V₁(π₁*, π₂*) ≥ V₁(π₁, π₂*) for all π₁
    V₂(π₁*, π₂*) ≥ V₂(π₁*, π₂) for all π₂

PARETO OPTIMALITY:
    No way to make one agent better without making another worse.
    Important for cooperative settings.

===============================================================
INDUCTIVE BIAS
===============================================================

1. OTHER AGENTS: Environment includes learning entities
2. GAME THEORY: Equilibrium concepts matter
3. COMMUNICATION: Agents may need to coordinate
4. CREDIT ASSIGNMENT: Who contributed to team reward?

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class MatrixGame:
    """
    Simple matrix game (normal form game).

    Each player chooses an action, payoffs determined by matrix.
    """

    def __init__(self, payoff_matrix):
        """
        payoff_matrix: (n_actions_1, n_actions_2, 2)
        payoff_matrix[a1, a2, 0] = reward for player 1
        payoff_matrix[a1, a2, 1] = reward for player 2
        """
        self.payoff_matrix = payoff_matrix
        self.n_actions_1 = payoff_matrix.shape[0]
        self.n_actions_2 = payoff_matrix.shape[1]

    def step(self, action_1, action_2):
        """
        Both players take actions simultaneously.
        Returns: (reward_1, reward_2)
        """
        r1 = self.payoff_matrix[action_1, action_2, 0]
        r2 = self.payoff_matrix[action_1, action_2, 1]
        return r1, r2


class GridWorldMultiAgent:
    """
    Multi-agent gridworld.

    Two agents navigate, can cooperate or compete.
    """

    def __init__(self, size=5, cooperative=True):
        self.size = size
        self.cooperative = cooperative
        self.n_actions = 5  # Up, Down, Left, Right, Stay
        self.reset()

    def reset(self):
        """Reset agents to random positions."""
        # Agent 1 starts top-left region
        self.pos_1 = (np.random.randint(0, 2), np.random.randint(0, 2))
        # Agent 2 starts bottom-right region
        self.pos_2 = (np.random.randint(self.size-2, self.size),
                      np.random.randint(self.size-2, self.size))
        # Goal in center
        self.goal = (self.size // 2, self.size // 2)
        return self._get_state()

    def _get_state(self):
        """Return state as tuple."""
        return (self.pos_1, self.pos_2)

    def _move(self, pos, action):
        """Move agent according to action."""
        row, col = pos
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)
        # action == 4: Stay
        return (row, col)

    def step(self, action_1, action_2):
        """
        Both agents take actions.
        Returns: (state, (reward_1, reward_2), done)
        """
        # Move both agents
        new_pos_1 = self._move(self.pos_1, action_1)
        new_pos_2 = self._move(self.pos_2, action_2)

        # Check collision
        if new_pos_1 == new_pos_2:
            # Agents collide - neither moves
            pass
        else:
            self.pos_1 = new_pos_1
            self.pos_2 = new_pos_2

        # Compute rewards
        if self.cooperative:
            # Cooperative: both get reward if EITHER reaches goal
            if self.pos_1 == self.goal or self.pos_2 == self.goal:
                r1 = r2 = 10.0
                done = True
            else:
                r1 = r2 = -0.1
                done = False
        else:
            # Competitive: first to goal wins
            if self.pos_1 == self.goal and self.pos_2 == self.goal:
                r1 = r2 = 0.0  # Tie
                done = True
            elif self.pos_1 == self.goal:
                r1 = 10.0
                r2 = -10.0
                done = True
            elif self.pos_2 == self.goal:
                r1 = -10.0
                r2 = 10.0
                done = True
            else:
                r1 = r2 = -0.1
                done = False

        return self._get_state(), (r1, r2), done


class IndependentQLearner:
    """
    Independent Q-learner for multi-agent setting.

    Each agent learns independently, treating other agents as part of environment.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table (state → action values)
        self.Q = {}

    def _get_q(self, state):
        """Get Q-values for state (initialize if needed)."""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def select_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self._get_q(state))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        target = reward
        if not done:
            target += self.gamma * np.max(self._get_q(next_state))

        q_values = self._get_q(state)
        q_values[action] += self.alpha * (target - q_values[action])


class JointActionLearner:
    """
    Joint Action Learner (JAL) for two-player matrix games.

    Maintains beliefs about opponent, best responds to expected opponent.
    """

    def __init__(self, n_actions_self, n_actions_opponent, alpha=0.1, epsilon=0.1):
        self.n_actions_self = n_actions_self
        self.n_actions_opp = n_actions_opponent
        self.alpha = alpha
        self.epsilon = epsilon

        # Joint Q-table: Q[my_action, opponent_action]
        self.Q = np.zeros((n_actions_self, n_actions_opponent))

        # Belief about opponent (count of opponent actions)
        self.opponent_counts = np.ones(n_actions_opponent)  # Initialize uniform

    def get_opponent_belief(self):
        """Return belief distribution over opponent actions."""
        return self.opponent_counts / np.sum(self.opponent_counts)

    def select_action(self):
        """Select action that best responds to expected opponent."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions_self)

        # Compute expected Q for each action (under belief about opponent)
        belief = self.get_opponent_belief()
        expected_q = self.Q @ belief
        return np.argmax(expected_q)

    def update(self, my_action, opponent_action, reward):
        """Update Q-value and opponent belief."""
        # Update Q
        self.Q[my_action, opponent_action] += self.alpha * (
            reward - self.Q[my_action, opponent_action]
        )

        # Update opponent belief
        self.opponent_counts[opponent_action] += 1


class SelfPlay:
    """
    Self-play training for competitive games.

    Agent plays against copies of itself.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.agent = IndependentQLearner(n_states, n_actions, alpha, gamma, epsilon)

    def train_episode(self, env):
        """Train by playing against self."""
        state = env.reset()

        # Agent plays both sides
        total_reward = 0
        steps = 0

        while True:
            # Both use same Q-table but separate actions
            action_1 = self.agent.select_action(state)
            action_2 = self.agent.select_action(state)  # Same policy

            next_state, (r1, r2), done = env.step(action_1, action_2)

            # Update from both perspectives
            self.agent.update(state, action_1, r1, next_state, done)
            self.agent.update(state, action_2, r2, next_state, done)

            total_reward += r1 + r2
            steps += 1
            state = next_state

            if done or steps > 100:
                break

        return total_reward


# ============================================================
# CLASSIC GAMES
# ============================================================

def prisoners_dilemma():
    """
    Classic Prisoner's Dilemma game.

    Actions: 0 = Cooperate, 1 = Defect
    """
    # Payoff matrix: (action_1, action_2) → (reward_1, reward_2)
    payoff = np.array([
        [[3, 3], [0, 5]],  # Player 1 cooperates
        [[5, 0], [1, 1]]   # Player 1 defects
    ], dtype=float)

    return MatrixGame(payoff)


def matching_pennies():
    """
    Matching Pennies - zero-sum game.

    Actions: 0 = Heads, 1 = Tails
    """
    payoff = np.array([
        [[1, -1], [-1, 1]],   # Player 1 shows Heads
        [[-1, 1], [1, -1]]    # Player 1 shows Tails
    ], dtype=float)

    return MatrixGame(payoff)


def coordination_game():
    """
    Coordination game - pure cooperation.

    Actions: 0 = Option A, 1 = Option B
    """
    payoff = np.array([
        [[2, 2], [0, 0]],    # Both choose A = good
        [[0, 0], [1, 1]]     # Both choose B = ok
    ], dtype=float)

    return MatrixGame(payoff)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_multi_agent_rl():
    """
    Comprehensive multi-agent RL visualization.
    """
    print("\n" + "="*60)
    print("MULTI-AGENT RL VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))
    np.random.seed(42)

    # ============ Plot 1: Prisoner's Dilemma Learning ============
    ax1 = fig.add_subplot(2, 3, 1)

    game = prisoners_dilemma()

    # Two JAL agents
    agent_1 = JointActionLearner(2, 2, alpha=0.05, epsilon=0.1)
    agent_2 = JointActionLearner(2, 2, alpha=0.05, epsilon=0.1)

    coop_rates_1 = []
    coop_rates_2 = []

    for episode in range(500):
        a1 = agent_1.select_action()
        a2 = agent_2.select_action()

        r1, r2 = game.step(a1, a2)

        agent_1.update(a1, a2, r1)
        agent_2.update(a2, a1, r2)

        # Track cooperation rate
        coop_rates_1.append(agent_1.opponent_counts[0] / np.sum(agent_1.opponent_counts))
        coop_rates_2.append(agent_2.opponent_counts[0] / np.sum(agent_2.opponent_counts))

    ax1.plot(coop_rates_1, 'b-', alpha=0.7, label='Agent 1 belief (opp cooperates)')
    ax1.plot(coop_rates_2, 'r-', alpha=0.7, label='Agent 2 belief (opp cooperates)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('P(Cooperate)')
    ax1.set_title("Prisoner's Dilemma\nAgents learn to defect (Nash)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============ Plot 2: Cooperative Gridworld ============
    ax2 = fig.add_subplot(2, 3, 2)

    env = GridWorldMultiAgent(size=5, cooperative=True)

    # Train two independent learners
    agent_1 = IndependentQLearner(n_states=1000, n_actions=5)
    agent_2 = IndependentQLearner(n_states=1000, n_actions=5)

    rewards_1 = []
    rewards_2 = []

    for episode in range(200):
        state = env.reset()
        ep_reward_1 = 0
        ep_reward_2 = 0

        for step in range(50):
            a1 = agent_1.select_action(state)
            a2 = agent_2.select_action(state)

            next_state, (r1, r2), done = env.step(a1, a2)

            agent_1.update(state, a1, r1, next_state, done)
            agent_2.update(state, a2, r2, next_state, done)

            ep_reward_1 += r1
            ep_reward_2 += r2
            state = next_state

            if done:
                break

        rewards_1.append(ep_reward_1)
        rewards_2.append(ep_reward_2)

    # Smooth
    window = 20
    smooth_1 = np.convolve(rewards_1, np.ones(window)/window, mode='valid')
    smooth_2 = np.convolve(rewards_2, np.ones(window)/window, mode='valid')

    ax2.plot(smooth_1, 'b-', linewidth=2, label='Agent 1')
    ax2.plot(smooth_2, 'r-', linewidth=2, label='Agent 2')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Cooperative Gridworld\nBoth agents learn to reach goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Competitive Gridworld ============
    ax3 = fig.add_subplot(2, 3, 3)

    env = GridWorldMultiAgent(size=5, cooperative=False)

    agent_1 = IndependentQLearner(n_states=1000, n_actions=5)
    agent_2 = IndependentQLearner(n_states=1000, n_actions=5)

    rewards_1 = []
    rewards_2 = []

    for episode in range(200):
        state = env.reset()
        ep_reward_1 = 0
        ep_reward_2 = 0

        for step in range(50):
            a1 = agent_1.select_action(state)
            a2 = agent_2.select_action(state)

            next_state, (r1, r2), done = env.step(a1, a2)

            agent_1.update(state, a1, r1, next_state, done)
            agent_2.update(state, a2, r2, next_state, done)

            ep_reward_1 += r1
            ep_reward_2 += r2
            state = next_state

            if done:
                break

        rewards_1.append(ep_reward_1)
        rewards_2.append(ep_reward_2)

    smooth_1 = np.convolve(rewards_1, np.ones(window)/window, mode='valid')
    smooth_2 = np.convolve(rewards_2, np.ones(window)/window, mode='valid')

    ax3.plot(smooth_1, 'b-', linewidth=2, label='Agent 1')
    ax3.plot(smooth_2, 'r-', linewidth=2, label='Agent 2')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Competitive Gridworld\nZero-sum: one wins, one loses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Self-Play Learning ============
    ax4 = fig.add_subplot(2, 3, 4)

    env = GridWorldMultiAgent(size=5, cooperative=False)
    self_play = SelfPlay(n_states=1000, n_actions=5)

    rewards = []
    for episode in range(200):
        r = self_play.train_episode(env)
        rewards.append(r)

    smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax4.plot(smooth, 'g-', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Combined Reward')
    ax4.set_title('Self-Play Training\nAgent learns by playing itself')
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Game Theory Matrix ============
    ax5 = fig.add_subplot(2, 3, 5)

    # Show payoff matrix for Prisoner's Dilemma
    game = prisoners_dilemma()
    payoffs = game.payoff_matrix

    # Create annotated heatmap
    data = payoffs[:, :, 0]  # Player 1's payoffs
    im = ax5.imshow(data, cmap='RdYlGn', aspect='equal')

    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['Cooperate', 'Defect'])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Cooperate', 'Defect'])
    ax5.set_xlabel('Player 2')
    ax5.set_ylabel('Player 1')

    # Annotate
    for i in range(2):
        for j in range(2):
            text = f'({payoffs[i,j,0]:.0f}, {payoffs[i,j,1]:.0f})'
            ax5.text(j, i, text, ha='center', va='center', fontsize=12)

    ax5.set_title("Prisoner's Dilemma Payoffs\n(Player 1, Player 2)")
    plt.colorbar(im, ax=ax5, label="Player 1's Payoff")

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    Multi-Agent RL
    ══════════════════════════════

    THE KEY IDEA:
    Multiple agents learning simultaneously!

    Each agent's optimal policy depends on others.

    SETTINGS:
    ┌────────────────────────────┐
    │ COOPERATIVE                │
    │   Shared reward            │
    │   Common goal              │
    ├────────────────────────────┤
    │ COMPETITIVE (Zero-Sum)     │
    │   One's gain = other's loss│
    ├────────────────────────────┤
    │ MIXED (General-Sum)        │
    │   Both cooperation and     │
    │   competition              │
    └────────────────────────────┘

    APPROACHES:
    • Independent Learners
    • Joint Action Learners
    • Self-Play
    • CTDE (centralized train)

    KEY CHALLENGE:
    Non-stationarity!
    (Other agents are learning too)
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Multi-Agent RL — Learning in Multi-Player Settings\n'
                 'Game Theory + Reinforcement Learning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments for multi-agent RL."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)

    # 1. Matrix game equilibria
    print("\n1. MATRIX GAME CONVERGENCE")
    print("-" * 40)

    games = [
        ("Prisoner's Dilemma", prisoners_dilemma()),
        ("Matching Pennies", matching_pennies()),
        ("Coordination", coordination_game())
    ]

    for name, game in games:
        agent_1 = JointActionLearner(2, 2, alpha=0.05, epsilon=0.1)
        agent_2 = JointActionLearner(2, 2, alpha=0.05, epsilon=0.1)

        for _ in range(1000):
            a1 = agent_1.select_action()
            a2 = agent_2.select_action()
            r1, r2 = game.step(a1, a2)
            agent_1.update(a1, a2, r1)
            agent_2.update(a2, a1, r2)

        # Final action probabilities
        p1_coop = agent_1.get_opponent_belief()[0]
        print(f"{name:<20}  P(cooperate)={p1_coop:.2f}")

    # 2. Cooperative vs Competitive
    print("\n2. COOPERATIVE vs COMPETITIVE")
    print("-" * 40)

    for cooperative in [True, False]:
        env = GridWorldMultiAgent(size=5, cooperative=cooperative)
        agent_1 = IndependentQLearner(n_states=1000, n_actions=5)
        agent_2 = IndependentQLearner(n_states=1000, n_actions=5)

        final_rewards = []
        for episode in range(100):
            state = env.reset()
            ep_reward = 0

            for step in range(50):
                a1 = agent_1.select_action(state)
                a2 = agent_2.select_action(state)
                next_state, (r1, r2), done = env.step(a1, a2)
                agent_1.update(state, a1, r1, next_state, done)
                agent_2.update(state, a2, r2, next_state, done)
                ep_reward += r1 + r2
                state = next_state
                if done:
                    break

            if episode >= 80:
                final_rewards.append(ep_reward)

        setting = "Cooperative" if cooperative else "Competitive"
        print(f"{setting:<12}  avg_combined_reward={np.mean(final_rewards):.2f}")

    # 3. Learning rate sensitivity
    print("\n3. LEARNING RATE SENSITIVITY")
    print("-" * 40)

    for alpha in [0.01, 0.05, 0.1, 0.3]:
        env = GridWorldMultiAgent(size=5, cooperative=True)
        agent_1 = IndependentQLearner(n_states=1000, n_actions=5, alpha=alpha)
        agent_2 = IndependentQLearner(n_states=1000, n_actions=5, alpha=alpha)

        final_rewards = []
        for episode in range(100):
            state = env.reset()
            ep_reward = 0

            for step in range(50):
                a1 = agent_1.select_action(state)
                a2 = agent_2.select_action(state)
                next_state, (r1, r2), done = env.step(a1, a2)
                agent_1.update(state, a1, r1, next_state, done)
                agent_2.update(state, a2, r2, next_state, done)
                ep_reward += r1 + r2
                state = next_state
                if done:
                    break

            if episode >= 80:
                final_rewards.append(ep_reward)

        print(f"α={alpha:.2f}  avg_reward={np.mean(final_rewards):.2f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("Multi-Agent RL — Learning in Multi-Player Settings")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_multi_agent_rl()
    save_path = '/Users/sid47/ML Algorithms/33_multi_agent_rl.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. Multi-Agent RL: Multiple agents learning in shared environment
2. Settings: Cooperative, Competitive (zero-sum), Mixed
3. Key challenge: Non-stationarity (others are learning too!)
4. Nash Equilibrium: No agent wants to deviate
5. Approaches: Independent, Joint Action, Self-Play, CTDE
6. Game theory concepts (Nash, Pareto) become important
7. Communication and coordination can help
    """)
