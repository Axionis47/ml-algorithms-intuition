"""
REINFORCEMENT LEARNING FUNDAMENTALS
====================================

Paradigm: SEQUENTIAL DECISION MAKING

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

An AGENT interacts with an ENVIRONMENT:

    State s_t → Action a_t → Reward r_t → Next State s_{t+1}

Goal: Learn a POLICY π(a|s) that maximizes CUMULATIVE reward.

THE BELLMAN EQUATION (the key insight):
    V(s) = max_a [R(s,a) + γ × V(s')]

    "The value of a state is the best immediate reward plus
     the discounted value of where you end up."

This recursive structure enables DYNAMIC PROGRAMMING!

===============================================================
KEY CONCEPTS
===============================================================

1. POLICY π(a|s)
   - Probability of taking action a in state s
   - What we want to learn!

2. VALUE FUNCTION V(s)
   - Expected return starting from state s
   - V(s) = E[Σ γ^t r_t | s_0 = s]

3. Q-FUNCTION Q(s,a)
   - Expected return from (state, action) pair
   - Q(s,a) = R(s,a) + γ × E[V(s')]

4. REWARD r_t
   - Immediate signal (good/bad)

5. RETURN G_t
   - Cumulative discounted reward: Σ γ^k r_{t+k}

6. DISCOUNT γ ∈ [0,1]
   - How much we care about future
   - γ=0: myopic, γ=1: far-sighted

===============================================================
EXPLORATION vs EXPLOITATION
===============================================================

The fundamental RL dilemma:
- EXPLOIT: Do what you know works (greedy)
- EXPLORE: Try new things (might find better)

STRATEGIES:
1. ε-greedy: With probability ε, take random action
2. UCB: Optimism in face of uncertainty
3. Thompson Sampling: Sample from belief, act greedily
4. Softmax/Boltzmann: Actions weighted by estimated value

===============================================================
INDUCTIVE BIAS
===============================================================

1. MARKOV ASSUMPTION
   - Future depends only on current state
   - P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, ..., s_t, a_0, ..., a_t)

2. STATIONARITY
   - Environment dynamics don't change over time

3. REWARD HYPOTHESIS
   - All goals can be described by maximizing expected cumulative reward

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# ENVIRONMENTS
# ============================================================

class GridWorld:
    """
    Simple navigation grid with walls and goal.

    The agent must navigate from start to goal while avoiding walls.
    This is the "Hello World" of RL environments.

    Grid encoding:
        0 = empty (can walk)
        1 = wall (blocked)
        2 = goal (terminal, +1 reward)
        3 = pit (terminal, -1 reward)
    """

    def __init__(self, size=5, walls=None, goal=None, pits=None, start=None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

        # Default configuration
        if walls is None:
            walls = [(1, 1), (2, 1), (1, 3), (3, 3)]
        if goal is None:
            goal = (size-1, size-1)
        if pits is None:
            pits = [(2, 3)]
        if start is None:
            start = (0, 0)

        self.start = start
        self.goal = goal
        self.pits = pits

        # Set up grid
        for w in walls:
            if 0 <= w[0] < size and 0 <= w[1] < size:
                self.grid[w] = 1
        self.grid[goal] = 2
        for p in pits:
            if 0 <= p[0] < size and 0 <= p[1] < size:
                self.grid[p] = 3

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4
        self.action_names = ['↑', '→', '↓', '←']
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.state = None
        self.reset()

    def reset(self):
        """Reset to start position."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take action, return (next_state, reward, done, info).

        THE ENVIRONMENT DYNAMICS:
        - Move in direction if not blocked
        - Get reward based on new cell
        - Episode ends at goal or pit
        """
        row, col = self.state
        dr, dc = self.action_deltas[action]
        new_row, new_col = row + dr, col + dc

        # Check boundaries and walls
        if (0 <= new_row < self.size and
            0 <= new_col < self.size and
            self.grid[new_row, new_col] != 1):
            self.state = (new_row, new_col)

        # Determine reward and termination
        cell = self.grid[self.state]

        if cell == 2:  # Goal
            return self.state, 1.0, True, {'terminal': 'goal'}
        elif cell == 3:  # Pit
            return self.state, -1.0, True, {'terminal': 'pit'}
        else:
            return self.state, -0.01, False, {}  # Small negative for each step

    def get_all_states(self):
        """Return all non-wall states."""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] != 1:
                    states.append((i, j))
        return states

    def is_terminal(self, state):
        """Check if state is terminal."""
        return self.grid[state] in [2, 3]

    def render(self, V=None, policy=None, ax=None):
        """Visualize the grid, optionally with values or policy."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Draw grid
        display = np.zeros((self.size, self.size, 3))

        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                if cell == 1:  # Wall
                    display[i, j] = [0.3, 0.3, 0.3]
                elif cell == 2:  # Goal
                    display[i, j] = [0, 0.8, 0]
                elif cell == 3:  # Pit
                    display[i, j] = [0.8, 0, 0]
                else:  # Empty
                    if V is not None:
                        # Color by value
                        v = V.get((i, j), 0)
                        display[i, j] = [0.5 - 0.5*v, 0.5 + 0.5*v, 0.5]
                    else:
                        display[i, j] = [0.9, 0.9, 0.9]

        ax.imshow(display)

        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Draw values or policy
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:
                    ax.text(j, i, '█', ha='center', va='center', fontsize=20)
                elif self.grid[i, j] == 2:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
                elif self.grid[i, j] == 3:
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
                elif V is not None:
                    ax.text(j, i, f'{V.get((i,j), 0):.2f}', ha='center', va='center', fontsize=10)
                elif policy is not None:
                    action = policy.get((i, j), 0)
                    ax.text(j, i, self.action_names[action], ha='center', va='center', fontsize=16)

        # Mark current position
        if self.state:
            ax.plot(self.state[1], self.state[0], 'bo', markersize=20, alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        return ax


class MultiArmedBandit:
    """
    K-armed bandit for exploration experiments.

    THE SIMPLEST RL PROBLEM:
    - No states (or single state)
    - Just pick arms and get rewards
    - Goal: Find the best arm

    This isolates the exploration-exploitation tradeoff.
    """

    def __init__(self, k=10, reward_type='gaussian'):
        self.k = k
        self.reward_type = reward_type

        # True arm values (hidden from agent)
        if reward_type == 'gaussian':
            self.q_star = np.random.randn(k)
        elif reward_type == 'bernoulli':
            self.q_star = np.random.rand(k)

        self.optimal_action = np.argmax(self.q_star)
        self.reset()

    def reset(self):
        """Reset statistics."""
        self.total_reward = 0
        self.n_steps = 0
        self.action_counts = np.zeros(self.k)
        return 0  # Single state

    def step(self, action):
        """
        Pull arm, get reward.

        Returns: (state, reward, done, info)
        """
        if self.reward_type == 'gaussian':
            reward = self.q_star[action] + np.random.randn()
        else:  # Bernoulli
            reward = float(np.random.rand() < self.q_star[action])

        self.action_counts[action] += 1
        self.total_reward += reward
        self.n_steps += 1

        is_optimal = (action == self.optimal_action)

        return 0, reward, False, {'optimal': is_optimal}


class CliffWalking:
    """
    Classic cliff walking environment.

    Start at bottom-left, goal at bottom-right.
    Cliff along the bottom edge - falling gives -100.
    Each step costs -1.

    This environment highlights the difference between
    safe (SARSA) and risky (Q-learning) policies.
    """

    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.start = (height-1, 0)
        self.goal = (height-1, width-1)

        # Cliff is the bottom row except start and goal
        self.cliff = [(height-1, j) for j in range(1, width-1)]

        self.n_actions = 4
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.state = None
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        dr, dc = self.action_deltas[action]
        new_row = np.clip(row + dr, 0, self.height - 1)
        new_col = np.clip(col + dc, 0, self.width - 1)

        self.state = (new_row, new_col)

        if self.state in self.cliff:
            # Fell off cliff!
            self.state = self.start
            return self.state, -100, False, {'fell': True}
        elif self.state == self.goal:
            return self.state, 0, True, {}
        else:
            return self.state, -1, False, {}


# ============================================================
# EXPLORATION STRATEGIES
# ============================================================

class EpsilonGreedy:
    """
    ε-greedy exploration.

    With probability ε: random action
    With probability 1-ε: greedy action

    Simple but effective baseline.
    """

    def __init__(self, epsilon=0.1, decay=1.0):
        self.epsilon = epsilon
        self.decay = decay

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def update(self):
        self.epsilon *= self.decay


class UCB:
    """
    Upper Confidence Bound exploration.

    THE UCB FORMULA:
        UCB(a) = Q(a) + c × √(log(t) / N(a))

    "Optimism in the face of uncertainty"
    - Explores actions with high uncertainty
    - Balances exploration and exploitation automatically
    """

    def __init__(self, c=2.0):
        self.c = c
        self.counts = None
        self.t = 0

    def select_action(self, q_values):
        n_actions = len(q_values)

        if self.counts is None:
            self.counts = np.zeros(n_actions)

        self.t += 1

        # Try each action once first
        if np.min(self.counts) == 0:
            return np.argmin(self.counts)

        # UCB formula
        ucb_values = q_values + self.c * np.sqrt(np.log(self.t) / self.counts)
        action = np.argmax(ucb_values)
        self.counts[action] += 1

        return action


class Softmax:
    """
    Softmax/Boltzmann exploration.

    P(a) ∝ exp(Q(a) / τ)

    τ = temperature:
    - High τ: uniform (explore)
    - Low τ: greedy (exploit)
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select_action(self, q_values):
        # Numerical stability
        q_shifted = q_values - np.max(q_values)
        exp_q = np.exp(q_shifted / self.temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)


# ============================================================
# VALUE ITERATION (Dynamic Programming)
# ============================================================

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    VALUE ITERATION ALGORITHM

    THE BELLMAN OPTIMALITY EQUATION:
        V*(s) = max_a [R(s,a) + γ × Σ P(s'|s,a) × V*(s')]

    For deterministic environments (P=1 for one s'):
        V*(s) = max_a [R(s,a) + γ × V*(s')]

    ALGORITHM:
    1. Initialize V(s) = 0 for all s
    2. Repeat until convergence:
       V(s) ← max_a [R(s,a) + γ × V(s')]
    3. Extract policy: π(s) = argmax_a [R(s,a) + γ × V(s')]

    GUARANTEED TO CONVERGE to optimal V* and π*!
    """
    states = env.get_all_states()
    V = {s: 0.0 for s in states}

    history = [dict(V)]

    for iteration in range(max_iterations):
        delta = 0

        for s in states:
            if env.is_terminal(s):
                continue

            old_v = V[s]

            # Find best action value
            action_values = []
            for a in range(env.n_actions):
                env.state = s
                s_next, r, done, _ = env.step(a)
                action_values.append(r + gamma * V[s_next])

            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))

        history.append(dict(V))

        if delta < theta:
            print(f"Value iteration converged in {iteration+1} iterations")
            break

    # Extract policy
    policy = {}
    for s in states:
        if env.is_terminal(s):
            policy[s] = 0
            continue

        action_values = []
        for a in range(env.n_actions):
            env.state = s
            s_next, r, done, _ = env.step(a)
            action_values.append(r + gamma * V[s_next])

        policy[s] = np.argmax(action_values)

    env.reset()
    return V, policy, history


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """
    POLICY ITERATION ALGORITHM

    Two steps:
    1. POLICY EVALUATION: Given π, compute V^π
    2. POLICY IMPROVEMENT: Given V^π, improve π

    Repeat until policy doesn't change.

    Often converges faster than value iteration!
    """
    states = env.get_all_states()

    # Initialize random policy
    policy = {s: np.random.randint(env.n_actions) for s in states}

    iterations = 0
    while True:
        # 1. POLICY EVALUATION
        V = {s: 0.0 for s in states}

        while True:
            delta = 0
            for s in states:
                if env.is_terminal(s):
                    continue

                old_v = V[s]
                a = policy[s]
                env.state = s
                s_next, r, done, _ = env.step(a)
                V[s] = r + gamma * V[s_next]
                delta = max(delta, abs(old_v - V[s]))

            if delta < theta:
                break

        # 2. POLICY IMPROVEMENT
        policy_stable = True

        for s in states:
            if env.is_terminal(s):
                continue

            old_action = policy[s]

            action_values = []
            for a in range(env.n_actions):
                env.state = s
                s_next, r, done, _ = env.step(a)
                action_values.append(r + gamma * V[s_next])

            policy[s] = np.argmax(action_values)

            if old_action != policy[s]:
                policy_stable = False

        iterations += 1

        if policy_stable:
            print(f"Policy iteration converged in {iterations} iterations")
            break

    env.reset()
    return V, policy


# ============================================================
# MONTE CARLO METHODS
# ============================================================

def monte_carlo_prediction(env, policy, gamma=0.99, n_episodes=1000):
    """
    MONTE CARLO VALUE ESTIMATION

    THE KEY INSIGHT:
        V(s) = E[G_t | S_t = s]

    We estimate this by SAMPLING:
    1. Generate episodes using policy
    2. For each state visited, record the return
    3. Average returns for each state

    First-visit MC: Only count first occurrence in episode
    Every-visit MC: Count all occurrences
    """
    V = defaultdict(float)
    returns = defaultdict(list)

    for episode in range(n_episodes):
        # Generate episode
        trajectory = []
        state = env.reset()
        done = False

        while not done:
            action = policy.get(state, np.random.randint(env.n_actions))
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        # Compute returns (backward)
        G = 0
        visited = set()

        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            G = gamma * G + reward

            # First-visit MC
            if state not in visited:
                visited.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])

    return dict(V)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_rl_fundamentals():
    """
    Create comprehensive RL fundamentals visualization:
    1. GridWorld and value iteration
    2. Exploration strategies comparison
    3. Bellman backup visualization
    4. Discount factor effect
    """
    print("\n" + "="*60)
    print("RL FUNDAMENTALS VISUALIZATION")
    print("="*60)

    fig = plt.figure(figsize=(16, 12))

    # ============ Plot 1: GridWorld Value Iteration ============
    ax1 = fig.add_subplot(2, 3, 1)

    env = GridWorld(size=5)
    V, policy, history = value_iteration(env, gamma=0.99)
    env.render(V=V, ax=ax1)
    ax1.set_title('GridWorld: Learned Values\nBrighter = Higher Value')

    # ============ Plot 2: Learned Policy ============
    ax2 = fig.add_subplot(2, 3, 2)

    env.reset()
    env.render(policy=policy, ax=ax2)
    ax2.set_title('GridWorld: Optimal Policy\nArrows show best action')

    # ============ Plot 3: Value Iteration Convergence ============
    ax3 = fig.add_subplot(2, 3, 3)

    # Track value of start state over iterations
    start_values = [h.get(env.start, 0) for h in history]
    ax3.plot(start_values, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('V(start)')
    ax3.set_title('Value Iteration Convergence\nStart state value over time')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: Exploration Strategies on Bandit ============
    ax4 = fig.add_subplot(2, 3, 4)

    np.random.seed(42)
    n_steps = 1000
    n_runs = 50

    strategies = {
        'ε-greedy (ε=0.1)': lambda: EpsilonGreedy(epsilon=0.1),
        'ε-greedy (ε=0.01)': lambda: EpsilonGreedy(epsilon=0.01),
        'UCB (c=2)': lambda: UCB(c=2),
        'Softmax (τ=0.5)': lambda: Softmax(temperature=0.5),
    }

    results = {name: np.zeros((n_runs, n_steps)) for name in strategies}

    for run in range(n_runs):
        bandit = MultiArmedBandit(k=10)

        for name, strategy_fn in strategies.items():
            strategy = strategy_fn()
            Q = np.zeros(bandit.k)
            N = np.zeros(bandit.k)

            bandit.reset()

            for step in range(n_steps):
                if name.startswith('UCB'):
                    action = strategy.select_action(Q)
                else:
                    action = strategy.select_action(Q)

                _, reward, _, info = bandit.step(action)

                # Update Q estimate
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]

                results[name][run, step] = info['optimal']

    # Plot average optimal action percentage
    for name, data in results.items():
        mean_optimal = np.mean(data, axis=0)
        # Smooth with moving average
        window = 50
        smoothed = np.convolve(mean_optimal, np.ones(window)/window, mode='valid')
        ax4.plot(smoothed, label=name, linewidth=1.5)

    ax4.set_xlabel('Step')
    ax4.set_ylabel('% Optimal Action')
    ax4.set_title('Exploration Strategies on 10-Armed Bandit\nHigher = Better')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Discount Factor Effect ============
    ax5 = fig.add_subplot(2, 3, 5)

    gammas = [0.5, 0.9, 0.99, 0.999]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

    for gamma, color in zip(gammas, colors):
        env = GridWorld(size=5)
        V, _, _ = value_iteration(env, gamma=gamma, max_iterations=500)

        # Get values as array
        values = [V.get(env.start, 0)]
        ax5.axhline(V.get(env.start, 0), color=color, linestyle='--', alpha=0.7,
                   label=f'γ={gamma}: V(start)={V.get(env.start, 0):.2f}')

    ax5.set_xlabel('Discount Factor γ')
    ax5.set_ylabel('V(start)')
    ax5.set_title('Effect of Discount Factor\nHigher γ = More farsighted')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ============ Plot 6: Summary ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """
    REINFORCEMENT LEARNING FUNDAMENTALS
    ════════════════════════════════════

    THE BELLMAN EQUATION:
    V(s) = max_a [R(s,a) + γ·V(s')]

    "Value = Best(Reward + Future)"

    KEY TRADEOFFS:
    ┌─────────────────────────────────┐
    │ Explore    ←→    Exploit       │
    │ (try new)       (use known)    │
    │                                 │
    │ Myopic     ←→    Farsighted    │
    │ (γ small)        (γ large)     │
    │                                 │
    │ Model-free ←→    Model-based   │
    │ (learn Q)        (learn P)     │
    └─────────────────────────────────┘

    VALUE ITERATION:
    Guaranteed to find optimal policy!

    EXPLORATION:
    ε-greedy: simple, effective
    UCB: principled, no hyperparameter
    """

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('REINFORCEMENT LEARNING — Sequential Decision Making\n'
                 'State → Action → Reward → Learn from experience',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    """Run ablation experiments to understand RL fundamentals."""

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    # 1. Discount factor effect
    print("\n1. EFFECT OF DISCOUNT FACTOR γ")
    print("-" * 40)

    for gamma in [0.5, 0.9, 0.99, 0.999]:
        env = GridWorld(size=5)
        V, policy, history = value_iteration(env, gamma=gamma)
        print(f"γ={gamma:.3f}  V(start)={V[env.start]:.3f}  iterations={len(history)}")

    print("→ Higher γ = higher values (future matters more)")
    print("→ γ≈1 may converge slower")

    # 2. Grid size effect
    print("\n2. EFFECT OF GRID SIZE")
    print("-" * 40)

    for size in [4, 6, 8, 10]:
        env = GridWorld(size=size, goal=(size-1, size-1))
        V, policy, history = value_iteration(env, gamma=0.99)
        n_states = len(env.get_all_states())
        print(f"size={size}  states={n_states}  iterations={len(history)}  V(start)={V[env.start]:.3f}")

    print("→ Larger grid = more states = slower convergence")
    print("→ Value of start decreases (farther from goal)")

    # 3. Exploration comparison on bandit
    print("\n3. EXPLORATION STRATEGIES (1000 steps, 10-armed bandit)")
    print("-" * 40)

    np.random.seed(42)
    n_steps = 1000

    strategies = [
        ('ε-greedy (ε=0.1)', EpsilonGreedy(epsilon=0.1)),
        ('ε-greedy (ε=0.01)', EpsilonGreedy(epsilon=0.01)),
        ('UCB (c=2)', UCB(c=2)),
        ('Softmax (τ=1)', Softmax(temperature=1.0)),
    ]

    for name, strategy in strategies:
        bandit = MultiArmedBandit(k=10)
        Q = np.zeros(bandit.k)
        N = np.zeros(bandit.k)

        total_reward = 0
        n_optimal = 0

        for step in range(n_steps):
            action = strategy.select_action(Q)
            _, reward, _, info = bandit.step(action)

            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            total_reward += reward
            n_optimal += info['optimal']

        print(f"{name:<25} total_reward={total_reward:>7.1f}  optimal%={100*n_optimal/n_steps:.1f}%")

    print("→ UCB often best for stationary problems")
    print("→ ε-greedy simple and competitive")

    # 4. Policy vs Value iteration
    print("\n4. VALUE ITERATION vs POLICY ITERATION")
    print("-" * 40)

    env = GridWorld(size=6)

    import time

    start = time.time()
    V_vi, policy_vi, history_vi = value_iteration(env, gamma=0.99)
    time_vi = time.time() - start

    start = time.time()
    V_pi, policy_pi = policy_iteration(env, gamma=0.99)
    time_pi = time.time() - start

    # Check if policies match
    policy_match = all(policy_vi[s] == policy_pi[s] for s in env.get_all_states() if not env.is_terminal(s))

    print(f"Value Iteration:  {len(history_vi)} iterations, {time_vi*1000:.1f}ms")
    print(f"Policy Iteration: converged, {time_pi*1000:.1f}ms")
    print(f"Same policy: {policy_match}")
    print("→ Both find optimal policy!")
    print("→ Policy iteration often faster (fewer total operations)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("REINFORCEMENT LEARNING FUNDAMENTALS")
    print("="*60)

    print(__doc__)

    # Run experiments
    ablation_experiments()

    # Create visualization
    fig = visualize_rl_fundamentals()
    save_path = '/Users/sid47/ML Algorithms/25_rl_fundamentals.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. RL = Learning from interaction with environment
2. Bellman equation: V(s) = max_a [R(s,a) + γ·V(s')]
3. Value iteration: guaranteed to find optimal policy
4. Exploration vs exploitation: fundamental tradeoff
5. Discount γ: controls myopic vs farsighted behavior
6. Markov assumption: future depends only on present
    """)
