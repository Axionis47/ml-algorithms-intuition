"""
MULTI-ARMED BANDITS — The Exploration vs Exploitation Dilemma

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

You're in a casino with K slot machines (arms). Each arm has an
unknown payout distribution. You have T pulls. How do you maximize
total reward?

THE FUNDAMENTAL DILEMMA:
    - EXPLOIT: Pull the arm that looks best so far
    - EXPLORE: Try other arms (might find a better one!)

If you only exploit → you might miss the best arm forever
If you only explore → you waste pulls on bad arms

This is the SIMPLEST RL setting: no states, no transitions.
Just actions and rewards. The core tension is pure.

===============================================================
THE KEY INSIGHT
===============================================================

Uncertainty is VALUABLE. An arm you haven't tried much might
be amazing. You should try uncertain arms more.

UCB captures this: "optimism in the face of uncertainty"
    Score(a) = estimated_reward(a) + exploration_bonus(a)

Thompson Sampling: Sample from your beliefs, act greedily
    Sample θ ~ Posterior, then pick argmax θ

===============================================================
REGRET — THE PERFORMANCE METRIC
===============================================================

Regret = (Optimal reward) - (Your reward)
       = T × μ* - Σ rewards

Good algorithms have regret growing as O(√T) or O(log T).
Bad algorithms (no exploration) have linear regret O(T).

===============================================================
INDUCTIVE BIAS
===============================================================

- Stationary: arm distributions don't change
- Independent: pulling one arm doesn't affect others
- Parametric beliefs: we assume Gaussian/Beta posteriors

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class BanditEnvironment:
    """Multi-armed bandit with Gaussian rewards."""

    def __init__(self, means: List[float], stds: List[float] = None):
        """
        Initialize bandit.

        means: true mean reward for each arm
        stds: standard deviation for each arm (default: all 1.0)
        """
        self.means = np.array(means)
        self.k = len(means)
        self.stds = np.ones(self.k) if stds is None else np.array(stds)
        self.optimal_arm = np.argmax(self.means)
        self.optimal_mean = self.means[self.optimal_arm]

    def pull(self, arm: int) -> float:
        """Pull an arm and get reward."""
        return np.random.normal(self.means[arm], self.stds[arm])

    def regret(self, arm: int) -> float:
        """Instantaneous regret of pulling this arm."""
        return self.optimal_mean - self.means[arm]


class BanditAgent:
    """Base class for bandit algorithms."""

    def __init__(self, k: int):
        self.k = k
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.k)  # pulls per arm
        self.values = np.zeros(self.k)  # estimated value per arm
        self.t = 0

    def select_arm(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        self.t += 1
        self.counts[arm] += 1
        # Incremental mean update
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class EpsilonGreedy(BanditAgent):
    """
    ε-greedy: With probability ε, explore randomly.

    Simple but effective. The parameter ε controls the
    explore-exploit tradeoff directly.
    """

    def __init__(self, k: int, epsilon: float = 0.1):
        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)  # explore
        else:
            return np.argmax(self.values)  # exploit


class DecayingEpsilonGreedy(BanditAgent):
    """
    ε-greedy with decaying ε.

    Start with high exploration, reduce over time.
    ε_t = min(1, ε_0 / t)
    """

    def __init__(self, k: int, epsilon_0: float = 1.0, decay_rate: float = 0.01):
        super().__init__(k)
        self.epsilon_0 = epsilon_0
        self.decay_rate = decay_rate

    def select_arm(self) -> int:
        epsilon = self.epsilon_0 / (1 + self.decay_rate * self.t)
        if np.random.random() < epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.values)


class UCB(BanditAgent):
    """
    Upper Confidence Bound (UCB1).

    THE KEY IDEA: Optimism in the face of uncertainty.

    Score(a) = Q(a) + c × √(log(t) / N(a))
               ↑        ↑
           estimated   exploration bonus
           reward      (high when N(a) is small)

    Arms we haven't tried much get a bonus.
    As we try them more, bonus shrinks.
    """

    def __init__(self, k: int, c: float = 2.0):
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        # First, try each arm once
        for a in range(self.k):
            if self.counts[a] == 0:
                return a

        # UCB formula
        exploration_bonus = self.c * np.sqrt(np.log(self.t + 1) / self.counts)
        ucb_values = self.values + exploration_bonus
        return np.argmax(ucb_values)

    def get_ucb_values(self) -> np.ndarray:
        """Return current UCB values for visualization."""
        if self.t == 0:
            return np.ones(self.k) * np.inf
        exploration_bonus = self.c * np.sqrt(np.log(self.t + 1) / np.maximum(self.counts, 1))
        return self.values + exploration_bonus


class ThompsonSampling(BanditAgent):
    """
    Thompson Sampling — Bayesian exploration.

    THE KEY IDEA: Probability matching.

    1. Maintain posterior belief about each arm's reward
    2. Sample from each posterior
    3. Pick the arm with highest sample

    For Gaussian rewards with known variance:
        Prior: μ ~ N(0, σ_0²)
        Posterior: μ | data ~ N(μ_n, σ_n²)

    Natural exploration: uncertain arms have high-variance
    posteriors, so sometimes sample high values.
    """

    def __init__(self, k: int, prior_mean: float = 0.0, prior_var: float = 1.0):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        super().__init__(k)
        # Posterior parameters (assuming known reward variance = 1)
        self.posterior_means = np.ones(k) * prior_mean
        self.posterior_vars = np.ones(k) * prior_var

    def reset(self):
        super().reset()
        self.posterior_means = np.ones(self.k) * self.prior_mean
        self.posterior_vars = np.ones(self.k) * self.prior_var

    def select_arm(self) -> int:
        # Sample from each arm's posterior
        samples = np.random.normal(self.posterior_means,
                                   np.sqrt(self.posterior_vars))
        return np.argmax(samples)

    def update(self, arm: int, reward: float):
        super().update(arm, reward)

        # Bayesian update for Gaussian likelihood with known variance (=1)
        # Posterior precision = prior precision + n × observation precision
        prior_precision = 1.0 / self.prior_var
        obs_precision = 1.0  # assuming variance = 1

        n = self.counts[arm]
        posterior_precision = prior_precision + n * obs_precision
        self.posterior_vars[arm] = 1.0 / posterior_precision

        # Posterior mean = precision-weighted average
        self.posterior_means[arm] = (
            prior_precision * self.prior_mean +
            n * obs_precision * self.values[arm]
        ) / posterior_precision


class Greedy(BanditAgent):
    """
    Pure greedy (no exploration).

    Always pick the arm with highest estimated value.
    This is the BASELINE that shows why exploration matters.
    """

    def __init__(self, k: int):
        super().__init__(k)

    def select_arm(self) -> int:
        # Initially random (to avoid all zeros)
        if self.t < self.k:
            return self.t % self.k
        return np.argmax(self.values)


def run_experiment(env: BanditEnvironment, agent: BanditAgent,
                   n_steps: int) -> Dict:
    """Run a single experiment and collect metrics."""
    agent.reset()

    rewards = []
    regrets = []
    arm_history = []
    cumulative_regret = 0

    for _ in range(n_steps):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)

        rewards.append(reward)
        regrets.append(env.regret(arm))
        cumulative_regret += env.regret(arm)
        arm_history.append(arm)

    return {
        'rewards': np.array(rewards),
        'regrets': np.array(regrets),
        'cumulative_regret': np.cumsum(regrets),
        'arm_history': np.array(arm_history),
        'final_counts': agent.counts.copy(),
        'final_values': agent.values.copy()
    }


def run_multiple_experiments(env: BanditEnvironment, agent: BanditAgent,
                            n_steps: int, n_runs: int) -> Dict:
    """Run multiple experiments and average."""
    all_cumulative_regrets = []
    all_arm_histories = []

    for _ in range(n_runs):
        results = run_experiment(env, agent, n_steps)
        all_cumulative_regrets.append(results['cumulative_regret'])
        all_arm_histories.append(results['arm_history'])

    return {
        'mean_cumulative_regret': np.mean(all_cumulative_regrets, axis=0),
        'std_cumulative_regret': np.std(all_cumulative_regrets, axis=0),
        'all_arm_histories': np.array(all_arm_histories)
    }


# =============================================================
# VISUALIZATION FUNCTIONS
# =============================================================

def visualize_regret_comparison(n_steps: int = 1000, n_runs: int = 50):
    """Compare cumulative regret across algorithms."""
    print("\n" + "="*60)
    print("REGRET COMPARISON")
    print("="*60)

    # Setup
    means = [0.1, 0.5, 0.3, 0.7, 0.4]  # Arm 3 is optimal
    env = BanditEnvironment(means)

    agents = {
        'ε-greedy (ε=0.1)': EpsilonGreedy(len(means), epsilon=0.1),
        'ε-greedy (ε=0.01)': EpsilonGreedy(len(means), epsilon=0.01),
        'UCB (c=2)': UCB(len(means), c=2.0),
        'Thompson Sampling': ThompsonSampling(len(means)),
        'Greedy (no exploration)': Greedy(len(means)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    # Run experiments
    results = {}
    for (name, agent), color in zip(agents.items(), colors):
        res = run_multiple_experiments(env, agent, n_steps, n_runs)
        results[name] = res

        # Plot mean regret with confidence band
        mean = res['mean_cumulative_regret']
        std = res['std_cumulative_regret']

        axes[0].plot(mean, label=name, color=color, linewidth=2)
        axes[0].fill_between(range(n_steps), mean - std, mean + std,
                            alpha=0.2, color=color)

    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Cumulative Regret Over Time\n(Lower is better)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Final regret bar chart
    final_regrets = [results[name]['mean_cumulative_regret'][-1]
                     for name in agents.keys()]
    bars = axes[1].bar(range(len(agents)), final_regrets, color=colors)
    axes[1].set_xticks(range(len(agents)))
    axes[1].set_xticklabels([name.split('(')[0].strip() for name in agents.keys()],
                           rotation=45, ha='right')
    axes[1].set_ylabel('Final Cumulative Regret')
    axes[1].set_title(f'Final Regret at T={n_steps}')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for bar, regret in zip(bars, final_regrets):
        axes[1].annotate(f'{regret:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Multi-Armed Bandit: {len(means)} Arms, Optimal Mean = {env.optimal_mean:.1f}',
                fontsize=12, y=1.02)
    plt.tight_layout()

    # Print analysis
    print(f"\nEnvironment: {len(means)} arms with means {means}")
    print(f"Optimal arm: {env.optimal_arm} (mean = {env.optimal_mean})")
    print(f"\nFinal cumulative regret (averaged over {n_runs} runs):")
    for name, regret in zip(agents.keys(), final_regrets):
        print(f"  {name}: {regret:.1f}")

    return fig


def visualize_arm_pulls(n_steps: int = 500):
    """Visualize which arms each algorithm pulls over time."""
    print("\n" + "="*60)
    print("ARM PULL DISTRIBUTION")
    print("="*60)

    means = [0.2, 0.8, 0.5, 0.3]  # Arm 1 is optimal
    env = BanditEnvironment(means)

    agents = {
        'ε-greedy': EpsilonGreedy(len(means), epsilon=0.1),
        'UCB': UCB(len(means), c=2.0),
        'Thompson': ThompsonSampling(len(means)),
        'Greedy': Greedy(len(means)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, agent) in zip(axes, agents.items()):
        results = run_experiment(env, agent, n_steps)

        # Create 2D histogram of arm pulls over time
        arm_history = results['arm_history']
        time_bins = np.arange(0, n_steps + 50, 50)

        # Count pulls per arm per time bin
        pull_counts = np.zeros((len(time_bins)-1, len(means)))
        for i in range(len(time_bins)-1):
            mask = (np.arange(n_steps) >= time_bins[i]) & (np.arange(n_steps) < time_bins[i+1])
            for arm in range(len(means)):
                pull_counts[i, arm] = np.sum(arm_history[mask] == arm)

        # Normalize to percentages
        pull_pcts = pull_counts / pull_counts.sum(axis=1, keepdims=True) * 100

        # Stacked area chart
        x = (time_bins[:-1] + time_bins[1:]) / 2
        colors_arm = plt.cm.Set2(np.linspace(0, 1, len(means)))

        ax.stackplot(x, pull_pcts.T, labels=[f'Arm {i} (μ={m:.1f})'
                     for i, m in enumerate(means)], colors=colors_arm, alpha=0.8)

        ax.axhline(y=100, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(0, n_steps)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('% of Pulls')
        ax.set_title(f'{name}\n(Final pulls: {results["final_counts"].astype(int)})')
        ax.legend(loc='center right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Arm Pull Distribution Over Time\n'
                 f'Optimal: Arm 1 (μ=0.8)', fontsize=12)
    plt.tight_layout()

    return fig


def visualize_ucb_confidence(n_steps: int = 200):
    """Visualize UCB confidence bounds shrinking over time."""
    print("\n" + "="*60)
    print("UCB CONFIDENCE BOUNDS")
    print("="*60)

    means = [0.3, 0.7, 0.5]
    env = BanditEnvironment(means)
    agent = UCB(len(means), c=2.0)

    # Collect UCB values at each step
    ucb_history = []
    value_history = []

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)

        ucb_history.append(agent.get_ucb_values().copy())
        value_history.append(agent.values.copy())

    ucb_history = np.array(ucb_history)
    value_history = np.array(value_history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    # Left: UCB values over time
    for i in range(len(means)):
        axes[0].plot(ucb_history[:, i], color=colors[i],
                    label=f'Arm {i} UCB (true μ={means[i]})', linewidth=2)
        axes[0].plot(value_history[:, i], color=colors[i],
                    linestyle='--', alpha=0.7, label=f'Arm {i} estimate')
        axes[0].axhline(y=means[i], color=colors[i], linestyle=':', alpha=0.5)

    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title('UCB Values Over Time\n(Solid=UCB, Dashed=Estimate, Dotted=True)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.5, 3.5)

    # Right: Confidence bonus shrinking
    exploration_bonuses = ucb_history - value_history

    for i in range(len(means)):
        axes[1].plot(exploration_bonuses[:, i], color=colors[i],
                    label=f'Arm {i}', linewidth=2)

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Exploration Bonus')
    axes[1].set_title('UCB Exploration Bonus Over Time\n(Shrinks as we learn)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('UCB: Optimism in the Face of Uncertainty', fontsize=12)
    plt.tight_layout()

    print("\nKey observation: UCB exploration bonus shrinks as √(log(t)/n)")
    print("Arms pulled less often keep higher bonuses → encourages exploration")

    return fig


def visualize_thompson_posteriors(n_steps: int = 100):
    """Visualize Thompson Sampling posterior evolution."""
    print("\n" + "="*60)
    print("THOMPSON SAMPLING POSTERIORS")
    print("="*60)

    means = [0.3, 0.7, 0.4]
    env = BanditEnvironment(means)
    agent = ThompsonSampling(len(means))

    # Collect posterior at key time points
    checkpoints = [0, 10, 30, 100]
    posteriors = {}

    for t in range(n_steps + 1):
        if t in checkpoints:
            posteriors[t] = {
                'means': agent.posterior_means.copy(),
                'vars': agent.posterior_vars.copy(),
                'counts': agent.counts.copy()
            }

        if t < n_steps:
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    x = np.linspace(-1, 2, 300)

    for ax, t in zip(axes, checkpoints):
        post = posteriors[t]

        for i in range(len(means)):
            # Plot posterior distribution
            y = np.exp(-0.5 * (x - post['means'][i])**2 / post['vars'][i])
            y = y / (np.sqrt(2 * np.pi * post['vars'][i]))
            ax.plot(x, y, color=colors[i], linewidth=2,
                   label=f'Arm {i} (n={int(post["counts"][i])})')
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.3)

            # Mark true mean
            ax.axvline(x=means[i], color=colors[i], linestyle=':', alpha=0.7)

        ax.set_xlabel('Reward')
        ax.set_ylabel('Density')
        ax.set_title(f't = {t}')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Thompson Sampling: Posterior Belief Evolution\n'
                 f'True means: {means}', fontsize=12)
    plt.tight_layout()

    print("\nKey observation: Posteriors concentrate around true means")
    print("Uncertain arms (wide distributions) get sampled more")

    return fig


# =============================================================
# ABLATION EXPERIMENTS
# =============================================================

def ablation_epsilon():
    """Ablation: Effect of ε in ε-greedy."""
    print("\n" + "="*60)
    print("ABLATION: EPSILON IN ε-GREEDY")
    print("="*60)

    means = [0.2, 0.8, 0.5]
    env = BanditEnvironment(means)

    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    n_steps = 1000
    n_runs = 30

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    final_regrets = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))

    for eps, color in zip(epsilons, colors):
        agent = EpsilonGreedy(len(means), epsilon=eps)
        results = run_multiple_experiments(env, agent, n_steps, n_runs)

        axes[0].plot(results['mean_cumulative_regret'],
                    color=color, label=f'ε={eps}', linewidth=2)
        final_regrets.append(results['mean_cumulative_regret'][-1])

    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Cumulative Regret for Different ε')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epsilons, final_regrets, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epsilon (ε)')
    axes[1].set_ylabel('Final Cumulative Regret')
    axes[1].set_title('Final Regret vs ε\n(U-shaped: too little or too much exploration hurts)')
    axes[1].grid(True, alpha=0.3)

    # Mark optimal
    best_idx = np.argmin(final_regrets)
    axes[1].scatter([epsilons[best_idx]], [final_regrets[best_idx]],
                   color='red', s=200, zorder=5, marker='*')
    axes[1].annotate(f'Best: ε={epsilons[best_idx]}',
                    xy=(epsilons[best_idx], final_regrets[best_idx]),
                    xytext=(10, 10), textcoords='offset points')

    plt.tight_layout()

    print(f"\nResults for T={n_steps}:")
    for eps, regret in zip(epsilons, final_regrets):
        print(f"  ε={eps:.2f}: regret={regret:.1f}")
    print(f"\nBest ε: {epsilons[best_idx]} (regret={final_regrets[best_idx]:.1f})")
    print("\nInsight: ε=0 (no exploration) often gets stuck on suboptimal arm")
    print("         ε=1 (random) wastes too many pulls exploring")

    return fig


def ablation_ucb_c():
    """Ablation: Effect of exploration constant c in UCB."""
    print("\n" + "="*60)
    print("ABLATION: UCB EXPLORATION CONSTANT c")
    print("="*60)

    means = [0.2, 0.7, 0.5, 0.3]
    env = BanditEnvironment(means)

    c_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    n_steps = 1000
    n_runs = 30

    fig, ax = plt.subplots(figsize=(10, 5))

    final_regrets = []
    colors = plt.cm.plasma(np.linspace(0, 1, len(c_values)))

    for c, color in zip(c_values, colors):
        agent = UCB(len(means), c=c)
        results = run_multiple_experiments(env, agent, n_steps, n_runs)

        ax.plot(results['mean_cumulative_regret'],
               color=color, label=f'c={c}', linewidth=2)
        final_regrets.append(results['mean_cumulative_regret'][-1])

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('UCB: Effect of Exploration Constant c\n'
                 'c=0 → greedy, c→∞ → always explore')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print(f"\nResults for T={n_steps}:")
    for c, regret in zip(c_values, final_regrets):
        print(f"  c={c:.1f}: regret={regret:.1f}")

    return fig


def ablation_num_arms():
    """Ablation: Effect of number of arms."""
    print("\n" + "="*60)
    print("ABLATION: NUMBER OF ARMS")
    print("="*60)

    k_values = [2, 5, 10, 20, 50]
    n_steps = 2000
    n_runs = 20

    fig, ax = plt.subplots(figsize=(10, 5))

    algorithms = {
        'ε-greedy': lambda k: EpsilonGreedy(k, epsilon=0.1),
        'UCB': lambda k: UCB(k, c=2.0),
        'Thompson': lambda k: ThompsonSampling(k),
    }

    markers = ['o', 's', '^']
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for (name, agent_fn), marker, color in zip(algorithms.items(), markers, colors):
        regrets = []
        for k in k_values:
            # Create environment with k arms, one optimal
            means = np.random.uniform(0.2, 0.5, k)
            means[0] = 0.8  # make first arm optimal
            env = BanditEnvironment(list(means))

            agent = agent_fn(k)
            results = run_multiple_experiments(env, agent, n_steps, n_runs)
            regrets.append(results['mean_cumulative_regret'][-1])

        ax.plot(k_values, regrets, marker=marker, color=color,
               label=name, linewidth=2, markersize=8)

    ax.set_xlabel('Number of Arms')
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Regret vs Number of Arms\n'
                 f'(T={n_steps}, more arms → harder exploration)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()

    print("\nInsight: More arms means harder exploration problem")
    print("Thompson Sampling often scales better with many arms")

    return fig


def ablation_time_horizon():
    """Ablation: Regret growth rate over time."""
    print("\n" + "="*60)
    print("ABLATION: REGRET GROWTH RATE")
    print("="*60)

    means = [0.3, 0.7, 0.5]
    env = BanditEnvironment(means)

    n_steps = 5000
    n_runs = 30

    algorithms = {
        'Greedy': Greedy(len(means)),
        'ε-greedy': EpsilonGreedy(len(means), epsilon=0.1),
        'UCB': UCB(len(means), c=2.0),
        'Thompson': ThompsonSampling(len(means)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#999999', '#e41a1c', '#377eb8', '#4daf4a']

    for (name, agent), color in zip(algorithms.items(), colors):
        results = run_multiple_experiments(env, agent, n_steps, n_runs)
        regret = results['mean_cumulative_regret']

        # Linear scale
        axes[0].plot(regret, color=color, label=name, linewidth=2)

        # Log-log scale (to see growth rate)
        t = np.arange(1, n_steps + 1)
        axes[1].plot(t, regret, color=color, label=name, linewidth=2)

    # Add reference lines for growth rates
    t = np.arange(1, n_steps + 1)
    axes[1].plot(t, 0.5 * t, 'k--', alpha=0.3, label='O(T) - linear')
    axes[1].plot(t, 10 * np.sqrt(t), 'k:', alpha=0.3, label='O(√T) - sublinear')

    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Cumulative Regret (Linear Scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time Step (log)')
    axes[1].set_ylabel('Cumulative Regret (log)')
    axes[1].set_title('Regret Growth Rate (Log-Log Scale)\n'
                      'Greedy: O(T), Good algorithms: O(√T) or O(log T)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    plt.tight_layout()

    print("\nKey insight:")
    print("  - Greedy (no exploration): Linear regret O(T) - keeps losing")
    print("  - Good algorithms: Sublinear regret O(√T) or O(log T)")
    print("  - Exploration is NECESSARY for sublinear regret")

    return fig


def ablation_nonstationary():
    """Ablation: Non-stationary bandits (arm means change)."""
    print("\n" + "="*60)
    print("ABLATION: NON-STATIONARY BANDITS")
    print("="*60)

    class NonstationaryBandit:
        """Bandit where arm means change over time."""

        def __init__(self, initial_means, change_points, new_means_list):
            self.means = np.array(initial_means)
            self.k = len(initial_means)
            self.change_points = change_points
            self.new_means_list = new_means_list
            self.t = 0
            self.change_idx = 0

        def pull(self, arm):
            # Check for change
            if (self.change_idx < len(self.change_points) and
                self.t >= self.change_points[self.change_idx]):
                self.means = np.array(self.new_means_list[self.change_idx])
                self.change_idx += 1

            self.t += 1
            return np.random.normal(self.means[arm], 1.0)

        @property
        def optimal_mean(self):
            return np.max(self.means)

        def regret(self, arm):
            return self.optimal_mean - self.means[arm]

    # Environment: best arm switches at t=500
    initial_means = [0.3, 0.8, 0.5]  # Arm 1 best initially
    change_points = [500]
    new_means_list = [[0.8, 0.3, 0.5]]  # Arm 0 becomes best

    n_steps = 1000
    n_runs = 30

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    algorithms = {
        'ε-greedy (ε=0.1)': lambda k: EpsilonGreedy(k, epsilon=0.1),
        'ε-greedy (ε=0.3)': lambda k: EpsilonGreedy(k, epsilon=0.3),  # More exploration
        'UCB': lambda k: UCB(k, c=2.0),
        'Decaying ε': lambda k: DecayingEpsilonGreedy(k, epsilon_0=1.0),
    }

    colors = ['#e41a1c', '#ff7f00', '#377eb8', '#984ea3']

    for (name, agent_fn), color in zip(algorithms.items(), colors):
        all_regrets = []

        for _ in range(n_runs):
            env = NonstationaryBandit(initial_means, change_points, new_means_list)
            agent = agent_fn(len(initial_means))

            regrets = []
            for t in range(n_steps):
                arm = agent.select_arm()
                reward = env.pull(arm)
                agent.update(arm, reward)
                regrets.append(env.regret(arm))

            all_regrets.append(np.cumsum(regrets))

        mean_regret = np.mean(all_regrets, axis=0)
        axes[0].plot(mean_regret, color=color, label=name, linewidth=2)

    axes[0].axvline(x=500, color='black', linestyle='--', alpha=0.5, label='Change point')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Non-Stationary Bandit\n(Best arm switches at t=500)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Show arm pulls around change point
    env = NonstationaryBandit(initial_means, change_points, new_means_list)
    agent = EpsilonGreedy(len(initial_means), epsilon=0.1)

    arm_history = []
    for t in range(n_steps):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)
        arm_history.append(arm)

    arm_history = np.array(arm_history)
    window = 50
    for arm in range(3):
        pulls = np.convolve(arm_history == arm, np.ones(window)/window, mode='valid')
        axes[1].plot(pulls, label=f'Arm {arm}', linewidth=2)

    axes[1].axvline(x=500, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Pull Frequency (smoothed)')
    axes[1].set_title('Arm Selection Over Time\n(ε-greedy slowly adapts after change)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    print("\nKey insight for non-stationary bandits:")
    print("  - UCB/Thompson: decay exploration → can't adapt quickly")
    print("  - Higher ε maintains exploration → adapts to changes")
    print("  - Specialized methods: sliding window, discounted UCB")

    return fig


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print("="*70)
    print(" " * 15 + "MULTI-ARMED BANDITS")
    print(" " * 10 + "Exploration vs Exploitation")
    print("="*70)

    print("""
THE FUNDAMENTAL DILEMMA:

    You're in a casino. Each slot machine has unknown payout.

    EXPLOIT → Pull the machine that looks best (safe but maybe wrong)
    EXPLORE → Try other machines (risky but might find better)

    Too much exploit → stuck on suboptimal machine forever
    Too much explore → waste pulls on bad machines

    This is the CORE TENSION in reinforcement learning.

THREE STRATEGIES:

    1. ε-greedy: Random exploration with probability ε
       Simple, but doesn't adapt exploration over time

    2. UCB: Optimism = estimate + uncertainty bonus
       Favors uncertain arms, bonus shrinks with pulls

    3. Thompson Sampling: Sample beliefs, act greedily
       Naturally balances exploration with confidence
    """)

    # Main visualizations
    figs = []

    figs.append(('regret', visualize_regret_comparison()))
    figs.append(('arms', visualize_arm_pulls()))
    figs.append(('ucb', visualize_ucb_confidence()))
    figs.append(('thompson', visualize_thompson_posteriors()))

    # Ablations
    print("\n" + "="*70)
    print(" " * 20 + "ABLATION EXPERIMENTS")
    print("="*70)

    figs.append(('abl_epsilon', ablation_epsilon()))
    figs.append(('abl_ucb_c', ablation_ucb_c()))
    figs.append(('abl_arms', ablation_num_arms()))
    figs.append(('abl_time', ablation_time_horizon()))
    figs.append(('abl_nonstat', ablation_nonstationary()))

    # Save all figures
    for name, fig in figs:
        save_path = f'/Users/sid47/ML Algorithms/25_bandits_{name}.png'
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # Summary
    print("\n" + "="*70)
    print(" " * 20 + "KEY TAKEAWAYS")
    print("="*70)
    print("""
1. EXPLORATION IS NECESSARY
   Without it, you get linear regret (keep losing forever)
   With it, you get sublinear regret (√T or log T)

2. UCB: OPTIMISM IN THE FACE OF UNCERTAINTY
   Score = estimate + exploration bonus
   Bonus shrinks as you pull arm more
   Automatically balances explore/exploit

3. THOMPSON SAMPLING: PROBABILITY MATCHING
   Sample from beliefs, act greedily on sample
   Naturally explores uncertain options
   Often best empirically

4. ε-GREEDY: SIMPLE BUT TUNABLE
   Fixed exploration rate
   Doesn't adapt, but works well with good ε
   Good baseline

5. NEXT: Add STATES → full RL
   Bandits are RL with no state transitions
   Q-learning, Policy Gradients extend these ideas
   """)
