"""
HIDDEN MARKOV MODEL — Paradigm: GENERATIVE (Sequential Latent States)

===============================================================
WHAT IT IS (THE CORE IDEA)
===============================================================

A GENERATIVE model for sequences where:
    - There's a HIDDEN state z_t at each time step
    - Hidden states follow a MARKOV chain: P(z_t | z_{t-1})
    - Observations are emitted from hidden states: P(x_t | z_t)

The "Markov" part: z_t only depends on z_{t-1} (memoryless).
The "Hidden" part: we only observe x_t, not z_t.

===============================================================
THE KEY INSIGHT: FACTORIZATION
===============================================================

Joint probability factorizes as:

P(x₁:T, z₁:T) = P(z₁) × Π_{t=2}^T P(z_t | z_{t-1}) × Π_{t=1}^T P(x_t | z_t)

This structured factorization enables EFFICIENT inference:
    - Forward-backward algorithm: O(T × K²) instead of O(K^T)
    - Viterbi decoding: find most likely state sequence

===============================================================
THE THREE HMM PROBLEMS
===============================================================

1. EVALUATION: P(x₁:T | model)
   "How likely is this sequence?"
   → Forward algorithm

2. DECODING: argmax_{z₁:T} P(z₁:T | x₁:T)
   "What's the most likely hidden state sequence?"
   → Viterbi algorithm

3. LEARNING: argmax_{θ} P(x₁:T | θ)
   "What parameters best explain the data?"
   → Baum-Welch (EM) algorithm

===============================================================
HMM PARAMETERS
===============================================================

π (initial state distribution): P(z₁ = k)
A (transition matrix): A[i,j] = P(z_t = j | z_{t-1} = i)
B (emission probabilities): P(x_t | z_t)

For Gaussian emissions: B[k] ~ N(μ_k, Σ_k)
For discrete emissions: B[k,v] = P(x_t = v | z_t = k)

===============================================================
FORWARD-BACKWARD vs VITERBI
===============================================================

FORWARD-BACKWARD:
    Computes P(z_t = k | x₁:T) for ALL t
    Uses sum over all paths through state k at time t
    "Soft" marginal posteriors

VITERBI:
    Finds argmax P(z₁:T | x₁:T)
    Uses max instead of sum
    "Hard" single best path

They can give DIFFERENT answers!
    Forward-backward: most likely state at each t (marginal)
    Viterbi: most likely PATH (joint)

===============================================================
INDUCTIVE BIAS
===============================================================

1. Markov assumption: z_t depends only on z_{t-1}
2. Conditional independence: x_t depends only on z_t
3. Finite discrete states
4. Stationary dynamics (same A, B at all times)

===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/sid47/ML Algorithms')


def create_hmm_dataset(n_sequences=100, seq_len=20, n_states=3, random_state=42):
    """
    Generate sequences from a true HMM.

    Returns sequences and true hidden states.
    """
    np.random.seed(random_state)

    # True HMM parameters
    # Initial state distribution
    pi = np.array([0.6, 0.3, 0.1])

    # Transition matrix (rows sum to 1)
    A = np.array([
        [0.7, 0.2, 0.1],  # From state 0
        [0.1, 0.7, 0.2],  # From state 1
        [0.2, 0.1, 0.7],  # From state 2
    ])

    # Emission means and variances (1D Gaussian emissions)
    means = np.array([-2.0, 0.0, 2.0])
    stds = np.array([0.5, 0.5, 0.5])

    X_list = []
    Z_list = []

    for _ in range(n_sequences):
        X = np.zeros(seq_len)
        Z = np.zeros(seq_len, dtype=int)

        # Sample initial state
        Z[0] = np.random.choice(n_states, p=pi)
        X[0] = np.random.normal(means[Z[0]], stds[Z[0]])

        # Sample sequence
        for t in range(1, seq_len):
            Z[t] = np.random.choice(n_states, p=A[Z[t-1]])
            X[t] = np.random.normal(means[Z[t]], stds[Z[t]])

        X_list.append(X)
        Z_list.append(Z)

    return X_list, Z_list


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.
    """

    def __init__(self, n_states=3, n_iter=100, tol=1e-4, random_state=None):
        """
        Parameters:
        -----------
        n_states : Number of hidden states
        n_iter : Maximum EM iterations
        tol : Convergence tolerance
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Parameters
        self.pi_ = None      # Initial state distribution
        self.A_ = None       # Transition matrix
        self.means_ = None   # Emission means
        self.vars_ = None    # Emission variances

    def _initialize(self, X_list):
        """Initialize parameters."""
        K = self.n_states

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Uniform initial distribution
        self.pi_ = np.ones(K) / K

        # Random transition matrix (normalize rows)
        self.A_ = np.random.rand(K, K)
        self.A_ /= self.A_.sum(axis=1, keepdims=True)

        # Initialize means by K-means-ish on all data
        all_data = np.concatenate(X_list)
        data_min, data_max = all_data.min(), all_data.max()
        self.means_ = np.linspace(data_min, data_max, K)

        # Initialize variances
        self.vars_ = np.ones(K) * np.var(all_data) / K

    def _emission_probs(self, X):
        """
        Compute emission probabilities P(x_t | z_t = k).

        Returns shape: (T, K)
        """
        T = len(X)
        K = self.n_states

        probs = np.zeros((T, K))
        for k in range(K):
            # Gaussian pdf
            probs[:, k] = (1 / np.sqrt(2 * np.pi * self.vars_[k])) * \
                          np.exp(-0.5 * (X - self.means_[k])**2 / self.vars_[k])

        # Prevent underflow
        probs = np.clip(probs, 1e-300, None)
        return probs

    def _forward(self, X):
        """
        Forward algorithm: compute α_t(k) = P(x_1:t, z_t = k)

        Uses scaling to prevent underflow.
        Returns: alpha (scaled), scale factors
        """
        T = len(X)
        K = self.n_states

        B = self._emission_probs(X)

        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        # Initialization: α_1(k) = π_k × B_k(x_1)
        alpha[0] = self.pi_ * B[0]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]

        # Recursion: α_t(k) = B_k(x_t) × Σ_j α_{t-1}(j) × A_{jk}
        for t in range(1, T):
            alpha[t] = B[t] * (alpha[t-1] @ self.A_)
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, X, scale):
        """
        Backward algorithm: compute β_t(k) = P(x_{t+1}:T | z_t = k)

        Uses same scaling factors from forward pass.
        """
        T = len(X)
        K = self.n_states

        B = self._emission_probs(X)

        beta = np.zeros((T, K))

        # Initialization: β_T(k) = 1
        beta[-1] = 1.0

        # Recursion: β_t(k) = Σ_j A_{kj} × B_j(x_{t+1}) × β_{t+1}(j)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A_ @ (B[t+1] * beta[t+1])) / scale[t+1]

        return beta

    def _e_step(self, X):
        """
        E-step: compute posterior probabilities.

        γ_t(k) = P(z_t = k | X)
        ξ_t(j,k) = P(z_t = j, z_{t+1} = k | X)
        """
        T = len(X)
        K = self.n_states

        alpha, scale = self._forward(X)
        beta = self._backward(X, scale)
        B = self._emission_probs(X)

        # γ_t(k) = α_t(k) × β_t(k) (already normalized by scaling)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        # ξ_t(j,k) = α_t(j) × A_{jk} × B_k(x_{t+1}) × β_{t+1}(k) / P(X)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi_t = np.outer(alpha[t], B[t+1] * beta[t+1]) * self.A_
            xi[t] = xi_t / xi_t.sum()

        # Log-likelihood
        log_likelihood = np.sum(np.log(scale))

        return gamma, xi, log_likelihood

    def _m_step(self, X_list, gamma_list, xi_list):
        """
        M-step: update parameters.
        """
        K = self.n_states

        # Accumulate statistics across all sequences
        gamma_sum_init = np.zeros(K)
        gamma_sum = np.zeros(K)
        xi_sum = np.zeros((K, K))
        weighted_sum = np.zeros(K)
        weighted_sq_sum = np.zeros(K)

        for X, gamma, xi in zip(X_list, gamma_list, xi_list):
            gamma_sum_init += gamma[0]
            gamma_sum += gamma.sum(axis=0)
            xi_sum += xi.sum(axis=0)

            for k in range(K):
                weighted_sum[k] += np.sum(gamma[:, k] * X)
                weighted_sq_sum[k] += np.sum(gamma[:, k] * X**2)

        # Update initial distribution
        self.pi_ = gamma_sum_init / len(X_list)

        # Update transition matrix
        # Normalize each row to get transition probabilities
        for k in range(K):
            row_sum = xi_sum[k].sum()
            if row_sum > 0:
                self.A_[k] = xi_sum[k] / row_sum
            else:
                # Fallback: uniform distribution if no transitions from state k
                self.A_[k] = np.ones(K) / K

        # Update emission parameters
        self.means_ = weighted_sum / gamma_sum
        self.vars_ = weighted_sq_sum / gamma_sum - self.means_**2
        self.vars_ = np.clip(self.vars_, 1e-4, None)  # Prevent zero variance

    def fit(self, X_list, verbose=True):
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        """
        self._initialize(X_list)

        log_likelihoods = []
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step for all sequences
            gamma_list = []
            xi_list = []
            total_ll = 0

            for X in X_list:
                gamma, xi, ll = self._e_step(X)
                gamma_list.append(gamma)
                xi_list.append(xi)
                total_ll += ll

            log_likelihoods.append(total_ll)

            # M-step
            self._m_step(X_list, gamma_list, xi_list)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: log-likelihood = {total_ll:.2f}")

            # Check convergence
            if abs(total_ll - prev_ll) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1}")
                break

            prev_ll = total_ll

        self.log_likelihoods_ = log_likelihoods
        return self

    def score(self, X):
        """Compute log-likelihood of a sequence."""
        _, scale = self._forward(X)
        return np.sum(np.log(scale))

    def predict(self, X):
        """
        Predict hidden states using FORWARD-BACKWARD (marginal MAP).
        """
        gamma, _, _ = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def viterbi(self, X):
        """
        Find most likely state sequence using VITERBI algorithm.

        Returns: best state sequence, its log probability
        """
        T = len(X)
        K = self.n_states

        B = self._emission_probs(X)

        # Viterbi uses log probabilities to avoid underflow
        log_A = np.log(self.A_ + 1e-300)
        log_pi = np.log(self.pi_ + 1e-300)
        log_B = np.log(B + 1e-300)

        # δ_t(k) = max_{z_1:t-1} log P(z_1:t-1, z_t=k, x_1:t)
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)  # Backpointers

        # Initialization
        delta[0] = log_pi + log_B[0]

        # Recursion
        for t in range(1, T):
            for k in range(K):
                # max over previous states
                candidates = delta[t-1] + log_A[:, k]
                psi[t, k] = np.argmax(candidates)
                delta[t, k] = candidates[psi[t, k]] + log_B[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        log_prob = delta[-1, states[-1]]

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, log_prob

    def sample(self, n_samples=20):
        """Generate a sample sequence from the model."""
        X = np.zeros(n_samples)
        Z = np.zeros(n_samples, dtype=int)

        # Sample initial state
        Z[0] = np.random.choice(self.n_states, p=self.pi_)
        X[0] = np.random.normal(self.means_[Z[0]], np.sqrt(self.vars_[Z[0]]))

        # Sample sequence
        for t in range(1, n_samples):
            Z[t] = np.random.choice(self.n_states, p=self.A_[Z[t-1]])
            X[t] = np.random.normal(self.means_[Z[t]], np.sqrt(self.vars_[Z[t]]))

        return X, Z


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

def ablation_experiments():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS")
    print("="*60)

    np.random.seed(42)
    X_list, Z_list = create_hmm_dataset(n_sequences=50, seq_len=30)

    # -------- Experiment 1: Number of States --------
    print("\n1. EFFECT OF NUMBER OF STATES")
    print("-" * 40)
    print("True model has 3 states")

    for n_states in [2, 3, 4, 5]:
        hmm = GaussianHMM(n_states=n_states, random_state=42)
        hmm.fit(X_list, verbose=False)

        total_ll = sum(hmm.score(X) for X in X_list)
        # BIC
        n_params = n_states - 1 + n_states * (n_states - 1) + 2 * n_states
        n_data = sum(len(X) for X in X_list)
        bic = -2 * total_ll + n_params * np.log(n_data)

        print(f"K={n_states} log_likelihood={total_ll:.1f} BIC={bic:.1f}")
    print("→ BIC helps identify correct number of states")

    # -------- Experiment 2: Viterbi vs Forward-Backward --------
    print("\n2. VITERBI vs FORWARD-BACKWARD DECODING")
    print("-" * 40)

    hmm = GaussianHMM(n_states=3, random_state=42)
    hmm.fit(X_list, verbose=False)

    # Decode one sequence both ways
    X_test = X_list[0]
    Z_true = Z_list[0]

    Z_fb = hmm.predict(X_test)  # Forward-backward (marginal)
    Z_vit, _ = hmm.viterbi(X_test)  # Viterbi (joint)

    fb_acc = np.mean(Z_fb == Z_true)
    vit_acc = np.mean(Z_vit == Z_true)
    agree = np.mean(Z_fb == Z_vit)

    print(f"Forward-Backward accuracy: {fb_acc:.3f}")
    print(f"Viterbi accuracy:          {vit_acc:.3f}")
    print(f"Agreement between methods: {agree:.3f}")
    print("→ Viterbi finds most likely PATH")
    print("→ Forward-Backward finds most likely STATE at each t")

    # -------- Experiment 3: Learned vs True Parameters --------
    print("\n3. LEARNED vs TRUE PARAMETERS")
    print("-" * 40)

    hmm = GaussianHMM(n_states=3, random_state=42)
    hmm.fit(X_list, verbose=False)

    print("True emission means:    [-2.0, 0.0, 2.0]")
    print(f"Learned emission means: {np.sort(hmm.means_).round(2)}")

    print("\nTrue initial dist:    [0.6, 0.3, 0.1]")
    print(f"Learned initial dist: {hmm.pi_.round(2)}")

    print("\nTrue transition (diagonal dominant):")
    print("Learned transition:")
    print(hmm.A_.round(2))
    print("→ HMM recovers parameters (up to state relabeling)")

    # -------- Experiment 4: Sequence Length Effect --------
    print("\n4. EFFECT OF SEQUENCE LENGTH")
    print("-" * 40)

    for seq_len in [10, 20, 50, 100]:
        X_list_len, Z_list_len = create_hmm_dataset(n_sequences=30, seq_len=seq_len)

        hmm = GaussianHMM(n_states=3, random_state=42)
        hmm.fit(X_list_len, verbose=False)

        # Decoding accuracy
        accs = []
        for X, Z_true in zip(X_list_len, Z_list_len):
            Z_pred, _ = hmm.viterbi(X)
            # Handle state relabeling
            from itertools import permutations
            best_acc = max(
                np.mean(np.array([p[z] for z in Z_pred]) == Z_true)
                for p in permutations(range(3))
            )
            accs.append(best_acc)

        print(f"seq_len={seq_len:<3} decoding_accuracy={np.mean(accs):.3f}")
    print("→ Longer sequences = better parameter estimation")

    # -------- Experiment 5: Sampling (Generative) --------
    print("\n5. SAMPLING FROM FITTED MODEL")
    print("-" * 40)

    hmm = GaussianHMM(n_states=3, random_state=42)
    hmm.fit(X_list, verbose=False)

    X_sample, Z_sample = hmm.sample(100)

    print(f"Sample mean:      {X_sample.mean():.3f}")
    print(f"Training mean:    {np.mean([X.mean() for X in X_list]):.3f}")
    print(f"Sample std:       {X_sample.std():.3f}")
    print(f"Training std:     {np.std(np.concatenate(X_list)):.3f}")
    print(f"State transitions in sample: {np.sum(Z_sample[1:] != Z_sample[:-1])}")
    print("→ HMM is GENERATIVE: can sample realistic sequences")

    # -------- Experiment 6: Initialization Sensitivity --------
    print("\n6. INITIALIZATION SENSITIVITY")
    print("-" * 40)

    log_likelihoods = []
    for seed in range(10):
        hmm = GaussianHMM(n_states=3, random_state=seed)
        hmm.fit(X_list, verbose=False)
        ll = sum(hmm.score(X) for X in X_list)
        log_likelihoods.append(ll)

    print(f"Log-likelihoods across 10 runs:")
    print(f"  Min: {min(log_likelihoods):.1f}")
    print(f"  Max: {max(log_likelihoods):.1f}")
    print(f"  Range: {max(log_likelihoods) - min(log_likelihoods):.1f}")
    print("→ Like GMM, HMM has local optima")


def visualize_hmm():
    """Visualize HMM decoding."""
    print("\n" + "="*60)
    print("HMM VISUALIZATION")
    print("="*60)

    np.random.seed(42)
    X_list, Z_list = create_hmm_dataset(n_sequences=20, seq_len=50)

    hmm = GaussianHMM(n_states=3, random_state=42)
    hmm.fit(X_list, verbose=False)

    # Select one sequence
    X = X_list[0]
    Z_true = Z_list[0]
    Z_vit, _ = hmm.viterbi(X)
    gamma, _, _ = hmm._e_step(X)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Observations and decoded states
    ax = axes[0]
    colors = ['red', 'green', 'blue']
    for t in range(len(X)):
        ax.scatter(t, X[t], c=colors[Z_vit[t]], s=30, alpha=0.7)
    ax.plot(X, 'k-', alpha=0.3)
    # Add horizontal lines for emission means
    for k, mean in enumerate(np.sort(hmm.means_)):
        ax.axhline(y=mean, color=colors[k], linestyle='--', alpha=0.5)
    ax.set_title('Observations (colored by Viterbi-decoded state)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    # Plot 2: True vs decoded states
    ax = axes[1]
    ax.plot(Z_true, 'b-', label='True states', alpha=0.7, linewidth=2)
    ax.plot(Z_vit, 'r--', label='Viterbi decoded', alpha=0.7, linewidth=2)
    ax.set_title('True vs Decoded Hidden States')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.set_yticks([0, 1, 2])

    # Plot 3: State probabilities (forward-backward)
    ax = axes[2]
    for k in range(3):
        ax.plot(gamma[:, k], label=f'P(z_t={k})', linewidth=2)
    ax.set_title('Forward-Backward State Probabilities')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.suptitle('HIDDEN MARKOV MODEL\n'
                 'Inferring hidden states from observations',
                 fontsize=12)
    plt.tight_layout()
    return fig


def benchmark_decoding():
    """Benchmark decoding accuracy."""
    print("\n" + "="*60)
    print("BENCHMARK: Decoding Accuracy")
    print("="*60)

    np.random.seed(42)
    X_list, Z_list = create_hmm_dataset(n_sequences=100, seq_len=30)

    # Split
    X_train, Z_train = X_list[:80], Z_list[:80]
    X_test, Z_test = X_list[80:], Z_list[80:]

    hmm = GaussianHMM(n_states=3, random_state=42)
    hmm.fit(X_train, verbose=False)

    # Evaluate on test
    from itertools import permutations

    vit_accs = []
    fb_accs = []

    for X, Z_true in zip(X_test, Z_test):
        Z_vit, _ = hmm.viterbi(X)
        Z_fb = hmm.predict(X)

        # Best permutation accuracy
        best_vit = max(
            np.mean(np.array([p[z] for z in Z_vit]) == Z_true)
            for p in permutations(range(3))
        )
        best_fb = max(
            np.mean(np.array([p[z] for z in Z_fb]) == Z_true)
            for p in permutations(range(3))
        )

        vit_accs.append(best_vit)
        fb_accs.append(best_fb)

    print(f"Viterbi decoding accuracy:         {np.mean(vit_accs):.3f}")
    print(f"Forward-Backward decoding accuracy: {np.mean(fb_accs):.3f}")

    return {'viterbi': np.mean(vit_accs), 'forward_backward': np.mean(fb_accs)}


if __name__ == '__main__':
    print("="*60)
    print("HMM — Hidden Markov Model")
    print("="*60)

    print("""
WHAT THIS MODEL IS:
    - Hidden states z_t follow a Markov chain
    - Observations x_t are emitted from hidden states
    - We observe x, want to infer z

THE THREE PROBLEMS:
    1. Evaluation: P(X) via Forward algorithm
    2. Decoding: best z sequence via Viterbi
    3. Learning: parameters via Baum-Welch (EM)

KEY INSIGHT:
    Markov structure enables O(TK²) inference
    (vs O(K^T) brute force)

FORWARD-BACKWARD vs VITERBI:
    FB: most likely state at each t (marginal)
    Viterbi: most likely PATH (joint)
    Can give different answers!

GENERATIVE:
    HMM can sample realistic sequences.
    """)

    ablation_experiments()
    results = benchmark_decoding()

    fig = visualize_hmm()
    save_path = '/Users/sid47/ML Algorithms/17_hmm.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close(fig)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
1. HMM: hidden Markov states + observed emissions
2. Forward algorithm: P(X) in O(TK²)
3. Viterbi: most likely state sequence
4. Baum-Welch: EM for parameter learning
5. Markov assumption enables efficient inference
6. GENERATIVE: models how sequences are produced
    """)
