"""
Quantum-Inspired Interference Operator for Population PDHG.

This module implements interference operators inspired by quantum mechanics
wave function interference. When two population members interact, their
solutions can combine in ways that potentially yield better solutions.

The key insight is:
- Constructive interference: When two good solutions point in similar directions,
  combining them reinforces that direction
- Destructive interference: When solutions conflict, the interference term
  can help escape local minima

Mathematical formulation:
    x_combined = α * x_i + β * x_j + γ * (x_i - x_j) * sign(E_j - E_i)

where:
    - α, β are weighted averaging coefficients
    - γ is the interference strength
    - E_i, E_j are energy (objective) values
    - The interference term biases toward the better solution
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse


@dataclass
class InterferenceConfig:
    """Configuration for quantum interference operator.

    Attributes:
        interference_rate: Fraction of population to apply interference to (0-1)
        alpha: Weight for better solution in averaging (should be > 0.5)
        gamma: Interference strength coefficient
        min_diversity: Minimum diversity threshold (skip interference if below)
        greedy_accept: If True, only accept if improvement
    """

    interference_rate: float = 0.2
    alpha: float = 0.7
    gamma: float = 0.1
    min_diversity: float = 1e-6
    greedy_accept: bool = True


class QuantumInterference:
    """Quantum-inspired interference operator for population evolution.

    Implements interference between pairs of population members to
    potentially create better solutions through constructive combination.

    Example:
        >>> interference = QuantumInterference()
        >>> new_state = interference.apply(state, A, b, c, lb, ub)
    """

    def __init__(self, config: Optional[InterferenceConfig] = None):
        """Initialize interference operator.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or InterferenceConfig()

    def compute_energy(
        self,
        x: np.ndarray,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        penalty: float = 100.0,
    ) -> np.ndarray:
        """Compute energy (objective + constraint violation) for population.

        Energy = c^T x + penalty * ||max(Ax - b, 0)||^2

        Args:
            x: Population solutions, shape (K, n)
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            penalty: Constraint violation penalty

        Returns:
            Energy values, shape (K,)
        """
        K = x.shape[0]

        # Objective value
        obj = x @ c  # (K,)

        # Constraint violation
        Ax = (A @ x.T).T  # (K, m)
        violation = np.maximum(Ax - b, 0)  # (K, m)
        penalty_term = penalty * np.sum(violation**2, axis=1)  # (K,)

        return obj + penalty_term

    def interfere_pair(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray,
        E_i: float,
        E_j: float,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply interference between two solutions.

        If E_i < E_j (i is better):
            x_new_i = (α + γ) * x_i + (1 - α - γ) * x_j  (biased toward i)
            x_new_j = (α - γ) * x_i + (1 - α + γ) * x_j  (biased toward i)

        Args:
            x_i, x_j: Two solutions to interfere, shape (n,)
            E_i, E_j: Their respective energy values
            lb, ub: Variable bounds

        Returns:
            Tuple of (new_x_i, new_x_j)
        """
        alpha = self.config.alpha
        gamma = self.config.gamma

        if E_i <= E_j:
            # i is better or equal
            # New solutions biased toward x_i
            weight_better = alpha + gamma
            weight_worse = 1 - alpha - gamma

            x_new_i = weight_better * x_i + weight_worse * x_j
            x_new_j = (alpha - gamma) * x_i + (1 - alpha + gamma) * x_j
        else:
            # j is better
            weight_better = alpha + gamma
            weight_worse = 1 - alpha - gamma

            x_new_i = (alpha - gamma) * x_j + (1 - alpha + gamma) * x_i
            x_new_j = weight_better * x_j + weight_worse * x_i

        # Project to bounds
        x_new_i = np.clip(x_new_i, lb, ub)
        x_new_j = np.clip(x_new_j, lb, ub)

        return x_new_i, x_new_j

    def apply(
        self,
        state: "PopulationState",
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        penalty: float = 100.0,
        seed: Optional[int] = None,
    ) -> Tuple["PopulationState", dict]:
        """Apply interference to the population.

        Randomly selects pairs of members and applies interference.
        Only accepts changes that improve (or don't worsen) solutions.

        Args:
            state: Current population state
            A, b, c: Problem data
            lb, ub: Variable bounds
            penalty: Constraint violation penalty for energy computation
            seed: Random seed

        Returns:
            Tuple of (updated_state, statistics)
        """
        if seed is not None:
            np.random.seed(seed)

        K = state.K
        x = state.x.copy()
        y = state.y.copy()

        # Compute current energies
        E = self.compute_energy(x, A, b, c, penalty)

        # Number of pairs to interfere
        n_pairs = int(self.config.interference_rate * K / 2)

        # Randomly select pairs
        indices = np.random.permutation(K)
        pairs = [(indices[2 * i], indices[2 * i + 1]) for i in range(n_pairs)]

        n_accepted = 0
        n_rejected = 0

        for i, j in pairs:
            # Apply interference
            x_new_i, x_new_j = self.interfere_pair(
                x[i], x[j], E[i], E[j], lb, ub
            )

            if self.config.greedy_accept:
                # Compute new energies
                E_new_i = self.compute_energy(x_new_i.reshape(1, -1), A, b, c, penalty)[0]
                E_new_j = self.compute_energy(x_new_j.reshape(1, -1), A, b, c, penalty)[0]

                # Accept if improvement
                if E_new_i < E[i]:
                    x[i] = x_new_i
                    E[i] = E_new_i
                    n_accepted += 1
                else:
                    n_rejected += 1

                if E_new_j < E[j]:
                    x[j] = x_new_j
                    E[j] = E_new_j
                    n_accepted += 1
                else:
                    n_rejected += 1
            else:
                # Always accept
                x[i] = x_new_i
                x[j] = x_new_j
                n_accepted += 2

        # Update objective values
        obj = x @ c

        # Update feasibility
        Ax = (A @ x.T).T
        primal_feas = np.linalg.norm(np.maximum(Ax - b, 0), axis=1)

        # Create new state
        from .pop_pdhg import PopulationState

        new_state = PopulationState(
            x=x,
            y=y,
            obj=obj,
            primal_feas=primal_feas,
            dual_feas=state.dual_feas.copy(),
            age=state.age.copy(),
        )

        stats = {
            "n_pairs": n_pairs,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "accept_rate": n_accepted / (n_accepted + n_rejected)
            if (n_accepted + n_rejected) > 0
            else 0.0,
        }

        return new_state, stats


class DiversityPreservingInterference(QuantumInterference):
    """Interference operator that explicitly preserves population diversity.

    Extends basic interference with diversity-aware selection and
    adaptive interference strength based on population diversity.
    """

    def __init__(
        self,
        config: Optional[InterferenceConfig] = None,
        diversity_threshold: float = 0.01,
    ):
        """Initialize diversity-preserving interference.

        Args:
            config: Base configuration
            diversity_threshold: Minimum diversity to maintain
        """
        super().__init__(config)
        self.diversity_threshold = diversity_threshold

    def compute_diversity(self, x: np.ndarray) -> float:
        """Compute population diversity as average pairwise distance."""
        K = x.shape[0]
        if K < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = np.linalg.norm(x[i] - x[j])
                total_dist += dist
                count += 1
        return total_dist / count if count > 0 else 0.0

    def apply(
        self,
        state: "PopulationState",
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        penalty: float = 100.0,
        seed: Optional[int] = None,
    ) -> Tuple["PopulationState", dict]:
        """Apply diversity-preserving interference.

        Reduces interference strength if diversity is low to prevent
        premature convergence.
        """
        # Check current diversity
        diversity = self.compute_diversity(state.x)

        # Adjust gamma based on diversity
        original_gamma = self.config.gamma
        if diversity < self.diversity_threshold:
            # Reduce interference when diversity is low
            self.config.gamma = original_gamma * (diversity / self.diversity_threshold)

        # Apply base interference
        result = super().apply(state, A, b, c, lb, ub, penalty, seed)

        # Restore original gamma
        self.config.gamma = original_gamma

        # Add diversity info to stats
        result[1]["diversity_before"] = diversity
        result[1]["diversity_after"] = self.compute_diversity(result[0].x)

        return result


if __name__ == "__main__":
    # Test interference operator
    print("Testing Quantum Interference Operator...")

    from src.population.pop_pdhg import PopulationPDHG, PopulationState
    from scipy import sparse

    # Simple LP
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e10, 1e10])

    # Initialize population
    solver = PopulationPDHG(A, b, c, lb, ub, population_size=8)
    state = solver.initialize_population(strategy="diverse", seed=42)

    print(f"Initial diversity: {solver.compute_diversity(state):.4f}")
    print(f"Initial objectives: {state.obj}")

    # Apply interference
    interference = QuantumInterference()
    new_state, stats = interference.apply(state, A, b, c, lb, ub, seed=42)

    print(f"\nAfter interference:")
    print(f"  Accept rate: {stats['accept_rate']:.2%}")
    print(f"  New objectives: {new_state.obj}")
    print(f"  New diversity: {solver.compute_diversity(new_state):.4f}")
