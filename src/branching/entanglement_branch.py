"""
Entanglement-Based Branching Strategy.

This module implements branching strategies based on population correlation
structure. The key innovation is using "entanglement" (correlation strength)
to guide branching decisions, potentially reducing the search tree size
compared to traditional heuristics.

Core ideas:
1. Variables with high entanglement affect many other variables when fixed
2. By branching on high-entanglement variables first, we propagate more
   information through the correlation structure
3. The correlation structure can also suggest which bound to tighten
   when a variable is fixed (measurement collapse propagation)

This is the third major innovation of the paper.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

from .correlation import CorrelationAnalyzer, CorrelationConfig


@dataclass
class BranchingConfig:
    """Configuration for entanglement branching.

    Attributes:
        alpha: Weight for entanglement vs fractional (0-1)
        temperature: Temperature for Boltzmann weighting
        use_collapse_propagation: Whether to propagate bounds after branching
        min_fractional: Minimum fractional value to consider for branching
        method: 'entanglement', 'fractional', 'combined', 'adaptive'
    """

    alpha: float = 0.5
    temperature: float = 1.0
    use_collapse_propagation: bool = True
    min_fractional: float = 0.01
    method: str = "combined"


@dataclass
class BranchingDecision:
    """A branching decision.

    Attributes:
        variable: Variable index to branch on
        value: Branching value
        score: Branching score
        entanglement_score: Entanglement component
        fractional_score: Fractional component
        direction: 'up' or 'down' for child nodes
    """

    variable: int
    value: float
    score: float
    entanglement_score: float
    fractional_score: float
    direction: str = "up"

    def __repr__(self):
        return (
            f"Branch(x{self.variable} {'<=' if self.direction == 'down' else '>='} "
            f"{self.value:.4f}, score={self.score:.3f})"
        )


@dataclass
class CollapsePropagation:
    """Result of measurement collapse propagation.

    When a variable is fixed, this describes the inferred bounds
    on other variables based on correlation structure.

    Attributes:
        fixed_var: The fixed variable
        fixed_value: The value it was fixed to
        tendencies: Dict mapping variable -> (lower_tendency, upper_tendency)
        confidence: Confidence in the propagation
    """

    fixed_var: int
    fixed_value: float
    tendencies: Dict[int, Tuple[float, float]]
    confidence: float


class EntanglementBranching:
    """Entanglement-based branching strategy.

    Uses population correlation structure to select branching variables
    and potentially propagate bound information.
    """

    def __init__(
        self,
        integer_vars: List[int],
        config: Optional[BranchingConfig] = None,
        correlation_config: Optional[CorrelationConfig] = None,
    ):
        """Initialize entanglement branching.

        Args:
            integer_vars: Indices of integer variables
            config: Branching configuration
            correlation_config: Correlation analysis configuration
        """
        self.integer_vars = integer_vars
        self.config = config or BranchingConfig()
        self.correlation_config = correlation_config or CorrelationConfig()

        self.analyzer = CorrelationAnalyzer(self.correlation_config)

        # History for adaptive strategy
        self.decision_history: List[BranchingDecision] = []

    def compute_weights(
        self,
        energies: np.ndarray,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Compute Boltzmann weights from energies.

        w_i ∝ exp(-E_i / T)

        Args:
            energies: Energy values, shape (K,)
            temperature: Temperature parameter

        Returns:
            Weights, shape (K,)
        """
        T = temperature if temperature is not None else self.config.temperature

        # Shift for numerical stability
        E_shifted = energies - np.min(energies)

        # Boltzmann weights
        log_weights = -E_shifted / T
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        return weights

    def select_branching_variable(
        self,
        X: np.ndarray,
        energies: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        current_solution: Optional[np.ndarray] = None,
    ) -> BranchingDecision:
        """Select the best variable for branching.

        Args:
            X: Population solutions, shape (K, n)
            energies: Energy values, shape (K,)
            lb, ub: Current variable bounds
            current_solution: Current best solution (optional)

        Returns:
            BranchingDecision with selected variable and value
        """
        # Compute weights
        weights = self.compute_weights(energies)

        # Compute correlation matrix
        corr = self.analyzer.compute_correlation_matrix(
            X, weights, self.integer_vars
        )

        # Compute scores
        ent_scores = self.analyzer.compute_entanglement_scores(corr)
        frac_scores = self.analyzer.compute_fractional_scores(
            X, weights, self.integer_vars
        )

        # Normalize scores
        def normalize(s):
            s_min, s_max = s.min(), s.max()
            if s_max - s_min < 1e-10:
                return np.ones_like(s) * 0.5
            return (s - s_min) / (s_max - s_min)

        ent_norm = normalize(ent_scores)
        frac_norm = normalize(frac_scores)

        # Combined score based on method
        if self.config.method == "entanglement":
            combined = ent_norm
        elif self.config.method == "fractional":
            combined = frac_norm
        else:  # combined or adaptive
            alpha = self.config.alpha
            combined = alpha * ent_norm + (1 - alpha) * frac_norm

        # Mask out non-fractional variables
        X_int = X[:, self.integer_vars]
        max_frac = np.max(np.abs(X_int - np.round(X_int)), axis=0)
        non_fractional_mask = max_frac < self.config.min_fractional

        combined = combined.copy()
        combined[non_fractional_mask] = -np.inf

        # Select best variable
        best_local_idx = np.argmax(combined)
        best_var = self.integer_vars[best_local_idx]

        # Get branching value
        branch_value = self.analyzer.get_branching_value(
            X, best_var, weights, method="weighted_median"
        )

        # Round to integer for integer variables
        branch_value = np.round(branch_value)

        # Create decision
        decision = BranchingDecision(
            variable=best_var,
            value=branch_value,
            score=combined[best_local_idx],
            entanglement_score=ent_scores[best_local_idx],
            fractional_score=frac_scores[best_local_idx],
        )

        # Record history
        self.decision_history.append(decision)

        return decision

    def propagate_collapse(
        self,
        corr_matrix: np.ndarray,
        fixed_var: int,
        fixed_value: float,
        lb: np.ndarray,
        ub: np.ndarray,
        threshold: float = 0.3,
    ) -> CollapsePropagation:
        """Propagate the effect of fixing a variable.

        When variable i is fixed to value v, variables with high
        correlation |ρ_ij| > threshold get suggested bound adjustments.

        Args:
            corr_matrix: Correlation matrix for integer variables
            fixed_var: Index of fixed variable (in integer var space)
            fixed_value: Value it was fixed to
            lb, ub: Current bounds
            threshold: Minimum correlation to propagate

        Returns:
            CollapsePropagation with suggested tendencies
        """
        tendencies = {}
        n = corr_matrix.shape[0]

        for j in range(n):
            if j == fixed_var:
                continue

            rho = corr_matrix[fixed_var, j]

            if abs(rho) < threshold:
                continue

            # Correlated variable j
            var_j = self.integer_vars[j]

            # Current center of bounds
            center = (lb[var_j] + ub[var_j]) / 2
            width = ub[var_j] - lb[var_j]

            # Tendency based on correlation sign
            # If ρ > 0: when i increases, j tends to increase
            # If ρ < 0: when i increases, j tends to decrease

            # Normalize fixed_value to [0, 1] within its bounds
            var_i = self.integer_vars[fixed_var]
            normalized_fixed = (fixed_value - lb[var_i]) / (ub[var_i] - lb[var_i] + 1e-10)

            # Infer tendency for j
            if rho > 0:
                # Positive correlation: j tends to same direction as i
                tendency = normalized_fixed
            else:
                # Negative correlation: j tends to opposite direction
                tendency = 1 - normalized_fixed

            # Map tendency to [lb, ub]
            suggested_value = lb[var_j] + tendency * width

            # Store as (lower_tightening, upper_tightening)
            # If tendency > 0.5, can tighten lower bound
            # If tendency < 0.5, can tighten upper bound
            lower_tight = suggested_value if tendency > 0.5 else lb[var_j]
            upper_tight = suggested_value if tendency < 0.5 else ub[var_j]

            tendencies[var_j] = (lower_tight, upper_tight)

        # Confidence based on average correlation strength
        correlations = [abs(corr_matrix[fixed_var, j]) for j in range(n) if j != fixed_var]
        confidence = np.mean(correlations) if correlations else 0.0

        return CollapsePropagation(
            fixed_var=fixed_var,
            fixed_value=fixed_value,
            tendencies=tendencies,
            confidence=confidence,
        )

    def get_child_directions(
        self,
        decision: BranchingDecision,
        X: np.ndarray,
        energies: np.ndarray,
    ) -> Tuple[str, str]:
        """Determine which child to explore first.

        Based on population statistics, determine if the "up" or "down"
        branch is more promising.

        Args:
            decision: Branching decision
            X: Population solutions
            energies: Energy values

        Returns:
            Tuple of (first_direction, second_direction)
        """
        var = decision.variable
        value = decision.value

        # Count population members in each branch
        up_mask = X[:, var] >= value
        down_mask = X[:, var] < value

        # Compute average energy in each branch
        if np.sum(up_mask) > 0:
            up_energy = np.mean(energies[up_mask])
        else:
            up_energy = float('inf')

        if np.sum(down_mask) > 0:
            down_energy = np.mean(energies[down_mask])
        else:
            down_energy = float('inf')

        # Explore lower energy branch first
        if up_energy <= down_energy:
            return "up", "down"
        else:
            return "down", "up"


class AdaptiveEntanglementBranching(EntanglementBranching):
    """Adaptive branching that learns from history.

    Adjusts alpha parameter based on which decisions led to
    good solutions.
    """

    def __init__(
        self,
        integer_vars: List[int],
        config: Optional[BranchingConfig] = None,
        correlation_config: Optional[CorrelationConfig] = None,
        learning_rate: float = 0.1,
    ):
        """Initialize adaptive branching."""
        super().__init__(integer_vars, config, correlation_config)
        self.learning_rate = learning_rate

        # Track success of entanglement vs fractional decisions
        self.entanglement_success = 0.5
        self.fractional_success = 0.5
        self.n_updates = 0

    def update_alpha(self, decision: BranchingDecision, success: bool):
        """Update alpha based on decision outcome.

        Args:
            decision: The branching decision
            success: Whether this decision led to improvement
        """
        if not success:
            return

        # Determine which component contributed more
        if decision.entanglement_score > decision.fractional_score:
            self.entanglement_success += self.learning_rate
        else:
            self.fractional_success += self.learning_rate

        self.n_updates += 1

        # Normalize
        total = self.entanglement_success + self.fractional_success
        self.entanglement_success /= total
        self.fractional_success /= total

        # Update alpha
        self.config.alpha = self.entanglement_success

    def select_branching_variable(
        self,
        X: np.ndarray,
        energies: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        current_solution: Optional[np.ndarray] = None,
    ) -> BranchingDecision:
        """Select variable with adaptive alpha."""
        # Use current adaptive alpha
        return super().select_branching_variable(
            X, energies, lb, ub, current_solution
        )


if __name__ == "__main__":
    # Test entanglement branching
    print("Testing Entanglement Branching")
    print("=" * 50)

    np.random.seed(42)

    # Create test population
    K, n = 20, 10
    integer_vars = [0, 1, 2, 3, 4]

    # Generate correlated population
    mean = np.zeros(n)
    cov = np.eye(n)
    cov[0, 1] = cov[1, 0] = 0.8
    cov[2, 3] = cov[3, 2] = -0.6

    X = np.random.multivariate_normal(mean, cov, K)
    X = np.clip(X * 0.3 + 0.5, 0, 1)

    # Generate energies (lower is better)
    energies = X @ np.random.randn(n) * 0.5

    # Current bounds
    lb = np.zeros(n)
    ub = np.ones(n)

    # Create branching strategy
    config = BranchingConfig(alpha=0.5, method="combined")
    branching = EntanglementBranching(integer_vars, config)

    # Select branching variable
    decision = branching.select_branching_variable(X, energies, lb, ub)

    print(f"Branching decision: {decision}")

    # Get child directions
    first, second = branching.get_child_directions(decision, X, energies)
    print(f"Child order: {first} first, then {second}")

    # Test collapse propagation
    corr = branching.analyzer.compute_correlation_matrix(X, integer_vars=integer_vars)
    propagation = branching.propagate_collapse(corr, 0, 1.0, lb, ub, threshold=0.3)

    print(f"\nCollapse propagation (fixed var 0 = 1):")
    print(f"  Confidence: {propagation.confidence:.3f}")
    for var, (lo, hi) in propagation.tendencies.items():
        print(f"  Var {var}: [{lo:.3f}, {hi:.3f}]")
