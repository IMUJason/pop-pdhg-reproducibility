"""
Correlation Analysis for Population-Based Branching.

This module computes correlation structures from population solutions
to guide branching decisions in Branch & Bound. The key insight is that
if two variables are highly correlated across good solutions, fixing one
provides information about the other.

Mathematical formulation:
    For population X ∈ R^(K×n) with K members and n variables:

    Correlation matrix: ρ_ij = Cov(X_i, X_j) / (σ_i * σ_j)

    Entanglement score: E_i = Σ_j |ρ_ij|  (sum of absolute correlations)

    High entanglement → fixing this variable affects many others
    Low entanglement → variable is relatively independent
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from scipy import sparse


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis.

    Attributes:
        min_samples: Minimum samples needed for reliable correlation
        regularization: Small value added to diagonal for stability
        use_rank: Use rank correlation (Spearman) instead of Pearson
        energy_weighted: Weight samples by inverse energy
    """

    min_samples: int = 5
    regularization: float = 1e-6
    use_rank: bool = False
    energy_weighted: bool = True


class CorrelationAnalyzer:
    """Analyzes correlation structure in population solutions.

    Computes:
    - Correlation matrix between variables
    - Entanglement scores (variable importance for branching)
    - Fractional scores (for traditional branching comparison)
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        """Initialize correlation analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or CorrelationConfig()

    def compute_correlation_matrix(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        integer_vars: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute weighted correlation matrix from population.

        Args:
            X: Population solutions, shape (K, n)
            weights: Optional weights for each sample, shape (K,)
            integer_vars: Indices of integer variables (compute only for these)

        Returns:
            Correlation matrix, shape (n_int, n_int) or (n, n)
        """
        K, n = X.shape

        # Select integer variables if specified
        if integer_vars is not None:
            X = X[:, integer_vars]
            n = len(integer_vars)

        # Check minimum samples
        if K < self.config.min_samples:
            # Return identity matrix if not enough samples
            return np.eye(n)

        # Apply weights
        if weights is None or not self.config.energy_weighted:
            weights = np.ones(K) / K
        else:
            weights = weights / np.sum(weights)

        # Use rank correlation if configured
        if self.config.use_rank:
            X = self._rank_transform(X)

        # Compute weighted mean
        mean = weights @ X  # (n,)

        # Compute weighted covariance
        X_centered = X - mean
        cov = (X_centered.T * weights) @ X_centered  # (n, n)

        # Add regularization for numerical stability
        cov += self.config.regularization * np.eye(n)

        # Compute correlation matrix
        std = np.sqrt(np.diag(cov))
        std[std < 1e-10] = 1.0  # Avoid division by zero

        corr = cov / (std[:, None] @ std[None, :])

        # Clamp to [-1, 1] for numerical stability
        corr = np.clip(corr, -1.0, 1.0)

        return corr

    def _rank_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to ranks for Spearman correlation.

        Args:
            X: Data matrix, shape (K, n)

        Returns:
            Rank-transformed matrix
        """
        K, n = X.shape
        X_rank = np.zeros_like(X)

        for j in range(n):
            ranks = np.argsort(np.argsort(X[:, j]))
            X_rank[:, j] = ranks / (K - 1)  # Normalize to [0, 1]

        return X_rank

    def compute_entanglement_scores(
        self,
        corr_matrix: np.ndarray,
        exclude_self: bool = True,
    ) -> np.ndarray:
        """Compute entanglement score for each variable.

        Entanglement score measures how much a variable is correlated
        with other variables. High entanglement means fixing this
        variable will affect many other variables.

        E_i = Σ_{j≠i} |ρ_ij|

        Args:
            corr_matrix: Correlation matrix, shape (n, n)
            exclude_self: Whether to exclude self-correlation

        Returns:
            Entanglement scores, shape (n,)
        """
        n = corr_matrix.shape[0]

        # Sum of absolute correlations
        abs_corr = np.abs(corr_matrix)

        if exclude_self:
            # Zero out diagonal
            abs_corr = abs_corr - np.diag(np.diag(abs_corr))

        scores = np.sum(abs_corr, axis=1)

        return scores

    def compute_fractional_scores(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        integer_vars: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute fractional score for each integer variable.

        Traditional branching heuristic: select variable closest to 0.5.
        We use weighted average distance from nearest integer.

        fractional_i = Σ_k w_k * |x_{ki} - round(x_{ki})|

        Args:
            X: Population solutions, shape (K, n)
            weights: Sample weights
            integer_vars: Indices of integer variables

        Returns:
            Fractional scores, shape (n_int,)
        """
        if integer_vars is not None:
            X = X[:, integer_vars]

        K, n = X.shape

        if weights is None:
            weights = np.ones(K) / K
        else:
            weights = weights / np.sum(weights)

        # Distance from nearest integer
        frac_dist = np.abs(X - np.round(X))

        # Weighted average
        scores = weights @ frac_dist

        return scores

    def compute_combined_scores(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        integer_vars: Optional[List[int]] = None,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute combined branching scores.

        Combines entanglement and fractional scores:
        score = alpha * normalized(entanglement) + (1-alpha) * normalized(fractional)

        Args:
            X: Population solutions
            weights: Sample weights
            integer_vars: Integer variable indices
            alpha: Weight for entanglement vs fractional

        Returns:
            Tuple of (combined_scores, entanglement_scores, fractional_scores)
        """
        # Compute correlation matrix
        corr = self.compute_correlation_matrix(X, weights, integer_vars)

        # Compute individual scores
        ent_scores = self.compute_entanglement_scores(corr)
        frac_scores = self.compute_fractional_scores(X, weights, integer_vars)

        # Normalize to [0, 1]
        def normalize(s):
            s_min, s_max = s.min(), s.max()
            if s_max - s_min < 1e-10:
                return np.ones_like(s) * 0.5
            return (s - s_min) / (s_max - s_min)

        ent_norm = normalize(ent_scores)
        frac_norm = normalize(frac_scores)

        # Combined score
        combined = alpha * ent_norm + (1 - alpha) * frac_norm

        return combined, ent_scores, frac_scores

    def select_branching_variable(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        integer_vars: Optional[List[int]] = None,
        alpha: float = 0.5,
        exclude_integral: bool = True,
        integral_tol: float = 1e-4,
    ) -> Tuple[int, float]:
        """Select the best variable for branching.

        Args:
            X: Population solutions, shape (K, n)
            weights: Sample weights
            integer_vars: Integer variable indices
            alpha: Entanglement weight
            exclude_integral: Exclude already-integral variables
            integral_tol: Tolerance for considering variable integral

        Returns:
            Tuple of (variable_index, score)
        """
        if integer_vars is None:
            integer_vars = list(range(X.shape[1]))

        # Compute scores
        combined, ent_scores, frac_scores = self.compute_combined_scores(
            X, weights, integer_vars, alpha
        )

        # Mask out integral variables
        if exclude_integral:
            # Check if variable is integral in all good solutions
            X_int = X[:, integer_vars]
            frac_dist = np.abs(X_int - np.round(X_int))
            max_frac = np.max(frac_dist, axis=0)
            integral_mask = max_frac < integral_tol

            # Set score to -inf for integral variables
            combined = combined.copy()
            combined[integral_mask] = -np.inf

        # Select best variable
        best_local_idx = np.argmax(combined)
        best_var = integer_vars[best_local_idx]
        best_score = combined[best_local_idx]

        return best_var, best_score

    def get_branching_value(
        self,
        X: np.ndarray,
        var_idx: int,
        weights: Optional[np.ndarray] = None,
        method: str = "weighted_median",
    ) -> float:
        """Get branching value for a variable.

        Args:
            X: Population solutions, shape (K, n)
            var_idx: Variable index
            weights: Sample weights
            method: 'weighted_median', 'mean', 'best_solution'

        Returns:
            Branching value
        """
        values = X[:, var_idx]

        if method == "mean":
            if weights is None:
                return np.mean(values)
            return np.sum(weights * values)

        elif method == "weighted_median":
            if weights is None:
                weights = np.ones(len(values)) / len(values)

            # Sort by value
            sorted_idx = np.argsort(values)
            sorted_weights = weights[sorted_idx]

            # Find median
            cumsum = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumsum, 0.5)

            return values[sorted_idx[median_idx]]

        elif method == "best_solution":
            if weights is None:
                # Use mean
                return np.mean(values)
            # Use value from best solution
            best_idx = np.argmax(weights)
            return values[best_idx]

        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test correlation analyzer
    print("Testing Correlation Analyzer")
    print("=" * 50)

    np.random.seed(42)

    # Create test population with correlated variables
    K, n = 20, 10
    integer_vars = [0, 1, 2, 3, 4]

    # Generate correlated data
    mean = np.zeros(n)
    cov = np.eye(n)
    # Add correlation between variables 0 and 1
    cov[0, 1] = cov[1, 0] = 0.8
    # Add correlation between variables 2 and 3
    cov[2, 3] = cov[3, 2] = -0.7

    X = np.random.multivariate_normal(mean, cov, K)
    # Clip to [0, 1] for binary-like behavior
    X = np.clip(X * 0.3 + 0.5, 0, 1)

    analyzer = CorrelationAnalyzer()

    # Compute correlation matrix
    corr = analyzer.compute_correlation_matrix(X, integer_vars=integer_vars)
    print("Correlation matrix (integer vars only):")
    print(np.round(corr, 2))

    # Compute entanglement scores
    ent = analyzer.compute_entanglement_scores(corr)
    print(f"\nEntanglement scores: {np.round(ent, 2)}")

    # Compute fractional scores
    frac = analyzer.compute_fractional_scores(X, integer_vars=integer_vars)
    print(f"Fractional scores: {np.round(frac, 2)}")

    # Select branching variable
    best_var, best_score = analyzer.select_branching_variable(
        X, integer_vars=integer_vars, alpha=0.5
    )
    print(f"\nBest branching variable: {best_var}, score: {best_score:.3f}")

    # Get branching value
    branch_val = analyzer.get_branching_value(X, best_var, method="weighted_median")
    print(f"Branching value: {branch_val:.3f}")
