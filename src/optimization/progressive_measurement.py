"""
Progressive Measurement for Integer Variables.

This module implements quantum-inspired progressive measurement for
gradually collapsing continuous relaxations to integer solutions.

Key concepts:
- Weak measurement: Partial collapse toward integer values
- Progressive strengthening: Gradually increase measurement strength
- Born rule: Probability-based selection from superposition
- Entanglement-aware: Consider variable correlations during measurement

References:
    - Quantum measurement theory
    - Progressive rounding in MIP
    - Born rule probability interpretation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import numpy as np


@dataclass
class MeasurementConfig:
    """Configuration for progressive measurement.

    Attributes:
        initial_strength: Initial measurement strength (0 = no collapse)
        final_strength: Final measurement strength (1 = full collapse)
        schedule: Strength schedule ("linear", "exponential", "cosine")
        entanglement_aware: Use correlation-aware measurement order
        born_rule_selection: Use probabilistic selection vs greedy
        adaptive: Adapt strength based on convergence
    """
    initial_strength: float = 0.1
    final_strength: float = 1.0
    schedule: str = "cosine"  # linear, exponential, cosine
    entanglement_aware: bool = True
    born_rule_selection: bool = True
    adaptive: bool = True
    min_improvement_threshold: float = 1e-6


@dataclass
class MeasurementStats:
    """Statistics for measurement operations."""
    total_measurements: int = 0
    integer_violations_before: float = 0.0
    integer_violations_after: float = 0.0
    obj_improvements: int = 0
    measurement_history: List[dict] = field(default_factory=list)


class ProgressiveMeasurement:
    """Progressive measurement for integer variable collapse.

    Implements quantum-inspired measurement that gradually collapses
    continuous relaxations toward integer solutions:

    1. Early iterations: Weak measurement (exploration)
    2. Mid iterations: Moderate measurement (refinement)
    3. Late iterations: Strong measurement (commitment)

    The collapse follows:
        x_measured = (1 - strength) * x_continuous + strength * round(x_continuous)

    Example:
        >>> config = MeasurementConfig(initial_strength=0.1, final_strength=1.0)
        >>> measurement = ProgressiveMeasurement(config)
        >>> x_new = measurement.measure(x, integer_vars=[0, 1, 2], progress=0.5)
    """

    def __init__(
        self,
        config: Optional[MeasurementConfig] = None,
        num_integer_vars: int = 0,
    ):
        """Initialize progressive measurement.

        Args:
            config: Measurement configuration
            num_integer_vars: Number of integer variables
        """
        self.config = config or MeasurementConfig()
        self.num_integer_vars = num_integer_vars

        # Statistics
        self.stats = MeasurementStats()

        # Current strength
        self.current_strength = self.config.initial_strength

        # Correlation matrix for entanglement-aware measurement
        self.correlation_matrix: Optional[np.ndarray] = None

        # History of fractionalities for adaptive adjustment
        self.fract_history: List[float] = []

    def compute_strength(self, progress: float) -> float:
        """Compute measurement strength based on progress.

        Args:
            progress: Solving progress in [0, 1]

        Returns:
            Current measurement strength
        """
        if self.config.schedule == "linear":
            strength = (
                self.config.initial_strength +
                progress * (self.config.final_strength - self.config.initial_strength)
            )

        elif self.config.schedule == "exponential":
            # Faster increase toward end
            t = progress ** 2
            strength = (
                self.config.initial_strength +
                t * (self.config.final_strength - self.config.initial_strength)
            )

        elif self.config.schedule == "cosine":
            # Smooth S-curve transition
            import math
            t = (1 - math.cos(math.pi * progress)) / 2
            strength = (
                self.config.initial_strength +
                t * (self.config.final_strength - self.config.initial_strength)
            )

        else:
            strength = (
                self.config.initial_strength +
                progress * (self.config.final_strength - self.config.initial_strength)
            )

        self.current_strength = strength
        return strength

    def compute_fractality(self, x: np.ndarray, integer_vars: List[int]) -> np.ndarray:
        """Compute fractional distance from integers for each variable.

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables

        Returns:
            Array of fractional distances
        """
        fract = np.zeros(len(integer_vars))
        for i, var_idx in enumerate(integer_vars):
            val = x[var_idx]
            fract[i] = abs(val - round(val))
        return fract

    def compute_measurement_order(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Determine optimal order for measuring integer variables.

        Uses entanglement-aware ordering:
        1. First measure variables that are already "close" to integer
        2. Then measure variables with low entanglement (fewer correlations)
        3. Last measure highly entangled variables

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables
            correlation_matrix: Variable correlation matrix

        Returns:
            Ordered list of variable indices
        """
        if not self.config.entanglement_aware or correlation_matrix is None:
            # Default: measure in given order
            return list(range(len(integer_vars)))

        # Compute fractionalities
        fract = self.compute_fractality(x, integer_vars)

        # Compute entanglement (sum of absolute correlations)
        entanglement = np.zeros(len(integer_vars))
        for i, var_idx in enumerate(integer_vars):
            if var_idx < correlation_matrix.shape[0]:
                entanglement[i] = np.sum(np.abs(correlation_matrix[var_idx, :]))

        # Normalize
        fract_norm = fract / (np.max(fract) + 1e-10)
        entanglement_norm = entanglement / (np.max(entanglement) + 1e-10)

        # Priority: low fractality (close to integer) and low entanglement
        priority = 0.5 * fract_norm + 0.5 * entanglement_norm

        # Sort by priority (lower = measure first)
        order = np.argsort(priority)

        return list(order)

    def born_rule_select(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        temperature: float = 1.0,
    ) -> int:
        """Select variable to measure using Born rule probability.

        Variables with fractional values closer to 0.5 have higher
        probability of being selected (maximum uncertainty principle).

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables
            temperature: Temperature for softmax

        Returns:
            Index of selected variable
        """
        fract = self.compute_fractality(x, integer_vars)

        # Uncertainty is maximized at 0.5 fractional distance
        uncertainty = 1.0 - 2.0 * np.abs(fract - 0.5)

        # Softmax selection
        weights = np.exp(uncertainty / temperature)
        probs = weights / np.sum(weights)

        return np.random.choice(len(integer_vars), p=probs)

    def measure_single(
        self,
        x: np.ndarray,
        var_idx: int,
        strength: float,
        born_rule: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Measure a single variable (partial or full collapse).

        Args:
            x: Solution vector
            var_idx: Index of variable to measure
            strength: Measurement strength in [0, 1]
            born_rule: Use probabilistic rounding
            temperature: Temperature for Born rule

        Returns:
            Updated solution vector
        """
        x_new = x.copy()
        val = x_new[var_idx]

        if born_rule:
            # Probabilistic rounding based on fractional part
            floor_val = np.floor(val)
            ceil_val = np.ceil(val)

            # Probability of rounding up
            p_up = val - floor_val

            # Apply measurement strength
            effective_p_up = (1 - strength) * 0.5 + strength * p_up

            if np.random.random() < effective_p_up:
                target = ceil_val
            else:
                target = floor_val
        else:
            # Deterministic rounding
            target = np.round(val)

        # Partial collapse
        x_new[var_idx] = (1 - strength) * val + strength * target

        return x_new

    def measure(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        progress: float,
        correlation_matrix: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply progressive measurement to integer variables.

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables
            progress: Solving progress in [0, 1]
            correlation_matrix: Variable correlation matrix
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Measured solution vector
        """
        if not integer_vars:
            return x

        # Compute current strength
        strength = self.compute_strength(progress)

        # Update statistics
        self.stats.total_measurements += 1
        fract_before = self.compute_fractality(x, integer_vars)
        self.stats.integer_violations_before += np.mean(fract_before)

        # Record fractality history
        self.fract_history.append(np.mean(fract_before))

        x_new = x.copy()

        # Determine measurement order
        order = self.compute_measurement_order(x, integer_vars, correlation_matrix)

        # Measure each variable
        for idx in order:
            var_idx = integer_vars[idx]

            if self.config.born_rule_selection:
                x_new = self.measure_single(
                    x_new, var_idx, strength,
                    born_rule=True,
                    temperature=max(0.1, 1.0 - progress)
                )
            else:
                x_new = self.measure_single(
                    x_new, var_idx, strength,
                    born_rule=False
                )

        # Enforce bounds
        if lb is not None and ub is not None:
            x_new = np.clip(x_new, lb, ub)

        # Final rounding if strength is high enough
        if strength > 0.9:
            for var_idx in integer_vars:
                x_new[var_idx] = np.round(x_new[var_idx])

        # Update statistics
        fract_after = self.compute_fractality(x_new, integer_vars)
        self.stats.integer_violations_after += np.mean(fract_after)

        self.stats.measurement_history.append({
            "progress": progress,
            "strength": strength,
            "fract_before": np.mean(fract_before),
            "fract_after": np.mean(fract_after),
        })

        return x_new

    def batch_measure(
        self,
        X: np.ndarray,
        integer_vars: List[int],
        progress: float,
        correlation_matrix: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply measurement to a batch of solutions.

        Args:
            X: Solution matrix (K x n)
            integer_vars: Indices of integer variables
            progress: Solving progress
            correlation_matrix: Variable correlation matrix
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Measured solution matrix
        """
        X_new = X.copy()
        for k in range(X.shape[0]):
            X_new[k] = self.measure(
                X_new[k], integer_vars, progress,
                correlation_matrix, lb, ub
            )
        return X_new

    def adapt_strength(self, improvement: float) -> None:
        """Adaptively adjust measurement strength.

        Args:
            improvement: Recent objective improvement
        """
        if not self.config.adaptive:
            return

        if improvement < self.config.min_improvement_threshold:
            # Not improving, increase strength faster
            self.config.initial_strength = min(
                0.5, self.config.initial_strength * 1.1
            )
        else:
            # Improving, keep current schedule
            pass

    def get_stats(self) -> dict:
        """Get measurement statistics."""
        avg_before = (
            self.stats.integer_violations_before / self.stats.total_measurements
            if self.stats.total_measurements > 0 else 0
        )
        avg_after = (
            self.stats.integer_violations_after / self.stats.total_measurements
            if self.stats.total_measurements > 0 else 0
        )

        return {
            "total_measurements": self.stats.total_measurements,
            "avg_fract_before": avg_before,
            "avg_fract_after": avg_after,
            "fract_reduction": avg_before - avg_after,
            "current_strength": self.current_strength,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = MeasurementStats()
        self.fract_history = []


class EntangledMeasurement(ProgressiveMeasurement):
    """Entanglement-aware measurement for correlated integer variables.

    Extends progressive measurement with:
    1. Correlation-aware measurement order
    2. Conditional probability updates after each measurement
    3. Belief propagation for inference

    Example:
        >>> measurement = EntangledMeasurement(config)
        >>> x_new = measurement.measure(x, integer_vars, progress=0.5,
        ...                              correlation_matrix=corr)
    """

    def __init__(
        self,
        config: Optional[MeasurementConfig] = None,
        num_integer_vars: int = 0,
    ):
        """Initialize entangled measurement."""
        super().__init__(config, num_integer_vars)

        # Belief state for each variable
        self.beliefs: Optional[np.ndarray] = None

    def initialize_beliefs(self, x: np.ndarray, integer_vars: List[int]) -> None:
        """Initialize belief states for integer variables.

        Belief is probability distribution over {floor, ceil}.

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables
        """
        n = len(integer_vars)
        self.beliefs = np.zeros(n)

        for i, var_idx in enumerate(integer_vars):
            val = x[var_idx]
            # Probability of being "up" (ceil)
            self.beliefs[i] = val - np.floor(val)

    def update_beliefs(
        self,
        measured_idx: int,
        measured_value: float,
        correlation_matrix: np.ndarray,
        integer_vars: List[int],
    ) -> None:
        """Update beliefs after measuring a variable.

        Uses correlation to update beliefs of unmeasured variables.

        Args:
            measured_idx: Index of measured variable in integer_vars
            measured_value: Value after measurement
            correlation_matrix: Full variable correlation matrix
            integer_vars: Indices of integer variables
        """
        if self.beliefs is None:
            return

        var_idx = integer_vars[measured_idx]

        for i, other_idx in enumerate(integer_vars):
            if i == measured_idx:
                continue

            # Get correlation between measured and this variable
            if var_idx < correlation_matrix.shape[0] and other_idx < correlation_matrix.shape[1]:
                corr = correlation_matrix[var_idx, other_idx]
            else:
                corr = 0

            # Update belief (simple correlation-based update)
            # Positive correlation: if measured is high, increase belief
            # Negative correlation: if measured is high, decrease belief
            measured_rounded = round(measured_value)
            is_up = measured_value - np.floor(measured_value) > 0.5

            if is_up:
                self.beliefs[i] += 0.1 * corr
            else:
                self.beliefs[i] -= 0.1 * corr

            # Clamp to [0, 1]
            self.beliefs[i] = np.clip(self.beliefs[i], 0, 1)

    def measure_entangled(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        progress: float,
        correlation_matrix: np.ndarray,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply entanglement-aware sequential measurement.

        Args:
            x: Solution vector
            integer_vars: Indices of integer variables
            progress: Solving progress
            correlation_matrix: Variable correlation matrix
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Measured solution vector
        """
        if not integer_vars:
            return x

        # Compute strength
        strength = self.compute_strength(progress)

        # Initialize beliefs
        self.initialize_beliefs(x, integer_vars)

        x_new = x.copy()
        remaining = set(range(len(integer_vars)))

        # Sequential measurement
        while remaining:
            # Select next variable to measure
            if self.config.born_rule_selection and np.random.random() < 0.5:
                # Born rule selection
                remaining_list = list(remaining)
                fract = np.array([
                    abs(x_new[integer_vars[i]] - round(x_new[integer_vars[i]]))
                    for i in remaining_list
                ])
                uncertainty = 1.0 - 2.0 * np.abs(fract - 0.5)
                temp = max(0.1, 1.0 - progress)
                weights = np.exp(uncertainty / temp)
                probs = weights / np.sum(weights)
                local_idx = np.random.choice(len(remaining_list), p=probs)
                idx = remaining_list[local_idx]
            else:
                # Greedy: select variable closest to integer
                idx = min(remaining, key=lambda i: abs(
                    x_new[integer_vars[i]] - round(x_new[integer_vars[i]])
                ))

            var_idx = integer_vars[idx]

            # Measure this variable
            old_value = x_new[var_idx]
            x_new = self.measure_single(x_new, var_idx, strength)

            # Update beliefs of remaining variables
            if correlation_matrix is not None:
                self.update_beliefs(idx, x_new[var_idx], correlation_matrix, integer_vars)

            # Remove from remaining
            remaining.remove(idx)

        # Enforce bounds
        if lb is not None and ub is not None:
            x_new = np.clip(x_new, lb, ub)

        # Final rounding if strength is high
        if strength > 0.9:
            for var_idx in integer_vars:
                x_new[var_idx] = np.round(x_new[var_idx])

        return x_new


if __name__ == "__main__":
    print("Testing Progressive Measurement...")

    # Test with a simple MIP
    np.random.seed(42)

    # Create test solution with fractional integer variables
    x = np.array([0.3, 0.7, 2.4, 3.6, 1.5, 4.2])
    integer_vars = [0, 1, 2, 3, 4, 5]

    # Create mock correlation matrix
    n = len(x)
    correlation_matrix = np.random.randn(n, n) * 0.5
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    # Test basic progressive measurement
    config = MeasurementConfig(
        initial_strength=0.1,
        final_strength=1.0,
        schedule="cosine",
        entanglement_aware=True,
        born_rule_selection=True,
    )

    measurement = ProgressiveMeasurement(config)

    print("\nTesting strength schedules:")
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        strength = measurement.compute_strength(progress)
        print(f"  Progress {progress:.2f}: strength = {strength:.4f}")

    print("\nTesting measurement at different progress levels:")
    for progress in [0.0, 0.5, 1.0]:
        x_measured = measurement.measure(x, integer_vars, progress, correlation_matrix)
        fract = measurement.compute_fractality(x_measured, integer_vars)
        print(f"  Progress {progress:.2f}: x = {x_measured}, fract = {np.mean(fract):.4f}")

    print("\nTesting entangled measurement:")
    entangled = EntangledMeasurement(config)
    x_entangled = entangled.measure_entangled(x, integer_vars, 0.8, correlation_matrix)
    print(f"  Result: {x_entangled}")

    print("\nStatistics:")
    print(f"  {measurement.get_stats()}")

    print("\nProgressive measurement test complete!")
