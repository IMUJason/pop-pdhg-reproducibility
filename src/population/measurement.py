"""
Quantum-Inspired Measurement (Rounding) Operator for Population PDHG.

This module implements measurement operators that extract integer-feasible
solutions from the continuous population. The name "measurement" comes from
quantum mechanics where measurement causes wave function collapse.

Key concepts:
- Born rule selection: Select population members with probability proportional
  to exp(-energy/T), not just the best
- Probabilistic rounding: Round continuous values probabilistically to maintain
  expected value
- Population vote: Use population statistics to make rounding decisions

These techniques help avoid commitment to a single solution too early and
leverage population diversity for better integer solutions.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from scipy import sparse


@dataclass
class MeasurementConfig:
    """Configuration for measurement operator.

    Attributes:
        temperature: Temperature for Born rule selection (higher = more exploration)
        rounding_strategy: 'deterministic', 'probabilistic', or 'population_vote'
        repair_infeasible: Whether to attempt repairing infeasible solutions
    """

    temperature: float = 1.0
    rounding_strategy: str = "probabilistic"
    repair_infeasible: bool = True


class QuantumMeasurement:
    """Quantum-inspired measurement operator for extracting integer solutions.

    Implements Born rule selection and probabilistic rounding to convert
    continuous relaxation solutions to integer-feasible solutions.

    Example:
        >>> measurement = QuantumMeasurement()
        >>> integer_solutions = measurement.measure(state, integer_vars, A, b, c)
    """

    def __init__(self, config: Optional[MeasurementConfig] = None):
        """Initialize measurement operator.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or MeasurementConfig()

    def born_rule_selection(
        self,
        energies: np.ndarray,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Select a population member using Born rule probabilities.

        P(select i) ∝ exp(-E_i / T)

        This is equivalent to Boltzmann sampling. Lower energy members
        have higher probability, but temperature controls exploration.

        Args:
            energies: Energy values for each member, shape (K,)
            temperature: Sampling temperature. Uses config if None.
            seed: Random seed

        Returns:
            Index of selected member
        """
        if seed is not None:
            np.random.seed(seed)

        T = temperature if temperature is not None else self.config.temperature

        # Shift energies for numerical stability
        E_shifted = energies - np.min(energies)

        # Compute probabilities
        log_probs = -E_shifted / T
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)

        # Sample
        return np.random.choice(len(energies), p=probs)

    def probabilistic_rounding(
        self,
        x_continuous: np.ndarray,
        integer_vars: List[int],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Round continuous solution probabilistically.

        For each integer variable x_i with fractional part f:
            P(x_i = floor) = 1 - f
            P(x_i = ceil) = f

        This preserves the expected value E[x_i] = x_i_continuous.

        Args:
            x_continuous: Continuous solution, shape (n,)
            integer_vars: Indices of integer variables
            seed: Random seed

        Returns:
            Rounded solution, shape (n,)
        """
        if seed is not None:
            np.random.seed(seed)

        x_int = x_continuous.copy()

        for i in integer_vars:
            val = x_int[i]
            floor_val = np.floor(val)
            ceil_val = np.ceil(val)
            frac = val - floor_val

            if np.random.random() < frac:
                x_int[i] = ceil_val
            else:
                x_int[i] = floor_val

        return x_int

    def deterministic_rounding(
        self, x_continuous: np.ndarray, integer_vars: List[int]
    ) -> np.ndarray:
        """Round to nearest integer.

        Args:
            x_continuous: Continuous solution
            integer_vars: Indices of integer variables

        Returns:
            Rounded solution
        """
        x_int = x_continuous.copy()
        x_int[integer_vars] = np.round(x_int[integer_vars])
        return x_int

    def population_vote(
        self,
        state: "PopulationState",
        integer_vars: List[int],
        energies: np.ndarray,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Determine integer values by weighted population vote.

        Each population member votes for integer values, weighted by
        their Boltzmann weight exp(-E_i/T).

        For each integer variable j:
            - Compute weighted average of population values
            - Round to nearest integer

        Args:
            state: Population state
            integer_vars: Indices of integer variables
            energies: Energy values for each member
            temperature: Voting temperature
            seed: Random seed

        Returns:
            Integer solution from voting
        """
        if seed is not None:
            np.random.seed(seed)

        T = temperature if temperature is not None else self.config.temperature
        K = state.K

        # Compute Boltzmann weights
        E_shifted = energies - np.min(energies)
        log_weights = -E_shifted / T
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        # Weighted average of integer variables
        x_voted = np.zeros(state.n)

        # For non-integer variables, use weighted average
        all_vars = set(range(state.n))
        non_integer_vars = list(all_vars - set(integer_vars))

        if non_integer_vars:
            x_voted[non_integer_vars] = weights @ state.x[:, non_integer_vars]

        # For integer variables, use weighted vote
        for j in integer_vars:
            # Get population values for this variable
            vals = state.x[:, j]

            # Weighted average
            weighted_avg = weights @ vals

            # Round to nearest integer
            x_voted[j] = np.round(weighted_avg)

        return x_voted

    def check_feasibility(
        self,
        x: np.ndarray,
        A: sparse.csr_matrix,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: List[int],
        tol: float = 1e-6,
    ) -> Tuple[bool, float, float]:
        """Check if a solution is feasible.

        Args:
            x: Solution to check
            A, b, lb, ub: Problem data
            integer_vars: Indices of integer variables
            tol: Tolerance for feasibility checks

        Returns:
            Tuple of (is_feasible, primal_violation, integrality_violation)
        """
        # Check bounds
        if np.any(x < lb - tol) or np.any(x > ub + tol):
            return False, float("inf"), float("inf")

        # Check constraints
        Ax = A @ x
        primal_violation = np.max(np.maximum(Ax - b, 0))

        # Check integrality
        if integer_vars:
            integrality_violation = np.max(
                np.minimum(
                    x[integer_vars] - np.floor(x[integer_vars]),
                    np.ceil(x[integer_vars]) - x[integer_vars],
                )
            )
        else:
            integrality_violation = 0.0

        is_feasible = (
            primal_violation <= tol
            and integrality_violation <= tol
        )

        return is_feasible, primal_violation, integrality_violation

    def simple_repair(
        self,
        x: np.ndarray,
        A: sparse.csr_matrix,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: List[int],
        max_iters: int = 100,
    ) -> np.ndarray:
        """Attempt simple repair of infeasible solution.

        For bound violations: clip to bounds
        For constraint violations: reduce variables proportionally

        Args:
            x: Potentially infeasible solution
            A, b, lb, ub: Problem data
            integer_vars: Indices of integer variables
            max_iters: Maximum repair iterations

        Returns:
            Repaired solution (may still be infeasible)
        """
        x_repaired = x.copy()

        # Clip to bounds
        x_repaired = np.clip(x_repaired, lb, ub)

        # Round integers
        x_repaired[integer_vars] = np.round(x_repaired[integer_vars])

        # Simple constraint repair: reduce variables proportionally
        for _ in range(max_iters):
            Ax = A @ x_repaired
            violations = Ax - b
            max_violation = np.max(violations)

            if max_violation <= 1e-6:
                break

            # Find most violated constraint
            most_violated = np.argmax(violations)

            # Reduce variables in that constraint
            row = A.getrow(most_violated).toarray().flatten()
            for j in range(len(row)):
                coef = row[j]
                if coef > 0:
                    # Reduce this variable
                    reduction = min(max_violation / coef, x_repaired[j] - lb[j])
                    x_repaired[j] -= reduction

            # Re-clip and round
            x_repaired = np.clip(x_repaired, lb, ub)
            x_repaired[integer_vars] = np.round(x_repaired[integer_vars])

        return x_repaired

    def measure(
        self,
        state: "PopulationState",
        integer_vars: List[int],
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        energies: Optional[np.ndarray] = None,
        n_samples: int = 5,
        seed: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, float, bool]]:
        """Extract integer-feasible solutions from population.

        Uses Born rule to select population members, then applies
        the configured rounding strategy.

        Args:
            state: Population state
            integer_vars: Indices of integer variables
            A, b, c, lb, ub: Problem data
            energies: Pre-computed energies (computed if None)
            n_samples: Number of solutions to extract
            seed: Random seed

        Returns:
            List of (solution, objective, is_feasible) tuples
        """
        if seed is not None:
            np.random.seed(seed)

        # Compute energies if not provided
        if energies is None:
            # Simple energy: objective + penalty for constraint violation
            Ax = (A @ state.x.T).T  # (K, m)
            violation = np.maximum(Ax - b, 0)
            penalty = 1000 * np.sum(violation**2, axis=1)
            energies = state.obj + penalty

        solutions = []

        for sample_idx in range(n_samples):
            # Select member using Born rule
            selected_idx = self.born_rule_selection(energies, seed=seed)

            # Get continuous solution
            x_cont = state.x[selected_idx].copy()

            # Apply rounding strategy
            if self.config.rounding_strategy == "deterministic":
                x_int = self.deterministic_rounding(x_cont, integer_vars)
            elif self.config.rounding_strategy == "probabilistic":
                x_int = self.probabilistic_rounding(
                    x_cont, integer_vars, seed=seed
                )
            elif self.config.rounding_strategy == "population_vote":
                x_int = self.population_vote(
                    state, integer_vars, energies, seed=seed
                )
            else:
                raise ValueError(
                    f"Unknown rounding strategy: {self.config.rounding_strategy}"
                )

            # Clip to bounds
            x_int = np.clip(x_int, lb, ub)

            # Check feasibility
            is_feas, primal_viol, int_viol = self.check_feasibility(
                x_int, A, b, lb, ub, integer_vars
            )

            # Attempt repair if infeasible
            if not is_feas and self.config.repair_infeasible:
                x_int = self.simple_repair(
                    x_int, A, b, lb, ub, integer_vars
                )
                is_feas, primal_viol, int_viol = self.check_feasibility(
                    x_int, A, b, lb, ub, integer_vars
                )

            # Compute objective
            obj = c @ x_int

            solutions.append((x_int, obj, is_feas))

        # Sort by objective (best first)
        solutions.sort(key=lambda s: s[1])

        return solutions


if __name__ == "__main__":
    # Test measurement operator
    print("Testing Quantum Measurement Operator...")

    from src.population.pop_pdhg import PopulationPDHG, PopulationState
    from scipy import sparse

    # Simple MIP: min -x - y s.t. x + y <= 1, x, y binary
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    integer_vars = [0, 1]

    # Initialize population
    solver = PopulationPDHG(A, b, c, lb, ub, population_size=16)
    state = solver.initialize_population(strategy="diverse", seed=42)

    # Run a few iterations
    for _ in range(100):
        state = solver.batch_pdhg_step(state)

    print(f"Population objectives: {state.obj}")

    # Apply measurement
    measurement = QuantumMeasurement(
        config=MeasurementConfig(
            temperature=0.5,
            rounding_strategy="probabilistic",
            repair_infeasible=True,
        )
    )

    solutions = measurement.measure(
        state, integer_vars, A, b, c, lb, ub, n_samples=10, seed=42
    )

    print(f"\nExtracted integer solutions:")
    for i, (x, obj, feas) in enumerate(solutions):
        feas_str = "feasible" if feas else "infeasible"
        print(f"  {i+1}: x={x}, obj={obj:.4f}, {feas_str}")
