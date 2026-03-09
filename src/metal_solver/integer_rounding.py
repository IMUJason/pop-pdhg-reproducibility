"""
Integer Rounding and Local Search for MIP

Progressive measurement and feasibility repair.
"""

import numpy as np
from typing import List, Optional


class ProgressiveMeasurement:
    """Progressive integer measurement (quantum-inspired rounding)."""

    def __init__(self, integer_vars: List[int], schedule='cosine'):
        """Initialize progressive measurement.

        Args:
            integer_vars: Indices of integer variables
            schedule: Rounding schedule ('cosine', 'linear', 'exponential')
        """
        self.integer_vars = integer_vars
        self.schedule = schedule

    def get_strength(self, iteration: int, max_iter: int) -> float:
        """Get measurement strength at current iteration.

        Returns:
            Strength in [0, 1] - 0 means no rounding, 1 means full rounding
        """
        progress = iteration / max_iter

        if self.schedule == 'cosine':
            # Cosine annealing: starts slow, accelerates
            return 0.5 * (1 - np.cos(progress * np.pi))
        elif self.schedule == 'linear':
            return progress
        elif self.schedule == 'exponential':
            return 1 - np.exp(-3 * progress)
        else:
            return progress

    def measure(self, x: np.ndarray, iteration: int, max_iter: int) -> np.ndarray:
        """Apply progressive measurement to integer variables.

        Args:
            x: Current solution (n,) or (K, n)
            iteration: Current iteration
            max_iter: Maximum iterations

        Returns:
            Partially rounded solution
        """
        strength = self.get_strength(iteration, max_iter)

        x_new = x.copy()

        if x.ndim == 1:
            # Single vector
            for i in self.integer_vars:
                target = np.round(x[i])
                x_new[i] = (1 - strength) * x[i] + strength * target
        else:
            # Batch
            for i in self.integer_vars:
                target = np.round(x[:, i])
                x_new[:, i] = (1 - strength) * x[:, i] + strength * target

        return x_new

    def finalize(self, x: np.ndarray) -> np.ndarray:
        """Final hard rounding to integers.

        Args:
            x: Solution to round

        Returns:
            Fully integer solution
        """
        x_new = x.copy()

        if x.ndim == 1:
            for i in self.integer_vars:
                x_new[i] = np.round(x[i])
        else:
            for i in self.integer_vars:
                x_new[:, i] = np.round(x[:, i])

        return x_new


class ConstraintAwareRepair:
    """Constraint-aware local search for integer feasibility repair.

    Uses gradient information to prioritize variables that most affect violated constraints.
    """

    def __init__(self, A, b, integer_vars, max_iter=100, constraint_sense=None):
        """Initialize constraint-aware repair.

        Args:
            A: Constraint matrix (m x n)
            b: RHS vector (m,)
            integer_vars: Integer variable indices
            max_iter: Maximum repair iterations
            constraint_sense: Optional list of constraint senses ('<', '>', '=')
        """
        self.A = A
        self.b = b
        self.integer_vars = np.array(integer_vars)
        self.max_iter = max_iter
        self.constraint_sense = constraint_sense
        self.m, self.n = A.shape

        # Pre-compute variable-constraint influence matrix
        self._compute_influence_matrix()

    def _compute_influence_matrix(self):
        """Pre-compute how much each variable affects each constraint."""
        # Influence matrix: |A[i,j]| for each constraint i and variable j
        self.influence = np.abs(self.A.toarray() if hasattr(self.A, 'toarray') else self.A)

        # For each constraint, find the most influential variables
        self.top_vars_per_constraint = {}
        for i in range(self.m):
            # Get indices of integer variables sorted by influence
            int_influences = [(j, self.influence[i, j]) for j in self.integer_vars]
            int_influences.sort(key=lambda x: x[1], reverse=True)
            self.top_vars_per_constraint[i] = [j for j, _ in int_influences[:10]]  # Top 10

    def compute_violation(self, x):
        """Compute constraint violation for each constraint."""
        Ax = self.A @ x

        if self.constraint_sense is None:
            return np.maximum(Ax - self.b, 0)

        violations = np.zeros(self.m)
        for i, s in enumerate(self.constraint_sense):
            if s == '<':
                violations[i] = max(Ax[i] - self.b[i], 0)
            elif s == '>':
                violations[i] = max(self.b[i] - Ax[i], 0)
            elif s == '=':
                violations[i] = abs(Ax[i] - self.b[i])
            else:
                violations[i] = max(Ax[i] - self.b[i], 0)

        return violations

    def compute_total_violation(self, x):
        """Total constraint violation."""
        return np.sum(self.compute_violation(x))

    def _get_gradient_direction(self, x, var_idx):
        """Compute gradient direction for a variable to reduce violations.

        Returns: direction (+1, -1, or 0) and expected improvement
        """
        violations = self.compute_violation(x)

        if np.sum(violations) < 1e-10:
            return 0, 0

        # Get current value and rounded value
        current_val = x[var_idx]
        rounded_val = np.round(current_val)

        # Try both directions
        improvements = {}

        for delta in [-1, 0, 1]:
            x_test = x.copy()
            new_val = rounded_val + delta
            x_test[var_idx] = new_val

            new_violations = self.compute_violation(x_test)
            improvement = np.sum(violations) - np.sum(new_violations)
            improvements[delta] = improvement

        # Choose best direction
        best_delta = max(improvements, key=improvements.get)
        return best_delta, improvements[best_delta]

    def repair(self, x, max_rounds=10):
        """Repair solution using constraint-aware local search.

        Args:
            x: Infeasible solution
            max_rounds: Maximum repair rounds

        Returns:
            (repaired_x, is_feasible)
        """
        x_current = x.copy()

        for round_idx in range(max_rounds):
            # Round all integer variables first
            for i in self.integer_vars:
                x_current[i] = np.round(x_current[i])

            violations = self.compute_violation(x_current)
            total_viol = np.sum(violations)

            if total_viol < 1e-6:
                return x_current, True

            # Identify most violated constraints
            violated_constraints = np.where(violations > 1e-10)[0]

            if len(violated_constraints) == 0:
                return x_current, True

            # Sort by violation amount
            violated_constraints = sorted(violated_constraints,
                                          key=lambda i: violations[i],
                                          reverse=True)

            # Try to fix top violated constraints
            improved = False

            for cons_idx in violated_constraints[:5]:  # Focus on top 5
                # Get influential variables for this constraint
                influential_vars = self.top_vars_per_constraint[cons_idx]

                for var_idx in influential_vars:
                    # Try different values for this variable
                    current_val = x_current[var_idx]

                    for delta in [-2, -1, 0, 1, 2]:
                        x_test = x_current.copy()
                        x_test[var_idx] = np.round(current_val) + delta

                        new_violations = self.compute_violation(x_test)
                        new_total = np.sum(new_violations)

                        if new_total < total_viol - 1e-10:
                            x_current = x_test
                            total_viol = new_total
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if not improved:
                # No improvement found, try random perturbation
                var_to_perturb = np.random.choice(self.integer_vars)
                current_val = x_current[var_to_perturb]
                x_current[var_to_perturb] = np.round(current_val) + np.random.choice([-2, -1, 1, 2])

        # Final check
        final_violations = self.compute_violation(x_current)
        is_feasible = np.sum(final_violations) < 1e-6

        return x_current, is_feasible


class LocalSearchRepair:
    """Local search to repair integer feasibility."""

    def __init__(self, A, b, integer_vars, max_iter=100, constraint_sense=None):
        """Initialize local search.

        Args:
            A: Constraint matrix
            b: RHS
            integer_vars: Integer variable indices
            max_iter: Maximum local search iterations
            constraint_sense: Optional list of constraint senses ('<', '>', '=')
                              If provided, violations are computed in original problem space
        """
        self.A = A
        self.b = b
        self.integer_vars = integer_vars
        self.max_iter = max_iter
        self.constraint_sense = constraint_sense

        # Use constraint-aware repair as the primary method
        self._constraint_aware = ConstraintAwareRepair(
            A, b, integer_vars, max_iter, constraint_sense
        )

    def compute_violation(self, x):
        """Compute constraint violation.

        If constraint_sense is provided, computes violations in original problem space.
        Otherwise, assumes standard form Ax <= b.
        """
        Ax = self.A @ x

        if self.constraint_sense is None:
            # Standard form: Ax <= b
            return np.maximum(Ax - self.b, 0)

        # Original problem space: handle different constraint types
        violations = np.zeros(len(self.b))
        for i, s in enumerate(self.constraint_sense):
            if s == '<':
                # Ax <= b
                violations[i] = max(Ax[i] - self.b[i], 0)
            elif s == '>':
                # Ax >= b  =>  b - Ax <= 0
                violations[i] = max(self.b[i] - Ax[i], 0)
            elif s == '=':
                # Ax = b  =>  |Ax - b|
                violations[i] = abs(Ax[i] - self.b[i])
            else:
                # Default to <=
                violations[i] = max(Ax[i] - self.b[i], 0)

        return violations

    def compute_total_violation(self, x):
        """Total constraint violation."""
        viol = self.compute_violation(x)
        return np.sum(viol)

    def one_opt_search(self, x, direction='minimize'):
        """1-OPT local search: try flipping each integer variable.

        Args:
            x: Current solution
            direction: 'minimize' or 'any_feasible'

        Returns:
            Best found solution
        """
        x_best = x.copy()
        viol_best = self.compute_total_violation(x_best)

        for idx in self.integer_vars:
            # Try current value +/- 1
            for delta in [-1, 1]:
                x_test = x.copy()
                x_test[idx] = np.round(x_test[idx]) + delta

                # Check if within bounds (if bounds exist)
                # TODO: Add bound checking

                viol_test = self.compute_total_violation(x_test)

                if viol_test < viol_best:
                    viol_best = viol_test
                    x_best = x_test

                    if direction == 'any_feasible' and viol_best < 1e-6:
                        return x_best

        return x_best

    def two_opt_search(self, x, max_pairs=100):
        """2-OPT local search: try swapping pairs of integer variables.

        Args:
            x: Current solution
            max_pairs: Maximum number of pairs to try

        Returns:
            Best found solution
        """
        x_best = x.copy()
        viol_best = self.compute_total_violation(x_best)

        # Random sample of pairs
        n_int = len(self.integer_vars)
        pairs_tried = 0

        for _ in range(max_pairs):
            i, j = np.random.choice(n_int, 2, replace=False)
            idx_i = self.integer_vars[i]
            idx_j = self.integer_vars[j]

            # Try swapping
            x_test = x.copy()
            x_test[idx_i], x_test[idx_j] = x_test[idx_j], x_test[idx_i]

            viol_test = self.compute_total_violation(x_test)

            if viol_test < viol_best:
                viol_best = viol_test
                x_best = x_test

            pairs_tried += 1

            if viol_best < 1e-6:
                break

        return x_best

    def repair(self, x, max_rounds=5):
        """Try to repair solution to feasibility.

        First tries constraint-aware repair, then falls back to traditional methods.

        Args:
            x: Infeasible solution
            max_rounds: Maximum repair rounds

        Returns:
            (repaired_x, is_feasible)
        """
        # First, try constraint-aware repair
        x_repaired, is_feas = self._constraint_aware.repair(x, max_rounds=max_rounds)

        if is_feas:
            return x_repaired, True

        # Fall back to traditional methods
        x_current = x_repaired.copy()

        for _ in range(max_rounds):
            # Round first
            for i in self.integer_vars:
                x_current[i] = np.round(x_current[i])

            # Check feasibility
            viol = self.compute_total_violation(x_current)
            if viol < 1e-6:
                return x_current, True

            # 1-OPT search
            x_current = self.one_opt_search(x_current)

            viol = self.compute_total_violation(x_current)
            if viol < 1e-6:
                return x_current, True

        return x_current, False

    def one_opt_search(self, x, direction='minimize'):
        """1-OPT local search: try flipping each integer variable.

        Args:
            x: Current solution
            direction: 'minimize' or 'any_feasible'

        Returns:
            Best found solution
        """
        x_best = x.copy()
        viol_best = self.compute_total_violation(x_best)

        for idx in self.integer_vars:
            # Try current value +/- 1
            for delta in [-1, 1]:
                x_test = x.copy()
                x_test[idx] = np.round(x_test[idx]) + delta

                # Check if within bounds (if bounds exist)
                # TODO: Add bound checking

                viol_test = self.compute_total_violation(x_test)

                if viol_test < viol_best:
                    viol_best = viol_test
                    x_best = x_test

                    if direction == 'any_feasible' and viol_best < 1e-6:
                        return x_best

        return x_best

    def two_opt_search(self, x, max_pairs=100):
        """2-OPT local search: try swapping pairs of integer variables.

        Args:
            x: Current solution
            max_pairs: Maximum number of pairs to try

        Returns:
            Best found solution
        """
        x_best = x.copy()
        viol_best = self.compute_total_violation(x_best)

        # Random sample of pairs
        n_int = len(self.integer_vars)
        pairs_tried = 0

        for _ in range(max_pairs):
            i, j = np.random.choice(n_int, 2, replace=False)
            idx_i = self.integer_vars[i]
            idx_j = self.integer_vars[j]

            # Try swapping
            x_test = x.copy()
            x_test[idx_i], x_test[idx_j] = x_test[idx_j], x_test[idx_i]

            viol_test = self.compute_total_violation(x_test)

            if viol_test < viol_best:
                viol_best = viol_test
                x_best = x_test

            pairs_tried += 1

            if viol_best < 1e-6:
                break

        return x_best

    def repair(self, x, max_rounds=5):
        """Try to repair solution to feasibility.

        Args:
            x: Infeasible solution
            max_rounds: Maximum repair rounds

        Returns:
            (repaired_x, is_feasible)
        """
        x_current = x.copy()

        for _ in range(max_rounds):
            # Round first
            for i in self.integer_vars:
                x_current[i] = np.round(x_current[i])

            # Check feasibility
            viol = self.compute_total_violation(x_current)
            if viol < 1e-6:
                return x_current, True

            # 1-OPT search
            x_current = self.one_opt_search(x_current)

            viol = self.compute_total_violation(x_current)
            if viol < 1e-6:
                return x_current, True

        return x_current, False


class IntegralityChecker:
    """Check integrality violation."""

    def __init__(self, integer_vars):
        self.integer_vars = integer_vars

    def check(self, x):
        """Check integrality violation.

        Returns:
            Maximum violation from integer values
        """
        if len(self.integer_vars) == 0:
            return 0.0

        violations = []
        for i in self.integer_vars:
            violations.append(abs(x[i] - round(x[i])))

        return max(violations)

    def is_integer_feasible(self, x, tol=1e-4):
        """Check if solution is integer feasible."""
        return self.check(x) < tol


if __name__ == "__main__":
    print("Testing Integer Rounding...")

    # Test progressive measurement
    int_vars = [0, 2, 4]
    measurer = ProgressiveMeasurement(int_vars, schedule='cosine')

    x = np.array([1.3, 2.0, 3.7, 4.0, 5.2])
    print(f"Original: {x}")

    x_half = measurer.measure(x, 50, 100)
    print(f"Halfway: {x_half}")

    x_final = measurer.finalize(x)
    print(f"Final: {x_final}")

    # Test local search
    from scipy import sparse

    A = sparse.csr_matrix([[1.0, 1.0, 1.0]])
    b = np.array([3.0])

    repair = LocalSearchRepair(A, b, int_vars)

    x_test = np.array([1.0, 2.0, 1.5])  # Infeasible
    x_repaired, is_feas = repair.repair(x_test)

    print(f"\nRepair test:")
    print(f"Original: {x_test}, viol={repair.compute_total_violation(x_test):.4f}")
    print(f"Repaired: {x_repaired}, viol={repair.compute_total_violation(x_repaired):.4f}")
    print(f"Is feasible: {is_feas}")

    print("\nTest passed!")
