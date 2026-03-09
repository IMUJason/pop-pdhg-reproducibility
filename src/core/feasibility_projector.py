"""
Feasibility Projection for General Constraints.

This module provides projection-based repair for problems with mixed
constraint types (<=, >=, =). The key idea is to use alternating projection
to find a point satisfying all constraints.

This is used as a post-processing step after PDHG iterations.
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple
import enum


class ConstraintSense(enum.IntEnum):
    LE = 0  # <=
    GE = 1  # >=
    EQ = 2  # ==


class FeasibilityProjector:
    """Projects solutions onto feasible region defined by mixed constraints.

    Uses Dykstra's alternating projection algorithm for finding the
    closest feasible point.
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        senses: List[ConstraintSense],
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """Initialize projector.

        Args:
            A: Constraint matrix
            b: Right-hand side
            senses: List of constraint senses
            lb: Variable lower bounds
            ub: Variable upper bounds
        """
        self.A = A
        self.b = b
        self.senses = np.array(senses)
        self.lb = lb
        self.ub = ub

        self.m, self.n = A.shape

        # Precompute constraint row norms for efficiency
        self.row_norms = np.array([A[i].dot(A[i].T).toarray().flatten()[0] for i in range(self.m)])
        self.row_norms = np.maximum(self.row_norms, 1e-10)  # Avoid division by zero

    def compute_violation(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute constraint violation for each constraint."""
        Ax = self.A @ x
        violations = np.zeros(self.m)

        for i in range(self.m):
            if self.senses[i] == ConstraintSense.LE:
                violations[i] = max(Ax[i] - self.b[i], 0)
            elif self.senses[i] == ConstraintSense.GE:
                violations[i] = max(self.b[i] - Ax[i], 0)
            elif self.senses[i] == ConstraintSense.EQ:
                violations[i] = abs(Ax[i] - self.b[i])

        return np.max(violations), violations

    def project_single_constraint(
        self,
        x: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Project x onto the i-th constraint.

        Returns the projection of x onto the half-space defined by
        the i-th constraint.
        """
        row = self.A[i].toarray().flatten()
        Ax = row @ x

        if self.senses[i] == ConstraintSense.LE:
            # Ax <= b
            if Ax <= self.b[i]:
                return x.copy()  # Already feasible
            # Project: x - (Ax - b) / ||a||^2 * a
            return x - (Ax - self.b[i]) / self.row_norms[i] * row

        elif self.senses[i] == ConstraintSense.GE:
            # Ax >= b
            if Ax >= self.b[i]:
                return x.copy()  # Already feasible
            # Project: x + (b - Ax) / ||a||^2 * a
            return x + (self.b[i] - Ax) / self.row_norms[i] * row

        elif self.senses[i] == ConstraintSense.EQ:
            # Ax = b
            if abs(Ax - self.b[i]) < 1e-10:
                return x.copy()  # Already feasible
            # Project: x + (b - Ax) / ||a||^2 * a
            return x + (self.b[i] - Ax) / self.row_norms[i] * row

    def project_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project onto variable bounds."""
        return np.clip(x, self.lb, self.ub)

    def project(
        self,
        x: np.ndarray,
        max_iters: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool, float]:
        """Project x onto feasible region using alternating projection.

        Args:
            x: Starting point
            max_iters: Maximum iterations
            tol: Tolerance for feasibility
            verbose: Print progress

        Returns:
            (projected_x, success, final_violation)
        """
        x_current = x.copy()

        # First, project to bounds
        x_current = self.project_bounds(x_current)

        # Check initial violation
        max_viol, _ = self.compute_violation(x_current)
        if max_viol <= tol:
            return x_current, True, max_viol

        # Dykstra's algorithm with bounds
        # Keep track of auxiliary variables for each constraint
        aux_vars = [np.zeros(self.n) for _ in range(self.m)]

        best_x = x_current.copy()
        best_viol = max_viol

        for iteration in range(max_iters):
            x_prev = x_current.copy()

            # Project onto each constraint in cyclic order
            for i in range(self.m):
                # Dykstra step: subtract auxiliary variable
                x_temp = x_current - aux_vars[i]

                # Project onto constraint i
                x_proj = self.project_single_constraint(x_temp, i)

                # Update auxiliary variable
                aux_vars[i] = x_proj - x_current

                # Update current point
                x_current = x_proj

                # Project to bounds after each constraint
                x_current = self.project_bounds(x_current)

            # Check violation
            max_viol, _ = self.compute_violation(x_current)

            if max_viol < best_viol:
                best_viol = max_viol
                best_x = x_current.copy()

            if max_viol <= tol:
                return x_current, True, max_viol

            # Check convergence
            change = np.linalg.norm(x_current - x_prev)
            if change < 1e-10 and max_viol > tol:
                # Stalled but not feasible
                break

        # Return best found
        return best_x, best_viol <= tol, best_viol

    def project_with_integer_rounding(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        max_iters: int = 1000,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, bool, float]:
        """Project with integer variable handling.

        Strategy:
        1. First do continuous projection
        2. Round integer variables
        3. Re-project to fix any constraint violations from rounding
        4. Use coordinate descent to optimize integer variables
        """
        if not integer_vars:
            # No integer variables, just do continuous projection
            return self.project(x, max_iters, tol)

        # Step 1: Continuous projection
        x_proj, success, viol = self.project(x, max_iters, tol)

        # Step 2: Round integer variables
        x_rounded = x_proj.copy()
        for i in integer_vars:
            x_rounded[i] = round(x_rounded[i])
        x_rounded = np.clip(x_rounded, self.lb, self.ub)

        # Step 3: Re-project to fix constraint violations from rounding
        x_proj2, success2, viol2 = self.project(x_rounded, max_iters, tol)

        # Use best of the two
        if viol2 < viol:
            x_best = x_proj2.copy()
            best_viol = viol2
        else:
            x_best = x_proj.copy()
            best_viol = viol

        # Step 4: Coordinate descent on integer variables
        # This is crucial for problems like assign1-5-8
        for cd_iter in range(200):  # More iterations for thoroughness
            improved = False

            for i in integer_vars:
                current_val = x_best[i]
                current_viol, _ = self.compute_violation(x_best)

                # Try neighboring integer values
                candidates = []
                if round(current_val - 1) >= self.lb[i]:
                    candidates.append(round(current_val - 1))
                candidates.append(round(current_val))
                if round(current_val + 1) <= self.ub[i]:
                    candidates.append(round(current_val + 1))

                best_val = current_val
                for val in candidates:
                    x_best[i] = val
                    new_viol, _ = self.compute_violation(x_best)
                    if new_viol < best_viol:
                        best_viol = new_viol
                        best_val = val
                        improved = True

                x_best[i] = best_val

            if not improved:
                break

            # Every 10 iterations, re-project continuous variables
            if cd_iter % 10 == 0 and cd_iter > 0:
                x_best, _, best_viol = self.project(x_best, max_iters=100, tol=tol)
                # Re-round integers after projection
                for i in integer_vars:
                    x_best[i] = round(x_best[i])
                x_best = np.clip(x_best, self.lb, self.ub)

        # Final projection and check
        x_final, success_final, viol_final = self.project(x_best, max_iters=200, tol=tol)

        # Final rounding
        for i in integer_vars:
            x_final[i] = round(x_final[i])
        x_final = np.clip(x_final, self.lb, self.ub)

        # Check final violation
        max_viol_final, _ = self.compute_violation(x_final)

        return x_final, max_viol_final <= tol, max_viol_final

    def repair_with_augmented_lagrangian(
        self,
        x: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        max_iters: int = 500,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, bool, float]:
        """Repair using augmented Lagrangian method.

        More aggressive than projection for difficult constraints.
        """
        x_current = x.copy()
        integer_vars = integer_vars or []

        # AL parameters
        rho = 1.0
        lambda_mult = np.zeros(self.m)

        best_x = x_current.copy()
        best_viol = float('inf')

        for iteration in range(max_iters):
            max_viol, violations = self.compute_violation(x_current)

            if max_viol < best_viol:
                best_viol = max_viol
                best_x = x_current.copy()

            if max_viol <= tol:
                return x_current, True, max_viol

            # Compute AL gradient
            grad = np.zeros(self.n)
            Ax = self.A @ x_current

            for i in range(self.m):
                if violations[i] > 0 or abs(lambda_mult[i]) > 0:
                    row = self.A[i].toarray().flatten()

                    if self.senses[i] == ConstraintSense.LE:
                        # Violation: Ax - b > 0
                        viol = Ax[i] - self.b[i]
                        al_term = lambda_mult[i] + rho * max(0, viol)
                        if al_term > 0:
                            grad += al_term * row

                    elif self.senses[i] == ConstraintSense.GE:
                        # Violation: b - Ax > 0
                        viol = self.b[i] - Ax[i]
                        al_term = lambda_mult[i] + rho * max(0, viol)
                        if al_term > 0:
                            grad -= al_term * row

                    elif self.senses[i] == ConstraintSense.EQ:
                        # Violation: |Ax - b| > 0
                        viol = Ax[i] - self.b[i]
                        al_term = lambda_mult[i] + rho * viol
                        grad += al_term * row

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm

            # Line search with aggressive step sizes
            best_improvement = 0
            best_candidate = x_current.copy()

            for step_scale in [2.0, 1.0, 0.5, 0.25, 0.1]:
                step_size = step_scale * min(max_viol, 10.0)
                x_candidate = x_current - step_size * grad
                x_candidate = np.clip(x_candidate, self.lb, self.ub)

                # Round integers
                for i in integer_vars:
                    x_candidate[i] = round(x_candidate[i])

                new_viol, _ = self.compute_violation(x_candidate)
                improvement = max_viol - new_viol

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = x_candidate.copy()

            if best_improvement > 0:
                x_current = best_candidate

                # Update multipliers
                Ax = self.A @ x_current
                for i in range(self.m):
                    if self.senses[i] == ConstraintSense.LE:
                        viol = Ax[i] - self.b[i]
                        lambda_mult[i] = max(0, lambda_mult[i] + rho * viol)
                    elif self.senses[i] == ConstraintSense.GE:
                        viol = self.b[i] - Ax[i]
                        lambda_mult[i] = max(0, lambda_mult[i] + rho * viol)
                    elif self.senses[i] == ConstraintSense.EQ:
                        viol = Ax[i] - self.b[i]
                        lambda_mult[i] += rho * viol

                # Increase penalty periodically
                if iteration > 0 and iteration % 50 == 0:
                    rho *= 2.0
            else:
                # Try random perturbation for integer vars
                if integer_vars and iteration % 20 == 0:
                    for i in integer_vars:
                        if np.random.rand() < 0.3:
                            x_current[i] = round(x_current[i] + np.random.choice([-1, 0, 1]))
                            x_current[i] = np.clip(x_current[i], self.lb[i], self.ub[i])

                    new_viol, _ = self.compute_violation(x_current)
                    if new_viol >= max_viol:
                        # Revert to best
                        x_current = best_x.copy()
                        break
                else:
                    break

        return best_x, best_viol <= tol, best_viol
