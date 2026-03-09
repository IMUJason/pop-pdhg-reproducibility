"""
Single-trajectory PDHG solver for ablation study.
No population, no tunneling - pure PDHG with rounding.
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, List
from scipy import sparse


class SinglePDHGSolver:
    """Single-trajectory PDHG for MIP - baseline for ablation study."""

    def __init__(self, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 constraint_sense: Optional[List[str]] = None):
        """Initialize single PDHG solver.

        Args:
            A: Constraint matrix
            b: RHS vector
            c: Objective coefficients
            lb: Lower bounds
            ub: Upper bounds
            integer_vars: Indices of integer variables
            constraint_sense: List of constraint senses ('<', '>', '=') for original problem
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float32)
        self.c = np.asarray(c, dtype=np.float32)
        self.lb = np.asarray(lb, dtype=np.float32)
        self.ub = np.asarray(ub, dtype=np.float32)
        self.integer_vars = integer_vars or []
        self.constraint_sense = constraint_sense

        self.m, self.n = A.shape

        # Step sizes
        self.eta, self.tau = self._compute_step_sizes()

    def _compute_step_sizes(self) -> Tuple[float, float]:
        """Compute PDHG step sizes using power iteration."""
        v = np.random.randn(self.n).astype(np.float32)
        v = v / np.linalg.norm(v)

        for _ in range(20):
            u = self.A @ v
            v_new = self.A.T @ u
            norm = np.linalg.norm(v_new)
            if norm > 1e-10:
                v = v_new / norm
            else:
                break

        norm_A = float(np.linalg.norm(self.A @ v))
        eta = tau = 0.99 / (norm_A + 1e-10)

        return eta, tau

    def _check_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check feasibility and return violation.

        If constraint_sense is provided, checks feasibility in original problem space.
        Otherwise assumes standard form Ax <= b.
        """
        Ax = self.A @ x

        if self.constraint_sense is None:
            # Standard form: Ax <= b
            violation = np.maximum(Ax - self.b, 0).max()
            return violation < 1e-4, float(violation)

        # Original problem space: handle different constraint types
        max_violation = 0.0
        for i, s in enumerate(self.constraint_sense):
            if s == '<':
                viol = max(Ax[i] - self.b[i], 0)
            elif s == '>':
                viol = max(self.b[i] - Ax[i], 0)
            elif s == '=':
                viol = abs(Ax[i] - self.b[i])
            else:
                viol = max(Ax[i] - self.b[i], 0)  # Default to <=
            max_violation = max(max_violation, viol)

        return max_violation < 1e-4, float(max_violation)

    def _check_integer_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check integer feasibility."""
        if not self.integer_vars:
            return True, 0.0

        violations = [abs(x[i] - round(x[i])) for i in self.integer_vars]
        max_viol = max(violations)
        return max_viol < 1e-4, max_viol

    def _progressive_rounding(self, x: np.ndarray, t: int, T_max: int) -> np.ndarray:
        """Apply progressive rounding based on iteration progress."""
        if not self.integer_vars or t < T_max * 0.3:
            return x

        x_rounded = x.copy()

        # Cosine schedule: more aggressive rounding later
        progress = (t - 0.3 * T_max) / (0.7 * T_max)
        threshold = 0.5 * (1 + np.cos(np.pi * (1 - progress)))

        for i in self.integer_vars:
            frac_part = abs(x[i] - round(x[i]))
            if frac_part < threshold:
                x_rounded[i] = round(x[i])

        return np.clip(x_rounded, self.lb, self.ub)

    def solve(self, max_iter: int = 10000, seed: int = 42,
              verbose: bool = False) -> Dict:
        """Solve using single-trajectory PDHG.

        Args:
            max_iter: Maximum iterations
            seed: Random seed
            verbose: Print progress

        Returns:
            Dictionary with solution info
        """
        np.random.seed(seed)

        # Initialize
        x = np.zeros(self.n, dtype=np.float32)
        y = np.zeros(self.m, dtype=np.float32)

        best_x = x.copy()
        best_obj = float('inf')
        best_feasible = False

        history = []
        start_time = time.time()

        for iteration in range(1, max_iter + 1):
            # PDHG iteration
            # Primal update
            x_new = x - self.eta * (self.c + self.A.T @ y)
            x_new = np.clip(x_new, self.lb, self.ub)

            # Extrapolation
            x_bar = 2 * x_new - x

            # Dual update
            y_new = y + self.tau * (self.A @ x_bar - self.b)
            y_new = np.maximum(y_new, 0)

            x, y = x_new, y_new

            # Progressive rounding (late phase)
            if iteration > max_iter * 0.5 and self.integer_vars:
                x_int = self._progressive_rounding(x, iteration, max_iter)
            else:
                x_int = x

            # Check and record
            if iteration % 100 == 0 or iteration == max_iter:
                is_feas, viol = self._check_feasibility(x_int)
                is_int_feas, int_viol = self._check_integer_feasibility(x_int)
                obj = float(self.c @ x_int)

                elapsed = time.time() - start_time

                history.append({
                    'iteration': iteration,
                    'obj': obj,
                    'primal_violation': viol,
                    'integrality_violation': int_viol,
                    'is_feasible': is_feas,
                    'is_integer_feasible': is_int_feas,
                    'time': elapsed
                })

                # Update best
                if is_feas and is_int_feas and obj < best_obj:
                    best_obj = obj
                    best_x = x_int.copy()
                    best_feasible = True

                if verbose:
                    print(f"Iter {iteration}: obj={obj:.4f}, "
                          f"feas={is_feas}, int_feas={is_int_feas}")

        total_time = time.time() - start_time

        # If we have a feasible solution (even without integer feasibility), report it
        # Check final state
        final_is_feas, final_viol = self._check_feasibility(x)
        final_is_int_feas, final_int_viol = self._check_integer_feasibility(x)
        final_obj = float(self.c @ x)

        # Use best feasible if found, otherwise use current
        if best_feasible:
            return {
                'x_best': best_x,
                'obj_best': best_obj,
                'is_feasible': True,
                'is_integer_feasible': True,
                'primal_violation': 0.0,
                'integrality_violation': 0.0,
                'iterations': max_iter,
                'solve_time': total_time,
                'history': history
            }
        else:
            # Return current state even if not integer feasible
            return {
                'x_best': x,
                'obj_best': final_obj,
                'is_feasible': final_is_feas,
                'is_integer_feasible': final_is_int_feas,
                'primal_violation': final_viol,
                'integrality_violation': final_int_viol,
                'iterations': max_iter,
                'solve_time': total_time,
                'history': history
            }


def solve_single_pdhg(A, b, c, lb, ub, integer_vars=None, max_iter=10000, seed=42):
    """Convenience function for single PDHG."""
    solver = SinglePDHGSolver(A, b, c, lb, ub, integer_vars)
    return solver.solve(max_iter=max_iter, seed=seed)
