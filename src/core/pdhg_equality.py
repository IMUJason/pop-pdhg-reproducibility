"""PDHG solver for equality constraints.

This module provides a PDHG implementation that solves problems with
equality constraints Ax = b, enabling unified handling of all constraint
types through slack variable transformation.

Key difference from standard PDHG:
- Standard PDHG: Ax <= b, dual variable y >= 0
- Equality PDHG: Ax = b, dual variable y unrestricted
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Callable
import time


class PDHGEquality:
    """PDHG solver for equality-constrained LP.

    Solves: min c^T x
            s.t. A x = b
                 lb <= x <= ub

    The dual variable y is unrestricted (can be positive or negative).
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """Initialize PDHG for equality constraints.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            c: Objective coefficients (n,)
            lb: Lower bounds (n,)
            ub: Upper bounds (n,)
        """
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub

        self.m, self.n = A.shape

        # Compute step sizes (same as standard PDHG)
        self._compute_step_sizes()

    def _compute_step_sizes(self):
        """Compute primal and dual step sizes."""
        # Estimate Lipschitz constant
        if self.m > 0:
            # Use power iteration to estimate ||A||_2
            np.random.seed(0)
            v = np.random.randn(self.n)
            v = v / np.linalg.norm(v)

            for _ in range(10):
                Av = self.A @ v
                v_new = self.A.T @ Av
                v_new_norm = np.linalg.norm(v_new)
                if v_new_norm < 1e-10:
                    break
                v = v_new / v_new_norm

            L = np.sqrt(np.linalg.norm(self.A @ v))
            self.tau = 1.0 / L if L > 0 else 1.0
            self.sigma = 1.0 / L if L > 0 else 1.0
        else:
            self.tau = 1.0
            self.sigma = 1.0

    def iterate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_bar: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one PDHG iteration for equality constraints.

        Args:
            x: Current primal solution
            y: Current dual solution
            x_bar: Extrapolated primal solution

        Returns:
            (x_new, y_new, x_bar_new): Updated solutions
        """
        # Primal update: x = proj_X(x - tau * (c + A^T y))
        x_new = x - self.tau * (self.c + self.A.T @ y)
        x_new = np.clip(x_new, self.lb, self.ub)

        # Dual update: y = y + sigma * (A @ x_bar - b)
        # KEY DIFFERENCE: No projection to y >= 0 for equality constraints
        y_new = y + self.sigma * (self.A @ x_bar - self.b)

        # Extrapolation: x_bar = 2 * x_new - x
        x_bar_new = 2 * x_new - x

        return x_new, y_new, x_bar_new

    def solve(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-6,
        verbose: bool = False,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Solve the equality-constrained LP.

        Args:
            x0: Initial primal solution
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
            callback: Optional callback function(iter, x, y, res)

        Returns:
            Dictionary with solution information
        """
        # Initialize
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = x0.copy()

        x = np.clip(x, self.lb, self.ub)
        y = np.zeros(self.m)
        x_bar = x.copy()

        # Tracking
        history = {'primal_res': [], 'dual_res': [], 'gap': [], 'obj': []}

        for k in range(max_iter):
            # Store old values for convergence check
            x_old = x.copy()

            # Iterate
            x, y, x_bar = self.iterate(x, y, x_bar)

            # Compute residuals
            primal_res = np.linalg.norm(self.A @ x - self.b) / (1 + np.linalg.norm(self.b))

            # Dual residual
            if k > 0:
                dual_res = np.linalg.norm(x - x_old) / (self.tau * (1 + np.linalg.norm(x)))
            else:
                dual_res = float('inf')

            # Duality gap
            obj_primal = self.c @ x

            # For gap, we need dual objective: max b^T y s.t. c + A^T y = 0 (approximately)
            # Use a simple estimate
            gap = abs(primal_res)

            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)
            history['gap'].append(gap)
            history['obj'].append(obj_primal)

            # Callback
            if callback:
                callback(k, x, y, primal_res)

            # Verbose output
            if verbose and (k % 500 == 0 or k == max_iter - 1):
                print(f"Iter {k}: obj={obj_primal:.4e}, p_res={primal_res:.2e}, "
                      f"d_res={dual_res:.2e}, gap={gap:.2e}")

            # Convergence check
            if primal_res < tol and dual_res < tol:
                if verbose:
                    print(f"Converged at iteration {k}")
                break

        return {
            'x': x,
            'y': y,
            'obj': self.c @ x,
            'primal_res': primal_res,
            'dual_res': dual_res,
            'gap': gap,
            'iterations': k + 1,
            'converged': primal_res < tol and dual_res < tol,
            'history': history,
        }


class PDHGEqualityAdaptive(PDHGEquality):
    """PDHG for equality constraints with adaptive restarts."""

    def solve(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-6,
        verbose: bool = False,
        restart_interval: int = 500,
    ) -> dict:
        """Solve with adaptive restarts."""
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = x0.copy()

        x = np.clip(x, self.lb, self.ub)
        y = np.zeros(self.m)
        x_bar = x.copy()

        best_x = x.copy()
        best_obj = self.c @ x
        best_res = float('inf')

        history = {'primal_res': [], 'dual_res': [], 'gap': [], 'obj': []}

        for k in range(max_iter):
            x_old = x.copy()

            # Iterate
            x, y, x_bar = self.iterate(x, y, x_bar)

            # Track best solution
            obj = self.c @ x
            primal_res = np.linalg.norm(self.A @ x - self.b) / (1 + np.linalg.norm(self.b))

            if primal_res < best_res and np.isfinite(obj):
                best_res = primal_res
                best_x = x.copy()
                best_obj = obj

            # Adaptive restart
            if k > 0 and k % restart_interval == 0:
                if verbose:
                    print(f"Restart at iteration {k}, best_res={best_res:.2e}")
                x = best_x.copy()
                x_bar = x.copy()
                y *= 0.5  # Reduce dual variables but don't zero them

            # Compute residuals
            dual_res = np.linalg.norm(x - x_old) / (self.tau * (1 + np.linalg.norm(x)))
            gap = abs(primal_res)

            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)
            history['gap'].append(gap)
            history['obj'].append(obj)

            if verbose and (k % 500 == 0 or k == max_iter - 1):
                print(f"Iter {k}: obj={obj:.4e}, p_res={primal_res:.2e}, "
                      f"best_res={best_res:.2e}")

            if primal_res < tol and dual_res < tol:
                if verbose:
                    print(f"Converged at iteration {k}")
                break

        return {
            'x': best_x,
            'y': y,
            'obj': best_obj,
            'primal_res': best_res,
            'dual_res': dual_res,
            'gap': gap,
            'iterations': k + 1,
            'converged': best_res < tol,
            'history': history,
        }
