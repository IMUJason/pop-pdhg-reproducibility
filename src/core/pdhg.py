"""
Primal-Dual Hybrid Gradient (PDHG) solver for Linear Programming.

This module implements the Chambolle-Pock PDHG algorithm with adaptive restarts,
following the cuPDLP design for improved convergence.

For the LP:
    min c^T x
    s.t. Ax <= b
         x >= 0

The dual is:
    max -b^T y
    s.t. A^T y + c >= 0
         y >= 0

The saddle point formulation is:
    min_x max_{y>=0} c^T x + y^T(Ax - b)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class PDHGResult:
    """Result of PDHG solver."""

    x: np.ndarray
    y: np.ndarray
    obj_primal: float
    obj_dual: float
    iterations: int
    primal_residual: float
    dual_residual: float
    gap: float
    converged: bool
    status: str
    history: list[dict] = field(default_factory=list)


class PDHG:
    """Primal-Dual Hybrid Gradient solver for LP.

    Solves LP in the form:
        min c^T x
        s.t. Ax <= b
             lb <= x <= ub

    For inequality constraints Ax <= b in a minimization problem,
    the dual variables y must be >= 0 (non-negative).
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """Initialize PDHG solver."""
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)

        self.m, self.n = A.shape

        assert len(self.b) == self.m
        assert len(self.c) == self.n
        assert len(self.lb) == self.n
        assert len(self.ub) == self.n

        self.norm_A = self._estimate_norm()
        # Step sizes: eta * tau * ||A||^2 < 1
        self.eta = 0.99 / self.norm_A
        self.tau = 0.99 / self.norm_A

    def _estimate_norm(self, n_iters: int = 20) -> float:
        """Estimate ||A||_2 using power iteration."""
        v = np.random.randn(self.n)
        v = v / np.linalg.norm(v)

        for _ in range(n_iters):
            u = self.A @ v
            v_new = self.A.T @ u
            norm = np.linalg.norm(v_new)
            if norm > 1e-10:
                v = v_new / norm
            else:
                break

        return np.linalg.norm(self.A @ v)

    def _project_box(self, x: np.ndarray) -> np.ndarray:
        """Project to box [lb, ub]."""
        return np.clip(x, self.lb, self.ub)

    def _project_dual(self, y: np.ndarray) -> np.ndarray:
        """Project dual to non-negative orthant (y >= 0 for <= constraints)."""
        return np.maximum(y, 0.0)

    def _compute_residuals(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute primal residual, dual residual, and duality gap."""
        # Primal feasibility: ||max(Ax - b, 0)||_inf
        Ax = self.A @ x
        primal_violation = np.maximum(Ax - self.b, 0)
        primal_res = np.linalg.norm(primal_violation, ord=np.inf) / (
            1 + np.linalg.norm(self.b, ord=np.inf)
        )

        # Dual feasibility: A^T y + c >= 0 (at x = 0)
        # For bounded x: reduced cost sign depends on position
        ATy = self.A.T @ y
        reduced_costs = ATy + self.c

        dual_violation = np.zeros(self.n)
        for i in range(self.n):
            at_lb = x[i] <= self.lb[i] + 1e-8
            at_ub = not np.isinf(self.ub[i]) and x[i] >= self.ub[i] - 1e-8

            if at_lb and at_ub:
                dual_violation[i] = 0
            elif at_lb:
                # At lower bound: reduced cost should be >= 0
                dual_violation[i] = max(0, -reduced_costs[i])
            elif at_ub:
                # At upper bound: reduced cost should be <= 0
                dual_violation[i] = max(0, reduced_costs[i])
            else:
                # In interior: reduced cost should be 0
                dual_violation[i] = abs(reduced_costs[i])

        dual_res = np.linalg.norm(dual_violation, ord=np.inf) / (
            1 + np.linalg.norm(self.c, ord=np.inf)
        )

        # Duality gap
        # Primal: c^T x
        # Dual: -b^T y (for y >= 0)
        obj_primal = self.c @ x
        obj_dual = -self.b @ y  # Dual gives upper bound on primal for max
        gap = abs(obj_primal - obj_dual) / (1 + abs(obj_primal) + abs(obj_dual))

        return primal_res, dual_res, gap

    def solve(
        self,
        max_iter: int = 10000,
        tol: float = 1e-6,
        check_interval: int = 40,
        verbose: bool = False,
        x_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> PDHGResult:
        """Solve LP using PDHG.

        The saddle point problem is:
            min_x max_{y>=0} L(x,y) = c^T x + y^T(Ax - b)

        PDHG updates:
            x^{k+1} = proj_{[lb,ub]}(x^k - eta * (c + A^T y^k))
            y^{k+1} = proj_{y>=0}(y^k + tau * (A(2x^{k+1} - x^k) - b))
        """
        x = x_init.copy() if x_init is not None else np.clip(np.zeros(self.n), self.lb, self.ub)
        y = y_init.copy() if y_init is not None else np.zeros(self.m)

        x_old = x.copy()
        history = []
        converged = False
        status = "max_iter"

        last_restart_iter = 0
        last_residual = float("inf")

        for k in range(1, max_iter + 1):
            x_old = x.copy()

            # Primal update: x = proj(x - eta * (c + A^T y))
            grad_x = self.A.T @ y + self.c
            x = self._project_box(x - self.eta * grad_x)

            # Extrapolation
            x_bar = 2 * x - x_old

            # Dual update: y = proj_{y>=0}(y + tau * (A x_bar - b))
            Ax_bar = self.A @ x_bar
            y = self._project_dual(y + self.tau * (Ax_bar - self.b))

            # Check convergence
            if k % check_interval == 0:
                primal_res, dual_res, gap = self._compute_residuals(x, y)
                history.append({
                    "iteration": k,
                    "primal_residual": primal_res,
                    "dual_residual": dual_res,
                    "gap": gap,
                })

                max_residual = max(primal_res, dual_res, gap)

                if verbose:
                    print(f"Iter {k}: p_res={primal_res:.2e}, d_res={dual_res:.2e}, gap={gap:.2e}")

                if max_residual < tol:
                    converged = True
                    status = "optimal"
                    break

                # Adaptive restart
                if max_residual > 2 * last_residual and k - last_restart_iter > 100:
                    if verbose:
                        print(f"Restart at iteration {k}")
                    self.norm_A = self._estimate_norm()
                    self.eta = 0.99 / self.norm_A
                    self.tau = 0.99 / self.norm_A
                    last_restart_iter = k
                    last_residual = float("inf")
                else:
                    last_residual = max_residual

        obj_primal = self.c @ x
        obj_dual = -self.b @ y
        primal_res, dual_res, gap = self._compute_residuals(x, y)

        return PDHGResult(
            x=x,
            y=y,
            obj_primal=obj_primal,
            obj_dual=obj_dual,
            iterations=k if converged else max_iter,
            primal_residual=primal_res,
            dual_residual=dual_res,
            gap=gap,
            converged=converged,
            status=status,
            history=history,
        )


def solve_lp(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    **kwargs,
) -> PDHGResult:
    """Convenience function to solve LP with PDHG."""
    solver = PDHG(A, b, c, lb, ub)
    return solver.solve(**kwargs)


if __name__ == "__main__":
    print("Testing PDHG solver...")

    # Test: min -x - y s.t. x + y <= 1, x, y >= 0
    # Optimal: x + y = 1, obj = -1
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e10, 1e10])

    solver = PDHG(A, b, c, lb, ub)
    print(f"||A|| = {solver.norm_A:.4f}, eta = tau = {solver.eta:.4f}")
    result = solver.solve(max_iter=5000, tol=1e-6, verbose=True)

    print(f"\nSolution: x = {result.x}")
    print(f"Objective: {result.obj_primal}")
    print(f"Dual objective: {result.obj_dual}")
    print(f"Converged: {result.converged}")
    print(f"y: {result.y}")
