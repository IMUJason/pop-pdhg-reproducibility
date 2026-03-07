"""
Population Primal-Dual Hybrid Gradient (Pop-PDHG) Solver.

This module implements a population-based PDHG algorithm where K solution
candidates are maintained and evolved simultaneously. The population provides:
1. Diversity: different trajectories explore different regions
2. Information sharing: through quantum-inspired interference operators
3. Statistics: population statistics guide branching decisions

References:
    - Chambolle & Pock (2011): Original PDHG
    - cuPDLP (2024): GPU-accelerated PDLP with restarts
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class PopulationState:
    """State of the population at a given iteration.

    Attributes:
        x: Primal solutions, shape (K, n)
        y: Dual solutions, shape (K, m)
        obj: Objective values for each member, shape (K,)
        primal_feas: Primal feasibility for each member, shape (K,)
        dual_feas: Dual feasibility for each member, shape (K,)
        age: Number of iterations since birth for each member, shape (K,)
    """

    x: np.ndarray
    y: np.ndarray
    obj: np.ndarray
    primal_feas: np.ndarray
    dual_feas: np.ndarray
    age: np.ndarray

    @property
    def K(self) -> int:
        """Population size."""
        return self.x.shape[0]

    @property
    def n(self) -> int:
        """Number of variables."""
        return self.x.shape[1]

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self.y.shape[1]


@dataclass
class PopPDHGResult:
    """Result of population PDHG solver.

    Attributes:
        x_best: Best primal solution found
        y_best: Corresponding dual solution
        obj_best: Best objective value
        state: Final population state
        iterations: Number of iterations performed
        converged: Whether converged
        status: Solver status string
        history: History of best objective values
    """

    x_best: np.ndarray
    y_best: np.ndarray
    obj_best: float
    state: PopulationState
    iterations: int
    converged: bool
    status: str
    history: list[dict] = field(default_factory=list)


class PopulationPDHG:
    """Population-based Primal-Dual Hybrid Gradient solver.

    Maintains K solution candidates and evolves them simultaneously using
    batch operations. The population approach provides:
    - Better exploration of the solution space
    - Robustness to local minima
    - Statistical information for branching decisions

    The algorithm solves LP in the form:
        min c^T x
        s.t. Ax <= b
             lb <= x <= ub
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        population_size: int = 16,
    ):
        """Initialize Population PDHG solver.

        Args:
            A: Constraint matrix (m x n, sparse CSR format)
            b: Right-hand side vector (m,)
            c: Objective coefficients (n,)
            lb: Variable lower bounds (n,)
            ub: Variable upper bounds (n,)
            population_size: Number of solution candidates (K)
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.K = population_size

        self.m, self.n = A.shape

        # Validate dimensions
        assert len(self.b) == self.m
        assert len(self.c) == self.n
        assert len(self.lb) == self.n
        assert len(self.ub) == self.n

        # Initialize step sizes
        self.norm_A = self._estimate_norm()
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

    def initialize_population(
        self, strategy: str = "diverse", seed: Optional[int] = None
    ) -> PopulationState:
        """Initialize the population with diverse starting points.

        Args:
            strategy: Initialization strategy
                - 'zeros': All members start at zero (clipped to bounds)
                - 'diverse': Different random perturbations
                - 'perturbed': Small perturbations from zero
            seed: Random seed for reproducibility

        Returns:
            Initial population state
        """
        if seed is not None:
            np.random.seed(seed)

        x = np.zeros((self.K, self.n))
        y = np.zeros((self.K, self.m))

        if strategy == "zeros":
            # All members start at zero (projected to bounds)
            x = np.clip(x, self.lb, self.ub)

        elif strategy == "diverse":
            # Different members explore different regions
            for k in range(self.K):
                if k == 0:
                    # First member at zero
                    x[k] = np.clip(np.zeros(self.n), self.lb, self.ub)
                elif k < self.K // 2:
                    # Random perturbations in positive direction
                    perturb = np.random.rand(self.n) * 0.1
                    x[k] = np.clip(perturb, self.lb, self.ub)
                else:
                    # Random perturbations with larger scale
                    perturb = np.random.randn(self.n) * 0.05
                    x[k] = np.clip(perturb, self.lb, self.ub)

        elif strategy == "perturbed":
            # Small perturbations from zero
            x = np.random.randn(self.K, self.n) * 0.01
            x = np.clip(x, self.lb, self.ub)

        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

        # Initialize objective values
        obj = x @ self.c

        # Initialize feasibility measures
        Ax = (self.A @ x.T).T  # (K, m)
        primal_feas = np.linalg.norm(np.maximum(Ax - self.b, 0), axis=1)

        dual_feas = np.zeros(self.K)

        # Initialize ages
        age = np.zeros(self.K, dtype=int)

        return PopulationState(
            x=x,
            y=y,
            obj=obj,
            primal_feas=primal_feas,
            dual_feas=dual_feas,
            age=age,
        )

    def batch_pdhg_step(
        self, state: PopulationState
    ) -> PopulationState:
        """Perform one PDHG step for all population members.

        Uses batch matrix operations for efficiency:
        - x_new = proj(x - eta * (c + A^T y))  for all K members
        - y_new = proj(y + tau * (A x_bar - b)) for all K members

        Args:
            state: Current population state

        Returns:
            Updated population state
        """
        x = state.x.copy()
        y = state.y.copy()

        # Primal update: x = proj(x - eta * (c + A^T y))
        # A^T y for batch: (K, n) = (K, m) @ (m, n)^T
        ATy = (y @ self.A.toarray()).reshape(self.K, self.n)  # (K, n)
        grad_x = ATy + self.c  # Broadcasting: (K, n) + (n,) -> (K, n)
        x_new = np.clip(x - self.eta * grad_x, self.lb, self.ub)

        # Extrapolation: x_bar = 2 * x_new - x
        x_bar = 2 * x_new - x

        # Dual update: y = proj(y + tau * (A x_bar - b))
        # A @ x_bar: (K, m) = (K, n) @ (n, m)^T
        Ax_bar = (x_bar @ self.A.toarray().T).reshape(self.K, self.m)  # (K, m)
        y_new = np.maximum(y + self.tau * (Ax_bar - self.b), 0)  # y >= 0 for <= constraints

        # Update objective values
        obj = x_new @ self.c

        # Update feasibility measures
        Ax_new = (self.A @ x_new.T).T  # (K, m)
        primal_feas = np.linalg.norm(np.maximum(Ax_new - self.b, 0), axis=1)

        # Dual feasibility (reduced cost check)
        ATy_new = (y_new @ self.A.toarray()).reshape(self.K, self.n)
        reduced_costs = ATy_new + self.c
        dual_feas = np.zeros(self.K)
        for k in range(self.K):
            for j in range(self.n):
                at_lb = x_new[k, j] <= self.lb[j] + 1e-8
                at_ub = not np.isinf(self.ub[j]) and x_new[k, j] >= self.ub[j] - 1e-8
                if at_lb:
                    dual_feas[k] += max(0, -reduced_costs[k, j]) ** 2
                elif at_ub:
                    dual_feas[k] += max(0, reduced_costs[k, j]) ** 2
                else:
                    dual_feas[k] += reduced_costs[k, j] ** 2
        dual_feas = np.sqrt(dual_feas)

        # Update ages
        age = state.age + 1

        return PopulationState(
            x=x_new,
            y=y_new,
            obj=obj,
            primal_feas=primal_feas,
            dual_feas=dual_feas,
            age=age,
        )

    def get_best(self, state: PopulationState) -> tuple[np.ndarray, np.ndarray, float]:
        """Get the best solution from the population.

        Selects the member with the lowest objective value among those
        with acceptable feasibility.

        Args:
            state: Population state

        Returns:
            Tuple of (best_x, best_y, best_obj)
        """
        # Prefer feasible solutions
        feas_threshold = 1e-4
        feasible_mask = state.primal_feas < feas_threshold

        if np.any(feasible_mask):
            # Among feasible, pick lowest objective
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = feasible_indices[np.argmin(state.obj[feasible_indices])]
        else:
            # If none feasible, pick best trade-off
            combined = state.obj + 1000 * state.primal_feas
            best_idx = np.argmin(combined)

        return state.x[best_idx].copy(), state.y[best_idx].copy(), state.obj[best_idx]

    def compute_diversity(self, state: PopulationState) -> float:
        """Compute population diversity as average pairwise distance.

        Args:
            state: Population state

        Returns:
            Diversity measure (average pairwise L2 distance)
        """
        total_dist = 0.0
        count = 0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                dist = np.linalg.norm(state.x[i] - state.x[j])
                total_dist += dist
                count += 1
        return total_dist / count if count > 0 else 0.0

    def solve(
        self,
        max_iter: int = 10000,
        tol: float = 1e-6,
        check_interval: int = 40,
        verbose: bool = False,
        init_strategy: str = "diverse",
        seed: Optional[int] = None,
    ) -> PopPDHGResult:
        """Solve LP using population PDHG.

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            check_interval: Interval for checking convergence
            verbose: Whether to print progress
            init_strategy: Population initialization strategy
            seed: Random seed

        Returns:
            PopPDHGResult with best solution and statistics
        """
        # Initialize population
        state = self.initialize_population(strategy=init_strategy, seed=seed)

        history = []
        converged = False
        status = "max_iter"

        best_obj_so_far = float("inf")

        for k in range(1, max_iter + 1):
            # Perform batch PDHG step
            state = self.batch_pdhg_step(state)

            # Check convergence
            if k % check_interval == 0:
                x_best, y_best, obj_best = self.get_best(state)
                diversity = self.compute_diversity(state)

                # Compute residuals for best solution
                Ax = self.A @ x_best
                primal_res = np.linalg.norm(np.maximum(Ax - self.b, 0), ord=np.inf) / (
                    1 + np.linalg.norm(self.b, ord=np.inf)
                )

                ATy = self.A.T @ y_best
                reduced = ATy + self.c
                dual_res = np.linalg.norm(reduced, ord=np.inf) / (
                    1 + np.linalg.norm(self.c, ord=np.inf)
                )

                gap = abs(obj_best - self.b @ y_best) / (
                    1 + abs(obj_best) + abs(self.b @ y_best)
                )

                max_residual = max(primal_res, dual_res, gap)

                if obj_best < best_obj_so_far:
                    best_obj_so_far = obj_best

                history.append(
                    {
                        "iteration": k,
                        "obj_best": obj_best,
                        "primal_res": primal_res,
                        "dual_res": dual_res,
                        "gap": gap,
                        "diversity": diversity,
                    }
                )

                if verbose:
                    print(
                        f"Iter {k}: obj={obj_best:.4e}, "
                        f"p_res={primal_res:.2e}, d_res={dual_res:.2e}, "
                        f"gap={gap:.2e}, div={diversity:.4f}"
                    )

                if max_residual < tol:
                    converged = True
                    status = "optimal"
                    break

        # Get final best solution
        x_best, y_best, obj_best = self.get_best(state)

        return PopPDHGResult(
            x_best=x_best,
            y_best=y_best,
            obj_best=obj_best,
            state=state,
            iterations=k if converged else max_iter,
            converged=converged,
            status=status,
            history=history,
        )


def solve_lp_population(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    population_size: int = 16,
    **kwargs,
) -> PopPDHGResult:
    """Convenience function to solve LP with population PDHG.

    Args:
        A: Constraint matrix
        b: Right-hand side
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        population_size: Number of solution candidates
        **kwargs: Additional arguments passed to solve()

    Returns:
        PopPDHGResult object
    """
    solver = PopulationPDHG(A, b, c, lb, ub, population_size)
    return solver.solve(**kwargs)


if __name__ == "__main__":
    # Test with a simple LP
    print("Testing Population PDHG...")

    # min -x - y s.t. x + y <= 1, x, y >= 0
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e10, 1e10])

    solver = PopulationPDHG(A, b, c, lb, ub, population_size=8)
    result = solver.solve(max_iter=2000, tol=1e-6, verbose=True, seed=42)

    print(f"\nBest solution: x = {result.x_best}")
    print(f"Best objective: {result.obj_best}")
    print(f"Converged: {result.converged}")
    print(f"Final diversity: {solver.compute_diversity(result.state):.4f}")
