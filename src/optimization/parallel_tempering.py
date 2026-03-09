"""
Parallel Tempering (Replica Exchange) for PDHG Optimization.

This module implements parallel tempering strategy inspired by quantum annealing
and statistical physics, allowing multiple replicas at different temperatures
to exchange states and improve exploration.

References:
    - Swendsen & Wang (1986): Replica Monte Carlo simulation
    - Geyer (1991): Markov chain Monte Carlo maximum likelihood
    - cuPDLP (2024): GPU-accelerated PDLP with restarts
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy import sparse
import copy


@dataclass
class ReplicaState:
    """State of a single replica in parallel tempering."""

    x: np.ndarray  # Primal solution
    y: np.ndarray  # Dual solution
    temperature: float
    energy: float
    obj_primal: float
    primal_feas: float
    exchange_count: int = 0
    iteration: int = 0


@dataclass
class ParallelTemperingResult:
    """Result of parallel tempering optimization."""

    x_best: np.ndarray
    y_best: np.ndarray
    obj_best: float
    replicas: List[ReplicaState]
    iterations: int
    converged: bool
    status: str
    exchange_history: List[dict] = field(default_factory=list)
    energy_history: List[dict] = field(default_factory=list)


class ParallelTemperingPDHG:
    """Parallel Tempering PDHG solver.

    Maintains multiple replicas at different temperatures, allowing
    high-temperature replicas to explore broadly while low-temperature
    replicas refine solutions. Periodic exchanges between replicas
    help escape local minima.
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_replicas: int = 8,
        T_min: float = 0.1,
        T_max: float = 10.0,
        population_size_per_replica: int = 1,
    ):
        """Initialize parallel tempering solver.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            c: Objective coefficients (n,)
            lb: Variable lower bounds (n,)
            ub: Variable upper bounds (n,)
            n_replicas: Number of temperature replicas
            T_min: Minimum temperature
            T_max: Maximum temperature
            population_size_per_replica: Population size for each replica
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)

        self.m, self.n = A.shape
        self.n_replicas = n_replicas

        # Geometric spacing of temperatures
        self.temperatures = np.geomspace(T_min, T_max, n_replicas)

        # Initialize step sizes
        self.norm_A = self._estimate_norm()
        self.eta = 0.99 / self.norm_A
        self.tau = 0.99 / self.norm_A

        # Replicas
        self.replicas: List[ReplicaState] = []

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

        return float(np.linalg.norm(self.A @ v))

    def _project_box(self, x: np.ndarray) -> np.ndarray:
        """Project to box [lb, ub]."""
        return np.clip(x, self.lb, self.ub)

    def _project_dual(self, y: np.ndarray) -> np.ndarray:
        """Project dual to non-negative orthant."""
        return np.maximum(y, 0.0)

    def _compute_energy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy (objective + constraint violation)."""
        obj = self.c @ x

        # Constraint violation
        Ax = self.A @ x
        violation = np.sum(np.maximum(Ax - self.b, 0) ** 2)

        return obj + violation

    def _compute_primal_feas(self, x: np.ndarray) -> float:
        """Compute primal feasibility."""
        Ax = self.A @ x
        return np.linalg.norm(np.maximum(Ax - self.b, 0), ord=np.inf)

    def initialize_replicas(self, seed: Optional[int] = None) -> None:
        """Initialize all replicas with diverse starting points.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.replicas = []

        for i, T in enumerate(self.temperatures):
            # Different initialization for each replica
            if i == 0:
                x = np.clip(np.zeros(self.n), self.lb, self.ub)
            elif i < self.n_replicas // 2:
                perturb = np.random.rand(self.n) * 0.1 * (T / self.temperatures[0])
                x = np.clip(perturb, self.lb, self.ub)
            else:
                perturb = np.random.randn(self.n) * 0.05 * T
                x = np.clip(perturb, self.lb, self.ub)

            y = np.zeros(self.m)

            energy = self._compute_energy(x, y)
            obj = self.c @ x
            feas = self._compute_primal_feas(x)

            self.replicas.append(ReplicaState(
                x=x, y=y, temperature=T, energy=energy,
                obj_primal=obj, primal_feas=feas
            ))

    def pdhg_step_with_temperature(
        self,
        replica: ReplicaState,
        noise_scale: float = 0.0,
    ) -> ReplicaState:
        """Perform one PDHG step with temperature-scaled noise.

        Args:
            replica: Current replica state
            noise_scale: Additional noise scale

        Returns:
            Updated replica state
        """
        x = replica.x.copy()
        y = replica.y.copy()
        T = replica.temperature

        x_old = x.copy()

        # Primal update with temperature-scaled noise
        grad_x = self.A.T @ y + self.c

        # Add thermal noise proportional to temperature
        if T > 0:
            noise = np.random.randn(self.n) * np.sqrt(T) * noise_scale * self.eta
            grad_x = grad_x + noise

        x = self._project_box(x - self.eta * grad_x)

        # Extrapolation
        x_bar = 2 * x - x_old

        # Dual update with temperature-scaled noise
        Ax_bar = self.A @ x_bar

        if T > 0:
            noise = np.random.randn(self.m) * np.sqrt(T) * noise_scale * self.tau
            Ax_bar = Ax_bar + noise

        y = self._project_dual(y + self.tau * (Ax_bar - self.b))

        # Compute new energy
        energy = self._compute_energy(x, y)
        obj = self.c @ x
        feas = self._compute_primal_feas(x)

        return ReplicaState(
            x=x, y=y, temperature=T, energy=energy,
            obj_primal=obj, primal_feas=feas,
            exchange_count=replica.exchange_count,
            iteration=replica.iteration + 1
        )

    def exchange_step(self) -> int:
        """Attempt to exchange states between adjacent replicas.

        Returns:
            Number of successful exchanges
        """
        n_exchanges = 0

        for i in range(len(self.replicas) - 1):
            r_i = self.replicas[i]
            r_j = self.replicas[i + 1]

            # Metropolis-Hastings exchange probability
            # P(accept) = min(1, exp((1/T_j - 1/T_i) * (E_i - E_j)))
            delta_beta = (1.0 / r_j.temperature) - (1.0 / r_i.temperature)
            delta_energy = r_i.energy - r_j.energy

            log_prob = delta_beta * delta_energy

            if np.log(np.random.random()) < log_prob:
                # Exchange states (but keep temperatures)
                j = i + 1
                self.replicas[i].x, self.replicas[j].x = r_j.x.copy(), r_i.x.copy()
                self.replicas[i].y, self.replicas[j].y = r_j.y.copy(), r_i.y.copy()
                self.replicas[i].energy, self.replicas[j].energy = r_j.energy, r_i.energy
                self.replicas[i].obj_primal, self.replicas[j].obj_primal = r_j.obj_primal, r_i.obj_primal
                self.replicas[i].primal_feas, self.replicas[j].primal_feas = r_j.primal_feas, r_i.primal_feas

                self.replicas[i].exchange_count += 1
                self.replicas[j].exchange_count += 1

                n_exchanges += 1

        return n_exchanges

    def get_best_replica(self) -> Tuple[ReplicaState, int]:
        """Get the best replica based on objective and feasibility.

        Returns:
            Tuple of (best replica, index)
        """
        # Prefer feasible solutions
        feas_threshold = 1e-4

        best_idx = 0
        best_score = float('inf')

        for i, r in enumerate(self.replicas):
            if r.primal_feas < feas_threshold:
                # Feasible: lower objective is better
                score = r.obj_primal
            else:
                # Infeasible: penalize constraint violation
                score = r.obj_primal + 1000 * r.primal_feas

            if score < best_score:
                best_score = score
                best_idx = i

        return self.replicas[best_idx], best_idx

    def solve(
        self,
        max_iter: int = 10000,
        tol: float = 1e-6,
        check_interval: int = 40,
        exchange_interval: int = 50,
        noise_scale: float = 0.1,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> ParallelTemperingResult:
        """Solve LP using parallel tempering PDHG.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            check_interval: Interval for convergence check
            exchange_interval: Interval for replica exchange
            noise_scale: Scale of thermal noise
            verbose: Print progress
            seed: Random seed

        Returns:
            ParallelTemperingResult with best solution
        """
        # Initialize replicas
        self.initialize_replicas(seed=seed)

        exchange_history = []
        energy_history = []
        converged = False
        status = "max_iter"

        for k in range(1, max_iter + 1):
            # Update all replicas
            for i, replica in enumerate(self.replicas):
                self.replicas[i] = self.pdhg_step_with_temperature(
                    replica, noise_scale=noise_scale
                )

            # Periodic exchange
            if k % exchange_interval == 0:
                n_exchanges = self.exchange_step()

                exchange_history.append({
                    'iteration': k,
                    'n_exchanges': n_exchanges,
                    'temperatures': [r.temperature for r in self.replicas],
                })

            # Convergence check
            if k % check_interval == 0:
                best_replica, best_idx = self.get_best_replica()

                # Compute residuals
                Ax = self.A @ best_replica.x
                primal_res = np.linalg.norm(np.maximum(Ax - self.b, 0), ord=np.inf) / (
                    1 + np.linalg.norm(self.b, ord=np.inf)
                )

                ATy = self.A.T @ best_replica.y
                reduced = ATy + self.c
                dual_res = np.linalg.norm(reduced, ord=np.inf) / (
                    1 + np.linalg.norm(self.c, ord=np.inf)
                )

                gap = abs(best_replica.obj_primal - self.b @ best_replica.y) / (
                    1 + abs(best_replica.obj_primal) + abs(self.b @ best_replica.y)
                )

                max_residual = max(primal_res, dual_res, gap)

                energy_history.append({
                    'iteration': k,
                    'best_idx': best_idx,
                    'best_obj': best_replica.obj_primal,
                    'best_feas': best_replica.primal_feas,
                    'primal_res': primal_res,
                    'dual_res': dual_res,
                    'gap': gap,
                    'energies': [r.energy for r in self.replicas],
                })

                if verbose:
                    print(f"Iter {k}: best_obj={best_replica.obj_primal:.4e}, "
                          f"p_res={primal_res:.2e}, d_res={dual_res:.2e}, "
                          f"gap={gap:.2e}, best_T={best_replica.temperature:.2f}")

                if max_residual < tol:
                    converged = True
                    status = "optimal"
                    break

        # Get final best solution
        best_replica, _ = self.get_best_replica()

        return ParallelTemperingResult(
            x_best=best_replica.x,
            y_best=best_replica.y,
            obj_best=best_replica.obj_primal,
            replicas=self.replicas,
            iterations=k if converged else max_iter,
            converged=converged,
            status=status,
            exchange_history=exchange_history,
            energy_history=energy_history,
        )


def solve_lp_parallel_tempering(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_replicas: int = 8,
    **kwargs,
) -> ParallelTemperingResult:
    """Convenience function to solve LP with parallel tempering.

    Args:
        A: Constraint matrix
        b: Right-hand side
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        n_replicas: Number of temperature replicas
        **kwargs: Additional arguments

    Returns:
        ParallelTemperingResult
    """
    solver = ParallelTemperingPDHG(A, b, c, lb, ub, n_replicas=n_replicas)
    return solver.solve(**kwargs)


if __name__ == "__main__":
    print("Testing Parallel Tempering PDHG...")

    # Test: min -x - y s.t. x + y <= 1, x, y >= 0
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1e10, 1e10])

    solver = ParallelTemperingPDHG(A, b, c, lb, ub, n_replicas=4)
    result = solver.solve(max_iter=2000, tol=1e-6, verbose=True, seed=42)

    print(f"\nBest solution: x = {result.x_best}")
    print(f"Best objective: {result.obj_best}")
    print(f"Converged: {result.converged}")
    print(f"Status: {result.status}")

    # Print exchange statistics
    total_exchanges = sum(r.exchange_count for r in result.replicas)
    print(f"Total exchanges: {total_exchanges}")
