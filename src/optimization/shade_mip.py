"""
SHADE (Success-History based Adaptive Differential Evolution) for MIP.

This module implements SHADE adapted for Mixed Integer Programming problems.
Based on:
- Tanabe & Fukunaga (2013): "Success-History Based Parameter Adaptation for Differential Evolution"
- IEEE CEC 2013

Adaptations for MIP:
- Integer/binary variables handled via rounding
- Constraints handled via penalty method
- Boundary repair for box constraints
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable
import numpy as np
from scipy import sparse


@dataclass
class SHADEConfig:
    """Configuration for SHADE algorithm.

    Attributes:
        population_size: Number of individuals (NP)
        max_evals: Maximum function evaluations
        memory_size: Historical memory size (H)
        arc_rate: Archive size rate
        p_best_rate: Rate for p-best selection
        tol: Convergence tolerance
    """
    population_size: int = 100
    max_evals: int = 100000
    memory_size: int = 5
    arc_rate: float = 2.0
    p_best_rate: float = 0.11
    tol: float = 1e-6


@dataclass
class SHADEResult:
    """Result of SHADE optimization.

    Attributes:
        x_best: Best solution found
        obj_best: Best objective value
        feasibility: Constraint violation
        evaluations: Number of evaluations used
        converged: Whether converged
        history: History of best values
        success_rate: Final success rate
    """
    x_best: np.ndarray
    obj_best: float
    feasibility: float
    evaluations: int
    converged: bool
    history: List[dict] = field(default_factory=list)
    success_rate: float = 0.0


class SHADEMIP:
    """SHADE algorithm for Mixed Integer Programming.

    Solves problems of the form:
        min c^T x
        s.t. Ax <= b
             lb <= x <= ub
             x_i integer for i in integer_vars
             x_i binary for i in binary_vars

    Features:
    - Success-history based parameter adaptation
    - Current-to-pbest/1 mutation
    - External archive
    - Integer handling via rounding
    - Constraint handling via penalty
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        binary_vars: Optional[List[int]] = None,
        config: Optional[SHADEConfig] = None,
        penalty_factor: float = 1000.0,
    ):
        """Initialize SHADE for MIP.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            c: Objective coefficients (n,)
            lb: Variable lower bounds (n,)
            ub: Variable upper bounds (n,)
            integer_vars: Indices of integer variables
            binary_vars: Indices of binary variables
            config: SHADE configuration
            penalty_factor: Penalty for constraint violation
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.config = config or SHADEConfig()
        self.penalty_factor = penalty_factor

        self.m, self.n = A.shape

        # Variable types
        self.integer_vars = integer_vars or []
        self.binary_vars = binary_vars or []
        self.all_integer_vars = list(set(self.integer_vars + self.binary_vars))

        # Archive for failed solutions
        self.archive: List[np.ndarray] = []
        self.arc_size = int(self.config.arc_rate * self.config.population_size)

        # Historical memory for F and Cr
        self.memory_F = np.ones(self.config.memory_size) * 0.5
        self.memory_Cr = np.ones(self.config.memory_size) * 0.5
        self.memory_idx = 0

        # Statistics
        self.eval_count = 0
        self.success_history = []

    def _is_integer_var(self, idx: int) -> bool:
        """Check if variable at index is integer or binary."""
        return idx in self.all_integer_vars

    def _round_to_integer(self, x: np.ndarray) -> np.ndarray:
        """Round integer/binary variables to nearest integer."""
        x_rounded = x.copy()
        for idx in self.all_integer_vars:
            x_rounded[idx] = np.round(x[idx])
        return x_rounded

    def _project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project solution to bounds."""
        return np.clip(x, self.lb, self.ub)

    def _compute_constraint_violation(self, x: np.ndarray) -> float:
        """Compute constraint violation (Ax <= b)."""
        Ax = self.A @ x
        violation = np.maximum(Ax - self.b, 0)
        return np.linalg.norm(violation)

    def _is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if solution is feasible."""
        return self._compute_constraint_violation(x) < tol

    def evaluate(self, x: np.ndarray) -> tuple[float, float]:
        """Evaluate solution with penalty for constraint violation.

        Args:
            x: Solution vector

        Returns:
            Tuple of (penalized_fitness, objective_value)
        """
        # Round integer variables for evaluation
        x_eval = self._round_to_integer(x)
        x_eval = self._project_to_bounds(x_eval)

        # Compute objective
        obj = self.c @ x_eval

        # Compute constraint violation
        violation = self._compute_constraint_violation(x_eval)

        # Penalized fitness (minimization)
        fitness = obj + self.penalty_factor * violation

        self.eval_count += 1

        return fitness, obj

    def initialize_population(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize population within bounds.

        Args:
            seed: Random seed

        Returns:
            Population array (NP x n)
        """
        if seed is not None:
            np.random.seed(seed)

        pop = np.zeros((self.config.population_size, self.n))

        for i in range(self.n):
            if self._is_integer_var(i):
                # For integer variables, sample integers
                low = int(np.ceil(self.lb[i]))
                high = int(np.floor(self.ub[i]))
                pop[:, i] = np.random.randint(low, high + 1, self.config.population_size)
            else:
                # For continuous variables, sample uniformly
                pop[:, i] = np.random.uniform(self.lb[i], self.ub[i], self.config.population_size)

        return pop

    def current_to_pbest_mutation(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        idx: int,
        F: float,
    ) -> np.ndarray:
        """Current-to-pbest/1 mutation.

        v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)

        Args:
            pop: Current population
            fitness: Fitness values
            idx: Index of current individual
            F: Scaling factor

        Returns:
            Mutated vector
        """
        NP = self.config.population_size

        # Select p-best individual
        p_best_count = max(2, int(self.config.p_best_rate * NP))
        sorted_indices = np.argsort(fitness)
        p_best_idx = np.random.choice(sorted_indices[:p_best_count])

        # Select random individuals
        candidates = [i for i in range(NP) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)

        # Select from archive if available
        if len(self.archive) > 0:
            r2_archive = np.random.choice(len(self.archive))
            x_r2 = self.archive[r2_archive]
        else:
            x_r2 = pop[r2]

        # Mutation
        v = pop[idx] + F * (pop[p_best_idx] - pop[idx]) + F * (pop[r1] - x_r2)

        return v

    def binomial_crossover(
        self,
        x: np.ndarray,
        v: np.ndarray,
        Cr: float,
    ) -> np.ndarray:
        """Binomial crossover.

        Args:
            x: Target vector
            v: Mutated vector
            Cr: Crossover probability

        Returns:
            Trial vector
        """
        j_rand = np.random.randint(self.n)
        u = np.zeros(self.n)

        for j in range(self.n):
            if np.random.random() < Cr or j == j_rand:
                u[j] = v[j]
            else:
                u[j] = x[j]

        return u

    def repair_bounds(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Repair out-of-bounds values.

        Uses the method: if u_j < lb_j, set to (lb_j + x_j) / 2
                         if u_j > ub_j, set to (ub_j + x_j) / 2

        Args:
            u: Trial vector
            x: Original vector

        Returns:
            Repaired vector
        """
        u_repaired = u.copy()

        for j in range(self.n):
            if u_repaired[j] < self.lb[j]:
                u_repaired[j] = (self.lb[j] + x[j]) / 2.0
            elif u_repaired[j] > self.ub[j]:
                u_repaired[j] = (self.ub[j] + x[j]) / 2.0

        return u_repaired

    def update_archive(self, x: np.ndarray):
        """Add failed solution to archive.

        Args:
            x: Solution to archive
        """
        self.archive.append(x.copy())

        # Limit archive size
        if len(self.archive) > self.arc_size:
            # Randomly remove one
            remove_idx = np.random.randint(len(self.archive))
            self.archive.pop(remove_idx)

    def solve(
        self,
        seed: Optional[int] = None,
        verbose: bool = False,
        check_interval: int = 1000,
    ) -> SHADEResult:
        """Run SHADE algorithm.

        Args:
            seed: Random seed
            verbose: Print progress
            check_interval: Evaluation interval for logging

        Returns:
            SHADEResult with best solution
        """
        # Initialize
        pop = self.initialize_population(seed)
        fitness = np.zeros(self.config.population_size)
        obj_values = np.zeros(self.config.population_size)

        for i in range(self.config.population_size):
            fitness[i], obj_values[i] = self.evaluate(pop[i])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        best_obj = obj_values[best_idx]
        best_feasibility = self._compute_constraint_violation(
            self._round_to_integer(best_x)
        )

        history = []
        if verbose:
            print(f"Initial: obj={best_obj:.4f}, feas={best_feasibility:.4e}")

        # Main loop
        generation = 0
        while self.eval_count < self.config.max_evals:
            # Success tracking for this generation
            S_F = []  # Successful F values
            S_Cr = []  # Successful Cr values
            improvements = 0

            for i in range(self.config.population_size):
                if self.eval_count >= self.config.max_evals:
                    break

                # Select memory index
                r = np.random.randint(self.config.memory_size)

                # Generate F (Cauchy distribution)
                while True:
                    F = np.random.standard_cauchy() * 0.1 + self.memory_F[r]
                    if F > 0:
                        break
                F = min(F, 1.0)

                # Generate Cr (Normal distribution)
                Cr = np.random.normal(self.memory_Cr[r], 0.1)
                Cr = np.clip(Cr, 0.0, 1.0)

                # Mutation
                v = self.current_to_pbest_mutation(pop, fitness, i, F)

                # Crossover
                u = self.binomial_crossover(pop[i], v, Cr)

                # Repair bounds
                u = self.repair_bounds(u, pop[i])

                # Evaluate trial vector
                u_fitness, u_obj = self.evaluate(u)

                # Selection
                if u_fitness <= fitness[i]:
                    # Save old individual to archive
                    self.update_archive(pop[i])

                    # Replace
                    pop[i] = u
                    fitness[i] = u_fitness
                    obj_values[i] = u_obj

                    # Record success
                    S_F.append(F)
                    S_Cr.append(Cr)
                    improvements += 1

                    # Update best
                    if u_fitness < best_fitness:
                        best_x = u.copy()
                        best_fitness = u_fitness
                        best_obj = u_obj
                        best_feasibility = self._compute_constraint_violation(
                            self._round_to_integer(best_x)
                        )

            # Update historical memory
            if len(S_F) > 0:
                # Weighted Lehmer mean for F
                weights = np.array(S_F) / np.sum(S_F)
                mean_F = np.sum(weights * np.array(S_F) ** 2) / np.sum(weights * np.array(S_F))

                # Weighted arithmetic mean for Cr
                mean_Cr = np.mean(S_Cr)

                # Update memory
                self.memory_F[self.memory_idx] = mean_F
                self.memory_Cr[self.memory_idx] = mean_Cr
                self.memory_idx = (self.memory_idx + 1) % self.config.memory_size

                self.success_history.append(len(S_F) / self.config.population_size)

            # Logging
            generation += 1
            if verbose and self.eval_count % check_interval < self.config.population_size:
                print(
                    f"Gen {generation}, Evals {self.eval_count}: "
                    f"obj={best_obj:.4f}, feas={best_feasibility:.4e}, "
                    f"success={len(S_F)}/{self.config.population_size}"
                )

            # History tracking
            if generation % 10 == 0:
                history.append({
                    "generation": generation,
                    "evaluations": self.eval_count,
                    "best_obj": best_obj,
                    "best_feasibility": best_feasibility,
                    "success_rate": len(S_F) / self.config.population_size if S_F else 0,
                })

            # Check convergence
            if best_feasibility < self.config.tol and generation > 100:
                if verbose:
                    print(f"Converged at generation {generation}")
                break

        # Final evaluation with rounding
        best_x_rounded = self._round_to_integer(best_x)
        best_x_rounded = self._project_to_bounds(best_x_rounded)
        final_obj = self.c @ best_x_rounded
        final_feas = self._compute_constraint_violation(best_x_rounded)

        success_rate = np.mean(self.success_history) if self.success_history else 0.0

        return SHADEResult(
            x_best=best_x_rounded,
            obj_best=final_obj,
            feasibility=final_feas,
            evaluations=self.eval_count,
            converged=final_feas < self.config.tol,
            history=history,
            success_rate=success_rate,
        )


def solve_mip_shade(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: Optional[List[int]] = None,
    binary_vars: Optional[List[int]] = None,
    population_size: int = 100,
    max_evals: int = 100000,
    seed: Optional[int] = None,
    verbose: bool = False,
    penalty_factor: float = 1000.0,
) -> SHADEResult:
    """Convenience function to solve MIP with SHADE.

    Args:
        A: Constraint matrix
        b: Right-hand side
        c: Objective coefficients
        lb: Lower bounds
        ub: Upper bounds
        integer_vars: Integer variable indices
        binary_vars: Binary variable indices
        population_size: Population size
        max_evals: Maximum evaluations
        seed: Random seed
        verbose: Print progress
        penalty_factor: Constraint penalty

    Returns:
        SHADEResult
    """
    config = SHADEConfig(
        population_size=population_size,
        max_evals=max_evals,
    )

    solver = SHADEMIP(
        A=A,
        b=b,
        c=c,
        lb=lb,
        ub=ub,
        integer_vars=integer_vars,
        binary_vars=binary_vars,
        config=config,
        penalty_factor=penalty_factor,
    )

    return solver.solve(seed=seed, verbose=verbose)


if __name__ == "__main__":
    # Test with a simple MIP
    print("Testing SHADE on simple MIP...")

    # min -x - 2y s.t. x + 2y <= 3, x, y binary
    A = sparse.csr_matrix([[1.0, 2.0]])
    b = np.array([3.0])
    c = np.array([-1.0, -2.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    binary_vars = [0, 1]

    result = solve_mip_shade(
        A=A,
        b=b,
        c=c,
        lb=lb,
        ub=ub,
        binary_vars=binary_vars,
        population_size=20,
        max_evals=5000,
        seed=42,
        verbose=True,
    )

    print(f"\nBest solution: x = {result.x_best}")
    print(f"Best objective: {result.obj_best}")
    print(f"Feasibility: {result.feasibility:.4e}")
    print(f"Converged: {result.converged}")
    print(f"Success rate: {result.success_rate:.2%}")
