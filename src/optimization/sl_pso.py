"""
SL-PSO (Social Learning Particle Swarm Optimization) for MIP problems.

Based on:
- "A social learning particle swarm optimization algorithm for scalable optimization"
  Information Sciences, 2015 (original SL-PSO)
- Extensions for mixed integer programming with constraint handling

Key features:
- No global/personal best storage
- Particles learn from demonstrators (better particles)
- Dimension-dependent parameter control
- Constraint handling via penalty functions
- Integer variable handling via rounding
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
import numpy as np
from scipy import sparse


@dataclass
class SLPSOConfig:
    """Configuration for SL-PSO algorithm.

    Attributes:
        population_size: Number of particles
        max_iter: Maximum iterations
        phi: Social learning probability (controls learning from mean)
        alpha: Learning rate for velocity update
        beta: Influence of demonstrators
        gamma: Random exploration factor
        penalty_coeff: Constraint violation penalty coefficient
        adaptive_penalty: Whether to adapt penalty coefficient
        seed: Random seed
    """
    population_size: int = 50
    max_iter: int = 5000
    phi: float = 0.5  # Social learning probability
    alpha: float = 0.5  # Learning rate
    beta: float = 0.5  # Demonstrator influence
    gamma: float = 0.1  # Exploration factor
    penalty_coeff: float = 1000.0
    adaptive_penalty: bool = True
    seed: Optional[int] = None


@dataclass
class SLPSOResult:
    """Result from SL-PSO optimization.

    Attributes:
        x_best: Best solution found
        obj_best: Best objective value
        violation_best: Constraint violation of best solution
        is_feasible: Whether best solution is feasible
        iterations: Number of iterations performed
        history: Convergence history
        feasibility_rate: Percentage of feasible solutions in population
        avg_violation: Average constraint violation
    """
    x_best: np.ndarray
    obj_best: float
    violation_best: float
    is_feasible: bool
    iterations: int
    history: List[dict] = field(default_factory=list)
    feasibility_rate: float = 0.0
    avg_violation: float = 0.0
    population_objs: np.ndarray = field(default_factory=lambda: np.array([]))
    population_violations: np.ndarray = field(default_factory=lambda: np.array([]))


class SLPSOSolver:
    """SL-PSO solver for Mixed Integer Programming problems.

    The algorithm uses social learning where particles learn from
    demonstrators (better particles) rather than storing personal/global bests.

    For MIP problems:
    - Integer variables are rounded for evaluation
    - Constraint violations are penalized in fitness
    - Feasibility tracking throughout optimization

    Example:
        >>> solver = SLPSOSolver(A, b, c, lb, ub, integer_vars=[0, 1, 2])
        >>> result = solver.solve()
        >>> print(f"Best objective: {result.obj_best}")
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        config: Optional[SLPSOConfig] = None,
    ):
        """Initialize SL-PSO solver.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            c: Objective coefficients (n,)
            lb: Variable lower bounds (n,)
            ub: Variable upper bounds (n,)
            integer_vars: Indices of integer variables
            config: Algorithm configuration
        """
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub
        self.integer_vars = integer_vars or []
        self.config = config or SLPSOConfig()

        self.m, self.n = A.shape
        self.N = self.config.population_size

        # Initialize population
        self.X = None  # Positions (N x n)
        self.V = None  # Velocities (N x n)
        self.fitness = None  # Fitness values (N,)
        self.violations = None  # Constraint violations (N,)

        # History
        self.history = []

        # Best solution found
        self.x_best = None
        self.obj_best = float('inf')
        self.violation_best = float('inf')

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def _initialize_population(self) -> None:
        """Initialize particle positions and velocities."""
        # Random initialization within bounds
        self.X = np.random.uniform(
            low=self.lb,
            high=self.ub,
            size=(self.N, self.n)
        )

        # Initialize velocities to small random values
        v_range = self.ub - self.lb
        self.V = np.random.uniform(
            low=-0.1 * v_range,
            high=0.1 * v_range,
            size=(self.N, self.n)
        )

        # Evaluate initial population
        self._evaluate_population()

    def _round_integer_vars(self, x: np.ndarray) -> np.ndarray:
        """Round integer variables to nearest integers.

        Args:
            x: Solution vector or matrix (n,) or (N x n)

        Returns:
            Solution with integer variables rounded
        """
        if not self.integer_vars:
            return x.copy()

        x_rounded = x.copy()
        if x.ndim == 1:
            x_rounded[self.integer_vars] = np.round(x[self.integer_vars])
        else:
            x_rounded[:, self.integer_vars] = np.round(x[:, self.integer_vars])
        return x_rounded

    def _compute_constraint_violation(self, x: np.ndarray) -> float:
        """Compute total constraint violation for a solution.

        Args:
            x: Solution vector (n,)

        Returns:
            Total constraint violation (L1 norm)
        """
        Ax = self.A @ x
        violation = np.maximum(Ax - self.b, 0)
        return np.sum(violation)

    def _repair_solution(self, x: np.ndarray, max_repair_iter: int = 3) -> np.ndarray:
        """Repair infeasible solution using gradient-based projection.

        Attempts to reduce constraint violations by moving toward feasible region.

        Args:
            x: Infeasible solution vector (n,)
            max_repair_iter: Maximum repair iterations

        Returns:
            Repaired solution (may still be infeasible but with reduced violation)
        """
        x_repair = x.copy()

        for _ in range(max_repair_iter):
            Ax = self.A @ x_repair
            violation = Ax - self.b

            # Check if feasible
            if np.all(violation <= 1e-6):
                break

            # Find most violated constraints
            violated = violation > 1e-6
            if not np.any(violated):
                break

            # Compute gradient direction for repair
            # Move in direction that reduces violation
            grad = np.zeros(self.n)
            for i in np.where(violated)[0]:
                row = self.A.getrow(i).toarray().flatten()
                grad += violation[i] * row

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break

            grad = grad / grad_norm

            # Line search to reduce violation
            step_size = 0.1
            for _ in range(5):
                x_new = x_repair - step_size * grad
                x_new = np.clip(x_new, self.lb, self.ub)

                new_violation = self._compute_constraint_violation(x_new)
                old_violation = self._compute_constraint_violation(x_repair)

                if new_violation < old_violation:
                    x_repair = x_new
                    break
                step_size *= 0.5

        return x_repair

    def _compute_objective(self, x: np.ndarray) -> float:
        """Compute objective value.

        Args:
            x: Solution vector (n,)

        Returns:
            Objective value
        """
        return self.c @ x

    def _evaluate_solution(self, x: np.ndarray, apply_repair: bool = False) -> Tuple[float, float]:
        """Evaluate a single solution.

        Args:
            x: Solution vector (n,)
            apply_repair: Whether to apply repair mechanism

        Returns:
            (objective, violation)
        """
        # Round integer variables for evaluation
        x_eval = self._round_integer_vars(x)

        # Clip to bounds
        x_eval = np.clip(x_eval, self.lb, self.ub)

        # Apply repair if solution is infeasible (only at final stage)
        if apply_repair:
            violation_before = self._compute_constraint_violation(x_eval)
            if violation_before > 1e-6:
                x_eval = self._repair_solution(x_eval)

        obj = self._compute_objective(x_eval)
        violation = self._compute_constraint_violation(x_eval)

        return obj, violation

    def _evaluate_population(self) -> None:
        """Evaluate entire population."""
        self.fitness = np.zeros(self.N)
        self.violations = np.zeros(self.N)

        for i in range(self.N):
            obj, viol = self._evaluate_solution(self.X[i])
            self.fitness[i] = obj
            self.violations[i] = viol

    def _compute_fitness_with_penalty(self, obj: float, violation: float) -> float:
        """Compute penalized fitness.

        Lower fitness is better (minimization).

        Args:
            obj: Objective value
            violation: Constraint violation

        Returns:
            Penalized fitness
        """
        return obj + self.config.penalty_coeff * violation

    def _sort_population(self) -> np.ndarray:
        """Sort population by fitness (penalized objective).

        Returns:
            Indices of sorted particles (best first)
        """
        penalized_fitness = np.array([
            self._compute_fitness_with_penalty(self.fitness[i], self.violations[i])
            for i in range(self.N)
        ])
        return np.argsort(penalized_fitness)

    def _update_velocity_slpso(
        self,
        i: int,
        sorted_indices: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Update velocity using SL-PSO social learning mechanism.

        Particle i learns from demonstrators (particles better than it).

        Args:
            i: Particle index
            sorted_indices: Sorted particle indices (best first)
            iteration: Current iteration

        Returns:
            New velocity vector
        """
        # Find position of particle i in sorted order
        pos_in_sorted = np.where(sorted_indices == i)[0][0]

        # Compute mean position of population
        x_mean = np.mean(self.X, axis=0)

        # Initialize new velocity
        v_new = np.zeros(self.n)

        # For each dimension
        for d in range(self.n):
            # Social learning probability (dimension-dependent)
            # Better particles have lower phi (less exploration)
            phi_d = self.config.phi * (1 - pos_in_sorted / self.N)

            if np.random.random() < phi_d:
                # Learn from population mean (social learning)
                v_new[d] = (
                    self.config.alpha * self.V[i, d] +
                    self.config.gamma * np.random.randn() * (x_mean[d] - self.X[i, d])
                )
            else:
                # Learn from demonstrators (better particles)
                # Select a random demonstrator from better particles
                if pos_in_sorted > 0:
                    demonstrator_idx = np.random.choice(sorted_indices[:pos_in_sorted])
                else:
                    # Best particle - learn from itself with small perturbation
                    demonstrator_idx = i

                demonstrator = self.X[demonstrator_idx]

                v_new[d] = (
                    self.config.alpha * self.V[i, d] +
                    self.config.beta * np.random.random() * (demonstrator[d] - self.X[i, d]) +
                    self.config.gamma * np.random.randn()
                )

        return v_new

    def _update_position(self, i: int) -> np.ndarray:
        """Update position of particle i.

        Args:
            i: Particle index

        Returns:
            New position vector
        """
        x_new = self.X[i] + self.V[i]

        # Reflect at boundaries
        for d in range(self.n):
            if x_new[d] < self.lb[d]:
                x_new[d] = self.lb[d] + np.random.random() * (self.ub[d] - self.lb[d])
                self.V[i, d] *= -0.5  # Reverse and dampen velocity
            elif x_new[d] > self.ub[d]:
                x_new[d] = self.ub[d] - np.random.random() * (self.ub[d] - self.lb[d])
                self.V[i, d] *= -0.5

        return x_new

    def _adapt_penalty(self, iteration: int) -> None:
        """Adapt penalty coefficient based on feasibility rate.

        Args:
            iteration: Current iteration
        """
        if not self.config.adaptive_penalty:
            return

        # Count feasible solutions
        feasible_count = np.sum(self.violations < 1e-6)
        feasibility_rate = feasible_count / self.N

        # Adapt penalty coefficient
        if feasibility_rate < 0.1:
            # Too few feasible solutions - increase penalty
            self.config.penalty_coeff *= 1.1
        elif feasibility_rate > 0.5:
            # Many feasible solutions - can decrease penalty
            self.config.penalty_coeff *= 0.95

        # Clamp penalty coefficient
        self.config.penalty_coeff = np.clip(self.config.penalty_coeff, 1.0, 1e6)

    def solve(self, verbose: bool = False) -> SLPSOResult:
        """Run SL-PSO optimization.

        Args:
            verbose: Print progress information

        Returns:
            SLPSOResult with best solution and statistics
        """
        # Initialize
        self._initialize_population()
        self.history = []

        # Track best feasible solution
        best_feasible_obj = float('inf')
        best_feasible_x = None
        best_feasible_violation = float('inf')

        for iteration in range(1, self.config.max_iter + 1):
            # Sort population by fitness
            sorted_indices = self._sort_population()

            # Update each particle
            for i in range(self.N):
                # Update velocity
                self.V[i] = self._update_velocity_slpso(i, sorted_indices, iteration)

                # Update position
                self.X[i] = self._update_position(i)

            # Evaluate updated population
            self._evaluate_population()

            # Adapt penalty coefficient
            self._adapt_penalty(iteration)

            # Update best feasible solution
            for i in range(self.N):
                if self.violations[i] < 1e-6 and self.fitness[i] < best_feasible_obj:
                    best_feasible_obj = self.fitness[i]
                    best_feasible_x = self.X[i].copy()
                    best_feasible_violation = self.violations[i]

            # Record history
            feasible_count = np.sum(self.violations < 1e-6)
            feasibility_rate = feasible_count / self.N

            self.history.append({
                'iteration': iteration,
                'best_obj': best_feasible_obj if best_feasible_x is not None else float('inf'),
                'avg_violation': np.mean(self.violations),
                'feasibility_rate': feasibility_rate,
                'penalty_coeff': self.config.penalty_coeff,
            })

            if verbose and iteration % 500 == 0:
                print(
                    f"Iter {iteration}: best_obj={best_feasible_obj:.4f}, "
                    f"feasibility={feasibility_rate:.2%}, "
                    f"penalty={self.config.penalty_coeff:.2e}"
                )

        # Determine final best solution
        if best_feasible_x is not None:
            x_best = best_feasible_x
            obj_best = best_feasible_obj
            violation_best = best_feasible_violation
            is_feasible = True
        else:
            # No feasible solution found - return best penalized
            sorted_indices = self._sort_population()
            best_idx = sorted_indices[0]
            x_best = self.X[best_idx]
            obj_best = self.fitness[best_idx]
            violation_best = self.violations[best_idx]
            is_feasible = False

        # Round integer variables in final solution
        x_best = self._round_integer_vars(x_best)
        x_best = np.clip(x_best, self.lb, self.ub)

        # Re-evaluate final solution
        obj_best, violation_best = self._evaluate_solution(x_best)
        is_feasible = violation_best < 1e-6

        return SLPSOResult(
            x_best=x_best,
            obj_best=obj_best,
            violation_best=violation_best,
            is_feasible=is_feasible,
            iterations=self.config.max_iter,
            history=self.history,
            feasibility_rate=feasibility_rate,
            avg_violation=np.mean(self.violations),
            population_objs=self.fitness.copy(),
            population_violations=self.violations.copy(),
        )


def solve_mip_slpso(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: Optional[List[int]] = None,
    population_size: int = 50,
    max_iter: int = 5000,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> SLPSOResult:
    """Solve MIP using SL-PSO.

    Convenience function for solving MIP problems with SL-PSO.

    Args:
        A: Constraint matrix
        b: Right-hand side
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        integer_vars: Indices of integer variables
        population_size: Number of particles
        max_iter: Maximum iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        SLPSOResult with solution and statistics
    """
    config = SLPSOConfig(
        population_size=population_size,
        max_iter=max_iter,
        seed=seed,
    )

    solver = SLPSOSolver(
        A=A,
        b=b,
        c=c,
        lb=lb,
        ub=ub,
        integer_vars=integer_vars,
        config=config,
    )

    return solver.solve(verbose=verbose)


if __name__ == "__main__":
    print("Testing SL-PSO on simple MIP problem...")

    # Simple test: min -x - 2y s.t. x + 2y <= 3, x, y binary
    A = sparse.csr_matrix([[1.0, 2.0]])
    b = np.array([3.0])
    c = np.array([-1.0, -2.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    integer_vars = [0, 1]

    result = solve_mip_slpso(
        A, b, c, lb, ub,
        integer_vars=integer_vars,
        population_size=30,
        max_iter=1000,
        seed=42,
        verbose=True,
    )

    print(f"\nBest solution: x = {result.x_best}")
    print(f"Best objective: {result.obj_best:.4f}")
    print(f"Constraint violation: {result.violation_best:.6e}")
    print(f"Is feasible: {result.is_feasible}")
    print(f"Feasibility rate: {result.feasibility_rate:.2%}")
    print(f"\nExpected: x=[1, 1], obj=-3, violation=0")
