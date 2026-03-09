"""
Genetic Algorithm (GA) for Mixed Integer Programming.

This module implements a real-coded GA adapted for MIP problems.
Based on:
- Holland (1975): "Adaptation in Natural and Artificial Systems"
- Deb (2000): "An Efficient Constraint Handling Method for Genetic Algorithms"

Adaptations for MIP:
- Integer/binary variables handled via rounding before evaluation
- Constraints handled via penalty method
- Tournament selection for constrained optimization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy import sparse


@dataclass
class GAConfig:
    """Configuration for GA algorithm.

    Attributes:
        population_size: Number of individuals
        max_generations: Maximum number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation per gene
        elite_size: Number of elite individuals to preserve
        tournament_size: Tournament size for selection
        tol: Convergence tolerance
    """
    population_size: int = 100
    max_generations: int = 1000
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    elite_size: int = 2
    tournament_size: int = 3
    tol: float = 1e-6


@dataclass
class GAResult:
    """Result of GA optimization."""
    x_best: np.ndarray
    obj_best: float
    feasibility: float
    evaluations: int
    converged: bool
    history: List[dict] = field(default_factory=list)


class GAMIP:
    """GA algorithm for Mixed Integer Programming."""

    def __init__(self, c: np.ndarray, A: sparse.csr_matrix, b: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 binary_vars: Optional[List[int]] = None,
                 config: Optional[GAConfig] = None,
                 seed: Optional[int] = None):
        self.c = c
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.integer_vars = integer_vars or []
        self.binary_vars = binary_vars or []
        self.config = config or GAConfig()
        self.rng = np.random.RandomState(seed)

        self.n = len(c)
        self.m = A.shape[0] if A is not None else 0
        self.population = None
        self.fitness = None
        self.violations = None

    def _evaluate(self, x: np.ndarray) -> tuple:
        """Evaluate objective and constraint violation."""
        obj = float(self.c @ x)

        if self.A is not None:
            violation = self.A @ x - self.b
            max_viol = np.maximum(0, violation).max()
        else:
            max_viol = 0.0

        return obj, max_viol

    def _penalty_objective(self, x: np.ndarray, penalty_coeff: float = 10000.0) -> float:
        """Compute penalized objective."""
        obj, viol = self._evaluate(x)
        return obj + penalty_coeff * viol

    def _repair_bounds(self, x: np.ndarray) -> np.ndarray:
        """Repair solution to satisfy box constraints."""
        x = np.clip(x, self.lb, self.ub)

        # Round integer variables
        if self.integer_vars:
            x[self.integer_vars] = np.round(x[self.integer_vars])
        if self.binary_vars:
            x[self.binary_vars] = np.clip(np.round(x[self.binary_vars]), 0, 1)

        return x

    def _tournament_select(self) -> int:
        """Tournament selection - returns index of selected individual."""
        tournament_indices = self.rng.choice(
            self.config.population_size,
            size=self.config.tournament_size,
            replace=False
        )

        # Select best in tournament (lowest fitness)
        best_idx = tournament_indices[0]
        best_fitness = self.fitness[best_idx]

        for idx in tournament_indices[1:]:
            if self.fitness[idx] < best_fitness:
                best_fitness = self.fitness[idx]
                best_idx = idx

        return best_idx

    def _simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) for real-coded GA."""
        eta = 20.0  # Distribution index

        child1 = np.copy(parent1)
        child2 = np.copy(parent2)

        if self.rng.rand() < self.config.crossover_rate:
            for i in range(self.n):
                if self.rng.rand() <= 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:
                        if parent1[i] < parent2[i]:
                            y1, y2 = parent1[i], parent2[i]
                        else:
                            y1, y2 = parent2[i], parent1[i]

                        beta = 1.0 + (2.0 * (y1 - self.lb[i]) / (y2 - y1))
                        alpha = 2.0 - beta ** (-(eta + 1.0))

                        rand = self.rng.rand()
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                        beta = 1.0 + (2.0 * (self.ub[i] - y2) / (y2 - y1))
                        alpha = 2.0 - beta ** (-(eta + 1.0))

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                        child1[i] = np.clip(c1, self.lb[i], self.ub[i])
                        child2[i] = np.clip(c2, self.lb[i], self.ub[i])

        return child1, child2

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Polynomial mutation for real-coded GA."""
        eta_m = 20.0  # Mutation distribution index
        mutant = np.copy(x)

        for i in range(self.n):
            if self.rng.rand() < self.config.mutation_rate:
                delta1 = (x[i] - self.lb[i]) / (self.ub[i] - self.lb[i])
                delta2 = (self.ub[i] - x[i]) / (self.ub[i] - self.lb[i])

                rand = self.rng.rand()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow

                mutant[i] = x[i] + delta_q * (self.ub[i] - self.lb[i])
                mutant[i] = np.clip(mutant[i], self.lb[i], self.ub[i])

        return mutant

    def optimize(self) -> GAResult:
        """Run GA optimization."""
        config = self.config
        NP = config.population_size

        # Initialize population
        self.population = np.zeros((NP, self.n))
        for i in range(NP):
            self.population[i] = self.rng.uniform(self.lb, self.ub)
            self.population[i] = self._repair_bounds(self.population[i])

        # Evaluate initial population
        self.fitness = np.zeros(NP)
        self.violations = np.zeros(NP)
        for i in range(NP):
            obj, viol = self._evaluate(self.population[i])
            self.violations[i] = viol
            self.fitness[i] = self._penalty_objective(self.population[i])

        # Track best
        best_idx = np.argmin(self.fitness)
        x_best = self.population[best_idx].copy()
        obj_best, feas_best = self._evaluate(x_best)

        history = [{'eval': NP, 'obj': obj_best, 'feas': feas_best}]
        n_evals = NP
        generation = 0

        # Main loop
        while generation < config.max_generations:
            generation += 1

            # Create new population
            new_population = np.zeros_like(self.population)
            new_fitness = np.zeros(NP)
            new_violations = np.zeros(NP)

            # Elitism: preserve best individuals
            sorted_indices = np.argsort(self.fitness)
            for i in range(config.elite_size):
                new_population[i] = self.population[sorted_indices[i]]
                new_fitness[i] = self.fitness[sorted_indices[i]]
                new_violations[i] = self.violations[sorted_indices[i]]

            # Generate offspring
            idx = config.elite_size
            while idx < NP:
                # Selection
                parent1_idx = self._tournament_select()
                parent2_idx = self._tournament_select()

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                # Crossover
                child1, child2 = self._simulated_binary_crossover(parent1, parent2)

                # Mutation
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)

                # Repair
                child1 = self._repair_bounds(child1)
                child2 = self._repair_bounds(child2)

                # Evaluate
                fitness1 = self._penalty_objective(child1)
                obj1, viol1 = self._evaluate(child1)

                new_population[idx] = child1
                new_fitness[idx] = fitness1
                new_violations[idx] = viol1
                idx += 1
                n_evals += 1

                if idx < NP:
                    fitness2 = self._penalty_objective(child2)
                    obj2, viol2 = self._evaluate(child2)

                    new_population[idx] = child2
                    new_fitness[idx] = fitness2
                    new_violations[idx] = viol2
                    idx += 1
                    n_evals += 1

            # Replace population
            self.population = new_population
            self.fitness = new_fitness
            self.violations = new_violations

            # Update best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self._penalty_objective(x_best):
                x_best = self.population[best_idx].copy()
                obj_best, feas_best = self._evaluate(x_best)

            # Record history
            if generation % 10 == 0 or generation == 1:
                history.append({'eval': n_evals, 'obj': obj_best, 'feas': feas_best})

            # Check convergence
            if feas_best < 1e-3 and generation > 50:
                # Check if no improvement
                recent_history = history[-10:]
                if len(recent_history) >= 10:
                    obj_range = max(h['obj'] for h in recent_history) - min(h['obj'] for h in recent_history)
                    if obj_range < config.tol:
                        break

        return GAResult(
            x_best=x_best,
            obj_best=obj_best,
            feasibility=feas_best,
            evaluations=n_evals,
            converged=feas_best < 1e-3,
            history=history
        )


def solve_mip_ga(c: np.ndarray, A: sparse.csr_matrix, b: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 binary_vars: Optional[List[int]] = None,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 seed: Optional[int] = None) -> GAResult:
    """Solve MIP using GA."""
    config = GAConfig(
        population_size=population_size,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    solver = GAMIP(c, A, b, lb, ub, integer_vars, binary_vars, config, seed)
    return solver.optimize()
