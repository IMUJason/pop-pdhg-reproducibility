"""
Genetic Algorithm (GA) solver for MIP - baseline comparison.

Standard GA with selection, crossover, and mutation operators.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Tuple
from scipy import sparse


class GeneticAlgorithmSolver:
    """Genetic Algorithm for MIP - classical baseline."""

    def __init__(self, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 constraint_sense: Optional[List[str]] = None,
                 population_size: int = 50):
        """Initialize GA solver.

        Args:
            A: Constraint matrix
            b: RHS vector
            c: Objective coefficients
            lb: Lower bounds
            ub: Upper bounds
            integer_vars: Indices of integer variables
            constraint_sense: List of constraint senses ('<', '>', '=')
            population_size: Size of population
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.integer_vars = integer_vars or []
        self.constraint_sense = constraint_sense
        self.population_size = population_size

        self.m, self.n = A.shape

    def _check_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check feasibility on original problem."""
        Ax = self.A @ x

        if self.constraint_sense is None:
            violation = np.maximum(Ax - self.b, 0).max()
            return violation < 1e-4, float(violation)

        max_violation = 0.0
        for i, s in enumerate(self.constraint_sense):
            if s == '<':
                viol = max(Ax[i] - self.b[i], 0)
            elif s == '>':
                viol = max(self.b[i] - Ax[i], 0)
            elif s == '=':
                viol = abs(Ax[i] - self.b[i])
            else:
                viol = max(Ax[i] - self.b[i], 0)
            max_violation = max(max_violation, viol)

        return max_violation < 1e-4, float(max_violation)

    def _check_integer_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check integer feasibility."""
        if not self.integer_vars:
            return True, 0.0

        violations = [abs(x[i] - round(x[i])) for i in self.integer_vars]
        max_viol = max(violations) if violations else 0.0
        return max_viol < 1e-4, max_viol

    def _compute_violation_penalty(self, x: np.ndarray) -> float:
        """Compute constraint violation penalty for fitness."""
        Ax = self.A @ x
        if self.constraint_sense is None:
            return np.sum(np.maximum(Ax - self.b, 0) ** 2)

        total = 0.0
        for i, s in enumerate(self.constraint_sense):
            if s == '<':
                viol = max(Ax[i] - self.b[i], 0)
            elif s == '>':
                viol = max(self.b[i] - Ax[i], 0)
            elif s == '=':
                viol = abs(Ax[i] - self.b[i])
            else:
                viol = max(Ax[i] - self.b[i], 0)
            total += viol ** 2
        return total

    def _fitness(self, x: np.ndarray, penalty_weight: float = 1e6) -> float:
        """Fitness function (higher is better).

        Uses penalty method for constraints.
        """
        obj = self.c @ x
        viol_penalty = self._compute_violation_penalty(x)
        # Negative because we minimize, and fitness should be maximized
        return -(obj + penalty_weight * viol_penalty)

    def _initialize_population(self) -> np.ndarray:
        """Initialize diverse population."""
        population = np.zeros((self.population_size, self.n))

        # Handle infinite bounds
        lb_clipped = np.clip(self.lb, -1e6, 1e6)
        ub_clipped = np.clip(self.ub, -1e6, 1e6)
        ub_clipped = np.maximum(ub_clipped, lb_clipped + 1e-6)

        for i in range(self.population_size):
            if i == 0:
                # Zero initialization
                population[i] = np.zeros(self.n)
            elif i < self.population_size // 3:
                # Random uniform
                population[i] = np.random.uniform(lb_clipped, ub_clipped)
            elif i < 2 * self.population_size // 3:
                # Random near lower bound
                population[i] = lb_clipped + np.random.rand(self.n) * 0.1 * (ub_clipped - lb_clipped)
            else:
                # Random near upper bound
                population[i] = ub_clipped - np.random.rand(self.n) * 0.1 * (ub_clipped - lb_clipped)

            # Round integer variables
            for j in self.integer_vars:
                population[i, j] = round(population[i, j])

            # Clip to bounds
            population[i] = np.clip(population[i], self.lb, self.ub)

        return population

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection."""
        selected = np.zeros_like(population)

        for i in range(self.population_size):
            # Tournament of 3
            candidates = np.random.choice(self.population_size, 3, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[winner].copy()

        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.rand(self.n) < 0.5

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        # Ensure integer variables stay integer
        for i in self.integer_vars:
            child1[i] = round(child1[i])
            child2[i] = round(child2[i])

        return np.clip(child1, self.lb, self.ub), np.clip(child2, self.lb, self.ub)

    def _mutation(self, x: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutation operator."""
        x_new = x.copy()

        for i in range(self.n):
            if np.random.random() < mutation_rate:
                if i in self.integer_vars:
                    # Integer mutation: flip or small change
                    if np.random.random() < 0.5:
                        x_new[i] = round(x[i] + np.random.choice([-1, 1]))
                    else:
                        lb_i = max(self.lb[i], -1e6)
                        ub_i = min(self.ub[i], 1e6)
                        ub_i = max(ub_i, lb_i + 1e-6)
                        x_new[i] = round(np.random.uniform(lb_i, ub_i))
                else:
                    # Continuous mutation: Gaussian perturbation
                    x_new[i] += np.random.randn() * 0.1 * (self.ub[i] - self.lb[i])

        return np.clip(x_new, self.lb, self.ub)

    def solve(self, max_iter: int = 1000, seed: int = 42,
              crossover_rate: float = 0.8,
              mutation_rate: float = 0.1,
              elitism: int = 2,
              verbose: bool = False) -> Dict:
        """Solve using Genetic Algorithm.

        Args:
            max_iter: Maximum generations
            seed: Random seed
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            elitism: Number of elite individuals to preserve
            verbose: Print progress

        Returns:
            Dictionary with solution info
        """
        np.random.seed(seed)

        # Initialize population
        population = self._initialize_population()
        fitness = np.array([self._fitness(x) for x in population])

        best_x = population[np.argmax(fitness)].copy()
        best_fitness = fitness.max()
        best_obj = float(self.c @ best_x)

        # Track best feasible
        best_feasible_x = None
        best_feasible_obj = float('inf')

        history = []
        start_time = time.time()

        for generation in range(1, max_iter + 1):
            # Selection
            selected = self._selection(population, fitness)

            # Crossover
            offspring = np.zeros_like(population)
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]

                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                offspring[i] = child1
                offspring[(i + 1) % self.population_size] = child2

            # Mutation
            for i in range(self.population_size):
                offspring[i] = self._mutation(offspring[i], mutation_rate)

            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness)[-elitism:]
            for i, idx in enumerate(elite_indices):
                offspring[i] = population[idx].copy()

            # Update population
            population = offspring
            fitness = np.array([self._fitness(x) for x in population])

            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fitness:
                best_fitness = fitness[best_idx]
                best_x = population[best_idx].copy()
                best_obj = float(self.c @ best_x)

            # Check for feasible solutions
            for i, x in enumerate(population):
                is_feas, _ = self._check_feasibility(x)
                is_int_feas, _ = self._check_integer_feasibility(x)
                if is_feas and is_int_feas:
                    obj = float(self.c @ x)
                    if obj < best_feasible_obj:
                        best_feasible_x = x.copy()
                        best_feasible_obj = obj

            # Record history
            if generation % 50 == 0 or generation == max_iter:
                elapsed = time.time() - start_time

                # Current best feasibility
                is_feas, primal_viol = self._check_feasibility(best_x)
                is_int_feas, int_viol = self._check_integer_feasibility(best_x)

                history.append({
                    'generation': generation,
                    'obj': best_obj,
                    'best_feasible_obj': best_feasible_obj if best_feasible_x is not None else float('inf'),
                    'max_fitness': best_fitness,
                    'is_feasible': is_feas and is_int_feas,
                    'time': elapsed
                })

                if verbose:
                    feas_str = "feas" if (is_feas and is_int_feas) else "infeas"
                    print(f"Gen {generation}: obj={best_obj:.4f}, "
                          f"best_feas={best_feasible_obj:.4f}, {feas_str}")

        total_time = time.time() - start_time

        # Use best feasible if found
        if best_feasible_x is not None:
            final_x = best_feasible_x
            is_feas, primal_viol = self._check_feasibility(final_x)
            is_int_feas, int_viol = self._check_integer_feasibility(final_x)
            final_obj = best_feasible_obj
        else:
            # No feasible found, return best overall
            final_x = best_x
            is_feas, primal_viol = self._check_feasibility(final_x)
            is_int_feas, int_viol = self._check_integer_feasibility(final_x)
            final_obj = float(self.c @ final_x)

            # Try to round integers for final solution
            x_rounded = final_x.copy()
            for i in self.integer_vars:
                x_rounded[i] = round(x_rounded[i])
            is_feas_r, _ = self._check_feasibility(x_rounded)
            is_int_feas_r, _ = self._check_integer_feasibility(x_rounded)

            if is_feas_r and is_int_feas_r:
                final_x = x_rounded
                is_feas = True
                is_int_feas = True
                final_obj = float(self.c @ final_x)

        return {
            'x_best': final_x,
            'obj_best': final_obj if (is_feas and is_int_feas) else float('inf'),
            'is_feasible': is_feas and is_int_feas,
            'is_integer_feasible': is_int_feas,
            'primal_violation': primal_viol,
            'integrality_violation': int_viol,
            'generations': max_iter,
            'solve_time': total_time,
            'history': history
        }


def solve_ga(A, b, c, lb, ub, integer_vars=None, constraint_sense=None,
             population_size=50, max_generations=1000, seed=42):
    """Convenience function for GA."""
    solver = GeneticAlgorithmSolver(A, b, c, lb, ub, integer_vars, constraint_sense,
                                    population_size)
    return solver.solve(max_iter=max_generations, seed=seed)
