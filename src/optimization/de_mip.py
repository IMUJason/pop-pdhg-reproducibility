"""
Differential Evolution (DE) for Mixed Integer Programming.

This module implements DE/rand/1/bin adapted for MIP problems.
Based on:
- Storn & Price (1997): "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization"

Adaptations for MIP:
- Integer/binary variables handled via rounding before evaluation
- Constraints handled via penalty method
- Boundary repair for box constraints
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from scipy import sparse


@dataclass
class DEConfig:
    """Configuration for DE algorithm.

    Attributes:
        population_size: Number of individuals (NP)
        max_evals: Maximum function evaluations
        F: Differential weight (mutation factor)
        CR: Crossover probability
        strategy: DE strategy ('rand/1/bin', 'best/1/bin', etc.)
        tol: Convergence tolerance
    """
    population_size: int = 100
    max_evals: int = 100000
    F: float = 0.8  # Mutation factor
    CR: float = 0.9  # Crossover probability
    strategy: str = 'rand/1/bin'
    tol: float = 1e-6


@dataclass
class DEResult:
    """Result of DE optimization."""
    x_best: np.ndarray
    obj_best: float
    feasibility: float
    evaluations: int
    converged: bool
    history: List[dict] = field(default_factory=list)


class DEMIP:
    """DE algorithm for Mixed Integer Programming."""

    def __init__(self, c: np.ndarray, A: sparse.csr_matrix, b: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 binary_vars: Optional[List[int]] = None,
                 config: Optional[DEConfig] = None,
                 seed: Optional[int] = None):
        self.c = c
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.integer_vars = integer_vars or []
        self.binary_vars = binary_vars or []
        self.config = config or DEConfig()
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

    def _mutate(self, idx: int) -> np.ndarray:
        """DE/rand/1 mutation."""
        # Select 3 random individuals different from idx
        candidates = [i for i in range(self.config.population_size) if i != idx]
        r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)

        x1, x2, x3 = self.population[r1], self.population[r2], self.population[r3]
        mutant = x1 + self.config.F * (x2 - x3)

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover."""
        trial = np.copy(target)
        j_rand = self.rng.randint(self.n)

        for j in range(self.n):
            if self.rng.rand() < self.config.CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def optimize(self) -> DEResult:
        """Run DE optimization."""
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

        # Main loop
        while n_evals < config.max_evals:
            for i in range(NP):
                # Mutation
                mutant = self._mutate(i)

                # Crossover
                trial = self._crossover(self.population[i], mutant)

                # Repair
                trial = self._repair_bounds(trial)

                # Evaluate trial
                trial_fitness = self._penalty_objective(trial)
                trial_obj, trial_viol = self._evaluate(trial)
                n_evals += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.violations[i] = trial_viol

                    # Update best
                    if trial_fitness < self.fitness[best_idx]:
                        best_idx = i
                        x_best = trial.copy()
                        obj_best = trial_obj
                        feas_best = trial_viol

                if n_evals >= config.max_evals:
                    break

            # Record history every NP evaluations
            if len(history) == 0 or n_evals - history[-1]['eval'] >= NP:
                history.append({'eval': n_evals, 'obj': obj_best, 'feas': feas_best})

        return DEResult(
            x_best=x_best,
            obj_best=obj_best,
            feasibility=feas_best,
            evaluations=n_evals,
            converged=feas_best < 1e-3,
            history=history
        )


def solve_mip_de(c: np.ndarray, A: sparse.csr_matrix, b: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 binary_vars: Optional[List[int]] = None,
                 population_size: int = 100,
                 max_evals: int = 100000,
                 F: float = 0.8,
                 CR: float = 0.9,
                 seed: Optional[int] = None) -> DEResult:
    """Solve MIP using DE."""
    config = DEConfig(
        population_size=population_size,
        max_evals=max_evals,
        F=F,
        CR=CR
    )
    solver = DEMIP(c, A, b, lb, ub, integer_vars, binary_vars, config, seed)
    return solver.optimize()
