"""
Simulated Annealing (SA) solver for MIP - baseline comparison.

Standard SA with temperature schedule and neighborhood search.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Tuple
from scipy import sparse


class SimulatedAnnealingSolver:
    """Simulated Annealing for MIP - classical baseline."""

    def __init__(self, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 integer_vars: Optional[List[int]] = None,
                 constraint_sense: Optional[List[str]] = None):
        """Initialize SA solver.

        Args:
            A: Constraint matrix
            b: RHS vector
            c: Objective coefficients
            lb: Lower bounds
            ub: Upper bounds
            integer_vars: Indices of integer variables
            constraint_sense: List of constraint senses ('<', '>', '=')
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.integer_vars = integer_vars or []
        self.constraint_sense = constraint_sense

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
        """Compute constraint violation penalty for energy function."""
        is_feas, max_viol = self._check_feasibility(x)
        if is_feas:
            return 0.0
        # Quadratic penalty
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

    def _energy(self, x: np.ndarray, penalty_weight: float = 1e6) -> float:
        """Energy function: objective + penalty for violations."""
        obj = self.c @ x
        viol_penalty = self._compute_violation_penalty(x)
        return obj + penalty_weight * viol_penalty

    def _generate_neighbor(self, x: np.ndarray, temperature: float) -> np.ndarray:
        """Generate neighbor solution.

        For integer variables: try flipping or small steps.
        For continuous: random perturbation.
        """
        x_new = x.copy()

        # Determine perturbation scale based on temperature
        scale = min(1.0, temperature) * 0.5

        # Perturb integer variables
        if self.integer_vars and np.random.random() < 0.7:
            # Focus on integer variables
            i = np.random.choice(self.integer_vars)
            current_val = x[i]
            # Try rounding or small step
            if np.random.random() < 0.5:
                x_new[i] = round(current_val)
            else:
                delta = np.random.choice([-1, 0, 1])
                x_new[i] = round(current_val) + delta
        else:
            # Perturb random variable
            i = np.random.randint(0, self.n)
            x_new[i] += np.random.randn() * scale

        # Clip to bounds
        x_new = np.clip(x_new, self.lb, self.ub)

        return x_new

    def solve(self, max_iter: int = 10000, seed: int = 42,
              initial_temperature: float = 100.0,
              cooling_rate: float = 0.995,
              verbose: bool = False) -> Dict:
        """Solve using Simulated Annealing.

        Args:
            max_iter: Maximum iterations
            seed: Random seed
            initial_temperature: Starting temperature
            cooling_rate: Temperature cooling rate per iteration
            verbose: Print progress

        Returns:
            Dictionary with solution info
        """
        np.random.seed(seed)

        # Initialize with random feasible-ish point
        # Handle infinite bounds by using a reasonable range
        lb_clipped = np.clip(self.lb, -1e6, 1e6)
        ub_clipped = np.clip(self.ub, -1e6, 1e6)
        ub_clipped = np.maximum(ub_clipped, lb_clipped + 1e-6)  # Ensure ub > lb

        x = np.random.uniform(lb_clipped, ub_clipped)
        # Round integer variables
        for i in self.integer_vars:
            x[i] = round(x[i])
        x = np.clip(x, self.lb, self.ub)

        best_x = x.copy()
        best_energy = self._energy(x)
        best_obj = float(self.c @ x)
        best_is_feas = False

        temperature = initial_temperature

        history = []
        start_time = time.time()

        for iteration in range(1, max_iter + 1):
            # Generate neighbor
            x_neighbor = self._generate_neighbor(x, temperature)

            # Compute energies
            E_current = self._energy(x)
            E_neighbor = self._energy(x_neighbor)

            # Acceptance criterion
            delta_E = E_neighbor - E_current

            if delta_E < 0:
                # Better solution, always accept
                x = x_neighbor
            else:
                # Worse solution, accept with probability
                acceptance_prob = np.exp(-delta_E / temperature)
                if np.random.random() < acceptance_prob:
                    x = x_neighbor

            # Update best
            is_feas, _ = self._check_feasibility(x)
            is_int_feas, _ = self._check_integer_feasibility(x)
            current_obj = float(self.c @ x)

            if is_feas and is_int_feas:
                if not best_is_feas or current_obj < best_obj:
                    best_x = x.copy()
                    best_obj = current_obj
                    best_is_feas = True
            elif not best_is_feas:
                # Track best even if not feasible (lower energy)
                if self._energy(x) < best_energy:
                    best_x = x.copy()
                    best_energy = self._energy(x)
                    best_obj = current_obj

            # Cool down
            temperature *= cooling_rate

            # Record history
            if iteration % 100 == 0 or iteration == max_iter:
                elapsed = time.time() - start_time
                history.append({
                    'iteration': iteration,
                    'obj': current_obj,
                    'temperature': temperature,
                    'is_feasible': is_feas and is_int_feas,
                    'time': elapsed
                })

                if verbose:
                    print(f"Iter {iteration}: T={temperature:.4f}, "
                          f"obj={current_obj:.4f}, feas={is_feas and is_int_feas}")

        total_time = time.time() - start_time

        # Final check
        is_feas, primal_viol = self._check_feasibility(best_x)
        is_int_feas, int_viol = self._check_integer_feasibility(best_x)

        # Round integer variables for final solution
        x_final = best_x.copy()
        for i in self.integer_vars:
            x_final[i] = round(x_final[i])

        # Re-check after rounding
        is_feas, primal_viol = self._check_feasibility(x_final)
        is_int_feas, int_viol = self._check_integer_feasibility(x_final)
        final_obj = float(self.c @ x_final)

        return {
            'x_best': x_final,
            'obj_best': final_obj if (is_feas and is_int_feas) else float('inf'),
            'is_feasible': is_feas and is_int_feas,
            'is_integer_feasible': is_int_feas,
            'primal_violation': primal_viol,
            'integrality_violation': int_viol,
            'iterations': max_iter,
            'solve_time': total_time,
            'history': history
        }


def solve_sa(A, b, c, lb, ub, integer_vars=None, constraint_sense=None,
             max_iter=10000, seed=42):
    """Convenience function for SA."""
    solver = SimulatedAnnealingSolver(A, b, c, lb, ub, integer_vars, constraint_sense)
    return solver.solve(max_iter=max_iter, seed=seed)
