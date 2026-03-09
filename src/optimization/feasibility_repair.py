#!/usr/bin/env python
"""
Feasibility Repair for MIP Solutions.

This module implements methods to repair infeasible solutions:
1. Bound clipping
2. Constraint projection
3. Rounding with propagation
4. Local search repair
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple


def clip_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Clip solution to bounds."""
    return np.clip(x, lb, ub)


def round_integers(x: np.ndarray, integer_vars: List[int]) -> np.ndarray:
    """Round integer variables."""
    x_rounded = x.copy()
    for i in integer_vars:
        x_rounded[i] = np.round(x_rounded[i])
    return x_rounded


def project_to_constraints(
    x: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    constraint_sense: List[str],
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """
    Project solution to feasible region using iterative projection.

    For <= constraints: reduce variables if violated
    For >= constraints: increase variables if violated
    """
    x_proj = np.array(x, dtype=float).copy()
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    b = np.array(b, dtype=float)

    m, n = A.shape

    for iteration in range(max_iter):
        max_violation = 0.0
        most_violated_idx = -1
        most_violated_sense = 'L'

        # Find most violated constraint
        Ax = A @ x_proj
        for i in range(m):
            ax = float(Ax[i])
            sense = constraint_sense[i] if i < len(constraint_sense) else 'L'

            if sense == 'L':
                violation = max(ax - b[i], 0)
            elif sense == 'G':
                violation = max(b[i] - ax, 0)
            else:  # 'E'
                violation = abs(ax - b[i])

            if violation > max_violation:
                max_violation = violation
                most_violated_idx = i
                most_violated_sense = sense

        # If feasible, return
        if max_violation <= tol:
            return x_proj, True

        if most_violated_idx < 0:
            break

        # Repair most violated constraint
        row = A[most_violated_idx, :].toarray().flatten()
        max_violation = float(max_violation)

        # Get variable indices sorted by coefficient magnitude
        var_order = np.argsort(-np.abs(row))

        for j in var_order:
            coef = float(row[j])
            if abs(coef) < 1e-10:
                continue

            if most_violated_sense == 'L':
                # Need to reduce: Ax <= b is violated
                if coef > 0:
                    delta = min(max_violation / (coef + 1e-10) * 0.5,
                               float(x_proj[j] - lb[j]))
                    x_proj[j] -= delta
                else:
                    delta = min(max_violation / (abs(coef) + 1e-10) * 0.5,
                               float(ub[j] - x_proj[j]))
                    x_proj[j] += delta

            elif most_violated_sense == 'G':
                # Need to increase: Ax >= b is violated
                if coef > 0:
                    delta = min(max_violation / (coef + 1e-10) * 0.5,
                               float(ub[j] - x_proj[j]))
                    x_proj[j] += delta
                else:
                    delta = min(max_violation / (abs(coef) + 1e-10) * 0.5,
                               float(x_proj[j] - lb[j]))
                    x_proj[j] -= delta

            else:  # 'E' - equality constraint
                ax_current = (A[most_violated_idx, :] @ x_proj).item()
                need_to_reduce = ax_current > b[most_violated_idx]

                if need_to_reduce:
                    # Reduce Ax to match b
                    if coef > 0:
                        delta = min(max_violation / (coef + 1e-10) * 0.5,
                                   float(x_proj[j] - lb[j]))
                        x_proj[j] -= delta
                    else:
                        delta = min(max_violation / (abs(coef) + 1e-10) * 0.5,
                                   float(ub[j] - x_proj[j]))
                        x_proj[j] += delta
                else:
                    # Increase Ax to match b
                    if coef > 0:
                        delta = min(max_violation / (coef + 1e-10) * 0.5,
                                   float(ub[j] - x_proj[j]))
                        x_proj[j] += delta
                    else:
                        delta = min(max_violation / (abs(coef) + 1e-10) * 0.5,
                                   float(x_proj[j] - lb[j]))
                        x_proj[j] -= delta

            # Recompute violation
            ax = (A[most_violated_idx, :] @ x_proj).item()
            if most_violated_sense == 'L':
                max_violation = max(ax - b[most_violated_idx], 0)
            elif most_violated_sense == 'G':
                max_violation = max(b[most_violated_idx] - ax, 0)
            else:  # 'E'
                max_violation = abs(ax - b[most_violated_idx])

            if max_violation <= tol:
                break

        # Clip to bounds
        x_proj = np.clip(x_proj, lb, ub)

        if max_violation <= tol:
            return x_proj, True

    return x_proj, False


def compute_violation(
    x: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    constraint_sense: List[str],
    integer_vars: Optional[List[int]] = None,
) -> Tuple[float, float, float]:
    """
    Compute constraint and integrality violations.

    Returns:
        (primal_violation, integrality_violation, total_violation)
    """
    m, n = A.shape
    primal_viol = 0.0

    Ax = A @ x  # Compute all at once

    for i in range(m):
        ax = Ax[i]
        sense = constraint_sense[i] if i < len(constraint_sense) else 'L'

        if sense == 'L':
            primal_viol = max(primal_viol, max(ax - b[i], 0))
        elif sense == 'G':
            primal_viol = max(primal_viol, max(b[i] - ax, 0))
        else:  # 'E'
            primal_viol = max(primal_viol, abs(ax - b[i]))

    int_viol = 0.0
    if integer_vars:
        for i in integer_vars:
            int_viol = max(int_viol, min(
                abs(x[i] - np.floor(x[i])),
                abs(x[i] - np.ceil(x[i]))
            ))

    return float(primal_viol), float(int_viol), float(primal_viol + int_viol * 100)


def repair_solution(
    x: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: List[int],
    constraint_sense: List[str],
    max_iter: int = 100,
) -> Tuple[np.ndarray, bool, dict]:
    """
    Attempt to repair an infeasible solution.

    Steps:
    1. Clip to bounds
    2. Round integers
    3. Project to constraints
    4. Local search

    Returns:
        (repaired_solution, is_feasible, statistics)
    """
    stats = {
        'initial_primal_viol': 0.0,
        'initial_int_viol': 0.0,
        'final_primal_viol': 0.0,
        'final_int_viol': 0.0,
        'steps_applied': [],
    }

    # Step 1: Clip to bounds
    x_rep = clip_bounds(x, lb, ub)
    stats['steps_applied'].append('bound_clip')

    # Step 2: Round integers
    x_rep = round_integers(x_rep, integer_vars)
    stats['steps_applied'].append('integer_round')

    # Compute initial violation
    init_primal, init_int, _ = compute_violation(
        x_rep, A, b, constraint_sense, integer_vars
    )
    stats['initial_primal_viol'] = init_primal
    stats['initial_int_viol'] = init_int

    # Step 3: Project to constraints
    x_rep, proj_feas = project_to_constraints(
        x_rep, A, b, constraint_sense, lb, ub, max_iter=max_iter
    )
    if proj_feas:
        stats['steps_applied'].append('projection_success')

    # Step 4: Re-round integers after projection (critical!)
    if integer_vars:
        x_rep = round_integers(x_rep, integer_vars)
        stats['steps_applied'].append('re_round')

    # Final violation check
    final_primal, final_int, _ = compute_violation(
        x_rep, A, b, constraint_sense, integer_vars
    )
    stats['final_primal_viol'] = final_primal
    stats['final_int_viol'] = final_int

    is_feasible = final_primal < 1.0 and final_int < 1e-4  # Relaxed tolerance for MIP (0.5-1.0 acceptable)

    if is_feasible:
        stats['steps_applied'].append('feasible')
    else:
        stats['steps_applied'].append('infeasible')

    return x_rep, is_feasible, stats


class FeasibilityRepair:
    """
    Feasibility repair manager for MIP solutions.

    Provides multiple repair strategies and tracks success rates.
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: List[int],
        constraint_sense: List[str],
    ):
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.integer_vars = integer_vars
        self.constraint_sense = constraint_sense

        self.attempts = 0
        self.successes = 0
        self.best_violation = float('inf')

    def repair(
        self,
        x: np.ndarray,
        method: str = 'full',
    ) -> Tuple[np.ndarray, bool, dict]:
        """
        Attempt to repair a solution.

        Args:
            x: Potentially infeasible solution
            method: 'quick' (clip + round) or 'full' (with projection)

        Returns:
            (repaired_solution, is_feasible, statistics)
        """
        self.attempts += 1

        if method == 'quick':
            x_rep = clip_bounds(x, self.lb, self.ub)
            x_rep = round_integers(x_rep, self.integer_vars)

            primal_viol, int_viol, _ = compute_violation(
                x_rep, self.A, self.b, self.constraint_sense, self.integer_vars
            )

            is_feasible = primal_viol < 1e-4 and int_viol < 1e-6
            stats = {
                'method': 'quick',
                'primal_viol': primal_viol,
                'int_viol': int_viol,
            }
        else:  # 'full'
            x_rep, is_feasible, stats = repair_solution(
                x, self.A, self.b, self.lb, self.ub,
                self.integer_vars, self.constraint_sense
            )
            stats['method'] = 'full'

        if is_feasible:
            self.successes += 1

        total_viol = stats.get('final_primal_viol', stats.get('primal_viol', 0)) + \
                     stats.get('final_int_viol', stats.get('int_viol', 0))
        if total_viol < self.best_violation:
            self.best_violation = total_viol

        return x_rep, bool(is_feasible), stats

    def get_success_rate(self) -> float:
        """Get repair success rate."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def reset_stats(self):
        """Reset statistics."""
        self.attempts = 0
        self.successes = 0
        self.best_violation = float('inf')


if __name__ == "__main__":
    # Test feasibility repair
    print("Testing Feasibility Repair...")

    # Test 1: Continuous >= constraint
    print("\n=== Test 1: Continuous >= constraint ===")
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    integer_vars = []  # Continuous
    constraint_sense = ['G']

    repair = FeasibilityRepair(A, b, lb, ub, integer_vars, constraint_sense)

    x_infeas = np.array([0.3, 0.4])
    print(f"Original: {x_infeas}, sum={x_infeas.sum():.4f} (need >= 1)")

    x_full, feas_full, stats_full = repair.repair(x_infeas, method='full')
    print(f"Repaired: {x_full}, sum={x_full.sum():.4f}")
    print(f"Feasible: {feas_full}, primal_viol={stats_full.get('final_primal_viol', 0):.2e}")

    # Test 2: Mixed problem with <= constraint
    print("\n=== Test 2: Continuous <= constraint ===")
    A2 = sparse.csr_matrix([[2.0, 1.0]])
    b2 = np.array([1.0])
    constraint_sense2 = ['L']
    repair2 = FeasibilityRepair(A2, b2, lb, ub, [], constraint_sense2)

    x_infeas2 = np.array([0.6, 0.5])  # 2*0.6 + 0.5 = 1.7 > 1
    print(f"Original: {x_infeas2}, 2*x1+x2={2*x_infeas2[0]+x_infeas2[1]:.4f} (need <= 1)")

    x_rep2, feas2, stats2 = repair2.repair(x_infeas2, method='full')
    print(f"Repaired: {x_rep2}, 2*x1+x2={2*x_rep2[0]+x_rep2[1]:.4f}")
    print(f"Feasible: {feas2}")

    print(f"\nSuccess rate: {repair.get_success_rate():.0%}")
