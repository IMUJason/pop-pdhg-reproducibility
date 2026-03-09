"""
Two-Phase Solver for General Constraint Types.

This solver handles problems with mixed constraint types (<=, >=, =) by:
Phase I: Find feasible solution using auxiliary problem
Phase II: Optimize from feasible point

Key insight: Instead of trying to make PDHG handle all constraints directly,
we use PDHG as a subroutine for solving auxiliary problems.
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import time

from core.problem_transformer import ProblemTransformer, ConstraintSense, TransformedProblem


@dataclass
class TwoPhaseResult:
    """Result of two-phase solving."""
    x_best: np.ndarray
    obj_best: float
    phase1_success: bool
    phase2_success: bool
    phase1_iters: int
    phase2_iters: int
    total_time: float
    final_violation: float


class TwoPhaseSolver:
    """Two-phase solver for problems with mixed constraint types.

    Phase I: Solve auxiliary problem to find feasible solution
             min sum(v_i) where v_i are constraint violations

    Phase II: From feasible point, optimize original objective
              while maintaining feasibility
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        senses: List[ConstraintSense],
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None,
    ):
        """Initialize two-phase solver.

        Args:
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            senses: List of constraint senses
            lb: Variable lower bounds
            ub: Variable upper bounds
            integer_vars: Indices of integer variables
        """
        self.A_orig = A.copy()
        self.b_orig = b.copy()
        self.c_orig = c.copy()
        self.senses = np.array(senses)
        self.lb_orig = lb.copy()
        self.ub_orig = ub.copy()
        self.integer_vars = integer_vars or []

        self.m, self.n = A.shape

        # Create problem transformer
        self.transformer = ProblemTransformer()
        self.transformed = self.transformer.transform(
            A, b, c, senses, lb, ub, integer_vars
        )

    def _build_phase_one_problem(self) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build Phase I problem: minimize constraint violation.

        Key insight: Convert all constraints to <= form WITH slack variables
        that directly represent violation.

        For Ax >= b (>= constraint):
        - Convert to -Ax <= -b
        - Add slack: -Ax - s <= -b where s >= 0
        - This means: if Ax < b (violation), we need s >= b - Ax > 0

        For Ax <= b (<= constraint):
        - Keep as Ax <= b
        - Add slack: Ax - s <= b where s >= 0
        - This means: if Ax > b (violation), we need s >= Ax - b > 0

        For Ax = b (= constraint):
        - Convert to two constraints: Ax <= b and -Ax <= -b
        - Add slack to both: Ax - s1 <= b and -Ax - s2 <= -b
        - Minimize s1 + s2

        Returns:
            A_phase1, b_phase1, c_phase1, lb_phase1, ub_phase1
        """
        # For = constraints, we need 2 slacks (one for each direction)
        n_slacks = self.m + np.sum(self.senses == ConstraintSense.EQ)
        n_total = self.n + n_slacks

        rows, cols, data = [], [], []
        slack_idx = self.n  # Current slack variable index
        rhs_list = []

        for i in range(self.m):
            row = self.A_orig.getrow(i).toarray().flatten()

            if self.senses[i] == ConstraintSense.LE:
                # <= constraint: Ax - s <= b
                # Add original coefficients
                for j in range(self.n):
                    if row[j] != 0:
                        rows.append(len(rhs_list))
                        cols.append(j)
                        data.append(row[j])
                # Slack: Ax - s <= b means s >= Ax - b (violation)
                rows.append(len(rhs_list))
                cols.append(slack_idx)
                data.append(-1.0)
                slack_idx += 1

                rhs_list.append(self.b_orig[i])

            elif self.senses[i] == ConstraintSense.GE:
                # >= constraint: convert to -Ax - s <= -b
                # Add negated coefficients
                for j in range(self.n):
                    if row[j] != 0:
                        rows.append(len(rhs_list))
                        cols.append(j)
                        data.append(-row[j])
                # Slack: -Ax - s <= -b means s >= b - Ax (violation)
                rows.append(len(rhs_list))
                cols.append(slack_idx)
                data.append(-1.0)
                slack_idx += 1

                rhs_list.append(-self.b_orig[i])

            elif self.senses[i] == ConstraintSense.EQ:
                # = constraint: split into two with separate slacks
                # Ax - s1 <= b
                for j in range(self.n):
                    if row[j] != 0:
                        rows.append(len(rhs_list))
                        cols.append(j)
                        data.append(row[j])
                rows.append(len(rhs_list))
                cols.append(slack_idx)
                data.append(-1.0)
                s1_idx = slack_idx
                slack_idx += 1
                rhs_list.append(self.b_orig[i])

                # -Ax - s2 <= -b
                for j in range(self.n):
                    if row[j] != 0:
                        rows.append(len(rhs_list))
                        cols.append(j)
                        data.append(-row[j])
                rows.append(len(rhs_list))
                cols.append(slack_idx)
                data.append(-1.0)
                s2_idx = slack_idx
                slack_idx += 1
                rhs_list.append(-self.b_orig[i])

        A_phase1 = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(rhs_list), n_total)
        )
        b_phase1 = np.array(rhs_list)

        # Objective: minimize sum of slacks
        c_phase1 = np.zeros(n_total)
        c_phase1[self.n:] = 1.0  # Minimize slacks

        # Bounds: original + slacks >= 0
        lb_phase1 = np.concatenate([self.lb_orig, np.zeros(n_slacks)])
        ub_phase1 = np.concatenate([self.ub_orig, np.full(n_slacks, np.inf)])

        return A_phase1, b_phase1, c_phase1, lb_phase1, ub_phase1

    def solve_phase_one(
        self,
        pdhg_solver_class,
        max_iters: int = 5000,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Solve Phase I to find feasible solution.

        Args:
            pdhg_solver_class: Class to use for solving (e.g., QuantumPopulationPDHG)
            max_iters: Maximum iterations
            tol: Tolerance for feasibility
            verbose: Print progress

        Returns:
            (feasible_x or None, best_violation)
        """
        if verbose:
            print("\nPhase I: Finding feasible solution...")

        # Build Phase I problem
        A_p1, b_p1, c_p1, lb_p1, ub_p1 = self._build_phase_one_problem()

        # Create solver
        solver = pdhg_solver_class(
            A_p1, b_p1, c_p1, lb_p1, ub_p1,
            population_size=16,
        )

        # Use diverse initialization (not zeros!)
        from population.pop_pdhg import PopulationState

        np.random.seed(42)
        K = 16
        X = np.zeros((K, len(lb_p1)))

        # Initialize original variables with small positive values
        for k in range(K):
            X[k, :self.n] = np.random.uniform(0.1, 0.5, self.n)
            X[k, :self.n] = np.clip(X[k, :self.n], self.lb_orig, self.ub_orig)

        # Initialize slacks to ensure constraints are satisfied
        # For each constraint row, compute required slack
        slack_row = 0
        for i in range(self.m):
            if self.senses[i] == ConstraintSense.LE:
                # Ax - s <= b, need s >= Ax - b
                Ax = self.A_orig[i] @ X[0, :self.n]
                slack_needed = max(Ax - self.b_orig[i], 0.1)
                X[:, self.n + slack_row] = slack_needed + np.random.uniform(0, 0.1, K)
                slack_row += 1
            elif self.senses[i] == ConstraintSense.GE:
                # -Ax - s <= -b, need s >= b - Ax
                Ax = self.A_orig[i] @ X[0, :self.n]
                slack_needed = max(self.b_orig[i] - Ax, 0.1)
                X[:, self.n + slack_row] = slack_needed + np.random.uniform(0, 0.1, K)
                slack_row += 1
            elif self.senses[i] == ConstraintSense.EQ:
                # Ax - s1 <= b, need s1 >= Ax - b
                # -Ax - s2 <= -b, need s2 >= b - Ax
                Ax = self.A_orig[i] @ X[0, :self.n]
                slack1_needed = max(Ax - self.b_orig[i], 0.1)
                slack2_needed = max(self.b_orig[i] - Ax, 0.1)
                X[:, self.n + slack_row] = slack1_needed + np.random.uniform(0, 0.1, K)
                slack_row += 1
                X[:, self.n + slack_row] = slack2_needed + np.random.uniform(0, 0.1, K)
                slack_row += 1

        Y = np.abs(np.random.randn(K, len(b_p1)))
        obj = np.array([c_p1 @ X[k] for k in range(K)])
        primal_feas = np.zeros(K)
        for k in range(K):
            Ax = A_p1 @ X[k]
            primal_feas[k] = np.linalg.norm(np.maximum(Ax - b_p1, 0), ord=np.inf)
        dual_feas = np.linalg.norm(A_p1.T @ Y.T, axis=0)
        age = np.zeros(K, dtype=int)

        state = PopulationState(x=X, y=Y, obj=obj, primal_feas=primal_feas, dual_feas=dual_feas, age=age)

        # Store state in solver
        solver.state = state

        # Solve
        result = solver.solve(
            max_iter=max_iters,
            tol=tol,
            seed=42,
            integer_vars=self.integer_vars,  # Integer vars are in original part
            use_enhanced_repair=True,
            use_feasibility_aware_tunnel=True,
            use_two_phase=False,
            verbose=verbose,
        )

        # Extract solution (original variables only)
        x_full = result.x_best
        x_orig = x_full[:self.n]

        # Check feasibility against ORIGINAL constraints
        max_viol, violations = self.transformed.compute_original_violation(x_full)

        if verbose:
            print(f"  Phase I complete:")
            print(f"    Slack sum: {np.sum(x_full[self.n:]):.4e}")
            print(f"    Original violation: {max_viol:.4e}")

        if max_viol <= tol:
            return x_orig, max_viol
        else:
            # Try repair
            x_repaired = self.transformer.repair_with_original_senses(
                x_full, self.integer_vars, max_iters=200
            )
            max_viol_repaired, _ = self.transformed.compute_original_violation(x_repaired)

            if max_viol_repaired <= tol:
                return x_repaired[:self.n], max_viol_repaired

            return None, min(max_viol, max_viol_repaired)

    def solve_phase_two(
        self,
        x_feas: np.ndarray,
        pdhg_solver_class,
        max_iters: int = 5000,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Solve Phase II: Optimize from feasible point.

        Args:
            x_feas: Feasible starting point
            pdhg_solver_class: Class to use for solving
            max_iters: Maximum iterations
            tol: Tolerance
            verbose: Print progress

        Returns:
            (x_opt, obj_value)
        """
        if verbose:
            print("\nPhase II: Optimizing from feasible point...")

        # Create solver with ORIGINAL transformed problem
        solver = pdhg_solver_class(
            self.transformed.A_pdh,
            self.transformed.b_pdh,
            self.transformed.c_expanded,
            self.transformed.lb,
            self.transformed.ub,
            population_size=16,
        )

        # Initialize all members from feasible point with small perturbations
        from population.pop_pdhg import PopulationState

        K = 16
        X = np.zeros((K, len(self.transformed.c_expanded)))

        for k in range(K):
            if k == 0:
                # First member at feasible point
                X[k, :self.n] = x_feas
            else:
                # Perturb
                perturb = np.random.randn(self.n) * 0.05
                X[k, :self.n] = x_feas + perturb
                X[k, :self.n] = np.clip(X[k, :self.n], self.lb_orig, self.ub_orig)

                # Round integers
                for i in self.integer_vars:
                    X[k, i] = round(X[k, i])

            # Compute slacks for >= constraints
            Ax = self.A_orig @ X[k, :self.n]
            for i, ge_idx in enumerate(self.transformed.ge_indices):
                slack_idx = self.n + i
                X[k, slack_idx] = max(Ax[ge_idx] - self.b_orig[ge_idx], 0.01)

        Y = np.abs(np.random.randn(K, len(self.transformed.b_pdh)))
        obj = np.array([self.transformed.c_expanded @ X[k] for k in range(K)])
        primal_feas = np.zeros(K)
        for k in range(K):
            Ax = self.transformed.A_pdh @ X[k]
            primal_feas[k] = np.linalg.norm(np.maximum(Ax - self.transformed.b_pdh, 0), ord=np.inf)
        dual_feas = np.linalg.norm(self.transformed.A_pdh.T @ Y.T, axis=0)
        age = np.zeros(K, dtype=int)

        state = PopulationState(x=X, y=Y, obj=obj, primal_feas=primal_feas, dual_feas=dual_feas, age=age)
        solver.state = state

        # Solve with stricter feasibility maintenance
        result = solver.solve(
            max_iter=max_iters,
            tol=tol,
            seed=42,
            integer_vars=self.integer_vars,
            use_enhanced_repair=True,
            use_feasibility_aware_tunnel=True,
            use_two_phase=False,
            verbose=verbose,
        )

        # Extract and check
        x_full = result.x_best
        x_orig = x_full[:self.n]

        max_viol, _ = self.transformed.compute_original_violation(x_full)

        if verbose:
            print(f"  Phase II complete:")
            print(f"    Objective: {self.c_orig @ x_orig:.4f}")
            print(f"    Constraint violation: {max_viol:.4e}")

        return x_orig, self.c_orig @ x_orig

    def solve(
        self,
        pdhg_solver_class,
        max_total_iters: int = 10000,
        tol: float = 1e-6,
        phase1_ratio: float = 0.4,
        verbose: bool = False,
    ) -> TwoPhaseResult:
        """Run complete two-phase solving.

        Args:
            pdhg_solver_class: Class to use for solving (e.g., QuantumPopulationPDHG)
            max_total_iters: Maximum total iterations
            tol: Tolerance
            phase1_ratio: Fraction of iterations for Phase 1
            verbose: Print progress

        Returns:
            TwoPhaseResult
        """
        start_time = time.time()

        phase1_iters = int(max_total_iters * phase1_ratio)
        phase2_iters = max_total_iters - phase1_iters

        # Phase I
        x_feas, violation = self.solve_phase_one(
            pdhg_solver_class,
            max_iters=phase1_iters,
            tol=tol,
            verbose=verbose,
        )

        phase1_success = x_feas is not None

        if not phase1_success:
            if verbose:
                print("\nPhase I failed to find feasible solution")

            return TwoPhaseResult(
                x_best=np.zeros(self.n),
                obj_best=float('inf'),
                phase1_success=False,
                phase2_success=False,
                phase1_iters=phase1_iters,
                phase2_iters=0,
                total_time=time.time() - start_time,
                final_violation=violation,
            )

        # Phase II
        x_opt, obj = self.solve_phase_two(
            x_feas,
            pdhg_solver_class,
            max_iters=phase2_iters,
            tol=tol,
            verbose=verbose,
        )

        # Final verification
        # Create dummy full vector to check violation
        x_full = np.zeros(self.n + self.transformed.n_slack)
        x_full[:self.n] = x_opt
        Ax = self.A_orig @ x_opt
        for i, ge_idx in enumerate(self.transformed.ge_indices):
            slack_idx = self.n + i
            x_full[slack_idx] = max(Ax[ge_idx] - self.b_orig[ge_idx], 0)

        final_viol, _ = self.transformed.compute_original_violation(x_full)

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*80}")
            print("Two-Phase Solving Complete")
            print(f"{'='*80}")
            print(f"Phase I success: {phase1_success}")
            print(f"Final objective: {obj:.4f}")
            print(f"Final violation: {final_viol:.4e}")
            print(f"Total time: {total_time:.1f}s")

        return TwoPhaseResult(
            x_best=x_opt,
            obj_best=obj,
            phase1_success=phase1_success,
            phase2_success=final_viol <= tol,
            phase1_iters=phase1_iters,
            phase2_iters=phase2_iters,
            total_time=total_time,
            final_violation=final_viol,
        )
