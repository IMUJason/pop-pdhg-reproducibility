"""
Problem Transformer for General Constraint Types.

This module transforms problems with mixed constraint types (<=, >=, =)
into a form suitable for PDHG while preserving the original problem structure.

Key insight: Instead of converting >= to <= (which makes x=0 artificially feasible),
we convert >= to = with slack variables, then handle via Augmented Lagrangian.

This approach:
1. PRESERVES PDHG's core algorithm (no changes to iterations)
2. Makes constraint violations detectable in the transformed space
3. Allows proper feasibility checking against original constraints
4. Uses slack variables to ensure zero is NOT artificially feasible
"""

import numpy as np
from scipy import sparse
import enum
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


class ConstraintSense(enum.IntEnum):
    """Constraint sense types."""
    LE = 0  # <=
    GE = 1  # >=
    EQ = 2  # ==


@dataclass
class TransformedProblem:
    """Transformed problem data structure.

    Contains both the transformed problem (for PDHG) and original data
    (for proper feasibility checking).
    """
    # Transformed matrices for PDHG (all <= form with slacks)
    A_pdh: sparse.csr_matrix
    b_pdh: np.ndarray

    # Original problem data
    A_orig: sparse.csr_matrix
    b_orig: np.ndarray
    senses: np.ndarray  # ConstraintSense values

    # Transformation info
    n_orig: int  # Original number of variables
    n_slack: int  # Number of slack variables added
    slack_indices: List[int]  # Indices of slack variables
    ge_indices: List[int]  # Indices of >= constraints
    eq_indices: List[int]  # Indices of == constraints

    # Variable bounds (including slacks)
    lb: np.ndarray
    ub: np.ndarray

    # Original objective
    c_orig: np.ndarray

    # Extended objective (with zero cost for slacks)
    c_expanded: np.ndarray

    def compute_original_violation(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute constraint violation using ORIGINAL senses.

        This is crucial for proper feasibility checking.

        Args:
            x: Solution vector (may include slack variables)

        Returns:
            (max_violation, violation_vector)
        """
        # Extract original variables (exclude slacks)
        x_orig = x[:self.n_orig]

        # Compute Ax for original constraints
        Ax = self.A_orig @ x_orig

        # Compute violations based on original senses
        violations = np.zeros(len(self.senses))

        for i, sense in enumerate(self.senses):
            if sense == ConstraintSense.LE:
                # Ax <= b, violation is max(Ax - b, 0)
                violations[i] = max(Ax[i] - self.b_orig[i], 0)
            elif sense == ConstraintSense.GE:
                # Ax >= b, violation is max(b - Ax, 0)
                violations[i] = max(self.b_orig[i] - Ax[i], 0)
            elif sense == ConstraintSense.EQ:
                # Ax == b, violation is |Ax - b|
                violations[i] = abs(Ax[i] - self.b_orig[i])

        max_viol = np.max(violations)
        return max_viol, violations

    def compute_original_objective(self, x: np.ndarray) -> float:
        """Compute objective using original variables only."""
        x_orig = x[:self.n_orig]
        return self.c_orig @ x_orig

    def extract_original_solution(self, x: np.ndarray) -> np.ndarray:
        """Extract solution for original variables (excluding slacks)."""
        return x[:self.n_orig].copy()

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check feasibility against original constraints."""
        max_viol, _ = self.compute_original_violation(x)
        return max_viol <= tol


class ProblemTransformer:
    """Transforms problems with mixed constraint types for PDHG.

    The transformation strategy:
    1. Keep <= constraints as-is
    2. Convert >= constraints to == with slack >= 0
       (Ax >= b becomes Ax - s = b, s >= 0)
    3. Keep == constraints as-is
    4. Convert all == to two <= constraints for PDHG

    This ensures that:
    - x=0 is NOT automatically feasible for >= constraints (s would be negative)
    - Slack variables must be non-negative
    - PDHG can handle the problem in standard form
    """

    def __init__(self):
        self.transformed = None

    def transform(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        senses: List[ConstraintSense],
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None
    ) -> TransformedProblem:
        """Transform problem with mixed constraint types.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            c: Objective coefficients (n,)
            senses: List of ConstraintSense for each constraint
            lb: Variable lower bounds
            ub: Variable upper bounds
            integer_vars: Indices of integer variables

        Returns:
            TransformedProblem with all necessary data
        """
        m, n = A.shape
        senses = np.array(senses)

        # Identify constraint types
        le_indices = np.where(senses == ConstraintSense.LE)[0].tolist()
        ge_indices = np.where(senses == ConstraintSense.GE)[0].tolist()
        eq_indices = np.where(senses == ConstraintSense.EQ)[0].tolist()

        n_le = len(le_indices)
        n_ge = len(ge_indices)
        n_eq = len(eq_indices)
        n_slack = n_ge  # One slack per >= constraint

        print(f"  Transforming: {n_le} <=, {n_ge} >= (with {n_slack} slacks), {n_eq} ==")

        # Build extended problem:
        # Original variables + slack variables (for >= constraints)
        n_total = n + n_slack

        # Build new constraint matrix for equality form first
        # For >= constraints: Ax - s = b, with s >= 0
        # This is equivalent to: Ax - b = s >= 0, i.e., Ax >= b

        rows = []
        cols = []
        data = []
        b_eq_extended = []

        # <= constraints: keep as Ax <= b
        for idx in le_indices:
            row = A.getrow(idx).toarray().flatten()
            for j in range(n):
                if row[j] != 0:
                    rows.append(len(b_eq_extended))
                    cols.append(j)
                    data.append(row[j])
            b_eq_extended.append(b[idx])

        # >= constraints: Ax - s = b (s is slack, s >= 0)
        for i, idx in enumerate(ge_indices):
            row = A.getrow(idx).toarray().flatten()
            # Original variables
            for j in range(n):
                if row[j] != 0:
                    rows.append(len(b_eq_extended))
                    cols.append(j)
                    data.append(row[j])
            # Slack variable (negative coefficient: -s)
            slack_idx = n + i
            rows.append(len(b_eq_extended))
            cols.append(slack_idx)
            data.append(-1.0)

            b_eq_extended.append(b[idx])

        # == constraints: keep as Ax = b
        for idx in eq_indices:
            row = A.getrow(idx).toarray().flatten()
            for j in range(n):
                if row[j] != 0:
                    rows.append(len(b_eq_extended))
                    cols.append(j)
                    data.append(row[j])
            b_eq_extended.append(b[idx])

        # Create equality-constrained matrix
        m_eq = len(b_eq_extended)
        A_eq = sparse.csr_matrix((data, (rows, cols)), shape=(m_eq, n_total))
        b_eq = np.array(b_eq_extended)

        # Convert to PDHG form (all <=):
        # Ax = b becomes: Ax <= b and -Ax <= -b
        m_pdh = 2 * m_eq
        A_pdh = sparse.vstack([A_eq, -A_eq], format='csr')
        b_pdh = np.hstack([b_eq, -b_eq])

        # Extended bounds (slacks are >= 0)
        lb_extended = np.hstack([lb, np.zeros(n_slack)])
        ub_extended = np.hstack([ub, np.full(n_slack, np.inf)])

        # Extended objective (slacks have zero cost)
        c_expanded = np.hstack([c, np.zeros(n_slack)])

        # Create result
        self.transformed = TransformedProblem(
            A_pdh=A_pdh,
            b_pdh=b_pdh,
            A_orig=A.copy(),
            b_orig=b.copy(),
            senses=senses,
            n_orig=n,
            n_slack=n_slack,
            slack_indices=list(range(n, n + n_slack)),
            ge_indices=ge_indices,
            eq_indices=eq_indices,
            lb=lb_extended,
            ub=ub_extended,
            c_orig=c.copy(),
            c_expanded=c_expanded
        )

        return self.transformed

    def get_initial_point(
        self,
        strategy: str = "feasible_seed",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate an initial point for the transformed problem.

        CRITICAL: For >= constraints, we CANNOT start from x=0 because
        that would require negative slack values.

        Strategies:
        - "feasible_seed": Start with small positive values, compute valid slacks
        - "random": Random point within bounds

        Args:
            strategy: Initialization strategy
            seed: Random seed

        Returns:
            Initial point x0 (length n_orig + n_slack)
        """
        if self.transformed is None:
            raise ValueError("Must call transform() first")

        if seed is not None:
            np.random.seed(seed)

        n_orig = self.transformed.n_orig
        n_slack = self.transformed.n_slack
        n_total = n_orig + n_slack
        lb = self.transformed.lb
        ub = self.transformed.ub

        if strategy == "feasible_seed":
            # Start with small positive values for original variables
            x0 = np.zeros(n_total)

            # Set original variables to small positive values
            for i in range(n_orig):
                if np.isfinite(lb[i]):
                    x0[i] = max(lb[i] + 0.1, 0.1)
                else:
                    x0[i] = 0.1

            # Compute slacks for >= constraints
            x_orig = x0[:n_orig]
            Ax = self.transformed.A_orig @ x_orig

            for i, ge_idx in enumerate(self.transformed.ge_indices):
                slack_idx = n_orig + i
                # slack = Ax - b (for Ax >= b)
                slack_val = Ax[ge_idx] - self.transformed.b_orig[ge_idx]
                x0[slack_idx] = max(slack_val, 0.01)  # Ensure non-negative

        elif strategy == "random":
            # Random within bounds (but ensure slacks are non-negative)
            x0 = np.zeros(n_total)
            for i in range(n_orig):
                if np.isfinite(lb[i]) and np.isfinite(ub[i]):
                    low = max(lb[i], 0.01)  # Start positive
                    x0[i] = np.random.uniform(low, ub[i])
                elif np.isfinite(lb[i]):
                    x0[i] = lb[i] + np.random.exponential(0.5)
                elif np.isfinite(ub[i]):
                    x0[i] = min(ub[i], max(0.01, np.random.exponential(0.5)))
                else:
                    x0[i] = np.abs(np.random.randn()) + 0.01

            # Compute slacks for >= constraints
            x_orig = x0[:n_orig]
            Ax = self.transformed.A_orig @ x_orig

            for i, ge_idx in enumerate(self.transformed.ge_indices):
                slack_idx = n_orig + i
                slack_val = Ax[ge_idx] - self.transformed.b_orig[ge_idx]
                x0[slack_idx] = max(slack_val, 0.01)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Clip to bounds
        x0 = np.clip(x0, lb, ub)

        return x0

    def repair_with_original_senses(
        self,
        x: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        max_iters: int = 200,
        tol: float = 1e-6
    ) -> np.ndarray:
        """Repair solution using original constraint senses.

        This is key: we repair based on ORIGINAL constraints, not PDHG form.

        Args:
            x: Solution vector (may include slacks)
            integer_vars: Integer variable indices (in original space)
            max_iters: Maximum iterations
            tol: Tolerance

        Returns:
            Repaired solution
        """
        if self.transformed is None:
            raise ValueError("Must call transform() first")

        x_repaired = x.copy()
        n_orig = self.transformed.n_orig
        integer_vars = integer_vars or []

        # Repair original variables
        for iteration in range(max_iters):
            max_viol, violations = self.transformed.compute_original_violation(x_repaired)

            if max_viol <= tol:
                break

            # Extract original solution
            x_orig = x_repaired[:n_orig]

            # Compute gradient in original space
            grad = np.zeros(n_orig)
            Ax = self.transformed.A_orig @ x_orig

            for i in range(len(self.transformed.senses)):
                if violations[i] > tol:
                    row = self.transformed.A_orig[i].toarray().flatten()
                    sense = self.transformed.senses[i]

                    if sense == ConstraintSense.LE:
                        # Need to reduce Ax
                        grad += row * violations[i]
                    elif sense == ConstraintSense.GE:
                        # Need to increase Ax
                        grad -= row * violations[i]
                    elif sense == ConstraintSense.EQ:
                        # Need to move toward b
                        sign = np.sign(Ax[i] - self.transformed.b_orig[i])
                        grad += sign * row * violations[i]

            # Normalize
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            grad = grad / grad_norm

            # Line search
            best_viol = max_viol
            best_x = x_repaired.copy()

            for step in [1.0, 0.5, 0.25, 0.1, 0.05]:
                x_test = x_repaired.copy()
                x_test[:n_orig] = x_orig - step * grad * min(max_viol, 1.0)
                x_test[:n_orig] = np.clip(x_test[:n_orig],
                                          self.transformed.lb[:n_orig],
                                          self.transformed.ub[:n_orig])

                # Round integers
                for i in integer_vars:
                    x_test[i] = round(x_test[i])

                # Update slacks
                new_orig = x_test[:n_orig]
                Ax_new = self.transformed.A_orig @ new_orig

                for i, ge_idx in enumerate(self.transformed.ge_indices):
                    slack_idx = n_orig + i
                    x_test[slack_idx] = max(Ax_new[ge_idx] - self.transformed.b_orig[ge_idx], 0)

                new_viol, _ = self.transformed.compute_original_violation(x_test)

                if new_viol < best_viol:
                    best_viol = new_viol
                    best_x = x_test.copy()

            if best_viol < max_viol:
                x_repaired = best_x
            else:
                break

        return x_repaired


def transform_problem_for_pdhg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    senses: List[ConstraintSense],
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: Optional[List[int]] = None
) -> Dict:
    """Convenience function to transform a problem for PDHG.

    Returns a dictionary with all necessary information for the solver.
    """
    transformer = ProblemTransformer()
    transformed = transformer.transform(A, b, c, senses, lb, ub, integer_vars)

    return {
        'transformer': transformer,
        'transformed': transformed,
        'A_pdh': transformed.A_pdh,
        'b_pdh': transformed.b_pdh,
        'c_expanded': transformed.c_expanded,
        'lb_expanded': transformed.lb,
        'ub_expanded': transformed.ub,
        'n_orig': transformed.n_orig,
        'n_slack': transformed.n_slack,
        'slack_indices': transformed.slack_indices,
    }
