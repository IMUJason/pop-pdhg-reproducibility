"""Transform mixed constraint types to equality constraints with slack variables.

This module transforms any LP/MIP with mixed constraints (<=, >=, =) into
an equivalent problem with only equality constraints and bound constraints.

Transformation:
    - Ax <= b    ->   Ax + s = b,   s >= 0
    - Ax >= b    ->   Ax - s = b,   s >= 0
    - Ax = b     ->   Ax = b        (no slack needed)

The transformed problem can be solved by PDHGEquality.
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple, Dict
import enum


class ConstraintSense(enum.IntEnum):
    """Constraint sense types."""
    LE = 0  # <=
    GE = 1  # >=
    EQ = 2  # =


class EqualityTransformer:
    """Transform mixed constraints to equality constraints with slack variables.

    This is the KEY enabler for solving problems with >= and = constraints.
    By converting all constraints to equalities, we avoid the all-zero solution
    being falsely feasible.
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        senses: List[ConstraintSense],
    ):
        """Initialize transformer.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            senses: List of constraint senses
        """
        self.A_orig = A.copy()
        self.b_orig = b.copy()
        self.senses = np.array(senses)
        self.m, self.n = A.shape

        # Identify constraint types
        self.le_mask = self.senses == ConstraintSense.LE
        self.ge_mask = self.senses == ConstraintSense.GE
        self.eq_mask = self.senses == ConstraintSense.EQ

        self.n_le = np.sum(self.le_mask)
        self.n_ge = np.sum(self.ge_mask)
        self.n_eq = np.sum(self.eq_mask)

        # Number of slack variables needed
        # For <=: add s >= 0, constraint becomes Ax + s = b
        # For >=: add s >= 0, constraint becomes Ax - s = b
        # For = : no slack needed
        self.n_slack = self.n_le + self.n_ge

        # Build transformation
        self._build_equality_form()

    def _build_equality_form(self):
        """Build the equality-constraint form with slack variables."""
        if self.n_slack == 0:
            # No transformation needed
            self.A_eq = self.A_orig.copy()
            self.b_eq = self.b_orig.copy()
            self.slack_indices = []
            self.original_indices = list(range(self.n))
            return

        # Build expanded constraint matrix
        # Variables: [original x, slack s]
        # Constraints: all equalities

        rows = []
        cols = []
        data = []

        slack_idx = 0
        self.slack_map = {}  # Map original constraint index to slack variable index

        for i in range(self.m):
            # Original constraint coefficients
            row_data = self.A_orig[i].toarray().flatten()
            for j in range(self.n):
                if abs(row_data[j]) > 1e-15:
                    rows.append(i)
                    cols.append(j)
                    data.append(row_data[j])

            # Add slack variable if needed
            if self.senses[i] == ConstraintSense.LE:
                # Ax <= b  ->  Ax + s = b, s >= 0
                rows.append(i)
                cols.append(self.n + slack_idx)
                data.append(1.0)
                self.slack_map[i] = self.n + slack_idx
                slack_idx += 1

            elif self.senses[i] == ConstraintSense.GE:
                # Ax >= b  ->  Ax - s = b, s >= 0
                # Or equivalently: -Ax + s = -b
                # Let's use: Ax - s = b with s >= 0
                rows.append(i)
                cols.append(self.n + slack_idx)
                data.append(-1.0)
                self.slack_map[i] = self.n + slack_idx
                slack_idx += 1

            # For EQ, no slack variable

        # Build matrix
        n_total_vars = self.n + self.n_slack
        self.A_eq = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.m, n_total_vars)
        )
        self.b_eq = self.b_orig.copy()

        # Track indices
        self.slack_indices = list(range(self.n, self.n + self.n_slack))
        self.original_indices = list(range(self.n))

    def get_equality_form(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Get the equality constraint form.

        Returns:
            (A_eq, b_eq): Equality constraint matrix and RHS
        """
        return self.A_eq, self.b_eq

    def expand_objective(self, c: np.ndarray) -> np.ndarray:
        """Expand objective vector to include slack variables.

        Slack variables have zero cost.

        Args:
            c: Original objective (n,)

        Returns:
            Expanded objective (n + n_slack,)
        """
        if self.n_slack == 0:
            return c.copy()
        return np.concatenate([c, np.zeros(self.n_slack)])

    def expand_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expand bounds to include slack variables.

        Slack variables are non-negative (>= 0).

        Args:
            lb: Original lower bounds (n,)
            ub: Original upper bounds (n,)

        Returns:
            (lb_expanded, ub_expanded): Expanded bounds
        """
        if self.n_slack == 0:
            return lb.copy(), ub.copy()

        # Original bounds + slack bounds (0 <= s <= inf)
        lb_expanded = np.concatenate([lb, np.zeros(self.n_slack)])
        ub_expanded = np.concatenate([ub, np.full(self.n_slack, np.inf)])

        return lb_expanded, ub_expanded

    def expand_integer_vars(self, integer_vars: List[int]) -> List[int]:
        """Expand integer variable list.

        Slack variables are continuous.

        Args:
            integer_vars: Original integer variable indices

        Returns:
            Integer variable indices in expanded space
        """
        # Slack variables are continuous, so only original variables remain integer
        return integer_vars

    def extract_original_solution(self, x_expanded: np.ndarray) -> np.ndarray:
        """Extract original solution from expanded solution.

        Args:
            x_expanded: Solution in expanded space

        Returns:
            Solution in original space
        """
        return x_expanded[:self.n]

    def extract_slack_values(self, x_expanded: np.ndarray) -> np.ndarray:
        """Extract slack variable values.

        Args:
            x_expanded: Solution in expanded space

        Returns:
            Slack variable values
        """
        if self.n_slack == 0:
            return np.array([])
        return x_expanded[self.n:]

    def compute_original_violation(
        self,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute constraint violation in original space.

        Args:
            x: Original solution vector

        Returns:
            (max_violation, violations_array)
        """
        Ax = self.A_orig @ x
        violations = np.zeros(self.m)

        # <= constraints
        violations[self.le_mask] = np.maximum(0, Ax[self.le_mask] - self.b_orig[self.le_mask])

        # >= constraints
        violations[self.ge_mask] = np.maximum(0, self.b_orig[self.ge_mask] - Ax[self.ge_mask])

        # = constraints
        violations[self.eq_mask] = np.abs(Ax[self.eq_mask] - self.b_orig[self.eq_mask])

        return np.max(violations), violations

    def verify_solution(
        self,
        x_expanded: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """Verify solution feasibility and extract information.

        Args:
            x_expanded: Solution in expanded space
            tol: Tolerance for feasibility

        Returns:
            Dictionary with verification results
        """
        x_orig = self.extract_original_solution(x_expanded)
        s_vals = self.extract_slack_values(x_expanded)

        # Check equality constraints
        Ax_eq = self.A_eq @ x_expanded
        eq_residual = np.linalg.norm(Ax_eq - self.b_eq)

        # Check original constraints
        max_viol_orig, violations_orig = self.compute_original_violation(x_orig)

        # Check slack non-negativity
        if len(s_vals) > 0:
            slack_viol = np.max(np.maximum(0, -s_vals))  # Should be 0 if s >= 0
        else:
            slack_viol = 0.0

        return {
            'x_orig': x_orig,
            's_vals': s_vals,
            'equality_residual': eq_residual,
            'max_original_violation': max_viol_orig,
            'violations_original': violations_orig,
            'slack_violation': slack_viol,
            'is_feasible': max_viol_orig <= tol and slack_viol <= tol,
        }


def transform_problem_for_equality_pdhg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    senses: List[ConstraintSense],
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: Optional[List[int]] = None,
) -> Dict:
    """Transform a mixed-constraint problem for equality PDHG.

    This is the main entry point for problem transformation.

    Args:
        A: Constraint matrix
        b: Right-hand side
        c: Objective coefficients
        senses: Constraint senses
        lb: Lower bounds
        ub: Upper bounds
        integer_vars: Integer variable indices

    Returns:
        Dictionary with transformed problem and transformer
    """
    transformer = EqualityTransformer(A, b, senses)

    A_eq, b_eq = transformer.get_equality_form()
    c_expanded = transformer.expand_objective(c)
    lb_expanded, ub_expanded = transformer.expand_bounds(lb, ub)
    integer_vars_expanded = transformer.expand_integer_vars(integer_vars or [])

    return {
        'A_eq': A_eq,
        'b_eq': b_eq,
        'c_expanded': c_expanded,
        'lb_expanded': lb_expanded,
        'ub_expanded': ub_expanded,
        'integer_vars': integer_vars_expanded,
        'transformer': transformer,
        'n_orig': transformer.n,
        'n_slack': transformer.n_slack,
    }
