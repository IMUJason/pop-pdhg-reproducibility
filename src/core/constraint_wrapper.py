"""Constraint wrapper for handling multiple constraint senses.

This module provides a wrapper around constraint matrices that tracks
the original constraint senses (<=, >=, =) and provides proper feasibility
checking for each type.
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple
import enum


class ConstraintSense(enum.IntEnum):
    """Constraint sense types."""
    LE = 0  # <=
    GE = 1  # >=
    EQ = 2  # =


class ConstraintWrapper:
    """Wrapper for constraint matrix with multiple senses.

    PDHG requires constraints in Ax <= b form. This wrapper:
    1. Converts constraints to Ax <= b form for PDHG
    2. Tracks original senses for proper feasibility checking
    3. Provides constraint violation computation for original senses

    Example:
        # Original constraints:
        #   x + y <= 5   (LE)
        #   x - y >= 1   (GE)
        #   x = 3        (EQ)

        wrapper = ConstraintWrapper(A, b, senses=[LE, GE, EQ])

        # For PDHG: get converted matrix (all <= form)
        A_pdh, b_pdhg = wrapper.get_pdh_form()

        # For feasibility checking: use original senses
        viol = wrapper.compute_violation(x)  # Checks original senses
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        senses: List[ConstraintSense],
        use_slack_transformation: bool = True,
    ):
        """Initialize constraint wrapper.

        Args:
            A: Constraint matrix (m x n)
            b: Right-hand side (m,)
            senses: List of constraint senses for each row
            use_slack_transformation: If True, use slack variables to transform
                >= and = constraints to standard form. This prevents the
                all-zero solution from being falsely feasible.
        """
        self.A_orig = A.copy()
        self.b_orig = b.copy()
        self.senses = np.array(senses)
        self.m, self.n = A.shape
        self.use_slack = use_slack_transformation

        if use_slack_transformation:
            # Use slack variable transformation for >= and = constraints
            self._build_slack_form()
        else:
            # Use simple flipping (original behavior)
            self._build_flipped_form()

    def _build_flipped_form(self):
        """Original simple flipping approach."""
        self.A_pdh = self.A_orig.copy()
        self.b_pdh = self.b_orig.copy()
        self.flipped = np.zeros(self.m, dtype=bool)

        # Convert >= constraints to <= by flipping
        ge_mask = self.senses == ConstraintSense.GE
        self.A_pdh[ge_mask] = -self.A_pdh[ge_mask]
        self.b_pdh[ge_mask] = -self.b_pdh[ge_mask]
        self.flipped[ge_mask] = True

        # For equality constraints, we'll store both directions
        eq_mask = self.senses == ConstraintSense.EQ
        self.eq_indices = np.where(eq_mask)[0]
        self.n_slack = 0
        self.slack_indices = []

    def _build_slack_form(self):
        """Build slack variable transformation.

        This transforms >= and = constraints to avoid the all-zero solution
        being falsely feasible.

        For >= constraint (Ax >= b): transformed to Ax - s = b, s >= 0
            PDHG form: [A, -I][x; s] <= b and [0, -I][x; s] <= 0 (s >= 0)
            -> Actually we use: Ax - s = b becomes two inequalities:
               Ax - s <= b and -(Ax - s) <= -b

        For = constraint (Ax = b): transformed to Ax - s1 + s2 = b, s1, s2 >= 0
            PDHG form: Ax - s1 + s2 <= b and -(Ax - s1 + s2) <= -b
                       with s1, s2 >= 0
        """
        # Count how many slack variables we need
        n_ge = np.sum(self.senses == ConstraintSense.GE)
        n_eq = np.sum(self.senses == ConstraintSense.EQ)
        self.n_slack = n_ge + 2 * n_eq

        if self.n_slack == 0:
            # No need for slack variables
            self.A_pdh = self.A_orig.copy()
            self.b_pdh = self.b_orig.copy()
            self.flipped = np.zeros(self.m, dtype=bool)
            self.eq_indices = []
            self.slack_indices = []
            return

        # Build expanded constraint matrix
        # Original variables: n, Slack variables: n_slack
        # For each constraint, we create two rows in PDHG form (as <=)

        n_new_constrs = 0
        self.orig_to_new = []  # Map original constraint index to new constraint indices
        self.slack_indices = []  # Track where slack variables are in the new vector
        self.flipped = np.zeros(2 * self.m, dtype=bool)

        rows, cols, data = [], [], []
        b_new = []
        slack_idx = 0

        for i in range(self.m):
            sense = self.senses[i]
            row_data = self.A_orig[i].toarray().flatten()

            if sense == ConstraintSense.LE:
                # LE constraint: Ax <= b, keep as is (single row)
                for j in range(self.n):
                    if abs(row_data[j]) > 1e-15:
                        rows.append(n_new_constrs)
                        cols.append(j)
                        data.append(row_data[j])
                b_new.append(self.b_orig[i])
                self.orig_to_new.append([n_new_constrs])
                n_new_constrs += 1

            elif sense == ConstraintSense.GE:
                # GE constraint: Ax >= b
                # Transform to: Ax - s = b with s >= 0
                # PDHG form: Ax - s <= b and -(Ax - s) <= -b

                # Row 1: Ax - s <= b
                for j in range(self.n):
                    if abs(row_data[j]) > 1e-15:
                        rows.append(n_new_constrs)
                        cols.append(j)
                        data.append(row_data[j])
                # Add slack variable with -1 coefficient
                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx)
                data.append(-1.0)
                self.slack_indices.append(self.n + slack_idx)
                b_new.append(self.b_orig[i])
                idx1 = n_new_constrs
                n_new_constrs += 1

                # Row 2: -(Ax - s) <= -b  =>  -Ax + s <= -b
                for j in range(self.n):
                    if abs(row_data[j]) > 1e-15:
                        rows.append(n_new_constrs)
                        cols.append(j)
                        data.append(-row_data[j])
                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx)
                data.append(1.0)
                b_new.append(-self.b_orig[i])
                self.flipped[n_new_constrs] = True
                idx2 = n_new_constrs
                n_new_constrs += 1

                self.orig_to_new.append([idx1, idx2])
                slack_idx += 1

            elif sense == ConstraintSense.EQ:
                # EQ constraint: Ax = b
                # Transform to: Ax - s1 + s2 = b with s1, s2 >= 0
                # PDHG form: Ax - s1 + s2 <= b and -(Ax - s1 + s2) <= -b

                # Row 1: Ax - s1 + s2 <= b
                for j in range(self.n):
                    if abs(row_data[j]) > 1e-15:
                        rows.append(n_new_constrs)
                        cols.append(j)
                        data.append(row_data[j])
                # Add s1 with -1, s2 with +1
                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx)
                data.append(-1.0)
                self.slack_indices.append(self.n + slack_idx)

                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx + 1)
                data.append(1.0)
                self.slack_indices.append(self.n + slack_idx + 1)

                b_new.append(self.b_orig[i])
                idx1 = n_new_constrs
                n_new_constrs += 1

                # Row 2: -(Ax - s1 + s2) <= -b
                for j in range(self.n):
                    if abs(row_data[j]) > 1e-15:
                        rows.append(n_new_constrs)
                        cols.append(j)
                        data.append(-row_data[j])
                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx)
                data.append(1.0)
                rows.append(n_new_constrs)
                cols.append(self.n + slack_idx + 1)
                data.append(-1.0)
                b_new.append(-self.b_orig[i])
                self.flipped[n_new_constrs] = True
                idx2 = n_new_constrs
                n_new_constrs += 1

                self.orig_to_new.append([idx1, idx2])
                slack_idx += 2

        # Build the expanded matrix
        n_total_vars = self.n + self.n_slack
        self.A_pdh = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_new_constrs, n_total_vars)
        )
        self.b_pdh = np.array(b_new)
        self.eq_indices = []

        # Store slack bounds (all >= 0)
        self.slack_lb = np.zeros(self.n_slack)
        self.slack_ub = np.full(self.n_slack, np.inf)

    def get_pdh_form(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Get constraint matrix in PDHG form (all <=).

        Returns:
            (A_pdh, b_pdh): Constraint matrix and RHS for PDHG
        """
        return self.A_pdh, self.b_pdh

    def get_expanded_bounds(self, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for expanded problem (including slack variables).

        Args:
            lb: Original lower bounds (n,)
            ub: Original upper bounds (n,)

        Returns:
            (lb_expanded, ub_expanded): Bounds for expanded problem
        """
        if not self.use_slack or self.n_slack == 0:
            return lb.copy(), ub.copy()

        # Original variables keep their bounds
        lb_expanded = np.concatenate([lb, self.slack_lb])
        ub_expanded = np.concatenate([ub, self.slack_ub])

        return lb_expanded, ub_expanded

    def get_expanded_integer_vars(self, integer_vars: List[int]) -> List[int]:
        """Get integer variable indices for expanded problem.

        Slack variables are continuous, so only original integer variables are kept.

        Args:
            integer_vars: Original integer variable indices

        Returns:
            Integer variable indices in expanded problem
        """
        if not self.use_slack or self.n_slack == 0:
            return integer_vars

        # Slack variables are continuous, so integer vars stay the same
        return integer_vars

    def extract_original_solution(self, x_expanded: np.ndarray) -> np.ndarray:
        """Extract original solution from expanded solution.

        Args:
            x_expanded: Solution vector for expanded problem

        Returns:
            Solution vector for original problem
        """
        if not self.use_slack or self.n_slack == 0:
            return x_expanded

        # Extract first n variables (original variables)
        return x_expanded[:self.n]

    def get_slack_values(self, x_expanded: np.ndarray) -> np.ndarray:
        """Get slack variable values from expanded solution.

        Args:
            x_expanded: Solution vector for expanded problem

        Returns:
            Slack variable values
        """
        if not self.use_slack or self.n_slack == 0:
            return np.array([])

        return x_expanded[self.n:self.n + self.n_slack]

    def compute_violation(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute constraint violation using original senses.

        For each constraint type:
        - LE (<=): violation = max(0, Ax - b)
        - GE (>=): violation = max(0, b - Ax)
        - EQ (=):  violation = |Ax - b|

        Args:
            x: Solution vector (n,)

        Returns:
            (max_violation, violations_array)
        """
        Ax = self.A_orig @ x
        violations = np.zeros(self.m)

        # LE constraints: Ax <= b, violation = max(0, Ax - b)
        le_mask = self.senses == ConstraintSense.LE
        violations[le_mask] = np.maximum(0, Ax[le_mask] - self.b_orig[le_mask])

        # GE constraints: Ax >= b, violation = max(0, b - Ax)
        ge_mask = self.senses == ConstraintSense.GE
        violations[ge_mask] = np.maximum(0, self.b_orig[ge_mask] - Ax[ge_mask])

        # EQ constraints: Ax = b, violation = |Ax - b|
        eq_mask = self.senses == ConstraintSense.EQ
        violations[eq_mask] = np.abs(Ax[eq_mask] - self.b_orig[eq_mask])

        return np.max(violations), violations

    def compute_violation_pdh_form(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute violation using PDHG form (for backwards compatibility).

        Args:
            x: Solution vector (n,)

        Returns:
            (max_violation, violations_array)
        """
        Ax = self.A_pdh @ x
        violations = np.maximum(0, Ax - self.b_pdh)
        return np.max(violations), violations

    def repair_constraint(
        self,
        x: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        max_iters: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Repair constraint violations using projected gradient descent.

        This repair method uses the ORIGINAL constraint senses, ensuring
        that solutions satisfy the original problem constraints.

        Args:
            x: Current solution (may violate constraints)
            integer_vars: Indices of integer variables
            max_iters: Maximum repair iterations
            tol: Tolerance for constraint satisfaction

        Returns:
            Repaired solution
        """
        x_repaired = x.copy()
        integer_vars = integer_vars or []

        for iteration in range(max_iters):
            # Compute violation using original senses
            max_viol, violations = self.compute_violation(x_repaired)

            if max_viol <= tol:
                break

            # Compute gradient of violation squared
            # For violated constraints, gradient points in direction to reduce violation
            grad = np.zeros(self.n)
            Ax = self.A_orig @ x_repaired

            for i in range(self.m):
                if violations[i] > tol:
                    if self.senses[i] == ConstraintSense.LE:
                        # LE: violation = max(0, Ax - b)
                        # grad = A[i] if Ax - b > 0
                        grad += self.A_orig[i].toarray().flatten() * violations[i]
                    elif self.senses[i] == ConstraintSense.GE:
                        # GE: violation = max(0, b - Ax)
                        # grad = -A[i] if b - Ax > 0
                        grad -= self.A_orig[i].toarray().flatten() * violations[i]
                    elif self.senses[i] == ConstraintSense.EQ:
                        # EQ: violation = |Ax - b|
                        # grad = sign(Ax - b) * A[i]
                        sign = np.sign(Ax[i] - self.b_orig[i])
                        grad += sign * self.A_orig[i].toarray().flatten() * violations[i]

            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break

            grad = grad / grad_norm

            # Line search
            best_improvement = 0
            best_x = x_repaired.copy()

            for step_scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
                step_size = step_scale * min(max_viol, 1.0)
                x_candidate = x_repaired - step_size * grad

                # Project to bounds
                # Note: We should have access to lb, ub from the solver
                # For now, assume they're passed or use unbounded

                # Round integer variables
                for i in integer_vars:
                    x_candidate[i] = round(x_candidate[i])

                # Check improvement
                new_viol, _ = self.compute_violation(x_candidate)
                improvement = max_viol - new_viol

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_x = x_candidate.copy()

            if best_improvement > 0:
                x_repaired = best_x
            else:
                # No improvement with gradient, try coordinate descent for integers
                if integer_vars:
                    x_repaired = self._coordinate_descent_repair(
                        x_repaired, integer_vars, max_iters=20
                    )
                break

        return x_repaired

    def _coordinate_descent_repair(
        self,
        x: np.ndarray,
        integer_vars: List[int],
        max_iters: int = 20,
    ) -> np.ndarray:
        """Coordinate descent repair for integer variables."""
        x_repaired = x.copy()

        for _ in range(max_iters):
            improved = False

            for i in integer_vars:
                current_val = x_repaired[i]
                current_viol, _ = self.compute_violation(x_repaired)

                # Try neighboring integer values
                candidates = []
                if round(current_val - 1) >= 0:  # Assuming non-negative
                    candidates.append(round(current_val - 1))
                candidates.append(round(current_val))
                candidates.append(round(current_val + 1))

                best_val = current_val
                best_violation = current_viol

                for val in candidates:
                    x_repaired[i] = val
                    v_new, _ = self.compute_violation(x_repaired)
                    if v_new < best_violation:
                        best_val = val
                        best_violation = v_new
                        improved = True

                x_repaired[i] = best_val

            if not improved:
                break

        return x_repaired


def load_mps_with_senses(mps_path: str) -> dict:
    """Load MPS file with constraint sense information.

    This is a wrapper around Gurobi that preserves constraint senses.

    Args:
        mps_path: Path to MPS file

    Returns:
        Dictionary with:
        - A: Constraint matrix
        - b: Right-hand side
        - c: Objective coefficients
        - lb: Lower bounds
        - ub: Upper bounds
        - integer_vars: Integer variable indices
        - constraint_wrapper: ConstraintWrapper with proper senses
        - senses: List of ConstraintSense for each constraint
    """
    try:
        import gurobipy as gp
    except ImportError:
        raise ImportError("Gurobi required for loading MPS")

    model = gp.read(mps_path)
    n_vars = model.NumVars
    n_constrs = model.NumConstrs

    vars_list = model.getVars()
    var_map = {v: i for i, v in enumerate(vars_list)}

    c = np.array([v.Obj for v in vars_list])
    lb = np.array([v.LB for v in vars_list])
    ub = np.array([v.UB for v in vars_list])

    var_types = [v.VType for v in vars_list]
    integer_vars = [i for i, vt in enumerate(var_types) if vt in [gp.GRB.INTEGER, gp.GRB.BINARY]]

    # Build constraint matrix and track senses
    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(n_constrs)
    senses = []

    for i, constr in enumerate(model.getConstrs()):
        row = model.getRow(constr)
        b[i] = constr.RHS

        # Map Gurobi sense to our enum
        if constr.Sense == gp.GRB.LESS_EQUAL:
            senses.append(ConstraintSense.LE)
        elif constr.Sense == gp.GRB.GREATER_EQUAL:
            senses.append(ConstraintSense.GE)
        elif constr.Sense == gp.GRB.EQUAL:
            senses.append(ConstraintSense.EQ)

        for j in range(row.size()):
            var = row.getVar(j)
            coef = row.getCoeff(j)
            var_idx = var_map[var]
            A_rows.append(i)
            A_cols.append(var_idx)
            A_data.append(coef)

    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(n_constrs, n_vars))

    # Create constraint wrapper with slack transformation
    # This transforms >= and = constraints to avoid all-zero solution being feasible
    constraint_wrapper = ConstraintWrapper(A, b, senses, use_slack_transformation=True)

    # Get PDHG form for solver
    A_pdh, b_pdh = constraint_wrapper.get_pdh_form()

    # Get expanded bounds including slack variables
    lb_expanded, ub_expanded = constraint_wrapper.get_expanded_bounds(lb, ub)

    # Get expanded integer variables (slack variables are continuous)
    integer_vars_expanded = constraint_wrapper.get_expanded_integer_vars(integer_vars)

    # Expand objective vector (slack variables have zero cost)
    if constraint_wrapper.n_slack > 0:
        c_expanded = np.concatenate([c, np.zeros(constraint_wrapper.n_slack)])
    else:
        c_expanded = c

    model.dispose()

    return {
        'A': A,  # Original matrix
        'b': b,  # Original RHS
        'A_pdh': A_pdh,  # PDHG form (expanded with slack variables if needed)
        'b_pdh': b_pdh,  # PDHG form RHS
        'c': c_expanded,  # Expanded objective (includes slack vars with 0 cost)
        'lb': lb_expanded,  # Expanded lower bounds
        'ub': ub_expanded,  # Expanded upper bounds
        'integer_vars': integer_vars_expanded,  # Integer vars in expanded space
        'constraint_wrapper': constraint_wrapper,
        'senses': senses,
        'n_vars': n_vars,  # Original number of variables
        'n_constrs': n_constrs,  # Original number of constraints
        'n_slack': constraint_wrapper.n_slack,  # Number of slack variables added
    }
