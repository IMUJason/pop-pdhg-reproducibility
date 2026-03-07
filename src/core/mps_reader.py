"""
MPS file reader for LP/MIP problems.

This module provides functionality to read MPS files and convert them
to standard form LP data structures suitable for PDHG solver.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class LPData:
    """Data structure for LP/MIP problems in standard form.

    Standard form: min c^T x, s.t. Ax <= b, x >= 0

    Attributes:
        A: Constraint matrix (scipy sparse CSR format)
        b: Right-hand side vector
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        var_types: Variable types ('C'=continuous, 'I'=integer, 'B'=binary)
        sense: Constraint senses ('<', '=', '>')
        m: Number of constraints
        n: Number of variables
        nnz: Number of non-zero elements
        obj_sense: Objective sense ('min' or 'max')
        integer_vars: Indices of integer variables
        binary_vars: Indices of binary variables
    """

    A: sparse.csr_matrix
    b: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    var_types: list[str]
    sense: list[str]
    m: int
    n: int
    nnz: int
    obj_sense: str = "min"
    integer_vars: list[int] = None
    binary_vars: list[int] = None

    def __post_init__(self):
        if self.integer_vars is None:
            self.integer_vars = []
        if self.binary_vars is None:
            self.binary_vars = []


def read_mps(filepath: str, use_gurobi: bool = True) -> LPData:
    """Read an MPS file and return LP data structure.

    Args:
        filepath: Path to the MPS file
        use_gurobi: If True, use Gurobi to read the file (more reliable).
                   If False, use scipy.io (limited support).

    Returns:
        LPData object containing the problem data

    Raises:
        FileNotFoundError: If the MPS file doesn't exist
        ValueError: If the file format is not supported
    """
    if use_gurobi:
        return _read_mps_gurobi(filepath)
    else:
        return _read_mps_scipy(filepath)


def _read_mps_gurobi(filepath: str) -> LPData:
    """Read MPS file using Gurobi (most reliable method)."""
    import gurobipy as gp

    model = gp.read(filepath)
    model.update()

    # Get constraint matrix
    A = model.getA()
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    else:
        A = A.tocsr()

    m, n = A.shape

    # Get RHS
    constrs = model.getConstrs()
    b = np.array([c.RHS for c in constrs])

    # Get constraint senses
    sense_map = {"<": "<", ">": ">", "=": "="}
    sense = [sense_map.get(c.Sense, "<") for c in constrs]

    # Get objective coefficients
    vars_list = model.getVars()
    c = np.array([v.Obj for v in vars_list])

    # Get variable bounds
    lb = np.array([v.LB for v in vars_list])
    ub = np.array([v.UB for v in vars_list])

    # Get variable types
    var_types = []
    integer_vars = []
    binary_vars = []

    for i, v in enumerate(vars_list):
        if v.VType == "B":
            var_types.append("B")
            binary_vars.append(i)
            integer_vars.append(i)
        elif v.VType == "I":
            var_types.append("I")
            integer_vars.append(i)
        else:
            var_types.append("C")

    # Get objective sense
    obj_sense = "min" if model.ModelSense == 1 else "max"

    return LPData(
        A=A,
        b=b,
        c=c,
        lb=lb,
        ub=ub,
        var_types=var_types,
        sense=sense,
        m=m,
        n=n,
        nnz=A.nnz,
        obj_sense=obj_sense,
        integer_vars=integer_vars,
        binary_vars=binary_vars,
    )


def _read_mps_scipy(filepath: str) -> LPData:
    """Read MPS file using scipy.io (limited support)."""
    from scipy.io import readsas

    # Note: scipy.io does not have native MPS reader
    # This is a placeholder - use Gurobi for reliable MPS reading
    raise NotImplementedError(
        "scipy.io does not support MPS files natively. "
        "Please use Gurobi (use_gurobi=True) or install pyscipopt."
    )


def to_standard_form(lp: LPData) -> LPData:
    """Convert LP to standard form: min c^T x, s.t. Ax <= b, x >= 0

    Transformations:
    - Convert >= constraints to <= by negating
    - Split = constraints into two <= constraints
    - Handle maximization by negating objective
    - Handle variable bounds by adjusting constraints

    Args:
        lp: Original LP data

    Returns:
        LP data in standard form
    """
    A_rows = []
    b_values = []
    sense_new = []

    for i, s in enumerate(lp.sense):
        row = lp.A.getrow(i)
        if s == "<":
            A_rows.append(row)
            b_values.append(lp.b[i])
            sense_new.append("<")
        elif s == ">":
            # Negate: Ax >= b => -Ax <= -b
            A_rows.append(-row)
            b_values.append(-lp.b[i])
            sense_new.append("<")
        elif s == "=":
            # Split: Ax = b => Ax <= b and -Ax <= -b
            A_rows.append(row)
            b_values.append(lp.b[i])
            sense_new.append("<")
            A_rows.append(-row)
            b_values.append(-lp.b[i])
            sense_new.append("<")

    A_new = sparse.vstack(A_rows, format="csr")
    b_new = np.array(b_values)

    # Handle maximization
    if lp.obj_sense == "max":
        c_new = -lp.c
    else:
        c_new = lp.c.copy()

    # Handle variable bounds (lb != 0)
    # For simplicity, we assume lb = 0 for standard form
    # If lb != 0, we need variable substitution
    # For now, just adjust ub - lb and b - A @ lb
    if not np.allclose(lp.lb, 0):
        # Shift: x' = x - lb, so x' >= 0
        # New constraint: A @ (x' + lb) <= b => A @ x' <= b - A @ lb
        b_adjusted = b_new - A_new @ lp.lb
        ub_adjusted = lp.ub - lp.lb
        lb_adjusted = np.zeros(lp.n)
    else:
        b_adjusted = b_new
        lb_adjusted = lp.lb.copy()
        ub_adjusted = lp.ub.copy()

    return LPData(
        A=A_new,
        b=b_adjusted,
        c=c_new,
        lb=lb_adjusted,
        ub=ub_adjusted,
        var_types=lp.var_types.copy(),
        sense=sense_new,
        m=A_new.shape[0],
        n=lp.n,
        nnz=A_new.nnz,
        obj_sense="min",  # Always minimize in standard form
        integer_vars=lp.integer_vars.copy() if lp.integer_vars else [],
        binary_vars=lp.binary_vars.copy() if lp.binary_vars else [],
    )


def get_stats(lp: LPData) -> dict:
    """Get statistics about the LP problem.

    Args:
        lp: LP data structure

    Returns:
        Dictionary with problem statistics
    """
    density = lp.nnz / (lp.m * lp.n) if lp.m * lp.n > 0 else 0

    return {
        "constraints": lp.m,
        "variables": lp.n,
        "nonzeros": lp.nnz,
        "density": f"{density:.6f}",
        "integer_vars": len(lp.integer_vars),
        "binary_vars": len(lp.binary_vars),
        "continuous_vars": lp.n - len(lp.integer_vars),
        "obj_sense": lp.obj_sense,
        "constraint_types": {
            "leq": lp.sense.count("<"),
            "geq": lp.sense.count(">"),
            "eq": lp.sense.count("="),
        },
    }


def print_stats(lp: LPData) -> None:
    """Print problem statistics in a formatted way."""
    stats = get_stats(lp)
    print("=" * 50)
    print("LP/MIP Problem Statistics")
    print("=" * 50)
    print(f"Constraints: {stats['constraints']}")
    print(f"Variables:   {stats['variables']}")
    print(f"Non-zeros:   {stats['nonzeros']}")
    print(f"Density:     {stats['density']}")
    print(f"Integer vars: {stats['integer_vars']}")
    print(f"Binary vars:  {stats['binary_vars']}")
    print(f"Continuous:   {stats['continuous_vars']}")
    print(f"Objective:    {stats['obj_sense']}")
    print(f"Constraint types:")
    print(f"  <= : {stats['constraint_types']['leq']}")
    print(f"  >= : {stats['constraint_types']['geq']}")
    print(f"  =  : {stats['constraint_types']['eq']}")
    print("=" * 50)


if __name__ == "__main__":
    # Test with a simple example
    import os
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            lp = read_mps(filepath)
            print_stats(lp)
        else:
            print(f"File not found: {filepath}")
    else:
        print("Usage: python mps_reader.py <mps_file>")
