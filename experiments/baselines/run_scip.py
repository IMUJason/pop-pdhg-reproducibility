"""
SCIP baseline solver script.

Provides functions to solve LP/MIP problems using SCIP (via PySCIPOpt).
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def solve_with_scip(
    mps_file: str,
    time_limit: float = 3600.0,
    lp_only: bool = False,
    output_flag: bool = False,
    threads: Optional[int] = None,
    mip_gap: float = 1e-4,
) -> dict:
    """Solve an MPS file using SCIP.

    Args:
        mps_file: Path to the MPS file
        time_limit: Time limit in seconds
        lp_only: If True, solve only LP relaxation
        output_flag: Whether to print SCIP output
        threads: Number of threads (None for auto)
        mip_gap: MIP optimality gap tolerance

    Returns:
        Dictionary with solution information:
            - obj: Objective value
            - time: Solve time in seconds
            - nodes: Number of B&B nodes (for MIP)
            - gap: MIP gap
            - lp_relaxation: LP relaxation value
            - status: Solver status string
            - x: Solution vector (if not too large)
    """
    from pyscipopt import Model, SCIP_PARAMSETTING

    # Read model
    model = Model()
    model.readProblem(mps_file)

    if not output_flag:
        model.hideOutput()

    # Set parameters
    model.setRealParam("limits/time", time_limit)
    model.setRealParam("limits/gap", mip_gap)

    if threads is not None:
        model.setIntParam("lp/threads", threads)

    # Get LP relaxation value first
    lp_relaxation = None
    if not lp_only:
        # Get LP relaxation
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.optimize()
        lp_relaxation = model.getObjVal()

        # Reset for MIP solve
        model.freeProb()
        model.readProblem(mps_file)
        if not output_flag:
            model.hideOutput()
        model.setRealParam("limits/time", time_limit)
        model.setRealParam("limits/gap", mip_gap)

    # Relax to LP if needed
    if lp_only:
        # Change all integer variables to continuous
        for var in model.getVars():
            if var.vtype() in ["INTEGER", "BINARY"]:
                model.chgVarType(var, "CONTINUOUS")

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    # Extract results
    status = model.getStatus()

    result = {
        "obj": None,
        "time": solve_time,
        "nodes": model.getNNodes(),
        "gap": model.getGap(),
        "lp_relaxation": lp_relaxation,
        "status": status.lower(),
        "x": None,
    }

    if model.getNSols() > 0:
        result["obj"] = model.getObjVal()

        # Save solution if not too large
        vars_list = model.getVars()
        if len(vars_list) <= 10000:
            result["x"] = [model.getVal(v) for v in vars_list]

    return result


def solve_lp_with_scip(
    A,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    output_flag: bool = False,
) -> dict:
    """Solve LP using SCIP with matrix input.

    Note: PySCIPOpt doesn't have direct matrix input, so we add
    variables and constraints one by one.

    Args:
        A: Constraint matrix (can be sparse)
        b: Right-hand side
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        output_flag: Whether to print output

    Returns:
        Dictionary with solution information
    """
    from pyscipopt import Model

    m, n = A.shape

    model = Model()
    if not output_flag:
        model.hideOutput()

    # Add variables
    x_vars = []
    for j in range(n):
        v = model.addVar(lb=lb[j], ub=ub[j], obj=c[j], name=f"x_{j}")
        x_vars.append(v)

    # Add constraints: A @ x <= b
    for i in range(m):
        # Get non-zero entries of row i
        if hasattr(A, "getrow"):
            # Sparse matrix
            row = A.getrow(i)
            coeffs = {x_vars[j]: val for j, val in zip(row.indices, row.data)}
        else:
            # Dense matrix
            coeffs = {x_vars[j]: A[i, j] for j in range(n) if A[i, j] != 0}

        model.addCons(sum(c * v for v, c in coeffs.items()) <= b[i], name=f"c_{i}")

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    status = model.getStatus()

    result = {
        "obj": None,
        "x": None,
        "time": solve_time,
        "status": status.lower(),
    }

    if model.getNSols() > 0:
        result["obj"] = model.getObjVal()
        result["x"] = [model.getVal(v) for v in x_vars]

    return result


def benchmark_scip(mps_files: list[str], output_dir: str, **kwargs) -> None:
    """Run SCIP on multiple MPS files and save results.

    Args:
        mps_files: List of MPS file paths
        output_dir: Directory to save results
        **kwargs: Additional arguments for solve_with_scip
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mps_file in mps_files:
        name = Path(mps_file).stem
        print(f"Solving {name}...")

        try:
            result = solve_with_scip(mps_file, **kwargs)
            result["file"] = mps_file
            results[name] = result

            print(
                f"  Status: {result['status']}, "
                f"Obj: {result['obj']}, "
                f"Time: {result['time']:.2f}s"
            )
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"status": "error", "error": str(e), "file": mps_file}

    # Save results
    output_file = output_dir / "scip_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mps_file = sys.argv[1]
        result = solve_with_scip(mps_file, output_flag=True)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    else:
        print("Usage: python run_scip.py <mps_file>")
