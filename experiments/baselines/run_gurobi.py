"""
Gurobi baseline solver script.

Provides functions to solve LP/MIP problems using Gurobi and compare results.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def solve_with_gurobi(
    mps_file: str,
    time_limit: float = 3600.0,
    lp_only: bool = False,
    output_flag: int = 0,
    threads: Optional[int] = None,
    mip_gap: float = 1e-4,
) -> dict:
    """Solve an MPS file using Gurobi.

    Args:
        mps_file: Path to the MPS file
        time_limit: Time limit in seconds
        lp_only: If True, solve only LP relaxation
        output_flag: Gurobi output flag (0=silent, 1=verbose)
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
    import gurobipy as gp

    # Read model
    model = gp.read(mps_file)
    model.setParam("OutputFlag", output_flag)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)

    if threads is not None:
        model.setParam("Threads", threads)

    # Get LP relaxation value first
    lp_relaxation = None
    if not lp_only:
        # Save current variable types
        vars_list = model.getVars()
        var_types = [v.VType for v in vars_list]

        # Relax to LP
        for v in vars_list:
            v.VType = "C"

        model.update()
        model.setParam("Method", 2)  # Use barrier for LP
        model.optimize()

        if model.Status == gp.GRB.Status.OPTIMAL:
            lp_relaxation = model.ObjVal

        # Restore variable types
        for v, vt in zip(vars_list, var_types):
            v.VType = vt
        model.update()

    # Solve
    if lp_only:
        model.setParam("Method", 2)  # Use barrier for LP
        # Relax all integers
        for v in model.getVars():
            v.VType = "C"
        model.update()

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    # Extract results
    status_map = {
        gp.GRB.Status.OPTIMAL: "optimal",
        gp.GRB.Status.INFEASIBLE: "infeasible",
        gp.GRB.Status.INF_OR_UNBD: "inf_or_unbd",
        gp.GRB.Status.UNBOUNDED: "unbounded",
        gp.GRB.Status.TIME_LIMIT: "time_limit",
        gp.GRB.Status.SOLUTION_LIMIT: "solution_limit",
        gp.GRB.Status.INTERRUPTED: "interrupted",
        gp.GRB.Status.ITERATION_LIMIT: "iteration_limit",
    }

    status = status_map.get(model.Status, f"unknown({model.Status})")

    result = {
        "obj": None,
        "time": solve_time,
        "nodes": 0,
        "gap": None,
        "lp_relaxation": lp_relaxation,
        "status": status,
        "x": None,
    }

    if model.Status == gp.GRB.Status.OPTIMAL or model.SolCount > 0:
        result["obj"] = model.ObjVal

        if not lp_only:
            result["nodes"] = model.NodeCount
            result["gap"] = model.MIPGap

        # Save solution if not too large
        vars_list = model.getVars()
        if len(vars_list) <= 10000:
            result["x"] = [v.X for v in vars_list]

    return result


def solve_lp_with_gurobi(
    A,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    output_flag: int = 0,
) -> dict:
    """Solve LP using Gurobi with matrix input.

    Args:
        A: Constraint matrix (can be sparse)
        b: Right-hand side
        c: Objective coefficients
        lb: Variable lower bounds
        ub: Variable upper bounds
        output_flag: Gurobi output flag

    Returns:
        Dictionary with solution information
    """
    import gurobipy as gp

    m, n = A.shape

    model = gp.Model()
    model.setParam("OutputFlag", output_flag)
    model.setParam("Method", 2)  # Barrier

    # Add variables
    x = model.addMVar(n, lb=lb, ub=ub, obj=c)

    # Add constraints: Ax <= b
    model.addMConstr(A, x, "<", b, name="constr")

    model.update()

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    status_map = {
        gp.GRB.Status.OPTIMAL: "optimal",
        gp.GRB.Status.INFEASIBLE: "infeasible",
        gp.GRB.Status.UNBOUNDED: "unbounded",
    }

    result = {
        "obj": None,
        "x": None,
        "time": solve_time,
        "status": status_map.get(model.Status, f"unknown({model.Status})"),
    }

    if model.Status == gp.GRB.Status.OPTIMAL:
        result["obj"] = model.ObjVal
        result["x"] = x.X.tolist()

    return result


def benchmark_gurobi(mps_files: list[str], output_dir: str, **kwargs) -> None:
    """Run Gurobi on multiple MPS files and save results.

    Args:
        mps_files: List of MPS file paths
        output_dir: Directory to save results
        **kwargs: Additional arguments for solve_with_gurobi
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mps_file in mps_files:
        name = Path(mps_file).stem
        print(f"Solving {name}...")

        try:
            result = solve_with_gurobi(mps_file, **kwargs)
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
    output_file = output_dir / "gurobi_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mps_file = sys.argv[1]
        result = solve_with_gurobi(mps_file, output_flag=1)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    else:
        print("Usage: python run_gurobi.py <mps_file>")
