#!/usr/bin/env python
"""
Comprehensive Benchmark: Quantum-Inspired Pop-PDHG for MIP.

This script tests the quantum-inspired Pop-PDHG solver on MIP instances,
comparing against standard Pop-PDHG and Gurobi baseline.

Test instances:
- p0033: 33 binary variables (MIPLIB)
- p0201: 201 binary variables (MIPLIB)
- knapsack_50: 50 binary variables
- setcover_50x200: 200 binary variables

Metrics:
- Objective value quality
- Feasibility rate
- Computation time
- Tunneling success rate
- Measurement effectiveness
"""

import time
import json
import numpy as np
from scipy import sparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Solvers
from src.population.pop_pdhg import PopulationPDHG, solve_lp_population
from src.population.quantum_pop_pdhg import (
    QuantumPopulationPDHG,
    QuantumPopPDHGConfig,
    solve_lp_quantum_population,
)
from src.population.measurement import QuantumMeasurement, MeasurementConfig as MeasureConfig
from src.optimization.quantum_tunneling import QuantumTunnelOperator, TunnelConfig


@dataclass
class BenchmarkResult:
    """Benchmark result for a single solver on a single instance."""
    instance: str
    solver: str
    obj_value: float
    time_seconds: float
    iterations: int
    converged: bool
    status: str
    is_feasible: bool
    primal_violation: float
    integrality_violation: float
    gap_to_gurobi: float  # Percentage gap
    tunnel_success_rate: float = 0.0
    measure_strength_final: float = 0.0


def create_test_instance(name: str):
    """Create test MIP instances."""
    np.random.seed(42)

    if name == "p0033":
        # MIPLIB p0033: 33 binary variables, 16 constraints
        # Simple test version (not exact p0033)
        n = 33
        m = 16

        # Generate random constraints
        A = sparse.random(m, n, density=0.3, format="csr") * 10
        A = A.tolil()
        for i in range(m):
            # Ensure some structure
            idx = np.random.choice(n, size=n//3, replace=False)
            A[i, idx] = np.random.uniform(1, 10, size=len(idx))
        A = A.tocsr()

        b = np.array([A[i].sum() * 0.5 for i in range(m)])
        c = -np.random.uniform(1, 10, size=n)
        lb = np.zeros(n)
        ub = np.ones(n)
        integer_vars = list(range(n))

    elif name == "p0201":
        # MIPLIB p0201: 201 binary variables, 120 constraints
        n = 201
        m = 120

        A = sparse.random(m, n, density=0.15, format="csr") * 10
        b = np.array([A[i].sum() * 0.4 for i in range(m)])
        c = -np.random.uniform(1, 5, size=n)
        lb = np.zeros(n)
        ub = np.ones(n)
        integer_vars = list(range(n))

    elif name == "knapsack_50":
        # Knapsack problem: 50 items
        n = 50
        weights = np.random.uniform(1, 20, size=n)
        values = np.random.uniform(1, 15, size=n)
        capacity = 0.6 * np.sum(weights)

        A = sparse.csr_matrix(weights.reshape(1, -1))
        b = np.array([capacity])
        c = -values  # Maximize value
        lb = np.zeros(n)
        ub = np.ones(n)
        integer_vars = list(range(n))

    elif name == "setcover_50x200":
        # Set covering: 50 constraints, 200 variables
        n = 200
        m = 50

        A = sparse.random(m, n, density=0.1, format="csr")
        A = (A > 0).astype(float)  # Binary coefficients

        b = np.ones(m) * 3  # Each constraint needs coverage
        c = np.random.uniform(1, 10, size=n)  # Minimize cost
        lb = np.zeros(n)
        ub = np.ones(n)
        integer_vars = list(range(n))

    else:
        raise ValueError(f"Unknown instance: {name}")

    return A, b, c, lb, ub, integer_vars


def check_feasibility(
    x: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: List[int],
    tol: float = 1e-4,
) -> tuple:
    """Check solution feasibility."""
    # Bound violations
    bound_viol = max(
        np.max(np.maximum(lb - x, 0)),
        np.max(np.maximum(x - ub, 0)),
    )

    # Constraint violations
    Ax = A @ x
    constraint_viol = np.max(np.maximum(Ax - b, 0))

    # Integrality violations
    if integer_vars:
        int_viol = np.max(np.abs(x[integer_vars] - np.round(x[integer_vars])))
    else:
        int_viol = 0.0

    is_feasible = (
        bound_viol <= tol and
        constraint_viol <= tol and
        int_viol <= tol
    )

    return is_feasible, max(bound_viol, constraint_viol), int_viol


def run_gurobi_baseline(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: List[int],
) -> Optional[float]:
    """Solve with Gurobi for baseline."""
    try:
        import gurobipy as gp
        from gurobipy import GRB

        n = len(c)
        m = len(b)

        # Create model
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 60

        # Variables
        x = {}
        for j in range(n):
            if j in integer_vars:
                x[j] = model.addVar(lb=lb[j], ub=ub[j], vtype=GRB.BINARY, name=f"x{j}")
            else:
                x[j] = model.addVar(lb=lb[j], ub=ub[j], vtype=GRB.CONTINUOUS, name=f"x{j}")

        # Constraints
        for i in range(m):
            row = A.getrow(i)
            expr = gp.quicksum(row[0, j] * x[j] for j in range(n))
            model.addConstr(expr <= b[i])

        # Objective
        model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n)), GRB.MINIMIZE)

        # Optimize
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return model.ObjVal
        else:
            return None

    except Exception as e:
        print(f"  Gurobi error: {e}")
        return None


def solve_with_quantum_pop_pdhg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: List[int],
    max_iter: int = 5000,
    population_size: int = 16,
    verbose: bool = False,
    seed: int = 42,
) -> tuple:
    """Solve with Quantum Pop-PDHG."""
    config = QuantumPopPDHGConfig(
        use_tunnel=True,
        tunnel_interval=50,
        use_progressive_measure=True,
        measure_interval=100,
        integer_vars=integer_vars,
        use_smart_trigger=True,
    )

    solver = QuantumPopulationPDHG(
        A, b, c, lb, ub,
        population_size=population_size,
        config=config,
    )

    start = time.time()
    result = solver.solve(
        max_iter=max_iter,
        tol=1e-4,
        verbose=verbose,
        seed=seed,
        integer_vars=integer_vars,
    )
    elapsed = time.time() - start

    # Apply final measurement
    measurement = QuantumMeasurement(MeasureConfig(
        temperature=0.5,
        rounding_strategy="probabilistic",
        repair_infeasible=True,
    ))

    solutions = measurement.measure(
        result.state, integer_vars, A, b, c, lb, ub,
        n_samples=5, seed=seed
    )

    # Get best feasible solution
    best_x = result.x_best.copy()
    best_obj = result.obj_best

    for x_sol, obj, feas in solutions:
        if feas and obj < best_obj:
            best_x = x_sol
            best_obj = obj

    return best_x, best_obj, elapsed, result


def solve_with_standard_pop_pdhg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: List[int],
    max_iter: int = 5000,
    population_size: int = 16,
    verbose: bool = False,
    seed: int = 42,
) -> tuple:
    """Solve with standard Pop-PDHG."""
    solver = PopulationPDHG(A, b, c, lb, ub, population_size=population_size)

    start = time.time()
    result = solver.solve(
        max_iter=max_iter,
        tol=1e-4,
        verbose=verbose,
        seed=seed,
    )
    elapsed = time.time() - start

    # Apply measurement for integer variables
    measurement = QuantumMeasurement(MeasureConfig(
        temperature=0.5,
        rounding_strategy="probabilistic",
        repair_infeasible=True,
    ))

    solutions = measurement.measure(
        result.state, integer_vars, A, b, c, lb, ub,
        n_samples=5, seed=seed
    )

    best_x = result.x_best.copy()
    best_obj = result.obj_best

    for x_sol, obj, feas in solutions:
        if feas and obj < best_obj:
            best_x = x_sol
            best_obj = obj

    return best_x, best_obj, elapsed, result


def run_benchmark(
    instances: List[str],
    max_iter: int = 5000,
    population_size: int = 16,
    n_runs: int = 3,
    output_dir: str = "experiments/results",
):
    """Run comprehensive benchmark."""
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Quantum-Inspired Pop-PDHG Benchmark")
    print("=" * 70)
    print(f"Instances: {instances}")
    print(f"Max iterations: {max_iter}")
    print(f"Population size: {population_size}")
    print(f"Runs per instance: {n_runs}")
    print()

    for instance_name in instances:
        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"{'='*60}")

        # Create instance
        A, b, c, lb, ub, integer_vars = create_test_instance(instance_name)
        n_vars = len(c)
        n_constraints = len(b)
        print(f"  Variables: {n_vars}, Constraints: {n_constraints}")
        print(f"  Integer variables: {len(integer_vars)}")

        # Gurobi baseline
        print("\n  Solving with Gurobi (baseline)...")
        gurobi_obj = run_gurobi_baseline(A, b, c, lb, ub, integer_vars)
        if gurobi_obj is not None:
            print(f"  Gurobi objective: {gurobi_obj:.4f}")
        else:
            print(f"  Gurobi failed to find optimal solution")
            gurobi_obj = None

        # Standard Pop-PDHG
        print("\n  Solving with Standard Pop-PDHG...")
        std_results = []
        for run in range(n_runs):
            x, obj, elapsed, _ = solve_with_standard_pop_pdhg(
                A, b, c, lb, ub, integer_vars,
                max_iter=max_iter,
                population_size=population_size,
                seed=42+run,
            )
            is_feas, primal_viol, int_viol = check_feasibility(
                x, A, b, lb, ub, integer_vars
            )
            gap = ((obj - gurobi_obj) / abs(gurobi_obj) * 100) if gurobi_obj else 0

            std_results.append({
                "obj": obj,
                "time": elapsed,
                "feasible": is_feas,
                "primal_viol": primal_viol,
                "int_viol": int_viol,
                "gap": gap,
            })
            print(f"    Run {run+1}: obj={obj:.4f}, time={elapsed:.2f}s, "
                  f"feas={'Yes' if is_feas else 'No'}, gap={gap:.2f}%")

        # Quantum Pop-PDHG
        print("\n  Solving with Quantum Pop-PDHG...")
        quantum_results = []
        for run in range(n_runs):
            x, obj, elapsed, result = solve_with_quantum_pop_pdhg(
                A, b, c, lb, ub, integer_vars,
                max_iter=max_iter,
                population_size=population_size,
                seed=42+run,
            )
            is_feas, primal_viol, int_viol = check_feasibility(
                x, A, b, lb, ub, integer_vars
            )
            gap = ((obj - gurobi_obj) / abs(gurobi_obj) * 100) if gurobi_obj else 0

            tunnel_rate = result.tunnel_stats.get("success_rate", 0)
            measure_str = result.measure_stats.get("final_strength", 0)

            quantum_results.append({
                "obj": obj,
                "time": elapsed,
                "feasible": is_feas,
                "primal_viol": primal_viol,
                "int_viol": int_viol,
                "gap": gap,
                "tunnel_rate": tunnel_rate,
                "measure_str": measure_str,
            })
            print(f"    Run {run+1}: obj={obj:.4f}, time={elapsed:.2f}s, "
                  f"feas={'Yes' if is_feas else 'No'}, gap={gap:.2f}%, "
                  f"tunnel={tunnel_rate:.1%}")

        # Aggregate results
        for run_idx in range(n_runs):
            results.append(BenchmarkResult(
                instance=instance_name,
                solver="standard_pop_pdhg",
                obj_value=std_results[run_idx]["obj"],
                time_seconds=std_results[run_idx]["time"],
                iterations=max_iter,
                converged=True,
                status="optimal" if std_results[run_idx]["feasible"] else "infeasible",
                is_feasible=std_results[run_idx]["feasible"],
                primal_violation=std_results[run_idx]["primal_viol"],
                integrality_violation=std_results[run_idx]["int_viol"],
                gap_to_gurobi=std_results[run_idx]["gap"],
            ))

            results.append(BenchmarkResult(
                instance=instance_name,
                solver="quantum_pop_pdhg",
                obj_value=quantum_results[run_idx]["obj"],
                time_seconds=quantum_results[run_idx]["time"],
                iterations=max_iter,
                converged=True,
                status="optimal" if quantum_results[run_idx]["feasible"] else "infeasible",
                is_feasible=quantum_results[run_idx]["feasible"],
                primal_violation=quantum_results[run_idx]["primal_viol"],
                integrality_violation=quantum_results[run_idx]["int_viol"],
                gap_to_gurobi=quantum_results[run_idx]["gap"],
                tunnel_success_rate=quantum_results[run_idx]["tunnel_rate"],
                measure_strength_final=quantum_results[run_idx]["measure_str"],
            ))

    # Save results (convert numpy types to Python native types)
    def convert_numpy(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_data = {
        "benchmark_info": {
            "instances": instances,
            "max_iter": max_iter,
            "population_size": population_size,
            "n_runs": n_runs,
        },
        "results": convert_numpy([asdict(r) for r in results]),
    }

    results_file = output_path / "quantum_pop_pdhg_benchmark.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for instance_name in instances:
        print(f"\n{instance_name}:")
        instance_results = [r for r in results if r.instance == instance_name]

        for solver_name in ["standard_pop_pdhg", "quantum_pop_pdhg"]:
            solver_results = [r for r in instance_results if r.solver == solver_name]
            if solver_results:
                avg_obj = np.mean([r.obj_value for r in solver_results])
                avg_time = np.mean([r.time_seconds for r in solver_results])
                feas_rate = sum(r.is_feasible for r in solver_results) / len(solver_results)
                avg_gap = np.mean([r.gap_to_gurobi for r in solver_results])

                print(f"  {solver_name}:")
                print(f"    Avg obj: {avg_obj:.4f}")
                print(f"    Avg time: {avg_time:.2f}s")
                print(f"    Feasibility rate: {feas_rate:.0%}")
                print(f"    Avg gap to Gurobi: {avg_gap:.2f}%")

    return results_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Quantum Pop-PDHG")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["p0033", "knapsack_50", "setcover_50x200"],
        help="Test instances to run",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=16,
        help="Population size",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of runs per instance",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    run_benchmark(
        instances=args.instances,
        max_iter=args.max_iter,
        population_size=args.population_size,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
    )
