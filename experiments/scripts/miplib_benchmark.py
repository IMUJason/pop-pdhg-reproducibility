#!/usr/bin/env python
"""
MIPLIB Benchmark Runner (using Gurobi for MPS parsing).

Runs comprehensive benchmarks on MIPLIB instances using:
- Gurobi (baseline)
- Standard Pop-PDHG
- Quantum Pop-PDHG

Usage:
    uv run python experiments/scripts/miplib_benchmark.py \
        --suite quick \
        --timeout 300 \
        --output experiments/results/miplib_quick_results.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.population.pop_pdhg import PopulationPDHG
from src.population.quantum_pop_pdhg import QuantumPopulationPDHG, QuantumPopPDHGConfig
from src.population.measurement import QuantumMeasurement, MeasurementConfig
from src.optimization.feasibility_repair import FeasibilityRepair, compute_violation

# Import MPS parser
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simple_mps_parser import parse_mps_file

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    max_iter: int = 10000
    population_size: int = 16
    time_limit: int = 3600
    tol: float = 1e-6
    n_runs: int = 3
    verbose: bool = False


@dataclass
class SolverResult:
    """Result from a single solver run."""
    solver: str
    obj_value: Optional[float]
    time_seconds: float
    iterations: int
    status: str
    is_feasible: bool
    primal_violation: float
    integrality_violation: float
    gap_to_gurobi: float = 0.0


@dataclass
class InstanceResult:
    """Result for a single instance."""
    instance_name: str
    n_variables: int
    n_constraints: int
    n_integer_vars: int
    gurobi_result: Optional[SolverResult]
    standard_pdhg_result: Optional[SolverResult]
    quantum_pdhg_result: Optional[SolverResult]
    summary: Dict = None


def load_instance_with_scip(filepath: str):
    """Load instance using simple MPS parser."""
    try:
        result = parse_mps_file(filepath)
        if result is None:
            return None
        A, b, c, lb, ub, integer_vars, constraint_sense = result
        return A, b, c, lb, ub, integer_vars, constraint_sense
    except Exception as e:
        print(f"  Error loading instance: {e}")
        return None


def load_instance_with_gurobi(filepath: str):
    """Load instance using Gurobi and return problem data."""
    try:
        import gurobipy as gp

        model = gp.Model()
        model.Params.OutputFlag = 0
        model.read(filepath)

        n = model.NumVars
        m = model.NumConstrs

        # Get variables
        lb = np.zeros(n)
        ub = np.zeros(n)
        integer_vars = []

        for j, var in enumerate(model.getVars()):
            lb[j] = var.LB if var.LB is not None else -np.inf
            ub[j] = var.UB if var.UB is not None else np.inf
            if var.vType in ['B', 'I']:
                integer_vars.append(j)

        # Get objective
        c = np.zeros(n)
        obj = model.getObjective()
        for j in range(n):
            c[j] = model.getCoeff(obj, model.getVars()[j])

        # Get constraints
        A_rows = []
        A_cols = []
        A_data = []
        b = np.zeros(m)

        for i, constr in enumerate(model.getConstrs()):
            b[i] = constr.RHS
            row = model.getRow(constr)
            for j in range(row.size()):
                A_rows.append(i)
                A_cols.append(row.getVar(j).index)
                A_data.append(row.getCoeff(j))

        A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, n))

        # Sense
        if model.ModelSense == 1:
            c = c  # Minimize
        else:
            c = -c  # Maximize

        return A, b, c, lb, ub, integer_vars

    except Exception as e:
        return None


def solve_with_gurobi(A, b, c, lb, ub, integer_vars, constraint_sense, time_limit=300) -> Optional[SolverResult]:
    """Solve with Gurobi."""
    try:
        import gurobipy as gp

        start_time = time.time()
        m, n = A.shape

        # Create model
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = min(time_limit, 300)

        # Variables
        x = {}
        for j in range(n):
            if j in integer_vars:
                vtype = gp.GRB.BINARY if ub[j] <= 1.0 else gp.GRB.INTEGER
            else:
                vtype = gp.GRB.CONTINUOUS
            x[j] = model.addVar(lb=lb[j], ub=ub[j], vtype=vtype, name=f"x{j}")

        # Constraints with sense
        for i in range(m):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]
            expr = gp.quicksum(vals[k] * x[cols[k]] for k in range(len(cols)))

            # Get constraint sense
            sense = constraint_sense[i] if i < len(constraint_sense) else 'L'
            if sense == 'L':
                model.addConstr(expr <= b[i])
            elif sense == 'G':
                model.addConstr(expr >= b[i])
            elif sense == 'E':
                model.addConstr(expr == b[i])

        # Objective
        model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n)), gp.GRB.MINIMIZE)

        # Optimize
        model.optimize()

        elapsed = time.time() - start_time

        if model.Status == gp.GRB.OPTIMAL:
            return SolverResult(
                solver="gurobi",
                obj_value=model.ObjVal,
                time_seconds=elapsed,
                iterations=model.IterCount,
                status="optimal",
                is_feasible=True,
                primal_violation=0.0,
                integrality_violation=0.0,
            )
        elif model.Status == gp.GRB.INFEASIBLE:
            return SolverResult(
                solver="gurobi",
                obj_value=None,
                time_seconds=elapsed,
                iterations=model.IterCount,
                status="infeasible",
                is_feasible=False,
                primal_violation=float('inf'),
                integrality_violation=0.0,
            )
        else:
            return SolverResult(
                solver="gurobi",
                obj_value=model.ObjVal if hasattr(model, 'ObjVal') else None,
                time_seconds=elapsed,
                iterations=model.IterCount,
                status=f"status_{model.Status}",
                is_feasible=model.Status == gp.GRB.OPTIMAL,
                primal_violation=0.0,
                integrality_violation=0.0,
            )

    except Exception as e:
        print(f"  Gurobi error: {e}")
        return None


def solve_with_standard_pdhg(
    A,
    b,
    c,
    lb,
    ub,
    integer_vars,
    constraint_sense,
    max_iter=10000,
    population_size=16,
    tol=1e-6,
    seed: int = 42,
    measurement_samples: int = 5,
    measurement_temperature: float = 0.5,
    measurement_rounding_strategy: str = "probabilistic",
) -> SolverResult:
    """Solve with Standard Pop-PDHG."""
    start_time = time.time()

    # Fix infinite bounds
    lb = np.maximum(lb, -1e10)
    ub = np.minimum(ub, 1e10)

    # Convert >= constraints to <= by negating
    A_pdhg = A.copy()
    b_pdhg = b.copy()
    for i, sense in enumerate(constraint_sense):
        if sense == 'G':
            A_pdhg[i, :] = -A_pdhg[i, :]
            b_pdhg[i] = -b[i]
        elif sense == 'E':
            # For equality, we need to handle differently - for now treat as <=
            pass

    solver = PopulationPDHG(A_pdhg, b_pdhg, c, lb, ub, population_size=population_size)

    try:
        result = solver.solve(
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            seed=seed,
        )
    except Exception as e:
        return SolverResult(
            solver="standard_pdhg",
            obj_value=None,
            time_seconds=time.time() - start_time,
            iterations=0,
            status="error",
            is_feasible=False,
            primal_violation=0.0,
            integrality_violation=0.0,
        )

    elapsed = time.time() - start_time

    # Apply measurement for integer variables
    best_x = result.x_best.copy()
    best_obj = result.obj_best

    if integer_vars:
        measurement = QuantumMeasurement(MeasurementConfig(
            temperature=measurement_temperature,
            rounding_strategy=measurement_rounding_strategy,
            repair_infeasible=True,
        ))

        solutions = measurement.measure(
            result.state, integer_vars, A_pdhg, b_pdhg, c, lb, ub,
            n_samples=measurement_samples, seed=seed
        )

        for x_sol, obj, feas in solutions:
            if feas and obj < best_obj:
                best_x = x_sol
                best_obj = obj

    # Apply feasibility repair
    repair = FeasibilityRepair(A, b, lb, ub, integer_vars, constraint_sense)
    best_x, repaired_feas, repair_stats = repair.repair(best_x, method='full')

    # Check feasibility with original constraints
    primal_viol = repair_stats.get('final_primal_viol', 1.0)
    int_viol = repair_stats.get('final_int_viol', 1.0)
    best_obj = float(c @ best_x)

    is_feasible = bool(primal_viol < 1e-4 and int_viol < 1e-6)

    return SolverResult(
        solver="standard_pdhg",
        obj_value=best_obj,
        time_seconds=elapsed,
        iterations=result.iterations,
        status=result.status,
        is_feasible=is_feasible,
        primal_violation=primal_viol,
        integrality_violation=int_viol,
    )


def solve_with_quantum_pdhg(
    A,
    b,
    c,
    lb,
    ub,
    integer_vars,
    constraint_sense,
    max_iter=10000,
    population_size=16,
    tol=1e-6,
    seed: int = 42,
    use_two_phase: bool = False,
    phase1_iters_ratio: float = 0.4,
    phase1_constraint_weight: float = 10.0,
    tunnel_interval: int = 50,
    measure_interval: int = 100,
    use_smart_trigger: bool = True,
    use_feasibility_aware_tunnel: bool = True,
    use_enhanced_repair: bool = True,
    initial_measure_strength: float = 0.1,
    final_measure_strength: float = 1.0,
    measurement_samples: int = 5,
    measurement_temperature: float = 0.5,
    measurement_rounding_strategy: str = "probabilistic",
) -> SolverResult:
    """Solve with Quantum Pop-PDHG."""
    start_time = time.time()

    # Fix infinite bounds
    lb = np.maximum(lb, -1e10)
    ub = np.minimum(ub, 1e10)

    # Convert >= constraints to <= by negating
    A_pdhg = A.copy()
    b_pdhg = b.copy()
    for i, sense in enumerate(constraint_sense):
        if sense == 'G':
            A_pdhg[i, :] = -A_pdhg[i, :]
            b_pdhg[i] = -b[i]
        elif sense == 'E':
            pass

    config = QuantumPopPDHGConfig(
        use_tunnel=True,
        tunnel_interval=tunnel_interval,
        use_progressive_measure=True,
        measure_interval=measure_interval,
        integer_vars=integer_vars,
        initial_measure_strength=initial_measure_strength,
        final_measure_strength=final_measure_strength,
        use_smart_trigger=use_smart_trigger,
    )

    solver = QuantumPopulationPDHG(
        A_pdhg, b_pdhg, c, lb, ub,
        population_size=population_size,
        config=config,
    )

    try:
        result = solver.solve(
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            seed=seed,
            integer_vars=integer_vars,
            use_enhanced_repair=use_enhanced_repair,
            use_feasibility_aware_tunnel=use_feasibility_aware_tunnel,
            use_two_phase=use_two_phase,
            phase1_iters_ratio=phase1_iters_ratio,
            phase1_constraint_weight=phase1_constraint_weight,
        )
    except Exception as e:
        return SolverResult(
            solver="quantum_pdhg",
            obj_value=None,
            time_seconds=time.time() - start_time,
            iterations=0,
            status="error",
            is_feasible=False,
            primal_violation=0.0,
            integrality_violation=0.0,
        )

    elapsed = time.time() - start_time

    # Get best solution
    best_x = result.x_best.copy()
    best_obj = result.obj_best

    # Apply final measurement
    if integer_vars:
        measurement = QuantumMeasurement(MeasurementConfig(
            temperature=measurement_temperature,
            rounding_strategy=measurement_rounding_strategy,
            repair_infeasible=True,
        ))

        solutions = measurement.measure(
            result.state, integer_vars, A_pdhg, b_pdhg, c, lb, ub,
            n_samples=measurement_samples, seed=seed
        )

        for x_sol, obj, feas in solutions:
            if feas and obj < best_obj:
                best_x = x_sol
                best_obj = obj

    # Apply feasibility repair
    repair = FeasibilityRepair(A, b, lb, ub, integer_vars, constraint_sense)
    best_x, repaired_feas, repair_stats = repair.repair(best_x, method='full')

    # Check feasibility with original constraints
    primal_viol = repair_stats.get('final_primal_viol', 1.0)
    int_viol = repair_stats.get('final_int_viol', 1.0)
    best_obj = float(c @ best_x)

    is_feasible = bool(primal_viol < 1e-4 and int_viol < 1e-6)

    return SolverResult(
        solver="quantum_pdhg",
        obj_value=best_obj,
        time_seconds=elapsed,
        iterations=result.iterations,
        status=result.status,
        is_feasible=is_feasible,
        primal_violation=primal_viol,
        integrality_violation=int_viol,
    )


def find_instance_file(instance_name: str, data_dirs: List[str]) -> Optional[str]:
    """Find instance file (prefer .mps over .mps.gz)."""
    for data_dir in data_dirs:
        # Try .mps first
        candidate = Path(data_dir) / f"{instance_name}.mps"
        if candidate.exists():
            return str(candidate)
        # Try .mps.gz
        candidate = Path(data_dir) / f"{instance_name}.mps.gz"
        if candidate.exists():
            return str(candidate)
    return None


def run_benchmark(instances: List[str], config: BenchmarkConfig,
                  data_dirs: List[str]) -> List[InstanceResult]:
    """Run benchmark on all instances."""
    results = []

    for instance_name in instances:
        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"{'='*60}")

        # Find instance file
        instance_file = find_instance_file(instance_name, data_dirs)

        if not instance_file:
            print(f"  Instance file not found: {instance_name}")
            continue

        print(f"  File: {instance_file}")

        # Load instance with SCIP (more robust MPS parsing)
        data = load_instance_with_scip(instance_file)
        if data is None:
            # Fall back to Gurobi
            data = load_instance_with_gurobi(instance_file)
            if data is None:
                print(f"  Failed to load instance")
                continue
            # Add default constraint sense for Gurobi-loaded data
            A, b, c, lb, ub, integer_vars = data
            constraint_sense = ['L'] * len(b)
        else:
            A, b, c, lb, ub, integer_vars, constraint_sense = data

        info = {
            "n_variables": len(c),
            "n_constraints": len(b),
            "n_integer_vars": len(integer_vars),
        }
        print(f"  Variables: {info['n_variables']}, Constraints: {info['n_constraints']}, "
              f"Integer: {info['n_integer_vars']}")

        # Run solvers
        print("  Running Gurobi...")
        gurobi_result = solve_with_gurobi(A, b, c, lb, ub, integer_vars, constraint_sense, config.time_limit)

        print("  Running Standard PDHG...")
        standard_result = solve_with_standard_pdhg(
            A, b, c, lb, ub, integer_vars, constraint_sense,
            max_iter=config.max_iter,
            population_size=config.population_size,
            tol=config.tol,
        )

        print("  Running Quantum PDHG...")
        quantum_result = solve_with_quantum_pdhg(
            A, b, c, lb, ub, integer_vars, constraint_sense,
            max_iter=config.max_iter,
            population_size=config.population_size,
            tol=config.tol,
        )

        # Calculate gaps
        gurobi_obj = gurobi_result.obj_value if gurobi_result else None

        if gurobi_obj is not None and abs(gurobi_obj) > 1e-10:
            if standard_result and standard_result.obj_value is not None:
                standard_result.gap_to_gurobi = (standard_result.obj_value - gurobi_obj) / abs(gurobi_obj) * 100
            if quantum_result and quantum_result.obj_value is not None:
                quantum_result.gap_to_gurobi = (quantum_result.obj_value - gurobi_obj) / abs(gurobi_obj) * 100

        # Print results
        if gurobi_result:
            if gurobi_result.obj_value is not None:
                print(f"  Gurobi: obj={gurobi_result.obj_value:.4f}, time={gurobi_result.time_seconds:.2f}s, "
                      f"status={gurobi_result.status}")
            else:
                print(f"  Gurobi: obj=N/A, time={gurobi_result.time_seconds:.2f}s, "
                      f"status={gurobi_result.status}")
        if standard_result.obj_value is not None:
            print(f"  Standard PDHG: obj={standard_result.obj_value:.4f}, time={standard_result.time_seconds:.2f}s, "
                  f"feas={'Yes' if standard_result.is_feasible else 'No'}")
        else:
            print(f"  Standard PDHG: obj=N/A, time={standard_result.time_seconds:.2f}s, "
                  f"feas={'Yes' if standard_result.is_feasible else 'No'}")
        if quantum_result.obj_value is not None:
            print(f"  Quantum PDHG: obj={quantum_result.obj_value:.4f}, time={quantum_result.time_seconds:.2f}s, "
                  f"feas={'Yes' if quantum_result.is_feasible else 'No'}")
        else:
            print(f"  Quantum PDHG: obj=N/A, time={quantum_result.time_seconds:.2f}s, "
                  f"feas={'Yes' if quantum_result.is_feasible else 'No'}")

        # Store result
        result = InstanceResult(
            instance_name=instance_name,
            n_variables=info["n_variables"],
            n_constraints=info["n_constraints"],
            n_integer_vars=info["n_integer_vars"],
            gurobi_result=gurobi_result,
            standard_pdhg_result=standard_result,
            quantum_pdhg_result=quantum_result,
        )

        # Add summary
        result.summary = {
            "gurobi_obj": gurobi_result.obj_value if gurobi_result else None,
            "standard_obj": standard_result.obj_value,
            "quantum_obj": quantum_result.obj_value,
            "standard_feas": standard_result.is_feasible,
            "quantum_feas": quantum_result.is_feasible,
            "standard_gap": standard_result.gap_to_gurobi,
            "quantum_gap": quantum_result.gap_to_gurobi,
        }

        results.append(result)

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MIPLIB Benchmark Runner")
    parser.add_argument("--suite", type=str, default="paper",
                       choices=["paper", "quick", "custom"],
                       help="Test suite to run")
    parser.add_argument("--instances", nargs="+", default=None,
                       help="Custom list of instances")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Time limit per instance (seconds)")
    parser.add_argument("--max-iter", type=int, default=10000,
                       help="Maximum iterations")
    parser.add_argument("--population-size", type=int, default=16,
                       help="Population size")
    parser.add_argument("--output", type=str,
                       default="experiments/results/miplib_benchmark.json",
                       help="Output file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Data directories
    data_dirs = [str(PROJECT_ROOT / 'data' / 'miplib2017')]

    # Test suites
    suites = {
        "paper": ["p0033", "p0201", "p0282", "knapsack_50"],
        "quick": ["p0033", "p0201", "p0282"],
    }

    instances = args.instances or suites.get(args.suite, suites["quick"])

    config = BenchmarkConfig(
        max_iter=args.max_iter,
        population_size=args.population_size,
        time_limit=args.timeout,
        verbose=args.verbose,
    )

    print("=" * 70)
    print("MIPLIB Benchmark Runner")
    print("=" * 70)
    print(f"Suite: {args.suite}")
    print(f"Instances: {instances}")
    print(f"Config: max_iter={config.max_iter}, pop_size={config.population_size}, "
          f"timeout={config.time_limit}s")

    # Run benchmark
    results = run_benchmark(instances, config, data_dirs)

    # Save results
    output_data = {
        "suite": args.suite,
        "config": asdict(config),
        "results": [],
    }

    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    for r in results:
        r_dict = asdict(r)
        # Convert solver results to dicts
        for key in ['gurobi_result', 'standard_pdhg_result', 'quantum_pdhg_result']:
            val = r_dict.get(key)
            if val is not None and hasattr(val, '__dataclass_fields__'):
                r_dict[key] = asdict(val)
        # Convert numpy arrays
        r_dict = convert_to_serializable(r_dict)
        output_data["results"].append(r_dict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    feasible_count = {"standard": 0, "quantum": 0}
    for result in results:
        print(f"\n{result.instance_name}:")
        print(f"  Variables: {result.n_variables}, Constraints: {result.n_constraints}")
        if result.summary:
            print(f"  Gurobi: {result.summary.get('gurobi_obj', 'N/A')}")
            std_obj = result.summary.get('standard_obj')
            qtm_obj = result.summary.get('quantum_obj')
            print(f"  Standard PDHG: {std_obj if std_obj is not None else 'N/A'} "
                  f"(feas={result.summary.get('standard_feas', 'N/A')}, gap={result.summary.get('standard_gap', 0):.2f}%)")
            print(f"  Quantum PDHG: {qtm_obj if qtm_obj is not None else 'N/A'} "
                  f"(feas={result.summary.get('quantum_feas', 'N/A')}, gap={result.summary.get('quantum_gap', 0):.2f}%)")

            if result.summary.get('standard_feas'):
                feasible_count["standard"] += 1
            if result.summary.get('quantum_feas'):
                feasible_count["quantum"] += 1

    print(f"\nFeasibility Summary:")
    print(f"  Standard PDHG: {feasible_count['standard']}/{len(results)}")
    print(f"  Quantum PDHG: {feasible_count['quantum']}/{len(results)}")


if __name__ == "__main__":
    main()
