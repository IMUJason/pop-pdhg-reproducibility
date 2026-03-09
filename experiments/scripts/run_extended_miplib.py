#!/usr/bin/env python
"""
Extended MIPLIB 2017 Benchmark for Quantum Pop-PDHG.

This script runs Quantum Pop-PDHG on an extended set of MIPLIB 2017 instances
to demonstrate scalability and general performance.

Selected instances: 15-20 small to medium-scale problems (< 5000 variables)
"""

import os
import sys
import json
import time
import gzip
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.population.quantum_pop_pdhg import (
    QuantumPopulationPDHG,
    QuantumPopPDHGConfig,
    QuantumPopPDHGResult,
)
from src.optimization.quantum_tunneling import TunnelConfig
from src.core.mps_reader import read_mps, LPData


@dataclass
class ExtendedBenchmarkResult:
    """Result for a single instance."""
    instance_name: str
    filepath: str

    # Problem stats
    n_variables: int
    n_constraints: int
    n_integer_vars: int
    n_binary_vars: int
    category: str

    # Gurobi baseline
    gurobi_obj: Optional[float] = None
    gurobi_time: Optional[float] = None
    gurobi_status: Optional[str] = None
    gurobi_gap: Optional[float] = None

    # Quantum Pop-PDHG results
    qp_obj: Optional[float] = None
    qp_time: Optional[float] = None
    qp_iterations: Optional[int] = None
    qp_status: Optional[str] = None
    qp_is_feasible: Optional[bool] = None
    qp_primal_violation: Optional[float] = None
    qp_integrality_violation: Optional[float] = None
    qp_gap_to_gurobi: Optional[float] = None

    # Quantum-specific stats
    tunnel_success_rate: Optional[float] = None
    measure_strength_final: Optional[float] = None
    n_tunnel_attempts: Optional[int] = None
    n_tunnel_success: Optional[int] = None

    # Error info
    error: Optional[str] = None


def decompress_if_needed(filepath: str) -> str:
    """Decompress .mps.gz file if needed."""
    if filepath.endswith('.mps.gz'):
        # Create temp decompressed file
        temp_path = filepath.replace('.mps.gz', '.mps')
        if not os.path.exists(temp_path):
            with gzip.open(filepath, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return temp_path
    return filepath


def load_mps_instance(filepath: str) -> Tuple:
    """Load MPS instance and return (A, b, c, lb, ub, integer_vars, vtype)."""
    # Decompress if needed
    actual_path = decompress_if_needed(filepath)

    # Use read_mps function
    result = read_mps(actual_path, use_gurobi=True)

    A = result.A
    b = result.b
    c = result.c
    lb = result.lb
    ub = result.ub
    vtype = result.var_types

    # Identify integer variables
    integer_vars = [i for i, vt in enumerate(vtype) if vt in ['B', 'I']]

    return A, b, c, lb, ub, integer_vars, vtype


def run_gurobi_baseline(filepath: str, time_limit: float = 300.0) -> Dict:
    """Run Gurobi to get baseline results."""
    try:
        import gurobipy as gp

        # Read model directly from file
        actual_path = decompress_if_needed(filepath)
        model = gp.read(actual_path)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit

        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        result = {
            'obj': None,
            'time': solve_time,
            'status': 'Unknown',
            'gap': None,
        }

        if model.SolCount > 0:
            result['obj'] = model.ObjVal

        if model.Status == gp.GRB.OPTIMAL:
            result['status'] = 'Optimal'
            result['gap'] = 0.0
        elif model.Status == gp.GRB.TIME_LIMIT:
            result['status'] = 'TimeLimit'
            if model.ObjBound != 0:
                result['gap'] = abs(model.ObjVal - model.ObjBound) / abs(model.ObjBound) * 100
        elif model.SolCount > 0:
            result['status'] = 'Feasible'

        return result

    except Exception as e:
        return {
            'obj': None,
            'time': None,
            'status': f'Error: {str(e)}',
            'gap': None,
        }


def run_quantum_pop_pdhg(
    filepath: str,
    max_iter: int = 5000,
) -> Dict:
    """Run Quantum Pop-PDHG on instance."""
    try:
        # Load instance
        A, b, c, lb, ub, integer_vars, vtype = load_mps_instance(filepath)

        # Convert to float64 for stability
        A = A.astype(np.float64)
        b = b.astype(np.float64)
        c = c.astype(np.float64)
        lb = lb.astype(np.float64)
        ub = ub.astype(np.float64)

        # Configure Quantum Pop-PDHG
        tunnel_config = TunnelConfig(
            tunnel_strength=1.0,
            tunnel_prob=0.05,
            multi_scale=True,
            adaptive=True,
            target_success_rate=0.2,
        )

        config = QuantumPopPDHGConfig(
            use_tunnel=True,
            tunnel_config=tunnel_config,
            tunnel_interval=50,
            use_progressive_measure=True,
            measure_interval=100,
            integer_vars=integer_vars,
            initial_measure_strength=0.1,
            final_measure_strength=1.0,
            use_smart_trigger=True,
        )

        # Create solver
        solver = QuantumPopulationPDHG(
            A=A,
            b=b,
            c=c,
            lb=lb,
            ub=ub,
            config=config,
        )

        # Solve
        start_time = time.time()
        result = solver.solve(
            max_iter=max_iter,
            integer_vars=integer_vars,
        )
        solve_time = time.time() - start_time

        # Check feasibility
        x = result.x_best

        # Constraint violation
        Ax = A @ x
        constraint_viol = np.max(np.maximum(Ax - b, 0)) if len(b) > 0 else 0.0

        # Bound violation
        bound_viol = max(
            np.max(np.maximum(lb - x, 0)),
            np.max(np.maximum(x - ub, 0)),
        )

        # Integrality violation
        if len(integer_vars) > 0:
            int_viol = np.max(np.abs(x[integer_vars] - np.round(x[integer_vars])))
        else:
            int_viol = 0.0

        is_feasible = (constraint_viol < 1e-4 and bound_viol < 1e-4 and int_viol < 1e-4)

        # Extract quantum stats
        tunnel_stats = getattr(result, 'tunnel_stats', {})
        measure_stats = getattr(result, 'measure_stats', {})

        return {
            'obj': result.obj_best,
            'time': solve_time,
            'iterations': result.iterations,
            'status': 'Converged' if result.converged else 'MaxIter',
            'is_feasible': is_feasible,
            'primal_violation': float(constraint_viol),
            'integrality_violation': float(int_viol),
            'tunnel_success_rate': tunnel_stats.get('success_rate', 0.0),
            'measure_strength_final': measure_stats.get('final_strength', 0.0),
            'n_tunnel_attempts': tunnel_stats.get('n_attempts', 0),
            'n_tunnel_success': tunnel_stats.get('n_success', 0),
        }

    except Exception as e:
        import traceback
        return {
            'error': f"{str(e)}\n{traceback.format_exc()}",
        }


def get_instance_info(filepath: str) -> Dict:
    """Get instance information using Gurobi."""
    try:
        import gurobipy as gp

        actual_path = decompress_if_needed(filepath)
        model = gp.read(actual_path)
        model.update()

        return {
            'n_variables': model.NumVars,
            'n_constraints': model.NumConstrs,
            'n_integer_vars': model.NumIntVars,
            'n_binary_vars': model.NumBinVars,
        }
    except Exception as e:
        print(f"    Warning: Could not get info for {filepath}: {e}")
        return {
            'n_variables': 0,
            'n_constraints': 0,
            'n_integer_vars': 0,
            'n_binary_vars': 0,
        }


def select_instances(data_dir: str, target_count: int = 20) -> List[Dict]:
    """Select instances for benchmarking."""
    # Priority instances from task description
    priority_names = [
        'assign1-5-8',
        'flugpl',
        'lseu',
        'markshare1',
        'markshare2',
        'mas284',
        'misc03',
        'misc07',
        'p2756',
    ]

    # Additional small-medium instances
    additional_candidates = [
        'p0033', 'p0201', 'p0282',
        'stein9', 'stein15', 'stein27',
        'enlight9', 'enlight13',
        'gt2',
        'rgn',
        'bell3a', 'bell5',
        'egout',
        'vpm1', 'vpm2',
        'khb05250',
        'lseu',
        'mod008',
        'pk1',
        'pp08a',
        'qiu',
        'rentacar',
        'mas284',
        'misc03',
        'lseu',
        'air01', 'air02', 'air03',
        'bm23',
        'cov1075',
        'diamond',
        'egout',
        'gen',
        'glass4',
        'gr4x6',
        'm100n500k4r1',
        'mine-166-5',
        'neos5',
        'neos8',
        'nsrand-ipx',
        'rout',
        'stein15',
        'swath1',
        'vpm1',
    ]

    selected = []
    data_path = Path(data_dir)

    # Search in all data directories
    search_dirs = [data_path / 'miplib2017']

    # First, find priority instances
    for name in priority_names:
        found = False
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in ['.mps', '.mps.gz']:
                filepath = search_dir / f"{name}{ext}"
                if filepath.exists():
                    info = get_instance_info(str(filepath))
                    selected.append({
                        'name': name,
                        'filepath': str(filepath),
                        **info,
                    })
                    found = True
                    break
            if found:
                break

    # Then add additional instances until we reach target
    for name in additional_candidates:
        if len(selected) >= target_count:
            break
        # Skip if already selected
        if any(s['name'] == name for s in selected):
            continue

        found = False
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in ['.mps', '.mps.gz']:
                filepath = search_dir / f"{name}{ext}"
                if filepath.exists():
                    info = get_instance_info(str(filepath))
                    # Only include if small enough (< 5000 vars)
                    if info['n_variables'] < 5000:
                        selected.append({
                            'name': name,
                            'filepath': str(filepath),
                            **info,
                        })
                        found = True
                        break
            if found:
                break

    return selected


def run_benchmark(
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_instances: int = 20,
    time_limit: float = 300.0,
):
    """Run extended MIPLIB benchmark."""
    data_dir = data_dir or str(PROJECT_ROOT / 'data')
    output_dir = output_dir or str(PROJECT_ROOT / 'experiments' / 'results')
    print("=" * 70)
    print("Extended MIPLIB 2017 Benchmark - Quantum Pop-PDHG")
    print("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select instances
    print("\nSelecting instances...")
    instances = select_instances(data_dir, target_count=max_instances)
    print(f"Selected {len(instances)} instances:")
    for inst in instances:
        print(f"  {inst['name']}: {inst['n_variables']} vars, {inst['n_constraints']} cons, "
              f"{inst['n_integer_vars']} int")

    # Run benchmark
    results = []

    for i, inst in enumerate(instances):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(instances)}] Testing: {inst['name']}")
        print(f"{'=' * 70}")

        result = ExtendedBenchmarkResult(
            instance_name=inst['name'],
            filepath=inst['filepath'],
            n_variables=inst['n_variables'],
            n_constraints=inst['n_constraints'],
            n_integer_vars=inst['n_integer_vars'],
            n_binary_vars=inst['n_binary_vars'],
            category='MIP' if inst['n_integer_vars'] > 0 else 'LP',
        )

        # Run Gurobi baseline
        print("  Running Gurobi baseline...")
        gurobi_result = run_gurobi_baseline(inst['filepath'], time_limit=time_limit)
        result.gurobi_obj = gurobi_result.get('obj')
        result.gurobi_time = gurobi_result.get('time')
        result.gurobi_status = gurobi_result.get('status')
        result.gurobi_gap = gurobi_result.get('gap')
        time_str = f"{result.gurobi_time:.2f}s" if result.gurobi_time is not None else "N/A"
        print(f"    Gurobi: obj={result.gurobi_obj}, time={time_str}, "
              f"status={result.gurobi_status}")

        # Run Quantum Pop-PDHG
        print("  Running Quantum Pop-PDHG...")
        qp_result = run_quantum_pop_pdhg(inst['filepath'])

        if 'error' in qp_result:
            result.error = qp_result['error']
            print(f"    ERROR: {result.error[:200]}")
        else:
            result.qp_obj = qp_result.get('obj')
            result.qp_time = qp_result.get('time')
            result.qp_iterations = qp_result.get('iterations')
            result.qp_status = qp_result.get('status')
            result.qp_is_feasible = qp_result.get('is_feasible')
            result.qp_primal_violation = qp_result.get('primal_violation')
            result.qp_integrality_violation = qp_result.get('integrality_violation')
            result.tunnel_success_rate = qp_result.get('tunnel_success_rate')
            result.measure_strength_final = qp_result.get('measure_strength_final')
            result.n_tunnel_attempts = qp_result.get('n_tunnel_attempts')
            result.n_tunnel_success = qp_result.get('n_tunnel_success')

            # Calculate gap to Gurobi
            if result.gurobi_obj is not None and result.qp_obj is not None:
                if abs(result.gurobi_obj) > 1e-6:
                    result.qp_gap_to_gurobi = abs(result.qp_obj - result.gurobi_obj) / abs(result.gurobi_obj) * 100
                else:
                    result.qp_gap_to_gurobi = abs(result.qp_obj - result.gurobi_obj) * 100

            print(f"    Quantum Pop-PDHG: obj={result.qp_obj}, time={result.qp_time:.2f}s, "
                  f"feasible={result.qp_is_feasible}")
            if result.qp_gap_to_gurobi is not None:
                print(f"    Gap to Gurobi: {result.qp_gap_to_gurobi:.2f}%")

        results.append(result)

        # Save intermediate results
        intermediate_file = output_path / 'extended_miplib_results_partial.json'
        with open(intermediate_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Save final results
    results_file = output_path / 'extended_miplib_results.json'
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to: {results_file}")

    # Generate summary
    generate_summary(results, output_path)

    return results


def generate_summary(results: List[ExtendedBenchmarkResult], output_path: Path):
    """Generate summary CSV and statistics."""
    import csv

    # Write CSV
    csv_file = output_path / 'miplib_summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Instance', 'Vars', 'Constraints', 'IntVars',
            'Gurobi_Obj', 'Gurobi_Time', 'Gurobi_Status',
            'QP_Obj', 'QP_Time', 'QP_Feasible', 'QP_Gap_%',
            'Tunnel_Rate_%', 'Error'
        ])

        for r in results:
            writer.writerow([
                r.instance_name,
                r.n_variables,
                r.n_constraints,
                r.n_integer_vars,
                r.gurobi_obj,
                r.gurobi_time,
                r.gurobi_status,
                r.qp_obj,
                r.qp_time,
                r.qp_is_feasible,
                r.qp_gap_to_gurobi,
                r.tunnel_success_rate * 100 if r.tunnel_success_rate else None,
                r.error[:100] if r.error else '',
            ])

    print(f"Summary CSV saved to: {csv_file}")

    # Print statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total = len(results)
    successful = sum(1 for r in results if r.error is None)
    feasible = sum(1 for r in results if r.qp_is_feasible)
    with_gurobi = sum(1 for r in results if r.gurobi_obj is not None)

    print(f"Total instances: {total}")
    print(f"Successfully solved by QP: {successful} ({100*successful/total:.1f}%)")
    print(f"Feasible solutions: {feasible} ({100*feasible/total:.1f}%)")
    print(f"Gurobi baselines: {with_gurobi}")

    if feasible > 0:
        avg_gap = np.mean([r.qp_gap_to_gurobi for r in results
                         if r.qp_is_feasible and r.qp_gap_to_gurobi is not None])
        print(f"Average gap to Gurobi (feasible only): {avg_gap:.2f}%")

    if successful > 0:
        avg_time = np.mean([r.qp_time for r in results if r.qp_time is not None])
        print(f"Average QP solve time: {avg_time:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended MIPLIB Benchmark")
    parser.add_argument("--data-dir", type=str,
                       default=str(PROJECT_ROOT / "data"),
                       help="Data directory")
    parser.add_argument("--output-dir", type=str,
                       default=str(PROJECT_ROOT / "experiments" / "results"),
                       help="Output directory")
    parser.add_argument("--max-instances", type=int, default=20,
                       help="Maximum number of instances to test")
    parser.add_argument("--time-limit", type=float, default=300.0,
                       help="Time limit per instance (seconds)")

    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        time_limit=args.time_limit,
    )
