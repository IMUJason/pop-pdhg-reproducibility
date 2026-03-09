"""
Ablation Study for Quantum Pop-PDHG.

Tests 5 configurations to validate contribution of each component:
1. Base Pop-PDHG (no tunneling, no progressive measurement)
2. Pop-PDHG + Quantum Tunneling only
3. Pop-PDHG + Progressive Measurement only
4. Pop-PDHG + HMC Dynamics only
5. Full Quantum Pop-PDHG (all components)
"""

import sys
import json
import time
import csv
from pathlib import Path
from dataclasses import asdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.population.quantum_pop_pdhg import (
    QuantumPopulationPDHG,
    QuantumPopPDHGConfig,
)
from src.core.mps_reader import read_mps
from src.optimization.feasibility_repair import compute_violation

# Test problems
TEST_PROBLEMS = [
    ("knapsack_50", "data/miplib2017/knapsack_50.mps"),
    ("p0033", "data/miplib2017/p0033.mps"),
    ("p0201", "data/miplib2017/p0201.mps"),
    ("p0282", "data/miplib2017/p0282.mps"),
]

# Gurobi optimal values verified on the current MIPLIB copies
GUROBI_OPTIMAL = {
    "knapsack_50": -2051.00,
    "p0033": -68.00,
    "p0201": -226.00,
    "p0282": -7899.413179214345,
}

# Ablation configurations
ABLATION_CONFIGS = [
    {
        "name": "Base Pop-PDHG",
        "use_tunnel": False,
        "use_progressive_measure": False,
        "use_smart_trigger": False,
    },
    {
        "name": "+ Quantum Tunneling",
        "use_tunnel": True,
        "use_progressive_measure": False,
        "use_smart_trigger": False,
    },
    {
        "name": "+ Progressive Measurement",
        "use_tunnel": False,
        "use_progressive_measure": True,
        "use_smart_trigger": False,
    },
    {
        "name": "+ HMC Dynamics",
        "use_tunnel": False,
        "use_progressive_measure": False,
        "use_smart_trigger": True,
    },
    {
        "name": "Full Quantum Pop-PDHG",
        "use_tunnel": True,
        "use_progressive_measure": True,
        "use_smart_trigger": True,
    },
]

SENSE_MAP = {
    "<": "L",
    ">": "G",
    "=": "E",
    "L": "L",
    "G": "G",
    "E": "E",
}


def normalize_constraint_sense(senses):
    """Normalize constraint senses to L/G/E format."""
    if not senses:
        return []
    return [SENSE_MAP.get(s, "L") for s in senses]


def evaluate_solution(x, A, b, lb, ub, constraint_sense, integer_vars):
    """Evaluate feasibility under original constraints, bounds, and integrality."""
    bound_viol = 0.0
    if len(lb) > 0:
        bound_viol = max(
            float(np.max(np.maximum(lb - x, 0.0))),
            float(np.max(np.maximum(x - ub, 0.0))),
        )

    primal_viol, int_viol, _ = compute_violation(
        x=x,
        A=A,
        b=b,
        constraint_sense=constraint_sense,
        integer_vars=integer_vars,
    )

    max_violation = max(bound_viol, primal_viol, int_viol)
    is_feasible = bool(bound_viol < 1e-6 and primal_viol < 1e-4 and int_viol < 1e-6)
    return is_feasible, max_violation, primal_viol, int_viol, bound_viol


def run_single_experiment(problem_name, problem_path, config_dict, seed=42):
    """Run a single ablation experiment."""
    print(f"\n  Running: {config_dict['name']}")

    # Load problem
    lp_data = read_mps(str(problem_path))

    if lp_data is None:
        print(f"    ERROR: Could not load problem {problem_name}")
        return None

    # Extract problem data
    A = lp_data.A
    b = lp_data.b
    c = lp_data.c
    lb = lp_data.lb
    ub = lp_data.ub
    constraint_sense = normalize_constraint_sense(lp_data.sense)

    # Use actual integer-variable metadata from the instance.
    integer_vars = list(lp_data.integer_vars or [])

    # Create config
    config = QuantumPopPDHGConfig(
        use_tunnel=config_dict["use_tunnel"],
        use_progressive_measure=config_dict["use_progressive_measure"],
        use_smart_trigger=config_dict["use_smart_trigger"],
        tunnel_interval=50,
        measure_interval=100,
        integer_vars=integer_vars,
        initial_measure_strength=0.1,
        final_measure_strength=1.0,
    )

    # Create solver and solve
    solver = QuantumPopulationPDHG(
        A=A, b=b, c=c, lb=lb, ub=ub,
        population_size=20,
        config=config
    )

    start_time = time.time()
    result = solver.solve(
        max_iter=1000,
        tol=1e-6,
        seed=seed,
        integer_vars=integer_vars,
        use_enhanced_repair=True,
        use_feasibility_aware_tunnel=True,
    )
    runtime = time.time() - start_time

    # Calculate metrics
    gurobi_opt = GUROBI_OPTIMAL.get(problem_name, 0)
    if gurobi_opt != 0:
        gap = abs((result.obj_best - gurobi_opt) / gurobi_opt) * 100
    else:
        gap = abs(result.obj_best - gurobi_opt)

    # Check feasibility
    x = result.x_best
    is_feasible, feasibility_violation, primal_viol, int_viol, bound_viol = evaluate_solution(
        x=x,
        A=A,
        b=b,
        lb=lb,
        ub=ub,
        constraint_sense=constraint_sense,
        integer_vars=integer_vars,
    )
    feasibility_rate = 100.0 if is_feasible else 0.0

    # Get tunneling stats if available
    tunnel_success_rate = 0.0
    tunnel_stats = getattr(result, "tunnel_stats", {}) or getattr(solver, "tunnel_stats", {})
    if tunnel_stats.get("attempts", 0) > 0:
        if "success_rate" in tunnel_stats:
            tunnel_success_rate = float(tunnel_stats["success_rate"]) * 100
        else:
            tunnel_success_rate = (
                float(tunnel_stats.get("successes", 0)) / float(tunnel_stats["attempts"])
            ) * 100

    return {
        "config_name": config_dict["name"],
        "problem": problem_name,
        "objective_value": float(result.obj_best),
        "gurobi_optimal": float(gurobi_opt),
        "gap_percent": float(gap),
        "feasibility_rate": float(feasibility_rate),
        "max_violation": float(feasibility_violation),
        "primal_violation": float(primal_viol),
        "integrality_violation": float(int_viol),
        "bound_violation": float(bound_viol),
        "runtime_seconds": float(runtime),
        "iterations": int(result.iterations),
        "tunnel_success_rate": float(tunnel_success_rate),
        "converged": bool(result.converged),
    }


def run_ablation_study(output_dir: Path | None = None):
    """Run complete ablation study."""
    print("=" * 70)
    print("Quantum Pop-PDHG Ablation Study")
    print("=" * 70)

    results = []
    output_dir = output_dir or (PROJECT_ROOT / "experiments" / "results")

    for problem_name, problem_path in TEST_PROBLEMS:
        print(f"\n{'='*70}")
        print(f"Problem: {problem_name}")
        print(f"{'='*70}")

        # Check if problem file exists
        full_path = PROJECT_ROOT / problem_path
        if not full_path.exists():
            print(f"  WARNING: Problem file not found: {full_path}")
            continue

        for config_dict in ABLATION_CONFIGS:
            result = run_single_experiment(problem_name, full_path, config_dict)
            if result:
                results.append(result)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "ablation_study_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {json_path}")

    # Save CSV summary
    csv_path = output_dir / "ablation_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"CSV summary saved to: {csv_path}")

    # Print summary table
    print_summary_table(results)

    return results


def print_summary_table(results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY SUMMARY")
    print("=" * 100)

    # Group by problem
    problems = sorted(set(r["problem"] for r in results))
    configs = [c["name"] for c in ABLATION_CONFIGS]

    for problem in problems:
        print(f"\n{problem.upper()} (Gurobi Optimal: {GUROBI_OPTIMAL.get(problem, 'N/A')})")
        print("-" * 100)
        print(f"{'Configuration':<30} {'Objective':>12} {'Gap%':>10} {'Feas%':>8} {'Time(s)':>10} {'Iter':>8} {'Tunnel%':>10}")
        print("-" * 100)

        for config_name in configs:
            for r in results:
                if r["problem"] == problem and r["config_name"] == config_name:
                    print(f"{r['config_name']:<30} "
                          f"{r['objective_value']:>12.2f} "
                          f"{r['gap_percent']:>10.2f} "
                          f"{r['feasibility_rate']:>8.1f} "
                          f"{r['runtime_seconds']:>10.2f} "
                          f"{r['iterations']:>8d} "
                          f"{r['tunnel_success_rate']:>10.1f}")

    print("\n" + "=" * 100)

    # Component contribution analysis
    print("\nCOMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 100)

    base_results = {r["problem"]: r for r in results if r["config_name"] == "Base Pop-PDHG"}
    full_results = {r["problem"]: r for r in results if r["config_name"] == "Full Quantum Pop-PDHG"}

    for problem in problems:
        if problem in base_results and problem in full_results:
            base = base_results[problem]
            full = full_results[problem]

            gap_improvement = base["gap_percent"] - full["gap_percent"]
            feas_improvement = full["feasibility_rate"] - base["feasibility_rate"]

            print(f"\n{problem}:")
            print(f"  Gap improvement: {base['gap_percent']:.2f}% -> {full['gap_percent']:.2f}% "
                  f"({gap_improvement:+.2f}%)")
            print(f"  Feasibility improvement: {base['feasibility_rate']:.1f}% -> {full['feasibility_rate']:.1f}% "
                  f"({feas_improvement:+.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Quantum Pop-PDHG ablation study")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "results"),
        help="Directory for ablation outputs",
    )
    args = parser.parse_args()

    run_ablation_study(output_dir=Path(args.output_dir))
