"""
Test SHADE on p0282 instance from MIPLIB 2017.

This script runs SHADE multiple times to evaluate:
1. Feasibility rate
2. Average solution gap
3. Success rate statistics
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.mps_reader import read_mps
from src.optimization.shade_mip import solve_mip_shade, SHADEConfig, SHADEMIP

# Optimal value from Gurobi
OPTIMAL_OBJ = -7899.413179214345


def run_single_test(seed: int, max_evals: int = 50000, population_size: int = 100, penalty_factor: float = 1000.0):
    """Run SHADE once with given seed."""
    lp = read_mps(str(PROJECT_ROOT / "data" / "miplib2017" / "p0282.mps"))

    result = solve_mip_shade(
        A=lp.A,
        b=lp.b,
        c=lp.c,
        lb=lp.lb,
        ub=lp.ub,
        integer_vars=lp.integer_vars,
        binary_vars=lp.binary_vars,
        population_size=population_size,
        max_evals=max_evals,
        seed=seed,
        verbose=False,
        penalty_factor=penalty_factor,
    )

    return result


def run_experiment(
    n_runs: int = 30,
    max_evals: int = 50000,
    population_size: int = 100,
    penalty_factor: float = 1000.0,
):
    """Run SHADE multiple times and collect statistics."""
    print("=" * 70)
    print("SHADE Experiment on p0282")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Runs: {n_runs}")
    print(f"  Max evaluations: {max_evals}")
    print(f"  Population size: {population_size}")
    print(f"  Penalty factor: {penalty_factor}")
    print(f"  Optimal objective: {OPTIMAL_OBJ:.4f}")
    print("=" * 70)

    results = []
    seeds = list(range(42, 42 + n_runs))

    for i, seed in enumerate(seeds):
        print(f"Run {i+1}/{n_runs} (seed={seed})...", end=" ")

        result = run_single_test(seed, max_evals, population_size, penalty_factor)

        # Check feasibility
        is_feasible = result.feasibility < 1e-4

        # Compute gap
        if is_feasible:
            gap = abs(result.obj_best - OPTIMAL_OBJ) / abs(OPTIMAL_OBJ) * 100
        else:
            gap = float('inf')

        results.append({
            "seed": seed,
            "obj": result.obj_best,
            "feasible": is_feasible,
            "feasibility": result.feasibility,
            "gap": gap,
            "evals": result.evaluations,
            "converged": result.converged,
            "success_rate": result.success_rate,
        })

        status = "FEASIBLE" if is_feasible else "INFEASIBLE"
        print(f"obj={result.obj_best:.2f}, feas={result.feasibility:.2e}, {status}")

    # Statistics
    feasible_results = [r for r in results if r["feasible"]]
    infeasible_results = [r for r in results if not r["feasible"]]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    feasibility_rate = len(feasible_results) / n_runs * 100
    print(f"Feasibility rate: {feasibility_rate:.1f}% ({len(feasible_results)}/{n_runs})")

    if feasible_results:
        gaps = [r["gap"] for r in feasible_results]
        objs = [r["obj"] for r in feasible_results]
        print(f"Average gap: {np.mean(gaps):.2f}%")
        print(f"Gap std: {np.std(gaps):.2f}%")
        print(f"Min gap: {np.min(gaps):.2f}%")
        print(f"Max gap: {np.max(gaps):.2f}%")
        print(f"Best objective: {np.min(objs):.4f}")
        print(f"Worst objective: {np.max(objs):.4f}")

    print(f"\nInfeasible runs: {len(infeasible_results)}")
    if infeasible_results:
        avg_violation = np.mean([r["feasibility"] for r in infeasible_results])
        print(f"Average constraint violation (infeasible): {avg_violation:.4e}")

    avg_evals = np.mean([r["evals"] for r in results])
    print(f"\nAverage evaluations: {avg_evals:.0f}")

    avg_success_rate = np.mean([r["success_rate"] for r in results])
    print(f"Average success rate: {avg_success_rate:.2%}")

    print("=" * 70)

    return results


def test_parameter_sensitivity():
    """Test different penalty factors."""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY: Penalty Factor")
    print("=" * 70)

    penalty_factors = [100, 500, 1000, 5000, 10000]
    n_runs = 10

    for penalty in penalty_factors:
        print(f"\nPenalty factor: {penalty}")
        print("-" * 50)

        feasible_count = 0
        gaps = []

        for seed in range(42, 42 + n_runs):
            result = run_single_test(seed, max_evals=30000, penalty_factor=penalty)
            is_feasible = result.feasibility < 1e-4

            if is_feasible:
                feasible_count += 1
                gap = abs(result.obj_best - OPTIMAL_OBJ) / abs(OPTIMAL_OBJ) * 100
                gaps.append(gap)

        feas_rate = feasible_count / n_runs * 100
        avg_gap = np.mean(gaps) if gaps else float('inf')

        print(f"  Feasibility: {feas_rate:.1f}% ({feasible_count}/{n_runs})")
        print(f"  Avg gap: {avg_gap:.2f}%")


def test_population_sizes():
    """Test different population sizes."""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY: Population Size")
    print("=" * 70)

    pop_sizes = [50, 100, 150, 200]
    n_runs = 10

    for pop_size in pop_sizes:
        print(f"\nPopulation size: {pop_size}")
        print("-" * 50)

        feasible_count = 0
        gaps = []

        for seed in range(42, 42 + n_runs):
            result = run_single_test(seed, max_evals=50000, population_size=pop_size)
            is_feasible = result.feasibility < 1e-4

            if is_feasible:
                feasible_count += 1
                gap = abs(result.obj_best - OPTIMAL_OBJ) / abs(OPTIMAL_OBJ) * 100
                gaps.append(gap)

        feas_rate = feasible_count / n_runs * 100
        avg_gap = np.mean(gaps) if gaps else float('inf')

        print(f"  Feasibility: {feas_rate:.1f}% ({feasible_count}/{n_runs})")
        print(f"  Avg gap: {avg_gap:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SHADE on p0282")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--evals", type=int, default=50000, help="Max evaluations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--penalty", type=float, default=1000.0, help="Penalty factor")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")

    args = parser.parse_args()

    # Main experiment
    results = run_experiment(
        n_runs=args.runs,
        max_evals=args.evals,
        population_size=args.pop,
        penalty_factor=args.penalty,
    )

    # Optional sensitivity analysis
    if args.sensitivity:
        test_parameter_sensitivity()
        test_population_sizes()
