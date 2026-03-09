#!/usr/bin/env python
"""
Run the SEC-oriented audited benchmark on the selected MIPLIB 2017 instances.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miplib_benchmark import solve_with_quantum_pdhg, solve_with_standard_pdhg
from miplib_sec_utils import load_lp_problem, normalize_constraint_sense, write_json


def objective_error_pct(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Relative error with safe zero handling."""
    if value is None or reference is None:
        return None
    scale = max(abs(reference), 1e-9)
    return abs(value - reference) / scale * 100.0


def run_gurobi_reference(instance_path: Path, time_limit: int) -> dict[str, Any]:
    """Solve an instance with Gurobi once as an exact-solver reference."""
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    start_time = time.time()
    model = gp.read(str(instance_path), env)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.optimize()
    elapsed = time.time() - start_time

    result = {
        "status_code": int(model.Status),
        "status": str(model.Status),
        "time_seconds": elapsed,
        "objective": None,
        "best_bound": None,
        "mip_gap": None,
        "solution_count": int(model.SolCount),
    }
    if model.SolCount > 0:
        result["objective"] = float(model.ObjVal)
    if hasattr(model, "ObjBound"):
        result["best_bound"] = float(model.ObjBound)
    if hasattr(model, "MIPGap"):
        result["mip_gap"] = float(model.MIPGap)
    return result


def aggregate_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated runs for one solver on one instance."""
    feasible_rows = [row for row in run_rows if row["is_feasible"]]

    def series(key: str) -> list[float]:
        return [float(row[key]) for row in run_rows if row.get(key) is not None]

    def feasible_series(key: str) -> list[float]:
        return [float(row[key]) for row in feasible_rows if row.get(key) is not None]

    primal_values = series("primal_violation")
    int_values = series("integrality_violation")
    time_values = series("time_seconds")
    error_values = feasible_series("reference_error_pct")

    summary = {
        "n_runs": len(run_rows),
        "feasible_runs": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / float(len(run_rows)) if run_rows else 0.0,
        "mean_primal_violation": statistics.mean(primal_values) if primal_values else None,
        "median_primal_violation": statistics.median(primal_values) if primal_values else None,
        "best_primal_violation": min(primal_values) if primal_values else None,
        "mean_integrality_violation": statistics.mean(int_values) if int_values else None,
        "mean_time_seconds": statistics.mean(time_values) if time_values else None,
        "median_time_seconds": statistics.median(time_values) if time_values else None,
        "best_feasible_objective": None,
        "mean_feasible_objective": None,
        "mean_reference_error_pct_feasible": statistics.mean(error_values) if error_values else None,
    }

    feasible_objectives = feasible_series("objective_original")
    if feasible_objectives:
        summary["best_feasible_objective"] = min(feasible_objectives)
        summary["mean_feasible_objective"] = statistics.mean(feasible_objectives)

    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV with a union of keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/results/sec_selected_instances.json",
        help="Selection manifest created by select_sec_benchmark_instances.py",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Iterations for Pop-PDHG solvers.")
    parser.add_argument("--population-size", type=int, default=12, help="Population size.")
    parser.add_argument("--time-limit", type=int, default=300, help="Gurobi time limit per instance.")
    parser.add_argument("--quantum-use-two-phase", action="store_true", help="Enable the optional two-phase controller.")
    parser.add_argument("--quantum-phase1-ratio", type=float, default=0.4, help="Fraction of iterations used in feasibility-first phase.")
    parser.add_argument("--quantum-phase1-weight", type=float, default=10.0, help="Dual constraint multiplier during phase 1.")
    parser.add_argument("--quantum-tunnel-interval", type=int, default=50, help="Tunnel interval for Quantum Pop-PDHG.")
    parser.add_argument("--quantum-measure-interval", type=int, default=100, help="Measurement interval for Quantum Pop-PDHG.")
    parser.add_argument("--quantum-measurement-samples", type=int, default=5, help="Number of final measurement samples for Quantum Pop-PDHG.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 23, 37],
        help="Random seeds for repeated runs.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/sec_extended_benchmark.json",
        help="JSON file for the raw and aggregated benchmark results.",
    )
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    summary_csv = output_path.with_suffix(".summary.csv")
    runs_csv = output_path.with_suffix(".runs.csv")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_instances = manifest["selected_instances"]

    raw_run_rows: list[dict[str, Any]] = []
    instance_summaries: list[dict[str, Any]] = []

    for instance_row in selected_instances:
        instance_name = instance_row["instance_name"]
        instance_path = Path(instance_row["copied_path"])
        print(f"\n=== {instance_name} ===")

        lp = load_lp_problem(instance_path)
        constraint_sense = normalize_constraint_sense(lp.sense)
        c_min = np.asarray(lp.c, dtype=np.float64)
        if lp.obj_sense == "max":
            c_min = -c_min

        gurobi_result = run_gurobi_reference(instance_path, args.time_limit)
        gurobi_result["reference_error_pct"] = objective_error_pct(
            gurobi_result.get("objective"),
            instance_row.get("reference_obj"),
        )

        solver_runs = {"standard_pdhg": [], "quantum_pdhg": []}
        for seed in args.seeds:
            std_result = solve_with_standard_pdhg(
                lp.A,
                lp.b,
                c_min,
                lp.lb,
                lp.ub,
                lp.integer_vars,
                constraint_sense,
                max_iter=args.max_iter,
                population_size=args.population_size,
                seed=seed,
            )
            q_result = solve_with_quantum_pdhg(
                lp.A,
                lp.b,
                c_min,
                lp.lb,
                lp.ub,
                lp.integer_vars,
                constraint_sense,
                max_iter=args.max_iter,
                population_size=args.population_size,
                seed=seed,
                use_two_phase=args.quantum_use_two_phase,
                phase1_iters_ratio=args.quantum_phase1_ratio,
                phase1_constraint_weight=args.quantum_phase1_weight,
                tunnel_interval=args.quantum_tunnel_interval,
                measure_interval=args.quantum_measure_interval,
                measurement_samples=args.quantum_measurement_samples,
            )

            for solver_name, result in (("standard_pdhg", std_result), ("quantum_pdhg", q_result)):
                if result.obj_value is None:
                    objective_original = None
                elif lp.obj_sense == "min":
                    objective_original = float(result.obj_value)
                else:
                    objective_original = -float(result.obj_value)
                run_row = {
                    "instance_name": instance_name,
                    "solver": solver_name,
                    "seed": seed,
                    "objective_sense": lp.obj_sense,
                    "objective_original": objective_original,
                    "reference_obj": instance_row.get("reference_obj"),
                    "reference_error_pct": objective_error_pct(
                        objective_original,
                        instance_row.get("reference_obj"),
                    ) if result.is_feasible else None,
                    "time_seconds": float(result.time_seconds),
                    "iterations": int(result.iterations),
                    "status": result.status,
                    "is_feasible": bool(result.is_feasible),
                    "primal_violation": float(result.primal_violation),
                    "integrality_violation": float(result.integrality_violation),
                }
                raw_run_rows.append(run_row)
                solver_runs[solver_name].append(run_row)

            print(
                f"  seed={seed}: std feas={std_result.is_feasible} "
                f"viol={std_result.primal_violation:.3e}, "
                f"q feas={q_result.is_feasible} viol={q_result.primal_violation:.3e}"
            )

        standard_summary = aggregate_runs(solver_runs["standard_pdhg"])
        quantum_summary = aggregate_runs(solver_runs["quantum_pdhg"])

        instance_summary = {
            "instance_name": instance_name,
            "n_vars": int(lp.n),
            "n_cons": int(lp.m),
            "n_integer_vars": int(len(lp.integer_vars)),
            "n_binary_vars": int(len(lp.binary_vars)),
            "reference_status": instance_row.get("solu_status"),
            "reference_obj": instance_row.get("reference_obj"),
            "gurobi": gurobi_result,
            "standard_pdhg": standard_summary,
            "quantum_pdhg": quantum_summary,
        }

        std_mean = standard_summary.get("mean_primal_violation")
        q_mean = quantum_summary.get("mean_primal_violation")
        if std_mean is not None and q_mean is not None:
            instance_summary["quantum_violation_reduction_pct"] = (
                (std_mean - q_mean) / max(abs(std_mean), 1e-9) * 100.0
            )
        else:
            instance_summary["quantum_violation_reduction_pct"] = None

        instance_summaries.append(instance_summary)

    overall = {
        "n_instances": len(instance_summaries),
        "seeds": args.seeds,
        "max_iter": args.max_iter,
        "population_size": args.population_size,
        "mean_standard_primal_violation": statistics.mean(
            row["standard_pdhg"]["mean_primal_violation"]
            for row in instance_summaries
            if row["standard_pdhg"]["mean_primal_violation"] is not None
        ),
        "mean_quantum_primal_violation": statistics.mean(
            row["quantum_pdhg"]["mean_primal_violation"]
            for row in instance_summaries
            if row["quantum_pdhg"]["mean_primal_violation"] is not None
        ),
        "quantum_wins_by_violation": int(
            sum(
                1
                for row in instance_summaries
                if row["standard_pdhg"]["mean_primal_violation"] is not None
                and row["quantum_pdhg"]["mean_primal_violation"] is not None
                and row["quantum_pdhg"]["mean_primal_violation"]
                < row["standard_pdhg"]["mean_primal_violation"]
            )
        ),
        "standard_total_feasible_runs": int(
            sum(row["standard_pdhg"]["feasible_runs"] for row in instance_summaries)
        ),
        "quantum_total_feasible_runs": int(
            sum(row["quantum_pdhg"]["feasible_runs"] for row in instance_summaries)
        ),
    }

    payload = {
        "manifest": str(manifest_path),
        "config": {
            "max_iter": args.max_iter,
            "population_size": args.population_size,
            "time_limit": args.time_limit,
            "seeds": args.seeds,
            "quantum_use_two_phase": args.quantum_use_two_phase,
            "quantum_phase1_ratio": args.quantum_phase1_ratio,
            "quantum_phase1_weight": args.quantum_phase1_weight,
            "quantum_tunnel_interval": args.quantum_tunnel_interval,
            "quantum_measure_interval": args.quantum_measure_interval,
            "quantum_measurement_samples": args.quantum_measurement_samples,
        },
        "overall_summary": overall,
        "instance_summaries": instance_summaries,
        "raw_runs": raw_run_rows,
    }
    write_json(output_path, payload)
    write_csv(
        summary_csv,
        [
            {
                "instance_name": row["instance_name"],
                "n_vars": row["n_vars"],
                "n_cons": row["n_cons"],
                "reference_obj": row["reference_obj"],
                "gurobi_objective": row["gurobi"]["objective"],
                "gurobi_time_seconds": row["gurobi"]["time_seconds"],
                "standard_feasible_runs": row["standard_pdhg"]["feasible_runs"],
                "quantum_feasible_runs": row["quantum_pdhg"]["feasible_runs"],
                "standard_mean_primal_violation": row["standard_pdhg"]["mean_primal_violation"],
                "quantum_mean_primal_violation": row["quantum_pdhg"]["mean_primal_violation"],
                "quantum_violation_reduction_pct": row["quantum_violation_reduction_pct"],
            }
            for row in instance_summaries
        ],
    )
    write_csv(runs_csv, raw_run_rows)

    print("\nBenchmark complete.")
    print(f"  JSON:    {output_path}")
    print(f"  Summary: {summary_csv}")
    print(f"  Runs:    {runs_csv}")


if __name__ == "__main__":
    main()
