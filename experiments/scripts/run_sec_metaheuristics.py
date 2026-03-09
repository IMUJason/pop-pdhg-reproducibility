#!/usr/bin/env python
"""
Benchmark SEC-style population/metaheuristic baselines on a representative subset.
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
from miplib_sec_utils import (
    build_leq_relaxation,
    load_lp_problem,
    normalize_constraint_sense,
    write_json,
)
from src.optimization.de_mip import solve_mip_de
from src.optimization.feasibility_repair import FeasibilityRepair
from src.optimization.ga_mip import solve_mip_ga
from src.optimization.shade_mip import solve_mip_shade
from src.optimization.sl_pso import solve_mip_slpso


def objective_error_pct(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None:
        return None
    scale = max(abs(reference), 1e-9)
    return abs(value - reference) / scale * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def evaluate_with_repair(lp, x_candidate: np.ndarray) -> tuple[np.ndarray, bool, float, float, float]:
    """Apply the shared repair layer and evaluate on the original problem."""
    senses = normalize_constraint_sense(lp.sense)
    repair = FeasibilityRepair(lp.A, lp.b, lp.lb, lp.ub, lp.integer_vars, senses)
    x_fixed, _, stats = repair.repair(np.asarray(x_candidate, dtype=np.float64), method="full")
    primal = float(stats.get("final_primal_viol", 1.0))
    int_viol = float(stats.get("final_int_viol", 1.0))
    feasible = bool(primal < 1e-4 and int_viol < 1e-6)
    objective = float(np.asarray(lp.c, dtype=np.float64) @ x_fixed)
    return x_fixed, feasible, objective, primal, int_viol


def sanitize_bounds(lb: np.ndarray, ub: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace infinite bounds with data-scaled finite surrogates for metaheuristics."""
    lb_safe = np.asarray(lb, dtype=np.float64).copy()
    ub_safe = np.asarray(ub, dtype=np.float64).copy()

    finite_pool = []
    finite_pool.extend(np.abs(lb_safe[np.isfinite(lb_safe)]).tolist())
    finite_pool.extend(np.abs(ub_safe[np.isfinite(ub_safe)]).tolist())
    finite_pool.extend(np.abs(np.asarray(rhs, dtype=np.float64)).tolist())
    finite_pool = [value for value in finite_pool if value > 0]

    if finite_pool:
        scale = float(np.percentile(finite_pool, 90))
    else:
        scale = 100.0
    scale = min(max(scale * 2.0, 100.0), 1e6)

    lb_safe[~np.isfinite(lb_safe)] = -scale
    ub_safe[~np.isfinite(ub_safe)] = scale
    ub_safe = np.maximum(ub_safe, lb_safe)
    return lb_safe, ub_safe


def run_solver(
    solver_name: str,
    lp,
    seed: int,
    pdhg_max_iter: int,
    pdhg_population_size: int,
    evo_population_size: int,
    evo_max_evals: int,
    slpso_max_iter: int,
    quantum_use_two_phase: bool,
    quantum_phase1_ratio: float,
    quantum_phase1_weight: float,
    quantum_tunnel_interval: int,
    quantum_measure_interval: int,
    quantum_measurement_samples: int,
) -> dict[str, Any]:
    """Run one solver on one instance with one seed."""
    senses = normalize_constraint_sense(lp.sense)
    c_min = np.asarray(lp.c, dtype=np.float64)
    if lp.obj_sense == "max":
        c_min = -c_min

    start_time = time.time()

    if solver_name == "standard_pdhg":
        result = solve_with_standard_pdhg(
            lp.A,
            lp.b,
            c_min,
            lp.lb,
            lp.ub,
            lp.integer_vars,
            senses,
            max_iter=pdhg_max_iter,
            population_size=pdhg_population_size,
            seed=seed,
        )
        objective = None if result.obj_value is None else float(result.obj_value)
        if objective is not None and lp.obj_sense == "max":
            objective = -objective
        return {
            "time_seconds": float(result.time_seconds),
            "iterations_or_evals": int(result.iterations),
            "status": result.status,
            "is_feasible": bool(result.is_feasible),
            "objective_original": objective,
            "primal_violation": float(result.primal_violation),
            "integrality_violation": float(result.integrality_violation),
        }

    if solver_name == "quantum_pdhg":
        result = solve_with_quantum_pdhg(
            lp.A,
            lp.b,
            c_min,
            lp.lb,
            lp.ub,
            lp.integer_vars,
            senses,
            max_iter=pdhg_max_iter,
            population_size=pdhg_population_size,
            seed=seed,
            use_two_phase=quantum_use_two_phase,
            phase1_iters_ratio=quantum_phase1_ratio,
            phase1_constraint_weight=quantum_phase1_weight,
            tunnel_interval=quantum_tunnel_interval,
            measure_interval=quantum_measure_interval,
            measurement_samples=quantum_measurement_samples,
        )
        objective = None if result.obj_value is None else float(result.obj_value)
        if objective is not None and lp.obj_sense == "max":
            objective = -objective
        return {
            "time_seconds": float(result.time_seconds),
            "iterations_or_evals": int(result.iterations),
            "status": result.status,
            "is_feasible": bool(result.is_feasible),
            "objective_original": objective,
            "primal_violation": float(result.primal_violation),
            "integrality_violation": float(result.integrality_violation),
        }

    a_leq, b_leq, c_leq, _ = build_leq_relaxation(lp)
    lb_safe, ub_safe = sanitize_bounds(lp.lb, lp.ub, lp.b)

    if solver_name == "ga":
        result = solve_mip_ga(
            c_leq,
            a_leq,
            b_leq,
            lb_safe,
            ub_safe,
            integer_vars=lp.integer_vars,
            binary_vars=lp.binary_vars,
            population_size=evo_population_size,
            max_generations=max(1, evo_max_evals // max(evo_population_size, 1)),
            seed=seed,
        )
        x_candidate = result.x_best
        iterations_or_evals = int(result.evaluations)
        status = "converged" if result.converged else "stopped"
    elif solver_name == "de":
        result = solve_mip_de(
            c_leq,
            a_leq,
            b_leq,
            lb_safe,
            ub_safe,
            integer_vars=lp.integer_vars,
            binary_vars=lp.binary_vars,
            population_size=evo_population_size,
            max_evals=evo_max_evals,
            seed=seed,
        )
        x_candidate = result.x_best
        iterations_or_evals = int(result.evaluations)
        status = "converged" if result.converged else "stopped"
    elif solver_name == "shade":
        result = solve_mip_shade(
            a_leq,
            b_leq,
            c_leq,
            lb_safe,
            ub_safe,
            integer_vars=lp.integer_vars,
            binary_vars=lp.binary_vars,
            population_size=evo_population_size,
            max_evals=evo_max_evals,
            seed=seed,
            verbose=False,
        )
        x_candidate = result.x_best
        iterations_or_evals = int(result.evaluations)
        status = "converged" if result.converged else "stopped"
    elif solver_name == "slpso":
        result = solve_mip_slpso(
            a_leq,
            b_leq,
            c_leq,
            lb_safe,
            ub_safe,
            integer_vars=lp.integer_vars,
            population_size=evo_population_size,
            max_iter=slpso_max_iter,
            seed=seed,
            verbose=False,
        )
        x_candidate = result.x_best
        iterations_or_evals = int(result.iterations)
        status = "converged" if result.is_feasible else "stopped"
    else:
        raise ValueError(f"Unsupported solver: {solver_name}")

    _, feasible, objective, primal, int_viol = evaluate_with_repair(lp, x_candidate)
    return {
        "time_seconds": float(time.time() - start_time),
        "iterations_or_evals": iterations_or_evals,
        "status": status,
        "is_feasible": feasible,
        "objective_original": objective,
        "primal_violation": primal,
        "integrality_violation": int_viol,
    }


def aggregate_solver_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible_rows = [row for row in rows if row["is_feasible"]]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    def feasible_values(key: str) -> list[float]:
        return [float(row[key]) for row in feasible_rows if row.get(key) is not None]

    primal = values("primal_violation")
    times = values("time_seconds")
    errors = feasible_values("reference_error_pct")
    feas_objs = feasible_values("objective_original")

    return {
        "n_runs": len(rows),
        "feasible_runs": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / float(len(rows)) if rows else 0.0,
        "mean_primal_violation": statistics.mean(primal) if primal else None,
        "median_primal_violation": statistics.median(primal) if primal else None,
        "best_primal_violation": min(primal) if primal else None,
        "mean_time_seconds": statistics.mean(times) if times else None,
        "mean_reference_error_pct_feasible": statistics.mean(errors) if errors else None,
        "best_feasible_objective": min(feas_objs) if feas_objs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/results/sec_selected_instances.json",
        help="Instance manifest from the selector script.",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=6,
        help="Number of smallest selected instances used for metaheuristic baselines.",
    )
    parser.add_argument("--pdhg-max-iter", type=int, default=1000)
    parser.add_argument("--pdhg-population-size", type=int, default=12)
    parser.add_argument("--quantum-use-two-phase", action="store_true")
    parser.add_argument("--quantum-phase1-ratio", type=float, default=0.4)
    parser.add_argument("--quantum-phase1-weight", type=float, default=10.0)
    parser.add_argument("--quantum-tunnel-interval", type=int, default=50)
    parser.add_argument("--quantum-measure-interval", type=int, default=100)
    parser.add_argument("--quantum-measurement-samples", type=int, default=5)
    parser.add_argument("--evo-population-size", type=int, default=24)
    parser.add_argument("--evo-max-evals", type=int, default=12000)
    parser.add_argument("--slpso-max-iter", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["ga", "de", "slpso", "shade", "standard_pdhg", "quantum_pdhg"],
    )
    parser.add_argument(
        "--output",
        default="experiments/results/sec_metaheuristic_benchmark.json",
        help="Output JSON file.",
    )
    args = parser.parse_args()

    manifest = json.loads((PROJECT_ROOT / args.manifest).resolve().read_text(encoding="utf-8"))
    selected_instances = sorted(
        manifest["selected_instances"],
        key=lambda row: (row["n_vars"], row["n_cons"], row["instance_name"]),
    )[: args.instance_limit]

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for instance_row in selected_instances:
        lp = load_lp_problem(instance_row["copied_path"])
        print(f"\n=== {instance_row['instance_name']} ===")

        per_solver_runs: dict[str, list[dict[str, Any]]] = {solver: [] for solver in args.solvers}
        for solver_name in args.solvers:
            for seed in args.seeds:
                metrics = run_solver(
                    solver_name=solver_name,
                    lp=lp,
                    seed=seed,
                    pdhg_max_iter=args.pdhg_max_iter,
                    pdhg_population_size=args.pdhg_population_size,
                    evo_population_size=args.evo_population_size,
                    evo_max_evals=args.evo_max_evals,
                    slpso_max_iter=args.slpso_max_iter,
                    quantum_use_two_phase=args.quantum_use_two_phase,
                    quantum_phase1_ratio=args.quantum_phase1_ratio,
                    quantum_phase1_weight=args.quantum_phase1_weight,
                    quantum_tunnel_interval=args.quantum_tunnel_interval,
                    quantum_measure_interval=args.quantum_measure_interval,
                    quantum_measurement_samples=args.quantum_measurement_samples,
                )
                row = {
                    "instance_name": instance_row["instance_name"],
                    "solver": solver_name,
                    "seed": seed,
                    "reference_obj": instance_row.get("reference_obj"),
                    "reference_error_pct": objective_error_pct(
                        metrics.get("objective_original"),
                        instance_row.get("reference_obj"),
                    ) if metrics.get("is_feasible") else None,
                    **metrics,
                }
                per_solver_runs[solver_name].append(row)
                raw_rows.append(row)

                print(
                    f"  {solver_name} seed={seed}: feas={row['is_feasible']} "
                    f"viol={row['primal_violation']:.3e} t={row['time_seconds']:.2f}s"
                )

        for solver_name in args.solvers:
            summary = aggregate_solver_rows(per_solver_runs[solver_name])
            summary_rows.append(
                {
                    "instance_name": instance_row["instance_name"],
                    "n_vars": instance_row["n_vars"],
                    "n_cons": instance_row["n_cons"],
                    "solver": solver_name,
                    **summary,
                }
            )

    output_path = (PROJECT_ROOT / args.output).resolve()
    payload = {
        "config": {
            "instance_limit": args.instance_limit,
            "pdhg_max_iter": args.pdhg_max_iter,
            "pdhg_population_size": args.pdhg_population_size,
            "quantum_use_two_phase": args.quantum_use_two_phase,
            "quantum_phase1_ratio": args.quantum_phase1_ratio,
            "quantum_phase1_weight": args.quantum_phase1_weight,
            "quantum_tunnel_interval": args.quantum_tunnel_interval,
            "quantum_measure_interval": args.quantum_measure_interval,
            "quantum_measurement_samples": args.quantum_measurement_samples,
            "evo_population_size": args.evo_population_size,
            "evo_max_evals": args.evo_max_evals,
            "slpso_max_iter": args.slpso_max_iter,
            "seeds": args.seeds,
            "solvers": args.solvers,
        },
        "instances": selected_instances,
        "summary_rows": summary_rows,
        "raw_rows": raw_rows,
    }
    write_json(output_path, payload)
    write_csv(output_path.with_suffix(".summary.csv"), summary_rows)
    write_csv(output_path.with_suffix(".runs.csv"), raw_rows)

    print("\nMetaheuristic benchmark complete.")
    print(f"  JSON: {output_path}")


if __name__ == "__main__":
    main()
