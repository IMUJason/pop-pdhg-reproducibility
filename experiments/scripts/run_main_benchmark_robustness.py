#!/usr/bin/env python
"""
Run a repeated-seed robustness study on the four audited paper instances.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from miplib_benchmark import find_instance_file, load_instance_with_scip, solve_with_quantum_pdhg, solve_with_standard_pdhg


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def objective_gap_pct(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    scale = max(abs(reference), 1e-9)
    return abs(value - reference) / scale * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["p0033", "p0201", "p0282", "knapsack_50"],
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101])
    parser.add_argument("--quantum-use-two-phase", action="store_true")
    parser.add_argument("--quantum-phase1-ratio", type=float, default=0.4)
    parser.add_argument("--quantum-phase1-weight", type=float, default=10.0)
    parser.add_argument("--quantum-tunnel-interval", type=int, default=50)
    parser.add_argument("--quantum-measure-interval", type=int, default=100)
    parser.add_argument("--quantum-measurement-samples", type=int, default=5)
    parser.add_argument(
        "--output",
        default="experiments/results/paper_benchmark_robustness.json",
    )
    args = parser.parse_args()

    data_dirs = [str(PROJECT_ROOT / "data" / "miplib2017")]
    raw_rows: list[dict[str, Any]] = []
    instance_summaries: list[dict[str, Any]] = []
    paired_primal_rows: list[tuple[float, float]] = []

    discordant_quantum = 0
    discordant_standard = 0

    for instance_name in args.instances:
        instance_path = find_instance_file(instance_name, data_dirs)
        if not instance_path:
            raise FileNotFoundError(f"Instance not found: {instance_name}")

        loaded = load_instance_with_scip(instance_path)
        if loaded is None:
            raise RuntimeError(f"Failed to load instance: {instance_name}")
        A, b, c, lb, ub, integer_vars, constraint_sense = loaded

        gurobi_result = None
        try:
            from miplib_benchmark import solve_with_gurobi

            gurobi_result = solve_with_gurobi(
                instance_path,
                time_limit=60,
            )
        except Exception:
            gurobi_result = None
        gurobi_obj = gurobi_result.obj_value if gurobi_result else None

        instance_runs = {"standard_pdhg": [], "quantum_pdhg": []}
        print(f"\n=== {instance_name} ===")
        for seed in args.seeds:
            std = solve_with_standard_pdhg(
                A,
                b,
                c,
                lb,
                ub,
                integer_vars,
                constraint_sense,
                max_iter=args.max_iter,
                population_size=args.population_size,
                seed=seed,
            )
            quantum = solve_with_quantum_pdhg(
                A,
                b,
                c,
                lb,
                ub,
                integer_vars,
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

            if std.is_feasible and not quantum.is_feasible:
                discordant_standard += 1
            elif quantum.is_feasible and not std.is_feasible:
                discordant_quantum += 1

            paired_primal_rows.append((float(std.primal_violation), float(quantum.primal_violation)))

            for solver_name, result in (("standard_pdhg", std), ("quantum_pdhg", quantum)):
                row = {
                    "instance_name": instance_name,
                    "solver": solver_name,
                    "seed": seed,
                    "objective": None if result.obj_value is None else float(result.obj_value),
                    "gap_to_gurobi_pct": objective_gap_pct(result.obj_value, gurobi_obj),
                    "time_seconds": float(result.time_seconds),
                    "iterations": int(result.iterations),
                    "status": result.status,
                    "is_feasible": bool(result.is_feasible),
                    "primal_violation": float(result.primal_violation),
                    "integrality_violation": float(result.integrality_violation),
                }
                raw_rows.append(row)
                instance_runs[solver_name].append(row)

            print(
                f"  seed={seed}: std feas={std.is_feasible} viol={std.primal_violation:.3e}, "
                f"q feas={quantum.is_feasible} viol={quantum.primal_violation:.3e}"
            )

        summary = {"instance_name": instance_name}
        for solver_name, rows in instance_runs.items():
            feasible_rows = [row for row in rows if row["is_feasible"]]
            primal = [float(row["primal_violation"]) for row in rows]
            gaps = [float(row["gap_to_gurobi_pct"]) for row in feasible_rows if row["gap_to_gurobi_pct"] is not None]
            times = [float(row["time_seconds"]) for row in rows]
            summary[solver_name] = {
                "n_runs": len(rows),
                "feasible_runs": len(feasible_rows),
                "feasibility_rate": len(feasible_rows) / float(len(rows)) if rows else 0.0,
                "mean_primal_violation": statistics.mean(primal) if primal else None,
                "median_primal_violation": statistics.median(primal) if primal else None,
                "mean_time_seconds": statistics.mean(times) if times else None,
                "mean_gap_to_gurobi_pct_feasible": statistics.mean(gaps) if gaps else None,
            }
        instance_summaries.append(summary)

    standard_primal = np.array([row[0] for row in paired_primal_rows], dtype=float)
    quantum_primal = np.array([row[1] for row in paired_primal_rows], dtype=float)
    wilcoxon_stat, wilcoxon_p = wilcoxon(
        quantum_primal,
        standard_primal,
        alternative="less",
        zero_method="wilcox",
    )

    discordant_total = discordant_quantum + discordant_standard
    if discordant_total > 0:
        mcnemar = binomtest(discordant_quantum, n=discordant_total, p=0.5, alternative="greater")
        mcnemar_payload = {
            "discordant_quantum_better": discordant_quantum,
            "discordant_standard_better": discordant_standard,
            "p_value": float(mcnemar.pvalue),
        }
    else:
        mcnemar_payload = {
            "discordant_quantum_better": 0,
            "discordant_standard_better": 0,
            "p_value": None,
        }

    payload = {
        "config": {
            "instances": args.instances,
            "max_iter": args.max_iter,
            "population_size": args.population_size,
            "seeds": args.seeds,
            "quantum_use_two_phase": args.quantum_use_two_phase,
            "quantum_phase1_ratio": args.quantum_phase1_ratio,
            "quantum_phase1_weight": args.quantum_phase1_weight,
            "quantum_tunnel_interval": args.quantum_tunnel_interval,
            "quantum_measure_interval": args.quantum_measure_interval,
            "quantum_measurement_samples": args.quantum_measurement_samples,
        },
        "instance_summaries": instance_summaries,
        "overall": {
            "standard_total_feasible_runs": int(sum(row["standard_pdhg"]["feasible_runs"] for row in instance_summaries)),
            "quantum_total_feasible_runs": int(sum(row["quantum_pdhg"]["feasible_runs"] for row in instance_summaries)),
            "paired_wilcoxon_quantum_less_violation": {
                "statistic": float(wilcoxon_stat),
                "p_value": float(wilcoxon_p),
            },
            "paired_feasibility_exact_sign_test": mcnemar_payload,
        },
        "raw_rows": raw_rows,
    }

    output_path = (PROJECT_ROOT / args.output).resolve()
    write_json(output_path, payload)
    write_csv(output_path.with_suffix(".summary.csv"), instance_summaries)
    write_csv(output_path.with_suffix(".runs.csv"), raw_rows)
    print(f"\nSaved robustness study to: {output_path}")


if __name__ == "__main__":
    main()
