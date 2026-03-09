#!/usr/bin/env python
"""
Statistical analysis for the SEC-oriented benchmark results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from miplib_sec_utils import write_json


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


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "p_value") -> list[dict[str, Any]]:
    """Apply Holm correction to a list of rows containing raw p-values."""
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float("inf") if item[1].get(p_key) is None else item[1][p_key],
    )
    m = len(rows)
    adjusted = [None] * m
    running_max = 0.0

    for rank, (original_idx, row) in enumerate(ordered):
        p_value = row.get(p_key)
        if p_value is None:
            adjusted[original_idx] = None
            continue
        candidate = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max

    result = []
    for row, p_adj in zip(rows, adjusted):
        updated = dict(row)
        updated["holm_adjusted_p"] = p_adj
        result.append(updated)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_extended(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["instance_summaries"]
    standard = np.array([row["standard_pdhg"]["mean_primal_violation"] for row in rows], dtype=float)
    quantum = np.array([row["quantum_pdhg"]["mean_primal_violation"] for row in rows], dtype=float)

    statistic, p_value = wilcoxon(quantum, standard, alternative="less", zero_method="wilcox")

    paired_rows = []
    quantum_better = 0
    standard_better = 0
    ties = 0
    for row, std_val, q_val in zip(rows, standard, quantum):
        relation = "tie"
        if q_val < std_val:
            relation = "quantum_better"
            quantum_better += 1
        elif q_val > std_val:
            relation = "standard_better"
            standard_better += 1
        else:
            ties += 1

        paired_rows.append(
            {
                "instance_name": row["instance_name"],
                "standard_mean_primal_violation": float(std_val),
                "quantum_mean_primal_violation": float(q_val),
                "quantum_violation_reduction_pct": row["quantum_violation_reduction_pct"],
                "winner": relation,
            }
        )

    return {
        "wilcoxon_quantum_less_than_standard": {
            "statistic": float(statistic),
            "p_value": float(p_value),
        },
        "paired_rows": paired_rows,
        "quantum_better_instances": quantum_better,
        "standard_better_instances": standard_better,
        "ties": ties,
    }


def analyze_meta(payload: dict[str, Any]) -> dict[str, Any]:
    summary_rows = payload["summary_rows"]
    instances = sorted({row["instance_name"] for row in summary_rows})
    solvers = payload["config"]["solvers"]

    matrix: dict[str, list[float]] = {solver: [] for solver in solvers}
    per_instance_ranks = []

    for instance_name in instances:
        instance_rows = {
            row["solver"]: row
            for row in summary_rows
            if row["instance_name"] == instance_name
        }
        values = [float(instance_rows[solver]["mean_primal_violation"]) for solver in solvers]
        ranks = rankdata(values, method="average")

        rank_row = {"instance_name": instance_name}
        for solver, value, rank in zip(solvers, values, ranks):
            matrix[solver].append(value)
            rank_row[f"{solver}_mean_primal_violation"] = value
            rank_row[f"{solver}_rank"] = float(rank)
        per_instance_ranks.append(rank_row)

    friedman_stat, friedman_p = friedmanchisquare(*[matrix[solver] for solver in solvers])

    average_ranks = []
    for solver in solvers:
        solver_ranks = [
            row[f"{solver}_rank"]
            for row in per_instance_ranks
            if f"{solver}_rank" in row
        ]
        average_ranks.append(
            {
                "solver": solver,
                "average_rank": float(np.mean(solver_ranks)),
                "mean_primal_violation": float(np.mean(matrix[solver])),
            }
        )

    pairwise = []
    quantum_values = np.array(matrix["quantum_pdhg"], dtype=float)
    for solver in solvers:
        if solver == "quantum_pdhg":
            continue
        comparator = np.array(matrix[solver], dtype=float)
        statistic, p_value = wilcoxon(quantum_values, comparator, alternative="less", zero_method="wilcox")
        pairwise.append(
            {
                "baseline_solver": solver,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "quantum_mean_violation": float(np.mean(quantum_values)),
                "baseline_mean_violation": float(np.mean(comparator)),
            }
        )

    pairwise = holm_adjust(pairwise)

    return {
        "friedman_test": {
            "statistic": float(friedman_stat),
            "p_value": float(friedman_p),
        },
        "average_ranks": sorted(average_ranks, key=lambda row: row["average_rank"]),
        "pairwise_quantum_vs_baselines": pairwise,
        "per_instance_ranks": per_instance_ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extended",
        default="experiments/results/sec_extended_benchmark.json",
        help="Extended benchmark JSON file.",
    )
    parser.add_argument(
        "--meta",
        default="experiments/results/sec_metaheuristic_benchmark.json",
        help="Metaheuristic benchmark JSON file.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/sec_statistics.json",
        help="Statistics JSON output.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    extended_path = (root / args.extended).resolve()
    meta_path = (root / args.meta).resolve()
    output_path = (root / args.output).resolve()

    extended_payload = load_json(extended_path)
    meta_payload = load_json(meta_path)

    result = {
        "extended": analyze_extended(extended_payload),
        "meta": analyze_meta(meta_payload),
    }
    write_json(output_path, result)
    write_csv(output_path.with_suffix(".extended.csv"), result["extended"]["paired_rows"])
    write_csv(output_path.with_suffix(".meta_ranks.csv"), result["meta"]["average_ranks"])
    write_csv(output_path.with_suffix(".meta_pairwise.csv"), result["meta"]["pairwise_quantum_vs_baselines"])

    print(f"Saved statistics to: {output_path}")


if __name__ == "__main__":
    main()

