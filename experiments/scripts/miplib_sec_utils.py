#!/usr/bin/env python
"""
Utilities for the SEC-oriented MIPLIB 2017 benchmark workflow.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.mps_reader import LPData, read_mps

SOLU_PATTERN = re.compile(r"^\s*(=\w+=)\s+(\S+)(?:\s+(.*?))?\s*$")

DEFAULT_SELECTION_RULES = {
    "min_vars": 100,
    "max_vars": 4000,
    "min_cons": 50,
    "max_cons": 7000,
    "min_integer_vars": 20,
    "min_integer_ratio": 0.30,
    "max_nnz": 100000,
    "max_density": 0.12,
    "allowed_statuses": {"=opt=", "=best="},
}


def to_builtin(value: Any) -> Any:
    """Convert numpy-heavy objects to JSON-safe builtins."""
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, sort_keys=False)


def parse_solu_file(solu_path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse an official MIPLIB `.solu` file."""
    solu_path = Path(solu_path)
    entries: dict[str, dict[str, Any]] = {}

    with solu_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = SOLU_PATTERN.match(raw_line)
            if not match:
                continue

            status, name, raw_value = match.groups()
            value = None
            if raw_value:
                raw_value = raw_value.strip()
                try:
                    value = float(raw_value)
                except ValueError:
                    value = None

            entries[name] = {
                "status": status,
                "value": value,
                "raw_value": raw_value.strip() if raw_value else None,
            }

    return entries


def load_official_instance_names(testset_path: str | Path) -> list[str]:
    """Load instance names from an official `.test` set file."""
    testset_path = Path(testset_path)
    instances = []
    for raw_line in testset_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = Path(line).name
        name = name.replace(".mps.gz", "").replace(".mps", "")
        instances.append(name)
    return instances


def resolve_instance_file(instance_name: str, search_dirs: Iterable[str | Path]) -> Optional[Path]:
    """Resolve an instance file in the provided directories."""
    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        for suffix in (".mps", ".mps.gz"):
            candidate = search_dir / f"{instance_name}{suffix}"
            if candidate.exists():
                return candidate
    return None


def normalize_constraint_sense(sense: Iterable[str]) -> list[str]:
    """Normalize Gurobi/LP senses to `L/G/E`."""
    mapping = {"<": "L", ">": "G", "=": "E", "L": "L", "G": "G", "E": "E"}
    return [mapping.get(item, "L") for item in sense]


def load_lp_problem(instance_path: str | Path) -> LPData:
    """Load an MPS/MPS.GZ instance with Gurobi-backed parsing."""
    return read_mps(str(instance_path), use_gurobi=True)


def build_leq_relaxation(lp: LPData) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, float]:
    """
    Convert a mixed-sense LP/MIP problem to a `<=` system for metaheuristics.

    Returns:
        `(A_leq, b_leq, c_min, objective_multiplier)` where the original
        objective value is `objective_multiplier * (c_min @ x)`.
    """
    rows = []
    rhs = []

    for row_idx, sense in enumerate(lp.sense):
        row = lp.A.getrow(row_idx)
        if sense == "<":
            rows.append(row)
            rhs.append(lp.b[row_idx])
        elif sense == ">":
            rows.append(-row)
            rhs.append(-lp.b[row_idx])
        else:
            rows.append(row)
            rhs.append(lp.b[row_idx])
            rows.append(-row)
            rhs.append(-lp.b[row_idx])

    a_leq = sparse.vstack(rows, format="csr")
    b_leq = np.asarray(rhs, dtype=np.float64)
    c_min = np.asarray(lp.c, dtype=np.float64)
    objective_multiplier = 1.0
    if lp.obj_sense == "max":
        c_min = -c_min
        objective_multiplier = -1.0

    return a_leq, b_leq, c_min, objective_multiplier


def compute_original_objective(lp: LPData, x: np.ndarray) -> float:
    """Compute the objective value in the original problem sense."""
    value = float(np.asarray(lp.c, dtype=np.float64) @ np.asarray(x, dtype=np.float64))
    return value if lp.obj_sense == "min" else value


def scan_instance_metadata(
    instance_name: str,
    instance_path: str | Path,
    solu_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Scan structural metadata for one instance."""
    lp = load_lp_problem(instance_path)
    density = lp.nnz / float(lp.m * lp.n) if lp.m > 0 and lp.n > 0 else 0.0
    n_finite_lb = int(np.isfinite(lp.lb).sum())
    n_finite_ub = int(np.isfinite(lp.ub).sum())
    finite_bound_ratio = (n_finite_lb + n_finite_ub) / float(2 * lp.n) if lp.n else 1.0
    eq_count = int(sum(1 for value in lp.sense if value == "="))
    ge_count = int(sum(1 for value in lp.sense if value == ">"))
    le_count = int(sum(1 for value in lp.sense if value == "<"))
    integer_ratio = len(lp.integer_vars) / float(lp.n) if lp.n else 0.0

    solu_entry = solu_entries.get(instance_name, {})
    row = {
        "instance_name": instance_name,
        "source_path": str(instance_path),
        "n_vars": int(lp.n),
        "n_cons": int(lp.m),
        "nnz": int(lp.nnz),
        "density": density,
        "n_integer_vars": int(len(lp.integer_vars)),
        "n_binary_vars": int(len(lp.binary_vars)),
        "n_continuous_vars": int(lp.n - len(lp.integer_vars)),
        "integer_ratio": integer_ratio,
        "finite_bound_ratio": finite_bound_ratio,
        "n_finite_lb": n_finite_lb,
        "n_finite_ub": n_finite_ub,
        "n_eq": eq_count,
        "n_ge": ge_count,
        "n_le": le_count,
        "obj_sense": lp.obj_sense,
        "solu_status": solu_entry.get("status"),
        "reference_obj": solu_entry.get("value"),
        "reference_obj_raw": solu_entry.get("raw_value"),
    }
    row["selection_score"] = suitability_score(row)
    return row


def structural_filter(
    row: dict[str, Any],
    rules: Optional[dict[str, Any]] = None,
) -> tuple[bool, list[str]]:
    """Return whether an instance satisfies the SEC benchmark filters."""
    rules = {**DEFAULT_SELECTION_RULES, **(rules or {})}
    reasons = []

    if row.get("solu_status") not in rules["allowed_statuses"]:
        reasons.append("unknown_reference")
    if row["n_vars"] < rules["min_vars"] or row["n_vars"] > rules["max_vars"]:
        reasons.append("vars_out_of_range")
    if row["n_cons"] < rules["min_cons"] or row["n_cons"] > rules["max_cons"]:
        reasons.append("cons_out_of_range")
    if row["n_integer_vars"] < rules["min_integer_vars"]:
        reasons.append("too_few_integer_vars")
    if row["integer_ratio"] < rules["min_integer_ratio"]:
        reasons.append("integer_ratio_too_low")
    if row["nnz"] > rules["max_nnz"]:
        reasons.append("too_many_nonzeros")
    if row["density"] > rules["max_density"]:
        reasons.append("too_dense")

    return (len(reasons) == 0, reasons)


def suitability_score(row: dict[str, Any]) -> float:
    """Score structural suitability for the SEC study."""
    density_penalty = min(row["density"] / 0.05, 1.0)
    size_penalty = min(max(row["n_vars"] / 4000.0, row["n_cons"] / 7000.0), 1.0)
    eq_penalty = row["n_eq"] / max(row["n_cons"], 1)
    status_bonus = 0.2 if row.get("solu_status") == "=opt=" else 0.0

    return (
        3.0 * row["integer_ratio"]
        + 2.0 * row["finite_bound_ratio"]
        - 0.8 * density_penalty
        - 0.5 * size_penalty
        - 0.4 * eq_penalty
        + status_bonus
    )


def size_stratified_selection(rows: list[dict[str, Any]], n_select: int) -> list[dict[str, Any]]:
    """Select a size-diverse shortlist from a filtered catalog."""
    if len(rows) <= n_select:
        return sorted(rows, key=lambda item: (item["n_vars"], item["n_cons"], item["instance_name"]))

    ranked = sorted(
        rows,
        key=lambda item: (item["n_vars"], item["n_cons"], -item["selection_score"], item["instance_name"]),
    )

    buckets = np.array_split(np.asarray(ranked, dtype=object), n_select)
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}

    for bucket in buckets:
        bucket_rows = list(bucket)
        ordered_bucket = sorted(
            bucket_rows,
            key=lambda item: (
                -item["selection_score"],
                item["nnz"],
                -item["integer_ratio"],
                item["instance_name"],
            ),
        )

        chosen = ordered_bucket[0]
        for candidate in ordered_bucket:
            family = instance_family(candidate["instance_name"])
            if family_counts.get(family, 0) < 2:
                chosen = candidate
                break

        selected.append(chosen)
        family = instance_family(chosen["instance_name"])
        family_counts[family] = family_counts.get(family, 0) + 1

    return sorted(selected, key=lambda item: (item["n_vars"], item["n_cons"], item["instance_name"]))


def instance_family(instance_name: str) -> str:
    """Infer a coarse instance family from the filename stem."""
    family = re.split(r"[-_\d]", instance_name, maxsplit=1)[0]
    return family or instance_name


def copy_instances(selected_rows: list[dict[str, Any]], target_dir: str | Path) -> list[str]:
    """Copy selected instances into the local Plan 1 data directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    keep_names = set()

    for row in selected_rows:
        source = Path(row["source_path"])
        destination = target_dir / source.name
        keep_names.add(destination.name)
        if not destination.exists() or source.stat().st_size != destination.stat().st_size:
            shutil.copy2(source, destination)
        copied.append(str(destination))

    for existing in target_dir.iterdir():
        if existing.is_file() and existing.name not in keep_names:
            existing.unlink()

    return copied


def summarize_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a catalog for reporting."""
    if not rows:
        return {
            "count": 0,
            "min_vars": None,
            "max_vars": None,
            "median_vars": None,
            "min_cons": None,
            "max_cons": None,
            "median_cons": None,
            "median_integer_ratio": None,
        }

    n_vars = sorted(row["n_vars"] for row in rows)
    n_cons = sorted(row["n_cons"] for row in rows)
    int_ratio = sorted(row["integer_ratio"] for row in rows)
    return {
        "count": len(rows),
        "min_vars": n_vars[0],
        "max_vars": n_vars[-1],
        "median_vars": int(n_vars[len(n_vars) // 2]),
        "min_cons": n_cons[0],
        "max_cons": n_cons[-1],
        "median_cons": int(n_cons[len(n_cons) // 2]),
        "median_integer_ratio": float(int_ratio[len(int_ratio) // 2]),
    }
