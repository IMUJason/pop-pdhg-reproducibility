#!/usr/bin/env python
"""
Select and copy a reproducible SEC-oriented MIPLIB 2017 benchmark subset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from miplib_sec_utils import (
    DEFAULT_SELECTION_RULES,
    copy_instances,
    load_official_instance_names,
    parse_solu_file,
    resolve_instance_file,
    scan_instance_metadata,
    size_stratified_selection,
    structural_filter,
    summarize_catalog,
    write_json,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a simple CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = []
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
        "--search-dir",
        action="append",
        required=True,
        help="Directory containing MPS or MPS.GZ files; can be repeated.",
    )
    parser.add_argument(
        "--solu",
        required=True,
        help="Path to the official MIPLIB .solu file.",
    )
    parser.add_argument(
        "--testset",
        required=True,
        help="Path to the official benchmark-v2.test file.",
    )
    parser.add_argument(
        "--target-dir",
        default="data/miplib2017_sec/instances",
        help="Local directory where selected instances will be copied.",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Directory for catalog and manifest outputs.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/miplib2017_sec",
        help="Directory for copied metadata artifacts.",
    )
    parser.add_argument("--n-select", type=int, default=12, help="Number of instances to keep.")
    parser.add_argument("--min-vars", type=int, default=DEFAULT_SELECTION_RULES["min_vars"])
    parser.add_argument("--max-vars", type=int, default=DEFAULT_SELECTION_RULES["max_vars"])
    parser.add_argument("--min-cons", type=int, default=DEFAULT_SELECTION_RULES["min_cons"])
    parser.add_argument("--max-cons", type=int, default=DEFAULT_SELECTION_RULES["max_cons"])
    parser.add_argument(
        "--min-integer-vars",
        type=int,
        default=DEFAULT_SELECTION_RULES["min_integer_vars"],
    )
    parser.add_argument(
        "--min-integer-ratio",
        type=float,
        default=DEFAULT_SELECTION_RULES["min_integer_ratio"],
    )
    parser.add_argument("--max-nnz", type=int, default=DEFAULT_SELECTION_RULES["max_nnz"])
    parser.add_argument("--max-density", type=float, default=DEFAULT_SELECTION_RULES["max_density"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    results_dir = (project_root / args.results_dir).resolve()
    metadata_dir = (project_root / args.metadata_dir).resolve()
    target_dir = (project_root / args.target_dir).resolve()

    rules = {
        "min_vars": args.min_vars,
        "max_vars": args.max_vars,
        "min_cons": args.min_cons,
        "max_cons": args.max_cons,
        "min_integer_vars": args.min_integer_vars,
        "min_integer_ratio": args.min_integer_ratio,
        "max_nnz": args.max_nnz,
        "max_density": args.max_density,
    }

    print("Loading official metadata...")
    solu_entries = parse_solu_file(args.solu)
    official_names = load_official_instance_names(args.testset)

    all_rows = []
    for index, instance_name in enumerate(official_names, start=1):
        instance_path = resolve_instance_file(instance_name, args.search_dir)
        if not instance_path:
            all_rows.append(
                {
                    "instance_name": instance_name,
                    "source_path": None,
                    "missing": True,
                }
            )
            continue

        row = scan_instance_metadata(instance_name, instance_path, solu_entries)
        is_selected_pool, reasons = structural_filter(row, rules)
        row["passes_structural_filter"] = is_selected_pool
        row["filter_reasons"] = ";".join(reasons)
        all_rows.append(row)

        if index % 40 == 0:
            print(f"  scanned {index}/{len(official_names)} instances")

    catalog_rows = sorted(all_rows, key=lambda item: item["instance_name"])
    filtered_rows = [row for row in catalog_rows if row.get("passes_structural_filter")]
    selected_rows = size_stratified_selection(filtered_rows, args.n_select)
    copied_paths = copy_instances(selected_rows, target_dir)

    for row, copied_path in zip(selected_rows, copied_paths):
        row["copied_path"] = copied_path

    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    catalog_json = results_dir / "sec_instance_catalog.json"
    catalog_csv = results_dir / "sec_instance_catalog.csv"
    filtered_json = results_dir / "sec_candidate_pool.json"
    filtered_csv = results_dir / "sec_candidate_pool.csv"
    manifest_json = results_dir / "sec_selected_instances.json"
    manifest_csv = results_dir / "sec_selected_instances.csv"

    write_json(catalog_json, catalog_rows)
    write_csv(catalog_csv, catalog_rows)
    write_json(filtered_json, filtered_rows)
    write_csv(filtered_csv, filtered_rows)

    manifest = {
        "selection_rules": rules,
        "benchmark_testset": str(Path(args.testset).resolve()),
        "solu_file": str(Path(args.solu).resolve()),
        "search_dirs": [str(Path(path).resolve()) for path in args.search_dir],
        "catalog_summary": summarize_catalog(catalog_rows),
        "candidate_summary": summarize_catalog(filtered_rows),
        "selected_summary": summarize_catalog(selected_rows),
        "selected_instances": selected_rows,
    }
    write_json(manifest_json, manifest)
    write_csv(manifest_csv, selected_rows)

    write_json(metadata_dir / "miplib2017_v36_solu.json", solu_entries)
    (metadata_dir / "benchmark_v2_instances.txt").write_text(
        "\n".join(official_names) + "\n",
        encoding="utf-8",
    )

    print("\nSelection complete.")
    print(f"  Catalog:   {catalog_json}")
    print(f"  Candidates:{filtered_json}")
    print(f"  Manifest:  {manifest_json}")
    print(f"  Copied to: {target_dir}")
    print("\nSelected instances:")
    for row in selected_rows:
        ref = row.get("reference_obj")
        ref_text = f"{ref:.6g}" if isinstance(ref, float) else str(ref)
        print(
            f"  - {row['instance_name']}: vars={row['n_vars']}, cons={row['n_cons']}, "
            f"int_ratio={row['integer_ratio']:.2f}, ref={ref_text}"
        )


if __name__ == "__main__":
    main()
