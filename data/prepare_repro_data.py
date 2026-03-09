#!/usr/bin/env python3
"""Prepare the local data layout for the Plan 1 reproducibility repository."""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "data" / "miplib2017"
OFFICIAL_DIR = PROJECT_ROOT / "data" / "miplib2017_sec" / "instances"
MIPLIB_2010_BASE_URL = "https://miplib2010.zib.de/miplib2/miplib"
MIPLIB_2017_BASE_URL = "https://miplib.zib.de/WebData/instances"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HP-PDHG-Repro/1.0)"}

MAIN_BENCHMARK_INSTANCES = ["p0033", "p0201", "p0282"]
OFFICIAL_SUBSET_INSTANCES = [
    "glass4",
    "neos-3046615-murg",
    "neos-2657525-crna",
    "ic97_potential",
    "cod105",
    "gmu-35-40",
    "neos-4954672-berkel",
    "mcsched",
    "gmu-35-50",
    "50v-10",
    "p200x1188c",
    "beasleyC3",
]

MAIN_BENCHMARK_URLS = {
    "p0033": f"{MIPLIB_2010_BASE_URL}/p0033.mps.gz",
    "p0201": f"{MIPLIB_2010_BASE_URL}/p0201.mps.gz",
    "p0282": f"{MIPLIB_2010_BASE_URL}/p0282.mps.gz",
}


def is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def download(url: str, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"[skip] {target}")
        return
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    tmp_target = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, tmp_target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if target.suffix == ".gz" and not is_gzip_file(tmp_target):
        preview = tmp_target.read_text(encoding="utf-8", errors="ignore")[:160]
        tmp_target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a gzip archive: {url}\nPreview: {preview!r}")
    tmp_target.replace(target)
    print(f"[saved] {target}")


def gunzip_file(archive_path: Path, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"[skip] {output_path}")
        return
    print(f"[gunzip] {archive_path.name} -> {output_path.name}")
    with gzip.open(archive_path, "rb") as source, output_path.open("wb") as sink:
        shutil.copyfileobj(source, sink)


def prepare_main(force: bool = False) -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    for name in MAIN_BENCHMARK_INSTANCES:
        archive = MAIN_DIR / f"{name}.mps.gz"
        plain = MAIN_DIR / f"{name}.mps"
        download(MAIN_BENCHMARK_URLS[name], archive, force=force)
        gunzip_file(archive, plain, force=force)


def prepare_official_subset(force: bool = False) -> None:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_SUBSET_INSTANCES:
        archive = OFFICIAL_DIR / f"{name}.mps.gz"
        download(f"{MIPLIB_2017_BASE_URL}/{name}.mps.gz", archive, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-benchmark", action="store_true", help="Download p0033/p0201/p0282.")
    parser.add_argument("--official-subset", action="store_true", help="Download the 12-instance official subset.")
    parser.add_argument("--paper-suite", action="store_true", help="Prepare all paper-relevant official instances.")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files.")
    args = parser.parse_args()

    selected_any = args.main_benchmark or args.official_subset or args.paper_suite
    if not selected_any:
        args.paper_suite = True

    print("=" * 72)
    print("Plan 1 data preparation")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"MIPLIB 2010:  {MIPLIB_2010_BASE_URL}")
    print(f"MIPLIB 2017:  {MIPLIB_2017_BASE_URL}")

    if args.paper_suite or args.main_benchmark:
        prepare_main(force=args.force)
    if args.paper_suite or args.official_subset:
        prepare_official_subset(force=args.force)

    print("=" * 72)
    print("Done.")
    print("Custom synthetic instance `knapsack_50.mps` is already tracked in the repository.")
    print("=" * 72)


if __name__ == "__main__":
    main()
