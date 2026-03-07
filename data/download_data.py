#!/usr/bin/env python3
"""
Download MIPLIB 2017 test instances for reproduction.

This script downloads the p0282 instance from MIPLIB 2017.
"""

import os
import urllib.request
from pathlib import Path


MIPLIB_URL = "https://miplib.zib.de/WebData/instances/"

INSTANCES = {
    "p0282.mps.gz": "p0282.mps.gz",
    # Add more instances as needed
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, str(dest_path))
        print(f"  Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Download all test instances."""
    data_dir = Path(__file__).parent

    print("=" * 70)
    print("Downloading MIPLIB 2017 test instances")
    print("=" * 70)

    for filename in INSTANCES:
        dest_path = data_dir / filename
        if dest_path.exists():
            print(f"{filename} already exists, skipping.")
            continue

        url = MIPLIB_URL + filename
        download_file(url, dest_path)

        # Decompress if .gz
        if filename.endswith('.gz'):
            import gzip
            mps_path = dest_path.with_suffix('')
            print(f"  Decompressing to {mps_path.name}...")
            with gzip.open(dest_path, 'rb') as f_in:
                with open(mps_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"  Decompressed.")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
