# Data Preparation

This repository does not track the official MIPLIB instances used in the paper.

## What is kept locally in Git

- `miplib2017/knapsack_50.mps`: small custom synthetic benchmark used in the audited paper suite

## What is not kept in Git

- official MIPLIB 2017 instances for the main audited benchmark
- official MIPLIB 2017 instances for the 12-instance extension subset

## Prepare the required local data

From the repository root:

```bash
uv run python data/prepare_repro_data.py --paper-suite
```

This will populate:

- `data/miplib2017/`
- `data/miplib2017_sec/instances/`

using files downloaded from the official ZIB-hosted MIPLIB archives.

The audited main suite (`p0033`, `p0201`, `p0282`) is fetched from the legacy MIPLIB archive, while the 12-instance SEC extension is fetched from the MIPLIB 2017 site.
