# `data/miplib2017`

This directory is used by the main audited benchmark, ablation study, and sensitivity study.

Tracked in Git:

- `knapsack_50.mps` (custom synthetic benchmark)

Not tracked in Git:

- `p0033.mps`
- `p0201.mps`
- `p0282.mps`
- any downloaded `.mps.gz` archives

To restore the official files locally, run:

```bash
uv run python data/prepare_repro_data.py --main-benchmark
```

These three audited instances are downloaded from the official legacy ZIB MIPLIB archive:

- `https://miplib2010.zib.de/miplib2/miplib/p0033.mps.gz`
- `https://miplib2010.zib.de/miplib2/miplib/p0201.mps.gz`
- `https://miplib2010.zib.de/miplib2/miplib/p0282.mps.gz`
