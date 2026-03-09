# `data/miplib2017_sec`

This directory is used by the official 12-instance MIPLIB 2017 extension benchmark.

The `instances/` directory is intentionally not tracked in Git. Populate it locally with:

```bash
uv run python data/prepare_repro_data.py --official-subset
```

The required instance names are hard-coded in the helper script to match the archived paper benchmark.
