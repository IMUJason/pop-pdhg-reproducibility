# HP-PDHG Plan 1 Reproducibility Repository

This directory is the minimal reproducibility workspace for the Plan 1 paper and codebase. It contains:

- the current solver implementation in `src/`
- the experiment scripts and archived result tables in `experiments/`
- the paper source in `paper/elsarticle/`
- lightweight tests in `tests/`

Official MIPLIB instances are **not version-controlled** in this repository. Readers can download them from the official MIPLIB website and place them in the expected local directories with the helper script in `data/prepare_repro_data.py`.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── src/                     # solver implementation
├── experiments/
│   ├── scripts/            # benchmark / ablation / sensitivity entry points
│   └── results/            # archived paper results used by the manuscript
├── data/
│   ├── prepare_repro_data.py
│   ├── miplib2017/         # main benchmark data directory
│   └── miplib2017_sec/     # official 12-instance extension directory
├── tests/
└── paper/elsarticle/       # current paper source
```

## Environment setup

Requirements:

- Python 3.10+
- `uv` for environment management
- Gurobi installed and licensed locally

Setup:

```bash
uv sync --extra dev
```

## Data policy

- `data/miplib2017/knapsack_50.mps` is kept in the repository because it is a small custom synthetic benchmark used in the paper.
- Official MIPLIB instances such as `p0033`, `p0201`, `p0282`, and the 12-instance extension subset are **not uploaded**.
- To prepare local data, run:

```bash
uv run python data/prepare_repro_data.py --paper-suite
```

This downloads the required official instances from the official MIPLIB websites:

- legacy library links page: `https://miplib.zib.de/links.html`
- MIPLIB 2010 archive home: `https://miplib2010.zib.de`
- official instance index: `https://miplib.zib.de/instances/index.html`
- MIPLIB 2017 download page: `https://miplib.zib.de/download.html`

The helper script fetches the main audited instances (`p0033`, `p0201`, `p0282`) from the legacy ZIB MIPLIB archive and the 12-instance extension subset from the MIPLIB 2017 instance server:

- `https://miplib2010.zib.de/miplib2/miplib/`
- `https://miplib.zib.de/WebData/instances/`

## Minimal verification

Run the lightweight self-check first:

```bash
uv run pytest tests/test_pdhg.py -v
```

Then run a paper-relevant smoke benchmark:

```bash
uv run python experiments/scripts/run_main_benchmark_robustness.py \
  --instances p0033 knapsack_50 \
  --seeds 11 17 \
  --max-iter 300 \
  --population-size 8
```

## Reproducing the paper workflow

### 1. Audited main benchmark robustness

```bash
uv run python experiments/scripts/run_main_benchmark_robustness.py
```

### 2. Ablation study

```bash
uv run python experiments/scripts/run_ablation_study.py
```

### 3. Parameter sensitivity

```bash
uv run python experiments/scripts/parameter_sensitivity.py --plot-only
```

### 4. Official MIPLIB 2017 extension

```bash
uv run python experiments/scripts/run_sec_extended_benchmark.py \
  --manifest experiments/results/sec_selected_instances.json
```

### 5. Population-based baseline comparison

```bash
uv run python experiments/scripts/run_sec_metaheuristics.py \
  --manifest experiments/results/sec_selected_instances.json
```

### 6. Compile the manuscript

```bash
cd paper/elsarticle
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Notes on reproducibility

- The archived `experiments/results/` files are the result tables and JSON/CSV artifacts used to build the current manuscript figures.
- The figure-generation script used by the paper lives in `paper/elsarticle/figures/make_figures.py`.
- The manuscript entry point is `paper/elsarticle/main.tex`.
- The current paper uses the ESWA-oriented framing, but the underlying implementation still contains some legacy names such as `Quantum Pop-PDHG` in code and archived results for continuity.

## Suggested starting points

- solver core: `src/population/quantum_pop_pdhg.py`
- audited benchmark: `experiments/scripts/run_main_benchmark_robustness.py`
- official extension: `experiments/scripts/run_sec_extended_benchmark.py`
- metaheuristic comparison: `experiments/scripts/run_sec_metaheuristics.py`
- manuscript source: `paper/elsarticle/main.tex`
