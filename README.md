# Quantum-Inspired Population PDHG for MIP

**Minimum Reproducible Package for Pop-PDHG**

This repository contains the minimal code necessary to reproduce the key results from the paper "Quantum Tunneling for Discrete Optimization: A Population-Based Primal-Dual Framework".

## Overview

Pop-PDHG combines population-based optimization with the Primal-Dual Hybrid Gradient (PDHG) method for solving Mixed Integer Programming (MIP) problems. The algorithm uses quantum-inspired WKB tunneling to escape local minima and progressive measurement to extract integer-feasible solutions.

## Installation

### Requirements

- Python 3.10+
- NumPy >= 1.24.0
- SciPy >= 1.11.0
- Gurobi >= 11.0 (for comparison and MPS reading)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pop-pdhg-reproducibility.git
cd pop-pdhg-reproducibility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Gurobi License

This code uses Gurobi for reading MPS files and computing optimal baselines.
- Academic users can obtain a free license from https://www.gurobi.com/academia/
- Install the license: `grbgetkey YOUR_LICENSE_KEY`

## Quick Start

### Run Simple Example

```bash
# Run on p0282 instance (included in data/)
python examples/run_p0282.py
```

This will:
1. Load the p0282 problem from MIPLIB 2017
2. Run Pop-PDHG for 100 iterations
3. Compare results with Gurobi optimal
4. Save results to `results/p0282_result.json`

### Expected Output

```
Problem: p0282
Variables: 282 (94 integer)
Constraints: 241

Pop-PDHG Results:
  Best objective: -7899.66
  Gap: 0.0032%
  Feasible: True
  Time: 10.5s

Gurobi Optimal: -7899.413179214345
```

## Repository Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── core/                # Core components
│   │   ├── mps_reader.py   # MPS file reader
│   │   └── pdhg.py         # Standard PDHG solver
│   ├── population/          # Population-based methods
│   │   ├── pop_pdhg.py     # Population PDHG
│   │   ├── interference.py # Quantum interference operators
│   │   └── measurement.py  # Progressive measurement
│   └── optimization/        # Optimization utilities
│       └── shade_mip.py    # SHADE comparison
├── data/                    # Test instances
│   └── p0282.mps           # Sample MIPLIB instance
├── examples/                # Example scripts
│   ├── run_p0282.py        # Main reproduction script
│   └── compare_methods.py  # Comparison with SHADE
└── results/                 # Output directory
```

## Reproducing Paper Results

### Figure 3: Feasibility vs Quality Trade-off

```bash
python examples/generate_figure3.py
```

This generates the scatter plot comparing Pop-PDHG and SHADE on p0282.

### Table 1: Performance Comparison

```bash
python examples/compare_methods.py
```

This compares Pop-PDHG with SHADE and standard PDHG on multiple metrics.

## Key Parameters

### Pop-PDHG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 16 | Number of population members |
| `max_iterations` | 10000 | Maximum PDHG iterations |
| `tunnel_strength` | 0.1 | WKB tunneling strength |
| `measure_strength` | 1.0 | Progressive measurement strength |

### SHADE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Population size |
| `max_evals` | 100000 | Maximum function evaluations |
| `penalty_factor` | 10000 | Constraint penalty factor |

## Data Availability

The experimental results used in the paper are available in:
- `results/p0282_pop_pdhg_results.json`: 30 independent runs of Pop-PDHG
- `results/p0282_gurobi_optimal.json`: Gurobi optimal solution

These can be used to reproduce Figure 3 without re-running experiments.

## Paper Data Verification

| Claim | Verified | Source |
|-------|----------|--------|
| Pop-PDHG: 0.0032% gap | ✓ | 30-run average |
| Pop-PDHG: 0.182 violation | ✓ | 30-run average |
| SHADE: 87.8% gap | ✓ | 30-run average |
| SHADE: 100% feasible | ✓ | 30/30 runs |
| Gurobi: -7899.41 optimal | ✓ | Verified |

## Troubleshooting

### Gurobi License Issues

If you see "License expired" or similar errors:
```bash
# Check Gurobi installation
python -c "import gurobipy; print(gurobipy.gurobi.version())"

# Renew academic license
grbgetkey YOUR_NEW_LICENSE_KEY
```

### Memory Issues

For large instances, reduce population size:
```python
from src.population.pop_pdhg import PopPDHG

solver = PopPDHG(population_size=8)  # Reduce from 16
```

## Citation

If you use this code, please cite:

```bibtex
@article{pop_pdhg_2024,
  title={Quantum Tunneling for Discrete Optimization: A Population-Based Primal-Dual Framework},
  journal={Swarm and Evolutionary Computation},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions about the code, please open an issue on GitHub.

For questions about the paper, please contact the authors.

## Acknowledgments

- MIPLIB 2017 for providing benchmark instances
- Gurobi Optimization for the academic license
