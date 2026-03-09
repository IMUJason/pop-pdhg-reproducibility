#!/usr/bin/env python
"""
Parameter Sensitivity Analysis for Quantum Pop-PDHG.

Tests the effect of each parameter on solution quality for high-performing instances:
- knapsack_50 (100% feasibility, 0.13% gap)
- p0033 (0% gap, optimal solution)
"""

import sys
import json
import time
import itertools
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.population.quantum_pop_pdhg import QuantumPopulationPDHG, QuantumPopPDHGConfig
from src.core.mps_reader import read_mps
from src.optimization.feasibility_repair import compute_violation

warnings.filterwarnings('ignore')

# Test instances (only high-performing ones)
TEST_INSTANCES = {
    "knapsack_50": {
        "path": "data/miplib2017/knapsack_50.mps",
        "gurobi_opt": -2051.0,
        "description": "50-item knapsack"
    },
    "p0033": {
        "path": "data/miplib2017/p0033.mps",
        "gurobi_opt": -68.00,
        "description": "MIPLIB p0033"
    }
}

SENSE_MAP = {
    "<": "L",
    ">": "G",
    "=": "E",
    "L": "L",
    "G": "G",
    "E": "E",
}


def normalize_constraint_sense(senses):
    """Normalize raw constraint senses to L/G/E format."""
    if not senses:
        return []
    return [SENSE_MAP.get(s, "L") for s in senses]


def evaluate_solution(problem_data: Dict, x: np.ndarray) -> Tuple[bool, float, float, float]:
    """Evaluate feasibility against original constraints, bounds, and integrality."""
    lb = problem_data["lb"]
    ub = problem_data["ub"]
    bound_viol = 0.0
    if len(lb) > 0:
        bound_viol = max(
            float(np.max(np.maximum(lb - x, 0.0))),
            float(np.max(np.maximum(x - ub, 0.0))),
        )

    primal_viol, int_viol, _ = compute_violation(
        x=x,
        A=problem_data["A"],
        b=problem_data["b"],
        constraint_sense=problem_data["constraint_sense"],
        integer_vars=problem_data["integer_vars"],
    )
    max_viol = max(bound_viol, primal_viol, int_viol)
    is_feasible = bool(bound_viol < 1e-6 and primal_viol < 1e-4 and int_viol < 1e-6)
    return is_feasible, max_viol, primal_viol, int_viol

# Baseline configuration (from successful runs)
BASELINE_CONFIG = {
    # Population parameters
    "population_size": 16,
    "elite_ratio": 0.2,

    # Quantum tunneling parameters
    "use_tunnel": True,
    "tunnel_interval": 50,
    "tunnel_strength": 1.0,
    "levy_alpha": 1.5,

    # Progressive measurement parameters
    "use_progressive_measure": True,
    "measure_interval": 100,
    "initial_measure_strength": 0.1,
    "final_measure_strength": 1.0,
    "use_cosine_schedule": True,

    # Smart trigger (HMC alternative)
    "use_smart_trigger": True,
    "use_smart_trigger": True,

    # Termination
    "max_iter": 1000,
    "tol": 1e-4,
}

# Parameter space for sensitivity analysis
PARAMETER_SPACE = {
    # Population parameters
    "population_size": [4, 8, 16, 32, 64],
    "elite_ratio": [0.1, 0.2, 0.3, 0.4],

    # Quantum tunneling parameters
    "tunnel_interval": [25, 50, 100, 200],
    "tunnel_strength": [0.5, 1.0, 2.0, 4.0],
    "levy_alpha": [1.0, 1.5, 2.0],

    # Progressive measurement parameters
    "measure_interval": [50, 100, 200, 400],
    "initial_measure_strength": [0.05, 0.1, 0.2, 0.5],
    "final_measure_strength": [0.5, 1.0, 2.0],

}


@dataclass
class SensitivityResult:
    """Result from a single parameter test."""
    instance: str
    parameter: str
    param_value: Any
    baseline_value: Any
    run_id: int

    # Solution quality
    obj_value: float
    gap_percent: float
    is_feasible: bool

    # Performance metrics
    runtime: float
    iterations: int
    converged: bool

    # Quantum-specific metrics
    tunnel_success_rate: float
    tunnel_attempts: int
    final_measure_strength: float

    # Comparison to baseline
    obj_diff: float = 0.0
    gap_diff: float = 0.0


def load_problem(instance_name: str):
    """Load MIP problem."""
    instance_info = TEST_INSTANCES[instance_name]
    full_path = PROJECT_ROOT / instance_info["path"]

    lp_data = read_mps(str(full_path))
    if lp_data is None:
        return None

    return {
        "A": lp_data.A,
        "b": lp_data.b,
        "c": lp_data.c,
        "lb": lp_data.lb,
        "ub": lp_data.ub,
        "integer_vars": list(lp_data.integer_vars or []),
        "constraint_sense": normalize_constraint_sense(lp_data.sense),
        "gurobi_opt": instance_info["gurobi_opt"]
    }


def create_config(param_name: str, param_value, baseline: Dict) -> QuantumPopPDHGConfig:
    """Create config with one parameter varied."""
    config_dict = baseline.copy()
    config_dict[param_name] = param_value

    # Handle boolean parameters that might be strings
    if param_name == "use_cosine_schedule" and isinstance(param_value, str):
        param_value = param_value == "True"
    if param_name == "use_smart_trigger" and isinstance(param_value, str):
        param_value = param_value == "True"

    return QuantumPopPDHGConfig(
        use_tunnel=config_dict["use_tunnel"],
        use_progressive_measure=config_dict["use_progressive_measure"],
        use_smart_trigger=config_dict["use_smart_trigger"],
        tunnel_interval=config_dict["tunnel_interval"],
        measure_interval=config_dict["measure_interval"],
        initial_measure_strength=config_dict["initial_measure_strength"],
        final_measure_strength=config_dict["final_measure_strength"],
        integer_vars=[],  # Set later
    )


def run_single_test(
    problem_data: Dict,
    param_name: str,
    param_value: Any,
    baseline: Dict,
    run_id: int,
    seed: int = 42
) -> SensitivityResult:
    """Run a single parameter test."""

    # Create config
    config = create_config(param_name, param_value, baseline)
    config.integer_vars = problem_data["integer_vars"]

    # Determine population size
    pop_size = param_value if param_name == "population_size" else baseline["population_size"]

    # Create solver
    solver = QuantumPopulationPDHG(
        A=problem_data["A"],
        b=problem_data["b"],
        c=problem_data["c"],
        lb=problem_data["lb"],
        ub=problem_data["ub"],
        population_size=pop_size,
        config=config
    )

    # Run solver
    start_time = time.time()
    try:
        result = solver.solve(
            max_iter=baseline["max_iter"],
            tol=baseline["tol"],
            seed=seed + run_id,
            integer_vars=problem_data["integer_vars"],
            use_enhanced_repair=True,
            use_feasibility_aware_tunnel=True,
        )
        runtime = time.time() - start_time

        # Calculate metrics
        gurobi_opt = problem_data["gurobi_opt"]
        gap = abs((result.obj_best - gurobi_opt) / gurobi_opt) * 100 if gurobi_opt != 0 else 0

        # Check feasibility
        x = result.x_best
        is_feasible, _, _, _ = evaluate_solution(problem_data, x)

        # Get tunnel stats
        tunnel_rate = result.tunnel_stats.get("success_rate", 0.0) if hasattr(result, 'tunnel_stats') else 0.0
        tunnel_attempts = result.tunnel_stats.get("attempts", 0) if hasattr(result, 'tunnel_stats') else 0

        # Get baseline value for comparison
        if param_name in baseline:
            baseline_val = baseline[param_name]
        else:
            # For 'baseline' runs, use the param_value itself
            baseline_val = param_value

        return SensitivityResult(
            instance="",
            parameter=param_name,
            param_value=param_value,
            baseline_value=baseline_val,
            run_id=run_id,
            obj_value=float(result.obj_best),
            gap_percent=float(gap),
            is_feasible=is_feasible,
            runtime=runtime,
            iterations=result.iterations,
            converged=result.converged,
            tunnel_success_rate=tunnel_rate,
            tunnel_attempts=tunnel_attempts,
            final_measure_strength=getattr(result, 'final_measure_strength', 0.0)
        )
    except Exception as e:
        print(f"    Error in run {run_id}: {e}")
        # Get baseline value for comparison
        if param_name in baseline:
            baseline_val = baseline[param_name]
        else:
            baseline_val = param_value
        return SensitivityResult(
            instance="",
            parameter=param_name,
            param_value=param_value,
            baseline_value=baseline_val,
            run_id=run_id,
            obj_value=float('inf'),
            gap_percent=float('inf'),
            is_feasible=False,
            runtime=0.0,
            iterations=0,
            converged=False,
            tunnel_success_rate=0.0,
            tunnel_attempts=0,
            final_measure_strength=0.0
        )


def run_baseline_test(problem_data: Dict, instance_name: str, n_runs: int = 5) -> List[SensitivityResult]:
    """Run baseline configuration multiple times."""
    print(f"\n  Running baseline configuration ({n_runs} runs)...")

    results = []
    for run_id in range(n_runs):
        # Run with first parameter in space as reference
        first_param = list(PARAMETER_SPACE.keys())[0]
        baseline_val = BASELINE_CONFIG[first_param]

        result = run_single_test(
            problem_data,
            "baseline",
            baseline_val,
            BASELINE_CONFIG,
            run_id,
            seed=42
        )
        result.instance = instance_name
        result.parameter = "baseline"
        result.baseline_value = baseline_val
        results.append(result)

    # Calculate average gap
    avg_gap = np.mean([r.gap_percent for r in results])
    print(f"    Baseline avg gap: {avg_gap:.2f}%")

    return results


def run_parameter_sweep(
    instance_name: str,
    param_name: str,
    param_values: List,
    problem_data: Dict,
    n_runs: int = 5
) -> List[SensitivityResult]:
    """Run sweep over a single parameter."""

    print(f"\n  Testing {param_name}: {param_values}")

    # First run baseline
    baseline_results = run_baseline_test(problem_data, instance_name, n_runs)
    baseline_gap = np.mean([r.gap_percent for r in baseline_results])

    all_results = baseline_results.copy()

    # Test each parameter value
    for param_value in param_values:
        print(f"    Value: {param_value}", end="")

        for run_id in range(n_runs):
            result = run_single_test(
                problem_data,
                param_name,
                param_value,
                BASELINE_CONFIG,
                run_id,
                seed=42 + run_id
            )
            result.instance = instance_name
            result.gap_diff = result.gap_percent - baseline_gap
            all_results.append(result)

        # Quick summary
        param_results = [r for r in all_results if r.parameter == param_name and r.param_value == param_value]
        avg_gap = np.mean([r.gap_percent for r in param_results])
        feas_rate = sum(r.is_feasible for r in param_results) / len(param_results) * 100
        print(f" -> gap={avg_gap:.2f}%, feas={feas_rate:.0f}%")

    return all_results


def run_sensitivity_analysis(output_dir: str = "experiments/results"):
    """Run complete sensitivity analysis."""

    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Instances: {list(TEST_INSTANCES.keys())}")
    print(f"Parameters to test: {len(PARAMETER_SPACE)}")
    print(f"Total parameter values: {sum(len(v) for v in PARAMETER_SPACE.values())}")
    print()

    all_results = []

    for instance_name in TEST_INSTANCES.keys():
        print(f"\n{'='*80}")
        print(f"Instance: {instance_name}")
        print(f"{'='*80}")

        # Load problem
        problem_data = load_problem(instance_name)
        if problem_data is None:
            print(f"  Failed to load {instance_name}, skipping")
            continue

        print(f"  Variables: {len(problem_data['c'])}")
        print(f"  Constraints: {len(problem_data['b'])}")
        print(f"  Gurobi optimal: {problem_data['gurobi_opt']}")

        # Test each parameter
        for param_name, param_values in PARAMETER_SPACE.items():
            results = run_parameter_sweep(
                instance_name,
                param_name,
                param_values,
                problem_data,
                n_runs=5
            )
            all_results.extend(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON
    def convert_value(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def deep_convert(obj):
        """Deep conversion of numpy types and nested structures."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        return obj

    results_dict = {
        "baseline_config": {k: deep_convert(v) for k, v in BASELINE_CONFIG.items()},
        "parameter_space": {k: [deep_convert(v) for v in vals] for k, vals in PARAMETER_SPACE.items()},
        "results": [
            {k: deep_convert(v) for k, v in asdict(r).items()}
            for r in all_results
        ]
    }

    results_file = output_path / "parameter_sensitivity_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")

    # Generate summary
    generate_summary(all_results, output_path)

    return all_results


def generate_summary(results: List[SensitivityResult], output_path: Path):
    """Generate summary statistics."""

    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 80)

    # Group by instance and parameter
    instances = set(r.instance for r in results)

    for instance in instances:
        if not instance:
            continue

        print(f"\n{instance.upper()}:")
        print("-" * 80)

        instance_results = [r for r in results if r.instance == instance]

        # Skip baseline-only results
        params_tested = set(r.parameter for r in instance_results if r.parameter != "baseline")

        for param in sorted(params_tested):
            param_results = [r for r in instance_results if r.parameter == param]

            # Group by parameter value
            values = sorted(set(r.param_value for r in param_results))

            print(f"\n  {param}:")
            print(f"    {'Value':<15} {'Avg Gap%':>10} {'Feas%':>8} {'Time(s)':>10} {'Tunnel%':>10}")
            print(f"    {'-'*60}")

            for val in values:
                val_results = [r for r in param_results if r.param_value == val]
                avg_gap = np.mean([r.gap_percent for r in val_results])
                feas_rate = sum(r.is_feasible for r in val_results) / len(val_results) * 100
                avg_time = np.mean([r.runtime for r in val_results])
                avg_tunnel = np.mean([r.tunnel_success_rate for r in val_results]) * 100

                print(f"    {str(val):<15} {avg_gap:>10.2f} {feas_rate:>8.1f} {avg_time:>10.2f} {avg_tunnel:>10.1f}")


def generate_sensitivity_plots(results_file: str, output_dir: str = "experiments/figures"):
    """Generate sensitivity plots from results."""

    print("\nGenerating sensitivity plots...")

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = [SensitivityResult(**r) for r in data["results"]]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create plots for each parameter
    instances = sorted(set(r.instance for r in results if r.instance))
    parameters = sorted(set(r.parameter for r in results if r.parameter != "baseline"))

    n_params = len(parameters)
    n_instances = len(instances)

    # Figure 1: Gap sensitivity curves
    fig, axes = plt.subplots(n_instances, n_params, figsize=(4*n_params, 4*n_instances))
    if n_instances == 1:
        axes = axes.reshape(1, -1)
    if n_params == 1:
        axes = axes.reshape(-1, 1)

    for i, instance in enumerate(instances):
        for j, param in enumerate(parameters):
            ax = axes[i, j]

            # Get baseline
            baseline_results = [r for r in results if r.instance == instance and r.parameter == "baseline"]
            baseline_gap = np.mean([r.gap_percent for r in baseline_results]) if baseline_results else 0

            # Get parameter results
            param_results = [r for r in results if r.instance == instance and r.parameter == param]

            if not param_results:
                continue

            # Group by value
            values = sorted(set(r.param_value for r in param_results))
            gaps = []
            gap_stds = []

            for val in values:
                val_results = [r.gap_percent for r in param_results if r.param_value == val]
                gaps.append(np.mean(val_results))
                gap_stds.append(np.std(val_results))

            # Plot
            ax.errorbar(values, gaps, yerr=gap_stds, marker='o', capsize=5, linewidth=2)
            ax.axhline(y=baseline_gap, color='r', linestyle='--', label='Baseline', alpha=0.7)
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Gap (%)')
            ax.set_title(f"{instance}: {param}")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    pdf_path = output_path / "parameter_sensitivity_gap.pdf"
    png_path = output_path / "parameter_sensitivity_gap.png"
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"  Gap sensitivity plots saved")
    plt.close()

    # Figure 2: Feasibility rate heatmap (for selected parameters)
    key_params = ["population_size", "tunnel_interval", "measure_interval"]

    for instance in instances:
        fig, axes = plt.subplots(1, len(key_params), figsize=(6*len(key_params), 5))
        if len(key_params) == 1:
            axes = [axes]

        for idx, param in enumerate(key_params):
            ax = axes[idx]

            param_results = [r for r in results if r.instance == instance and r.parameter == param]
            if not param_results:
                continue

            values = sorted(set(r.param_value for r in param_results))
            feas_rates = []

            for val in values:
                val_results = [r for r in param_results if r.param_value == val]
                feas_rate = sum(r.is_feasible for r in val_results) / len(val_results) * 100
                feas_rates.append(feas_rate)

            ax.bar(range(len(values)), feas_rates, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values], rotation=45)
            ax.set_ylabel('Feasibility Rate (%)')
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_title(f"{instance}")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        pdf_path = output_path / f"feasibility_{instance}.pdf"
        png_path = output_path / f"feasibility_{instance}.png"
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Feasibility plots for {instance} saved")
        plt.close()

    print(f"\nAll plots saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parameter Sensitivity Analysis")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")
    parser.add_argument("--results-file", default="experiments/results/parameter_sensitivity_results.json",
                        help="Results file for plotting")

    args = parser.parse_args()

    if args.plot_only:
        generate_sensitivity_plots(args.results_file, "experiments/figures")
    else:
        # Run full analysis
        results = run_sensitivity_analysis(args.output_dir)

        # Generate plots
        results_file = Path(args.output_dir) / "parameter_sensitivity_results.json"
        generate_sensitivity_plots(str(results_file), "experiments/figures")

        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results: {results_file}")
        print(f"Plots: experiments/figures/parameter_sensitivity_*.pdf")
