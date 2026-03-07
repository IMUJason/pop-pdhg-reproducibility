#!/usr/bin/env python3
"""
Reproduce Pop-PDHG results on p0282 instance from MIPLIB 2017.

This script:
1. Loads the p0282 problem
2. Runs Pop-PDHG with standard parameters
3. Compares with Gurobi optimal
4. Saves results to JSON
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.mps_reader import read_mps
from population.pop_pdhg import PopPDHG


def run_gurobi_baseline(lp_data):
    """Run Gurobi to get optimal solution."""
    try:
        import gurobipy as gp

        model = gp.Model("baseline")
        model.setParam('OutputFlag', 0)

        # Create variables
        n = len(lp_data.c)
        x = []
        for i in range(n):
            if i in lp_data.binary_vars:
                x.append(model.addVar(vtype=gp.GRB.BINARY, lb=lp_data.lb[i], ub=lp_data.ub[i]))
            elif i in lp_data.integer_vars:
                x.append(model.addVar(vtype=gp.GRB.INTEGER, lb=lp_data.lb[i], ub=lp_data.ub[i]))
            else:
                x.append(model.addVar(lb=lp_data.lb[i], ub=lp_data.ub[i]))

        # Set objective
        model.setObjective(sum(lp_data.c[i] * x[i] for i in range(n)), gp.GRB.MINIMIZE)

        # Add constraints
        m = len(lp_data.b)
        for i in range(m):
            row = lp_data.A.getrow(i)
            indices = row.indices
            data = row.data
            expr = sum(data[j] * x[indices[j]] for j in range(len(indices)))

            if lp_data.sense[i] == '<':
                model.addConstr(expr <= lp_data.b[i])
            elif lp_data.sense[i] == '>':
                model.addConstr(expr >= lp_data.b[i])
            else:
                model.addConstr(expr == lp_data.b[i])

        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            return {
                'status': 'optimal',
                'obj': model.objVal,
                'time': model.Runtime,
                'gap': 0.0
            }
        else:
            return {'status': 'failed', 'obj': None}

    except Exception as e:
        print(f"Gurobi error: {e}")
        return {'status': 'error', 'obj': None}


def run_pop_pdhg(lp_data, max_iter=1000, population_size=16, seed=42):
    """Run Pop-PDHG solver."""
    print(f"\nRunning Pop-PDHG...")
    print(f"  Population size: {population_size}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Seed: {seed}")

    solver = PopPDHG(
        population_size=population_size,
        max_iterations=max_iter,
        random_seed=seed,
        tunnel_strength=0.1,
        measure_strength=1.0,
        verbose=True
    )

    start_time = time.time()
    result = solver.solve(
        A=lp_data.A,
        b=lp_data.b,
        c=lp_data.c,
        lb=lp_data.lb,
        ub=lp_data.ub,
        integer_vars=lp_data.integer_vars,
        binary_vars=lp_data.binary_vars
    )
    elapsed = time.time() - start_time

    return {
        'obj': result.get('obj_best', None),
        'feasible': result.get('is_feasible', False),
        'violation': result.get('max_constraint_violation', None),
        'iterations': result.get('iterations', 0),
        'time': elapsed,
        'gap': result.get('gap_percent', None)
    }


def main():
    """Main function."""
    print("=" * 70)
    print("Pop-PDHG Reproduction: p0282 instance")
    print("=" * 70)

    # Load problem
    data_path = Path(__file__).parent.parent / "data" / "p0282.mps"
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please download p0282.mps from MIPLIB 2017")
        return

    print(f"\nLoading problem from {data_path}...")
    lp_data = read_mps(str(data_path))

    if lp_data is None:
        print("Error: Failed to load problem")
        return

    print(f"\nProblem Statistics:")
    print(f"  Name: {lp_data.name}")
    print(f"  Variables: {lp_data.n}")
    print(f"  Constraints: {lp_data.m}")
    print(f"  Non-zeros: {lp_data.nnz}")
    print(f"  Integer vars: {len(lp_data.integer_vars)}")
    print(f"  Binary vars: {len(lp_data.binary_vars)}")

    # Run Gurobi baseline
    print("\n" + "=" * 70)
    print("Running Gurobi baseline...")
    print("=" * 70)
    gurobi_result = run_gurobi_baseline(lp_data)

    if gurobi_result['status'] == 'optimal':
        print(f"\nGurobi Optimal:")
        print(f"  Objective: {gurobi_result['obj']:.6f}")
        print(f"  Time: {gurobi_result['time']:.2f}s")
        optimal_obj = gurobi_result['obj']
    else:
        print("Warning: Gurobi failed to find optimal solution")
        optimal_obj = None

    # Run Pop-PDHG
    print("\n" + "=" * 70)
    print("Running Pop-PDHG")
    print("=" * 70)
    pop_result = run_pop_pdhg(
        lp_data,
        max_iter=1000,
        population_size=16,
        seed=42
    )

    print(f"\nPop-PDHG Results:")
    print(f"  Best objective: {pop_result['obj']:.6f}" if pop_result['obj'] else "  Best objective: N/A")
    print(f"  Feasible: {pop_result['feasible']}")
    print(f"  Max violation: {pop_result['violation']:.6f}" if pop_result['violation'] else "  Max violation: N/A")
    print(f"  Iterations: {pop_result['iterations']}")
    print(f"  Time: {pop_result['time']:.2f}s")

    if optimal_obj and pop_result['obj']:
        gap = abs(pop_result['obj'] - optimal_obj) / abs(optimal_obj) * 100
        print(f"  Gap: {gap:.4f}%")

    # Save results
    results = {
        'problem': 'p0282',
        'gurobi': gurobi_result,
        'pop_pdhg': pop_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    output_path = Path(__file__).parent.parent / "results" / "p0282_result.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
