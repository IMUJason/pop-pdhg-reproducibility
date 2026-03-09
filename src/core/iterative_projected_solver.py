"""
Iterative Projected PDHG Solver.

This solver combines PDHG optimization with feasibility projection in an
iterative loop. The key idea is:

1. Run PDHG for a fixed number of iterations
2. Project the solution onto the feasible region (handling >= and = constraints)
3. If projection succeeds and improves objective, use as warm start
4. Repeat until converged or max cycles reached

This approach:
- Preserves PDHG's core algorithm (quantum tunneling, population, etc.)
- Handles arbitrary constraint types through projection
- Progressively improves solution quality
- Does NOT require modifying the PDHG solver itself
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import time

from core.feasibility_projector import FeasibilityProjector, ConstraintSense
from population.quantum_pop_pdhg import QuantumPopulationPDHG, QuantumPopPDHGConfig
from population.pop_pdhg import PopulationState


@dataclass
class IterativeSolverResult:
    """Result of iterative projected solving."""
    x_best: np.ndarray
    obj_best: float
    cycles: int
    total_iters: int
    total_time: float
    feasible: bool
    final_violation: float
    history: List[Dict]


class IterativeProjectedSolver:
    """Iterative solver that alternates between PDHG and projection.

    Algorithm:
    ---------
    Initialize x from diverse starting points
    For cycle = 1, 2, ..., max_cycles:
        1. Run PDHG for N iterations starting from current x
        2. Project PDHG output onto feasible region
        3. If projection feasible and objective improved:
           - Update best solution
           - Use projected point as next initial point
        4. Else:
           - Try different random restart
           - Or increase PDHG iterations

    The projection step handles >= and = constraints properly,
    while PDHG operates on the standard <= form.
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        senses: List[ConstraintSense],
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None,
    ):
        """Initialize iterative solver.

        Args:
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            senses: List of constraint senses
            lb: Variable lower bounds
            ub: Variable upper bounds
            integer_vars: Indices of integer variables
        """
        self.A_orig = A.copy()
        self.b_orig = b.copy()
        self.c_orig = c.copy()
        self.senses = senses
        self.lb_orig = lb.copy()
        self.ub_orig = ub.copy()
        self.integer_vars = integer_vars or []

        self.m, self.n = A.shape

        # Create feasibility projector
        self.projector = FeasibilityProjector(A, b, senses, lb, ub)

        # Convert to PDHG form (all <=)
        self.A_pdh = A.copy()
        self.b_pdh = b.copy()
        for i in range(len(senses)):
            if senses[i] == ConstraintSense.GE:
                # >= becomes <= by negation
                self.A_pdh[i] = -self.A_pdh[i]
                self.b_pdh[i] = -self.b_pdh[i]

        print(f"  IterativeSolver: {self.n} vars, {len(senses)} constraints")
        print(f"    PDHG form: {sum(1 for s in senses if s == ConstraintSense.LE)} <=, "
              f"{sum(1 for s in senses if s == ConstraintSense.GE)} >= (converted), "
              f"{sum(1 for s in senses if s == ConstraintSense.EQ)} = (converted)")

    def _compute_original_objective(self, x: np.ndarray) -> float:
        """Compute objective in original space."""
        return self.c_orig @ x

    def _check_original_feasibility(self, x: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
        """Check feasibility against original constraints."""
        max_viol, _ = self.projector.compute_violation(x)
        return max_viol <= tol, max_viol

    def _create_pdh_solver(self, config: Optional[QuantumPopPDHGConfig] = None) -> QuantumPopulationPDHG:
        """Create PDHG solver with transformed constraints."""
        if config is None:
            config = QuantumPopPDHGConfig(
                use_tunnel=True,
                tunnel_interval=50,
                use_progressive_measure=True,
                measure_interval=100,
                integer_vars=self.integer_vars,
            )

        return QuantumPopulationPDHG(
            self.A_pdh, self.b_pdh, self.c_orig,
            self.lb_orig, self.ub_orig,
            population_size=16,
            config=config,
        )

    def _initialize_population_from_point(
        self,
        solver: QuantumPopulationPDHG,
        x0: np.ndarray,
        seed: int = 42,
    ) -> PopulationState:
        """Initialize population from a given starting point."""
        np.random.seed(seed)
        K = solver.K

        X = np.zeros((K, self.n))
        for k in range(K):
            if k == 0:
                # First member at x0
                X[k] = x0.copy()
            else:
                # Add perturbations for diversity
                noise = np.random.randn(self.n) * 0.05
                X[k] = x0 + noise
                X[k] = np.clip(X[k], self.lb_orig, self.ub_orig)

                # Round integers
                for i in self.integer_vars:
                    X[k, i] = round(X[k, i])

        Y = np.abs(np.random.randn(K, solver.m))

        # Compute objectives
        obj = np.array([self.c_orig @ X[k] for k in range(K)])

        # Compute primal feasibility
        primal_feas = np.zeros(K)
        for k in range(K):
            Ax = solver.A @ X[k]
            primal_feas[k] = np.linalg.norm(np.maximum(Ax - solver.b, 0), ord=np.inf)

        dual_feas = np.linalg.norm(solver.A.T @ Y.T, axis=0)
        age = np.zeros(K, dtype=int)

        return PopulationState(x=X, y=Y, obj=obj, primal_feas=primal_feas, dual_feas=dual_feas, age=age)

    def solve(
        self,
        max_cycles: int = 10,
        iters_per_cycle: int = 1000,
        verbose: bool = True,
        tol: float = 1e-6,
        use_adaptive_iters: bool = True,
    ) -> IterativeSolverResult:
        """Run iterative projected solving.

        Args:
            max_cycles: Maximum number of PDHG+projection cycles
            iters_per_cycle: PDHG iterations per cycle
            verbose: Print progress
            tol: Tolerance for feasibility
            use_adaptive_iters: Increase iterations if stuck

        Returns:
            IterativeSolverResult
        """
        start_time = time.time()

        if verbose:
            print("\n" + "=" * 80)
            print("Iterative Projected PDHG Solver")
            print("=" * 80)
            print(f"Max cycles: {max_cycles}, Iters per cycle: {iters_per_cycle}")

        # Track best solution found
        best_x = None
        best_obj = float('inf')
        best_viol = float('inf')
        history = []

        # Initial point: try to find something reasonable
        # Start with small positive values (not zero!)
        x_current = np.random.uniform(0.1, 0.5, self.n)
        x_current = np.clip(x_current, self.lb_orig, self.ub_orig)

        total_iters = 0

        for cycle in range(max_cycles):
            cycle_start = time.time()

            if verbose:
                print(f"\n--- Cycle {cycle + 1}/{max_cycles} ---")

            # Adjust iterations if stuck
            current_iters = iters_per_cycle
            if use_adaptive_iters and cycle > 0:
                current_iters = int(iters_per_cycle * (1 + cycle * 0.5))

            # Step 1: Run PDHG
            if verbose:
                print(f"Running PDHG for {current_iters} iterations...")

            solver = self._create_pdh_solver()
            state = self._initialize_population_from_point(solver, x_current, seed=42 + cycle)
            solver.state = state

            result = solver.solve(
                max_iter=current_iters,
                tol=tol,
                seed=42 + cycle,
                integer_vars=self.integer_vars,
                use_enhanced_repair=True,
                use_feasibility_aware_tunnel=True,
                verbose=False,
            )

            x_pdh = result.x_best
            obj_pdh = self._compute_original_objective(x_pdh)
            is_feas_pdh, viol_pdh = self._check_original_feasibility(x_pdh)

            total_iters += result.iterations

            if verbose:
                print(f"  PDHG: obj={obj_pdh:.2f}, violation={viol_pdh:.4e}, feasible={is_feas_pdh}")

            # Step 2: Project to feasible region
            if verbose:
                print(f"Projecting onto feasible region...")

            x_proj, proj_success, viol_proj = self.projector.project_with_integer_rounding(
                x_pdh, self.integer_vars, max_iters=1000, tol=tol
            )

            obj_proj = self._compute_original_objective(x_proj)

            if verbose:
                print(f"  Proj: obj={obj_proj:.2f}, violation={viol_proj:.4e}, success={proj_success}")

            # Step 3: Try augmented Lagrangian if projection failed
            if not proj_success and viol_proj > 1e-3:
                if verbose:
                    print(f"Trying AL repair...")

                x_al, al_success, viol_al = self.projector.repair_with_augmented_lagrangian(
                    x_pdh, self.integer_vars, max_iters=300, tol=tol
                )

                obj_al = self._compute_original_objective(x_al)

                if verbose:
                    print(f"  AL: obj={obj_al:.2f}, violation={viol_al:.4e}, success={al_success}")

                # Use best of projection and AL
                if viol_al < viol_proj:
                    x_proj = x_al
                    obj_proj = obj_al
                    viol_proj = viol_al
                    proj_success = al_success

            # Step 4: Update best solution
            improved = False
            if proj_success or viol_proj < best_viol:
                if obj_proj < best_obj or viol_proj < best_viol:
                    best_x = x_proj.copy()
                    best_obj = obj_proj
                    best_viol = viol_proj
                    improved = True

                    if verbose:
                        print(f"  ✓ New best: obj={best_obj:.2f}, violation={best_viol:.4e}")

            # Step 5: Set up for next cycle
            if improved:
                # Use projected feasible point as next initial point
                x_current = x_proj.copy()
            else:
                # Try random restart with perturbation
                if verbose:
                    print(f"  No improvement, trying random restart...")

                noise = np.random.randn(self.n) * (0.1 + cycle * 0.05)
                x_current = x_proj + noise
                x_current = np.clip(x_current, self.lb_orig, self.ub_orig)

                for i in self.integer_vars:
                    x_current[i] = round(x_current[i])

            # Record history
            cycle_time = time.time() - cycle_start
            history.append({
                'cycle': cycle + 1,
                'pdh_obj': obj_pdh,
                'pdh_viol': viol_pdh,
                'proj_obj': obj_proj,
                'proj_viol': viol_proj,
                'best_obj': best_obj,
                'best_viol': best_viol,
                'time': cycle_time,
                'improved': improved,
            })

            # Check if converged
            if best_viol <= tol and best_obj < float('inf'):
                if verbose:
                    print(f"\n✓ Converged to feasible solution!")
                break

            # Check if stuck
            if cycle > 2 and not improved:
                # Check if we keep getting the same solution
                recent = history[-3:]
                if all(abs(r['best_viol'] - best_viol) < 1e-6 for r in recent):
                    if verbose:
                        print(f"\n⚠ Stalled, stopping early")
                    break

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*80}")
            print("FINAL RESULTS")
            print(f"{'='*80}")
            print(f"Cycles: {len(history)}")
            print(f"Total PDHG iterations: {total_iters}")
            print(f"Total time: {total_time:.1f}s")

            if best_x is not None:
                print(f"Best objective: {best_obj:.4f}")
                print(f"Best violation: {best_viol:.4e}")
                print(f"Feasible: {best_viol <= tol}")
            else:
                print("No feasible solution found")

        # If no solution found, return last projection attempt
        if best_x is None:
            best_x = x_proj
            best_obj = obj_proj
            best_viol = viol_proj

        return IterativeSolverResult(
            x_best=best_x,
            obj_best=best_obj,
            cycles=len(history),
            total_iters=total_iters,
            total_time=total_time,
            feasible=best_viol <= tol,
            final_violation=best_viol,
            history=history,
        )
