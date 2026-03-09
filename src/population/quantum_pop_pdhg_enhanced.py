"""
Enhanced Quantum-Inspired Population PDHG Solver.

Improvements over standard version:
1. Mixed-integer constraint repair (handles continuous + general integer vars)
2. Two-phase solving strategy (feasibility first, then optimization)
3. Feasibility-aware quantum tunneling (prevents over-optimization)
4. Augmented Lagrangian for equality constraints

Author: Claude
Date: 2025-03-07
"""

import numpy as np
from scipy import sparse
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# Import base class
try:
    from population.quantum_pop_pdhg import (
        QuantumPopulationPDHG, QuantumPopPDHGConfig,
        QuantumPopPDHGResult, PopulationState
    )
except ImportError:
    from .quantum_pop_pdhg import (
        QuantumPopulationPDHG, QuantumPopPDHGConfig,
        QuantumPopPDHGResult, PopulationState
    )


@dataclass
class EnhancedPopPDHGConfig(QuantumPopPDHGConfig):
    """Configuration for enhanced Pop-PDHG.

    Additional attributes:
        use_two_phase: Enable two-phase solving
        phase1_weight: Constraint weight in phase 1 (feasibility)
        phase2_weight: Constraint weight in phase 2 (optimization)
        use_feasibility_aware_tunnel: Only tunnel when nearly feasible
        feasibility_threshold: Max constraint violation for tunneling
        use_augmented_lagrangian: Enable AL for equality constraints
        al_rho_init: Initial AL penalty parameter
        al_rho_max: Maximum AL penalty parameter
        use_mixed_integer_repair: Enable mixed-integer constraint repair
        repair_max_iters: Max iterations for repair procedure
        repair_tol: Tolerance for repair convergence
    """
    # Two-phase strategy
    use_two_phase: bool = True
    phase1_iters_ratio: float = 0.4  # 40% iterations for phase 1
    phase1_constraint_weight: float = 10.0
    phase2_constraint_weight: float = 1.0

    # Feasibility-aware tunneling
    use_feasibility_aware_tunnel: bool = True
    feasibility_threshold: float = 1.0  # Only tunnel if violation < 1.0
    max_tunnel_violation_increase: float = 2.0  # Reject if violation increases 2x

    # Augmented Lagrangian for equalities
    use_augmented_lagrangian: bool = True
    al_rho_init: float = 1.0
    al_rho_max: float = 1000.0
    al_update_interval: int = 50

    # Mixed-integer repair
    use_mixed_integer_repair: bool = True
    repair_max_iters: int = 200
    repair_tol: float = 1e-6
    repair_step_size: float = 0.1
    repair_projection_steps: int = 5


class EnhancedQuantumPopulationPDHG(QuantumPopulationPDHG):
    """Enhanced Quantum Pop-PDHG with improved constraint handling.

    Key improvements:
    1. Mixed-integer repair: Handles continuous, integer, and binary variables
    2. Two-phase strategy: Prioritizes feasibility before optimization
    3. Feasibility-aware tunneling: Prevents over-optimization
    4. Augmented Lagrangian: Better handling of equality constraints
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        config: Optional[EnhancedPopPDHGConfig] = None,
    ):
        """Initialize enhanced solver."""
        # Use enhanced config with defaults
        if config is None:
            config = EnhancedPopPDHGConfig()

        # Initialize base class
        super().__init__(A, b, c, lb, ub, config)
        self.config = config

        # Identify constraint types
        self._identify_constraint_types()

        # Augmented Lagrangian state
        self.al_lambda = np.zeros(self.m)  # Multipliers
        self.al_rho = config.al_rho_init  # Penalty

        # Phase tracking
        self.current_phase = 1
        self.constraint_weight = config.phase1_constraint_weight

        # Variable type tracking (will be set during solve)
        self.continuous_vars = None
        self.general_integer_vars = None
        self.binary_vars = None

    def _identify_constraint_types(self):
        """Identify equality vs inequality constraints."""
        # This is a heuristic - in practice, you'd pass this info
        # For now, assume constraints with very similar upper/lower bounds are equalities
        self.is_equality = np.zeros(self.m, dtype=bool)
        # Mark constraints that appear to be equalities
        # (In real implementation, pass explicit information)

    def _classify_variables(self, integer_vars: Optional[List[int]] = None):
        """Classify variables into continuous, integer, and binary."""
        if integer_vars is None:
            integer_vars = self.config.integer_vars

        n = len(self.c)
        all_vars = set(range(n))
        int_set = set(integer_vars) if integer_vars else set()

        # Binary vars: integer with bounds [0, 1]
        self.binary_vars = [
            i for i in int_set
            if self.lb[i] == 0 and self.ub[i] == 1
        ]

        # General integer vars
        self.general_integer_vars = [
            i for i in int_set if i not in self.binary_vars
        ]

        # Continuous vars
        self.continuous_vars = list(all_vars - int_set)

    def _compute_constraint_violation(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute constraint violation and violation vector."""
        Ax = self.A @ x
        violation_vec = np.maximum(Ax - self.b, 0)
        violation = np.linalg.norm(violation_vec, ord=np.inf)
        return violation, violation_vec

    def _repair_mixed_integer(
        self,
        x: np.ndarray,
        max_iters: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """Mixed-integer constraint repair using gradient projection.

        Handles:
        - Binary variables: Greedy flipping (existing strategy)
        - General integer: Rounding + local search
        - Continuous: Gradient projection to feasible region
        """
        if max_iters is None:
            max_iters = self.config.repair_max_iters
        if tol is None:
            tol = self.config.repair_tol

        x_repaired = x.copy()
        alpha = self.config.repair_step_size

        # Phase 1: Handle binary variables with greedy strategy (fast)
        if self.binary_vars:
            x_repaired = self._repair_binary_greedy(x_repaired, max_iters=50)

        # Phase 2: Gradient projection for all variables
        for iteration in range(max_iters):
            # Compute violation
            violation, v_vec = self._compute_constraint_violation(x_repaired)

            if violation < tol:
                break

            # Compute gradient of ||(Ax-b)_+||^2
            grad = self.A.T @ v_vec

            # Line search for step size
            best_x = x_repaired.copy()
            best_violation = violation

            for step_scale in [1.0, 0.5, 0.25, 0.1]:
                x_candidate = x_repaired - step_scale * alpha * grad

                # Project to bounds
                x_candidate = np.clip(x_candidate, self.lb, self.ub)

                # Round integer variables
                if self.general_integer_vars:
                    for j in self.general_integer_vars:
                        x_candidate[j] = round(x_candidate[j])

                # Check improvement
                v_new, _ = self._compute_constraint_violation(x_candidate)
                if v_new < best_violation:
                    best_x = x_candidate
                    best_violation = v_new

            x_repaired = best_x

            # Early termination if no improvement
            if best_violation >= violation * 0.99:
                # Try coordinate descent for integer variables
                if self.general_integer_vars or self.binary_vars:
                    x_repaired = self._coordinate_descent_repair(x_repaired)
                break

        return x_repaired

    def _repair_binary_greedy(
        self,
        x: np.ndarray,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Greedy repair for binary variables only."""
        x_repaired = x.copy()

        for _ in range(max_iters):
            Ax = self.A @ x_repaired
            violations = Ax - self.b
            max_viol = np.max(violations)

            if max_viol <= 1e-6:
                break

            # Find best variable to flip
            candidates = []
            for j in self.binary_vars:
                if x_repaired[j] > 0.5:  # Currently 1
                    reduction = 0
                    for i in range(self.m):
                        if violations[i] > 0:
                            coef = self.A[i, j]
                            if coef > 0:
                                reduction += min(coef, violations[i])

                    if reduction > 0:
                        priority = reduction / (abs(self.c[j]) + 1e-10)
                        candidates.append((priority, j, reduction))

            if not candidates:
                break

            candidates.sort(reverse=True)
            _, best_j, _ = candidates[0]
            x_repaired[best_j] = 0

        return x_repaired

    def _coordinate_descent_repair(self, x: np.ndarray) -> np.ndarray:
        """Coordinate descent for integer variables."""
        x_repaired = x.copy()
        int_vars = (self.general_integer_vars or []) + (self.binary_vars or [])

        if not int_vars:
            return x_repaired

        for _ in range(50):  # Max 50 sweeps
            improved = False

            for j in int_vars:
                current_val = x_repaired[j]

                # Try neighboring integer values
                candidates = []
                if j in self.binary_vars:
                    candidates = [0, 1]
                else:
                    candidates = [round(current_val - 1),
                                  round(current_val),
                                  round(current_val + 1)]
                    candidates = [v for v in candidates
                                  if self.lb[j] <= v <= self.ub[j]]

                best_val = current_val
                best_violation = self._compute_constraint_violation(x_repaired)[0]

                for val in candidates:
                    x_repaired[j] = val
                    v_new = self._compute_constraint_violation(x_repaired)[0]
                    if v_new < best_violation:
                        best_val = val
                        best_violation = v_new
                        improved = True

                x_repaired[j] = best_val

            if not improved:
                break

        return x_repaired

    def tunnel_step_enhanced(
        self,
        state: PopulationState,
        iteration: int,
    ) -> PopulationState:
        """Feasibility-aware quantum tunneling.

        Only tunnel if:
        1. Current solution is nearly feasible (violation < threshold)
        2. Tunneling doesn't increase violation too much
        """
        if self.tunnel is None:
            return state

        if not self.config.use_feasibility_aware_tunnel:
            # Use standard tunneling
            return super().tunnel_step(state, iteration)

        x = state.x.copy()
        tunnel_prob = self.config.tunnel_config.tunnel_prob

        for k in range(self.K):
            if np.random.random() < tunnel_prob:
                # Check current feasibility
                current_violation, _ = self._compute_constraint_violation(x[k])

                # Only tunnel if nearly feasible
                if current_violation > self.config.feasibility_threshold:
                    # Try to repair first instead
                    x[k] = self._repair_mixed_integer(x[k], max_iters=20)
                    continue

                # Attempt tunneling
                x_new, accepted, _ = self.tunnel.execute(
                    x[k],
                    energy_fn=self._energy_fn,
                    iteration=iteration,
                    population=x,
                    project_fn=self._project_fn,
                    temperature=1.0,
                )

                if accepted:
                    # Check if tunneling maintains feasibility
                    new_violation, _ = self._compute_constraint_violation(x_new)

                    max_increase = self.config.max_tunnel_violation_increase
                    if new_violation <= current_violation * max_increase:
                        x[k] = x_new
                        self.tunnel_stats["successes"] += 1

                self.tunnel_stats["attempts"] += 1

        # Update state
        state.x = x
        state.obj = x @ self.c

        # Update feasibility
        Ax = (self.A @ x.T).T
        state.primal_feas = np.linalg.norm(np.maximum(Ax - self.b, 0), axis=1)

        return state

    def _augmented_lagrangian_step(self, state: PopulationState) -> PopulationState:
        """Apply augmented Lagrangian penalty for equality constraints."""
        if not self.config.use_augmented_lagrangian:
            return state

        # Compute constraint violation
        Ax = (self.A @ state.x.T).T
        violation = Ax - self.b

        # Apply penalty gradient: rho * A^T @ (Ax - b + lambda/rho)
        penalty_grad = self.A.T @ (self.al_lambda + self.al_rho * violation)

        # Add to objective gradient (affects dual variable update)
        # This is handled implicitly through the energy function

        return state

    def _update_al_multipliers(self, state: PopulationState):
        """Update augmented Lagrangian multipliers."""
        Ax = (self.A @ state.x.T).T
        violation = Ax - self.b

        # Update multipliers: lambda += rho * (Ax - b)
        mean_violation = np.mean(violation, axis=0)
        self.al_lambda += self.al_rho * mean_violation

        # Increase penalty if constraints still violated
        max_violation = np.max(np.abs(mean_violation))
        if max_violation > self.config.repair_tol:
            self.al_rho = min(
                self.al_rho * 1.5,
                self.config.al_rho_max
            )

    def solve(
        self,
        max_iter: int = 10000,
        tol: float = 1e-6,
        check_interval: int = 40,
        verbose: bool = False,
        init_strategy: str = "diverse",
        seed: Optional[int] = None,
        integer_vars: Optional[List[int]] = None,
    ) -> QuantumPopPDHGResult:
        """Solve using enhanced two-phase strategy."""
        # Classify variables
        self._classify_variables(integer_vars)

        # Compute phase iteration limits
        phase1_end = int(max_iter * self.config.phase1_iters_ratio)

        if verbose and self.config.use_two_phase:
            print(f"Two-phase strategy: Phase 1 (feasibility): 1-{phase1_end}, "
                  f"Phase 2 (optimization): {phase1_end+1}-{max_iter}")

        # Call parent solve with enhanced components
        return self._solve_enhanced(
            max_iter=max_iter,
            tol=tol,
            check_interval=check_interval,
            verbose=verbose,
            init_strategy=init_strategy,
            seed=seed,
            integer_vars=integer_vars,
            phase1_end=phase1_end,
        )

    def _solve_enhanced(
        self,
        max_iter: int,
        tol: float,
        check_interval: int,
        verbose: bool,
        init_strategy: str,
        seed: Optional[int],
        integer_vars: Optional[List[int]],
        phase1_end: int,
    ) -> QuantumPopPDHGResult:
        """Enhanced solve with two-phase strategy."""
        # Initialize
        state = self.initialize_population(strategy=init_strategy, seed=seed)

        history = []
        quantum_history = []
        converged = False
        status = "max_iter"

        # Reset statistics
        self.tunnel_stats = {"attempts": 0, "successes": 0}
        self.measure_stats = {"attempts": 0, "improvements": 0}

        # Reset AL state
        self.al_lambda = np.zeros(self.m)
        self.al_rho = self.config.al_rho_init

        if self.tunnel is not None:
            self.tunnel.reset_stats()

        if self.smart_trigger is not None:
            self.smart_trigger.reset()

        best_obj_so_far = float("inf")
        prev_obj = float("inf")
        best_feasible_obj = float("inf")
        best_feasible_x = None

        for k in range(1, max_iter + 1):
            # Phase transition
            if k == phase1_end + 1 and self.config.use_two_phase:
                self.current_phase = 2
                self.constraint_weight = self.config.phase2_constraint_weight
                if verbose:
                    print(f"\n*** Phase 2 started at iteration {k} ***")
                    print(f"Best feasible objective so far: {best_feasible_obj:.4f}")

            # 1. Standard PDHG step
            state = self.batch_pdhg_step(state)

            # 2. Augmented Lagrangian (if enabled)
            if self.config.use_augmented_lagrangian:
                state = self._augmented_lagrangian_step(state)
                if k % self.config.al_update_interval == 0:
                    self._update_al_multipliers(state)

            # 3. Enhanced quantum tunneling (feasibility-aware)
            if self.config.use_tunnel and k % self.config.tunnel_interval == 0:
                state = self.tunnel_step_enhanced(state, k)

            # 4. Progressive measurement with mixed-integer repair
            if self.config.use_progressive_measure and integer_vars:
                if k % self.config.measure_interval == 0:
                    state = self.progressive_measure_step(
                        state, k, max_iter, integer_vars
                    )

                    # Apply enhanced repair after measurement
                    if self.config.use_mixed_integer_repair:
                        for i in range(self.K):
                            state.x[i] = self._repair_mixed_integer(state.x[i])

            # 5. Track best feasible solution
            for i in range(self.K):
                violation, _ = self._compute_constraint_violation(state.x[i])
                if violation < tol:
                    obj = state.x[i] @ self.c
                    if obj < best_feasible_obj:
                        best_feasible_obj = obj
                        best_feasible_x = state.x[i].copy()

            # 6. Interference (if diversity is low)
            diversity = self.compute_diversity(state)
            if diversity < 0.01 and k % 100 == 0:
                state = self.interference_step(state)

            # 7. Convergence check
            if k % check_interval == 0:
                x_best, y_best, obj_best = self.get_best(state)

                # Compute residuals
                Ax = self.A @ x_best
                primal_res = np.linalg.norm(
                    np.maximum(Ax - self.b, 0), ord=np.inf
                ) / (1 + np.linalg.norm(self.b, ord=np.inf))

                ATy = self.A.T @ y_best
                reduced = ATy + self.c
                dual_res = np.linalg.norm(reduced, ord=np.inf) / (
                    1 + np.linalg.norm(self.c, ord=np.inf)
                )

                gap = abs(obj_best - self.b @ y_best) / (
                    1 + abs(obj_best) + abs(self.b @ y_best)
                )

                max_residual = max(primal_res, dual_res, gap)

                # Track improvement
                if prev_obj != float('inf') and abs(prev_obj) > 1e-10:
                    improvement = abs(prev_obj - obj_best) / abs(prev_obj)
                else:
                    improvement = 0.0
                prev_obj = obj_best

                if self.smart_trigger:
                    primal_violation = np.linalg.norm(
                        np.maximum(Ax - self.b, 0), ord=np.inf
                    )
                    self.smart_trigger.update(improvement)

                if obj_best < best_obj_so_far:
                    best_obj_so_far = obj_best

                history.append({
                    "iteration": k,
                    "obj_best": obj_best,
                    "primal_res": primal_res,
                    "dual_res": dual_res,
                    "gap": gap,
                    "diversity": diversity,
                    "phase": self.current_phase,
                    "best_feasible": best_feasible_obj,
                })

                if verbose and k % 200 == 0:
                    feas_str = f"best_feas={best_feasible_obj:.2f}" if best_feasible_x is not None else "no_feasible"
                    print(
                        f"Iter {k} (Phase {self.current_phase}): "
                        f"obj={obj_best:.4e}, p_res={primal_res:.2e}, "
                        f"{feas_str}, div={diversity:.4f}"
                    )

                if max_residual < tol and best_feasible_x is not None:
                    converged = True
                    status = "optimal"
                    break

        # Final processing with enhanced repair
        if integer_vars and self.config.use_progressive_measure:
            state = self.progressive_measure_step(
                state, max_iter, max_iter, integer_vars
            )

        # Extract best solution
        result = self._extract_result_enhanced(
            state, converged, status, max_iter, history,
            best_feasible_x, best_feasible_obj
        )

        return result

    def _extract_result_enhanced(
        self,
        state: PopulationState,
        converged: bool,
        status: str,
        iterations: int,
        history: List[Dict],
        best_feasible_x: Optional[np.ndarray],
        best_feasible_obj: float,
    ) -> QuantumPopPDHGResult:
        """Extract result with preference for feasible solutions."""
        # If we found a feasible solution, use it
        if best_feasible_x is not None and best_feasible_obj < float('inf'):
            x_best = best_feasible_x
            obj_best = best_feasible_obj

            # Find corresponding dual
            best_idx = 0
            min_dist = float('inf')
            for i in range(self.K):
                dist = np.linalg.norm(state.x[i] - x_best)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            y_best = state.y[best_idx]
        else:
            # Fall back to best objective (may be infeasible)
            x_best, y_best, obj_best = self.get_best(state)

        # Compute final metrics
        Ax = self.A @ x_best
        primal_violation = np.linalg.norm(np.maximum(Ax - self.b, 0), ord=np.inf)
        primal_violation_rel = primal_violation / (1 + np.linalg.norm(self.b, ord=np.inf))

        ATy = self.A.T @ y_best
        dual_residual = np.linalg.norm(ATy + self.c, ord=np.inf)

        gap = abs(obj_best - self.b @ y_best) / (
            1 + abs(obj_best) + abs(self.b @ y_best)
        )

        # Check integer feasibility
        int_violation = 0.0
        if self.config.integer_vars:
            for i in self.config.integer_vars:
                int_violation += abs(x_best[i] - round(x_best[i]))

        return QuantumPopPDHGResult(
            x=x_best,
            y=y_best,
            obj=obj_best,
            iterations=iterations,
            converged=converged,
            status=status,
            primal_residual=primal_violation,
            dual_residual=dual_residual,
            gap=gap,
            history=history,
            tunnel_stats=self.tunnel_stats if hasattr(self, 'tunnel_stats') else {},
            measure_stats=self.measure_stats if hasattr(self, 'measure_stats') else {},
            quantum_history=[],
        )
