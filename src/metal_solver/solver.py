"""
Metal-Accelerated Quantum-Inspired MIP Solver

Main solver integrating all components for heterogeneous CPU-GPU execution.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import time
import numpy as np
from scipy import sparse

# Core components
from .kernels import PDHGKernels, BatchPDHGStep, TunnelKernels, AdaptiveTunnelStrength
from .integer_rounding import ProgressiveMeasurement, LocalSearchRepair, IntegralityChecker
from .schedulers import HeterogeneousScheduler, AdaptiveBatchSize, PerformanceMonitor


@dataclass
class SolverConfig:
    """Configuration for MetalQuantumMIPSolver."""
    # Population
    population_size: int = 16
    adaptive_population: bool = True

    # Iterations
    max_iter: int = 10000
    check_interval: int = 100

    # PDHG parameters
    step_size_scale: float = 0.99

    # Quantum tunneling
    use_tunneling: bool = True
    tunnel_interval: int = 50
    tunnel_initial_strength: float = 1.0

    # Integer handling
    use_progressive_rounding: bool = True
    rounding_schedule: str = 'cosine'
    rounding_interval: int = 100

    # Local search
    use_local_search: bool = True
    local_search_interval: int = 200
    local_search_max_iter: int = 50

    # Device selection
    prefer_gpu: bool = True

    # Convergence
    tol: float = 1e-6
    stagnation_patience: int = 10


@dataclass
class SolverResult:
    """Result from MetalQuantumMIPSolver."""
    x_best: np.ndarray
    obj_best: float
    is_feasible: bool
    is_integer_feasible: bool
    primal_violation: float
    integrality_violation: float
    iterations: int
    solve_time: float
    tunnel_stats: Dict = field(default_factory=dict)
    device_stats: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)


class MetalQuantumMIPSolver:
    """Quantum-inspired MIP solver with Metal GPU acceleration.

    Features:
    - Heterogeneous CPU-GPU execution (Apple Silicon optimized)
    - Quantum tunneling for global exploration
    - Progressive measurement for integer rounding
    - Local search for feasibility repair
    """

    def __init__(self, problem,
                 config: Optional[SolverConfig] = None):
        """Initialize solver.

        Args:
            problem: LPData object with original problem (will be converted to standard form internally)
            config: Solver configuration
        """
        from core.mps_reader import to_standard_form, LPData

        # Save original problem for feasibility checking
        self.original_problem = problem
        # Store original bounds for coordinate conversion
        self._original_lb = np.asarray(problem.lb, dtype=np.float64) if hasattr(problem, 'lb') else np.zeros(problem.n)
        self._using_bounds_adjustment = not np.allclose(self._original_lb, 0)

        # Convert to standard form for solving
        if hasattr(problem, 'sense'):
            std_problem = to_standard_form(problem)
        else:
            std_problem = problem

        self.A = std_problem.A
        # Use float64 for numerical stability
        self.b = np.asarray(std_problem.b, dtype=np.float64)
        self.c = np.asarray(std_problem.c, dtype=np.float64)
        self.lb = np.asarray(std_problem.lb, dtype=np.float64)
        self.ub = np.asarray(std_problem.ub, dtype=np.float64)
        self.integer_vars = std_problem.integer_vars if hasattr(std_problem, 'integer_vars') else []

        self.config = config or SolverConfig()

        self.m, self.n = self.A.shape

        # Compute problem statistics
        self.problem_size = {
            'n_vars': self.n,
            'n_constrs': self.m,
            'nnz': self.A.nnz
        }

        # Initialize components
        self._init_components()

    def _std_to_orig(self, x_std):
        """Convert solution from standard form to original problem space."""
        x = x_std
        # First undo Ruiz scaling if applied
        if hasattr(self, '_ruiz_scaling_applied') and self._ruiz_scaling_applied:
            x = x * (1.0 / self._ruiz_E if hasattr(self, '_ruiz_E') else 1.0)
        # Then undo bounds adjustment if applied
        if hasattr(self, '_using_bounds_adjustment') and self._using_bounds_adjustment:
            x = x + self._original_lb
        return x

    def _orig_to_std(self, x_orig):
        """Convert solution from original problem space to standard form."""
        x = x_orig
        # First apply bounds adjustment if needed
        if hasattr(self, '_using_bounds_adjustment') and self._using_bounds_adjustment:
            x = x - self._original_lb
        # Then apply Ruiz scaling if applied
        if hasattr(self, '_ruiz_scaling_applied') and self._ruiz_scaling_applied:
            x = x * self._ruiz_E if hasattr(self, '_ruiz_E') else x
        return x

    def _apply_ruiz_scaling(self):
        """Apply simple diagonal scaling to improve problem conditioning.

        Uses a single-pass row and column scaling to normalize matrix entries.
        More stable than iterative Ruiz for poorly conditioned problems.
        """
        A = self.A
        m, n = A.shape

        # Compute row norms (infinity norm)
        row_norms = np.zeros(m)
        for i in range(m):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            if row_end > row_start:
                row_norms[i] = np.max(np.abs(A.data[row_start:row_end]))

        # Clamp to avoid division by zero and extreme values
        row_norms = np.clip(row_norms, 1e-10, 1e10)

        # Row scaling (make each row have max entry ~1)
        D = 1.0 / row_norms
        D = np.clip(D, 1e-10, 1e10)  # Prevent extreme values

        # Apply row scaling: A_scaled = D * A
        A_scaled = A.copy()
        for i in range(m):
            row_start = A_scaled.indptr[i]
            row_end = A_scaled.indptr[i + 1]
            if row_end > row_start:
                A_scaled.data[row_start:row_end] *= D[i]

        # Apply scaling to b: b_scaled = D * b
        b_scaled = self.b * D

        # Store scaling info (no column scaling for stability)
        self.A = A_scaled
        self.b = b_scaled
        self._ruiz_D = D
        self._ruiz_E = np.ones(n)  # Identity for columns
        self._ruiz_scaling_applied = True

        print(f"Applied row scaling: D range [{D.min():.4e}, {D.max():.4e}]")

    def _unscale_solution(self, x_scaled):
        """Unscale solution back to original space.

        For row scaling only: x_original = x_scaled (no change needed)
        """
        if not hasattr(self, '_ruiz_scaling_applied') or not self._ruiz_scaling_applied:
            return x_scaled
        # With row scaling only, solution stays in original space
        return x_scaled

    def _init_components(self):
        """Initialize all solver components."""
        # Determine population size
        if self.config.adaptive_population:
            batch_adjuster = AdaptiveBatchSize(self.config.population_size)
            self.K = batch_adjuster.get_batch_size(self.n, self.m)
        else:
            self.K = self.config.population_size

        print(f"Population size: {self.K}")

        # Apply Ruiz scaling for ill-conditioned problems
        # Check if problem might be ill-conditioned (large range in A or b)
        A_range = self.A.max() - self.A.min()
        b_range = np.max(self.b) - np.min(self.b) if len(self.b) > 0 else 0
        if A_range > 1e3 or b_range > 1e3:
            print(f"Detected potentially ill-conditioned problem (A_range={A_range:.2e}, b_range={b_range:.2e})")
            print("Applying Ruiz scaling...")
            self._apply_ruiz_scaling()
        else:
            self._ruiz_scaling_applied = False

        # Normalize objective to balance with constraints
        norm_c = np.linalg.norm(self.c)
        if norm_c > 1e-10:
            # Scale c to have similar magnitude to constraint terms
            # This prevents objective from dominating PDHG updates
            self.c_scale = min(1.0, 100.0 / norm_c)  # Normalize if c is very large
            self.c = self.c * self.c_scale
            print(f"Normalized objective: scale={self.c_scale:.4f}, norm_c={norm_c:.2f}")
        else:
            self.c_scale = 1.0

        # PDHG step sizes
        norm_A = self._estimate_norm_A()
        self.eta = self.config.step_size_scale / norm_A
        self.tau = self.config.step_size_scale / norm_A

        # Kernels
        self.pdhg_step = BatchPDHGStep(PDHGKernels(use_gpu=self.config.prefer_gpu))
        self.tunnel_kernels = TunnelKernels(use_gpu=self.config.prefer_gpu)
        self.tunnel_strength = AdaptiveTunnelStrength(
            initial_strength=self.config.tunnel_initial_strength
        )

        # Integer handling
        if self.integer_vars:
            self.measurer = ProgressiveMeasurement(
                self.integer_vars,
                schedule=self.config.rounding_schedule
            )
            # Use ORIGINAL problem for repair to avoid standard form conversion issues
            # This ensures feasibility is measured in the original problem space
            if hasattr(self.original_problem, 'sense'):
                # Original problem available - use it for repair
                self.repair = LocalSearchRepair(
                    self.original_problem.A,
                    self.original_problem.b,
                    self.integer_vars,
                    max_iter=self.config.local_search_max_iter,
                    constraint_sense=self.original_problem.sense
                )
            else:
                # Fallback to standard form (should not happen for MIP)
                self.repair = LocalSearchRepair(
                    self.A, self.b, self.integer_vars,
                    max_iter=self.config.local_search_max_iter
                )
            self.int_checker = IntegralityChecker(self.integer_vars)
        else:
            self.measurer = None
            self.repair = None
            self.int_checker = None

        # Scheduler and monitor
        self.scheduler = HeterogeneousScheduler()
        self.monitor = PerformanceMonitor()

        # State
        self.population_x = None
        self.population_y = None
        self.population_obj = None
        self.best_x = None
        self.best_obj = float('inf')
        self.history = []

    def _estimate_norm_A(self) -> float:
        """Estimate ||A||_2 using power iteration."""
        v = np.random.randn(self.n).astype(np.float32)
        v = v / np.linalg.norm(v)

        for _ in range(20):
            u = self.A @ v
            v_new = self.A.T @ u
            norm = np.linalg.norm(v_new)
            if norm > 1e-10:
                v = v_new / norm
            else:
                break

        return float(np.linalg.norm(self.A @ v))

    def _init_population(self, seed: Optional[int] = None):
        """Initialize population with diverse starting points."""
        if seed is not None:
            np.random.seed(seed)

        # Use float64 for numerical stability
        self.population_x = np.zeros((self.K, self.n), dtype=np.float64)
        self.population_y = np.zeros((self.K, self.m), dtype=np.float64)

        # Diverse initialization for primal variables
        for k in range(self.K):
            if k == 0:
                # Zero initialization
                self.population_x[k] = np.clip(
                    np.zeros(self.n), self.lb, self.ub
                )
            elif k < self.K // 2:
                # Random in [0, 0.1]
                perturb = np.random.rand(self.n) * 0.1
                self.population_x[k] = np.clip(perturb, self.lb, self.ub)
            else:
                # Random normal
                perturb = np.random.randn(self.n) * 0.05
                self.population_x[k] = np.clip(perturb, self.lb, self.ub)

        # Diverse initialization for dual variables
        # Different initial dual points help explore different constraint combinations
        for k in range(self.K):
            if k == 0:
                self.population_y[k] = np.zeros(self.m)
            else:
                # Small random initialization for diversity
                scale = 0.01 * k
                self.population_y[k] = np.abs(np.random.randn(self.m) * scale)

        # Initial objective values
        self.population_obj = self.population_x @ self.c

    def _check_original_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check feasibility on original problem (not standard form).

        Returns:
            (is_feasible, max_violation)
        """
        if not hasattr(self.original_problem, 'sense'):
            # No original problem data, use standard form
            Ax = self.A @ x
            viol = np.max(np.maximum(Ax - self.b, 0))
            return viol < 1e-4, viol

        # Check on original problem
        A_orig = self.original_problem.A
        b_orig = self.original_problem.b
        sense = self.original_problem.sense

        Ax = A_orig @ x
        max_viol = 0.0

        for i, s in enumerate(sense):
            if s == '<':
                viol = max(Ax[i] - b_orig[i], 0)
            elif s == '>':
                viol = max(b_orig[i] - Ax[i], 0)
            elif s == '=':
                viol = abs(Ax[i] - b_orig[i])
            else:
                viol = max(Ax[i] - b_orig[i], 0)  # Default to <=

            max_viol = max(max_viol, viol)

        return max_viol < 1e-4, max_viol

    def _select_best(self) -> Tuple[np.ndarray, float]:
        """Select best solution from population."""
        # Compute feasibility on ORIGINAL problem
        feas_scores = []
        for k in range(self.K):
            _, viol = self._check_original_feasibility(self.population_x[k])
            feas_scores.append(viol)

        feas_scores = np.array(feas_scores)

        # Prefer feasible solutions
        feasible_mask = feas_scores < 1e-4

        if np.any(feasible_mask):
            # Among feasible, pick lowest objective
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = feasible_indices[np.argmin(self.population_obj[feasible_indices])]
        else:
            # Strongly penalize infeasibility - use problem-dependent penalty
            # Use max(|obj|) * 100 + max_violation * 10000
            max_obj = np.max(np.abs(self.population_obj))
            penalty_base = max(max_obj * 10, 1000.0)
            combined = self.population_obj + penalty_base * feas_scores * 100
            best_idx = np.argmin(combined)

        return self.population_x[best_idx].copy(), float(self.population_obj[best_idx])

    def _apply_tunneling(self, iteration: int):
        """Apply quantum tunneling to population with feasibility preservation.

        Key insight: Only apply tunneling to infeasible population members,
        and use WKB probability to prefer feasible destinations.
        """
        if not self.config.use_tunneling:
            return

        if iteration % self.config.tunnel_interval != 0:
            return

        # Early phase: don't tunnel, let PDHG converge
        if iteration < self.config.max_iter * 0.2:
            return

        strength = self.tunnel_strength.get_strength()
        success_count = 0

        for k in range(self.K):
            x_current = self.population_x[k]

            # Check current feasibility
            is_feas_current, viol_current = self._check_original_feasibility(x_current)

            # If already feasible, be very conservative
            if is_feas_current and np.random.random() > 0.1:
                # 90% chance to skip tunneling for feasible solutions
                continue

            # Decide tunnel type based on feasibility
            if is_feas_current:
                # For feasible: small population-guided perturbation only
                x_new = self.tunnel_kernels.population_guided_jump(
                    x_current, self.population_x, elite_frac=0.3
                )
                # Small noise
                x_new = x_new + np.random.randn(self.n) * strength * 0.05
            else:
                # For infeasible: can use more aggressive strategies
                r = np.random.random()
                if r < 0.4:
                    x_new = self.tunnel_kernels.levy_flight(x_current, scale=strength)
                elif r < 0.7:
                    x_new = self.tunnel_kernels.population_guided_jump(
                        x_current, self.population_x
                    )
                else:
                    x_new = x_current + np.random.randn(self.n) * strength * 0.1

            # Clip to bounds
            x_new = np.clip(x_new, self.lb, self.ub)

            # Check new feasibility BEFORE WKB acceptance
            is_feas_new, viol_new = self._check_original_feasibility(x_new)

            # WKB energy: balance objective and constraint violation
            # Quantum-inspired: treat violation as potential barrier
            def energy(x):
                obj = self.c @ x
                # Use original problem for energy calculation
                is_feas, max_viol = self._check_original_feasibility(x)
                # Penalty for violation (potential barrier)
                penalty = 1e6 * max_viol if max_viol > 1e-4 else 0
                return obj + penalty

            E_current = energy(x_current)
            E_new = energy(x_new)

            # WKB acceptance with quantum-inspired probability
            accepted, prob = self.tunnel_kernels.wkb_acceptance(
                x_current, x_new, lambda x: energy(x), temperature=strength
            )

            # Additional feasibility-aware criterion
            if is_feas_current and not is_feas_new:
                # Would break feasibility - require higher probability
                if prob < 0.5:  # Must be strong improvement
                    accepted = False

            if accepted:
                self.population_x[k] = x_new
                self.population_obj[k] = self.c @ x_new
                success_count += 1

        # Update tunnel strength
        success_rate = success_count / self.K
        self.tunnel_strength.update(success_rate)

    def _apply_integer_rounding(self, iteration: int):
        """Apply progressive integer rounding, preserving feasibility."""
        if not self.integer_vars:
            return

        if not self.config.use_progressive_rounding:
            return

        if iteration % self.config.rounding_interval != 0:
            return

        # Apply to all population members, but only if rounding preserves feasibility
        for k in range(self.K):
            x_before = self.population_x[k].copy()

            # Apply rounding
            x_rounded = self.measurer.measure(
                x_before, iteration, self.config.max_iter
            )
            x_rounded = np.clip(x_rounded, self.lb, self.ub)

            # Check if rounding preserves feasibility
            is_feas_before, _ = self._check_original_feasibility(x_before)
            is_feas_after, _ = self._check_original_feasibility(x_rounded)

            if is_feas_before and not is_feas_after:
                # Rounding broke feasibility, try to repair
                if self.repair:
                    # Convert to original space for repair
                    x_rounded_orig = self._std_to_orig(x_rounded)
                    x_repaired_orig, repaired_feas = self.repair.repair(x_rounded_orig)
                    # Convert back to standard form
                    x_repaired = self._orig_to_std(x_repaired_orig)
                    x_repaired = np.clip(x_repaired, self.lb, self.ub)

                    is_orig_feas, _ = self._check_original_feasibility(x_repaired)

                    if is_orig_feas:
                        self.population_x[k] = x_repaired
                    else:
                        # Keep original if repair fails
                        self.population_x[k] = x_before
                else:
                    # No repair available, keep original
                    self.population_x[k] = x_before
            else:
                # Rounding is fine or original was already infeasible
                self.population_x[k] = x_rounded

        # Update objectives
        self.population_obj = self.population_x @ self.c

    def _apply_local_search(self):
        """Apply local search to best solution."""
        if not self.integer_vars:
            return

        if not self.config.use_local_search:
            return

        # Find current best
        x_best, _ = self._select_best()

        # Convert from standard form to original space for repair
        x_best_orig = self._std_to_orig(x_best)

        # Try to repair on ORIGINAL problem
        x_repaired_orig, is_feas = self.repair.repair(x_best_orig)

        # Convert back to standard form
        x_repaired = self._orig_to_std(x_repaired_orig)
        x_repaired = np.clip(x_repaired, self.lb, self.ub)

        if is_feas:
            # Verify on original problem
            is_orig_feas, _ = self._check_original_feasibility(x_repaired)
            if is_orig_feas:
                # Replace worst population member
                worst_idx = np.argmax(self.population_obj)
                self.population_x[worst_idx] = x_repaired
                self.population_obj[worst_idx] = self.c @ x_repaired

    def _restore_feasibility(self, iteration: int):
        """Periodically restore feasibility for infeasible population members.

        Uses aggressive dual gradient projection to push solutions toward feasible region.
        """
        # Only apply every 50 iterations
        if iteration % 50 != 0:
            return

        for k in range(self.K):
            # Check feasibility on original problem
            is_feas, viol = self._check_original_feasibility(self.population_x[k])
            if is_feas:
                continue

            # Aggressive feasibility restoration using multiple steps
            x_current = self.population_x[k].copy()

            for _ in range(20):  # Multiple projection steps
                # Check current violation
                Ax = self.A @ x_current
                violation = np.maximum(Ax - self.b, 0)
                max_viol = np.max(violation)

                if max_viol < 1e-6:
                    break

                # Gradient of constraint violation
                grad_viol = self.A.T @ violation
                grad_norm = np.linalg.norm(grad_viol)

                if grad_norm < 1e-10:
                    break

                # Adaptive step size based on violation magnitude
                step_size = min(1.0, max_viol / (grad_norm + 1e-10))
                x_current -= step_size * grad_viol

                # Clip to bounds
                x_current = np.clip(x_current, self.lb, self.ub)

            # Only accept if improved
            new_viol = np.maximum(self.A @ x_current - self.b, 0).max()
            if new_viol < viol:
                self.population_x[k] = x_current

        # Update objectives after restoration
        self.population_obj = self.population_x @ self.c

    def solve(self, seed: Optional[int] = None,
              verbose: bool = False) -> SolverResult:
        """Solve MIP using Metal-accelerated quantum-inspired algorithm.

        Args:
            seed: Random seed
            verbose: Print progress

        Returns:
            SolverResult with solution and statistics
        """
        start_time = time.time()

        # Initialize
        self._init_population(seed)
        stagnation_count = 0
        prev_best_obj = float('inf')

        if verbose:
            print(f"Starting solve: n={self.n}, m={self.m}, K={self.K}")
            print(f"Integer vars: {len(self.integer_vars)}")

        # Track feasibility for restart detection
        prev_feas_count = 0
        best_feasible_x = None
        best_feasible_obj = float('inf')

        for iteration in range(1, self.config.max_iter + 1):
            iter_start = time.time()

            # 1. PDHG step
            device = self.scheduler.decide_device(self.problem_size, 'pdhg')
            self.population_x, self.population_y = self.pdhg_step.step(
                self.population_x, self.population_y,
                self.A, self.b, self.c,
                self.eta, self.tau, self.lb, self.ub
            )

            # Update objectives
            self.population_obj = self.population_x @ self.c

            # 2. Quantum tunneling
            self._apply_tunneling(iteration)

            # 3. Integer rounding
            self._apply_integer_rounding(iteration)

            # 4. Local search (periodic)
            if iteration % self.config.local_search_interval == 0:
                self._apply_local_search()

            # 5. Feasibility restoration (periodic)
            self._restore_feasibility(iteration)

            # 6. Select best
            self.best_x, self.best_obj = self._select_best()

            # Track best feasible solution found
            is_feas_now, _ = self._check_original_feasibility(self.best_x)
            if is_feas_now and self.best_obj < best_feasible_obj:
                best_feasible_x = self.best_x.copy()
                best_feasible_obj = float(self.best_obj)

            # 7. Restart mechanism: detect divergence and restart
            if iteration % 100 == 0:
                feas_count = sum(
                    1 for k in range(self.K)
                    if self._check_original_feasibility(self.population_x[k])[0]
                )

                # Compute primal-dual residual for best solution
                if self.best_x is not None:
                    Ax = self.A @ self.best_x
                    primal_residual = np.linalg.norm(np.maximum(Ax - self.b, 0))
                    # Dual residual: ||c + A^T y|| (projected to bounds)
                    grad = self.c + self.A.T @ self.population_y[0]
                    dual_residual = np.linalg.norm(grad)
                    total_residual = primal_residual + dual_residual
                else:
                    total_residual = float('inf')

                # Track residual history
                if not hasattr(self, '_residual_history'):
                    self._residual_history = []
                self._residual_history.append(total_residual)

                # If feasibility degraded OR residual increased significantly, restart
                residual_grew = False
                if len(self._residual_history) >= 3:
                    # Check if residual is trending up
                    recent_avg = sum(self._residual_history[-3:]) / 3
                    older_avg = sum(self._residual_history[-6:-3]) / 3 if len(self._residual_history) >= 6 else recent_avg
                    residual_grew = recent_avg > older_avg * 1.5 and iteration > 500

                should_restart = (
                    (feas_count < prev_feas_count - 1 and prev_feas_count >= 2) or
                    residual_grew
                )

                if should_restart and best_feasible_x is not None:
                    if verbose:
                        if residual_grew:
                            print(f"Iter {iteration}: Restarting due to residual growth")
                        else:
                            print(f"Iter {iteration}: Restarting due to divergence "
                                  f"(feas {prev_feas_count} -> {feas_count})")

                    # Restart: reset dual variables, keep primal close to best feasible
                    for k in range(self.K):
                        # Start from best feasible with varying perturbations
                        if k == 0:
                            self.population_x[k] = best_feasible_x.copy()
                        else:
                            # Increasing perturbation for diversity
                            scale = 0.01 * k
                            noise = np.random.randn(self.n) * scale
                            x_perturbed = best_feasible_x + noise
                            self.population_x[k] = np.clip(
                                x_perturbed, self.lb, self.ub
                            )
                        self.population_y[k] = np.zeros(self.m, dtype=np.float64)

                    self.population_obj = self.population_x @ self.c

                    # Clear residual history after restart
                    self._residual_history = []

                    # Force feasibility check after restart
                    feas_count = sum(
                        1 for k in range(self.K)
                        if self._check_original_feasibility(self.population_x[k])[0]
                    )

                prev_feas_count = feas_count

            # Record iteration time
            iter_time = time.time() - iter_start
            self.monitor.record_iteration(
                float(self.best_obj), False, iter_time
            )

            # Check convergence
            if iteration % self.config.check_interval == 0:
                # Check feasibility on ORIGINAL problem
                is_feas, primal_viol = self._check_original_feasibility(self.best_x)

                # Check integrality
                if self.int_checker:
                    int_viol = self.int_checker.check(self.best_x)
                    is_int_feas = self.int_checker.is_integer_feasible(self.best_x)
                else:
                    int_viol = 0.0
                    is_int_feas = True

                self.history.append({
                    'iteration': iteration,
                    'obj': float(self.best_obj),
                    'primal_violation': float(primal_viol),
                    'integrality_violation': int_viol,
                    'is_feasible': bool(is_feas),
                    'is_integer_feasible': is_int_feas
                })

                if verbose:
                    print(f"Iter {iteration}: obj={self.best_obj:.4f}, "
                          f"feas={is_feas}, int_feas={is_int_feas}")

                # Stagnation check
                if abs(prev_best_obj - self.best_obj) < self.config.tol:
                    stagnation_count += 1
                    if stagnation_count >= self.config.stagnation_patience:
                        if verbose:
                            print(f"Stagnated at iteration {iteration}")
                        break
                else:
                    stagnation_count = 0

                prev_best_obj = self.best_obj

        # Finalize
        solve_time = time.time() - start_time

        # Final rounding - only if it preserves feasibility
        if self.measurer:
            x_rounded = self.measurer.finalize(self.best_x)
            is_feas_rounded, viol_rounded = self._check_original_feasibility(x_rounded)

            if is_feas_rounded:
                # Rounding maintains feasibility, use it
                self.best_x = x_rounded
                self.best_obj = float(self.c @ self.best_x)
            else:
                # Rounding breaks feasibility - try to repair
                # Convert to original space for repair
                x_rounded_orig = self._std_to_orig(x_rounded)
                x_repaired_orig, repaired_feas = self.repair.repair(x_rounded_orig)
                # Convert back to standard form
                x_repaired = self._orig_to_std(x_repaired_orig)
                x_repaired = np.clip(x_repaired, self.lb, self.ub)

                is_orig_feas, _ = self._check_original_feasibility(x_repaired)

                if is_orig_feas:
                    # Repaired version is feasible
                    self.best_x = x_repaired
                    self.best_obj = float(self.c @ self.best_x)
                # else: keep the original fractional solution (it's feasible)

        # Try to repair ALL population members to find a feasible solution
        if self.repair and self.integer_vars:
            # First check if best is already feasible
            is_feas_check, _ = self._check_original_feasibility(self.best_x)

            if not is_feas_check:
                # Try to repair best first
                # Convert to original space for repair
                best_x_orig = self._std_to_orig(self.best_x)
                x_repaired_orig, repaired_feas = self.repair.repair(best_x_orig)
                # Convert back to standard form
                x_repaired = self._orig_to_std(x_repaired_orig)
                x_repaired = np.clip(x_repaired, self.lb, self.ub)

                if repaired_feas:
                    is_orig_feas, _ = self._check_original_feasibility(x_repaired)
                    if is_orig_feas:
                        self.best_x = x_repaired
                        self.best_obj = float(self.c @ self.best_x)

                # If still not feasible, try all population members
                is_feas_check, _ = self._check_original_feasibility(self.best_x)
                if not is_feas_check:
                    best_feas_obj = float('inf')
                    best_feas_x = None

                    for k in range(self.K):
                        x_k = self.population_x[k].copy()
                        # Round to integers first
                        for i in self.integer_vars:
                            x_k[i] = np.round(x_k[i])

                        # Convert to original space for repair
                        x_k_orig = self._std_to_orig(x_k)
                        x_repaired_orig, is_feas = self.repair.repair(x_k_orig)
                        # Convert back to standard form
                        x_repaired = self._orig_to_std(x_repaired_orig)
                        x_repaired = np.clip(x_repaired, self.lb, self.ub)

                        if is_feas:
                            # Verify on original problem
                            is_orig_feas, _ = self._check_original_feasibility(x_repaired)
                            if is_orig_feas:
                                obj = float(self.c @ x_repaired)
                                if obj < best_feas_obj:
                                    best_feas_obj = obj
                                    best_feas_x = x_repaired

                    if best_feas_x is not None:
                        self.best_x = best_feas_x
                        self.best_obj = best_feas_obj

        # Final checks on ORIGINAL problem
        is_feas, primal_viol = self._check_original_feasibility(self.best_x)

        if self.int_checker:
            int_viol = self.int_checker.check(self.best_x)
            is_int_feas = self.int_checker.is_integer_feasible(self.best_x)
        else:
            int_viol = 0.0
            is_int_feas = True

        # Unscale solution from Ruiz scaling if applied
        if hasattr(self, '_ruiz_scaling_applied') and self._ruiz_scaling_applied:
            self.best_x = self._unscale_solution(self.best_x)
            # Re-check feasibility on original unscaled solution
            is_feas, primal_viol = self._check_original_feasibility(self.best_x)
            if self.int_checker:
                int_viol = self.int_checker.check(self.best_x)
                is_int_feas = self.int_checker.is_integer_feasible(self.best_x)
            # Recompute objective (should be consistent with original problem)
            self.best_obj = float(self.c @ (self.best_x * (self._ruiz_E if hasattr(self, '_ruiz_E') else 1)))

        # Unscale objective for reporting
        obj_unscaled = float(self.best_obj) / self.c_scale

        # Also unscale history
        for h in self.history:
            h['obj'] = float(h['obj']) / self.c_scale

        return SolverResult(
            x_best=self.best_x,
            obj_best=obj_unscaled,
            is_feasible=is_feas,
            is_integer_feasible=is_int_feas,
            primal_violation=primal_viol,
            integrality_violation=int_viol,
            iterations=iteration,
            solve_time=solve_time,
            tunnel_stats=self.tunnel_strength.__dict__,
            device_stats=self.scheduler.get_stats(),
            history=self.history
        )


# Convenience function
def solve_mip(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray,
              lb: np.ndarray, ub: np.ndarray,
              integer_vars: Optional[List[int]] = None,
              **kwargs) -> SolverResult:
    """Solve MIP using Metal-accelerated quantum-inspired algorithm."""
    config = SolverConfig(**kwargs)
    solver = MetalQuantumMIPSolver(A, b, c, lb, ub, integer_vars, config)
    return solver.solve()


if __name__ == "__main__":
    print("Testing MetalQuantumMIPSolver...")

    # Test problem
    A = sparse.csr_matrix([[1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0]])
    b = np.array([1.0, 1.0])
    c = np.array([-1.0, -2.0, -1.0])
    lb = np.zeros(3)
    ub = np.ones(3) * 10

    config = SolverConfig(
        population_size=4,
        max_iter=1000,
        use_tunneling=True
    )

    solver = MetalQuantumMIPSolver(A, b, c, lb, ub, None, config)
    result = solver.solve(seed=42, verbose=True)

    print(f"\nResult:")
    print(f"  x = {result.x_best}")
    print(f"  obj = {result.obj_best:.4f}")
    print(f"  feasible = {result.is_feasible}")
    print(f"  time = {result.solve_time:.4f}s")
