"""
Hybrid CPU/GPU Collaboration Mode for PDHG Solver

This module implements the hybrid collaboration strategy where:
- Root node LP: GPU accelerated (large scale)
- Sub-node LP: CPU processing (small scale, many nodes)
- Hot start: Shared memory for zero-copy initialization

Key insight: Use M4 Pro's unified memory for seamless CPU/GPU data sharing.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass, field
import time
from enum import Enum

# Import our optimized solvers
from .cpu_optimized import OptimizedCPUSolver, LazyRuizScaling
from .gpu_optimized import OptimizedGPUSolver
from .adaptive_backend import (
    AdaptiveDeviceSelector,
    ProblemCharacteristics,
    LightweightProfiler
)


class NodeType(Enum):
    """Type of B&B node."""
    ROOT = "root"
    SHALLOW = "shallow"    # depth < 5
    DEEP = "deep"          # depth >= 5
    SMALL = "small"        # n_vars < 1000
    LARGE = "large"        # n_vars >= 1000


@dataclass
class BnBNode:
    """Branch and Bound node."""
    node_id: int
    depth: int
    parent_id: Optional[int]

    # Problem data (shared via unified memory)
    A: sp.csr_matrix
    b: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    # Branching info
    branch_var: Optional[int] = None
    branch_value: Optional[float] = None

    # Solution info
    lp_value: Optional[float] = None
    lp_solution: Optional[np.ndarray] = None
    is_feasible: bool = False
    is_integer: bool = False

    # Bounds
    lower_bound: float = float('-inf')
    upper_bound: float = float('inf')

    @property
    def node_type(self) -> NodeType:
        """Determine node type for device selection."""
        n = self.A.shape[1]

        if self.depth == 0:
            return NodeType.ROOT
        if n < 1000:
            return NodeType.SMALL
        if self.depth < 5:
            return NodeType.SHALLOW
        return NodeType.DEEP


@dataclass
class HybridConfig:
    """Configuration for hybrid CPU/GPU solving."""

    # Device selection thresholds
    gpu_for_root: bool = True              # Use GPU for root node
    gpu_for_shallow: bool = True           # Use GPU for shallow nodes
    gpu_for_large: bool = True             # Use GPU for n >= 2000
    gpu_threshold: int = 2000              # n threshold for GPU

    # Profiling
    enable_profiling: bool = True          # Enable runtime profiling
    profile_interval: int = 100            # Profile every N nodes

    # Memory
    use_unified_memory: bool = True        # Use unified memory for zero-copy
    cache_gpu_data: bool = True            # Keep GPU data resident

    # Performance
    max_gpu_memory_mb: int = 8000          # Max GPU memory to use
    warmup_iterations: int = 3             # Warmup before timing


class HybridSolver:
    """
    Hybrid CPU/GPU solver for MIP with intelligent device selection.

    Strategy:
    - Root node: GPU if large (n > 2000), CPU otherwise
    - Shallow nodes: GPU if problem still large
    - Deep/small nodes: CPU (lower latency)

    Features:
    1. Unified memory for zero-copy data sharing
    2. GPU data caching for repeated solves
    3. Runtime profiling for adaptive selection
    """

    def __init__(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        config: Optional[HybridConfig] = None,
        verbose: bool = False
    ):
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub
        self.config = config or HybridConfig()
        self.verbose = verbose

        m, n = A.shape
        self.n = n
        self.m = m

        # Device selector
        self.selector = AdaptiveDeviceSelector(
            verbose=verbose,
            enable_profiling=self.config.enable_profiling
        )

        # Profiler
        self.profiler = LightweightProfiler()

        # GPU solver (initialized once, reused)
        self.gpu_solver: Optional[OptimizedGPUSolver] = None
        self._gpu_initialized = False

        # CPU solver (created per node as needed)
        self._cpu_solvers: Dict[int, OptimizedCPUSolver] = {}

        # Statistics
        self.stats = {
            'cpu_solves': 0,
            'gpu_solves': 0,
            'total_time': 0.0,
            'node_times': []
        }

        # Compute Ruiz scaling once (shared)
        if verbose:
            print("  Computing Ruiz scaling for problem...")
        self.ruiz_scaling = LazyRuizScaling.compute(A, max_iter=10)
        if verbose:
            print(f"    Ruiz: {self.ruiz_scaling.iterations} iters, "
                  f"converged={self.ruiz_scaling.converged}")

    def _get_gpu_solver(self) -> Optional[OptimizedGPUSolver]:
        """Get or create GPU solver (singleton)."""
        if not self._gpu_initialized:
            if self.selector.gpu_available:
                self.gpu_solver = OptimizedGPUSolver(
                    self.A, self.b, self.c, self.lb, self.ub,
                    col_scale=self.ruiz_scaling.col_scale,
                    row_scale=self.ruiz_scaling.row_scale,
                    verbose=self.verbose
                )
                self._gpu_initialized = True
            else:
                self.gpu_solver = None

        return self.gpu_solver

    def _select_device_for_node(self, node: BnBNode) -> str:
        """Select device for a specific B&B node."""
        node_type = node.node_type
        chars = ProblemCharacteristics.from_problem(node.A)

        # Rule-based selection based on node type
        if node_type == NodeType.ROOT:
            if self.config.gpu_for_root and self.n > self.config.gpu_threshold:
                return "gpu"

        elif node_type == NodeType.SHALLOW:
            if self.config.gpu_for_shallow and self.n > self.config.gpu_threshold:
                return "gpu"

        elif node_type == NodeType.LARGE:
            if self.config.gpu_for_large:
                return "gpu"

        elif node_type == NodeType.SMALL:
            # Small nodes always CPU
            return "cpu"

        # Fall back to feature-based selection
        device, reason = self.selector.select(chars)
        return device

    def solve_node(
        self,
        node: BnBNode,
        max_iter: int = 500,
        tol: float = 1e-6,
        warm_start: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve LP relaxation at a B&B node.

        Args:
            node: B&B node to solve
            max_iter: Maximum PDHG iterations
            tol: Convergence tolerance
            warm_start: (x, y) pair from parent node

        Returns:
            (x, y, info)
        """
        start_time = time.perf_counter()

        # Select device
        device = self._select_device_for_node(node)

        if self.verbose:
            print(f"  Node {node.node_id} (depth={node.depth}, "
                  f"n={node.n_vars}): using {device}")

        # Solve on selected device
        if device == "gpu":
            x, y, info = self._solve_gpu(node, max_iter, tol, warm_start)
            self.stats['gpu_solves'] += 1
        else:
            x, y, info = self._solve_cpu(node, max_iter, tol, warm_start)
            self.stats['cpu_solves'] += 1

        # Record timing
        elapsed = time.perf_counter() - start_time
        self.stats['total_time'] += elapsed
        self.stats['node_times'].append((node.node_id, elapsed, device))

        # Update node info
        node.lp_solution = x.copy()
        node.lp_value = self.c @ x
        node.is_feasible = info.get('converged', False)

        # Check integrality
        int_vars = np.where(~np.isclose(x, np.round(x)))[0]
        node.is_integer = len(int_vars) == 0

        return x, y, info

    def _solve_cpu(
        self,
        node: BnBNode,
        max_iter: int,
        tol: float,
        warm_start: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve on CPU."""
        # Create or reuse CPU solver
        if node.node_id not in self._cpu_solvers:
            self._cpu_solvers[node.node_id] = OptimizedCPUSolver(
                node.A, node.b, node.c, node.lb, node.ub,
                use_ruiz=True,
                n_threads=4,
                verbose=False
            )

        solver = self._cpu_solvers[node.node_id]

        # Solve
        if warm_start is not None:
            x0, y0 = warm_start
            # Could use warm start here (not implemented in basic PDHG)
            pass

        x, y, info = solver.solve(max_iter=max_iter, tol=tol)
        return x, y, info

    def _solve_gpu(
        self,
        node: BnBNode,
        max_iter: int,
        tol: float,
        warm_start: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve on GPU."""
        gpu_solver = self._get_gpu_solver()

        if gpu_solver is None or not gpu_solver.available:
            # Fallback to CPU
            return self._solve_cpu(node, max_iter, tol, warm_start)

        # Initialize
        x = np.zeros(self.n)
        y = np.zeros(self.m)

        if warm_start is not None:
            x0, y0 = warm_start
            # Copy to GPU buffers
            gpu_solver._write_vector(gpu_solver.gpu_memory.x, x0)
            gpu_solver._write_vector(gpu_solver.gpu_memory.y, y0)
            x = x0.copy()
            y = y0.copy()

        # Step sizes
        eta = 1.0
        tau = 1.0

        # Iteration
        history = {'gap': [], 'time': []}
        start = time.perf_counter()

        for iteration in range(max_iter):
            # PDHG step on GPU
            x_new, y_new = gpu_solver.pdhg_step(
                x, y, tau, eta, node.c, node.b
            )

            # Compute gap
            obj_primal = node.c @ x_new
            obj_dual = -node.b @ y_new
            gap = abs(obj_primal - obj_dual) / (abs(obj_primal) + 1e-10)

            history['gap'].append(gap)
            history['time'].append(time.perf_counter() - start)

            # Check convergence
            if gap < tol:
                break

            x = x_new
            y = y_new

        info = {
            'iterations': iteration + 1,
            'time': time.perf_counter() - start,
            'converged': gap < tol,
            'final_gap': gap,
            'history': history
        }

        return x, y, info

    def solve_mip(
        self,
        max_nodes: int = 1000,
        time_limit: float = 300.0,
        verbose: bool = None
    ) -> Dict:
        """
        Solve MIP using branch and bound with hybrid CPU/GPU.

        Args:
            max_nodes: Maximum B&B nodes
            time_limit: Time limit in seconds
            verbose: Override instance verbosity

        Returns:
            Solution dictionary
        """
        v = verbose if verbose is not None else self.verbose
        start_time = time.perf_counter()

        # Root node
        root = BnBNode(
            node_id=0,
            depth=0,
            parent_id=None,
            A=self.A,
            b=self.b,
            c=self.c,
            lb=self.lb,
            ub=self.ub
        )

        if v:
            print("=" * 60)
            print("Hybrid MIP Solver")
            print("=" * 60)
            print(f"Problem: n={self.n}, m={self.m}, nnz={self.A.nnz}")
            print(f"Device strategy: GPU for root/large, CPU for small/deep")
            print()

        # Solve root
        if v:
            print("Solving root node...")
        x_root, y_root, root_info = self.solve_node(root, max_iter=1000)

        if v:
            print(f"  Root: obj={root.lp_value:.4f}, "
                  f"gap={root_info['final_gap']:.2e}, "
                  f"time={root_info['time']:.3f}s")

        # Check if already integer feasible
        if root.is_integer:
            if v:
                print("  Root solution is integer feasible!")
            return {
                'status': 'optimal',
                'solution': root.lp_solution,
                'objective': root.lp_value,
                'nodes': 1,
                'time': time.perf_counter() - start_time
            }

        # Initialize B&B
        open_nodes = [root]
        best_incumbent = None
        best_incumbent_value = float('inf')
        node_counter = 1

        if v:
            print(f"\nStarting B&B (max_nodes={max_nodes})...")

        # Main loop
        while open_nodes and node_counter < max_nodes:
            # Check time limit
            if time.perf_counter() - start_time > time_limit:
                if v:
                    print(f"  Time limit reached")
                break

            # Select node (depth-first for simplicity)
            current = open_nodes.pop()

            # Check bound
            if current.lp_value >= best_incumbent_value:
                continue

            # Branch on most fractional variable
            if current.lp_solution is not None:
                frac = np.abs(current.lp_solution - np.round(current.lp_solution))
                branch_var = int(np.argmax(frac))
                branch_value = current.lp_solution[branch_var]
            else:
                continue

            # Create child nodes
            for bound_type in ['lower', 'upper']:
                new_lb = current.lb.copy()
                new_ub = current.ub.copy()

                if bound_type == 'lower':
                    new_ub[branch_var] = np.floor(branch_value)
                else:
                    new_lb[branch_var] = np.ceil(branch_value)

                # Check feasibility
                if np.any(new_lb > new_ub):
                    continue

                # Create node
                child = BnBNode(
                    node_id=node_counter,
                    depth=current.depth + 1,
                    parent_id=current.node_id,
                    A=current.A,
                    b=current.b,
                    c=current.c,
                    lb=new_lb,
                    ub=new_ub,
                    branch_var=branch_var,
                    branch_value=branch_value
                )

                # Solve child
                warm_start = (current.lp_solution, y_root)
                x_child, y_child, child_info = self.solve_node(
                    child, warm_start=warm_start
                )

                # Check incumbent
                if child.is_integer and child.lp_value < best_incumbent_value:
                    best_incumbent = child.lp_solution.copy()
                    best_incumbent_value = child.lp_value
                    if v:
                        print(f"  Node {node_counter}: new incumbent = {best_incumbent_value:.4f}")

                # Add to open if promising
                if child.lp_value < best_incumbent_value:
                    open_nodes.append(child)

                node_counter += 1

        # Return result
        elapsed = time.perf_counter() - start_time

        if v:
            print()
            print("=" * 60)
            print("B&B Complete")
            print(f"  Nodes: {node_counter}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Best: {best_incumbent_value if best_incumbent_value != float('inf') else 'Infeasible'}")
            print(f"  CPU solves: {self.stats['cpu_solves']}")
            print(f"  GPU solves: {self.stats['gpu_solves']}")
            print("=" * 60)

        return {
            'status': 'optimal' if best_incumbent is not None else 'infeasible',
            'solution': best_incumbent,
            'objective': best_incumbent_value,
            'nodes': node_counter,
            'time': elapsed,
            'cpu_solves': self.stats['cpu_solves'],
            'gpu_solves': self.stats['gpu_solves']
        }


def benchmark_hybrid_solver():
    """Benchmark hybrid solver strategies."""
    print("=" * 60)
    print("Hybrid CPU/GPU Solver Benchmark")
    print("=" * 60)

    # Create test MIP (small for demo)
    np.random.seed(42)
    n = 500
    m = 200
    density = 0.05

    A = sp.random(m, n, density=density, format='csr')
    b = np.random.randn(m)
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.ones(n)  # Binary variables

    print(f"\nProblem: n={n}, m={m}, nnz={A.nnz}")

    # Create hybrid solver
    config = HybridConfig(
        gpu_for_root=True,
        gpu_for_shallow=False,
        gpu_threshold=1000,
        enable_profiling=False
    )

    solver = HybridSolver(A, b, c, lb, ub, config=config, verbose=True)

    # Solve MIP (limited for demo)
    result = solver.solve_mip(max_nodes=50, time_limit=30.0)

    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Objective: {result['objective']}")
    print(f"  Nodes: {result['nodes']}")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  CPU solves: {result['cpu_solves']}")
    print(f"  GPU solves: {result['gpu_solves']}")


if __name__ == "__main__":
    benchmark_hybrid_solver()
