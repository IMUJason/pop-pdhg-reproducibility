"""
CPU Path Optimizations for Heterogeneous PDHG Solver

This module provides optimized CPU implementations for the main computation path.

Optimizations:
1. Lazy Ruiz Scaling - compute scaling factors without modifying matrix
2. Memory Pre-allocation - avoid repeated allocations
3. SIMD Vectorization - leverage Accelerate BLAS
4. Parallel Preprocessing - multi-threaded Ruiz factor computation
5. Hybrid CSR/CSC Storage - avoid runtime transpose conversion
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time

# Import hybrid storage (optional)
try:
    from .hybrid_storage import HybridSparseMatrix
    HYBRID_STORAGE_AVAILABLE = True
except ImportError:
    HYBRID_STORAGE_AVAILABLE = False


@dataclass
class LazyRuizScaling:
    """
    Lazy Ruiz Scaling - compute scaling factors without modifying the matrix.

    Memory: O(m+n) instead of O(nnz)

    Key insight: Instead of creating A_scaled = R @ A @ C, we store
    row_scale and col_scale vectors and apply them during SpMV.
    """
    row_scale: np.ndarray  # shape (m,)
    col_scale: np.ndarray  # shape (n,)
    converged: bool
    iterations: int

    @classmethod
    def compute(
        cls,
        A: sp.csr_matrix,
        max_iter: int = 10,
        tol: float = 1e-8,
        n_threads: int = 1
    ) -> 'LazyRuizScaling':
        """
        Compute Ruiz scaling factors iteratively.

        Ruiz scaling equilibrates row and column norms to 1.

        Args:
            A: Sparse matrix in CSR format
            max_iter: Maximum iterations
            tol: Convergence tolerance
            n_threads: Number of threads for parallel computation

        Returns:
            LazyRuizScaling object with row_scale and col_scale
        """
        m, n = A.shape

        # Initialize scaling factors
        row_scale = np.ones(m)
        col_scale = np.ones(n)

        # Pre-compute row and column structures
        row_nnz = np.diff(A.indptr)

        converged = False
        for iteration in range(max_iter):
            # Compute row norms
            if n_threads > 1 and m > 1000:
                row_norms = _parallel_row_norms(A.data, A.indices, A.indptr, m, n_threads)
            else:
                row_norms = _sequential_row_norms(A.data, A.indices, A.indptr, m)

            # Apply row scaling
            row_scale *= 1.0 / np.maximum(row_norms, 1e-10)

            # Compute column norms (with row scaling applied)
            col_norms = _column_norms_with_scaling(
                A.data, A.indices, A.indptr, n, row_scale
            )

            # Apply column scaling
            col_scale *= 1.0 / np.maximum(col_norms, 1e-10)

            # Check convergence
            row_change = np.max(np.abs(row_norms - 1.0))
            col_change = np.max(np.abs(col_norms - 1.0))

            if row_change < tol and col_change < tol:
                converged = True
                break

        return cls(
            row_scale=row_scale,
            col_scale=col_scale,
            converged=converged,
            iterations=iteration + 1
        )

    def apply_to_vector(self, x: np.ndarray, scale: str = 'col') -> np.ndarray:
        """Apply scaling to a vector."""
        if scale == 'col':
            return self.col_scale * x
        elif scale == 'row':
            return self.row_scale * x
        else:
            raise ValueError(f"Unknown scale type: {scale}")


def _sequential_row_norms(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    m: int
) -> np.ndarray:
    """Compute row norms sequentially."""
    row_norms = np.zeros(m)
    for i in range(m):
        start, end = indptr[i], indptr[i + 1]
        row_data = data[start:end]
        row_norms[i] = np.linalg.norm(row_data)
    return row_norms


def _parallel_row_norms(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    m: int,
    n_threads: int
) -> np.ndarray:
    """Compute row norms in parallel using chunking."""
    from concurrent.futures import ThreadPoolExecutor

    chunk_size = (m + n_threads - 1) // n_threads

    def compute_chunk(start_row: int, end_row: int) -> np.ndarray:
        norms = np.zeros(end_row - start_row)
        for i in range(start_row, end_row):
            start, end = indptr[i], indptr[i + 1]
            row_data = data[start:end]
            norms[i - start_row] = np.linalg.norm(row_data)
        return norms

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, m)
            if start < end:
                futures.append(executor.submit(compute_chunk, start, end))

        results = []
        for future in futures:
            results.append(future.result())

    return np.concatenate(results)


def _column_norms_with_scaling(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n: int,
    row_scale: np.ndarray
) -> np.ndarray:
    """Compute column norms with row scaling applied."""
    col_norms_sq = np.zeros(n)

    # Iterate over all non-zeros
    m = len(indptr) - 1
    for row_idx in range(m):
        start, end = indptr[row_idx], indptr[row_idx + 1]
        rs = row_scale[row_idx]

        for k in range(start, end):
            col_idx = indices[k]
            val = data[k] * rs
            col_norms_sq[col_idx] += val * val

    return np.sqrt(col_norms_sq)


@dataclass
class PreallocatedWorkspace:
    """
    Pre-allocated workspace for PDHG iterations.

    Avoids repeated memory allocations during solve.
    """
    # Primal variables
    x: np.ndarray
    x_new: np.ndarray

    # Dual variables
    y: np.ndarray
    y_new: np.ndarray

    # Gradient buffers
    grad_x: np.ndarray
    grad_y: np.ndarray

    # Residual buffers
    primal_residual: np.ndarray
    dual_residual: np.ndarray

    # Ruiz scaling (optional)
    row_scale: Optional[np.ndarray] = None
    col_scale: Optional[np.ndarray] = None

    @classmethod
    def create(
        cls,
        n_vars: int,
        n_constraints: int,
        batch_size: int = 1,
        dtype: np.dtype = np.float64
    ) -> 'PreallocatedWorkspace':
        """Create workspace with pre-allocated arrays."""
        if batch_size == 1:
            x_shape = (n_vars,)
            y_shape = (n_constraints,)
        else:
            x_shape = (batch_size, n_vars)
            y_shape = (batch_size, n_constraints)

        return cls(
            x=np.zeros(x_shape, dtype=dtype),
            x_new=np.zeros(x_shape, dtype=dtype),
            y=np.zeros(y_shape, dtype=dtype),
            y_new=np.zeros(y_shape, dtype=dtype),
            grad_x=np.zeros(x_shape, dtype=dtype),
            grad_y=np.zeros(y_shape, dtype=dtype),
            primal_residual=np.zeros(y_shape, dtype=dtype),
            dual_residual=np.zeros(x_shape, dtype=dtype)
        )

    def reset(self):
        """Reset all buffers to zero."""
        for attr in ['x', 'x_new', 'y', 'y_new', 'grad_x', 'grad_y',
                     'primal_residual', 'dual_residual']:
            arr = getattr(self, attr)
            if arr is not None:
                arr.fill(0)


class OptimizedCPUSolver:
    """
    Optimized CPU solver for PDHG with all enhancements.

    Features:
    1. Lazy Ruiz scaling (no matrix modification)
    2. Pre-allocated workspace
    3. SIMD-vectorized operations (via NumPy/Accelerate)
    4. Parallel preprocessing
    """

    def __init__(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        use_ruiz: bool = True,
        use_csc: bool = True,  # NEW: Enable hybrid storage
        n_threads: int = 1,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.n_threads = n_threads

        # Store matrix with hybrid CSR/CSC storage (if enabled)
        if use_csc and HYBRID_STORAGE_AVAILABLE:
            self.A_storage = HybridSparseMatrix(A, use_csc=True, verbose=verbose)
            self.A = self.A_storage.get_csr()
            if verbose:
                print(f"  Hybrid CSR/CSC storage enabled")
        else:
            self.A_storage = None
            self.A = A

        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub

        m, n = A.shape

        # Compute Ruiz scaling if requested
        self.ruiz_scaling: Optional[LazyRuizScaling] = None
        if use_ruiz:
            if verbose:
                print(f"  Computing Ruiz scaling (n_threads={n_threads})...")
            start = time.perf_counter()
            self.ruiz_scaling = LazyRuizScaling.compute(
                A, max_iter=10, n_threads=n_threads
            )
            elapsed = (time.perf_counter() - start) * 1000
            if verbose:
                print(f"    Done: {self.ruiz_scaling.iterations} iters, "
                      f"converged={self.ruiz_scaling.converged}, {elapsed:.2f}ms")

        # Pre-allocate workspace
        self.workspace = PreallocatedWorkspace.create(n_vars=n, n_constraints=m)

        # Cache for scaled vectors
        self._x_scaled_cache = np.zeros(n)
        self._y_scaled_cache = np.zeros(m)

    def sparse_matvec_scaled(
        self,
        x: np.ndarray,
        transpose: bool = False
    ) -> np.ndarray:
        """
        Sparse matrix-vector multiply with Ruiz scaling applied.

        Computes: y = row_scale * (A @ (col_scale * x))
        or: x = col_scale * (A.T @ (row_scale * y))

        This avoids creating the scaled matrix explicitly.

        Uses hybrid CSR/CSC storage for efficient transpose operations.
        """
        if self.ruiz_scaling is None:
            # No scaling - direct SpMV
            if self.A_storage is not None:
                # Use hybrid storage (avoids runtime CSR->CSC conversion)
                return self.A_storage.matvec(x, transpose=transpose)
            else:
                # Fallback to direct SpMV
                if transpose:
                    return self.A.T @ x
                else:
                    return self.A @ x

        # Apply scaling
        if transpose:
            # y_scaled = A.T @ (row_scale * x)
            # result = col_scale * y_scaled
            self._y_scaled_cache[:] = self.ruiz_scaling.row_scale * x
            if self.A_storage is not None:
                y_scaled = self.A_storage.matvec(self._y_scaled_cache, transpose=True)
            else:
                y_scaled = self.A.T @ self._y_scaled_cache
            return self.ruiz_scaling.col_scale * y_scaled
        else:
            # y_scaled = A @ (col_scale * x)
            # result = row_scale * y_scaled
            self._x_scaled_cache[:] = self.ruiz_scaling.col_scale * x
            if self.A_storage is not None:
                y_scaled = self.A_storage.matvec(self._x_scaled_cache, transpose=False)
            else:
                y_scaled = self.A @ self._x_scaled_cache
            return self.ruiz_scaling.row_scale * y_scaled

    def pdhg_iteration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tau: float,
        sigma: float,
        theta: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single PDHG iteration with Ruiz scaling.

        Primal update:
            x_new = proj_X(x - tau * (c + A_scaled.T @ y))

        Dual update:
            y_new = proj_Y(y + sigma * (A_scaled @ (2*x_new - x) - b))

        Returns:
            (x_new, y_new)
        """
        # === Primal Update ===
        # grad_x = c + A_scaled.T @ y
        grad_x = self.c + self.sparse_matvec_scaled(y, transpose=True)

        # x_new = proj_X(x - tau * grad_x)
        x_new = self._project_primal(x - tau * grad_x)

        # === Dual Update ===
        # Extrapolation: x_bar = 2*x_new - x
        x_bar = 2 * x_new - x

        # grad_y = A_scaled @ x_bar - b
        grad_y = self.sparse_matvec_scaled(x_bar) - self.b

        # y_new = proj_Y(y + sigma * grad_y)
        y_new = self._project_dual(y + sigma * grad_y)

        return x_new, y_new

    def _project_primal(self, x: np.ndarray) -> np.ndarray:
        """Project onto primal feasible set (box constraints)."""
        return np.clip(x, self.lb, self.ub)

    def _project_dual(self, y: np.ndarray) -> np.ndarray:
        """Project onto dual feasible set (y >= 0 for <= constraints)."""
        return np.maximum(y, 0)

    def solve(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        eta: Optional[float] = None,
        tau: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve LP using PDHG with all optimizations.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            eta: Primal step size (auto-computed if None)
            tau: Dual step size (auto-computed if None)

        Returns:
            (x_opt, y_opt, info_dict)
        """
        m, n = self.A.shape

        # Auto-compute step sizes
        if eta is None or tau is None:
            # Estimate operator norm
            if self.ruiz_scaling is not None:
                # Scaled operator norm is approximately 1
                norm_est = 1.0
            else:
                # Use sparse norm estimate (1-norm as approximation)
                norm_est = sp.linalg.norm(self.A, ord=2)

            if eta is None:
                eta = 1.0 / norm_est
            if tau is None:
                tau = 1.0 / norm_est

        # Initialize
        x = np.zeros(n)
        y = np.zeros(m)

        # Iteration history
        history = {
            'obj_primal': [],
            'obj_dual': [],
            'gap': [],
            'primal_res': [],
            'dual_res': []
        }

        start_time = time.perf_counter()

        for iteration in range(max_iter):
            # PDHG iteration
            x, y = self.pdhg_iteration(x, y, tau, eta)

            # Compute metrics every 100 iterations
            if (iteration + 1) % 100 == 0 or iteration == 0:
                obj_primal = self.c @ x
                obj_dual = -self.b @ y

                # Duality gap
                gap = abs(obj_primal - obj_dual) / (abs(obj_primal) + 1e-10)

                # Residuals
                Ax = self.sparse_matvec_scaled(x)
                primal_res = np.max(np.maximum(Ax - self.b, 0))

                ATy = self.sparse_matvec_scaled(y, transpose=True)
                dual_res = np.max(np.abs(self.c + ATy))

                history['obj_primal'].append(obj_primal)
                history['obj_dual'].append(obj_dual)
                history['gap'].append(gap)
                history['primal_res'].append(primal_res)
                history['dual_res'].append(dual_res)

                if self.verbose:
                    print(f"  Iter {iteration + 1}: obj_p={obj_primal:.4f}, "
                          f"gap={gap:.2e}, p_res={primal_res:.2e}")

                # Check convergence
                if gap < tol and primal_res < tol and dual_res < tol:
                    break

        elapsed = time.perf_counter() - start_time

        info = {
            'iterations': iteration + 1,
            'time': elapsed,
            'converged': gap < tol and primal_res < tol and dual_res < tol,
            'final_gap': gap,
            'final_primal_res': primal_res,
            'final_dual_res': dual_res,
            'history': history
        }

        return x, y, info


def benchmark_cpu_optimizations():
    """Benchmark different CPU optimization strategies."""
    import scipy.sparse as sp

    print("=" * 60)
    print("CPU Optimization Benchmark")
    print("=" * 60)

    # Create test problem
    np.random.seed(42)
    n = 1000
    m = 500
    density = 0.02

    A = sp.random(m, n, density=density, format='csr')
    b = np.random.randn(m)
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.ones(n)

    # Benchmark 1: No Ruiz scaling
    print("\n1. No Ruiz Scaling:")
    solver1 = OptimizedCPUSolver(A, b, c, lb, ub, use_ruiz=False, verbose=True)
    x1, y1, info1 = solver1.solve(max_iter=500, tol=1e-4)
    print(f"   Time: {info1['time']:.3f}s, Iterations: {info1['iterations']}")

    # Benchmark 2: With Ruiz scaling (sequential)
    print("\n2. With Ruiz Scaling (sequential):")
    solver2 = OptimizedCPUSolver(A, b, c, lb, ub, use_ruiz=True, n_threads=1, verbose=True)
    x2, y2, info2 = solver2.solve(max_iter=500, tol=1e-4)
    print(f"   Time: {info2['time']:.3f}s, Iterations: {info2['iterations']}")

    # Benchmark 3: With Ruiz scaling (parallel)
    print("\n3. With Ruiz Scaling (parallel, 4 threads):")
    solver3 = OptimizedCPUSolver(A, b, c, lb, ub, use_ruiz=True, n_threads=4, verbose=True)
    x3, y3, info3 = solver3.solve(max_iter=500, tol=1e-4)
    print(f"   Time: {info3['time']:.3f}s, Iterations: {info3['iterations']}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"  Speedup (Ruiz vs No Ruiz): {info1['time'] / info2['time']:.2f}x")
    print(f"  Speedup (Parallel vs Seq): {info2['time'] / info3['time']:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_cpu_optimizations()
