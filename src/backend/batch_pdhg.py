"""
Parallel Population PDHG Solver

Uses multiprocessing to parallelize population-based PDHG across all CPU cores.
Apple M4 Pro has 14 cores (10P + 4E), enabling massive parallelization.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time


@dataclass
class ParticleResult:
    """Result from a single particle."""
    x: np.ndarray
    y: np.ndarray
    obj: float
    feasible: bool
    iterations: int


def _run_single_pdhg(args):
    """Run PDHG for a single particle (for multiprocessing)."""
    A_data, A_indices, A_indptr, b, c, lb, ub, seed, max_iter, eta, tau = args

    np.random.seed(seed)
    n = len(c)
    m = len(b)

    # Reconstruct sparse matrix
    A = sp.csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    AT = A.T.tocsr()

    # Initialize with random perturbation
    x = np.random.uniform(lb, ub).astype(np.float64)
    y = np.zeros(m, dtype=np.float64)

    best_obj = float('inf')
    best_x = x.copy()

    for k in range(max_iter):
        # PDHG step
        Ax = A @ x
        ATy = AT @ y

        # Primal update
        grad_x = ATy + c
        x_new = np.clip(x - eta * grad_x, lb, ub)

        # Over-relaxed
        x_bar = 2 * x_new - x

        # Dual update
        Ax_bar = A @ x_bar
        grad_y = Ax_bar - b
        y_new = np.maximum(y + tau * grad_y, 0)

        x = x_new
        y = y_new

        # Track best
        if k % 50 == 0:
            obj = float(c @ x)
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()

    return ParticleResult(
        x=best_x,
        y=y,
        obj=best_obj,
        feasible=True,  # Will be checked later
        iterations=max_iter
    )


class ParallelPopulationPDHG:
    """
    Parallel population-based PDHG solver.

    Uses all available CPU cores to run multiple PDHG instances in parallel.
    """

    def __init__(self, A: sp.csr_matrix, b: np.ndarray, c: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray):
        self.A = A.tocsr()
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.m, self.n = A.shape

        # Pre-extract sparse matrix data for serialization
        self.A_data = self.A.data.astype(np.float64)
        self.A_indices = self.A.indices
        self.A_indptr = self.A.indptr

        # Detect available cores
        self.n_cores = mp.cpu_count()
        print(f"ParallelPopulationPDHG initialized")
        print(f"  Problem size: {self.m} x {self.n}")
        print(f"  Available cores: {self.n_cores}")

    def solve(self, population_size: int = 1000,
              max_iter: int = 500,
              eta: float = 0.1,
              tau: float = 0.1,
              verbose: bool = False) -> Tuple[np.ndarray, List[ParticleResult]]:
        """
        Run parallel population PDHG.

        Args:
            population_size: Number of particles
            max_iter: Iterations per particle
            eta: Primal step size
            tau: Dual step size
            verbose: Print progress

        Returns:
            Tuple of (best_x, all_results)
        """
        start_time = time.time()

        # Prepare arguments for parallel execution
        base_seed = np.random.randint(0, 2**31)
        args_list = [
            (self.A_data, self.A_indices, self.A_indptr,
             self.b, self.c, self.lb, self.ub,
             base_seed + i, max_iter, eta, tau)
            for i in range(population_size)
        ]

        # Run in parallel using ProcessPoolExecutor
        # Use fewer workers than cores to avoid memory pressure
        n_workers = min(self.n_cores - 1, population_size // 10)
        n_workers = max(1, n_workers)

        if verbose:
            print(f"  Running {population_size} particles with {n_workers} workers...")

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_run_single_pdhg, args_list))

        # Find best result
        best_idx = np.argmin([r.obj for r in results])
        best_result = results[best_idx]

        elapsed = time.time() - start_time
        if verbose:
            print(f"  Completed in {elapsed:.2f}s")
            print(f"  Best objective: {best_result.obj:.4e}")

        return best_result.x, results


class BatchPDHGNumba:
    """
    Batch PDHG using Numba JIT for vectorized operations.

    This is faster than multiprocessing for small to medium populations
    because it avoids process spawn overhead.
    """

    def __init__(self, A: sp.csr_matrix, b: np.ndarray, c: np.ndarray,
                 lb: np.ndarray, ub: np.ndarray,
                 constraint_sense: Optional[List[str]] = None):
        self.A = A.tocsr()
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.m, self.n = A.shape

        # Constraint sense: 'L' (<=), 'G' (>=), 'E' (equality)
        if constraint_sense is None:
            self.constraint_sense = ['L'] * self.m
        else:
            self.constraint_sense = constraint_sense

        # Convert >= constraints to <= by negating
        self._convert_constraints()

        # Convert to dense for batch operations (for small problems)
        self.A_dense = self.A.toarray()
        self.AT_dense = self.A_dense.T

        # Create masks for constraint types
        self.is_equality = np.array([s == 'E' for s in self.constraint_sense])
        self.is_inequality = ~self.is_equality

        print(f"BatchPDHGNumba initialized")
        print(f"  Problem size: {self.m} x {self.n}")

    def _convert_constraints(self):
        """Convert >= constraints to <= by negating."""
        for i, sense in enumerate(self.constraint_sense):
            if sense == 'G':
                self.A[i, :] = -self.A[i, :]
                self.b[i] = -self.b[i]
                self.constraint_sense[i] = 'L'

    def solve(self, population_size: int = 1000,
              max_iter: int = 500,
              eta: float = 0.1,
              tau: float = 0.1,
              verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run batch PDHG using vectorized operations.

        Handles:
        - <= constraints: dual y >= 0
        - >= constraints: converted to <= in constructor
        - = constraints: dual y unrestricted
        """
        start_time = time.time()

        K = population_size
        n = self.n
        m = self.m

        # Initialize population
        x = np.random.uniform(self.lb, self.ub, size=(K, n))
        y = np.zeros((K, m))

        best_obj = float('inf')
        best_x = None

        for k in range(max_iter):
            # Batch matrix multiply: (K, n) @ (n, m).T = (K, m)
            Ax = x @ self.AT_dense  # (K, m)

            # Batch primal gradient
            ATy = y @ self.A_dense  # (K, n)
            grad_x = ATy + self.c  # Broadcasting

            # Primal update
            x_new = np.clip(x - eta * grad_x, self.lb, self.ub)

            # Over-relaxed
            x_bar = 2 * x_new - x

            # Dual gradient
            Ax_bar = x_bar @ self.AT_dense
            grad_y = Ax_bar - self.b

            # Dual update with proper constraint handling
            y_new = y + tau * grad_y

            # Project inequality duals to >= 0, leave equality duals unrestricted
            y_new[:, self.is_inequality] = np.maximum(y_new[:, self.is_inequality], 0)

            x = x_new
            y = y_new

            # Track best
            if k % 50 == 0:
                objs = x @ self.c  # (K,)
                min_idx = np.argmin(objs)
                if objs[min_idx] < best_obj:
                    best_obj = objs[min_idx]
                    best_x = x[min_idx].copy()

                if verbose and k % 100 == 0:
                    print(f"  Iter {k}: best_obj={best_obj:.4e}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"  Completed in {elapsed:.2f}s")
            print(f"  Best objective: {best_obj:.4e}")

        return best_x, best_obj
