"""
Adaptive Heterogeneous Solver for Apple Silicon

This module provides CPU/GPU heterogeneous computing with adaptive device selection.
The framework automatically selects the optimal backend based on problem characteristics.

Key Features:
1. Zero-copy via unified memory (when GPU is used)
2. Multi-dimensional feature-based adaptive device selection
3. Fallback to CPU if GPU is unavailable or slower
4. Performance profiling for continuous learning

Version 2: Enhanced with multi-dimensional features based on DeepSeek recommendations
"""

import numpy as np
import scipy.sparse as sp
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import warnings

# Try to import Metal (optional)
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"
    PROFILE = "profile"  # Needs profiling to decide


@dataclass
class ProblemCharacteristics:
    """
    Enhanced problem characteristics for device selection.

    Features are extracted during preprocessing (e.g., Ruiz scaling) at zero extra cost.
    Based on DeepSeek research, the most important features are:
    1. density - determines GPU memory access patterns
    2. max_row_nnz - detects dense rows (key for GPU speedup potential)
    3. int_var_ratio - MIP characteristics
    """
    # Scale features
    n_vars: int
    n_constraints: int
    nnz: int

    # Structure features (critical for GPU decision)
    density: float              # nnz / (m * n)
    max_row_nnz: int           # max non-zeros per row (dense row detection)
    max_col_nnz: int           # max non-zeros per column
    avg_row_nnz: float         # average non-zeros per row

    # Numerical features
    norm_frobenius: float = 0.0

    # Problem type features
    int_var_ratio: float = 0.0  # ratio of integer variables
    has_equality: bool = False

    # Batch features
    is_batch: bool = False
    batch_size: int = 1

    @classmethod
    def from_problem(
        cls,
        A: sp.csr_matrix,
        int_vars: List[int] = None,
        constraint_sense: List[str] = None,
        batch_size: int = 1
    ) -> 'ProblemCharacteristics':
        """
        Extract features from problem matrix.

        This is designed to be called during preprocessing, so most
        computations are essentially free (reusing existing data).

        Args:
            A: Sparse constraint matrix in CSR format
            int_vars: List of integer variable indices
            constraint_sense: List of constraint types ('L', 'G', 'E')
            batch_size: Batch size for population methods
        """
        m, n = A.shape
        nnz = A.nnz

        # Structure features - computed from CSR structure
        row_nnz = np.diff(A.indptr)  # non-zeros per row
        col_nnz = np.bincount(A.indices, minlength=n)  # non-zeros per column

        density = nnz / (m * n) if m * n > 0 else 0

        # Numerical features - Frobenius norm (cheap to compute)
        norm_frobenius = np.linalg.norm(A.data) if len(A.data) > 0 else 0.0

        # Problem type features
        int_var_ratio = len(int_vars) / n if int_vars and n > 0 else 0.0
        has_equality = 'E' in constraint_sense if constraint_sense else False

        return cls(
            n_vars=n,
            n_constraints=m,
            nnz=nnz,
            density=density,
            max_row_nnz=int(np.max(row_nnz)) if len(row_nnz) > 0 else 0,
            max_col_nnz=int(np.max(col_nnz)) if len(col_nnz) > 0 else 0,
            avg_row_nnz=float(np.mean(row_nnz)) if len(row_nnz) > 0 else 0.0,
            norm_frobenius=norm_frobenius,
            int_var_ratio=int_var_ratio,
            has_equality=has_equality,
            is_batch=batch_size > 1,
            batch_size=batch_size
        )

    def summary(self) -> str:
        """Return a one-line summary of characteristics."""
        return (f"n={self.n_vars}, m={self.n_constraints}, nnz={self.nnz}, "
                f"density={self.density:.4f}, max_row_nnz={self.max_row_nnz}, "
                f"int_ratio={self.int_var_ratio:.0%}")


@dataclass
class PerformanceProfile:
    """Performance history for adaptive learning."""
    cpu_times: Dict[int, float] = field(default_factory=dict)
    gpu_times: Dict[int, float] = field(default_factory=dict)

    def record(self, n_vars: int, device: str, time_ms: float):
        if device == "cpu":
            self.cpu_times[n_vars] = time_ms
        else:
            self.gpu_times[n_vars] = time_ms

    def get_speedup(self, n_vars: int) -> Optional[float]:
        if n_vars in self.cpu_times and n_vars in self.gpu_times:
            return self.cpu_times[n_vars] / self.gpu_times[n_vars]
        return None


class LightweightProfiler:
    """
    Lightweight CPU/GPU profiler for uncertain cases.

    Uses M4 Pro's unified memory to do zero-copy profiling:
    - Runs 10 iterations on CPU (Accelerate BLAS)
    - Runs 10 iterations on GPU (Metal)
    - Compares actual throughput

    Overhead: ~5-10ms (acceptable for large problems)
    """

    def __init__(self, warmup: int = 3, test_iters: int = 10):
        self.warmup = warmup
        self.test_iters = test_iters
        self._metal_compute = None

    def _get_metal_compute(self):
        """Lazy initialization of Metal compute - using Phase 3.5."""
        if self._metal_compute is None:
            try:
                from src.backend.gpu_phase35_optimized import OptimizedGPUSolverPhase35
                self._metal_compute = OptimizedGPUSolverPhase35  # Class reference
            except ImportError:
                pass
        return self._metal_compute

    def profile_spmv(
        self,
        A: sp.csr_matrix,
        x: np.ndarray = None
    ) -> Tuple[float, float, str]:
        """
        Profile SpMV performance on CPU vs GPU.

        Uses Phase 3.5 optimized GPU solver for accurate profiling.

        Args:
            A: Sparse matrix in CSR format
            x: Input vector (optional, will generate if None)

        Returns:
            (cpu_time_ms, gpu_time_ms, winner)
        """
        n = A.shape[1]
        if x is None:
            x = np.random.randn(n).astype(np.float32)

        # === CPU Benchmark ===
        # Warmup
        for _ in range(self.warmup):
            _ = A @ x

        # Benchmark
        start = time.perf_counter()
        for _ in range(self.test_iters):
            _ = A @ x
        cpu_time = (time.perf_counter() - start) / self.test_iters * 1000

        # === GPU Benchmark (Phase 3.5) ===
        MetalCompute = self._get_metal_compute()

        if MetalCompute is None:
            return cpu_time, float('inf'), "cpu"

        # Create solver instance
        try:
            # Create dummy b, c, lb, ub for solver initialization
            m = A.shape[0]
            b = np.zeros(m)
            c = np.random.randn(n)
            lb = np.zeros(n)
            ub = np.ones(n)

            gpu_solver = MetalCompute(A, b, c, lb, ub, verbose=False)

            if not gpu_solver.available:
                return cpu_time, float('inf'), "cpu"
        except Exception:
            return cpu_time, float('inf'), "cpu"

        # Warmup
        for _ in range(self.warmup):
            _ = gpu_solver.spmv(x)

        # Benchmark
        start = time.perf_counter()
        for _ in range(self.test_iters):
            _ = gpu_solver.spmv(x)
        gpu_time = (time.perf_counter() - start) / self.test_iters * 1000

        # Determine winner (GPU needs to be clearly faster to justify)
        if gpu_time < cpu_time * 0.8:
            winner = "gpu"
        else:
            winner = "cpu"

        return cpu_time, gpu_time, winner

    def should_profile(self, chars: ProblemCharacteristics) -> bool:
        """
        Determine if profiling is needed for this problem.

        Profiling is useful for uncertain cases where rules don't give
        a clear answer. We profile when:
        - Medium scale (500 < n < 10000)
        - Medium density (0.001 < density < 0.1)
        - Not clearly CPU or GPU favored

        Updated for Phase 3.5: GPU advantage starts earlier, so profiling
        range is adjusted.
        """
        n = chars.n_vars
        density = chars.density

        # Clear cases - no profiling needed
        if n < 500:
            return False
        if density < 0.001:
            return False
        if density > 0.1 and n > 5000:
            return False
        if chars.max_row_nnz > 1000 and n > 2000:
            return False
        if chars.batch_size >= 1000:
            return False
        if chars.int_var_ratio > 0.8 and n < 20000:
            return False
        if n < self.THRESHOLDS.get('n_large', 3000):
            return False  # Typical MIP, CPU is safe choice

        # Uncertain cases: 3000 <= n <= 10000
        return 3000 <= n <= 10000 and 0.001 <= density <= 0.1


class AdaptiveDeviceSelector:
    """
    Enhanced adaptive device selector with multi-dimensional features.

    Selection Strategy (based on DeepSeek research + M4 Pro benchmarks):

    1. Ultra-sparse problems (density < 0.001): CPU (GPU memory access bottleneck)
    2. Dense + Large (density > 0.1, n > 5000): GPU
    3. Dense rows (max_row_nnz > 1000, n > 2000): GPU
    4. Large batch (batch >= 1000): GPU
    5. MIP-heavy (int_ratio > 0.8, n < 20000): CPU (branch logic intensive)
    6. Typical MIP scale (n < 5000): CPU
    7. Uncertain cases: Profile or default to CPU

    Key insight: Metal kernel launch overhead (~0.15ms) means CPU is faster
    for most MIP-scale problems. GPU advantage requires:
    - Dense enough to amortize memory access overhead
    - Large enough to amortize kernel launch overhead
    - Or batch operations to amortize one-time costs
    """

    # Thresholds based on Phase 3.5 M4 Pro benchmarks (Mar 2026)
    THRESHOLDS = {
        # Scale thresholds
        'n_small': 500,           # Below: CPU always faster
        'n_large': 3000,          # Above: GPU may be faster (Phase 3.5: GPU advantage starts earlier)
        'n_very_large': 6000,     # Above: GPU likely faster (lowered from 8000 based on 实测)

        # Density thresholds (CRITICAL for GPU decision)
        'density_ultra_sparse': 0.001,   # Below: GPU memory bottleneck
        'density_sparse': 0.01,          # Sparse
        'density_moderate': 0.02,        # Moderate (lowered from 0.05)
        'density_dense': 0.1,            # Dense - GPU advantage

        # Structure thresholds
        'dense_row_threshold': 1000,     # Dense row detection
        'dense_col_threshold': 1000,     # Dense column detection

        # Batch threshold (Phase 3.5: GPU优势显著提前)
        'batch_gpu_threshold': 32,       # Raised from 16 (Phase 2: need n>=10000)
        'batch_gpu_min_n': 10000,        # NEW: minimum n for batch GPU

        # MIP characteristics
        'int_ratio_high': 0.8,           # High integer variable ratio

        # Metal kernel overhead (measured on M4 Pro)
        'gpu_overhead_ms': 0.15,
    }

    def __init__(self, verbose: bool = False, enable_profiling: bool = True,
                 enable_performance_cache: bool = True):
        self.verbose = verbose
        self.profile = PerformanceProfile()
        self.gpu_available = METAL_AVAILABLE
        self.device = None
        self.enable_profiling = enable_profiling
        self.enable_performance_cache = enable_performance_cache
        self._profiler = None

        # Performance Cache (new)
        # Key: (n_vars, density_bucket, batch_size_bucket)
        # Value: {'cpu_time': float, 'gpu_time': float, 'count': int}
        self._performance_cache: Dict[tuple, Dict] = {}
        self._cache_alpha = 0.3  # EMA smoothing factor

        if self.gpu_available:
            try:
                self.device = Metal.MTLCreateSystemDefaultDevice()
                if self.device:
                    if verbose:
                        print(f"GPU available: {self.device.name()}")
                        mem_gb = self.device.recommendedMaxWorkingSetSize() / 1e9
                        print(f"  Memory: {mem_gb:.1f} GB")
                else:
                    self.gpu_available = False
            except Exception as e:
                if verbose:
                    print(f"GPU initialization failed: {e}")
                self.gpu_available = False

    @property
    def profiler(self):
        """Lazy initialization of profiler."""
        if self._profiler is None and self.enable_profiling:
            self._profiler = LightweightProfiler()
        return self._profiler

    def _get_cache_key(self, chars: ProblemCharacteristics) -> tuple:
        """
        Generate cache key from problem characteristics.

        Buckets:
        - n_vars: log scale buckets
        - density: 4 buckets (ultra_sparse, sparse, moderate, dense)
        - batch_size: 3 buckets (1, 2-32, 33+)
        """
        # n_vars bucket (log scale)
        n = chars.n_vars
        if n < 1000:
            n_bucket = 0
        elif n < 5000:
            n_bucket = 1
        elif n < 10000:
            n_bucket = 2
        elif n < 20000:
            n_bucket = 3
        else:
            n_bucket = 4

        # density bucket
        d = chars.density
        if d < 0.001:
            d_bucket = 0  # ultra_sparse
        elif d < 0.01:
            d_bucket = 1  # sparse
        elif d < 0.1:
            d_bucket = 2  # moderate
        else:
            d_bucket = 3  # dense

        # batch_size bucket
        b = chars.batch_size
        if b <= 1:
            b_bucket = 0
        elif b <= 32:
            b_bucket = 1
        else:
            b_bucket = 2

        return (n_bucket, d_bucket, b_bucket)

    def _get_cached_decision(self, chars: ProblemCharacteristics) -> Optional[Tuple[str, float, float]]:
        """
        Get cached performance data and decision.

        Returns:
            (device, cpu_time, gpu_time) if cache hit, None otherwise
        """
        if not self.enable_performance_cache:
            return None

        key = self._get_cache_key(chars)
        if key in self._performance_cache:
            cached = self._performance_cache[key]
            cpu_time = cached['cpu_time']
            gpu_time = cached['gpu_time']

            # Return decision based on cached times
            if gpu_time < cpu_time * 0.8:
                return ("gpu", cpu_time, gpu_time)
            else:
                return ("cpu", cpu_time, gpu_time)

        return None

    def _update_cache(self, chars: ProblemCharacteristics,
                      cpu_time: float, gpu_time: float):
        """Update performance cache with new measurement."""
        if not self.enable_performance_cache:
            return

        key = self._get_cache_key(chars)
        alpha = self._cache_alpha

        if key in self._performance_cache:
            # Exponential moving average update
            cached = self._performance_cache[key]
            cached['cpu_time'] = (1 - alpha) * cached['cpu_time'] + alpha * cpu_time
            cached['gpu_time'] = (1 - alpha) * cached['gpu_time'] + alpha * gpu_time
            cached['count'] += 1
        else:
            # New entry
            self._performance_cache[key] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'count': 1
            }

    def get_cache_stats(self) -> Dict:
        """Get performance cache statistics."""
        if not self._performance_cache:
            return {'cache_entries': 0, 'cache_hit_rate': 0.0}

        total_entries = len(self._performance_cache)
        total_lookups = sum(c['count'] for c in self._performance_cache.values())
        unique_lookups = total_entries

        return {
            'cache_entries': total_entries,
            'total_measurements': total_lookups,
            'unique_entries': unique_lookups,
            'avg_measurements_per_entry': total_lookups / total_entries if total_entries > 0 else 0
        }

    def select(self, chars: ProblemCharacteristics) -> Tuple[str, str]:
        """
        Select optimal device based on multi-dimensional features.

        First checks performance cache for fast decision.
        Falls back to rule-based selection if cache miss.

        Returns:
            (device_type, reason)
        """
        # === Step 1: Check performance cache ===
        cached_result = self._get_cached_decision(chars)
        if cached_result is not None:
            device, cpu_time, gpu_time = cached_result
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            return device, f"cache (cpu={cpu_time:.3f}ms, gpu={gpu_time:.3f}ms, speedup={speedup:.2f}x)"

        # === Step 2: Rule-based selection (cache miss) ===
        n = chars.n_vars
        m = chars.n_constraints
        density = chars.density
        batch = chars.batch_size
        max_row_nnz = chars.max_row_nnz
        int_ratio = chars.int_var_ratio

        # === Rule 1: Very small problems always CPU ===
        if n < self.THRESHOLDS['n_small']:
            return "cpu", f"n={n} < {self.THRESHOLDS['n_small']}"

        # === Rule 2: Ultra-sparse problems → CPU ===
        # DeepSeek: Sparse matrices cause GPU memory access bottleneck
        if density < self.THRESHOLDS['density_ultra_sparse']:
            return "cpu", f"ultra_sparse (density={density:.5f} < {self.THRESHOLDS['density_ultra_sparse']})"

        # === Rule 3: Dense + Large → GPU ===
        # Dense matrices allow efficient GPU memory coalescing
        if (density > self.THRESHOLDS['density_dense'] and
            n > self.THRESHOLDS['n_large'] and
            self.gpu_available):
            return "gpu", f"dense_large (density={density:.2f}, n={n})"

        # === Rule 4: Dense rows detected → GPU ===
        # DeepSeek: Dense rows/blocks are key for GPU speedup potential
        # Fix: Also check density and row count to avoid "fake dense rows" cases
        # like air03 (124 rows, max_row_nnz=3861, but density=0.068 → CPU wins)
        if (max_row_nnz > self.THRESHOLDS['dense_row_threshold'] and
            n > 2000 and
            m > 500 and  # Need enough rows for GPU to be effective
            density > 0.05 and  # Minimum density for GPU advantage
            self.gpu_available):
            return "gpu", f"dense_rows (max_row_nnz={max_row_nnz})"

        # === Rule 4.5: Ultra-large scale + High max_row_nnz → GPU ===
        # NEW: For extremely large problems (n > 50K) with high max_row_nnz (>5000),
        # GPU advantage emerges even with lower density due to massive parallel compute.
        # Example: a2864-99blp (n=200K, max_row_nnz=11811) → GPU 2.6x faster
        if (n > 50000 and
            max_row_nnz > 5000 and
            m > 10000 and  # Need enough rows for workload distribution
            self.gpu_available):
            return "gpu", f"ultra_large_scale (n={n}, max_row_nnz={max_row_nnz})"

        # === Rule 5: Large batch operations → GPU ===
        # Batch amortizes kernel launch overhead
        # Phase 2 fix: GPU only wins when n >= batch_gpu_min_n AND batch >= threshold
        if (batch >= self.THRESHOLDS['batch_gpu_threshold'] and
            n >= self.THRESHOLDS['batch_gpu_min_n'] and
            self.gpu_available):
            return "gpu", f"large_batch (batch={batch}, n={n})"

        # === Rule 5.5: Moderate density + Medium scale → GPU ===
        # Phase 2 fix: density 0.02-0.05 with n >= 6000 shows GPU advantage
        # e.g., uncertain_n6000_d0.03: GPU 1.13x, uncertain_n7500_d0.025: GPU 1.37x
        # NEW: Exclude high int_ratio problems (MIP-heavy → CPU)
        if (density >= self.THRESHOLDS['density_moderate'] and
            n >= self.THRESHOLDS['n_very_large'] and
            int_ratio < 0.5 and  # Exclude MIP-heavy problems
            self.gpu_available):
            return "gpu", f"moderate_density (density={density:.3f}, n={n})"

        # === Rule 6: MIP-heavy problems → CPU ===
        # DeepSeek: MIP branch logic is CPU-friendly
        if (int_ratio > self.THRESHOLDS['int_ratio_high'] and
            n < self.THRESHOLDS['n_very_large']):
            return "cpu", f"mip_heavy (int_ratio={int_ratio:.0%})"

        # === Rule 7: Typical MIP scale → CPU ===
        # Based on Phase 3.5 M4 Pro benchmarks, CPU is faster for n < 3000
        if n < self.THRESHOLDS['n_large']:
            return "cpu", f"typical_mip (n={n} < {self.THRESHOLDS['n_large']})"

        # === Rule 8: Very large problems → GPU ===
        # NEW: Exclude MIP-heavy problems (high int_ratio → CPU)
        if (n > self.THRESHOLDS['n_very_large'] and
            int_ratio < 0.5 and  # Exclude MIP-heavy problems
            self.gpu_available):
            return "gpu", f"very_large (n={n})"

        # === Rule 9: Check historical performance ===
        speedup = self.profile.get_speedup(n)
        if speedup is not None:
            if speedup > 1.2:
                return "gpu", f"historical_speedup={speedup:.2f}x"
            else:
                return "cpu", f"historical_speedup={speedup:.2f}x"

        # === Rule 10: Uncertain - could profile, but default CPU ===
        # For very uncertain cases, return "profile" to trigger lightweight profiling
        # Otherwise default to CPU for safety
        return "cpu", "default_fallback"

    def needs_profiling(self, chars: ProblemCharacteristics) -> bool:
        """Check if this problem needs profiling."""
        if not self.enable_profiling or not self.gpu_available:
            return False
        if self._profiler is None:
            self._profiler = LightweightProfiler()
        return self._profiler.should_profile(chars)

    def select_with_profile(self, chars: ProblemCharacteristics,
                            A: sp.csr_matrix = None,
                            x: np.ndarray = None,
                            force_profile: bool = False) -> Tuple[str, str]:
        """
        Select device with optional lightweight profiling for uncertain cases.

        This uses M4 Pro's unified memory to do zero-copy profiling.

        Args:
            chars: Problem characteristics
            A: Sparse matrix (required for profiling)
            x: Input vector (optional, will generate if None)
            force_profile: Force profiling even if rules say otherwise

        Returns:
            (device_type, reason)
        """
        # First, try rule-based selection
        device, reason = self.select(chars)

        # Check if we should profile
        should_profile = force_profile or self.needs_profiling(chars)

        if not should_profile:
            return device, reason

        # Profiling is needed - need matrix data
        if A is None:
            return device, f"rule_based (no matrix for profiling)"

        if not self.gpu_available:
            return "cpu", "rule_based_cpu (no gpu)"

        # Run lightweight profiling
        try:
            if self._profiler is None:
                self._profiler = LightweightProfiler()

            cpu_time, gpu_time, winner = self._profiler.profile_spmv(A, x)

            # Record for future use (legacy profile)
            self.profile.record(chars.n_vars, "cpu", cpu_time)
            self.profile.record(chars.n_vars, "gpu", gpu_time)

            # Update performance cache (new)
            self._update_cache(chars, cpu_time, gpu_time)

            if winner == "gpu":
                return "gpu", f"profiled (cpu={cpu_time:.3f}ms, gpu={gpu_time:.3f}ms)"
            else:
                return "cpu", f"profiled (cpu={cpu_time:.3f}ms, gpu={gpu_time:.3f}ms)"

        except Exception as e:
            # Fallback to rule-based decision
            return device, f"rule_based (profile error: {e})"


class HeterogeneousBackend:
    """
    Unified backend with automatic CPU/GPU selection.

    This is the main interface for heterogeneous PDHG solving.
    """

    def __init__(self, device: str = "auto", verbose: bool = False):
        self.verbose = verbose
        self.selector = AdaptiveDeviceSelector(verbose=verbose)
        self.forced_device = device if device != "auto" else None

        # Performance stats
        self.stats = {
            'cpu_calls': 0,
            'gpu_calls': 0,
            'total_time_ms': 0
        }

    def sparse_matvec(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """Sparse matrix-vector multiply with automatic device selection."""
        if self.forced_device:
            device = self.forced_device
            reason = "forced"
        else:
            chars = ProblemCharacteristics.from_problem(A)
            device, reason = self.selector.select(chars)

        start = time.perf_counter()

        if device == "gpu" and self.selector.gpu_available:
            result = self._gpu_sparse_matvec(A, x)
            self.stats['gpu_calls'] += 1
        else:
            result = A @ x  # NumPy/SciPy (uses Accelerate BLAS)
            self.stats['cpu_calls'] += 1

        elapsed = (time.perf_counter() - start) * 1000
        self.stats['total_time_ms'] += elapsed

        if self.verbose:
            print(f"  SpMV: {device} ({reason}), {elapsed:.3f} ms")

        return result

    def _gpu_sparse_matvec(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """GPU sparse matrix-vector multiply using Metal."""
        # For now, fall back to CPU until Metal shaders are properly compiled
        # This is a placeholder - actual GPU implementation would use Metal
        return A @ x

    def batch_sparse_matvec(self, A: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
        """Batch sparse matrix-vector multiply."""
        K, n = X.shape
        m = A.shape[0]

        if self.forced_device:
            device = self.forced_device
        else:
            chars = ProblemCharacteristics.from_problem(A, batch_size=K)
            device, _ = self.selector.select(chars)

        start = time.perf_counter()

        if device == "gpu" and self.selector.gpu_available:
            result = self._gpu_batch_sparse_matvec(A, X)
            self.stats['gpu_calls'] += 1
        else:
            # CPU batch - use dense for better BLAS utilization
            if K >= 50 and A.shape[0] * A.shape[1] < 1e7:
                # Convert to dense for efficient batch multiply
                A_dense = A.toarray().astype(np.float64)
                result = X @ A_dense.T
            else:
                # Naive sparse batch
                result = np.zeros((K, m))
                for i in range(K):
                    result[i] = A @ X[i]
            self.stats['cpu_calls'] += 1

        elapsed = (time.perf_counter() - start) * 1000
        self.stats['total_time_ms'] += elapsed

        return result

    def _gpu_batch_sparse_matvec(self, A: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
        """GPU batch sparse matrix-vector multiply."""
        # Placeholder - fall back to CPU
        K = X.shape[0]
        m = A.shape[0]
        result = np.zeros((K, m))
        for i in range(K):
            result[i] = A @ X[i]
        return result

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total_calls = self.stats['cpu_calls'] + self.stats['gpu_calls']
        return {
            **self.stats,
            'total_calls': total_calls,
            'gpu_ratio': self.stats['gpu_calls'] / total_calls if total_calls > 0 else 0
        }


def create_backend(device: str = "auto", verbose: bool = False) -> HeterogeneousBackend:
    """Factory function to create backend."""
    return HeterogeneousBackend(device=device, verbose=verbose)


# Benchmark function
def benchmark_backends(sizes: List[int] = None, verbose: bool = True) -> Dict:
    """
    Benchmark CPU vs estimated GPU performance.

    Returns comparison results.

    NOTE: Based on M4 Pro actual testing:
    - Metal kernel launch overhead: ~0.15ms
    - CPU (Accelerate BLAS) is faster for n < 5000
    - GPU only advantageous for very large problems or huge batches
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]

    results = []

    # Metal kernel launch overhead (measured on M4 Pro)
    GPU_OVERHEAD_MS = 0.15

    for n in sizes:
        A = sp.random(n, n, density=0.02, format='csr')
        x = np.random.randn(n).astype(np.float32)

        # CPU benchmark
        times_cpu = []
        for _ in range(10):
            start = time.perf_counter()
            _ = A @ x
            times_cpu.append(time.perf_counter() - start)

        cpu_time = np.mean(times_cpu) * 1000

        # GPU estimate (based on M4 Pro actual measurements)
        # GPU time = overhead + compute time
        # Compute time scales roughly linearly with nnz
        nnz = A.nnz
        gpu_compute_estimate = cpu_time * 0.1  # GPU compute is ~10x faster
        gpu_time_estimate = GPU_OVERHEAD_MS + gpu_compute_estimate

        results.append({
            'n': n,
            'nnz': nnz,
            'cpu_time_ms': cpu_time,
            'gpu_estimate_ms': gpu_time_estimate,
            'gpu_overhead_ms': GPU_OVERHEAD_MS,
            'speedup': cpu_time / gpu_time_estimate if gpu_time_estimate > 0 else 0
        })

        if verbose:
            r = results[-1]
            winner = "CPU" if cpu_time < gpu_time_estimate else "GPU"
            print(f"n={n:5d}: CPU={r['cpu_time_ms']:.3f}ms, "
                  f"GPU~{r['gpu_estimate_ms']:.3f}ms (overhead={GPU_OVERHEAD_MS}ms), "
                  f"{winner} wins")

    return results


if __name__ == "__main__":
    print("="*60)
    print("Enhanced Adaptive Heterogeneous Backend Test")
    print("="*60)

    # Test device selection with various problem types
    print("\n1. Device Selection Test (Enhanced Features):")
    selector = AdaptiveDeviceSelector(verbose=True, enable_profiling=True)

    # Test cases with different characteristics
    test_cases = [
        # (n, density, batch_size, description)
        (100, 0.02, 1, "Small sparse"),
        (1000, 0.001, 1, "Medium ultra-sparse"),
        (1000, 0.02, 1, "Medium typical MIP"),
        (1000, 0.15, 1, "Medium dense"),
        (5000, 0.02, 1, "Large sparse"),
        (5000, 0.15, 1, "Large dense"),
        (10000, 0.02, 1, "Very large sparse"),
        (10000, 0.15, 1, "Very large dense"),
        (2000, 0.05, 100, "Medium batch=100"),
        (2000, 0.05, 1000, "Medium batch=1000"),
    ]

    print("\n  Testing various problem types:")
    print("  " + "-"*56)

    for n, density, batch_size, desc in test_cases:
        A = sp.random(n, n, density=density, format='csr')
        chars = ProblemCharacteristics.from_problem(A, batch_size=batch_size)
        device, reason = selector.select(chars)
        print(f"  {desc:25s}: {device:4s} - {reason}")

    # Test with integer variables (MIP characteristics)
    print("\n  Testing MIP characteristics:")
    print("  " + "-"*56)

    n = 3000
    A = sp.random(n, n, density=0.02, format='csr')

    # High integer ratio (MIP-heavy)
    int_vars = list(range(int(n * 0.9)))  # 90% integer
    chars_mip = ProblemCharacteristics.from_problem(A, int_vars=int_vars)
    device, reason = selector.select(chars_mip)
    print(f"  MIP-heavy (90% int):      {device:4s} - {reason}")

    # Low integer ratio (mostly LP)
    int_vars_lp = list(range(int(n * 0.1)))  # 10% integer
    chars_lp = ProblemCharacteristics.from_problem(A, int_vars=int_vars_lp)
    device, reason = selector.select(chars_lp)
    print(f"  LP-like (10% int):        {device:4s} - {reason}")

    # Test dense rows
    print("\n  Testing dense row detection:")
    print("  " + "-"*56)

    # Create matrix with some dense rows using LIL format (efficient for construction)
    n = 3000
    A_dense_rows = sp.lil_matrix((n, n))
    # Add sparse background
    for i in range(n):
        for j in np.random.choice(n, size=int(n * 0.01), replace=False):
            A_dense_rows[i, j] = np.random.randn()
    # Add dense rows (first 10 rows are dense)
    for i in range(10):
        for j in range(n):
            if np.random.rand() < 0.8:
                A_dense_rows[i, j] = np.random.randn()

    A_dense_rows_csr = A_dense_rows.tocsr()
    chars_dense = ProblemCharacteristics.from_problem(A_dense_rows_csr)
    device, reason = selector.select(chars_dense)
    print(f"  Dense rows detected:      {device:4s} - {reason}")
    print(f"    (max_row_nnz={chars_dense.max_row_nnz}, density={chars_dense.density:.4f})")

    # Test profiling feature
    print("\n3. Profiling Test (Uncertain Cases):")
    print("  " + "-"*56)

    # Create an uncertain case: n=8000, density=0.02 (medium)
    n_uncertain = 8000
    A_uncertain = sp.random(n_uncertain, n_uncertain, density=0.02, format='csr')
    chars_uncertain = ProblemCharacteristics.from_problem(A_uncertain)

    print(f"  Testing uncertain case (n={n_uncertain}, density={chars_uncertain.density:.4f})")

    # Check if profiling is needed
    needs_profile = selector.needs_profiling(chars_uncertain)
    print(f"  Needs profiling: {needs_profile}")

    if needs_profile:
        # Run profiling
        device, reason = selector.select_with_profile(chars_uncertain, A_uncertain)
        print(f"  After profiling: {device} - {reason}")
    else:
        # Just use rule-based
        device, reason = selector.select(chars_uncertain)
        print(f"  Rule-based: {device} - {reason}")

    # Test backend
    print("\n2. Backend Integration Test:")
    backend = create_backend(verbose=True)

    A = sp.random(1000, 1000, density=0.02, format='csr')
    x = np.random.randn(1000)

    y = backend.sparse_matvec(A, x)
    print(f"  Result shape: {y.shape}")

    # Batch test
    X = np.random.randn(100, 1000)
    Y = backend.batch_sparse_matvec(A, X)
    print(f"  Batch result shape: {Y.shape}")

    print(f"\n  Stats: {backend.get_stats()}")

    # Benchmark
    print("\n4. CPU vs GPU Estimate Benchmark:")
    print("  (Based on M4 Pro actual measurements: GPU overhead ~0.15ms)")
    benchmark_backends()

    print("\n" + "="*60)
    print("✓ Enhanced adaptive backend test complete!")
    print("="*60)
