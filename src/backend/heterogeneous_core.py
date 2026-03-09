"""
Heterogeneous Computing Framework for Apple Silicon

This module provides a unified interface for CPU/GPU heterogeneous computing,
with automatic device selection based on problem characteristics.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    HeterogeneousSolver                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Profiler    │  │ DeviceSelector│  │ AdaptiveScheduler│   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ComputeBackend (抽象层)                 │    │
│  │  ┌──────────────┐          ┌────────────────────┐   │    │
│  │  │ CPUBackend   │          │ MetalGPUBackend    │   │    │
│  │  │ (NumPy+BLAS) │          │ (Metal Shaders)    │   │    │
│  │  └──────────────┘          └────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                    统一内存                                  │
│              (Unified Memory - Zero Copy)                   │
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
import scipy.sparse as sp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import os

# Metal imports
try:
    import Metal
    import MetalPerformanceShaders as MPS
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("Warning: Metal framework not available, GPU acceleration disabled")


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass
class ProblemCharacteristics:
    """Characteristics of an optimization problem for device selection."""
    n_vars: int
    n_constraints: int
    nnz: int  # Non-zeros in constraint matrix
    n_int_vars: int
    density: float
    estimated_memory_mb: float

    @classmethod
    def from_problem(cls, A: sp.csr_matrix, int_vars: List[int] = None) -> 'ProblemCharacteristics':
        m, n = A.shape
        nnz = A.nnz
        density = nnz / (m * n) if m * n > 0 else 0
        estimated_memory = (nnz * 8 + (m + n) * 8 * 3) / 1024 / 1024  # MB

        return cls(
            n_vars=n,
            n_constraints=m,
            nnz=nnz,
            n_int_vars=len(int_vars) if int_vars else 0,
            density=density,
            estimated_memory_mb=estimated_memory
        )


@dataclass
class PerformanceProfile:
    """Historical performance data for adaptive scheduling."""
    cpu_times: List[float] = field(default_factory=list)
    gpu_times: List[float] = field(default_factory=list)
    problem_sizes: List[int] = field(default_factory=list)

    def add_result(self, n_vars: int, cpu_time: float, gpu_time: float):
        self.problem_sizes.append(n_vars)
        self.cpu_times.append(cpu_time)
        self.gpu_times.append(gpu_time)

    def get_recommended_device(self, n_vars: int) -> DeviceType:
        """Recommend device based on historical data."""
        if len(self.problem_sizes) < 3:
            return DeviceType.AUTO  # Not enough data

        # Find similar-sized problems
        sizes = np.array(self.problem_sizes)
        cpu_times = np.array(self.cpu_times)
        gpu_times = np.array(self.gpu_times)

        # Weight by similarity
        weights = 1.0 / (np.abs(sizes - n_vars) + 1)

        avg_cpu = np.average(cpu_times, weights=weights)
        avg_gpu = np.average(gpu_times, weights=weights)

        return DeviceType.GPU if avg_gpu < avg_cpu * 0.8 else DeviceType.CPU


class ComputeBackend(ABC):
    """Abstract base class for compute backends."""

    @abstractmethod
    def sparse_matvec(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """Compute A @ x"""
        pass

    @abstractmethod
    def sparse_matvec_T(self, A: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
        """Compute A^T @ y"""
        pass

    @abstractmethod
    def batch_sparse_matvec(self, A: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
        """Compute A @ X^T for batch of vectors (efficient GPU version)"""
        pass

    @abstractmethod
    def get_device_name(self) -> str:
        pass


class CPUBackend(ComputeBackend):
    """CPU backend using NumPy with Accelerate BLAS."""

    def __init__(self):
        self.name = "CPU (Accelerate BLAS)"
        # Check if Accelerate is being used
        try:
            import Accelerate
            self.has_accelerate = True
        except ImportError:
            self.has_accelerate = False

    def sparse_matvec(self, A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        return A @ x

    def sparse_matvec_T(self, A: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
        return A.T @ y

    def batch_sparse_matvec(self, A: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
        """Batch SpMV: compute A @ X^T for each row of X."""
        # Naive implementation - could be optimized with threading
        K = X.shape[0]
        results = np.zeros((K, A.shape[0]))
        for i in range(K):
            results[i] = A @ X[i]
        return results

    def get_device_name(self) -> str:
        return self.name


class MetalGPUBackend(ComputeBackend):
    """
    GPU backend using Metal Performance Shaders.

    Key optimizations for M-chip architecture:
    1. Zero-copy via unified memory
    2. Fused operations to minimize kernel launches
    3. Batch processing for population algorithms
    """

    def __init__(self):
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal framework not available")

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("No Metal device available")

        self.command_queue = self.device.newCommandQueue()
        self.name = f"GPU ({self.device.name()})"

        # Cached buffers for reuse
        self._buffer_cache: Dict[str, Any] = {}

        # Compile shaders
        self._compile_shaders()

        print(f"MetalGPUBackend initialized: {self.device.name()}")
        print(f"  Recommended max working set: {self.device.recommendedMaxWorkingSetSize() / 1e9:.1f} GB")

    def _compile_shaders(self):
        """Compile Metal shaders for PDHG operations."""
        # Metal Shader Language (MSL) code
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        // Sparse matrix-vector multiply (CSR format)
        // Each thread computes one element of the output vector
        kernel void spmv_csr(
            device const int* rowptr [[buffer(0)]],
            device const int* colind [[buffer(1)]],
            device const float* values [[buffer(2)]],
            device const float* x [[buffer(3)]],
            device float* y [[buffer(4)]],
            device const float* row_scale [[buffer(5)]],  // For Ruiz scaling
            device const float* col_scale [[buffer(6)]],  // For Ruiz scaling
            uint tid [[thread_position_in_grid]],
            uint grid_size [[threads_per_grid]]
        ) {
            if (tid >= grid_size) return;

            int row = tid;
            int start = rowptr[row];
            int end = rowptr[row + 1];

            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                int col = colind[i];
                float val = values[i];
                // Apply column scaling if provided
                if (col_scale != nullptr) {
                    val *= col_scale[col];
                }
                sum += val * x[col];
            }

            // Apply row scaling if provided
            if (row_scale != nullptr) {
                sum *= row_scale[row];
            }

            y[row] = sum;
        }

        // Batch sparse matrix-vector multiply
        // Each thread group handles one batch element
        kernel void batch_spmv_csr(
            device const int* rowptr [[buffer(0)]],
            device const int* colind [[buffer(1)]],
            device const float* values [[buffer(2)]],
            device const float* X [[buffer(3)]],      // (K, n) row-major
            device float* Y [[buffer(4)]],            // (K, m) row-major
            device const float* row_scale [[buffer(5)]],
            device const float* col_scale [[buffer(6)]],
            constant int& n_vars [[buffer(7)]],
            uint2 tid [[thread_position_in_grid]],
            uint2 grid_size [[threads_per_grid]]
        ) {
            uint batch_idx = tid.y;
            uint row = tid.x;

            if (batch_idx >= grid_size.y || row >= grid_size.x) return;

            int start = rowptr[row];
            int end = rowptr[row + 1];

            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                int col = colind[i];
                float val = values[i];
                if (col_scale != nullptr) {
                    val *= col_scale[col];
                }
                // X is row-major: X[batch_idx, col]
                sum += val * X[batch_idx * n_vars + col];
            }

            if (row_scale != nullptr) {
                sum *= row_scale[row];
            }

            Y[batch_idx * grid_size.x + row] = sum;
        }

        // Vector operations kernel (for fused updates)
        kernel void pdhg_update(
            device const float* x [[buffer(0)]],
            device const float* grad_x [[buffer(1)]],
            device const float* lb [[buffer(2)]],
            device const float* ub [[buffer(3)]],
            device float* x_new [[buffer(4)]],
            device const float& eta [[buffer(5)]],
            uint tid [[thread_position_in_grid]],
            uint grid_size [[threads_per_grid]]
        ) {
            if (tid >= grid_size) return;

            // Primal update with projection
            float val = x[tid] - eta * grad_x[tid];
            x_new[tid] = clamp(val, lb[tid], ub[tid]);
        }

        // Dual update with constraint sense handling
        kernel void dual_update(
            device const float* y [[buffer(0)]],
            device const float* grad_y [[buffer(1)]],
            device float* y_new [[buffer(2)]],
            device const bool* is_equality [[buffer(3)]],  // True if equality constraint
            device const float& tau [[buffer(4)]],
            uint tid [[thread_position_in_grid]],
            uint grid_size [[threads_per_grid]]
        ) {
            if (tid >= grid_size) return;

            float val = y[tid] + tau * grad_y[tid];

            // Project to y >= 0 only for inequality constraints
            if (!is_equality[tid]) {
                val = max(val, 0.0f);
            }

            y_new[tid] = val;
        }

        // Batch PDHG update (population-based)
        kernel void batch_pdhg_step(
            device const float* X [[buffer(0)]],      // (K, n) current population
            device const float* ATy [[buffer(1)]],    // (K, n) gradient
            device const float* c [[buffer(2)]],      // (n,) objective
            device const float* lb [[buffer(3)]],     // (n,)
            device const float* ub [[buffer(4)]],     // (n,)
            device float* X_new [[buffer(5)]],        // (K, n) updated
            device const float& eta [[buffer(6)]],
            constant int& n_vars [[buffer(7)]],
            uint2 tid [[thread_position_in_grid]],
            uint2 grid_size [[threads_per_grid]]
        ) {
            uint batch_idx = tid.y;
            uint var_idx = tid.x;

            if (batch_idx >= grid_size.y || var_idx >= grid_size.x) return;

            uint idx = batch_idx * n_vars + var_idx;

            // Primal gradient
            float grad = ATy[idx] + c[var_idx];

            // Primal update
            float val = X[idx] - eta * grad;
            X_new[idx] = clamp(val, lb[var_idx], ub[var_idx]);
        }

        // Reduction: find minimum objective in batch
        kernel void find_min_obj(
            device const float* X [[buffer(0)]],      // (K, n)
            device const float* c [[buffer(1)]],      // (n,)
            device float* objectives [[buffer(2)]],   // (K,) objectives
            device float* min_obj [[buffer(3)]],      // scalar output
            device int* min_idx [[buffer(4)]],        // scalar output
            constant int& n_vars [[buffer(5)]],
            threadgroup float* shared_obj [[threadgroup(0)]],
            threadgroup int* shared_idx [[threadgroup(1)]],
            uint tid [[thread_position_in_grid]],
            uint local_tid [[thread_position_in_threadgroup]],
            uint group_size [[threads_per_threadgroup]],
            uint grid_size [[threads_per_grid]]
        ) {
            // Compute objective for this batch element
            uint batch_idx = tid;
            if (batch_idx >= grid_size) return;

            float obj = 0.0f;
            for (int j = 0; j < n_vars; j++) {
                obj += X[batch_idx * n_vars + j] * c[j];
            }
            objectives[batch_idx] = obj;

            // Parallel reduction to find minimum
            shared_obj[local_tid] = obj;
            shared_idx[local_tid] = batch_idx;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = group_size / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    if (shared_obj[local_tid + stride] < shared_obj[local_tid]) {
                        shared_obj[local_tid] = shared_obj[local_tid + stride];
                        shared_idx[local_tid] = shared_idx[local_tid + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (local_tid == 0) {
                min_obj[0] = shared_obj[0];
                min_idx[0] = shared_idx[0];
            }
        }
        """

        # Create shader library
        result = self.device.newLibraryWithSource_options_error_(
            shader_source, None, None
        )

        # Handle PyObjC return value (might be tuple or single value)
        if isinstance(result, tuple):
            self.shader_library, error = result
            if error:
                print(f"Warning: Metal shader compilation error: {error}")
        else:
            self.shader_library = result

        if self.shader_library is None:
            print("Warning: Failed to compile Metal shaders, falling back to CPU")
            self.shaders_available = False
            return

        self.shaders_available = True

        # Get kernel functions
        self.spmv_kernel = self.shader_library.newFunctionWithName_("spmv_csr")
        self.batch_spmv_kernel = self.shader_library.newFunctionWithName_("batch_spmv_csr")
        self.pdhg_update_kernel = self.shader_library.newFunctionWithName_("pdhg_update")
        self.dual_update_kernel = self.shader_library.newFunctionWithName_("dual_update")
        self.batch_pdhg_kernel = self.shader_library.newFunctionWithName_("batch_pdhg_step")
        self.find_min_kernel = self.shader_library.newFunctionWithName_("find_min_obj")

        # Create compute pipelines
        result = self.device.newComputePipelineStateWithFunction_error_(
            self.spmv_kernel, None
        )
        if isinstance(result, tuple):
            self.spmv_pipeline, error = result
        else:
            self.spmv_pipeline = result

        result = self.device.newComputePipelineStateWithFunction_error_(
            self.batch_spmv_kernel, None
        )
        if isinstance(result, tuple):
            self.batch_spmv_pipeline, error = result
        else:
            self.batch_spmv_pipeline = result

        print("  Metal shaders compiled successfully")

    def _create_buffer(self, data: np.ndarray, name: str = None) -> 'MTLBuffer':
        """Create a Metal buffer from numpy array (zero-copy if possible)."""
        # Ensure correct dtype
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Create buffer with shared storage for zero-copy
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            data.nbytes,
            Metal.MTLResourceStorageModeShared  # Unified memory
        )

        if name:
            self._buffer_cache[name] = buffer

        return buffer

    def _upload_sparse_matrix(self, A: sp.csr_matrix) -> Tuple['MTLBuffer', 'MTLBuffer', 'MTLBuffer']:
        """Upload sparse matrix in CSR format to GPU."""
        A_csr = A.tocsr()

        rowptr = self._create_buffer(A_csr.indptr.astype(np.int32))
        colind = self._create_buffer(A_csr.indices.astype(np.int32))
        values = self._create_buffer(A_csr.data.astype(np.float32))

        return rowptr, colind, values

    def sparse_matvec(self, A: sp.csr_matrix, x: np.ndarray,
                      row_scale: np.ndarray = None,
                      col_scale: np.ndarray = None) -> np.ndarray:
        """Compute A @ x using GPU."""
        if not self.shaders_available:
            return A @ x

        m, n = A.shape

        # Upload data
        rowptr, colind, values = self._upload_sparse_matrix(A)
        x_buffer = self._create_buffer(x.astype(np.float32))
        y_buffer = self.device.newBufferWithLength_options_(
            m * 4, Metal.MTLResourceStorageModeShared
        )

        # Optional scaling buffers
        row_scale_buffer = self._create_buffer(row_scale) if row_scale is not None else None
        col_scale_buffer = self._create_buffer(col_scale) if col_scale is not None else None

        # Create command buffer
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.spmv_pipeline)
        encoder.setBuffer_offset_atIndex_(rowptr, 0, 0)
        encoder.setBuffer_offset_atIndex_(colind, 0, 1)
        encoder.setBuffer_offset_atIndex_(values, 0, 2)
        encoder.setBuffer_offset_atIndex_(x_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(y_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(row_scale_buffer, 0, 5) if row_scale_buffer else None
        encoder.setBuffer_offset_atIndex_(col_scale_buffer, 0, 6) if col_scale_buffer else None

        # Dispatch
        threadgroup_size = self.spmv_pipeline.maxTotalThreadsPerThreadgroup()
        grid_size = m
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(grid_size, 1, 1),
            Metal.MTLSizeMake(min(threadgroup_size, grid_size), 1, 1)
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read result
        result = np.frombuffer(y_buffer.contents(), dtype=np.float32, count=m)
        return result

    def sparse_matvec_T(self, A: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
        """Compute A^T @ y using GPU."""
        # For transpose, we use CSC format (which is CSR of A^T)
        A_csc = A.tocsc()
        return self.sparse_matvec(A_csc, y)

    def batch_sparse_matvec(self, A: sp.csr_matrix, X: np.ndarray,
                            row_scale: np.ndarray = None,
                            col_scale: np.ndarray = None) -> np.ndarray:
        """Batch SpMV: compute A @ X^T for each row of X using GPU."""
        if not self.shaders_available:
            return super().batch_sparse_matvec(A, X)

        K, n = X.shape
        m = A.shape[0]

        # Upload data
        rowptr, colind, values = self._upload_sparse_matrix(A)
        X_buffer = self._create_buffer(X.astype(np.float32).flatten())
        Y_buffer = self.device.newBufferWithLength_options_(
            K * m * 4, Metal.MTLResourceStorageModeShared
        )

        row_scale_buffer = self._create_buffer(row_scale) if row_scale is not None else None
        col_scale_buffer = self._create_buffer(col_scale) if col_scale is not None else None

        n_vars_buffer = self._create_buffer(np.array([n], dtype=np.int32))

        # Create command buffer
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.batch_spmv_pipeline)
        encoder.setBuffer_offset_atIndex_(rowptr, 0, 0)
        encoder.setBuffer_offset_atIndex_(colind, 0, 1)
        encoder.setBuffer_offset_atIndex_(values, 0, 2)
        encoder.setBuffer_offset_atIndex_(X_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(Y_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(row_scale_buffer, 0, 5) if row_scale_buffer else None
        encoder.setBuffer_offset_atIndex_(col_scale_buffer, 0, 6) if col_scale_buffer else None
        encoder.setBuffer_offset_atIndex_(n_vars_buffer, 0, 7)

        # Dispatch with 2D grid
        threadgroup_size = 32
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((m + threadgroup_size - 1) // threadgroup_size,
                             (K + threadgroup_size - 1) // threadgroup_size, 1),
            Metal.MTLSizeMake(threadgroup_size, threadgroup_size, 1)
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read result
        result = np.frombuffer(Y_buffer.contents(), dtype=np.float32, count=K * m).reshape(K, m)
        return result

    def get_device_name(self) -> str:
        return self.name


class DeviceSelector:
    """
    Adaptive device selector based on problem characteristics.

    Selection strategy:
    1. Small problems (< threshold_small): CPU (avoid GPU overhead)
    2. Medium problems (threshold_small to threshold_large): Benchmark both
    3. Large problems (> threshold_large): GPU

    Thresholds are tuned based on M-chip architecture.
    """

    def __init__(self):
        # Tunable thresholds (based on empirical testing)
        self.threshold_small = 500      # < 500 vars: CPU always faster
        self.threshold_large = 5000     # > 5000 vars: GPU likely faster

        # Performance history for adaptive learning
        self.profile = PerformanceProfile()

        # Try to initialize GPU
        self.gpu_available = METAL_AVAILABLE
        self.gpu_backend = None

        if self.gpu_available:
            try:
                self.gpu_backend = MetalGPUBackend()
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                self.gpu_available = False

        # CPU backend always available
        self.cpu_backend = CPUBackend()

        # Warm-up GPU
        if self.gpu_available:
            self._warmup_gpu()

    def _warmup_gpu(self):
        """Warm up GPU with a small computation to initialize kernels."""
        try:
            A = sp.eye(100, format='csr')
            x = np.ones(100, dtype=np.float32)
            _ = self.gpu_backend.sparse_matvec(A, x)
            print("  GPU warmup complete")
        except Exception as e:
            print(f"  GPU warmup failed: {e}")

    def select_device(self, chars: ProblemCharacteristics,
                      force: DeviceType = DeviceType.AUTO) -> ComputeBackend:
        """
        Select the best compute device for a problem.

        Args:
            chars: Problem characteristics
            force: Force specific device (for benchmarking)

        Returns:
            ComputeBackend instance
        """
        if force == DeviceType.CPU:
            return self.cpu_backend
        if force == DeviceType.GPU:
            return self.gpu_backend if self.gpu_available else self.cpu_backend

        # Auto selection logic
        n = chars.n_vars

        # Check historical performance first
        if len(self.profile.problem_sizes) >= 5:
            recommended = self.profile.get_recommended_device(n)
            if recommended == DeviceType.CPU:
                return self.cpu_backend
            if recommended == DeviceType.GPU and self.gpu_available:
                return self.gpu_backend

        # Rule-based selection
        if n < self.threshold_small:
            return self.cpu_backend

        if n > self.threshold_large and self.gpu_available:
            return self.gpu_backend

        # Medium-sized: prefer GPU if available, but could benchmark
        # For now, use CPU for medium problems (conservative)
        return self.cpu_backend

    def benchmark(self, A: sp.csr_matrix, x: np.ndarray, n_runs: int = 3) -> Dict:
        """
        Benchmark CPU vs GPU performance for a specific problem.

        Returns timing comparison.
        """
        results = {
            'cpu_time': [],
            'gpu_time': [],
            'n_vars': A.shape[1]
        }

        # CPU benchmark
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.cpu_backend.sparse_matvec(A, x)
            results['cpu_time'].append(time.perf_counter() - start)

        # GPU benchmark
        if self.gpu_available:
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.gpu_backend.sparse_matvec(A, x)
                results['gpu_time'].append(time.perf_counter() - start)

            # Record for adaptive learning
            self.profile.add_result(
                A.shape[1],
                np.mean(results['cpu_time']),
                np.mean(results['gpu_time'])
            )

        return results


class HeterogeneousSolver:
    """
    Main solver class with automatic CPU/GPU selection.

    This is the primary interface for heterogeneous PDHG solving.
    """

    def __init__(self, device: DeviceType = DeviceType.AUTO):
        self.selector = DeviceSelector()
        self.device_preference = device

    def solve_pdhg(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        constraint_sense: List[str] = None,
        max_iter: int = 2000,
        population_size: int = 1,  # 1 for single, >1 for batch
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Solve LP using heterogeneous PDHG.

        Automatically selects CPU or GPU based on problem characteristics.
        """
        m, n = A.shape

        # Analyze problem
        chars = ProblemCharacteristics.from_problem(A)

        # Select device
        backend = self.selector.select_device(chars, self.device_preference)

        if verbose:
            print(f"  Using device: {backend.get_device_name()}")
            print(f"  Problem: {n} vars, {m} constrs, density={chars.density:.4f}")

        # Convert constraint sense
        if constraint_sense is None:
            constraint_sense = ['L'] * m

        sense_work = list(constraint_sense)
        A_work = A.tocsr().copy()
        b_work = b.copy()

        # Convert >= to <=
        for i, sense in enumerate(sense_work):
            if sense == 'G':
                A_work[i, :] = -A_work[i, :]
                b_work[i] = -b_work[i]
                sense_work[i] = 'L'

        is_equality = np.array([s == 'E' for s in sense_work])
        is_inequality = ~is_equality

        # Initialize
        x = np.zeros(n, dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)
        x = np.clip(x, lb, ub)

        # Adaptive step sizes
        A_dense = A_work.toarray()
        norm_A = np.linalg.norm(A_dense, ord=2) + 1e-10
        eta = 1.0 / norm_A
        tau = 1.0 / norm_A

        best_obj = float('inf')
        best_x = x.copy()

        AT = A_work.T.tocsr()

        start_time = time.time()

        for k in range(max_iter):
            # Matrix operations using selected backend
            Ax = backend.sparse_matvec(A_work, x)
            ATy = backend.sparse_matvec(AT, y)

            # Primal update
            grad_x = ATy + c
            x_new = np.clip(x - eta * grad_x, lb, ub)

            # Over-relaxed
            x_bar = 2 * x_new - x

            # Dual gradient
            Ax_bar = backend.sparse_matvec(A_work, x_bar)
            grad_y = Ax_bar - b_work

            # Dual update
            y_new = y + tau * grad_y
            y_new[is_inequality] = np.maximum(y_new[is_inequality], 0)

            x = x_new
            y = y_new

            # Track best
            if k % 50 == 0:
                obj = float(c @ x)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()

        solve_time = time.time() - start_time

        stats = {
            'device': backend.get_device_name(),
            'iterations': max_iter,
            'solve_time': solve_time,
            'n_vars': n,
            'n_constraints': m
        }

        return best_x, best_obj, stats


# Convenience function
def get_heterogeneous_solver(device: str = "auto") -> HeterogeneousSolver:
    """
    Get a heterogeneous solver instance.

    Args:
        device: "cpu", "gpu", or "auto"

    Returns:
        HeterogeneousSolver instance
    """
    device_map = {
        "cpu": DeviceType.CPU,
        "gpu": DeviceType.GPU,
        "auto": DeviceType.AUTO
    }
    return HeterogeneousSolver(device=device_map.get(device, DeviceType.AUTO))


if __name__ == "__main__":
    print("="*60)
    print("Heterogeneous Computing Framework Test")
    print("="*60)

    # Test device selector
    selector = DeviceSelector()

    # Test with different problem sizes
    test_sizes = [100, 500, 1000, 5000]

    print("\nDevice Selection Test:")
    for n in test_sizes:
        A = sp.random(n, n, density=0.1, format='csr')
        chars = ProblemCharacteristics.from_problem(A)
        backend = selector.select_device(chars)
        print(f"  n={n:5d}: {backend.get_device_name()}")

    # Benchmark comparison
    print("\nBenchmark Test (n=1000):")
    A = sp.random(1000, 1000, density=0.05, format='csr')
    x = np.random.randn(1000).astype(np.float32)

    results = selector.benchmark(A, x, n_runs=5)

    print(f"  CPU avg: {np.mean(results['cpu_time'])*1000:.2f} ms")
    if results['gpu_time']:
        print(f"  GPU avg: {np.mean(results['gpu_time'])*1000:.2f} ms")
        speedup = np.mean(results['cpu_time']) / np.mean(results['gpu_time'])
        print(f"  Speedup: {speedup:.2f}x")

    print("\n✓ Heterogeneous framework test complete!")
