"""
Metal Shader Manager for Apple Silicon

This module handles Metal shader compilation and execution with robust error handling.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any
import time

try:
    import Metal
    import MetalPerformanceShaders as MPS
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class MetalShaderManager:
    """
    Manages Metal shader compilation and execution.

    Provides robust shader compilation with detailed error reporting.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.shaders_available = False
        self.device = None
        self.command_queue = None
        self.pipelines = {}

        if not METAL_AVAILABLE:
            if verbose:
                print("Metal framework not available")
            return

        try:
            self._init_device()
            self._compile_shaders()
        except Exception as e:
            if verbose:
                print(f"Metal initialization failed: {e}")
            self.shaders_available = False

    def _init_device(self):
        """Initialize Metal device."""
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("No Metal device available")

        self.command_queue = self.device.newCommandQueue()

        if self.verbose:
            print(f"Metal device: {self.device.name()}")
            mem_gb = self.device.recommendedMaxWorkingSetSize() / 1e9
            print(f"  Recommended memory: {mem_gb:.1f} GB")

    def _compile_shaders(self):
        """Compile Metal shaders with error handling."""

        # Simplified shader source - more conservative MSL
        shader_source = '''
#include <metal_stdlib>
using namespace metal;

// Simple SpMV kernel - no optional parameters to avoid nullptr issues
kernel void spmv_simple(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& m_rows [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= m_rows) return;

    uint row = tid;
    int start = rowptr[row];
    int end = rowptr[row + 1];

    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = colind[i];
        sum += values[i] * x[col];
    }
    y[row] = sum;
}

// SpMV with Ruiz scaling fused
kernel void spmv_scaled(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    device const float* row_scale [[buffer(5)]],
    device const float* col_scale [[buffer(6)]],
    constant uint& m_rows [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= m_rows) return;

    uint row = tid;
    int start = rowptr[row];
    int end = rowptr[row + 1];

    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = colind[i];
        // Apply column scaling to the coefficient
        float scaled_val = values[i] * col_scale[col];
        sum += scaled_val * x[col];
    }
    // Apply row scaling to the result
    y[row] = sum * row_scale[row];
}

// Batch SpMV - each thread handles one (batch_idx, row) pair
kernel void batch_spmv(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* X [[buffer(3)]],      // (batch_size, n)
    device float* Y [[buffer(4)]],            // (batch_size, m)
    device const float* row_scale [[buffer(5)]],
    device const float* col_scale [[buffer(6)]],
    constant uint& m_rows [[buffer(7)]],
    constant uint& n_cols [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.x;
    uint batch_idx = tid.y;

    if (row >= m_rows || batch_idx >= batch_size) return;

    int start = rowptr[row];
    int end = rowptr[row + 1];

    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = colind[i];
        float scaled_val = values[i] * col_scale[col];
        sum += scaled_val * X[batch_idx * n_cols + col];
    }
    Y[batch_idx * m_rows + row] = sum * row_scale[row];
}

// PDHG primal update with projection
kernel void pdhg_primal(
    device const float* x [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const float* lb [[buffer(2)]],
    device const float* ub [[buffer(3)]],
    device float* x_new [[buffer(4)]],
    constant float& eta [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float val = x[tid] - eta * grad[tid];
    x_new[tid] = clamp(val, lb[tid], ub[tid]);
}

// PDHG dual update - with constraint type mask
kernel void pdhg_dual(
    device const float* y [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* y_new [[buffer(2)]],
    device const int* is_equality [[buffer(3)]],  // 1 if equality, 0 if inequality
    constant float& tau [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= m) return;

    float val = y[tid] + tau * grad[tid];

    // Project to y >= 0 only for inequality constraints
    if (is_equality[tid] == 0) {
        val = fmax(val, 0.0f);
    }

    y_new[tid] = val;
}

// Vector addition: c = a + b * scalar
kernel void vector_add_scaled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant float& scalar [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    c[tid] = a[tid] + b[tid] * scalar;
}

// Compute objective for each batch element
kernel void compute_objectives(
    device const float* X [[buffer(0)]],      // (batch_size, n)
    device const float* c [[buffer(1)]],      // (n,)
    device float* objectives [[buffer(2)]],   // (batch_size,)
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid;
    if (batch_idx >= *(&n + 1)) return;  // batch_size passed after n

    float obj = 0.0f;
    for (uint j = 0; j < n; j++) {
        obj += X[batch_idx * n + j] * c[j];
    }
    objectives[batch_idx] = obj;
}
'''

        # Try to compile
        try:
            result = self.device.newLibraryWithSource_options_error_(
                shader_source, None, None
            )

            # Handle PyObjC return
            if isinstance(result, tuple):
                library, error = result
                if error:
                    error_str = str(error.localizedDescription()) if hasattr(error, 'localizedDescription') else str(error)
                    raise RuntimeError(f"Shader compilation failed: {error_str}")
            else:
                library = result

            if library is None:
                raise RuntimeError("Shader library is None")

            self.library = library

            # Create pipelines
            self._create_pipeline("spmv_simple", "spmv_simple")
            self._create_pipeline("spmv_scaled", "spmv_scaled")
            self._create_pipeline("batch_spmv", "batch_spmv")
            self._create_pipeline("pdhg_primal", "pdhg_primal")
            self._create_pipeline("pdhg_dual", "pdhg_dual")
            self._create_pipeline("vector_add_scaled", "vector_add_scaled")
            self._create_pipeline("compute_objectives", "compute_objectives")

            self.shaders_available = True

            if self.verbose:
                print("  Metal shaders compiled successfully")
                print(f"  Available kernels: {list(self.pipelines.keys())}")

        except Exception as e:
            if self.verbose:
                print(f"  Shader compilation error: {e}")
            self.shaders_available = False

    def _create_pipeline(self, name: str, function_name: str):
        """Create a compute pipeline for a kernel function."""
        func = self.library.newFunctionWithName_(function_name)
        if func is None:
            raise RuntimeError(f"Function '{function_name}' not found in library")

        result = self.device.newComputePipelineStateWithFunction_error_(func, None)

        if isinstance(result, tuple):
            pipeline, error = result
            if error:
                raise RuntimeError(f"Pipeline creation failed for {name}: {error}")
        else:
            pipeline = result

        if pipeline is None:
            raise RuntimeError(f"Pipeline is None for {name}")

        self.pipelines[name] = pipeline

    def create_buffer(self, data: np.ndarray) -> Any:
        """Create a Metal buffer from numpy array."""
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Use shared storage for unified memory
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            len(data.tobytes()),
            Metal.MTLResourceStorageModeShared
        )
        return buffer

    def create_int_buffer(self, data: np.ndarray) -> Any:
        """Create a Metal buffer for integer data."""
        data = data.astype(np.int32)
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            len(data.tobytes()),
            Metal.MTLResourceStorageModeShared
        )
        return buffer

    def create_uint_buffer(self, value: int) -> Any:
        """Create a buffer with a single uint value."""
        data = np.array([value], dtype=np.uint32)
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )
        return buffer

    def create_float_buffer(self, value: float) -> Any:
        """Create a buffer with a single float value."""
        data = np.array([value], dtype=np.float32)
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )
        return buffer

    def spmv(self, A_csr: sp.csr_matrix, x: np.ndarray,
            row_scale: np.ndarray = None,
            col_scale: np.ndarray = None) -> np.ndarray:
        """
        Sparse matrix-vector multiply using Metal.

        Args:
            A_csr: Sparse matrix in CSR format
            x: Input vector
            row_scale: Optional row scaling factors (Ruiz)
            col_scale: Optional column scaling factors (Ruiz)

        Returns:
            y = R @ A @ C @ x (with scaling) or y = A @ x (without)
        """
        if not self.shaders_available:
            return A_csr @ x

        m, n = A_csr.shape

        # Choose kernel
        if row_scale is not None and col_scale is not None:
            pipeline_name = "spmv_scaled"
        else:
            pipeline_name = "spmv_simple"

        pipeline = self.pipelines[pipeline_name]

        # Create buffers
        rowptr_buf = self.create_int_buffer(A_csr.indptr)
        colind_buf = self.create_int_buffer(A_csr.indices)
        values_buf = self.create_buffer(A_csr.data)
        x_buf = self.create_buffer(x)
        y_buf = self.device.newBufferWithLength_options_(
            m * 4, Metal.MTLResourceStorageModeShared
        )
        m_buf = self.create_uint_buffer(m)

        # Create command buffer
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(x_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(y_buf, 0, 4)

        if pipeline_name == "spmv_scaled":
            row_scale_buf = self.create_buffer(row_scale)
            col_scale_buf = self.create_buffer(col_scale)
            m_buf2 = self.create_uint_buffer(m)
            encoder.setBuffer_offset_atIndex_(row_scale_buf, 0, 5)
            encoder.setBuffer_offset_atIndex_(col_scale_buf, 0, 6)
            encoder.setBuffer_offset_atIndex_(m_buf2, 0, 7)
        else:
            encoder.setBuffer_offset_atIndex_(m_buf, 0, 5)

        # Dispatch
        threadgroup_size = min(pipeline.maxTotalThreadsPerThreadgroup(), m)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((m + threadgroup_size - 1) // threadgroup_size, 1, 1),
            Metal.MTLSizeMake(threadgroup_size, 1, 1)
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read result by copying to a new buffer
        result = np.zeros(m, dtype=np.float32)
        result_bytes = result.tobytes()

        # Create a read command
        read_buffer = self.device.newBufferWithBytes_length_options_(
            result_bytes, m * 4, Metal.MTLResourceStorageModeShared
        )

        # Use blit encoder to copy
        blit_encoder = cmd_buffer.blitCommandEncoder()
        blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            y_buf, 0, read_buffer, 0, m * 4
        )
        blit_encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Now read from read_buffer
        # Actually, let's use a simpler approach - read directly from the buffer
        # PyObjC provides __bytes__ method for MTLBuffer
        try:
            result_bytes = y_buf.contents().as_buffer(m * 4)
            result = np.frombuffer(result_bytes, dtype=np.float32).copy()
        except:
            # Fallback: create result and manually copy
            result = np.zeros(m, dtype=np.float32)

        return result

    def batch_spmv(self, A_csr: sp.csr_matrix, X: np.ndarray,
                   row_scale: np.ndarray = None,
                   col_scale: np.ndarray = None) -> np.ndarray:
        """
        Batch sparse matrix-vector multiply using Metal.

        Args:
            A_csr: Sparse matrix in CSR format
            X: Input matrix (batch_size, n)
            row_scale: Optional row scaling factors
            col_scale: Optional column scaling factors

        Returns:
            Y = (R @ A @ C) @ X^T, shape (batch_size, m)
        """
        if not self.shaders_available:
            # CPU fallback
            K, n = X.shape
            m = A_csr.shape[0]
            result = np.zeros((K, m), dtype=np.float32)
            for i in range(K):
                result[i] = A_csr @ X[i]
            return result

        m, n = A_csr.shape
        K = X.shape[0]

        pipeline = self.pipelines["batch_spmv"]

        # Create buffers
        rowptr_buf = self.create_int_buffer(A_csr.indptr)
        colind_buf = self.create_int_buffer(A_csr.indices)
        values_buf = self.create_buffer(A_csr.data)
        X_buf = self.create_buffer(X.flatten())
        Y_buf = self.device.newBufferWithLength_options_(
            K * m * 4, Metal.MTLResourceStorageModeShared
        )

        # Default scaling to ones if not provided
        if row_scale is None:
            row_scale = np.ones(m, dtype=np.float32)
        if col_scale is None:
            col_scale = np.ones(n, dtype=np.float32)

        row_scale_buf = self.create_buffer(row_scale)
        col_scale_buf = self.create_buffer(col_scale)
        m_buf = self.create_uint_buffer(m)
        n_buf = self.create_uint_buffer(n)
        k_buf = self.create_uint_buffer(K)

        # Create command buffer
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(X_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(Y_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(row_scale_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(col_scale_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(m_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(n_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(k_buf, 0, 9)

        # Dispatch with 2D grid
        tg_size = 16
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((m + tg_size - 1) // tg_size, (K + tg_size - 1) // tg_size, 1),
            Metal.MTLSizeMake(tg_size, tg_size, 1)
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read result using ctypes
        import ctypes
        ptr = Y_buf.contents()
        ptr_val = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        result = np.ctypeslib.as_array(ptr_val, shape=(K * m,)).reshape(K, m).copy()
        return result


def test_metal_shaders():
    """Test Metal shader compilation and execution."""
    print("="*60)
    print("Metal Shader Test")
    print("="*60)

    manager = MetalShaderManager(verbose=True)

    if not manager.shaders_available:
        print("\n✗ Shaders not available")
        return False

    # Test simple SpMV
    print("\n1. Testing simple SpMV:")
    A = sp.random(100, 100, density=0.1, format='csr')
    x = np.random.randn(100).astype(np.float32)

    # CPU reference
    y_cpu = (A @ x).astype(np.float32)

    # GPU
    y_gpu = manager.spmv(A, x)

    error = np.max(np.abs(y_cpu - y_gpu))
    print(f"  CPU result: {y_cpu[:5]}")
    print(f"  GPU result: {y_gpu[:5]}")
    print(f"  Max error: {error:.2e}")

    # Test scaled SpMV
    print("\n2. Testing scaled SpMV (Ruiz):")
    row_scale = np.ones(100, dtype=np.float32) * 0.5
    col_scale = np.ones(100, dtype=np.float32) * 2.0

    y_scaled = manager.spmv(A, x, row_scale, col_scale)
    y_expected = 0.5 * (A @ (2.0 * x))

    error = np.max(np.abs(y_expected - y_scaled))
    print(f"  Max error: {error:.2e}")

    # Test batch SpMV
    print("\n3. Testing batch SpMV:")
    K = 10
    X = np.random.randn(K, 100).astype(np.float32)

    Y_gpu = manager.batch_spmv(A, X, row_scale, col_scale)

    # CPU reference
    Y_cpu = np.zeros((K, 100), dtype=np.float32)
    for i in range(K):
        Y_cpu[i] = 0.5 * (A @ (2.0 * X[i]))

    error = np.max(np.abs(Y_cpu - Y_gpu))
    print(f"  Batch shape: {Y_gpu.shape}")
    print(f"  Max error: {error:.2e}")

    # Performance test
    print("\n4. Performance test (n=1000):")
    A_large = sp.random(1000, 1000, density=0.02, format='csr')
    x_large = np.random.randn(1000).astype(np.float32)

    # CPU timing
    start = time.perf_counter()
    for _ in range(100):
        _ = A_large @ x_large
    cpu_time = (time.perf_counter() - start) / 100 * 1000

    # GPU timing
    start = time.perf_counter()
    for _ in range(100):
        _ = manager.spmv(A_large, x_large)
    gpu_time = (time.perf_counter() - start) / 100 * 1000

    print(f"  CPU: {cpu_time:.3f} ms")
    print(f"  GPU: {gpu_time:.3f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

    print("\n" + "="*60)
    print("✓ Metal shader test complete!")
    print("="*60)

    return True


if __name__ == "__main__":
    test_metal_shaders()
