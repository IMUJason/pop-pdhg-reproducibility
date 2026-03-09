"""
Metal Shader Manager - Simplified and Robust

This module provides a working Metal GPU implementation for Apple Silicon.
Uses shared buffers that can be read directly without complex copying.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any
import time
import ctypes

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class MetalCompute:
    """
    Simplified Metal GPU compute interface.

    Uses shared storage buffers for direct CPU/GPU memory access.
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

    def _compile_shaders(self):
        """Compile Metal shaders."""

        # Simple shader with no optional parameters
        shader_source = '''
#include <metal_stdlib>
using namespace metal;

kernel void spmv(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int row = int(tid);
    int start = rowptr[row];
    int end = rowptr[row + 1];

    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += values[i] * x[colind[i]];
    }
    y[row] = sum;
}

kernel void saxpy(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* z [[buffer(2)]],
    constant float& a [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    z[tid] = a * x[tid] + y[tid];
}

kernel void vmax_inplace(
    device float* x [[buffer(0)]],
    constant float& val [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    x[tid] = max(x[tid], val);
}

kernel void vclip(
    device const float* x [[buffer(0)]],
    device const float* lb [[buffer(1)]],
    device const float* ub [[buffer(2)]],
    device float* y [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    y[tid] = clamp(x[tid], lb[tid], ub[tid]);
}
'''

        # Compile
        result = self.device.newLibraryWithSource_options_error_(shader_source, None, None)

        if isinstance(result, tuple):
            library, error = result
            if error:
                raise RuntimeError(f"Shader compilation failed: {error}")
        else:
            library = result

        if library is None:
            raise RuntimeError("Shader library is None")

        self.library = library

        # Create pipelines
        for name in ["spmv", "saxpy", "vmax_inplace", "vclip"]:
            func = self.library.newFunctionWithName_(name)
            if func is None:
                raise RuntimeError(f"Function '{name}' not found")

            result = self.device.newComputePipelineStateWithFunction_error_(func, None)
            if isinstance(result, tuple):
                pipeline, _ = result
            else:
                pipeline = result

            if pipeline:
                self.pipelines[name] = pipeline

        self.shaders_available = len(self.pipelines) == 4

        if self.verbose:
            print(f"  Compiled {len(self.pipelines)} kernels")

    def _create_buffer(self, size_bytes: int) -> Any:
        """Create an empty shared buffer."""
        return self.device.newBufferWithLength_options_(
            size_bytes,
            Metal.MTLResourceStorageModeShared
        )

    def _create_buffer_from_data(self, data: np.ndarray) -> Any:
        """Create a buffer and initialize with data."""
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            len(data.tobytes()),
            Metal.MTLResourceStorageModeShared
        )

    def _create_int_buffer_from_data(self, data: np.ndarray) -> Any:
        """Create a buffer with int32 data."""
        data = data.astype(np.int32)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            len(data.tobytes()),
            Metal.MTLResourceStorageModeShared
        )

    def _read_buffer(self, buffer: Any, n: int) -> np.ndarray:
        """Read float32 values from a buffer using as_buffer() method."""
        contents = buffer.contents()
        # as_buffer() is the correct PyObjC method for reading MTLBuffer contents
        byte_buffer = contents.as_buffer(n * 4)
        return np.frombuffer(byte_buffer, dtype=np.float32).copy()

    def spmv(self, A_csr: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """Sparse matrix-vector multiply on GPU."""
        if not self.shaders_available:
            return A_csr @ x

        m, n = A_csr.shape

        # Create buffers
        rowptr_buf = self._create_int_buffer_from_data(A_csr.indptr)
        colind_buf = self._create_int_buffer_from_data(A_csr.indices)
        values_buf = self._create_buffer_from_data(A_csr.data.astype(np.float32))
        x_buf = self._create_buffer_from_data(x.astype(np.float32))
        y_buf = self._create_buffer(m * 4)

        # Execute kernel
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.pipelines["spmv"])
        encoder.setBuffer_offset_atIndex_(rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(x_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(y_buf, 0, 4)

        # Dispatch
        tg_size = self.pipelines["spmv"].maxTotalThreadsPerThreadgroup()
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((m + tg_size - 1) // tg_size, 1, 1),
            Metal.MTLSizeMake(min(tg_size, m), 1, 1)
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read result
        return self._read_buffer(y_buf, m)


def test_metal():
    """Test Metal GPU computation."""
    print("="*60)
    print("Metal GPU Compute Test")
    print("="*60)

    compute = MetalCompute(verbose=True)

    if not compute.shaders_available:
        print("\n✗ Shaders not available, test skipped")
        return

    # Test SpMV
    print("\n1. SpMV Correctness Test:")
    np.random.seed(42)

    for n in [100, 500, 1000]:
        A = sp.random(n, n, density=0.05, format='csr')
        x = np.random.randn(n).astype(np.float32)

        y_cpu = (A @ x).astype(np.float32)
        y_gpu = compute.spmv(A, x)

        error = np.max(np.abs(y_cpu - y_gpu))
        status = "✓" if error < 1e-5 else "✗"
        print(f"  n={n:4d}: max_error={error:.2e} {status}")

    # Performance test
    print("\n2. Performance Test (n=2000, 100 runs):")
    n = 2000
    A = sp.random(n, n, density=0.02, format='csr')
    x = np.random.randn(n).astype(np.float32)

    # CPU
    start = time.perf_counter()
    for _ in range(100):
        _ = A @ x
    cpu_time = (time.perf_counter() - start) / 100 * 1000

    # GPU
    start = time.perf_counter()
    for _ in range(100):
        _ = compute.spmv(A, x)
    gpu_time = (time.perf_counter() - start) / 100 * 1000

    print(f"  CPU: {cpu_time:.3f} ms")
    print(f"  GPU: {gpu_time:.3f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

    print("\n" + "="*60)
    print("✓ Metal test complete!")
    print("="*60)


if __name__ == "__main__":
    test_metal()
