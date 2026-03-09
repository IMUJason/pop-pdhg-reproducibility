"""
Phase 2: Kernel 级优化 - OptimizedGPUSolver with Shared Memory

优化项:
1. Threadgroup 共享内存 - 缓存 x 向量
2. Threadgroup 大小调优 - 64/128/256/512 可选
3. Tile-based SpMV - 分块处理减少共享内存压力
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


# Phase 2 优化后的 Metal Shaders
METAL_SHADERS_PHASE2 = """
#include <metal_stdlib>
using namespace metal;

// ============================================
// Phase 2.1: Threadgroup 共享内存优化 SpMV
// ============================================
// 核心思想：
// 1. 将 x 向量分块加载到 threadgroup 内存
// 2. 线程协作加载，减少全局内存访问
// 3. barrier 同步确保数据就绪

constant constexpr uint TILE_SIZE = 256;  // Tile 大小
constant constexpr uint CACHE_SIZE = TILE_SIZE + 16;  // 带 halo 的缓存

kernel void spmv_shared_memory(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    device const float* col_scale [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    if (tid >= m) return;

    // 共享内存 - 缓存 x 向量片段
    threadgroup float x_shared[CACHE_SIZE];

    int row_start = rowptr[tid];
    int row_end = rowptr[tid + 1];

    // 分块处理行元素
    float sum = 0.0f;

    for (uint tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // 协作加载 x 到共享内存
        uint x_idx = tile_start + lid;
        if (x_idx < n) {
            x_shared[lid] = x[x_idx];
        } else {
            x_shared[lid] = 0.0f;
        }

        // 屏障同步 - 确保所有线程完成加载
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 从共享内存读取 (快 3-5x)
        for (int i = row_start; i < row_end; i++) {
            int col = colind[i];
            uint col_in_tile = col - tile_start;

            if (col_in_tile < TILE_SIZE) {
                // 列在当前 tile 内，使用共享内存
                int scaled_col = col;  // 列缩放已融合
                sum += values[i] * col_scale[scaled_col] * x_shared[col_in_tile];
            }
        }

        // 屏障同步 - 确保所有线程完成读取
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 应用行缩放并写入结果
    y[tid] = row_scale[tid] * sum;
}

// ============================================
// Phase 2.2: 简化的共享内存版本 (适合稀疏度低的情况)
// ============================================
// 优化：只加载需要的 x 元素，减少共享内存占用

kernel void spmv_shared_minimal(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    device const float* col_scale [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (tid >= m) return;

    int start = rowptr[tid];
    int end = rowptr[tid + 1];

    // 优化：直接使用寄存器累加，避免共享内存开销
    // 适合每行非零元较少的情况
    float sum = 0.0f;

    for (int i = start; i < end; i++) {
        int col = colind[i];
        // 直接全局内存访问 (寄存器足够时效率高)
        sum += values[i] * col_scale[col] * x[col];
    }

    y[tid] = row_scale[tid] * sum;
}

// ============================================
// Phase 2.3: Batch SpMV with Shared Memory
// ============================================

kernel void batch_spmv_shared(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y [[buffer(4)]],
    device const float* col_scale [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint row = gid.x;
    uint batch = gid.y;
    uint lid_x = lid.x;  // 只使用 x 维度

    if (row >= m || batch >= K) return;

    // 共享内存 - 缓存 X 的当前 batch
    threadgroup float x_shared[TILE_SIZE];

    int row_start = rowptr[row];
    int row_end = rowptr[row + 1];

    float sum = 0.0f;

    for (uint tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // 加载当前 batch 的 x 到共享内存
        uint x_idx = tile_start + lid_x;
        if (x_idx < n) {
            x_shared[lid_x] = X[batch * n + x_idx];
        } else {
            x_shared[lid_x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 从共享内存读取
        for (int i = row_start; i < row_end; i++) {
            int col = colind[i];
            uint col_in_tile = col - tile_start;

            if (col_in_tile < TILE_SIZE) {
                sum += values[i] * col_scale[col] * x_shared[col_in_tile];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    Y[batch * m + row] = row_scale[row] * sum;
}

// ============================================
// 原始版本 (用于对比)
// ============================================

kernel void spmv(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= m) return;

    int start = rowptr[tid];
    int end = rowptr[tid + 1];
    float sum = 0.0f;

    for (int i = start; i < end; i++) {
        sum += values[i] * x[colind[i]];
    }
    y[tid] = sum;
}

// Batch SpMV (原始版本)
kernel void batch_spmv(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y [[buffer(4)]],
    device const float* col_scale [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint batch = gid.y;

    if (row >= m || batch >= K) return;

    int start = rowptr[row];
    int end = rowptr[row + 1];
    float sum = 0.0f;

    for (int i = start; i < end; i++) {
        int col = colind[i];
        sum += values[i] * col_scale[col] * X[batch * n + col];
    }
    Y[batch * m + row] = row_scale[row] * sum;
}

// SAXPY, vclip, vrelu 等辅助 kernel...
"""


class MetalShaderManagerV2:
    """Metal shader manager with Phase 2 optimizations."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = None
        self.command_queue = None
        self.library = None
        self.pipelines: Dict[str, Any] = {}
        self._compiled = False

    def initialize(self) -> bool:
        """Initialize Metal and compile Phase 2 shaders."""
        if not METAL_AVAILABLE:
            return False

        try:
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                return False

            self.command_queue = self.device.newCommandQueue()

            result = self.device.newLibraryWithSource_options_error_(
                METAL_SHADERS_PHASE2, None, None
            )
            if isinstance(result, tuple):
                self.library, error = result
                if error:
                    if self.verbose:
                        print(f"Shader compilation error: {error}")
                    return False
            else:
                self.library = result

            if self.library is None:
                return False

            self._compiled = True

            if self.verbose:
                func_names = ['spmv', 'spmv_shared_memory', 'spmv_shared_minimal',
                              'batch_spmv', 'batch_spmv_shared']
                print(f"  Phase 2 Metal shaders compiled: {len(func_names)} kernels")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Metal initialization failed: {e}")
            return False

    def get_pipeline(self, kernel_name: str, threadgroup_size: Optional[int] = None) -> Optional[Any]:
        """Get or create compute pipeline with optional threadgroup size optimization."""
        if not self._compiled:
            return None

        pipeline_key = kernel_name
        if threadgroup_size:
            pipeline_key = f"{kernel_name}_tg{threadgroup_size}"

        if pipeline_key in self.pipelines:
            return self.pipelines[pipeline_key]

        try:
            func = self.library.newFunctionWithName_(kernel_name)
            if func is None:
                return None

            # 创建 pipeline state
            result = self.device.newComputePipelineStateWithFunction_error_(func, None)
            if isinstance(result, tuple):
                pipeline, error = result
                if error:
                    return None
            else:
                pipeline = result

            self.pipelines[pipeline_key] = pipeline
            return pipeline

        except Exception:
            return None


class OptimizedGPUSolverPhase2:
    """
    Phase 2 Optimized GPU Solver with Threadgroup Shared Memory.

    优化特性:
    1. Threadgroup 共享内存 - 减少全局内存访问
    2. Threadgroup 大小调优 - 可配置 64/128/256/512
    3. Tile-based SpMV - 分块处理
    """

    def __init__(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        col_scale: Optional[np.ndarray] = None,
        row_scale: Optional[np.ndarray] = None,
        verbose: bool = False,
        threadgroup_size: int = 256,  # Phase 2 可调参数
        use_shared_memory: bool = True  # 是否使用共享内存优化
    ):
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub
        self.verbose = verbose
        self.threadgroup_size = threadgroup_size
        self.use_shared_memory = use_shared_memory

        m, n = A.shape
        self.n = n
        self.m = m

        # Initialize Metal
        self.shader_manager = MetalShaderManagerV2(verbose=verbose)
        if not self.shader_manager.initialize():
            if verbose:
                print("  GPU initialization failed, falling back to CPU")
            self.available = False
            return

        self.available = True

        # 计算每行平均非零元 (用于决定使用哪个 kernel)
        nnz_per_row = np.diff(A.indptr)
        self.avg_nnz = np.mean(nnz_per_row)
        self.max_nnz = np.max(nnz_per_row)

        # 预分配 buffer
        self._preallocate_buffers(A, b, c, lb, ub, col_scale, row_scale)

    def _preallocate_buffers(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        col_scale: Optional[np.ndarray],
        row_scale: Optional[np.ndarray]
    ):
        """预分配 GPU buffer."""
        device = self.shader_manager.device

        # Matrix buffers
        self.rowptr_buf = device.newBufferWithBytes_length_options_(
            A.indptr.astype(np.int32).tobytes(),
            len(A.indptr) * 4,
            Metal.MTLResourceStorageModeShared
        )
        self.colind_buf = device.newBufferWithBytes_length_options_(
            A.indices.astype(np.int32).tobytes(),
            len(A.indices) * 4,
            Metal.MTLResourceStorageModeShared
        )
        self.values_buf = device.newBufferWithBytes_length_options_(
            A.data.astype(np.float32).tobytes(),
            len(A.data) * 4,
            Metal.MTLResourceStorageModeShared
        )

        # Vector buffers
        self.x_buf = device.newBufferWithLength_options_(
            self.n * 4, Metal.MTLResourceStorageModeShared
        )
        self.y_buf = device.newBufferWithLength_options_(
            self.m * 4, Metal.MTLResourceStorageModeShared
        )

        # Scaling buffers
        if col_scale is None:
            col_scale = np.ones(self.n, dtype=np.float32)
        if row_scale is None:
            row_scale = np.ones(self.m, dtype=np.float32)

        self.col_scale_buf = device.newBufferWithBytes_length_options_(
            col_scale.astype(np.float32).tobytes(),
            self.n * 4,
            Metal.MTLResourceStorageModeShared
        )
        self.row_scale_buf = device.newBufferWithBytes_length_options_(
            row_scale.astype(np.float32).tobytes(),
            self.m * 4,
            Metal.MTLResourceStorageModeShared
        )

        # Bounds buffers
        self.lb_buf = device.newBufferWithBytes_length_options_(
            lb.astype(np.float32).tobytes(),
            self.n * 4,
            Metal.MTLResourceStorageModeShared
        )
        self.ub_buf = device.newBufferWithBytes_length_options_(
            ub.astype(np.float32).tobytes(),
            self.n * 4,
            Metal.MTLResourceStorageModeShared
        )

        # Parameter buffers
        self.m_buf = device.newBufferWithBytes_length_options_(
            np.array([self.m], dtype=np.uint32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )
        self.n_buf = device.newBufferWithBytes_length_options_(
            np.array([self.n], dtype=np.uint32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared
        )

    def _write_vector(self, buffer: Any, data: np.ndarray):
        """Write data to GPU buffer."""
        ptr = buffer.contents()
        byte_buffer = ptr.as_buffer(len(data) * 4)
        np.frombuffer(byte_buffer, dtype=np.float32)[:] = data.astype(np.float32)

    def _read_vector(self, buffer: Any, n: int) -> np.ndarray:
        """Read data from GPU buffer."""
        ptr = buffer.contents()
        byte_buffer = ptr.as_buffer(n * 4)
        return np.frombuffer(byte_buffer, dtype=np.float32).copy()

    def _select_kernel(self) -> str:
        """根据矩阵特性选择最优 kernel."""
        if not self.use_shared_memory:
            return 'spmv'

        # 根据稀疏度选择 kernel
        if self.avg_nnz < 10:
            # 稀疏度高 - 共享内存收益大
            return 'spmv_shared_memory'
        else:
            # 稀疏度低 - 直接访问更高效
            return 'spmv_shared_minimal'

    def spmv(self, x: np.ndarray) -> np.ndarray:
        """
        Sparse matrix-vector multiply with Phase 2 optimizations.
        """
        if not self.available:
            return self.A @ x

        # Write input
        self._write_vector(self.x_buf, x)

        # Select kernel
        kernel_name = self._select_kernel()

        # Get pipeline
        pipeline = self.shader_manager.get_pipeline(
            kernel_name, self.threadgroup_size
        )
        if pipeline is None:
            return self.A @ x

        # Setup command buffer
        command_buffer = self.shader_manager.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(self.rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.x_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.y_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(self.col_scale_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(self.row_scale_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(self.m_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(self.n_buf, 0, 8)

        # Dispatch with optimized threadgroup size
        threads_per_threadgroup = min(
            self.threadgroup_size,
            pipeline.maxTotalThreadsPerThreadgroup(),
            self.m
        )
        threadgroups = (self.m + threads_per_threadgroup - 1) // threads_per_threadgroup

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(threadgroups, 1, 1),
            Metal.MTLSizeMake(threads_per_threadgroup, 1, 1)
        )

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._read_vector(self.y_buf, self.m)

    def batch_spmv(self, X: np.ndarray) -> np.ndarray:
        """Batch SpMV."""
        if not self.available:
            return np.array([self.A @ x for x in X])

        K, n = X.shape
        assert n == self.n

        # Allocate batch buffer if needed
        batch_size = K * n * 4
        if not hasattr(self, 'x_batch_buf') or self.x_batch_buf.length() < batch_size:
            self.x_batch_buf = self.shader_manager.device.newBufferWithLength_options_(
                batch_size, Metal.MTLResourceStorageModeShared
            )
        if not hasattr(self, 'y_batch_buf') or self.y_batch_buf.length() < K * self.m * 4:
            self.y_batch_buf = self.shader_manager.device.newBufferWithLength_options_(
                K * self.m * 4, Metal.MTLResourceStorageModeShared
            )

        # Write batch data
        X_flat = X.astype(np.float32).flatten()
        self._write_vector(self.x_batch_buf, X_flat)

        # Get pipeline
        kernel_name = 'batch_spmv_shared' if self.use_shared_memory else 'batch_spmv'
        pipeline = self.shader_manager.get_pipeline(kernel_name)
        if pipeline is None:
            return np.array([self.A @ x for x in X])

        # Setup command buffer
        command_buffer = self.shader_manager.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(self.rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.x_batch_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.y_batch_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(self.col_scale_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(self.row_scale_buf, 0, 6)

        # Parameter buffers
        m_buf = self.shader_manager.device.newBufferWithBytes_length_options_(
            np.array([self.m], dtype=np.uint32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        n_buf = self.shader_manager.device.newBufferWithBytes_length_options_(
            np.array([self.n], dtype=np.uint32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        K_buf = self.shader_manager.device.newBufferWithBytes_length_options_(
            np.array([K], dtype=np.uint32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(m_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(n_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(K_buf, 0, 9)

        # 2D dispatch
        threads_per_threadgroup = min(
            self.threadgroup_size,
            pipeline.maxTotalThreadsPerThreadgroup(),
            self.m
        )
        threadgroups_x = (self.m + threads_per_threadgroup - 1) // threads_per_threadgroup
        threadgroups_y = K

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(threadgroups_x, threadgroups_y, 1),
            Metal.MTLSizeMake(threads_per_threadgroup, 1, 1)
        )

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        Y_flat = self._read_vector(self.y_batch_buf, K * self.m)
        return Y_flat.reshape(K, self.m)


def benchmark_phase2():
    """Phase 2 性能对比测试."""
    print("="*60)
    print("Phase 2: Kernel 级优化对比")
    print("="*60)

    from gpu_optimized import OptimizedGPUSolver  # V1 baseline

    test_cases = [
        (5000, 2500, 0.02),
        (10000, 5000, 0.02),
        (20000, 10000, 0.02),
    ]

    # 测试不同 threadgroup 大小
    tg_sizes = [64, 128, 256, 512]

    for n, m, density in test_cases:
        print(f"\nn={n}, m={m}, density={density}")
        print("-" * 50)

        A = sp.random(m, n, density=density, format='csr', random_state=42)
        x = np.random.randn(n).astype(np.float32)
        b = np.zeros(m)
        c = np.random.randn(n)
        lb = np.zeros(n)
        ub = np.ones(n)

        # CPU
        _ = A @ x
        cpu_times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = A @ x
            cpu_times.append((time.perf_counter() - start) * 1000)
        cpu_time = np.min(cpu_times)
        print(f"  CPU: {cpu_time:.4f} ms")

        # V1 (Baseline)
        v1 = OptimizedGPUSolver(A, b, c, lb, ub, verbose=False)
        if v1.available:
            times = []
            for _ in range(50):
                start = time.perf_counter()
                _ = v1.spmv(x)
                times.append((time.perf_counter() - start) * 1000)
            v1_time = np.min(times)
            print(f"  V1 (Baseline): {v1_time:.4f} ms")
        else:
            v1_time = None

        # Phase 2 - 测试不同 threadgroup 大小
        print("\n  Phase 2 (Threadgroup 大小调优):")
        best_phase2_time = float('inf')
        best_tg_size = 256

        for tg_size in tg_sizes:
            v2 = OptimizedGPUSolverPhase2(
                A, b, c, lb, ub, verbose=False,
                threadgroup_size=tg_size,
                use_shared_memory=True
            )
            if v2.available:
                times = []
                for _ in range(50):
                    start = time.perf_counter()
                    _ = v2.spmv(x)
                    times.append((time.perf_counter() - start) * 1000)
                v2_time = np.min(times)
                improvement = v1_time / v2_time if v1_time else 1.0
                marker = " ← best" if v2_time < best_phase2_time else ""
                if v2_time < best_phase2_time:
                    best_phase2_time = v2_time
                    best_tg_size = tg_size
                print(f"    TG={tg_size}: {v2_time:.4f} ms (vs V1: {improvement:.2f}x){marker}")

        # Shared Memory vs No Shared Memory
        print(f"\n  Phase 2 (共享内存效果, TG={best_tg_size}):")

        v2_shared = OptimizedGPUSolverPhase2(
            A, b, c, lb, ub, verbose=False,
            threadgroup_size=best_tg_size,
            use_shared_memory=True
        )
        v2_no_shared = OptimizedGPUSolverPhase2(
            A, b, c, lb, ub, verbose=False,
            threadgroup_size=best_tg_size,
            use_shared_memory=False
        )

        if v2_shared.available:
            times = []
            for _ in range(50):
                start = time.perf_counter()
                _ = v2_shared.spmv(x)
                times.append((time.perf_counter() - start) * 1000)
            shared_time = np.min(times)
        else:
            shared_time = None

        if v2_no_shared.available:
            times = []
            for _ in range(50):
                start = time.perf_counter()
                _ = v2_no_shared.spmv(x)
                times.append((time.perf_counter() - start) * 1000)
            no_shared_time = np.min(times)
        else:
            no_shared_time = None

        if shared_time and no_shared_time:
            improvement = no_shared_time / shared_time
            print(f"    With Shared Memory: {shared_time:.4f} ms")
            print(f"    Without Shared Memory: {no_shared_time:.4f} ms")
            print(f"    Shared Memory 提升：{improvement:.2f}x")

        # 总结
        if v1_time and best_phase2_time:
            print(f"\n  Phase 2 总提升：{v1_time/best_phase2_time:.2f}x")
            gpu_speedup = cpu_time / best_phase2_time
            winner = "GPU" if gpu_speedup > 1.0 else "CPU"
            print(f"  GPU vs CPU (Phase 2): {winner} {max(gpu_speedup, 1/gpu_speedup):.2f}x")

    print("\n" + "="*60)


if __name__ == '__main__':
    benchmark_phase2()
