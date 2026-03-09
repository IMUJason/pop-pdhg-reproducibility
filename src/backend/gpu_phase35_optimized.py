"""
Phase 3.5: 深度优化 - 共享内存 + Batch 策略重构

核心优化:
1. 智能共享内存 - 根据稀疏模式动态选择 tile 大小
2. 混合 Batch 策略 - occupancy 感知，动态选择 1D vs 2D
3. Register 压力优化 - 减少 per-thread 寄存器使用
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import time

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False



# Phase 3.5 优化后的 Metal Shaders
METAL_SHADERS_PHASE35 = """
#include <metal_stdlib>
using namespace metal;

// ============================================
// Phase 3.5 优化 1: 智能共享内存 (减少 barrier)
// ============================================
// 核心思想:
// 1. 只在一个 tile 内处理，避免多次 barrier
// 2. 使用更小的 tile 大小，减少共享内存占用
// 3. 针对列索引局部性优化

constant constexpr uint OPT_TILE_SIZE = 128;  // 更小的 tile

// 优化版：单 tile 共享内存 (适合列索引集中的情况)
kernel void spmv_shared_smart(
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
    uint lid [[thread_position_in_threadgroup]]
) {
    if (tid >= m) return;

    int row_start = rowptr[tid];
    int row_end = rowptr[tid + 1];

    // 找出本行列索引范围
    int min_col = n;
    int max_col = 0;
    for (int i = row_start; i < row_end; i++) {
        int col = colind[i];
        if (col < min_col) min_col = col;
        if (col > max_col) max_col = col;
    }

    int col_range = max_col - min_col + 1;

    // 如果列范围小于 tile 大小，使用共享内存
    if (col_range <= (int)OPT_TILE_SIZE) {
        threadgroup float x_shared[OPT_TILE_SIZE];

        // 协作加载相关 x 到共享内存
        for (uint i = lid; i < (uint)col_range; i += OPT_TILE_SIZE) {
            x_shared[i] = x[min_col + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 从共享内存读取 (只有一次 barrier)
        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            int col = colind[i];
            sum += values[i] * col_scale[col] * x_shared[col - min_col];
        }
        y[tid] = row_scale[tid] * sum;
    } else {
        // 列范围太大，直接全局内存访问
        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            int col = colind[i];
            sum += values[i] * col_scale[col] * x[col];
        }
        y[tid] = row_scale[tid] * sum;
    }
}

// ============================================
// Phase 3.5 优化 2: 优化的 Batch SpMV
// ============================================
// 核心思想:
// 1. occupancy 感知 - 避免过度并行
// 2. 分层处理 - 大 batch 时分组处理

// 1D 版本：顺序处理每个 batch (适合大 K)
kernel void batch_spmv_1d(
    device const int* rowptr [[buffer(0)]],
    device const int* colind [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* X [[buffer(3)]],  // (K, n) row-major
    device float* Y [[buffer(4)]],        // (K, m) row-major
    device const float* col_scale [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]  // (row, 1, batch)
) {
    uint row = gid.x;
    uint batch = gid.z;  // 使用 z 维度

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

// 2D 版本：并行处理 (适合小 K)
kernel void batch_spmv_2d_opt(
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
    // uint lid_x = lid.x;  // 暂不使用

    if (row >= m || batch >= K) return;

    int start = rowptr[row];
    int end = rowptr[row + 1];

    // 优化：使用寄存器累加，避免共享内存
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = colind[i];
        sum += values[i] * col_scale[col] * X[batch * n + col];
    }
    Y[batch * m + row] = row_scale[row] * sum;
}

// ============================================
// 原始版本保留
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
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        int col = colind[i];
        sum += values[i] * col_scale[col] * x[col];
    }
    y[tid] = row_scale[tid] * sum;
}

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
"""


class MetalShaderManagerV35:
    """Phase 3.5 Shader Manager."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = None
        self.command_queue = None
        self.library = None
        self.pipelines: Dict[str, Any] = {}
        self._compiled = False

    def initialize(self) -> bool:
        if not METAL_AVAILABLE:
            return False

        try:
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                return False

            self.command_queue = self.device.newCommandQueue()

            result = self.device.newLibraryWithSource_options_error_(
                METAL_SHADERS_PHASE35, None, None
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
                func_names = ['spmv', 'spmv_shared_minimal', 'spmv_shared_smart',
                              'batch_spmv', 'batch_spmv_1d', 'batch_spmv_2d_opt']
                print(f"  Phase 3.5 shaders compiled: {len(func_names)} kernels")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Metal initialization failed: {e}")
            return False

    def get_pipeline(self, kernel_name: str) -> Optional[Any]:
        if not self._compiled:
            return None

        if kernel_name in self.pipelines:
            return self.pipelines[kernel_name]

        try:
            func = self.library.newFunctionWithName_(kernel_name)
            if func is None:
                return None

            result = self.device.newComputePipelineStateWithFunction_error_(func, None)
            if isinstance(result, tuple):
                pipeline, error = result
                if error:
                    return None
            else:
                pipeline = result

            self.pipelines[kernel_name] = pipeline
            return pipeline

        except Exception:
            return None


class OptimizedGPUSolverPhase35:
    """
    Phase 3.5: 深度优化 GPU Solver

    优化特性:
    1. 智能共享内存 - 根据列索引范围动态选择
    2. occupancy 感知 Batch 策略 - 动态 1D vs 2D
    3. Register 压力优化
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
        threadgroup_size: int = 256
    ):
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub
        self.verbose = verbose
        self.threadgroup_size = threadgroup_size

        m, n = A.shape
        self.n = n
        self.m = m

        # 分析稀疏模式
        nnz_per_row = np.diff(A.indptr)
        self.avg_nnz = float(np.mean(nnz_per_row))
        self.max_nnz = int(np.max(nnz_per_row))

        # 分析列索引范围 (用于共享内存决策)
        col_ranges = []
        for i in range(m):
            start, end = A.indptr[i], A.indptr[i + 1]
            if end > start:
                col_range = A.indices[start:end].max() - A.indices[start:end].min()
                col_ranges.append(col_range)
        self.avg_col_range = float(np.mean(col_ranges)) if col_ranges else 0

        # Initialize Metal
        self.shader_manager = MetalShaderManagerV35(verbose=verbose)
        if not self.shader_manager.initialize():
            if verbose:
                print("  GPU initialization failed, falling back to CPU")
            self.available = False
            return

        self.available = True

        # occupancy 阈值 (根据 M4 Pro 硬件)
        # 测试优化：320 阈值让 n=5000 使用 2D, n≥10000 使用 1D
        self.max_threadgroups = 350  # 从 512 调整到 350

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
        """预分配所有 GPU buffer."""
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

        # Parameter buffers (预分配，复用)
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

        # Batch parameter buffers (预分配最大可能大小)
        self.K_buf_max = device.newBufferWithBytes_length_options_(
            np.array([1024], dtype=np.uint32).tobytes(),  # 最大支持 K=1024
            4,
            Metal.MTLResourceStorageModeShared
        )

    def _write_vector(self, buffer: Any, data: np.ndarray):
        ptr = buffer.contents()
        byte_buffer = ptr.as_buffer(len(data) * 4)
        np.frombuffer(byte_buffer, dtype=np.float32)[:] = data.astype(np.float32)

    def _read_vector(self, buffer: Any, n: int) -> np.ndarray:
        ptr = buffer.contents()
        byte_buffer = ptr.as_buffer(n * 4)
        return np.frombuffer(byte_buffer, dtype=np.float32).copy()

    def _select_spmv_kernel(self) -> str:
        """智能选择 SpMV kernel."""
        # 如果平均列范围小，使用共享内存
        if self.avg_col_range < 128:
            return 'spmv_shared_smart'
        else:
            return 'spmv_shared_minimal'

    def spmv(self, x: np.ndarray) -> np.ndarray:
        """SpMV with Phase 3.5 optimizations."""
        if not self.available:
            return self.A @ x

        self._write_vector(self.x_buf, x)

        kernel_name = self._select_spmv_kernel()
        pipeline = self.shader_manager.get_pipeline(kernel_name)
        if pipeline is None:
            return self.A @ x

        command_buffer = self.shader_manager.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

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
        """
        Phase 3.5 occupancy-aware Batch SpMV.

        核心优化:
        1. occupancy 感知 - 动态选择 1D vs 2D
        2. 分组处理 - 大 batch 时分批
        """
        if not self.available:
            return np.array([self.A @ x for x in X])

        K, n = X.shape
        assert n == self.n

        # 计算 occupancy
        threadgroups_per_batch = (self.m + self.threadgroup_size - 1) // self.threadgroup_size
        total_threadgroups = threadgroups_per_batch * K

        # Phase 3.5: occupancy 感知策略
        if total_threadgroups > self.max_threadgroups:
            # occupancy 过高，使用 1D 策略 + 分组处理
            return self._batch_spmv_1d_grouped(X)
        else:
            # occupancy 正常，使用 2D 并行
            return self._batch_spmv_2d(X)

    def _batch_spmv_2d(self, X: np.ndarray) -> np.ndarray:
        """2D Batch SpMV (适合小 K)."""
        K, n = X.shape

        # 分配/扩展 batch buffer
        self._ensure_batch_buffer(K, n)

        # Write batch data
        X_flat = X.astype(np.float32).flatten()
        self._write_vector(self.x_batch_buf, X_flat)

        # Get pipeline
        pipeline = self.shader_manager.get_pipeline('batch_spmv_2d_opt')
        if pipeline is None:
            return np.array([self.A @ x for x in X])

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

        # Write K parameter (复用预分配 buffer)
        self._write_vector(self.K_buf_max, np.array([K], dtype=np.uint32))

        encoder.setBuffer_offset_atIndex_(self.m_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(self.n_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(self.K_buf_max, 0, 9)

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

    def _batch_spmv_1d_grouped(self, X: np.ndarray, group_size: int = 8) -> np.ndarray:
        """
        1D Grouped Batch SpMV (适合大 K).

        核心思想:
        1. 将 K 个 batch 分成多组
        2. 每组使用 1D kernel 顺序处理
        3. 避免 occupancy 过载
        """
        K, n = X.shape
        results = []

        for g in range(0, K, group_size):
            end = min(g + group_size, K)
            X_group = X[g:end]

            # 1D kernel: (row, 1, batch)
            self._ensure_batch_buffer(end - g, n)

            X_flat = X_group.astype(np.float32).flatten()
            self._write_vector(self.x_batch_buf, X_flat)

            pipeline = self.shader_manager.get_pipeline('batch_spmv_1d')
            if pipeline is None:
                results.append(np.array([self.A @ x for x in X_group]))
                continue

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

            self._write_vector(self.K_buf_max, np.array([end - g], dtype=np.uint32))
            encoder.setBuffer_offset_atIndex_(self.m_buf, 0, 7)
            encoder.setBuffer_offset_atIndex_(self.n_buf, 0, 8)
            encoder.setBuffer_offset_atIndex_(self.K_buf_max, 0, 9)

            # 3D dispatch: (threadgroups_x, 1, group_K)
            threads_per_threadgroup = min(
                self.threadgroup_size,
                pipeline.maxTotalThreadsPerThreadgroup(),
                self.m
            )
            threadgroups_x = (self.m + threads_per_threadgroup - 1) // threads_per_threadgroup

            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(threadgroups_x, 1, end - g),
                Metal.MTLSizeMake(threads_per_threadgroup, 1, 1)
            )

            encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            Y_flat = self._read_vector(self.y_batch_buf, (end - g) * self.m)
            results.append(Y_flat.reshape(end - g, self.m))

        return np.vstack(results)

    def _ensure_batch_buffer(self, K: int, n: int):
        """确保 batch buffer 足够大."""
        device = self.shader_manager.device
        batch_size = K * n * 4

        if not hasattr(self, 'x_batch_buf') or self.x_batch_buf.length() < batch_size:
            self.x_batch_buf = device.newBufferWithLength_options_(
                batch_size, Metal.MTLResourceStorageModeShared
            )
        if not hasattr(self, 'y_batch_buf') or self.y_batch_buf.length() < K * self.m * 4:
            self.y_batch_buf = device.newBufferWithLength_options_(
                K * self.m * 4, Metal.MTLResourceStorageModeShared
            )


def benchmark_phase35():
    """Phase 3.5 对比测试."""
    print("="*70)
    print("Phase 3.5: 深度优化对比测试")
    print("="*70)

    from gpu_optimized import OptimizedGPUSolver  # V1
    from gpu_phase2_optimized import OptimizedGPUSolverPhase2  # Phase 2
    from gpu_phase3_optimized import OptimizedGPUSolverPhase3  # Phase 3

    test_cases = [
        (5000, 2500, 0.02),
        (10000, 5000, 0.02),
        (20000, 10000, 0.02),
    ]

    K = 32  # Batch size

    for n, m, density in test_cases:
        print(f"\nn={n}, m={m}, density={density}, K={K}")
        print("-" * 60)

        A = sp.random(m, n, density=density, format='csr', random_state=42)
        X = np.random.randn(K, n).astype(np.float32)
        b = np.zeros(m)
        c = np.random.randn(n)
        lb = np.zeros(n)
        ub = np.ones(n)

        # CPU
        _ = np.array([A @ x for x in X])
        cpu_times = [(time.perf_counter() - start) * 1000
                     for start in [time.perf_counter() for _ in range(10)]
                     for _ in [np.array([A @ x for x in X])]]
        cpu_time = np.min(cpu_times)
        print(f"CPU Batch: {cpu_time:.4f} ms")

        # V1
        v1 = OptimizedGPUSolver(A, b, c, lb, ub, verbose=False)
        if v1.available:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = np.array([v1.spmv(X[k]) for k in range(K)])
                times.append((time.perf_counter() - start) * 1000)
            v1_time = np.min(times)
            print(f"V1 (Sequential): {v1_time:.4f} ms")

        # Phase 2
        v2 = OptimizedGPUSolverPhase2(A, b, c, lb, ub, verbose=False,
                                       threadgroup_size=256, use_shared_memory=False)
        if v2.available:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = np.array([v2.spmv(X[k]) for k in range(K)])
                times.append((time.perf_counter() - start) * 1000)
            v2_time = np.min(times)
            print(f"Phase 2 (Sequential): {v2_time:.4f} ms")

        # Phase 3 (旧 2D)
        v3 = OptimizedGPUSolverPhase3(A, b, c, lb, ub, verbose=False,
                                       threadgroup_size=256,
                                       use_async=False, use_batch_submit=True)
        if v3.available:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = v3.batch_spmv(X)
                times.append((time.perf_counter() - start) * 1000)
            v3_time = np.min(times)
            print(f"Phase 3 (2D): {v3_time:.4f} ms")

        # Phase 3.5 (新 occupancy-aware)
        v35 = OptimizedGPUSolverPhase35(A, b, c, lb, ub, verbose=False,
                                         threadgroup_size=256)
        if v35.available:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = v35.batch_spmv(X)
                times.append((time.perf_counter() - start) * 1000)
            v35_time = np.min(times)
            print(f"Phase 3.5 (occupancy-aware): {v35_time:.4f} ms")

            # GPU vs CPU
            gpu_speedup = cpu_time / v35_time
            print(f"GPU vs CPU: {'GPU' if gpu_speedup > 1 else 'CPU'} {max(gpu_speedup, 1/gpu_speedup):.2f}x")

            # 共享内存效果
            if v35.avg_col_range < 128:
                print(f"  智能共享内存启用 (avg_col_range={v35.avg_col_range:.1f})")
            else:
                print(f"  智能共享内存跳过 (avg_col_range={v35.avg_col_range:.1f})")

        # 分析
        print(f"\n分析:")
        print(f"  avg_nnz/row: {np.mean(np.diff(A.indptr)):.1f}")
        print(f"  avg_col_range: {v35.avg_col_range if v35.available else 'N/A'}")
        threadgroups_needed = ((m + 255) // 256) * K
        print(f"  threadgroups needed: {threadgroups_needed}")
        print(f"  max_threadgroups: {v35.max_threadgroups if v35.available else 'N/A'}")

    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark_phase35()
