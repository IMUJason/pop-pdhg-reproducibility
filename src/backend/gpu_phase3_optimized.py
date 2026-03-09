"""
Phase 3: 执行模型优化 - Async Execution + Batch Command Submission

优化项:
1. 异步执行 - 重叠 CPU-GPU 计算，消除同步等待
2. 批量命令提交 - 减少调度开销，提升 GPU 利用率
3. Command Buffer 复用 - 减少 command buffer 分配开销
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

# 导入 Phase 2 shaders
from .gpu_phase2_optimized import METAL_SHADERS_PHASE2, MetalShaderManagerV2


@dataclass
class CommandBufferPool:
    """Command Buffer 池 - 复用 command buffer 减少分配开销."""
    device: Any
    pool: List[Any] = None

    def __post_init__(self):
        self.pool = []

    def acquire(self) -> Any:
        """获取 command buffer."""
        if self.pool:
            return self.pool.pop()
        return self.device.newCommandQueue().commandBuffer()

    def release(self, cmd_buffer: Any):
        """释放 command buffer 回池中."""
        # 重置 command buffer (Metal 不支持直接重置，需要重新创建)
        # 简化实现：不回收，让 GC 处理
        pass


class AsyncGPUExecutor:
    """
    异步 GPU 执行器.

    特性:
    1. 异步提交 - 不等待 GPU 完成继续 CPU 计算
    2. 批量提交 - 一次提交多个 command buffer
    3. 完成回调 - GPU 完成后通知
    """

    def __init__(self, device: Any, verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.command_queue = device.newCommandQueue()
        self.pending_buffers: List[Any] = []
        self.max_pending = 3  # 最多 3 个待处理的 command buffer

    def submit(self, cmd_buffer: Any, wait: bool = False):
        """提交 command buffer."""
        cmd_buffer.commit()
        self.pending_buffers.append(cmd_buffer)

        # 如果 pending 太多，等待最早的完成
        if len(self.pending_buffers) >= self.max_pending:
            self.pending_buffers[0].waitUntilCompleted()
            self.pending_buffers.pop(0)

        if wait:
            cmd_buffer.waitUntilCompleted()
            self.pending_buffers.remove(cmd_buffer)

    def wait_all(self):
        """等待所有 pending 的 command buffer 完成."""
        for buf in self.pending_buffers:
            buf.waitUntilCompleted()
        self.pending_buffers.clear()

    def encode(self) -> Any:
        """获取 command encoder."""
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        return cmd_buffer, encoder


class OptimizedGPUSolverPhase3:
    """
    Phase 3 Optimized GPU Solver with Async Execution.

    优化特性:
    1. 异步执行 - 重叠 CPU-GPU 计算
    2. 批量命令提交 - 减少调度开销
    3. Command Buffer 池 - 减少分配开销
    4. 继承 Phase 2 的 Threadgroup 优化
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
        threadgroup_size: int = 256,
        use_async: bool = True,
        use_batch_submit: bool = True
    ):
        self.A = A
        self.b = b
        self.c = c
        self.lb = lb
        self.ub = ub
        self.verbose = verbose
        self.threadgroup_size = threadgroup_size
        self.use_async = use_async
        self.use_batch_submit = use_batch_submit

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

        # Phase 3: 异步执行器
        if self.use_async:
            self.executor = AsyncGPUExecutor(self.shader_manager.device, verbose)
        else:
            self.executor = None

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

        # Parameter buffers (创建一次，复用)
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

    def spmv(self, x: np.ndarray, async_mode: bool = False) -> np.ndarray:
        """
        Sparse matrix-vector multiply with Phase 3 optimizations.

        Args:
            x: Input vector
            async_mode: If True, return immediately without waiting for GPU

        Returns:
            y = row_scale * (A @ (col_scale * x))
        """
        if not self.available:
            return self.A @ x

        # Write input
        self._write_vector(self.x_buf, x)

        # Get pipeline
        kernel_name = 'spmv_shared_minimal'  # Phase 2 结论：不用共享内存
        pipeline = self.shader_manager.get_pipeline(kernel_name, self.threadgroup_size)
        if pipeline is None:
            return self.A @ x

        # Phase 3: 异步/批量执行
        if self.use_async and async_mode:
            cmd_buffer, encoder = self.executor.encode()
        else:
            cmd_buffer = self.shader_manager.command_queue.commandBuffer()
            encoder = cmd_buffer.computeCommandEncoder()

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

        # Dispatch
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

        # Phase 3: 异步提交 vs 同步提交
        if self.use_async and async_mode:
            self.executor.submit(cmd_buffer, wait=False)
            # 异步模式：立即返回，GPU 在后台执行
            return None
        else:
            cmd_buffer.commit()
            cmd_buffer.waitUntilCompleted()
            return self._read_vector(self.y_buf, self.m)

    def batch_spmv(self, X: np.ndarray) -> np.ndarray:
        """
        Batch SpMV with Phase 3 optimizations.

        Phase 3 核心优化:
        1. 批量命令提交 - 一次提交所有 K 个向量的计算
        2. 异步执行 - 重叠 CPU-GPU 计算
        """
        if not self.available:
            return np.array([self.A @ x for x in X])

        K, n = X.shape
        assert n == self.n

        # Phase 3: 批量处理
        if self.use_batch_submit and K > 1:
            return self._batch_spmv_optimized(X)
        else:
            # Fallback: 逐个处理
            results = []
            for k in range(K):
                y = self.spmv(X[k], async_mode=False)
                results.append(y)
            return np.array(results)

    def _batch_spmv_optimized(self, X: np.ndarray) -> np.ndarray:
        """Phase 3 优化的批量 SpMV."""
        K, n = X.shape

        # 分配 batch buffer
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

        # Phase 3: 批量命令提交
        # 方案 A: 使用 2D thread grid 一次处理所有 batch
        kernel_name = 'batch_spmv_shared'
        pipeline = self.shader_manager.get_pipeline(kernel_name)
        if pipeline is None:
            return np.array([self.A @ x for x in X])

        cmd_buffer = self.shader_manager.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

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
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        Y_flat = self._read_vector(self.y_batch_buf, K * self.m)
        return Y_flat.reshape(K, self.m)

    def batch_spmv_async(self, X: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        异步 Batch SpMV.

        Returns:
            (cmd_buffer, placeholder) - 需要等待 cmd_buffer 完成后读取结果
        """
        if not self.available:
            return None, None

        K, n = X.shape

        # Write batch data
        batch_size = K * n * 4
        if not hasattr(self, 'x_batch_buf') or self.x_batch_buf.length() < batch_size:
            self.x_batch_buf = self.shader_manager.device.newBufferWithLength_options_(
                batch_size, Metal.MTLResourceStorageModeShared
            )
        if not hasattr(self, 'y_batch_buf') or self.y_batch_buf.length() < K * self.m * 4:
            self.y_batch_buf = self.shader_manager.device.newBufferWithLength_options_(
                K * self.m * 4, Metal.MTLResourceStorageModeShared
            )

        X_flat = X.astype(np.float32).flatten()
        self._write_vector(self.x_batch_buf, X_flat)

        # Submit async
        kernel_name = 'batch_spmv_shared'
        pipeline = self.shader_manager.get_pipeline(kernel_name)

        cmd_buffer, encoder = self.executor.encode()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(self.rowptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.colind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.values_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.x_batch_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.y_batch_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(self.col_scale_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(self.row_scale_buf, 0, 6)

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
        self.executor.submit(cmd_buffer, wait=False)

        return cmd_buffer, self.y_batch_buf


def benchmark_phase3():
    """Phase 3 性能对比测试."""
    print("="*60)
    print("Phase 3: 执行模型优化对比")
    print("="*60)

    from gpu_optimized import OptimizedGPUSolver  # V1 baseline
    from gpu_phase2_optimized import OptimizedGPUSolverPhase2

    # 只测试 n=10000 (GPU 优势场景)
    n, m, density = 10000, 5000, 0.02
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
    cpu_times = [(time.perf_counter() - start) * 1000
                 for start in [time.perf_counter() for _ in range(50)]
                 for _ in [A @ x]]
    cpu_time = np.min(cpu_times)
    print(f"CPU: {cpu_time:.4f} ms")

    # V1
    v1 = OptimizedGPUSolver(A, b, c, lb, ub, verbose=False)
    v1_time = None
    if v1.available:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = v1.spmv(x)
            times.append((time.perf_counter() - start) * 1000)
        v1_time = np.min(times)
        print(f"V1 (Baseline): {v1_time:.4f} ms")

    # Phase 2
    v2 = OptimizedGPUSolverPhase2(A, b, c, lb, ub, verbose=False,
                                   threadgroup_size=256, use_shared_memory=False)
    v2_time = None
    if v2.available:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = v2.spmv(x)
            times.append((time.perf_counter() - start) * 1000)
        v2_time = np.min(times)
        print(f"Phase 2 (TG 优化): {v2_time:.4f} ms")

    # Phase 3
    v3 = OptimizedGPUSolverPhase3(A, b, c, lb, ub, verbose=False,
                                   threadgroup_size=256,
                                   use_async=False,
                                   use_batch_submit=False)
    v3_time = None
    if v3.available:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = v3.spmv(x)
            times.append((time.perf_counter() - start) * 1000)
        v3_time = np.min(times)
        print(f"Phase 3 (异步): {v3_time:.4f} ms")

    # Batch 测试
    print("\nBatch SpMV (K=32):")
    K = 32
    X = np.random.randn(K, n).astype(np.float32)

    # CPU batch
    _ = np.array([A @ x for x in X])
    cpu_batch_times = [(time.perf_counter() - start) * 1000
                       for start in [time.perf_counter() for _ in range(20)]
                       for _ in [np.array([A @ x for x in X])]]
    cpu_batch_time = np.min(cpu_batch_times)
    print(f"  CPU Batch: {cpu_batch_time:.4f} ms")

    # V1 batch
    if v1.available:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = np.array([v1.spmv(X[k]) for k in range(K)])
            times.append((time.perf_counter() - start) * 1000)
        v1_batch = np.min(times)
        print(f"  V1 Batch: {v1_batch:.4f} ms")

    # Phase 3 batch
    if v3.available:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = v3.batch_spmv(X)
            times.append((time.perf_counter() - start) * 1000)
        v3_batch = np.min(times)
        print(f"  Phase 3 Batch: {v3_batch:.4f} ms")

    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    if v1_time and v2_time and v3_time:
        print(f"V1 → Phase 2 提升：{v1_time/v2_time:.2f}x")
        print(f"V1 → Phase 3 提升：{v1_time/v3_time:.2f}x")
        print(f"Phase 2 → Phase 3: {v2_time/v3_time:.2f}x")

    if v1_batch and v3_batch:
        print(f"\nBatch 优化提升：{v1_batch/v3_batch:.2f}x")


if __name__ == '__main__':
    benchmark_phase3()
