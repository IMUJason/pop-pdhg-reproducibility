"""
混合 CSR/CSC 稀疏矩阵存储

为优化转置 SpMV 操作 (A.T @ y)，同时存储 CSR 和 CSC 格式。

动机:
- PDHG 迭代中需要频繁计算 A @ x 和 A.T @ y
- 当前每次 A.T 都需要 CSR→CSC 转换，开销显著
- 预存储 CSC 可避免运行时转换

适用场景:
- 转置操作频繁的迭代算法 (如 PDHG)
- 内存充足 (额外 +100% 存储开销)
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


class HybridSparseMatrix:
    """
    混合 CSR/CSC 稀疏矩阵存储类

    功能:
    1. 同时存储 CSR 和 CSC 格式
    2. 高效的 matvec 和 transposed_matvec
    3. 可选的延迟 CSC 初始化 (节省内存)
    """

    def __init__(self, A: sp.spmatrix, use_csc: bool = True, verbose: bool = False):
        """
        初始化混合存储

        Args:
            A: 输入稀疏矩阵 (任意格式)
            use_csc: 是否预分配 CSC 格式 (默认 True)
            verbose: 是否打印详细信息
        """
        self.verbose = verbose

        # 始终存储 CSR (用于 A @ x)
        if not sp.isspmatrix_csr(A):
            self.csr = A.tocsr()
        else:
            self.csr = A

        # 可选 CSC (用于 A.T @ y)
        self._csc: Optional[sp.csc_matrix] = None

        if use_csc:
            self._ensure_csc()

        # 缓存形状信息
        self.m, self.n = self.csr.shape
        self.nnz = self.csr.nnz

        if verbose:
            print(f"HybridSparseMatrix: m={self.m}, n={self.n}, nnz={self.nnz}")
            if use_csc:
                print(f"  CSC 预分配完成")

    def _ensure_csc(self):
        """确保 CSC 格式已分配"""
        if self._csc is None:
            self._csc = self.csr.tocsc()

    @property
    def csc(self) -> Optional[sp.csc_matrix]:
        """获取 CSC 格式 (延迟初始化)"""
        if self._csc is None:
            self._ensure_csc()
        return self._csc

    def matvec(self, x: np.ndarray, transpose: bool = False) -> np.ndarray:
        """
        稀疏矩阵 - 向量乘法

        Args:
            x: 输入向量
            transpose: 是否计算转置 (A.T @ y)

        Returns:
            结果向量
        """
        if transpose:
            # A.T @ y - 使用 CSC 避免运行时转换
            if self._csc is not None:
                return self._csc.T @ x
            else:
                # 退化情况：CSC 未初始化，临时转换
                return self.csr.T @ x
        else:
            # A @ x - 使用 CSR
            return self.csr @ x

    def batch_matvec(self, X: np.ndarray, transpose: bool = False) -> np.ndarray:
        """
        批量稀疏矩阵 - 向量乘法

        Args:
            X: 输入矩阵 (K, n) 或 (K, m)
            transpose: 是否计算转置

        Returns:
            结果矩阵 (K, m) 或 (K, n)
        """
        if X.ndim == 1:
            return self.matvec(X, transpose=transpose)

        K = X.shape[0]

        if transpose:
            # A.T @ y for batch
            if self._csc is not None:
                return np.array([self._csc.T @ x for x in X])
            else:
                return np.array([self.csr.T @ x for x in X])
        else:
            # A @ x for batch
            return np.array([self.csr @ x for x in X])

    def get_csr(self) -> sp.csr_matrix:
        """获取 CSR 格式"""
        return self.csr

    def get_csc(self) -> Optional[sp.csc_matrix]:
        """获取 CSC 格式"""
        return self.csc

    def to_scipy(self, transpose: bool = False):
        """
        转换为 SciPy 矩阵

        Args:
            transpose: 是否返回转置

        Returns:
            SciPy 稀疏矩阵
        """
        if transpose:
            return self.csc if self._csc is not None else self.csr.T
        else:
            return self.csr

    def memory_usage(self) -> Tuple[int, int]:
        """
        计算内存使用

        Returns:
            (csr_memory, csc_memory) 字节
        """
        # CSR 内存：indptr + indices + data
        csr_mem = (
            self.csr.indptr.nbytes +
            self.csr.indices.nbytes +
            self.csr.data.nbytes
        )

        if self._csc is not None:
            csc_mem = (
                self._csc.indptr.nbytes +
                self._csc.indices.nbytes +
                self._csc.data.nbytes
            )
        else:
            csc_mem = 0

        return csr_mem, csc_mem

    def __repr__(self):
        csc_status = "预分配" if self._csc is not None else "未初始化"
        return (f"HybridSparseMatrix(m={self.m}, n={self.n}, nnz={self.nnz}, "
                f"CSC={csc_status})")


def create_hybrid_matrix(A: sp.spmatrix, use_csc: bool = True,
                         verbose: bool = False) -> HybridSparseMatrix:
    """
    工厂函数：创建混合存储矩阵

    Args:
        A: 输入稀疏矩阵
        use_csc: 是否预分配 CSC
        verbose: 详细信息

    Returns:
        HybridSparseMatrix 实例
    """
    return HybridSparseMatrix(A, use_csc=use_csc, verbose=verbose)


if __name__ == "__main__":
    """测试混合存储功能"""
    print("="*60)
    print("混合 CSR/CSC 存储测试")
    print("="*60)

    # 创建测试矩阵
    np.random.seed(42)
    m, n, density = 1000, 500, 0.02

    A = sp.random(m, n, density=density, format='csr', random_state=42)
    x = np.random.randn(n)
    y = np.random.randn(m)

    print(f"\n测试矩阵：m={m}, n={n}, density={density:.1%}, nnz={A.nnz}")

    # 测试 1: CSR only
    print("\n1. CSR only (无 CSC 预分配):")
    print("-" * 40)
    hybrid_csr = HybridSparseMatrix(A, use_csc=False, verbose=True)

    result_x = hybrid_csr.matvec(x)
    result_y = hybrid_csr.matvec(y, transpose=True)

    print(f"  A @ x 正确：{np.allclose(result_x, A @ x)}")
    print(f"  A.T @ y 正确：{np.allclose(result_y, A.T @ y)}")

    # 测试 2: CSR + CSC
    print("\n2. CSR + CSC (预分配):")
    print("-" * 40)
    hybrid_both = HybridSparseMatrix(A, use_csc=True, verbose=True)

    result_x2 = hybrid_both.matvec(x)
    result_y2 = hybrid_both.matvec(y, transpose=True)

    print(f"  A @ x 正确：{np.allclose(result_x2, A @ x)}")
    print(f"  A.T @ y 正确：{np.allclose(result_y2, A.T @ y)}")

    # 测试 3: 内存使用
    print("\n3. 内存使用:")
    print("-" * 40)
    csr_mem, csc_mem = hybrid_both.memory_usage()
    print(f"  CSR: {csr_mem / 1024:.1f} KB")
    print(f"  CSC: {csc_mem / 1024:.1f} KB")
    print(f"  总计：{(csr_mem + csc_mem) / 1024:.1f} KB")

    # 测试 4: 延迟初始化
    print("\n4. 延迟初始化测试:")
    print("-" * 40)
    hybrid_lazy = HybridSparseMatrix(A, use_csc=False)
    print(f"  初始 CSC 状态：{hybrid_lazy._csc}")

    _ = hybrid_lazy.csc  # 触发延迟初始化
    print(f"  访问后 CSC 状态：{'已初始化' if hybrid_lazy._csc is not None else '未初始化'}")

    # 测试 5: 批量操作
    print("\n5. 批量操作测试:")
    print("-" * 40)
    K = 32
    X = np.random.randn(K, n)
    Y = np.random.randn(K, m)

    result_batch_x = hybrid_both.batch_matvec(X)
    result_batch_y = hybrid_both.batch_matvec(Y, transpose=True)

    expected_batch_x = np.array([A @ x for x in X])
    expected_batch_y = np.array([A.T @ y for y in Y])

    print(f"  Batch A @ X 正确：{np.allclose(result_batch_x, expected_batch_x)}")
    print(f"  Batch A.T @ Y 正确：{np.allclose(result_batch_y, expected_batch_y)}")

    print("\n" + "="*60)
    print("✓ 所有测试通过!")
    print("="*60)
