"""
PDHG Core Operations - Metal GPU Implementation with CPU Fallback
"""

import numpy as np
from scipy import sparse

# Try to import Metal
try:
    from ..device import MetalDevice, UnifiedMemoryBuffer
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class PDHGKernels:
    """GPU/CPU kernels for PDHG primal-dual iterations."""

    def __init__(self, use_gpu=True):
        """Initialize PDHG kernels.

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and METAL_AVAILABLE
        self.device = None

        if self.use_gpu:
            try:
                self.device = MetalDevice.default()
                if not self.device.is_available:
                    self.use_gpu = False
            except:
                self.use_gpu = False

        if not self.use_gpu:
            self._init_cpu_fallback()

    def _init_cpu_fallback(self):
        """Initialize CPU fallback (numpy)."""
        self.device = None
        print("PDHG Kernels: Using CPU (numpy) fallback")

    def primal_update(self, x, y, A, c, eta, lb, ub):
        """Primal update: x = proj(x - eta * (c + A^T @ y))

        Supports batch operation for multiple population members.

        Args:
            x: Current primal variables (K, n) or (n,)
            y: Current dual variables (K, m) or (m,)
            A: Constraint matrix (m, n) sparse
            c: Objective coefficients (n,)
            eta: Primal step size
            lb, ub: Variable bounds

        Returns:
            Updated x
        """
        if self.use_gpu:
            return self._primal_update_gpu(x, y, A, c, eta, lb, ub)
        else:
            return self._primal_update_cpu(x, y, A, c, eta, lb, ub)

    def _primal_update_cpu(self, x, y, A, c, eta, lb, ub):
        """CPU implementation using numpy."""
        # Handle both single vector and batch
        if x.ndim == 1:
            # Single vector
            ATy = A.T @ y
            grad = c + ATy
            x_new = x - eta * grad
            return np.clip(x_new, lb, ub)
        else:
            # Batch: (K, n)
            K = x.shape[0]
            x_new = np.zeros_like(x)

            # Use efficient sparse matrix operations
            AT = A.T.tocsr()
            for k in range(K):
                ATy = AT @ y[k]
                grad = c + ATy
                x_new[k] = np.clip(x[k] - eta * grad, lb, ub)

            return x_new

    def _primal_update_gpu(self, x, y, A, c, eta, lb, ub):
        """GPU implementation using Metal (placeholder for now).

        For Phase 1, falls back to CPU with unified memory.
        """
        # TODO: Implement actual Metal kernel
        # For now, use CPU implementation but ensure unified memory
        return self._primal_update_cpu(x, y, A, c, eta, lb, ub)

    def dual_update(self, y, x_bar, A, b, tau):
        """Dual update: y = proj_{>=0}(y + tau * (A @ x_bar - b))

        Args:
            y: Current dual variables (K, m) or (m,)
            x_bar: Extrapolated primal (K, n) or (n,)
            A: Constraint matrix (m, n)
            b: RHS vector (m,)
            tau: Dual step size

        Returns:
            Updated y
        """
        if self.use_gpu:
            return self._dual_update_gpu(y, x_bar, A, b, tau)
        else:
            return self._dual_update_cpu(y, x_bar, A, b, tau)

    def _dual_update_cpu(self, y, x_bar, A, b, tau):
        """CPU implementation."""
        if y.ndim == 1:
            # Single vector
            Ax = A @ x_bar
            y_new = y + tau * (Ax - b)
            return np.maximum(y_new, 0)  # Project to y >= 0
        else:
            # Batch
            K = y.shape[0]
            y_new = np.zeros_like(y)

            for k in range(K):
                Ax = A @ x_bar[k]
                y_new[k] = np.maximum(y[k] + tau * (Ax - b), 0)

            return y_new

    def _dual_update_gpu(self, y, x_bar, A, b, tau):
        """GPU implementation (placeholder)."""
        return self._dual_update_cpu(y, x_bar, A, b, tau)

    def compute_objective(self, x, c):
        """Compute objective: c^T @ x

        Args:
            x: Primal variables (K, n) or (n,)
            c: Objective coefficients (n,)

        Returns:
            Objective value(s)
        """
        if x.ndim == 1:
            return c @ x
        else:
            # Batch: return (K,) vector
            return x @ c

    def compute_constraint_violation(self, x, A, b):
        """Compute primal feasibility violation.

        Args:
            x: Primal variables (K, n) or (n,)
            A: Constraint matrix
            b: RHS

        Returns:
            Maximum violation (scalar or vector of K)
        """
        if x.ndim == 1:
            Ax = A @ x
            violation = np.maximum(Ax - b, 0)
            return np.max(violation)
        else:
            # Batch
            K = x.shape[0]
            violations = np.zeros(K)
            for k in range(K):
                Ax = A @ x[k]
                violation = np.maximum(Ax - b, 0)
                violations[k] = np.max(violation)
            return violations


class BatchPDHGStep:
    """Complete batch PDHG iteration."""

    def __init__(self, kernels=None):
        self.kernels = kernels or PDHGKernels()

    def step(self, x, y, A, b, c, eta, tau, lb, ub):
        """One complete PDHG step for all population members.

        Args:
            x: (K, n) population primal variables
            y: (K, m) population dual variables
            A, b, c: Problem data
            eta, tau: Step sizes
            lb, ub: Bounds

        Returns:
            x_new, y_new: Updated variables
        """
        # Primal update
        x_new = self.kernels.primal_update(x, y, A, c, eta, lb, ub)

        # Extrapolation
        x_bar = 2 * x_new - x

        # Dual update
        y_new = self.kernels.dual_update(y, x_bar, A, b, tau)

        return x_new, y_new


if __name__ == "__main__":
    # Test kernels
    print("Testing PDHG Kernels...")

    # Create test problem
    n, m = 100, 50
    K = 4  # Population size

    A = sparse.random(m, n, density=0.1, format='csr')
    b = np.random.rand(m).astype(np.float32)
    c = np.random.randn(n).astype(np.float32)
    lb = np.zeros(n, dtype=np.float32)
    ub = np.ones(n, dtype=np.float32) * 10

    eta = tau = 0.01

    # Initialize population
    x = np.random.rand(K, n).astype(np.float32)
    y = np.random.rand(K, m).astype(np.float32)

    # Test kernels
    kernels = PDHGKernels(use_gpu=False)

    x_new, y_new = BatchPDHGStep(kernels).step(x, y, A, b, c, eta, tau, lb, ub)

    print(f"Input x shape: {x.shape}, y shape: {y.shape}")
    print(f"Output x shape: {x_new.shape}, y shape: {y_new.shape}")
    print(f"Objective values: {kernels.compute_objective(x_new, c)}")
    print("Test passed!")
