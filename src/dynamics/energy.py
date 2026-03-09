"""
Energy Functions for HMC-PDHG.

This module defines energy functions (Hamiltonians) for LP/MIP problems
in the HMC framework. The energy function combines:
1. Objective function contribution
2. Constraint violation penalty
3. Optional regularization terms

For LP: min c^T x s.t. Ax <= b, lb <= x <= ub

The potential energy is:
    U(x) = c^T x + penalty * ||max(Ax - b, 0)||^2 + I_{[lb,ub]}(x)

where I_{[lb,ub]}(x) is the indicator function (0 if x in bounds, ∞ otherwise).

The kinetic energy is:
    K(p) = ||p||^2 / (2m)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
from scipy import sparse


@dataclass
class EnergyConfig:
    """Configuration for energy function.

    Attributes:
        penalty_init: Initial constraint violation penalty
        penalty_growth: Penalty growth factor per HMC step
        penalty_max: Maximum penalty value
        barrier_coeff: Barrier function coefficient for bounds
        use_barrier: Whether to use log barrier (vs hard projection)
    """

    penalty_init: float = 100.0
    penalty_growth: float = 1.1
    penalty_max: float = 1e6
    barrier_coeff: float = 0.01
    use_barrier: bool = False


class LPEnergyFunction:
    """Energy function for LP problems in HMC framework.

    Provides potential energy, gradient, and related utilities
    for sampling from LP solution distributions.

    The energy is designed to:
    - Be smooth (for gradient-based sampling)
    - Heavily penalize constraint violations
    - Guide toward optimal objective value
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        config: Optional[EnergyConfig] = None,
    ):
        """Initialize energy function.

        Args:
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            lb, ub: Variable bounds
            config: Energy configuration
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.config = config or EnergyConfig()

        self.m, self.n = A.shape

        # Current penalty (can be updated during solving)
        self.penalty = self.config.penalty_init

    def potential_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute potential energy for a batch of solutions.

        U(x) = c^T x + penalty * ||max(Ax - b, 0)||^2

        Args:
            x: Solutions, shape (K, n) or (n,)

        Returns:
            Energy values, shape (K,) or scalar
        """
        single_dim = x.ndim == 1
        if single_dim:
            x = x.reshape(1, -1)

        # Objective contribution
        obj = x @ self.c  # (K,)

        # Constraint violation
        Ax = (self.A @ x.T).T  # (K, m)
        violation = np.maximum(Ax - self.b, 0)  # (K, m)
        penalty_term = self.penalty * np.sum(violation**2, axis=1)  # (K,)

        # Total energy
        energy = obj + penalty_term

        # Add barrier for bounds if using barrier method
        if self.config.use_barrier:
            # Log barrier: -coeff * sum(log(x - lb) + log(ub - x))
            barrier = 0.0
            for j in range(self.n):
                # Only add barrier if strictly inside bounds
                dist_lb = x[:, j] - self.lb[j]
                dist_ub = self.ub[j] - x[:, j]

                # Penalize out-of-bounds heavily
                barrier += np.where(
                    (dist_lb > 1e-10) & (dist_ub > 1e-10),
                    -self.config.barrier_coeff * (np.log(dist_lb) + np.log(dist_ub)),
                    1e10,  # Large penalty for out-of-bounds
                )
            energy = energy + barrier

        if single_dim:
            return energy[0]
        return energy

    def grad_potential(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of potential energy.

        ∇U = c + 2 * penalty * A^T max(Ax - b, 0)

        Args:
            x: Solutions, shape (K, n) or (n,)

        Returns:
            Gradient, shape (K, n) or (n,)
        """
        single_dim = x.ndim == 1
        if single_dim:
            x = x.reshape(1, -1)

        K = x.shape[0]

        # Objective gradient: constant c
        grad = np.broadcast_to(self.c, (K, self.n)).copy()  # (K, n)

        # Constraint violation gradient
        Ax = (self.A @ x.T).T  # (K, m)
        violation = np.maximum(Ax - self.b, 0)  # (K, m)

        # A^T @ violation: (K, n) = (K, m) @ (m, n)^T
        # For sparse matrix, do it row by row
        grad_violation = np.zeros((K, self.n))
        for k in range(K):
            grad_violation[k] = self.A.T @ violation[k]

        grad = grad + 2 * self.penalty * grad_violation

        # Add barrier gradient if using barrier method
        if self.config.use_barrier:
            for j in range(self.n):
                dist_lb = x[:, j] - self.lb[j]
                dist_ub = self.ub[j] - x[:, j]

                # d/dx log(x - lb) = 1/(x - lb)
                # d/dx log(ub - x) = -1/(ub - x)
                barrier_grad = np.where(
                    (dist_lb > 1e-10) & (dist_ub > 1e-10),
                    -self.config.barrier_coeff * (1.0 / dist_lb - 1.0 / dist_ub),
                    0.0,  # No gradient if out of bounds (handled by projection)
                )
                grad[:, j] += barrier_grad

        if single_dim:
            return grad[0]
        return grad

    def kinetic_energy(self, p: np.ndarray, mass: float = 1.0) -> np.ndarray:
        """Compute kinetic energy.

        K(p) = ||p||^2 / (2m)

        Args:
            p: Momentum, shape (K, n) or (n,)
            mass: Mass parameter

        Returns:
            Kinetic energy, shape (K,) or scalar
        """
        return np.sum(p**2, axis=-1) / (2 * mass)

    def hamiltonian(
        self, x: np.ndarray, p: np.ndarray, mass: float = 1.0
    ) -> np.ndarray:
        """Compute total Hamiltonian.

        H(x, p) = U(x) + K(p)

        Args:
            x: Position, shape (K, n) or (n,)
            p: Momentum, shape (K, n) or (n,)
            mass: Mass parameter

        Returns:
            Hamiltonian values, shape (K,) or scalar
        """
        return self.potential_energy(x) + self.kinetic_energy(p, mass)

    def update_penalty(self, factor: Optional[float] = None) -> None:
        """Update the constraint penalty coefficient.

        Args:
            factor: Growth factor. Uses config.penalty_growth if None.
        """
        if factor is None:
            factor = self.config.penalty_growth

        self.penalty = min(self.penalty * factor, self.config.penalty_max)

    def project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project solution to variable bounds.

        Args:
            x: Solution, shape (K, n) or (n,)

        Returns:
            Projected solution
        """
        return np.clip(x, self.lb, self.ub)


class MIPHardEnergyFunction(LPEnergyFunction):
    """Energy function for MIP problems with integer variables.

    Extends LP energy with penalty for integrality violation.

    Additional energy term:
        U_int(x) = penalty_int * sum_j (x_j - round(x_j))^2 for integer j
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: list[int],
        config: Optional[EnergyConfig] = None,
        int_penalty: float = 10.0,
    ):
        """Initialize MIP energy function.

        Args:
            A, b, c, lb, ub: Problem data
            integer_vars: Indices of integer variables
            config: Energy configuration
            int_penalty: Integrality violation penalty
        """
        super().__init__(A, b, c, lb, ub, config)
        self.integer_vars = integer_vars
        self.int_penalty = int_penalty

    def potential_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute potential energy with integrality penalty."""
        energy = super().potential_energy(x)

        single_dim = x.ndim == 1
        if single_dim:
            x = x.reshape(1, -1)
            energy = energy.reshape(1)

        # Add integrality penalty
        if self.integer_vars:
            x_int = x[:, self.integer_vars]
            int_violation = np.sum(
                (x_int - np.round(x_int)) ** 2, axis=1
            )
            energy = energy + self.int_penalty * int_violation

        if single_dim:
            return energy[0]
        return energy

    def grad_potential(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient with integrality term."""
        grad = super().grad_potential(x)

        single_dim = x.ndim == 1
        if single_dim:
            x = x.reshape(1, -1)
            grad = grad.reshape(1, -1)

        # Add integrality gradient
        if self.integer_vars:
            x_int = x[:, self.integer_vars]
            int_grad = 2 * self.int_penalty * (x_int - np.round(x_int))
            grad[:, self.integer_vars] += int_grad

        if single_dim:
            return grad[0]
        return grad


def create_energy_function(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_vars: Optional[list[int]] = None,
    config: Optional[EnergyConfig] = None,
) -> LPEnergyFunction:
    """Factory function to create appropriate energy function.

    Args:
        A, b, c, lb, ub: Problem data
        integer_vars: Indices of integer variables (None for pure LP)
        config: Energy configuration

    Returns:
        Energy function instance
    """
    if integer_vars:
        return MIPHardEnergyFunction(
            A, b, c, lb, ub, integer_vars, config
        )
    else:
        return LPEnergyFunction(A, b, c, lb, ub, config)


if __name__ == "__main__":
    # Test energy function
    print("Testing Energy Function")
    print("=" * 50)

    from scipy import sparse

    # Simple LP: min -x - y s.t. x + y <= 1
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])

    energy = LPEnergyFunction(A, b, c, lb, ub)

    # Test at different points
    test_points = [
        np.array([0.5, 0.5]),   # Optimal
        np.array([1.0, 1.0]),   # Infeasible
        np.array([0.0, 0.0]),   # Feasible but not optimal
    ]

    print("\nSingle point tests:")
    for x in test_points:
        U = energy.potential_energy(x)
        grad = energy.grad_potential(x)
        Ax = A @ x
        feas = "feasible" if Ax[0] <= b[0] + 1e-6 else "infeasible"
        print(f"  x = {x}: U = {U:.4f} ({feas})")
        print(f"       ∇U = {grad}")

    # Test batch
    print("\nBatch test:")
    x_batch = np.array(test_points)
    U_batch = energy.potential_energy(x_batch)
    print(f"  Energies: {U_batch}")
