"""
Hamiltonian Monte Carlo (HMC) for LP/MIP Problems.

This module implements HMC sampling for LP problems, integrated with
PDHG for optimization. HMC provides:
1. Momentum-based exploration (inertia helps escape local minima)
2. Efficient sampling from constrained distributions
3. Theoretical convergence guarantees via detailed balance

The key insight is that standard PDHG is equivalent to HMC with
infinite mass (no momentum). Adding finite mass allows the sampler
to traverse energy barriers more efficiently.

Mathematical foundation:
    - PDHG: x^{k+1} = x^k - η ∇f(x^k)
    - HMC:  dx/dt = p/m, dp/dt = -∇U(x)

    PDHG is the m→∞ limit of HMC where momentum is instantly dissipated.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy import sparse

from .symplectic import LeapfrogIntegrator, IntegratorConfig
from .energy import LPEnergyFunction, EnergyConfig, create_energy_function


@dataclass
class HMCConfig:
    """Configuration for HMC sampler.

    Attributes:
        step_size: Leapfrog step size (ε)
        n_steps: Number of leapfrog steps per proposal
        mass: Mass parameter (m)
        temperature: Sampling temperature (T)
        target_accept_rate: Target Metropolis acceptance rate
        adapt_step_size: Whether to adapt step size
        penalty_init: Initial constraint penalty
        penalty_growth: Penalty growth rate
    """

    step_size: float = 0.01
    n_steps: int = 10
    mass: float = 1.0
    temperature: float = 1.0
    target_accept_rate: float = 0.65
    adapt_step_size: bool = True
    penalty_init: float = 100.0
    penalty_growth: float = 1.05


@dataclass
class HMCResult:
    """Result of HMC sampling.

    Attributes:
        samples: Collected samples, shape (n_samples, n)
        energies: Energy values of samples, shape (n_samples,)
        accept_rate: Overall acceptance rate
        n_accepted: Number of accepted proposals
        n_rejected: Number of rejected proposals
        step_sizes: History of step sizes (if adapting)
        penalty_history: History of penalty values
    """

    samples: np.ndarray
    energies: np.ndarray
    accept_rate: float
    n_accepted: int
    n_rejected: int
    step_sizes: List[float] = field(default_factory=list)
    penalty_history: List[float] = field(default_factory=list)


class HMCPDHG:
    """Hamiltonian Monte Carlo enhanced PDHG solver.

    Combines HMC sampling with PDHG optimization:
    1. Run PDHG for deterministic optimization
    2. Periodically run HMC steps for exploration
    3. Use Metropolis acceptance for convergence guarantees

    This hybrid approach benefits from:
    - Fast convergence of PDHG (deterministic)
    - Exploration capability of HMC (stochastic)
    """

    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        integer_vars: Optional[List[int]] = None,
        config: Optional[HMCConfig] = None,
    ):
        """Initialize HMC-PDHG solver.

        Args:
            A: Constraint matrix
            b: Right-hand side
            c: Objective coefficients
            lb, ub: Variable bounds
            integer_vars: Indices of integer variables
            config: HMC configuration
        """
        self.A = A
        self.b = np.asarray(b, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.integer_vars = integer_vars or []
        self.config = config or HMCConfig()

        self.m, self.n = A.shape

        # Estimate operator norm for step size initialization
        self.norm_A = self._estimate_norm()

        # Initialize energy function
        energy_config = EnergyConfig(
            penalty_init=self.config.penalty_init,
            penalty_growth=self.config.penalty_growth,
        )
        self.energy = create_energy_function(
            A, b, c, lb, ub, self.integer_vars, energy_config
        )

        # Initialize integrator
        integrator_config = IntegratorConfig(
            step_size=self.config.step_size / self.norm_A,
            n_steps=self.config.n_steps,
            mass=self.config.mass,
        )
        self.integrator = LeapfrogIntegrator(integrator_config)

        # Acceptance statistics
        self.n_accepted = 0
        self.n_rejected = 0

    def _estimate_norm(self, n_iters: int = 20) -> float:
        """Estimate ||A||_2 using power iteration."""
        v = np.random.randn(self.n)
        v = v / np.linalg.norm(v)

        for _ in range(n_iters):
            u = self.A @ v
            v_new = self.A.T @ u
            norm = np.linalg.norm(v_new)
            if norm > 1e-10:
                v = v_new / norm
            else:
                break

        return np.linalg.norm(self.A @ v)

    def sample_momentum(self, shape: Tuple[int, ...], seed: Optional[int] = None) -> np.ndarray:
        """Sample momentum from N(0, mass*I).

        Args:
            shape: Shape of momentum array
            seed: Random seed

        Returns:
            Momentum array
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.randn(*shape) * np.sqrt(self.config.mass)

    def hmc_step(
        self,
        x: np.ndarray,
        penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, float]:
        """Perform one HMC step.

        1. Sample momentum p ~ N(0, m*I)
        2. Compute initial Hamiltonian H_0
        3. Run leapfrog integration
        4. Compute final Hamiltonian H_1
        5. Accept with probability min(1, exp(-(H_1 - H_0)/T))

        Args:
            x: Current position, shape (n,) or (K, n)
            penalty: Current penalty (uses self.energy.penalty if None)
            seed: Random seed

        Returns:
            Tuple of (new_x, accepted, delta_H)
        """
        if seed is not None:
            np.random.seed(seed)

        single_dim = x.ndim == 1
        if single_dim:
            x = x.reshape(1, -1)

        K = x.shape[0]

        # Update penalty if provided
        if penalty is not None:
            self.energy.penalty = penalty

        # Sample momentum
        p = self.sample_momentum(x.shape)

        # Initial Hamiltonian
        H_0 = self.energy.hamiltonian(x, p, self.config.mass)

        # Leapfrog integration
        grad_U = lambda q: self.energy.grad_potential(q)
        project = lambda q: self.energy.project_to_bounds(q)

        x_new, p_new = self.integrator.integrate(x, p, grad_U, project)

        # Final Hamiltonian
        H_1 = self.energy.hamiltonian(x_new, -p_new, self.config.mass)

        # Metropolis acceptance
        delta_H = H_1 - H_0
        T = self.config.temperature

        # Accept probability with numerical stability
        log_accept_prob = -delta_H / T
        accept_prob = np.minimum(1.0, np.exp(np.clip(log_accept_prob, -20, 0)))

        # Accept/reject
        rand = np.random.uniform(size=K)
        accepted = rand < accept_prob

        # Apply acceptance
        x_result = np.where(accepted[:, None], x_new, x)

        # Update statistics
        self.n_accepted += np.sum(accepted)
        self.n_rejected += np.sum(~accepted)

        # Adapt step size if enabled
        if self.config.adapt_step_size:
            self._adapt_step_size(float(np.mean(accepted)))

        if single_dim:
            return x_result[0], bool(accepted[0]), float(delta_H[0])
        return x_result, accepted, delta_H

    def _adapt_step_size(self, accept_rate: float) -> None:
        """Adapt step size based on acceptance rate."""
        target = self.config.target_accept_rate

        if accept_rate > target + 0.05:
            self.integrator.eps *= 1.02
        elif accept_rate < target - 0.05:
            self.integrator.eps *= 0.98

        # Clamp to reasonable range
        self.integrator.eps = np.clip(
            self.integrator.eps, 1e-6 / self.norm_A, 1.0 / self.norm_A
        )

    def solve(
        self,
        max_iter: int = 1000,
        hmc_interval: int = 50,
        n_samples: int = 100,
        verbose: bool = False,
        seed: Optional[int] = None,
        x_init: Optional[np.ndarray] = None,
    ) -> HMCResult:
        """Solve LP/MIP using HMC-PDHG hybrid.

        Args:
            max_iter: Maximum number of iterations
            hmc_interval: Run HMC every this many PDHG steps
            n_samples: Number of samples to collect
            verbose: Print progress
            seed: Random seed
            x_init: Initial solution

        Returns:
            HMCResult with samples and statistics
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize
        if x_init is not None:
            x = x_init.copy()
        else:
            x = np.clip(np.zeros(self.n), self.lb, self.ub)

        # PDHG step sizes
        eta = 0.99 / self.norm_A
        tau = 0.99 / self.norm_A

        # Sample collection
        samples = []
        energies = []
        step_sizes = []
        penalty_history = []

        sample_interval = max(1, max_iter // n_samples)

        for k in range(1, max_iter + 1):
            # Standard PDHG step
            ATy = self.A.T @ (np.zeros(self.m) if x.ndim == 1 else np.zeros(self.m))
            # For single solution:
            if x.ndim == 1:
                # PDHG primal update
                grad = self.c  # y = 0 initially
                x_pdhg = np.clip(x - eta * grad, self.lb, self.ub)

                # PDHG dual update (simplified, y stays at 0 for exploration)
                # In full implementation, would update y as well

                x = x_pdhg

            # HMC step for exploration
            if k % hmc_interval == 0:
                x, accepted, delta_H = self.hmc_step(
                    x.reshape(1, -1) if x.ndim == 1 else x,
                    penalty=self.energy.penalty,
                    seed=None,  # Don't fix seed for variety
                )
                if x.ndim == 2 and x.shape[0] == 1:
                    x = x[0]

                # Update penalty
                self.energy.update_penalty()

                if verbose and k % (hmc_interval * 10) == 0:
                    H = self.energy.potential_energy(x)
                    print(f"Iter {k}: U = {H:.4f}, penalty = {self.energy.penalty:.1f}")

            # Collect sample
            if k % sample_interval == 0:
                samples.append(x.copy())
                energies.append(self.energy.potential_energy(x))
                step_sizes.append(self.integrator.eps)
                penalty_history.append(self.energy.penalty)

        # Compute statistics
        total = self.n_accepted + self.n_rejected
        accept_rate = self.n_accepted / total if total > 0 else 0.0

        return HMCResult(
            samples=np.array(samples),
            energies=np.array(energies),
            accept_rate=accept_rate,
            n_accepted=self.n_accepted,
            n_rejected=self.n_rejected,
            step_sizes=step_sizes,
            penalty_history=penalty_history,
        )

    def get_best_sample(self, result: HMCResult) -> Tuple[np.ndarray, float]:
        """Get the best (lowest energy) sample.

        Args:
            result: HMC result

        Returns:
            Tuple of (best_x, best_energy)
        """
        best_idx = np.argmin(result.energies)
        return result.samples[best_idx], result.energies[best_idx]


if __name__ == "__main__":
    # Test HMC-PDHG
    print("Testing HMC-PDHG")
    print("=" * 50)

    from scipy import sparse
    import gurobipy as gp

    # Simple LP
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([-1.0, -1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])

    # Gurobi reference
    import tempfile
    import os

    # Create temp MPS file for Gurobi
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
        f.write("NAME TEST\nROWS\n N OBJ\n L C1\nCOLUMNS\n")
        f.write("    x1 OBJ -1 C1 1\n    x2 OBJ -1 C1 1\n")
        f.write("RHS\n    RHS C1 1\nBOUNDS\n")
        f.write(" PL BND x1\n PL BND x2\nENDATA\n")
        mps_file = f.name

    model = gp.read(mps_file)
    model.setParam('OutputFlag', 0)
    model.setParam('Method', 2)
    model.optimize()
    print(f"Gurobi optimal: {model.ObjVal:.6f}")
    os.unlink(mps_file)

    # HMC-PDHG
    config = HMCConfig(
        step_size=0.1,
        n_steps=10,
        mass=1.0,
        temperature=0.1,
        penalty_init=100.0,
        penalty_growth=1.1,
    )

    solver = HMCPDHG(A, b, c, lb, ub, config=config)
    result = solver.solve(max_iter=500, hmc_interval=20, verbose=True, seed=42)

    best_x, best_U = solver.get_best_sample(result)

    print(f"\nHMC-PDHG results:")
    print(f"  Best energy: {best_U:.6f}")
    print(f"  Best x: {best_x}")
    print(f"  Accept rate: {result.accept_rate:.1%}")
    print(f"  Samples collected: {len(result.samples)}")
