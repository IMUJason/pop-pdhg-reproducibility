"""
Adaptive HMC with Dual Averaging Step Size and Mass Matrix Adaptation.

This module implements state-of-the-art HMC improvements:
1. Dual averaging step size adaptation (Hoffman & Gelman, 2014)
2. Mass matrix adaptation from samples
3. Reflective boundary handling
4. Smart HMC trigger based on PDHG stagnation

References:
    - Hoffman & Gelman (2014): The No-U-Turn Sampler
    - Neal (2011): MCMC Using Hamiltonian Dynamics
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List
import numpy as np


@dataclass
class DualAveragingConfig:
    """Configuration for dual averaging step size adaptation."""

    eps_init: float = 1.0
    target_accept: float = 0.65
    gamma: float = 0.05
    t0: int = 10
    kappa: float = 0.75


@dataclass
class DualAveragingState:
    """State of dual averaging adaptation."""

    mu: float
    log_eps: float
    log_eps_bar: float
    H_bar: float
    t: int


class DualAveragingStepSize:
    """Dual averaging step size adaptation.

    Implements Hoffman & Gelman (2014) algorithm for automatic
    step size tuning in HMC.

    log(eps_{t+1}) = mu - sqrt(t)/gamma * H_bar
    where H_bar tracks deviation from target acceptance rate.
    """

    def __init__(self, config: Optional[DualAveragingConfig] = None):
        """Initialize dual averaging.

        Args:
            config: Configuration parameters
        """
        self.config = config or DualAveragingConfig()
        self.state = None

    def initialize(self, eps_init: Optional[float] = None) -> None:
        """Initialize adaptation state.

        Args:
            eps_init: Initial step size (uses config if not provided)
        """
        eps = eps_init or self.config.eps_init

        self.state = DualAveragingState(
            mu=np.log(10 * eps),
            log_eps=np.log(eps),
            log_eps_bar=0.0,
            H_bar=0.0,
            t=0,
        )

    def update(self, accept_prob: float) -> float:
        """Update step size based on acceptance probability.

        Args:
            accept_prob: Acceptance probability from last HMC step

        Returns:
            New step size
        """
        if self.state is None:
            self.initialize()

        self.state.t += 1

        # Update H_bar (running average of accept - target)
        w = 1.0 / (self.state.t + self.config.t0)
        self.state.H_bar = (1 - w) * self.state.H_bar + w * (self.config.target_accept - accept_prob)

        # Update log(eps)
        self.state.log_eps = self.state.mu - np.sqrt(self.state.t) / self.config.gamma * self.state.H_bar

        # Update log(eps_bar) (exponential moving average for final step size)
        m = self.state.t ** (-self.config.kappa)
        self.state.log_eps_bar = m * self.state.log_eps + (1 - m) * self.state.log_eps_bar

        return np.exp(self.state.log_eps)

    def get_step_size(self) -> float:
        """Get current step size."""
        if self.state is None:
            self.initialize()
        return np.exp(self.state.log_eps)

    def get_final_step_size(self) -> float:
        """Get final adapted step size."""
        if self.state is None:
            self.initialize()
        return np.exp(self.state.log_eps_bar)


@dataclass
class MassMatrixConfig:
    """Configuration for mass matrix adaptation."""

    adapt_start: int = 100
    adapt_end: int = 1000
    regularization: float = 1e-3


class MassMatrixAdapter:
    """Adaptive mass matrix estimation from samples.

    Estimates the covariance of samples and uses it as the mass matrix
    in HMC. This can significantly improve sampling efficiency for
    ill-conditioned problems.
    """

    def __init__(
        self,
        n_vars: int,
        config: Optional[MassMatrixConfig] = None,
    ):
        """Initialize mass matrix adapter.

        Args:
            n_vars: Number of variables
            config: Configuration parameters
        """
        self.n = n_vars
        self.config = config or MassMatrixConfig()

        # Welford's algorithm for online covariance estimation
        self.count = 0
        self.mean = np.zeros(n_vars)
        self.M2 = np.zeros((n_vars, n_vars))

        # Mass matrices
        self.mass_matrix = np.eye(n_vars)
        self.inv_mass_matrix = np.eye(n_vars)
        self.chol_mass = np.eye(n_vars)

    def update(self, x: np.ndarray, iteration: int) -> None:
        """Update covariance estimate with new sample.

        Args:
            x: Sample point
            iteration: Current iteration number
        """
        if iteration < self.config.adapt_start:
            return

        self.count += 1

        # Welford's online algorithm
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

        if iteration >= self.config.adapt_end and self.count > 1:
            self._finalize()

    def _finalize(self) -> None:
        """Compute final mass matrix from accumulated samples."""
        cov = self.M2 / (self.count - 1)

        # Add regularization
        reg = self.config.regularization * np.trace(cov) / self.n
        cov_reg = cov + reg * np.eye(self.n)

        try:
            self.mass_matrix = cov_reg
            self.inv_mass_matrix = np.linalg.inv(cov_reg)
            self.chol_mass = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            # Fall back to diagonal approximation
            diag_var = np.diag(cov_reg)
            self.mass_matrix = np.diag(diag_var)
            self.inv_mass_matrix = np.diag(1.0 / np.maximum(diag_var, 1e-10))
            self.chol_mass = np.diag(np.sqrt(np.maximum(diag_var, 1e-10)))

    def sample_momentum(self) -> np.ndarray:
        """Sample momentum from N(0, M).

        Returns:
            Momentum vector
        """
        z = np.random.randn(self.n)
        return self.chol_mass @ z

    def kinetic_energy(self, p: np.ndarray) -> float:
        """Compute kinetic energy p^T M^{-1} p / 2.

        Args:
            p: Momentum vector

        Returns:
            Kinetic energy
        """
        return 0.5 * p @ self.inv_mass_matrix @ p


class ReflectiveLeapfrog:
    """Leapfrog integrator with reflective boundary conditions.

    When a trajectory hits a boundary, it reflects both position
    and momentum, preserving the symplectic structure better than
    hard projection.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        step_size: float = 0.01,
    ):
        """Initialize reflective leapfrog integrator.

        Args:
            lb: Lower bounds
            ub: Upper bounds
            step_size: Integration step size
        """
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.eps = step_size

    def reflect(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reflect position and momentum at boundaries.

        Args:
            q: Position
            p: Momentum

        Returns:
            Tuple of (reflected q, reflected p)
        """
        q = q.copy()
        p = p.copy()

        for j in range(len(q)):
            if q[j] < self.lb[j]:
                overshoot = self.lb[j] - q[j]
                q[j] = self.lb[j] + overshoot
                p[j] = -p[j]
            elif not np.isinf(self.ub[j]) and q[j] > self.ub[j]:
                overshoot = q[j] - self.ub[j]
                q[j] = self.ub[j] - overshoot
                p[j] = -p[j]

        return q, p

    def integrate(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_U: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Hamiltonian dynamics with reflection.

        Args:
            q: Initial position
            p: Initial momentum
            grad_U: Gradient of potential energy
            n_steps: Number of leapfrog steps

        Returns:
            Tuple of (final q, final p)
        """
        q = q.copy()
        p = p.copy()

        # Half step for momentum
        p = p - (self.eps / 2) * grad_U(q)

        # Full steps
        for i in range(n_steps - 1):
            q = q + self.eps * p
            q, p = self.reflect(q, p)
            p = p - self.eps * grad_U(q)

        # Final position step
        q = q + self.eps * p
        q, p = self.reflect(q, p)

        # Half step for momentum
        p = p - (self.eps / 2) * grad_U(q)

        # Negate momentum for reversibility
        return q, -p


@dataclass
class AdaptiveHMCConfig:
    """Configuration for adaptive HMC."""

    # Step size
    step_size_init: float = 0.01
    adapt_step_size: bool = True
    target_accept_rate: float = 0.65

    # Trajectory length
    n_steps: int = 10
    jitter_L: bool = True
    jitter_range: float = 0.2

    # Mass matrix
    adapt_mass_matrix: bool = True
    mass_adapt_start: int = 100
    mass_adapt_end: int = 1000

    # Boundary handling
    use_reflection: bool = True

    # Temperature
    temperature_init: float = 1.0
    temperature_final: float = 0.01
    temperature_decay: float = 0.995


@dataclass
class HMCResult:
    """Result of HMC step."""

    q: np.ndarray
    p: np.ndarray
    accepted: bool
    accept_prob: float
    delta_H: float
    step_size: float
    kinetic_energy: float
    potential_energy: float


class AdaptiveHMC:
    """Adaptive HMC with automatic step size and mass matrix tuning.

    Combines:
    - Dual averaging step size adaptation
    - Mass matrix adaptation from samples
    - Reflective boundary conditions
    - Temperature annealing
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        config: Optional[AdaptiveHMCConfig] = None,
    ):
        """Initialize adaptive HMC.

        Args:
            lb: Lower bounds
            ub: Upper bounds
            config: Configuration
        """
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.n = len(lb)
        self.config = config or AdaptiveHMCConfig()

        # Step size adapter
        self.step_adapter = DualAveragingStepSize(
            DualAveragingConfig(
                eps_init=self.config.step_size_init,
                target_accept=self.config.target_accept_rate,
            )
        )
        self.step_adapter.initialize()

        # Mass matrix adapter
        self.mass_adapter = MassMatrixAdapter(
            self.n,
            MassMatrixConfig(
                adapt_start=self.config.mass_adapt_start,
                adapt_end=self.config.mass_adapt_end,
            ),
        )

        # Leapfrog integrator
        self.integrator = ReflectiveLeapfrog(
            self.lb, self.ub, self.step_adapter.get_step_size()
        )

        # State
        self.iteration = 0
        self.temperature = self.config.temperature_init

    def sample_momentum(self) -> np.ndarray:
        """Sample momentum from adapted distribution."""
        return self.mass_adapter.sample_momentum()

    def kinetic_energy(self, p: np.ndarray) -> float:
        """Compute kinetic energy."""
        return self.mass_adapter.kinetic_energy(p)

    def hamiltonian(
        self,
        q: np.ndarray,
        p: np.ndarray,
        potential_fn: Callable[[np.ndarray], float],
    ) -> float:
        """Compute total Hamiltonian H = U + K.

        Args:
            q: Position
            p: Momentum
            potential_fn: Potential energy function

        Returns:
            Total Hamiltonian
        """
        U = potential_fn(q)
        K = self.kinetic_energy(p)
        return U + K

    def step(
        self,
        q: np.ndarray,
        potential_fn: Callable[[np.ndarray], float],
        grad_potential: Callable[[np.ndarray], np.ndarray],
    ) -> HMCResult:
        """Perform one adaptive HMC step.

        Args:
            q: Current position
            potential_fn: Potential energy function
            grad_potential: Gradient of potential energy

        Returns:
            HMCResult with new state and diagnostics
        """
        self.iteration += 1

        # Update temperature
        self.temperature = max(
            self.config.temperature_final,
            self.temperature * self.config.temperature_decay
        )

        # Sample momentum
        p = self.sample_momentum()

        # Compute initial Hamiltonian
        H0 = self.hamiltonian(q, p, potential_fn)

        # Get current step size
        eps = self.step_adapter.get_step_size()
        self.integrator.eps = eps

        # Determine trajectory length (with optional jitter)
        if self.config.jitter_L:
            jitter = int(self.config.n_steps * self.config.jitter_range * (2 * np.random.rand() - 1))
            L = max(1, self.config.n_steps + jitter)
        else:
            L = self.config.n_steps

        # Integrate
        q_new, p_new = self.integrator.integrate(q, p, grad_potential, L)

        # Compute final Hamiltonian
        H1 = self.hamiltonian(q_new, p_new, potential_fn)

        # Metropolis acceptance
        delta_H = H1 - H0
        accept_prob = min(1.0, np.exp(-delta_H / self.temperature))

        accepted = np.random.rand() < accept_prob

        if accepted:
            q_final = q_new
            p_final = p_new
        else:
            q_final = q
            p_final = p

        # Adapt step size
        if self.config.adapt_step_size:
            self.step_adapter.update(accept_prob)

        # Adapt mass matrix
        if self.config.adapt_mass_matrix:
            self.mass_adapter.update(q_final, self.iteration)

        return HMCResult(
            q=q_final,
            p=p_final,
            accepted=accepted,
            accept_prob=accept_prob,
            delta_H=delta_H,
            step_size=eps,
            kinetic_energy=self.kinetic_energy(p_final),
            potential_energy=potential_fn(q_final),
        )


class SmartHMCTrigger:
    """Smart HMC trigger based on PDHG stagnation detection.

    Determines when HMC exploration should be triggered during
    PDHG optimization based on convergence behavior.
    """

    def __init__(
        self,
        stagnation_threshold: int = 50,
        improvement_threshold: float = 1e-8,
        min_interval: int = 100,
        violation_threshold: float = 1.0,
    ):
        """Initialize smart trigger.

        Args:
            stagnation_threshold: Number of iterations to check for stagnation
            improvement_threshold: Minimum improvement to not be considered stagnant
            min_interval: Minimum iterations between HMC triggers
            violation_threshold: Constraint violation threshold for urgent trigger
        """
        self.stagnation_threshold = stagnation_threshold
        self.improvement_threshold = improvement_threshold
        self.min_interval = min_interval
        self.violation_threshold = violation_threshold

        self.improvement_history: List[float] = []
        self.last_trigger = 0

    def update(self, improvement: float) -> None:
        """Record improvement from last PDHG step.

        Args:
            improvement: Change in solution norm or objective
        """
        self.improvement_history.append(improvement)

    def should_trigger(self, primal_violation: float, iteration: int) -> bool:
        """Determine if HMC should be triggered.

        Args:
            primal_violation: Current constraint violation
            iteration: Current iteration number

        Returns:
            True if HMC should be triggered
        """
        # Check minimum interval
        if iteration - self.last_trigger < self.min_interval:
            return False

        # Not enough history
        if len(self.improvement_history) < self.stagnation_threshold:
            return False

        recent = self.improvement_history[-self.stagnation_threshold:]

        # Condition 1: Improvement stagnation
        if np.mean(np.abs(recent)) < self.improvement_threshold:
            self.last_trigger = iteration
            return True

        # Condition 2: Periodic trigger
        if iteration % 200 == 0:
            self.last_trigger = iteration
            return True

        # Condition 3: High violation with slow progress
        if primal_violation > self.violation_threshold:
            if np.mean(np.abs(recent)) < self.improvement_threshold * 100:
                self.last_trigger = iteration
                return True

        return False

    def reset(self) -> None:
        """Reset trigger state."""
        self.improvement_history = []
        self.last_trigger = 0


if __name__ == "__main__":
    print("Testing Adaptive HMC...")

    # Simple 2D test: minimize (x-1)^2 + (y-2)^2
    def potential(q):
        return (q[0] - 1) ** 2 + (q[1] - 2) ** 2

    def grad_potential(q):
        return 2 * np.array([q[0] - 1, q[1] - 2])

    lb = np.array([-10.0, -10.0])
    ub = np.array([10.0, 10.0])

    hmc = AdaptiveHMC(lb, ub)

    # Run HMC
    q = np.array([0.0, 0.0])
    samples = []

    for i in range(500):
        result = hmc.step(q, potential, grad_potential)
        q = result.q
        samples.append(q.copy())

        if i % 100 == 0:
            print(f"Iter {i}: q={q}, accepted={result.accepted}, "
                  f"step_size={result.step_size:.4f}")

    samples = np.array(samples)

    # Discard burn-in
    samples = samples[200:]

    print(f"\nMean: {np.mean(samples, axis=0)}")
    print(f"Std: {np.std(samples, axis=0)}")
    print(f"Expected mean: [1.0, 2.0]")
