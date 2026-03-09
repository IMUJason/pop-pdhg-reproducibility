"""
Symplectic Integrators for Hamiltonian Monte Carlo.

This module implements symplectic (volume-preserving) integrators for
Hamiltonian dynamics. The key property of symplectic integrators is
that they preserve the phase space volume (Liouville's theorem),
which ensures the detailed balance condition for HMC.

The most commonly used integrator is Leapfrog (Störmer-Verlet),
which is second-order accurate and time-reversible.

Mathematical background:
    Hamiltonian: H(q, p) = U(q) + K(p)
    where U(q) is potential energy, K(p) = ||p||²/(2m) is kinetic energy

    Hamilton's equations:
        dq/dt = ∂H/∂p = p/m
        dp/dt = -∂H/∂q = -∇U(q)

    Leapfrog discretization:
        p_{1/2} = p_0 - (ε/2) ∇U(q_0)
        q_1 = q_0 + ε (p_{1/2}/m)
        p_1 = p_{1/2} - (ε/2) ∇U(q_1)
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class IntegratorConfig:
    """Configuration for symplectic integrator.

    Attributes:
        step_size: Time step size (ε)
        n_steps: Number of integration steps per trajectory
        mass: Mass parameter (m), controls inertia
        adaptive: Whether to use adaptive step size
        target_accept_rate: Target Metropolis acceptance rate
    """

    step_size: float = 0.01
    n_steps: int = 10
    mass: float = 1.0
    adaptive: bool = False
    target_accept_rate: float = 0.65


class LeapfrogIntegrator:
    """Leapfrog (Störmer-Verlet) symplectic integrator.

    The leapfrog integrator is:
    - Second-order accurate
    - Time-reversible
    - Symplectic (preserves phase space volume)
    - Simple to implement

    These properties make it ideal for HMC sampling.

    Example:
        >>> integrator = LeapfrogIntegrator(step_size=0.01, n_steps=20)
        >>> q_final, p_final = integrator.integrate(q0, p0, grad_U)
    """

    def __init__(self, config: Optional[IntegratorConfig] = None):
        """Initialize leapfrog integrator.

        Args:
            config: Integrator configuration. Uses defaults if None.
        """
        self.config = config or IntegratorConfig()
        self.eps = self.config.step_size
        self.n_steps = self.config.n_steps
        self.mass = self.config.mass

    def integrate(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        grad_U: Callable[[np.ndarray], np.ndarray],
        project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform n_steps of leapfrog integration.

        Args:
            q0: Initial position, shape (n,) or (K, n) for batch
            p0: Initial momentum, same shape as q0
            grad_U: Gradient of potential energy, q -> ∇U(q)
            project: Optional projection function for constrained dynamics

        Returns:
            Tuple of (q_final, p_final)
        """
        q = q0.copy()
        p = p0.copy()

        # Half step for momentum
        p = p - (self.eps / 2) * grad_U(q)

        # Full steps
        for step in range(self.n_steps - 1):
            # Full step for position
            q = q + self.eps * (p / self.mass)

            # Project if needed
            if project is not None:
                q = project(q)

            # Full step for momentum
            p = p - self.eps * grad_U(q)

        # Final full step for position
        q = q + self.eps * (p / self.mass)
        if project is not None:
            q = project(q)

        # Half step for momentum
        p = p - (self.eps / 2) * grad_U(q)

        # Negate momentum for time reversibility
        p = -p

        return q, p

    def integrate_batch(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        grad_U: Callable[[np.ndarray], np.ndarray],
        project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate a batch of trajectories in parallel.

        Args:
            q0: Initial positions, shape (K, n)
            p0: Initial momenta, shape (K, n)
            grad_U: Batch gradient function, (K, n) -> (K, n)
            project: Optional projection function

        Returns:
            Tuple of (q_final, p_final), both shape (K, n)
        """
        return self.integrate(q0, p0, grad_U, project)


class Ruth4Integrator:
    """Fourth-order Ruth symplectic integrator.

    A 4th order symplectic integrator using a composition of
    leapfrog-like steps with carefully chosen coefficients.

    More accurate than leapfrog but requires 4 gradient evaluations
    per step (vs 2 for leapfrog).

    Coefficients from: Ruth (1983), "A canonical integration technique"
    """

    # Coefficients for 4th order Ruth integrator
    C = np.array([
        1.0 / (2 * (2 - 2**(1/3))),
        (1 - 2**(1/3)) / (2 * (2 - 2**(1/3))),
        (1 - 2**(1/3)) / (2 * (2 - 2**(1/3))),
        1.0 / (2 * (2 - 2**(1/3))),
    ])

    D = np.array([
        0.0,
        1.0 / (2 - 2**(1/3)),
        -2**(1/3) / (2 - 2**(1/3)),
        1.0 / (2 - 2**(1/3)),
    ])

    def __init__(self, config: Optional[IntegratorConfig] = None):
        """Initialize Ruth 4 integrator."""
        self.config = config or IntegratorConfig()
        self.eps = self.config.step_size
        self.n_steps = self.config.n_steps
        self.mass = self.config.mass

    def integrate(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        grad_U: Callable[[np.ndarray], np.ndarray],
        project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform 4th order Ruth integration."""
        q = q0.copy()
        p = p0.copy()

        for _ in range(self.n_steps):
            for c, d in zip(self.C, self.D):
                # Position update
                q = q + c * self.eps * (p / self.mass)
                if project is not None:
                    q = project(q)
                # Momentum update
                p = p - d * self.eps * grad_U(q)

        return q, -p  # Negate for time reversibility


class AdaptiveLeapfrog(LeapfrogIntegrator):
    """Leapfrog integrator with adaptive step size.

    Adjusts step size based on Metropolis acceptance rate to
    maintain target acceptance rate (typically 60-80%).
    """

    def __init__(self, config: Optional[IntegratorConfig] = None):
        """Initialize adaptive integrator."""
        super().__init__(config)
        self.target_rate = self.config.target_accept_rate

        # History for adaptation
        self.accept_history: list[bool] = []
        self.adapt_window = 50

    def adapt_step_size(self, accepted: bool) -> None:
        """Adapt step size based on acceptance.

        Uses simple multiplicative adjustment:
        - If accepted: increase step size slightly
        - If rejected: decrease step size

        Args:
            accepted: Whether the last proposal was accepted
        """
        self.accept_history.append(accepted)

        if len(self.accept_history) >= self.adapt_window:
            recent_rate = np.mean(self.accept_history[-self.adapt_window:])

            # Multiplicative adjustment
            if recent_rate > self.target_rate + 0.05:
                self.eps *= 1.1  # Increase
            elif recent_rate < self.target_rate - 0.05:
                self.eps *= 0.9  # Decrease

            # Clamp to reasonable range
            self.eps = np.clip(self.eps, 1e-6, 1.0)


if __name__ == "__main__":
    # Test leapfrog integrator on simple harmonic oscillator
    print("Testing Leapfrog Integrator on Harmonic Oscillator")
    print("=" * 50)

    # Harmonic oscillator: U(q) = 0.5 * k * q^2
    k = 1.0
    m = 1.0

    def grad_U(q):
        return k * q

    # Initial conditions
    q0 = np.array([1.0])
    p0 = np.array([0.0])

    # Exact solution: q(t) = cos(ωt), p(t) = -mω sin(ωt), where ω = sqrt(k/m)
    omega = np.sqrt(k / m)

    # Integrate
    config = IntegratorConfig(step_size=0.1, n_steps=100, mass=m)
    integrator = LeapfrogIntegrator(config)

    q, p = integrator.integrate(q0, p0, grad_U)

    T = config.step_size * config.n_steps
    q_exact = np.cos(omega * T)
    p_exact = -m * omega * np.sin(omega * T)

    print(f"Final time T = {T:.2f}")
    print(f"Numerical:  q = {q[0]:.6f}, p = {p[0]:.6f}")
    print(f"Exact:      q = {q_exact:.6f}, p = {p_exact:.6f}")
    print(f"Error:      |Δq| = {abs(q[0] - q_exact):.2e}, |Δp| = {abs(p[0] - p_exact):.2e}")

    # Check energy conservation
    E0 = 0.5 * k * q0[0]**2 + 0.5 * p0[0]**2 / m
    E_final = 0.5 * k * q[0]**2 + 0.5 * p[0]**2 / m
    print(f"\nEnergy conservation:")
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Final energy:   {E_final:.6f}")
    print(f"  Relative error: {abs(E_final - E0) / E0:.2e}")
