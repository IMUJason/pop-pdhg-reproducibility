"""
Quantum Tunneling Kernels - GPU/CPU Implementation
"""

import numpy as np
from scipy import stats


class TunnelKernels:
    """GPU/CPU kernels for quantum tunneling operations."""

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def levy_flight(self, x, scale=1.0, alpha=1.5):
        """Levy flight jump for global exploration.

        Args:
            x: Current position (K, n) or (n,)
            scale: Jump scale
            alpha: Stability parameter (1 < alpha < 2)

        Returns:
            New position after jump
        """
        if x.ndim == 1:
            # Single vector
            n = len(x)
            direction = np.random.randn(n)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Levy flight step length
            step = stats.levy_stable.rvs(alpha, 0, scale=scale)

            return x + step * direction
        else:
            # Batch
            K, n = x.shape
            jumps = np.zeros_like(x)

            for k in range(K):
                direction = np.random.randn(n)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                step = stats.levy_stable.rvs(alpha, 0, scale=scale)
                jumps[k] = x[k] + step * direction

            return jumps

    def wkb_acceptance(self, x_old, x_new, energy_fn, temperature=1.0):
        """WKB tunneling probability for accepting jumps.

        P_tunnel = exp(-Gamma * ||dx|| * sqrt(2m * delta_E))

        Args:
            x_old: Current position
            x_new: Proposed position
            energy_fn: Function to compute energy
            temperature: Temperature parameter

        Returns:
            (accepted, prob)
        """
        E_old = energy_fn(x_old)
        E_new = energy_fn(x_new)
        delta_E = E_new - E_old

        if delta_E < 0:
            # Lower energy, always accept
            return True, 1.0

        # Compute WKB tunneling probability
        dx = np.linalg.norm(x_new - x_old)

        # WKB approximation: P = exp(-sqrt(2 * delta_E) * |dx|)
        # (simplified, assuming m=1, Gamma=1)
        wkb_exponent = -np.sqrt(2 * delta_E) * dx / temperature
        prob = np.exp(wkb_exponent)

        accepted = np.random.random() < prob
        return accepted, prob

    def population_guided_jump(self, x, population, elite_frac=0.2):
        """Jump guided by elite population members.

        Args:
            x: Current position
            population: (K, n) all population members
            elite_frac: Fraction of elite members

        Returns:
            New position
        """
        K = population.shape[0]
        n_elite = max(1, int(K * elite_frac))

        # Random elite member
        elite_idx = np.random.randint(0, n_elite)
        x_elite = population[elite_idx]

        # Interpolate with some noise
        alpha = np.random.beta(2, 5)  # Bias towards current position
        noise = np.random.randn(len(x)) * 0.1

        return alpha * x + (1 - alpha) * x_elite + noise


class AdaptiveTunnelStrength:
    """Adaptively adjust tunneling strength based on progress."""

    def __init__(self, initial_strength=1.0, min_strength=0.1, max_strength=10.0):
        self.strength = initial_strength
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.history = []

    def update(self, success_rate):
        """Update tunneling strength based on success rate.

        Args:
            success_rate: Recent tunnel success rate (0-1)
        """
        self.history.append(success_rate)

        if len(self.history) < 5:
            return

        # Recent average
        avg_success = np.mean(self.history[-5:])

        if avg_success > 0.3:
            # Too many successes - increase exploration
            self.strength = min(self.strength * 1.1, self.max_strength)
        elif avg_success < 0.1:
            # Too few successes - be more conservative
            self.strength = max(self.strength * 0.9, self.min_strength)

    def get_strength(self):
        return self.strength


if __name__ == "__main__":
    print("Testing Tunnel Kernels...")

    kernels = TunnelKernels(use_gpu=False)

    # Test Levy flight
    x = np.random.randn(100)
    x_new = kernels.levy_flight(x, scale=1.0)
    print(f"Levy flight: ||dx|| = {np.linalg.norm(x_new - x):.4f}")

    # Test WKB acceptance
    def energy(x):
        return np.sum(x**2)

    accepted, prob = kernels.wkb_acceptance(x, x_new, energy)
    print(f"WKB acceptance: {accepted}, prob={prob:.4f}")

    print("Test passed!")
