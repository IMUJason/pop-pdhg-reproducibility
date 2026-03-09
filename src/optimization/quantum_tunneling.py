"""
Quantum Tunneling Operator for Optimization.

This module implements quantum tunneling-inspired non-local jumps to help
escape local minima in optimization landscapes.

Key concepts:
- WKB approximation for tunneling probability: P ≈ exp(-Γ · ||Δx|| · √(ΔV))
- Multi-scale exploration using Lévy flights
- History-guided tunneling to best points
- Population-guided tunneling between members

References:
    - WKB approximation in quantum mechanics
    - Simulated quantum annealing (SQA)
    - Path integral Monte Carlo (PIMC)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import numpy as np


@dataclass
class TunnelConfig:
    """Configuration for quantum tunneling operator.

    Attributes:
        tunnel_strength: Strength parameter Γ in WKB formula
        tunnel_prob: Probability of attempting tunneling each iteration
        multi_scale: Whether to use multi-scale tunneling
        scales: List of scales for multi-scale tunneling
        adaptive: Whether to adaptively adjust strength
        target_success_rate: Target success rate for adaptive adjustment
    """
    tunnel_strength: float = 1.0
    tunnel_prob: float = 0.05
    multi_scale: bool = True
    scales: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    adaptive: bool = True
    target_success_rate: float = 0.2

    # History settings
    max_history: int = 50


@dataclass
class TunnelStats:
    """Statistics for tunneling operations."""
    attempts: int = 0
    successes: int = 0
    total_energy_improvement: float = 0.0
    by_strategy: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_improvement(self) -> float:
        return self.total_energy_improvement / self.successes if self.successes > 0 else 0.0


class QuantumTunnelOperator:
    """Quantum tunneling operator for optimization.

    Implements non-local jumps inspired by quantum tunneling:
    - WKB approximation for tunneling probability
    - Multi-scale Lévy flight exploration
    - History-guided jumps to best points
    - Population-guided jumps between members

    Example:
        >>> tunnel = QuantumTunnelOperator(norm_A=1.5)
        >>> x_new, accepted = tunnel.execute(x, energy_fn, iteration=100)
    """

    def __init__(
        self,
        config: Optional[TunnelConfig] = None,
        norm_A: float = 1.0,
    ):
        """Initialize quantum tunneling operator.

        Args:
            config: Tunneling configuration
            norm_A: Norm of constraint matrix for scaling
        """
        self.config = config or TunnelConfig()
        self.norm_A = norm_A

        # Statistics
        self.stats = TunnelStats()

        # History of best points for guided tunneling
        self.best_points: List[Tuple[np.ndarray, float]] = []

        # Current scale index for multi-scale
        self.scale_idx = 0

    def tunnel_probability_wkb(
        self,
        distance: float,
        delta_V: float,
    ) -> float:
        """Calculate tunneling probability using WKB approximation.

        P_tunnel ≈ exp(-Γ · ||Δx|| · √(ΔV))

        For "high and narrow" barriers, this gives higher probability
        than simulated annealing (which depends on barrier height linearly).

        Args:
            distance: Distance between current and target points
            delta_V: Energy barrier (max(V_target, V_current) - min)

        Returns:
            Tunneling probability in [0, 1]
        """
        if delta_V < 1e-10:
            return 1.0

        exponent = -self.config.tunnel_strength * distance * np.sqrt(delta_V) / self.norm_A
        return float(np.exp(np.clip(exponent, -50, 0)))

    def propose_jump(
        self,
        x: np.ndarray,
        iteration: int,
        population: Optional[np.ndarray] = None,
        strategy: str = "auto",
    ) -> Tuple[np.ndarray, str]:
        """Propose a tunneling jump target.

        Strategies:
        1. random: Multi-scale Lévy flight
        2. history: Jump toward historical best points
        3. population: Jump toward other population members

        Args:
            x: Current position
            iteration: Current iteration number
            population: Optional array of population members (K x n)
            strategy: Strategy to use ("auto", "random", "history", "population")

        Returns:
            Tuple of (target position, strategy used)
        """
        n = len(x)

        # Auto-select strategy
        if strategy == "auto":
            has_population = population is not None and len(population) > 0
            has_history = len(self.best_points) > 0

            if has_population and has_history:
                strategy = np.random.choice(
                    ["random", "history", "population"],
                    p=[0.4, 0.3, 0.3]
                )
            elif has_history:
                strategy = np.random.choice(["random", "history"], p=[0.7, 0.3])
            else:
                strategy = "random"

        if strategy == "random":
            x_target = self._levy_flight_jump(x, iteration)

        elif strategy == "history":
            x_target = self._history_guided_jump(x)

        elif strategy == "population":
            x_target = self._population_guided_jump(x, population)

        else:
            x_target = self._levy_flight_jump(x, iteration)

        return x_target, strategy

    def _levy_flight_jump(self, x: np.ndarray, iteration: int) -> np.ndarray:
        """Generate Lévy flight jump for exploration.

        Lévy flights produce long-tailed jump distributions, allowing
        for occasional large jumps while mostly staying local.
        """
        n = len(x)

        # Select scale for multi-scale tunneling
        if self.config.multi_scale:
            scale = self.config.scales[iteration % len(self.config.scales)]
        else:
            scale = 0.1

        # Mantegna algorithm for Lévy stable distribution
        alpha = 1.5  # Lévy index (1 < α < 2)

        u = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, size=n)
        v = np.random.exponential(1.0, size=n)

        # Lévy step
        step = (np.sin(alpha * u) / (np.cos(u) ** (1 / alpha))) * \
               (np.cos((alpha - 1) * u) / v) ** ((alpha - 1) / alpha)

        # Handle numerical issues
        step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale by norm_A for problem-appropriate scaling
        return x + scale * step / self.norm_A

    def _history_guided_jump(self, x: np.ndarray) -> np.ndarray:
        """Jump toward a historical best point."""
        if not self.best_points:
            return self._levy_flight_jump(x, 0)

        # Select a best point (weighted toward better points)
        idx = np.random.randint(min(10, len(self.best_points)))
        target_x, _ = self.best_points[idx]

        # Add noise for exploration
        noise = np.random.randn(len(x)) * 0.1 / self.norm_A
        return target_x + noise

    def _population_guided_jump(
        self,
        x: np.ndarray,
        population: np.ndarray,
    ) -> np.ndarray:
        """Jump toward another population member."""
        if population is None or len(population) == 0:
            return self._levy_flight_jump(x, 0)

        # Select random other member
        idx = np.random.randint(len(population))
        target_x = population[idx].copy()

        # Add small perturbation
        target_x += np.random.randn(len(x)) * 0.05 / self.norm_A

        return target_x

    def execute(
        self,
        x: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        iteration: int,
        population: Optional[np.ndarray] = None,
        project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, bool, float]:
        """Execute a tunneling operation.

        Args:
            x: Current position
            energy_fn: Function computing energy (objective + penalty)
            iteration: Current iteration number
            population: Optional population for guided tunneling
            project_fn: Optional projection function for bounds
            temperature: Temperature for Metropolis acceptance

        Returns:
            Tuple of (new position, accepted, energy change)
        """
        self.stats.attempts += 1

        # Current energy
        V_current = energy_fn(x)

        # Propose jump
        x_target, strategy = self.propose_jump(x, iteration, population)

        # Project if needed
        if project_fn is not None:
            x_target = project_fn(x_target)

        # Target energy
        V_target = energy_fn(x_target)

        # Calculate tunneling probability (WKB)
        distance = np.linalg.norm(x_target - x)
        delta_V = max(V_current, V_target) - min(V_current, V_target)
        P_tunnel = self.tunnel_probability_wkb(distance, delta_V)

        # Combined acceptance probability
        delta_E = V_target - V_current

        if delta_E < 0:
            # Energy decrease: accept with tunneling probability
            P_accept = P_tunnel
        else:
            # Energy increase: combine tunneling with thermal acceptance
            P_accept = P_tunnel * np.exp(-delta_E / temperature)

        # Accept or reject
        if np.random.random() < P_accept:
            # Update statistics
            self.stats.successes += 1
            self.stats.total_energy_improvement += max(0, -delta_E)

            if strategy not in self.stats.by_strategy:
                self.stats.by_strategy[strategy] = {"attempts": 0, "successes": 0}
            self.stats.by_strategy[strategy]["attempts"] += 1
            self.stats.by_strategy[strategy]["successes"] += 1

            # Update history
            self._update_history(x_target, V_target)

            return x_target, True, delta_E

        return x, False, 0.0

    def _update_history(self, x: np.ndarray, energy: float) -> None:
        """Update history of best points."""
        self.best_points.append((x.copy(), energy))

        # Sort by energy and keep best
        self.best_points.sort(key=lambda p: p[1])
        self.best_points = self.best_points[:self.config.max_history]

    def adapt_strength(self) -> None:
        """Adapt tunneling strength based on success rate."""
        if not self.config.adaptive or self.stats.attempts < 10:
            return

        current_rate = self.stats.success_rate
        target = self.config.target_success_rate

        if current_rate > target + 0.1:
            # Too many successes, increase strength (harder to tunnel)
            self.config.tunnel_strength *= 1.1
        elif current_rate < target - 0.1:
            # Too few successes, decrease strength (easier to tunnel)
            self.config.tunnel_strength *= 0.9

        # Keep in reasonable range
        self.config.tunnel_strength = np.clip(
            self.config.tunnel_strength, 0.1, 10.0
        )

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = TunnelStats()

    def get_stats(self) -> dict:
        """Get tunneling statistics."""
        return {
            "attempts": self.stats.attempts,
            "successes": self.stats.successes,
            "success_rate": self.stats.success_rate,
            "avg_improvement": self.stats.avg_improvement,
            "by_strategy": self.stats.by_strategy,
            "current_strength": self.config.tunnel_strength,
            "history_size": len(self.best_points),
        }


class MultiScaleTunnel:
    """Multi-scale tunneling for hierarchical exploration.

    Explores at multiple scales simultaneously:
    - Fine scale: Local refinement
    - Medium scale: Basin exploration
    - Coarse scale: Global jumps
    """

    def __init__(
        self,
        scales: List[float] = None,
        norm_A: float = 1.0,
    ):
        """Initialize multi-scale tunneling.

        Args:
            scales: List of scale factors
            norm_A: Norm of constraint matrix
        """
        self.scales = scales or [0.01, 0.1, 1.0, 10.0]
        self.norm_A = norm_A

        # Separate operators for each scale
        self.operators = [
            QuantumTunnelOperator(
                TunnelConfig(tunnel_strength=1.0 / scale, tunnel_prob=0.05),
                norm_A=norm_A,
            )
            for scale in self.scales
        ]

    def execute(
        self,
        x: np.ndarray,
        energy_fn: Callable,
        iteration: int,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, float]:
        """Execute tunneling at appropriate scale."""
        # Cycle through scales
        scale_idx = iteration % len(self.scales)
        return self.operators[scale_idx].execute(x, energy_fn, iteration, **kwargs)

    def get_stats(self) -> List[dict]:
        """Get statistics for all scales."""
        return [op.get_stats() for op in self.operators]


if __name__ == "__main__":
    print("Testing Quantum Tunneling Operator...")

    # Simple 2D test: minimize (x-1)^2 + (y-2)^2
    def energy(q):
        return (q[0] - 1) ** 2 + (q[1] - 2) ** 2

    tunnel = QuantumTunnelOperator(norm_A=1.0)

    # Start at origin
    x = np.array([0.0, 0.0])

    print(f"Initial: x={x}, energy={energy(x):.4f}")

    # Run tunneling
    for i in range(100):
        x_new, accepted, delta_E = tunnel.execute(
            x, energy, i, temperature=1.0
        )
        if accepted:
            x = x_new
            print(f"  Iter {i}: accepted jump to {x}, energy={energy(x):.4f}")

    print(f"\nFinal: x={x}, energy={energy(x):.4f}")
    print(f"Stats: {tunnel.get_stats()}")
    print(f"Expected: x=[1.0, 2.0], energy=0.0")
