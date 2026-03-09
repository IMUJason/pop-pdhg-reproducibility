"""
Optimization module for Quantum-Inspired MIP Solver.

This module contains advanced optimization techniques:
- Parallel Tempering (Replica Exchange)
- Adaptive HMC with dual averaging
- Quantum Tunneling Operator
- Simulated Bifurcation
- Adaptive restart strategies
"""

from .parallel_tempering import (
    ParallelTemperingPDHG,
    ParallelTemperingResult,
    ReplicaState,
    solve_lp_parallel_tempering,
)

from .adaptive_hmc import (
    AdaptiveHMC,
    AdaptiveHMCConfig,
    HMCResult,
    DualAveragingStepSize,
    MassMatrixAdapter,
    ReflectiveLeapfrog,
    SmartHMCTrigger,
)

from .quantum_tunneling import (
    QuantumTunnelOperator,
    TunnelConfig,
    TunnelStats,
    MultiScaleTunnel,
)

from .progressive_measurement import (
    ProgressiveMeasurement,
    EntangledMeasurement,
    MeasurementConfig,
    MeasurementStats,
)

from .feasibility_repair import (
    FeasibilityRepair,
    repair_solution,
    compute_violation,
    clip_bounds,
    round_integers,
)

__all__ = [
    # Parallel Tempering
    "ParallelTemperingPDHG",
    "ParallelTemperingResult",
    "ReplicaState",
    "solve_lp_parallel_tempering",
    # Adaptive HMC
    "AdaptiveHMC",
    "AdaptiveHMCConfig",
    "HMCResult",
    "DualAveragingStepSize",
    "MassMatrixAdapter",
    "ReflectiveLeapfrog",
    "SmartHMCTrigger",
    # Quantum Tunneling
    "QuantumTunnelOperator",
    "TunnelConfig",
    "TunnelStats",
    "MultiScaleTunnel",
    # Progressive Measurement
    "ProgressiveMeasurement",
    "EntangledMeasurement",
    "MeasurementConfig",
    "MeasurementStats",
    # Feasibility Repair
    "FeasibilityRepair",
    "repair_solution",
    "compute_violation",
    "clip_bounds",
    "round_integers",
]
