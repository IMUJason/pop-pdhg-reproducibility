"""
Metal-Accelerated Quantum-Inspired MIP Solver

利用 Apple Silicon 统一内存架构实现高性能异构求解。
"""

__version__ = "0.1.0"

from .solver import MetalQuantumMIPSolver, SolverConfig, SolverResult
from .device import MetalDevice, UnifiedMemoryBuffer

__all__ = [
    "MetalQuantumMIPSolver",
    "SolverConfig",
    "SolverResult",
    "MetalDevice",
    "UnifiedMemoryBuffer",
]
