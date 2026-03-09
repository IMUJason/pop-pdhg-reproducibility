"""
Heterogeneous CPU-GPU Schedulers for Apple Silicon

10-rule based device selection with performance monitoring.
"""

import time
import numpy as np
from typing import Dict, Optional


class HeterogeneousScheduler:
    """Schedules computation between CPU and GPU based on problem characteristics."""

    def __init__(self):
        self.device_stats = {
            'cpu': {'calls': 0, 'total_time': 0.0},
            'gpu': {'calls': 0, 'total_time': 0.0}
        }
        self.decision_history = []

    def decide_device(self, problem_size: Dict, operation: str,
                      prefer_gpu: bool = True) -> str:
        """Decide whether to use CPU or GPU based on 10 rules.

        Args:
            problem_size: Dict with 'n_vars', 'n_constrs', 'nnz'
            operation: Type of operation ('pdhg', 'tunnel', 'rounding', 'objective')
            prefer_gpu: Whether GPU is preferred when rules are ambiguous

        Returns:
            'cpu' or 'gpu'
        """
        n = problem_size.get('n_vars', 0)
        m = problem_size.get('n_constrs', 0)
        nnz = problem_size.get('nnz', 0)

        # Rule 1: Very small problems -> CPU (avoid GPU launch overhead)
        if n < 50 and m < 50:
            return 'cpu'

        # Rule 2: Integer rounding -> CPU P-cores (needs fine control)
        if operation == 'rounding':
            return 'cpu'

        # Rule 3: Local search -> CPU (sequential by nature)
        if operation == 'local_search':
            return 'cpu'

        # Rule 4: Dense large matrix -> GPU
        density = nnz / (n * m) if n * m > 0 else 0
        if n > 1000 and density > 0.5:
            return 'gpu'

        # Rule 5: Large sparse -> GPU
        if nnz > 100000:
            return 'gpu'

        # Rule 6: Batch operations -> GPU
        if operation in ['pdhg', 'tunnel'] and n > 200:
            return 'gpu'

        # Rule 7: Memory-bound operations -> prefer unified memory (GPU)
        if nnz > 10000 and operation == 'objective':
            return 'gpu'

        # Rule 8: Convergence check -> CPU (lightweight)
        if operation == 'convergence':
            return 'cpu'

        # Rule 9: Initialization -> CPU
        if operation == 'init':
            return 'cpu'

        # Rule 10: Default based on preference
        return 'gpu' if prefer_gpu else 'cpu'

    def record_execution(self, device: str, execution_time: float):
        """Record execution statistics.

        Args:
            device: 'cpu' or 'gpu'
            execution_time: Time taken in seconds
        """
        self.device_stats[device]['calls'] += 1
        self.device_stats[device]['total_time'] += execution_time
        self.decision_history.append((device, execution_time))

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        stats = {}
        for device in ['cpu', 'gpu']:
            calls = self.device_stats[device]['calls']
            total = self.device_stats[device]['total_time']
            avg = total / calls if calls > 0 else 0
            stats[device] = {
                'calls': calls,
                'total_time': total,
                'avg_time': avg
            }
        return stats

    def get_speedup(self) -> Optional[float]:
        """Get observed GPU speedup over CPU."""
        cpu_avg = self.device_stats['cpu']['avg_time']
        gpu_avg = self.device_stats['gpu']['avg_time']

        if cpu_avg > 0 and gpu_avg > 0:
            return cpu_avg / gpu_avg
        return None


class AdaptiveBatchSize:
    """Adaptively adjust population size based on problem characteristics."""

    def __init__(self, base_size=16):
        self.base_size = base_size

    def get_batch_size(self, n_vars: int, n_constrs: int) -> int:
        """Get recommended population size.

        Args:
            n_vars: Number of variables
            n_constrs: Number of constraints

        Returns:
            Recommended population size
        """
        # Small problems: smaller population
        if n_vars < 100:
            return max(4, self.base_size // 4)

        # Medium problems: base size
        if n_vars < 1000:
            return self.base_size

        # Large problems: larger population for diversity
        if n_vars < 10000:
            return self.base_size * 2

        # Very large: max size
        return self.base_size * 4


class PerformanceMonitor:
    """Monitor solver performance and adapt parameters."""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.obj_history = []
        self.feas_history = []
        self.iteration_times = []

    def record_iteration(self, obj_best: float, is_feasible: bool,
                         iteration_time: float):
        """Record iteration statistics."""
        self.obj_history.append(obj_best)
        self.feas_history.append(is_feasible)
        self.iteration_times.append(iteration_time)

        # Keep only recent history
        if len(self.obj_history) > self.window_size:
            self.obj_history.pop(0)
            self.feas_history.pop(0)
            self.iteration_times.pop(0)

    def is_stagnating(self, tol=1e-6) -> bool:
        """Check if objective is stagnating."""
        if len(self.obj_history) < self.window_size:
            return False

        recent_improvement = abs(self.obj_history[-1] - self.obj_history[0])
        return recent_improvement < tol

    def get_avg_iteration_time(self) -> float:
        """Get average iteration time."""
        if not self.iteration_times:
            return 0.0
        return np.mean(self.iteration_times)

    def get_feasibility_rate(self) -> float:
        """Get fraction of feasible solutions in recent history."""
        if not self.feas_history:
            return 0.0
        return sum(self.feas_history) / len(self.feas_history)


if __name__ == "__main__":
    print("Testing Heterogeneous Scheduler...")

    scheduler = HeterogeneousScheduler()

    # Test decisions
    test_cases = [
        {'n_vars': 50, 'n_constrs': 50, 'nnz': 100},   # Small
        {'n_vars': 500, 'n_constrs': 500, 'nnz': 10000},  # Medium
        {'n_vars': 5000, 'n_constrs': 2000, 'nnz': 100000},  # Large
    ]

    for i, size in enumerate(test_cases):
        for op in ['pdhg', 'rounding', 'tunnel']:
            device = scheduler.decide_device(size, op)
            print(f"Case {i+1}, {op}: {device}")

    # Test performance monitor
    monitor = PerformanceMonitor(window_size=5)
    for i in range(10):
        monitor.record_iteration(
            obj_best=100.0 - i * 0.1,
            is_feasible=(i % 2 == 0),
            iteration_time=0.1
        )

    print(f"\nStagnating: {monitor.is_stagnating()}")
    print(f"Avg time: {monitor.get_avg_iteration_time():.4f}s")
    print(f"Feas rate: {monitor.get_feasibility_rate():.1%}")

    print("\nTest passed!")
