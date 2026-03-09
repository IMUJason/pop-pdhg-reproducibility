"""
Comprehensive Solution Monitor for Plan1 Experiments

Integrates:
1. Hardware monitoring (CPU/GPU/Memory/Power)
2. Device selection logging
3. Performance statistics
4. Iteration-by-iteration tracking
5. GPU hardware specs (cores, frequency, bandwidth)

Usage:
    from solution_monitor import SolutionMonitor

    monitor = SolutionMonitor(output_dir="experiments/monitoring")
    monitor.start_experiment("instance_name", "method_name")

    # During solve:
    monitor.record_iteration(iter_num, device_used, time_ms, obj_value)
    monitor.record_hardware_snapshot()

    # After solve:
    report = monitor.finish_experiment()
    report.save()
"""

import time
import json
import threading
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import subprocess

# Import enhanced hardware monitor
try:
    from .enhanced_hardware_monitor import EnhancedHardwareMonitor, AppleSiliconQuery
    HAS_ENHANCED_HW = True
except ImportError:
    HAS_ENHANCED_HW = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    import scipy.sparse as sp
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IterationRecord:
    """Single iteration record."""
    iter_num: int
    timestamp: float  # seconds since experiment start
    device: str  # 'cpu' or 'gpu'
    device_reason: str
    spmv_time_ms: float
    total_iter_time_ms: float
    objective_value: float = 0.0
    primal_residual: float = 0.0
    dual_residual: float = 0.0

    # Hardware snapshot (optional)
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    gpu_device_util: float = 0.0
    gpu_memory_mb: float = 0.0
    power_watts: float = 0.0


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    instance_name: str
    method_name: str
    start_time: str
    end_time: str
    total_duration_sec: float
    total_iterations: int

    # Device usage
    cpu_iterations: int = 0
    gpu_iterations: int = 0
    gpu_ratio: float = 0.0

    # Performance
    avg_spmv_time_ms: float = 0.0
    avg_iter_time_ms: float = 0.0
    max_spmv_time_ms: float = 0.0
    min_spmv_time_ms: float = 0.0

    # Hardware averages
    avg_cpu_percent: float = 0.0
    avg_memory_gb: float = 0.0
    avg_gpu_device_util: float = 0.0
    avg_power_watts: float = 0.0

    # Convergence
    final_objective: float = 0.0
    final_primal_residual: float = 0.0
    final_dual_residual: float = 0.0

    # Platform info
    platform_info: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HardwareSnapshot:
    """Single hardware measurement."""
    timestamp: float
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    memory_bandwidth_gbps: float = 0.0  # Real-time estimate
    gpu_device_util: float = 0.0
    gpu_renderer_util: float = 0.0
    gpu_tiler_util: float = 0.0
    gpu_memory_mb: float = 0.0
    power_watts: float = 0.0
    energy_joules: float = 0.0

    # GPU Hardware specs (static)
    gpu_core_count: int = 0
    gpu_frequency_mhz: int = 0
    theoretical_bandwidth_gbps: float = 0.0


# ============================================================================
# Monitor Classes
# ============================================================================

class SolutionMonitor:
    """
    Comprehensive solution monitor for Plan1 experiments.

    Features:
    1. Iteration-by-iteration tracking
    2. Hardware monitoring (background thread)
    3. Device selection logging
    4. Performance statistics
    5. Automatic report generation
    """

    def __init__(self, output_dir: str = "experiments/monitoring",
                 sample_interval: float = 1.0,
                 verbose: bool = True):
        """
        Initialize solution monitor.

        Args:
            output_dir: Directory for output files
            sample_interval: Hardware sampling interval (seconds)
            verbose: Print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.sample_interval = sample_interval

        # State
        self._running = False
        self._experiment_active = False
        self._start_time = 0.0
        self._experiment_name = ""
        self._method_name = ""

        # Data storage
        self._iterations: List[IterationRecord] = []
        self._hardware_samples: List[HardwareSnapshot] = []
        self._device_decisions: List[Dict] = []

        # Background monitoring
        self._monitor_thread: Optional[threading.Thread] = None

        # Platform info and hardware specs
        self._platform_info = self._get_platform_info()
        self._hw_specs = self._get_gpu_hardware_specs()

        if verbose:
            print(f"SolutionMonitor initialized: {self.output_dir}")
            print(f"  Platform: {self._platform_info.get('cpu', 'Unknown')}")
            if self._hw_specs:
                print(f"  GPU Cores: {self._hw_specs.get('gpu_core_count', 'N/A')}")
                print(f"  GPU Frequency: {self._hw_specs.get('gpu_frequency_mhz', 'N/A')} MHz")
                print(f"  Memory Bandwidth: {self._hw_specs.get('theoretical_bandwidth_gbps', 'N/A')} GB/s")

    def _get_platform_info(self) -> Dict:
        """Get platform information."""
        info = {
            'system': 'unknown',
            'machine': 'unknown',
            'cpu': 'unknown',
            'timestamp': datetime.now().isoformat()
        }

        import platform
        info['system'] = platform.system()
        info['machine'] = platform.machine()

        if info['system'] == 'Darwin':
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    info['cpu'] = result.stdout.strip()
            except:
                pass

        return info

    def _get_gpu_hardware_specs(self) -> Dict:
        """Get GPU hardware specifications."""
        if HAS_ENHANCED_HW:
            return AppleSiliconQuery.get_full_specs()
        return {}

    def _get_platform_info(self) -> Dict:
        """Get platform information."""
        info = {
            'system': 'unknown',
            'machine': 'unknown',
            'cpu': 'unknown',
            'timestamp': datetime.now().isoformat()
        }

        import platform
        info['system'] = platform.system()
        info['machine'] = platform.machine()

        if info['system'] == 'Darwin':
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    info['cpu'] = result.stdout.strip()

                result = subprocess.run(
                    ['sysctl', '-n', 'hw.gpucount'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    info['gpu_cores'] = int(result.stdout.strip())
            except:
                pass

        return info

    def _get_hardware_snapshot(self) -> HardwareSnapshot:
        """Collect current hardware metrics."""
        snapshot = HardwareSnapshot(
            timestamp=time.perf_counter() - self._start_time
        )

        # Add GPU hardware specs (static)
        if self._hw_specs:
            snapshot.gpu_core_count = self._hw_specs.get('gpu_cores', 0)
            snapshot.gpu_frequency_mhz = self._hw_specs.get('gpu_frequency_mhz', 0)
            snapshot.theoretical_bandwidth_gbps = self._hw_specs.get('memory_bandwidth_gbps', 0)

        if HAS_PSUTIL:
            snapshot.cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            snapshot.memory_used_gb = mem.used / (1024**3)
            snapshot.memory_percent = mem.percent

        # GPU metrics (macOS only via ioreg)
        if self._platform_info.get('system') == 'Darwin':
            try:
                result = subprocess.run(
                    ['ioreg', '-l'],
                    capture_output=True, text=True, timeout=1
                )
                output = result.stdout

                # Extract GPU metrics
                if 'PerformanceStatistics' in output:
                    # Parse GPU utilization
                    match = re.search(r'"Device Utilization %"=(\d+)', output)
                    if match:
                        snapshot.gpu_device_util = float(match.group(1))

                    match = re.search(r'"Renderer Utilization %"=(\d+)', output)
                    if match:
                        snapshot.gpu_renderer_util = float(match.group(1))

                    match = re.search(r'"Tiler Utilization %"=(\d+)', output)
                    if match:
                        snapshot.gpu_tiler_util = float(match.group(1))

                    match = re.search(r'"In use system memory"=(\d+)', output)
                    if match:
                        snapshot.gpu_memory_mb = float(match.group(1)) / (1024**2)
            except:
                pass

        # Power estimation (use EnhancedHardwareMonitor if available)
        if HAS_ENHANCED_HW:
            from .enhanced_hardware_monitor import PowerEstimator
            snapshot.power_watts = PowerEstimator.estimate_power(
                snapshot.cpu_percent,
                snapshot.gpu_device_util,
                snapshot.memory_bandwidth_gbps
            )
        else:
            # Simple estimation
            snapshot.power_watts = 5.0 + (snapshot.cpu_percent / 100.0) * 20.0

        return snapshot

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running and self._experiment_active:
            snapshot = self._get_hardware_snapshot()
            self._hardware_samples.append(snapshot)
            time.sleep(self.sample_interval)

    def start_experiment(self, instance_name: str, method_name: str):
        """
        Start a new experiment.

        Args:
            instance_name: Name of test instance
            method_name: Name of method being tested
        """
        self._experiment_name = instance_name
        self._method_name = method_name
        self._start_time = time.perf_counter()
        self._experiment_active = True
        self._running = True
        self._iterations = []
        self._hardware_samples = []
        self._device_decisions = []

        # Start background monitoring
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        if self.verbose:
            print(f"\n[SolutionMonitor] Started: {instance_name} / {method_name}")

    def record_iteration(self, iter_num: int, device: str, device_reason: str,
                        spmv_time_ms: float, total_iter_time_ms: float,
                        objective: float = 0.0, primal_residual: float = 0.0,
                        dual_residual: float = 0.0):
        """Record a single iteration."""
        record = IterationRecord(
            iter_num=iter_num,
            timestamp=time.perf_counter() - self._start_time,
            device=device,
            device_reason=device_reason,
            spmv_time_ms=spmv_time_ms,
            total_iter_time_ms=total_iter_time_ms,
            objective_value=objective,
            primal_residual=primal_residual,
            dual_residual=dual_residual
        )

        # Add latest hardware snapshot
        if self._hardware_samples:
            latest_hw = self._hardware_samples[-1]
            record.cpu_percent = latest_hw.cpu_percent
            record.memory_used_gb = latest_hw.memory_used_gb
            record.gpu_device_util = latest_hw.gpu_device_util
            record.gpu_memory_mb = latest_hw.gpu_memory_mb
            record.power_watts = latest_hw.power_watts

        self._iterations.append(record)

        # Track device decision
        self._device_decisions.append({
            'iter': iter_num,
            'device': device,
            'reason': device_reason
        })

    def record_device_selection(self, iter_num: int, device: str, reason: str,
                               cpu_time: float = 0.0, gpu_time: float = 0.0):
        """Record a device selection decision."""
        self._device_decisions.append({
            'iter': iter_num,
            'device': device,
            'reason': reason,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time
        })

    def finish_experiment(self) -> ExperimentSummary:
        """
        Finish experiment and generate summary.

        Returns:
            ExperimentSummary object
        """
        self._running = False
        self._experiment_active = False

        # Wait for monitor thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        end_time = time.perf_counter()
        duration = end_time - self._start_time

        # Generate summary
        summary = ExperimentSummary(
            instance_name=self._experiment_name,
            method_name=self._method_name,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_duration_sec=duration,
            total_iterations=len(self._iterations),
            platform_info=self._platform_info
        )

        if self._iterations:
            # Device usage
            summary.gpu_iterations = sum(1 for i in self._iterations if i.device == 'gpu')
            summary.cpu_iterations = len(self._iterations) - summary.gpu_iterations
            summary.gpu_ratio = summary.gpu_iterations / len(self._iterations)

            # Performance
            spmv_times = [i.spmv_time_ms for i in self._iterations]
            iter_times = [i.total_iter_time_ms for i in self._iterations]

            summary.avg_spmv_time_ms = np.mean(spmv_times)
            summary.avg_iter_time_ms = np.mean(iter_times)
            summary.max_spmv_time_ms = np.max(spmv_times)
            summary.min_spmv_time_ms = np.min(spmv_times)

            # Hardware averages
            if HAS_PSUTIL:
                summary.avg_cpu_percent = np.mean([i.cpu_percent for i in self._iterations])
                summary.avg_memory_gb = np.mean([i.memory_used_gb for i in self._iterations])

            # Convergence
            last = self._iterations[-1]
            summary.final_objective = last.objective_value
            summary.final_primal_residual = last.primal_residual
            summary.final_dual_residual = last.dual_residual

        if self.verbose:
            print(f"\n[SolutionMonitor] Finished: {self._experiment_name}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Iterations: {len(self._iterations)}")
            print(f"  GPU ratio: {summary.gpu_ratio:.1%}")

        return summary

    def save_report(self, summary: ExperimentSummary, filename_prefix: str = None):
        """Save monitoring report to files."""
        if filename_prefix is None:
            filename_prefix = f"{self._experiment_name}_{self._method_name}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.output_dir / f"{filename_prefix}_{timestamp}"

        # Save summary
        summary_path = base_path.with_suffix('.summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        # Save iteration details
        iter_path = base_path.with_suffix('.iterations.json')
        iter_data = [asdict(i) for i in self._iterations]
        with open(iter_path, 'w') as f:
            json.dump(iter_data, f, indent=2)

        # Save device decisions
        device_path = base_path.with_suffix('.devices.json')
        with open(device_path, 'w') as f:
            json.dump(self._device_decisions, f, indent=2)

        # Save hardware samples
        hw_path = base_path.with_suffix('.hardware.json')
        hw_data = [asdict(s) for s in self._hardware_samples]
        with open(hw_path, 'w') as f:
            json.dump(hw_data, f, indent=2)

        # Save markdown report
        self._save_markdown_report(summary, base_path.with_suffix('.md'))

        if self.verbose:
            print(f"\n[SolutionMonitor] Reports saved:")
            print(f"  {summary_path}")

    def _save_markdown_report(self, summary: ExperimentSummary, path: Path):
        """Save markdown summary report."""
        report = f"""# Solution Monitor Report

## Experiment Info
- **Instance**: {summary.instance_name}
- **Method**: {summary.method_name}
- **Start**: {summary.start_time}
- **End**: {summary.end_time}
- **Duration**: {summary.total_duration_sec:.2f}s
- **Platform**: {summary.platform_info.get('cpu', 'Unknown')}

## Device Usage
| Metric | Value |
|--------|-------|
| Total Iterations | {summary.total_iterations} |
| CPU Iterations | {summary.cpu_iterations} |
| GPU Iterations | {summary.gpu_iterations} |
| GPU Ratio | {summary.gpu_ratio:.1%} |

## Performance
| Metric | Value |
|--------|-------|
| Avg SpMV Time | {summary.avg_spmv_time_ms:.3f} ms |
| Avg Iter Time | {summary.avg_iter_time_ms:.3f} ms |
| Max SpMV Time | {summary.max_spmv_time_ms:.3f} ms |
| Min SpMV Time | {summary.min_spmv_time_ms:.3f} ms |

## Hardware Usage (Average)
| Metric | Value |
|--------|-------|
| CPU Utilization | {summary.avg_cpu_percent:.1f}% |
| Memory Used | {summary.avg_memory_gb:.2f} GB |
| GPU Device Util | {summary.avg_gpu_device_util:.1f}% |
| Power | {summary.avg_power_watts:.1f} W |

## Convergence
| Metric | Value |
|--------|-------|
| Final Objective | {summary.final_objective:.6f} |
| Final Primal Residual | {summary.final_primal_residual:.6e} |
| Final Dual Residual | {summary.final_dual_residual:.6e} |

---
*Generated by SolutionMonitor*
"""
        with open(path, 'w') as f:
            f.write(report)


# ============================================================================
# Integration Helper
# ============================================================================

class MonitoredPDHG:
    """
    PDHG solver wrapper with automatic monitoring.

    Usage:
        monitor = SolutionMonitor()
        pdhg = MonitoredPDHG(solver, monitor)

        x, y, history = pdhg.solve(A, b, c, max_iter=1000)
    """

    def __init__(self, solver, monitor: SolutionMonitor, instance_name: str = "instance"):
        self.solver = solver
        self.monitor = monitor
        self.instance_name = instance_name

    def solve(self, A, b, c, lb, ub, max_iter=1000, **kwargs):
        """Solve with monitoring."""
        method_name = type(self.solver).__name__
        self.monitor.start_experiment(self.instance_name, method_name)

        # Run solve and record iterations
        history = self.solver.solve(A, b, c, lb, ub, max_iter=max_iter,
                                    callback=self._iteration_callback)

        summary = self.monitor.finish_experiment()
        self.monitor.save_report(summary)

        return history

    def _iteration_callback(self, iter_num, x, y, obj, residuals, device_info):
        """Called at each iteration to record data."""
        self.monitor.record_iteration(
            iter_num=iter_num,
            device=device_info.get('device', 'cpu'),
            device_reason=device_info.get('reason', ''),
            spmv_time_ms=device_info.get('spmv_time_ms', 0),
            total_iter_time_ms=device_info.get('iter_time_ms', 0),
            objective=obj,
            primal_residual=residuals.get('primal', 0),
            dual_residual=residuals.get('dual', 0)
        )


if __name__ == "__main__":
    """Test SolutionMonitor"""
    print("="*60)
    print("SolutionMonitor Test")
    print("="*60)

    monitor = SolutionMonitor(verbose=True)

    # Simulate experiment
    monitor.start_experiment("test_instance", "test_method")

    # Simulate iterations
    for i in range(10):
        device = 'gpu' if i % 3 == 0 else 'cpu'
        monitor.record_iteration(
            iter_num=i,
            device=device,
            device_reason=f"test_{device}",
            spmv_time_ms=0.5 if device == 'gpu' else 0.3,
            total_iter_time_ms=1.0,
            objective=100 - i * 5,
            primal_residual=1e-3 / (i + 1),
            dual_residual=1e-3 / (i + 1)
        )
        time.sleep(0.5)

    summary = monitor.finish_experiment()
    monitor.save_report(summary)

    print("\n✓ Test complete!")
