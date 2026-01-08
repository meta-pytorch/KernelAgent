"""Benchmarking system for kernel performance measurement.

Simplified structure with just 3 files:
- benchmark.py: Main Benchmark class and BenchmarkLockManager
- timing.py: All utilities (timing + model/kernel loading)
- kernel_subprocess.py: Subprocess runner for kernel isolation
"""

# Core benchmarking
from .benchmark import Benchmark, BenchmarkLockManager

# All utilities from timing module
from .timing import (
    # Model/kernel loading
    CompilationError,
    import_module,
    load_kernel_function,
    load_problem_interface,
    prepare_inputs,
    prepare_pytorch_model,
    # Timing functions
    clear_l2_cache,
    compute_timing_stats,
    time_with_cuda_events,
    time_with_triton_do_bench,
)

__all__ = [
    # Core benchmarking
    "Benchmark",
    "BenchmarkLockManager",
    # Model/kernel loading
    "CompilationError",
    "import_module",
    "load_kernel_function",
    "load_problem_interface",
    "prepare_inputs",
    "prepare_pytorch_model",
    # Timing utilities
    "clear_l2_cache",
    "compute_timing_stats",
    "time_with_cuda_events",
    "time_with_triton_do_bench",
]
