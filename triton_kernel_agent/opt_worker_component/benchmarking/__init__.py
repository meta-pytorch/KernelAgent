# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarking infrastructure for kernel performance measurement."""

# Core benchmarking
from .benchmark import Benchmark, BenchmarkLockManager

# All utilities from timing module
from .timing import (
    # Timing functions
    clear_l2_cache,
    # Model/kernel loading
    CompilationError,
    compute_timing_stats,
    import_module,
    load_kernel_function,
    load_problem_interface,
    prepare_inputs,
    prepare_pytorch_model,
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
