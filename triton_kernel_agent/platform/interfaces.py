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

"""Abstract interfaces for platform-specific optimization components.

These interfaces define the seams where platform-specific code
(NVIDIA CUDA/NCU, AMD ROCm, Intel XPU, etc.) enters the optimization
manager. Implement these to support a new backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class KernelVerifier(ABC):
    """Verifies kernel correctness before optimization begins."""

    @abstractmethod
    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        """Check that *kernel_code* produces correct results.

        Args:
            kernel_code: Kernel source code to verify.
            problem_file: Path to ``problem.py`` defining Model and get_inputs().
            test_code: Test source code for correctness checking.

        Returns:
            ``True`` if the kernel passes verification.
        """
        ...


class KernelBenchmarker(ABC):
    """Benchmarks kernels and reference implementations."""

    @abstractmethod
    def benchmark_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
    ) -> float:
        """Benchmark a kernel and return its execution time.

        Args:
            kernel_code: Kernel source code to benchmark.
            problem_file: Path to ``problem.py``.

        Returns:
            Execution time in milliseconds, or ``float("inf")`` on failure.
        """
        ...

    @abstractmethod
    def benchmark_reference(
        self,
        problem_file: Path,
    ) -> float:
        """Benchmark the eager reference implementation (e.g. PyTorch eager).

        Args:
            problem_file: Path to ``problem.py``.

        Returns:
            Execution time in milliseconds, or ``float("inf")`` on failure.
        """
        ...

    @abstractmethod
    def benchmark_reference_compiled(
        self,
        problem_file: Path,
    ) -> float:
        """Benchmark the compiler-optimized reference (e.g. ``torch.compile``).

        Args:
            problem_file: Path to ``problem.py``.

        Returns:
            Execution time in milliseconds, or ``float("inf")`` on failure.
        """
        ...


class WorkerRunner(ABC):
    """Spawns and manages optimization workers for a single round."""

    @abstractmethod
    def run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: str,
        pytorch_baseline: float,
        shared_history: list[dict],
        shared_reflexions: list[dict],
    ) -> list[dict[str, Any]]:
        """Run optimization workers for one round and collect results.

        Each *candidate* dict contains at least:
        - ``parent``: a ``ProgramEntry`` with the kernel to optimize from
        - ``bottleneck_id``: which bottleneck direction to explore

        Each result dict should contain at least:
        - ``success``: bool
        - ``worker_id``: int
        - ``kernel_code``: str | None
        - ``time_ms``: float
        - ``parent_id``: str
        - ``attempt``: dict | None  (for shared history)
        - ``reflexion``: dict | None  (for shared reflexions)

        Args:
            candidates: Candidate specs from the search strategy.
            round_num: Current round number.
            problem_file: Path to ``problem.py``.
            test_code: Test code for correctness verification.
            pytorch_baseline: PyTorch eager baseline time in ms.
            shared_history: Recent optimization attempt history.
            shared_reflexions: Recent reflexion entries.

        Returns:
            List of per-worker result dicts.
        """
        ...
