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


# =====================================================================
# Inner-worker component interfaces
#
# These define the seams *inside* each optimization worker where
# platform-specific profiling, analysis, and hardware description enter.
# They can be injected into ``OptimizationWorker`` via its constructor
# (which propagates through ``worker_kwargs`` from the manager level).
# =====================================================================


class AcceleratorSpecsProvider(ABC):
    """Provides hardware specifications for the target accelerator."""

    @abstractmethod
    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        """Return hardware specs for the named device.

        Args:
            device_name: Device identifier (e.g. ``"NVIDIA H100 NVL 94GB"``).
                         ``None`` means auto-detect.

        Returns:
            Dict with keys like ``name``, ``architecture``,
            ``peak_fp32_tflops``, ``peak_memory_bw_gbps``, ``sm_count``, etc.
        """
        ...


class KernelProfilerBase(ABC):
    """Profiles a compiled kernel to collect hardware performance counters."""

    @abstractmethod
    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        """Profile a kernel and return results.

        The return value should be compatible with the
        ``ProfilerResults`` protocol (has ``.metrics`` dict attribute).

        Args:
            kernel_file: Path to the kernel source file.
            problem_file: Path to ``problem.py``.
            round_num: Current optimisation round (used for file naming).
            max_retries: Maximum retry attempts on transient failures.

        Returns:
            Profiler results object, or ``None`` if profiling failed.
        """
        ...


class RooflineAnalyzerBase(ABC):
    """Classifies a kernel as memory-bound, compute-bound, or underutilised."""

    @abstractmethod
    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        """Analyse profiler metrics and return a roofline result.

        The return value should be compatible with ``RooflineResult``
        (has ``efficiency_pct``, ``compute_sol_pct``, ``memory_sol_pct``,
        ``bottleneck``, ``at_roofline``, ``headroom_pct``,
        ``uses_tensor_cores``, ``to_dict()``).
        """
        ...

    @abstractmethod
    def should_stop(self, result: Any) -> tuple[bool, str]:
        """Whether optimisation should terminate (at roofline or converged).

        Returns:
            ``(stop, reason)`` tuple.
        """
        ...

    @abstractmethod
    def reset_history(self) -> None:
        """Reset convergence tracking for a new optimisation run."""
        ...


class BottleneckAnalyzerBase(ABC):
    """Diagnoses performance bottlenecks from profiler metrics."""

    @abstractmethod
    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        """Analyse kernel bottlenecks.

        Returns:
            List of ``BottleneckResult``-compatible objects (each has
            ``category``, ``summary``, ``reasoning``, ``root_causes``,
            ``recommended_fixes``, ``to_dict()``).
            Empty list if analysis fails.
        """
        ...


class RAGPrescriberBase(ABC):
    """Retrieves optimisation patterns from a knowledge base."""

    @abstractmethod
    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        """Embed *query* and find the closest optimisation node.

        Returns:
            ``(opt_node_or_None, similarity_scores)``
        """
        ...

    @abstractmethod
    def build_context(self, opt_node: Any) -> str:
        """Build an LLM-consumable context string from *opt_node*.

        Returns:
            Technique descriptions and code examples as a string.
        """
        ...
