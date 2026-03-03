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

"""No-op implementations of platform interfaces.

These pass the input kernel through unchanged, making them useful for:
- Dry-run testing of the optimisation pipeline
- CI environments without GPU hardware
- Integration tests that exercise the manager/worker plumbing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RAGPrescriberBase,
    RooflineAnalyzerBase,
    WorkerRunner,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manager-level no-ops
# ---------------------------------------------------------------------------


class NoOpVerifier(KernelVerifier):
    """Always reports the kernel as correct."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        logger.info("[noop] Skipping verification — returning True")
        return True


class NoOpBenchmarker(KernelBenchmarker):
    """Returns a placeholder time (``1.0 ms``) for every benchmark.

    Returning a finite time (rather than ``inf``) ensures the strategy
    records the initial kernel as a valid baseline so the pipeline can
    report success at the end of the run.
    """

    _PLACEHOLDER_MS = 1.0

    def __init__(self, **kwargs: Any) -> None:
        pass

    def benchmark_kernel(self, kernel_code: str, problem_file: Path) -> float:
        logger.info(
            "[noop] Skipping kernel benchmark — returning %.1f ms", self._PLACEHOLDER_MS
        )
        return self._PLACEHOLDER_MS

    def benchmark_reference(self, problem_file: Path) -> float:
        logger.info(
            "[noop] Skipping reference benchmark — returning %.1f ms",
            self._PLACEHOLDER_MS,
        )
        return self._PLACEHOLDER_MS

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        logger.info(
            "[noop] Skipping compiled benchmark — returning %.1f ms",
            self._PLACEHOLDER_MS,
        )
        return self._PLACEHOLDER_MS


class NoOpWorkerRunner(WorkerRunner):
    """Returns the parent kernel unchanged (no actual optimisation).

    Each candidate yields a ``success: True`` result with the parent's
    kernel code and timing, so the strategy sees "no improvement" and
    terminates normally.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

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
        logger.info(
            "[noop] Returning %d candidate(s) unchanged (round %d)",
            len(candidates),
            round_num,
        )
        results: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            parent = candidate["parent"]
            results.append(
                {
                    "success": True,
                    "worker_id": i,
                    "kernel_code": parent.kernel_code,
                    "time_ms": parent.metrics.time_ms,
                    "parent_id": parent.program_id,
                    "attempt": None,
                    "reflexion": None,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Worker-level no-ops
# ---------------------------------------------------------------------------


class NoOpSpecsProvider(AcceleratorSpecsProvider):
    """Returns a minimal stub GPU spec dict."""

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        return {
            "name": device_name or "noop",
            "architecture": "noop",
            "peak_fp32_tflops": 0.0,
            "peak_memory_bw_gbps": 0.0,
            "sm_count": 0,
        }


class NoOpProfiler(KernelProfilerBase):
    """Always reports profiling as failed (returns ``None``)."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        logger.info("[noop] Skipping profiling — returning None")
        return None


class NoOpRooflineAnalyzer(RooflineAnalyzerBase):
    """Reports zero efficiency and always signals stop."""

    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        return {
            "efficiency_pct": 0.0,
            "compute_sol_pct": 0.0,
            "memory_sol_pct": 0.0,
            "bottleneck": "unknown",
            "at_roofline": False,
            "headroom_pct": 100.0,
            "uses_tensor_cores": False,
        }

    def should_stop(self, result: Any) -> tuple[bool, str]:
        return True, "noop roofline — always stop"

    def reset_history(self) -> None:
        pass


class NoOpBottleneckAnalyzer(BottleneckAnalyzerBase):
    """Returns an empty bottleneck list (no diagnosis)."""

    def __init__(self, **kwargs: Any) -> None:
        self.roofline = NoOpRooflineAnalyzer()

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        logger.info("[noop] Skipping bottleneck analysis — returning []")
        return []


class NoOpRAGPrescriber(RAGPrescriberBase):
    """Returns no RAG results."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        return None, []

    def build_context(self, opt_node: Any) -> str:
        return ""
