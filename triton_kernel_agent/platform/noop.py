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

These pass-through implementations allow ``OptimizationManager`` to run
without any real hardware.  Every component simply prints that it was
called and returns a neutral default so the optimization loop terminates
quickly and returns the input kernel unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.interfaces import (
    KernelBenchmarker,
    KernelVerifier,
    WorkerRunner,
)


class NoOpVerifier(KernelVerifier):
    """Always reports the kernel as correct."""

    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        print(
            "[NoOpVerifier] verify() called — skipping verification, returning True"
        )
        return True


class NoOpBenchmarker(KernelBenchmarker):
    """Returns ``inf`` for every benchmark (no timing hardware available)."""

    def benchmark_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
    ) -> float:
        print("[NoOpBenchmarker] benchmark_kernel() called — returning inf")
        return float("inf")

    def benchmark_reference(
        self,
        problem_file: Path,
    ) -> float:
        print("[NoOpBenchmarker] benchmark_reference() called — returning inf")
        return float("inf")

    def benchmark_reference_compiled(
        self,
        problem_file: Path,
    ) -> float:
        print(
            "[NoOpBenchmarker] benchmark_reference_compiled() called — returning inf"
        )
        return float("inf")


class NoOpWorkerRunner(WorkerRunner):
    """Returns no-improvement results so the strategy terminates immediately."""

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
        print(
            f"[NoOpWorkerRunner] run_workers() called for round {round_num} "
            f"with {len(candidates)} candidate(s) — returning no improvements"
        )
        return [
            {
                "success": False,
                "worker_id": i,
                "kernel_code": candidate["parent"].kernel_code,
                "time_ms": float("inf"),
                "parent_id": candidate["parent"].program_id,
                "attempt": None,
                "reflexion": None,
            }
            for i, candidate in enumerate(candidates)
        ]
