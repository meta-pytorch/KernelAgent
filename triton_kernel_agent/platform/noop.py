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
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RAGPrescriberBase,
    RooflineAnalyzerBase,
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


# =====================================================================
# Inner-worker no-op components
#
# These replace profiling, analysis, and hardware spec components
# *inside* each optimization worker.  When injected, the LLM-driven
# optimization loop still runs but without hardware profiling data,
# so it falls back to generic optimization or skips profiling-dependent
# steps.
# =====================================================================


class NoOpAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Returns a generic specs dict with no real hardware info."""

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        print(
            f"[NoOpAcceleratorSpecsProvider] get_specs(device_name={device_name!r}) "
            "called — returning generic specs"
        )
        return {
            "name": device_name or "generic",
            "architecture": "unknown",
            "peak_fp32_tflops": 0.0,
            "peak_fp16_tflops": 0.0,
            "peak_bf16_tflops": 0.0,
            "peak_memory_bw_gbps": 0.0,
            "sm_count": 0,
            "max_threads_per_sm": 0,
            "l1_cache_kb": 0,
            "l2_cache_mb": 0,
            "memory_gb": 0,
            "memory_type": "unknown",
        }


class NoOpKernelProfiler(KernelProfilerBase):
    """Always reports profiling failure (returns ``None``)."""

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> None:
        print(
            f"[NoOpKernelProfiler] profile_kernel() called for round {round_num} "
            "— returning None (no profiler available)"
        )
        return None


class NoOpRooflineAnalyzer(RooflineAnalyzerBase):
    """Returns a neutral roofline result; never triggers early stopping."""

    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        print("[NoOpRooflineAnalyzer] analyze() called — returning neutral result")
        from dataclasses import dataclass, field

        @dataclass
        class _StubRooflineResult:
            compute_sol_pct: float = 0.0
            memory_sol_pct: float = 0.0
            efficiency_pct: float = 0.0
            at_roofline: bool = False
            headroom_pct: float = 100.0
            bottleneck: str = "unknown"
            uses_tensor_cores: bool = False
            warnings: list[str] = field(default_factory=lambda: ["no-op analyzer"])

            def to_dict(self) -> dict[str, Any]:
                return {
                    "compute_sol_pct": self.compute_sol_pct,
                    "memory_sol_pct": self.memory_sol_pct,
                    "efficiency_pct": self.efficiency_pct,
                    "at_roofline": self.at_roofline,
                    "headroom_pct": self.headroom_pct,
                    "bottleneck": self.bottleneck,
                    "uses_tensor_cores": self.uses_tensor_cores,
                    "warnings": self.warnings,
                }

        return _StubRooflineResult()

    def should_stop(self, result: Any) -> tuple[bool, str]:
        print("[NoOpRooflineAnalyzer] should_stop() called — returning False")
        return False, ""

    def reset_history(self) -> None:
        print("[NoOpRooflineAnalyzer] reset_history() called")


class NoOpBottleneckAnalyzer(BottleneckAnalyzerBase):
    """Returns an empty bottleneck list so profiling-dependent steps are skipped."""

    def __init__(self) -> None:
        # Orchestrator accesses .roofline on the analyzer for inline roofline
        # analysis, so expose a no-op roofline analyzer too.
        self.roofline = NoOpRooflineAnalyzer()

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        print(
            f"[NoOpBottleneckAnalyzer] analyze() called for round {round_num} "
            "— returning empty list"
        )
        return []


class NoOpRAGPrescriber(RAGPrescriberBase):
    """Returns no retrieval results."""

    def retrieve(self, query: str) -> tuple[None, dict]:
        print(
            f"[NoOpRAGPrescriber] retrieve() called with query={query!r} "
            "— returning (None, {{}})"
        )
        return None, {}

    def build_context(self, opt_node: Any) -> str:
        print("[NoOpRAGPrescriber] build_context() called — returning empty string")
        return ""
