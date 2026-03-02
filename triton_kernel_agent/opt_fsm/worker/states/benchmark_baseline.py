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

"""BENCHMARK_BASELINE state — benchmark baseline kernel and profile SOL."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    _get_triton_kernel_metrics,
)
from triton_kernel_agent.worker_util import _write_kernel_file


class BenchmarkBaseline(State):
    """Benchmark baseline kernel, profile for SOL, and set up PyTorch baseline.

    Replicates OptimizationOrchestrator._benchmark_baseline().

    Transitions:
        always -> ProfileKernel
    """

    def execute(self, ctx: WorkerContext) -> str:
        ctx.logger.info("=" * 80)
        ctx.logger.info("Starting hardware-guided optimization")
        ctx.logger.info("=" * 80)

        # Initialize state
        ctx.current_kernel = ctx.initial_kernel
        ctx.error_feedback = ""
        ctx.roofline_analyzer.reset_history()
        ctx.problem_description = ctx.problem_file.read_text()
        ctx.logger.info(f"Problem: {ctx.problem_description[:100]}...")

        # Benchmark or use known time
        baseline_sol = 0.0
        kernel_file_round = ctx.artifact_dir / "kernel_round_0.py"
        kernel_file_round.write_text(ctx.initial_kernel)

        if ctx.known_kernel_time and ctx.known_kernel_time != float("inf"):
            best_time = ctx.known_kernel_time
            baseline_results = {"time_ms": ctx.known_kernel_time, "speedup": 1.0}
            ctx.logger.info(f"Using known kernel time: {best_time:.4f} ms")
        else:
            _write_kernel_file(ctx.kernel_file, ctx.initial_kernel, ctx.logger)
            baseline_results = ctx.benchmarker.benchmark_kernel(
                kernel_file_round, ctx.problem_file
            )
            best_time = baseline_results["time_ms"]
            ctx.logger.info(f"Baseline time: {best_time:.4f} ms")

        # Profile baseline for SOL
        baseline_metrics = self._profile_kernel_for_sol(ctx, ctx.initial_kernel, 0)
        if baseline_metrics:
            baseline_sol = baseline_metrics.get("efficiency_pct", 0.0)
            bottleneck = baseline_metrics.get("bottleneck", "unknown")
            compute_sol = baseline_metrics.get("compute_sol_pct", 0.0)
            memory_sol = baseline_metrics.get("memory_sol_pct", 0.0)
            ctx.logger.info(
                f"Baseline SOL: {baseline_sol:.1f}% ({bottleneck}-bound, "
                f"Compute: {compute_sol:.1f}%, Memory: {memory_sol:.1f}%)"
            )

        # PyTorch baseline
        if ctx.pytorch_baseline_time is not None:
            pytorch_time = ctx.pytorch_baseline_time
            if pytorch_time == float("inf"):
                pytorch_time = None
            else:
                ctx.logger.info(
                    f"PyTorch baseline: {pytorch_time:.4f} ms (pre-computed)"
                )
        else:
            pytorch_results = ctx.benchmarker.benchmark_pytorch(ctx.problem_file)
            pytorch_time = pytorch_results.get("time_ms", float("inf"))
            if pytorch_time != float("inf"):
                ctx.logger.info(f"PyTorch baseline: {pytorch_time:.4f} ms")
            else:
                pytorch_time = None

        # Store results
        ctx.best_time = best_time
        ctx.baseline_results = baseline_results
        ctx.baseline_sol = baseline_sol
        ctx.pytorch_baseline_time = pytorch_time

        # Initialize two-kernel tracking
        ctx.best_runtime_kernel = ctx.initial_kernel
        ctx.best_runtime_time = best_time
        ctx.best_runtime_sol = baseline_sol
        ctx.best_sol_kernel = ctx.initial_kernel
        ctx.best_sol_time = best_time
        ctx.best_sol_sol = baseline_sol

        return "ProfileKernel"

    @staticmethod
    def _profile_kernel_for_sol(
        ctx: WorkerContext, kernel_code: str, round_num: int
    ) -> dict | None:
        """Profile a kernel to get its SOL metrics."""
        try:
            kernel_file = ctx.artifact_dir / f"kernel_round_{round_num}_sol.py"
            kernel_file.write_text(kernel_code)

            profiler_results = ctx.profiler.profile_kernel(
                kernel_file, ctx.problem_file, round_num
            )

            if profiler_results is None or not profiler_results.metrics:
                return None

            ncu_metrics = profiler_results.metrics
            flat_metrics = _get_triton_kernel_metrics(ncu_metrics)
            roofline_result = ctx.roofline_analyzer.analyze(ncu_metrics=flat_metrics)

            return {
                "efficiency_pct": roofline_result.efficiency_pct,
                "compute_sol_pct": roofline_result.compute_sol_pct,
                "memory_sol_pct": roofline_result.memory_sol_pct,
                "bottleneck": roofline_result.bottleneck,
                "roofline_result": roofline_result,
                "ncu_metrics": ncu_metrics,
            }
        except Exception as e:
            ctx.logger.warning(f"[{round_num}] SOL profiling failed: {e}")
            return None
