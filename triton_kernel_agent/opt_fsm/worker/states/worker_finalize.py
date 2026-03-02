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

"""WORKER_FINALIZE state — build final worker result and terminate."""

from __future__ import annotations

from dataclasses import asdict

from triton_kernel_agent.opt_fsm.engine import TERMINAL, State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    _get_triton_kernel_metrics,
)


class WorkerFinalize(State):
    """Finalize worker results.

    Profiles the final best kernel, computes performance metrics, and
    writes the result. Replicates OptimizationOrchestrator._finalize_results().

    Transitions:
        always -> TERMINAL
    """

    def execute(self, ctx: WorkerContext) -> str:
        # Profile final best kernel if we had improvements
        if ctx.best_round_num > 0:
            final_kernel_file = (
                ctx.artifact_dir / f"kernel_round_{ctx.best_round_num}.py"
            )
            if final_kernel_file.exists():
                ctx.logger.info(
                    f"Profiling final best kernel (round {ctx.best_round_num})..."
                )
                final_profiler_results = ctx.profiler.profile_kernel(
                    final_kernel_file, ctx.problem_file, ctx.best_round_num
                )
                if final_profiler_results and final_profiler_results.metrics:
                    ctx.best_ncu_metrics = final_profiler_results.metrics
                    final_flat_metrics = _get_triton_kernel_metrics(
                        ctx.best_ncu_metrics
                    )
                    final_roofline = ctx.roofline_analyzer.analyze(
                        ncu_metrics=final_flat_metrics
                    )
                    ctx.logger.info(
                        f"Final roofline (kernel_round_{ctx.best_round_num}): "
                        f"{final_roofline.bottleneck}-bound, "
                        f"{final_roofline.efficiency_pct:.1f}% SOL "
                        f"(Compute: {final_roofline.compute_sol_pct:.1f}%, "
                        f"Memory: {final_roofline.memory_sol_pct:.1f}%)"
                    )

        # Build finalized result
        best_runtime_kernel = ctx.best_runtime_kernel
        best_runtime_time = ctx.best_runtime_time
        best_runtime_sol = ctx.best_runtime_sol

        ctx.logger.info("")
        ctx.logger.info("=" * 80)
        ctx.logger.info("OPTIMIZATION COMPLETE")
        if ctx.early_stop_reason:
            ctx.logger.info(f"   (Early termination: {ctx.early_stop_reason})")
        ctx.logger.info("=" * 80)

        baseline_time = ctx.baseline_results.get("time_ms", float("inf"))
        baseline_speedup = (
            baseline_time / best_runtime_time if best_runtime_time > 0 else 0
        )
        improvement_percent = (
            (baseline_time - best_runtime_time) / baseline_time * 100
            if baseline_time > 0
            else 0
        )

        ctx.logger.info("Final Results - BEST BY RUNTIME:")
        ctx.logger.info(f"   Time: {best_runtime_time:.4f} ms")
        ctx.logger.info(f"   SOL:  {best_runtime_sol:.1f}%")
        ctx.logger.info(f"   Baseline time: {baseline_time:.4f} ms")
        ctx.logger.info(f"   Speedup vs baseline: {baseline_speedup:.2f}x")

        if ctx.best_sol_kernel != best_runtime_kernel:
            ctx.logger.info("")
            ctx.logger.info("BEST BY SOL (different kernel):")
            ctx.logger.info(f"   Time: {ctx.best_sol_time:.4f} ms")
            ctx.logger.info(f"   SOL:  {ctx.best_sol_sol:.1f}%")

        if ctx.pytorch_baseline_time and ctx.pytorch_baseline_time != float("inf"):
            pytorch_speedup = ctx.pytorch_baseline_time / best_runtime_time
            ctx.logger.info(
                f"   PyTorch baseline: {ctx.pytorch_baseline_time:.4f} ms"
            )
            ctx.logger.info(f"   Speedup vs PyTorch: {pytorch_speedup:.2f}x")

        ctx.logger.info(f"   Improvement: {improvement_percent:.1f}%")

        # Save best kernel
        best_kernel_file = ctx.output_dir / "best_kernel.py"
        best_kernel_file.write_text(best_runtime_kernel)

        # Build performance metrics
        perf_metrics = {
            "baseline_time_ms": baseline_time,
            "best_time_ms": best_runtime_time,
            "best_runtime_sol_pct": best_runtime_sol,
            "speedup": baseline_speedup,
            "rounds": ctx.max_opt_rounds,
        }

        if ctx.best_sol_kernel != best_runtime_kernel:
            perf_metrics["best_sol_time_ms"] = ctx.best_sol_time
            perf_metrics["best_sol_sol_pct"] = ctx.best_sol_sol

        if ctx.best_bottleneck_category:
            perf_metrics["bottleneck_addressed"] = ctx.best_bottleneck_category
            perf_metrics["bottleneck_category"] = ctx.best_bottleneck_category

        if ctx.best_ncu_metrics:
            kernel_metrics = next(iter(ctx.best_ncu_metrics.values()), {})
            perf_metrics["memory_throughput"] = kernel_metrics.get(
                "dram__throughput.avg.pct_of_peak_sustained_elapsed"
            )
            perf_metrics["compute_throughput"] = kernel_metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            )

        if ctx.early_stop_reason:
            perf_metrics["early_stop_reason"] = ctx.early_stop_reason

        if ctx.attempt_history:
            perf_metrics["last_attempt"] = asdict(ctx.attempt_history[-1])
        if ctx.reflexions:
            perf_metrics["last_reflexion"] = asdict(ctx.reflexions[-1])

        ctx.success = best_runtime_time != float("inf") and ctx.any_verified
        ctx.result_kernel = best_runtime_kernel
        ctx.result_metrics = perf_metrics

        return TERMINAL
