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

"""UPDATE_BEST_KERNELS state — two-kernel tracking and divergence check."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class UpdateKernels(State):
    """Update best-by-runtime and best-by-SOL kernels.

    Implements two-kernel tracking and divergence reversion, replicating
    OptimizationOrchestrator._update_kernels().

    Transitions:
        optimized kernel is best runtime AND at roofline -> CheckEarlyStop
        otherwise -> WorkerFinalize
    """

    def execute(self, ctx: WorkerContext) -> str:
        rn = ctx.round_num
        new_time = ctx.new_time
        new_sol = ctx.new_sol
        optimized_kernel = ctx.optimized_kernel

        # Check for runtime improvement
        if new_time < ctx.best_runtime_time:
            speedup = ctx.best_runtime_time / new_time
            improvement = (
                (ctx.best_runtime_time - new_time) / ctx.best_runtime_time * 100
            )
            ctx.logger.info(
                f"[{rn}] NEW BEST RUNTIME! {new_time:.4f} ms "
                f"(speedup: {speedup:.2f}x, improvement: {improvement:.1f}%)"
            )
            if new_sol > 0:
                ctx.logger.info(f"[{rn}] SOL: {new_sol:.1f}%")
            ctx.best_runtime_kernel = optimized_kernel
            ctx.best_runtime_time = new_time
            ctx.best_runtime_sol = new_sol

        # Check for SOL improvement
        if new_sol > ctx.best_sol_sol:
            ctx.logger.info(
                f"[{rn}] NEW BEST SOL! {ctx.best_sol_sol:.1f}% -> {new_sol:.1f}%"
            )
            ctx.logger.info(
                f"[{rn}]    Runtime: {new_time:.4f} ms "
                f"(best runtime: {ctx.best_runtime_time:.4f} ms)"
            )
            ctx.best_sol_kernel = optimized_kernel
            ctx.best_sol_time = new_time
            ctx.best_sol_sol = new_sol

        # Divergence check
        divergence = (
            (new_time - ctx.best_runtime_time) / ctx.best_runtime_time * 100
            if ctx.best_runtime_time > 0
            else 0
        )

        if divergence > ctx.divergence_threshold:
            ctx.logger.warning(
                f"[{rn}] Excessive divergence ({divergence:.1f}%), "
                f"reverting to best runtime kernel"
            )
            ctx.current_kernel = ctx.best_runtime_kernel
        else:
            if new_time >= ctx.best_runtime_time and new_sol <= ctx.best_sol_sol:
                ctx.logger.info(
                    f"[{rn}] No improvement: {new_time:.4f} ms vs "
                    f"best {ctx.best_runtime_time:.4f} ms"
                )
                if new_sol > 0:
                    ctx.logger.info(
                        f"[{rn}] SOL: {new_sol:.1f}% (best: {ctx.best_sol_sol:.1f}%)"
                    )
            ctx.current_kernel = optimized_kernel

        # Track metadata
        if new_time < ctx.best_runtime_time or new_sol > ctx.best_sol_sol:
            ctx.best_round_num = rn
            ctx.best_bottleneck_category = ctx.primary_bottleneck.category
            if ctx.new_kernel_metrics:
                ctx.best_ncu_metrics = ctx.new_kernel_metrics.get("ncu_metrics")

        # Roofline check
        if ctx.new_kernel_metrics:
            roofline_check = ctx.new_kernel_metrics.get("roofline_result")
            if roofline_check:
                ctx.logger.info(
                    f"[{rn}] Roofline: {roofline_check.bottleneck}-bound, "
                    f"{roofline_check.efficiency_pct:.1f}% SOL "
                    f"(Compute: {roofline_check.compute_sol_pct:.1f}%, "
                    f"Memory: {roofline_check.memory_sol_pct:.1f}%)"
                )

                if (
                    ctx.best_runtime_kernel == optimized_kernel
                    and roofline_check.at_roofline
                ):
                    return "CheckEarlyStop"

        return "WorkerFinalize"
