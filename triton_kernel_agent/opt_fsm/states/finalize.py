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

"""FINALIZE state — build final result dict and terminate."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import TERMINAL, State


class Finalize(State):
    """Build the final result dict and terminate the FSM.

    If ctx.result is already populated (e.g., by VerifyInitialKernel on failure),
    this state just logs and returns TERMINAL. Otherwise it builds the result
    from the strategy's best program, matching OptimizationManager.run_optimization().

    Transitions:
        always -> TERMINAL
    """

    def execute(self, ctx: OptimizationContext) -> str:
        # If result was already set (e.g., initial verification failure), just log
        if ctx.result:
            ctx.logger.info("")
            ctx.logger.info("=" * 80)
            ctx.logger.info("OPTIMIZATION COMPLETE")
            ctx.logger.info("=" * 80)
            return TERMINAL

        best = ctx.strategy.get_best_program()

        ctx.logger.info("")
        ctx.logger.info("=" * 80)
        ctx.logger.info("OPTIMIZATION COMPLETE")
        ctx.logger.info("=" * 80)

        if best:
            ctx.logger.info(f"Best time: {best.metrics.time_ms:.4f}ms")
            if (
                ctx.initial_kernel_time_ms != float("inf")
                and best.metrics.time_ms > 0
            ):
                speedup = ctx.initial_kernel_time_ms / best.metrics.time_ms
                ctx.logger.info(f"Speedup vs initial kernel: {speedup:.2f}x")
            if (
                ctx.pytorch_baseline_ms != float("inf")
                and best.metrics.time_ms > 0
            ):
                speedup_pt = ctx.pytorch_baseline_ms / best.metrics.time_ms
                ctx.logger.info(f"Speedup vs PyTorch eager: {speedup_pt:.2f}x")

        ctx.result = {
            "success": best is not None and best.metrics.time_ms != float("inf"),
            "kernel_code": best.kernel_code if best else None,
            "best_time_ms": best.metrics.time_ms if best else float("inf"),
            "total_rounds": ctx.round_num,
            "pytorch_baseline_ms": ctx.pytorch_baseline_ms,
            "pytorch_compile_ms": ctx.pytorch_compile_ms,
            "initial_kernel_time_ms": ctx.initial_kernel_time_ms,
            "top_kernels": [
                {
                    "kernel_code": p.kernel_code,
                    "time_ms": p.metrics.time_ms,
                    "generation": p.generation,
                    "program_id": p.program_id,
                }
                for p in ctx.database.get_top_k(5)
            ],
        }

        return TERMINAL
