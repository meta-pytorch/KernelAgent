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

"""PROFILE_FOR_SOL state — NCU profiling for roofline efficiency metrics."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    _get_triton_kernel_metrics,
)


class ProfileSol(State):
    """Profile the optimized kernel to get SOL efficiency metrics.

    Transitions:
        always -> GenerateReflexion
    """

    def execute(self, ctx: WorkerContext) -> str:
        rn = ctx.round_num

        try:
            kernel_file = ctx.artifact_dir / f"kernel_round_{rn}_sol.py"
            kernel_file.write_text(ctx.optimized_kernel)

            profiler_results = ctx.profiler.profile_kernel(
                kernel_file, ctx.problem_file, rn
            )

            if profiler_results is not None and profiler_results.metrics:
                ncu_metrics = profiler_results.metrics
                flat_metrics = _get_triton_kernel_metrics(ncu_metrics)
                roofline_result = ctx.roofline_analyzer.analyze(
                    ncu_metrics=flat_metrics
                )

                ctx.new_kernel_metrics = {
                    "efficiency_pct": roofline_result.efficiency_pct,
                    "compute_sol_pct": roofline_result.compute_sol_pct,
                    "memory_sol_pct": roofline_result.memory_sol_pct,
                    "bottleneck": roofline_result.bottleneck,
                    "roofline_result": roofline_result,
                    "ncu_metrics": ncu_metrics,
                }
                ctx.new_sol = roofline_result.efficiency_pct
            else:
                ctx.new_kernel_metrics = None
                ctx.new_sol = 0.0
        except Exception as e:
            ctx.logger.warning(f"[{rn}] SOL profiling failed: {e}")
            ctx.new_kernel_metrics = None
            ctx.new_sol = 0.0

        return "GenerateReflexion"
