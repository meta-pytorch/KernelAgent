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

"""ANALYZE_BOTTLENECK state — LLM-based bottleneck analysis of NCU metrics."""

from __future__ import annotations

import json

from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import BottleneckResult
from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class AnalyzeBottleneck(State):
    """Analyze kernel bottlenecks using LLM and NCU metrics.

    Replicates the second part of OptimizationOrchestrator._profile_and_analyze().

    Transitions:
        no analysis available -> WorkerFinalize (failure)
        analysis ok          -> GenerateKernel
    """

    def execute(self, ctx: WorkerContext) -> str:
        rn = ctx.round_num

        if not ctx.ncu_metrics:
            ctx.logger.warning(f"[{rn}] No NCU metrics, skipping round")
            ctx.bottleneck_results = None
            return "WorkerFinalize"

        # Roofline analysis
        flat_metrics = (
            next(iter(ctx.ncu_metrics.values()), {}) if ctx.ncu_metrics else {}
        )
        roofline_result = ctx.bottleneck_analyzer.roofline.analyze(flat_metrics)
        ctx.roofline_result = roofline_result

        # Bottleneck analysis
        if ctx.bottleneck_override:
            ctx.logger.info(
                f"[{rn}] Using pre-computed bottleneck: "
                f"{ctx.bottleneck_override}-bound (with LLM analysis for details)"
            )
            llm_results = ctx.bottleneck_analyzer.analyze(
                ctx.current_kernel, ctx.ncu_metrics, rn, roofline_result
            )
            if llm_results:
                bottleneck_results = [
                    BottleneckResult(
                        category=ctx.bottleneck_override,
                        summary=f"Pre-computed: {ctx.bottleneck_override}-bound kernel",
                        reasoning=r.reasoning,
                        root_causes=r.root_causes,
                        recommended_fixes=r.recommended_fixes,
                    )
                    for r in llm_results
                ]
            else:
                ctx.logger.warning(
                    f"[{rn}] LLM analysis failed, using empty root_causes/fixes"
                )
                bottleneck_results = [
                    BottleneckResult(
                        category=ctx.bottleneck_override,
                        summary=f"Pre-computed: {ctx.bottleneck_override}-bound kernel",
                        reasoning="Classification based on operation arithmetic intensity",
                        root_causes=[],
                        recommended_fixes=[],
                    )
                ]
        else:
            ctx.logger.info(f"[{rn}] Analyzing bottleneck...")
            bottleneck_results = ctx.bottleneck_analyzer.analyze(
                ctx.current_kernel, ctx.ncu_metrics, rn, roofline_result
            )

        if not bottleneck_results:
            ctx.logger.warning(f"[{rn}] No analysis available, skipping round")
            ctx.bottleneck_results = None
            return "WorkerFinalize"

        # Save strategy
        strategy_file = ctx.artifact_dir / f"round{rn:03d}_strategy.json"
        with open(strategy_file, "w") as f:
            json.dump([r.to_dict() for r in bottleneck_results], f, indent=2)

        ctx.bottleneck_results = bottleneck_results

        # Select primary bottleneck
        if (
            ctx.bottleneck_id is not None
            and len(bottleneck_results) >= ctx.bottleneck_id
        ):
            ctx.primary_bottleneck = bottleneck_results[ctx.bottleneck_id - 1]
        else:
            ctx.primary_bottleneck = bottleneck_results[0]

        return "GenerateKernel"
