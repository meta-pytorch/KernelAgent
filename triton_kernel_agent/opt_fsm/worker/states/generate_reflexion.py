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

"""GENERATE_REFLEXION state — LLM self-reflection on the optimization attempt."""

from __future__ import annotations

import json
import re

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationAttempt,
    Reflexion,
    extract_triton_config,
)


class GenerateReflexion(State):
    """Generate a reflexion for the current optimization attempt.

    Handles both successful attempts (with full metrics) and failed attempts.
    Populates attempt_history and reflexions on the context.

    Transitions:
        from success path -> UpdateKernels
        from failure path -> WorkerFinalize
    """

    def execute(self, ctx: WorkerContext) -> str:
        attempt = ctx.current_attempt
        if attempt is None:
            return "WorkerFinalize"

        # Determine if this came from the success path (has new_time) or failure path
        is_success_path = (
            attempt.passed_verification
            and ctx.new_time != float("inf")
            and ctx.optimized_kernel is not None
        )

        if is_success_path:
            self._complete_successful_attempt(ctx, attempt)

        # Add attempt to history
        ctx.attempt_history.append(attempt)

        # Generate reflexion
        reflexion = self._generate_reflexion(ctx, attempt)
        if reflexion:
            ctx.reflexions.append(reflexion)

        if is_success_path:
            return "UpdateKernels"
        return "WorkerFinalize"

    def _complete_successful_attempt(
        self, ctx: WorkerContext, attempt: OptimizationAttempt
    ) -> None:
        """Fill in benchmark results for a successful attempt."""
        new_config = extract_triton_config(ctx.optimized_kernel)
        config_changes = {}
        for key in set(ctx.current_config.keys()) | set(new_config.keys()):
            old_val = ctx.current_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                config_changes[key] = f"{old_val}->{new_val}"

        improvement_pct = (
            ((ctx.best_runtime_time - ctx.new_time) / ctx.best_runtime_time * 100)
            if ctx.best_runtime_time > 0
            else 0
        )

        attempt.time_after_ms = ctx.new_time
        attempt.improvement_pct = improvement_pct
        attempt.is_improvement = ctx.new_time < ctx.best_runtime_time
        attempt.passed_verification = True
        attempt.config_changes = config_changes
        ctx.any_verified = True

        if ctx.new_kernel_metrics:
            attempt.compute_sol_pct = ctx.new_kernel_metrics.get(
                "compute_sol_pct", 0.0
            )
            attempt.memory_sol_pct = ctx.new_kernel_metrics.get(
                "memory_sol_pct", 0.0
            )
            attempt.combined_sol_pct = ctx.new_sol

    def _generate_reflexion(
        self, ctx: WorkerContext, attempt: OptimizationAttempt
    ) -> Reflexion | None:
        """Generate reflexion, mirroring OptimizationOrchestrator._generate_reflexion."""
        if not attempt.passed_verification:
            return Reflexion(
                round_num=attempt.round_num,
                root_cause_diagnosed=attempt.root_cause,
                fix_applied=attempt.recommended_fix,
                expected_outcome="Improve performance",
                actual_outcome="Failed verification",
                performance_delta_pct=0.0,
                was_diagnosis_correct=False,
                was_fix_effective=False,
                reasoning=(
                    f"Attempt failed verification: "
                    f"{attempt.error_message[:200] if attempt.error_message else 'Unknown error'}"
                ),
                lessons=["Ensure generated code passes correctness checks"],
                avoid_patterns=[
                    f"Similar approach to round {attempt.round_num} that failed verification"
                ],
                try_patterns=[],
            )

        try:
            reflexion_prompt = ctx.prompt_manager.render_reflexion_prompt(attempt)
            messages = [{"role": "user", "content": reflexion_prompt}]
            response = ctx.provider.get_response(
                ctx.model, messages, max_tokens=2048
            )
            response_text = response.content

            reflexion_file = (
                ctx.artifact_dir / f"round{attempt.round_num:03d}_reflexion.txt"
            )
            with open(reflexion_file, "w") as f:
                f.write(response_text)

            return self._parse_reflexion_response(response_text, attempt)

        except Exception as e:
            ctx.logger.warning(
                f"[{attempt.round_num}] Failed to generate reflexion: {e}"
            )
            return self._fallback_reflexion(attempt)

    @staticmethod
    def _fallback_reflexion(
        attempt: OptimizationAttempt, reasoning: str | None = None
    ) -> Reflexion:
        return Reflexion(
            round_num=attempt.round_num,
            root_cause_diagnosed=attempt.root_cause,
            fix_applied=attempt.recommended_fix,
            expected_outcome="Improve performance by addressing bottleneck",
            actual_outcome="Improved" if attempt.is_improvement else "No improvement",
            performance_delta_pct=attempt.improvement_pct,
            was_diagnosis_correct=attempt.is_improvement,
            was_fix_effective=attempt.is_improvement,
            reasoning=reasoning
            or f"Performance changed by {attempt.improvement_pct:+.1f}%",
            lessons=[],
            avoid_patterns=[],
            try_patterns=[],
        )

    @staticmethod
    def _parse_reflexion_response(
        response_text: str, attempt: OptimizationAttempt
    ) -> Reflexion:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Reflexion(
                    round_num=attempt.round_num,
                    root_cause_diagnosed=attempt.root_cause,
                    fix_applied=attempt.recommended_fix,
                    expected_outcome=data.get(
                        "expected_outcome", "Improve performance"
                    ),
                    actual_outcome=data.get("actual_outcome", ""),
                    performance_delta_pct=attempt.improvement_pct,
                    was_diagnosis_correct=data.get(
                        "was_diagnosis_correct", attempt.is_improvement
                    ),
                    was_fix_effective=data.get(
                        "was_fix_effective", attempt.is_improvement
                    ),
                    reasoning=data.get("reasoning", ""),
                    lessons=data.get("lessons", []),
                    avoid_patterns=data.get("avoid_patterns", []),
                    try_patterns=data.get("try_patterns", []),
                )
            except json.JSONDecodeError:
                pass

        return GenerateReflexion._fallback_reflexion(
            attempt,
            reasoning=(
                f"Applied {attempt.recommended_fix}. "
                f"Performance changed by {attempt.improvement_pct:+.1f}%"
            ),
        )
