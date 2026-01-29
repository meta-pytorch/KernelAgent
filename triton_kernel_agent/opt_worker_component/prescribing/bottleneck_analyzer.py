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

"""Bottleneck analysis using NCU metrics and LLM."""

from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
    ProfilerResults,
)
import logging
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt import (
    build_judge_optimization_prompt,
    extract_judge_response,
)
from triton_kernel_agent.worker_util import (
    _call_llm,
    _save_debug_file,
)
from utils.providers.base import BaseProvider


class BottleneckAnalyzer:
    """Analyzes NCU metrics to identify performance bottlenecks using LLM.

    This class wraps the Judge LLM workflow:
    1. Build prompt from kernel code, problem description, NCU metrics, GPU specs
    2. Call LLM to analyze bottlenecks
    3. Parse response to extract dual-bottleneck analysis
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        gpu_specs: dict[str, Any],
        high_reasoning_effort: bool = True,
        logs_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.model = model
        self.gpu_specs = gpu_specs
        self.high_reasoning_effort = high_reasoning_effort
        self.logs_dir = logs_dir
        self.logger = logger or logging.getLogger(__name__)

    def analyze_bottleneck(
        self,
        kernel_code: str,
        problem_description: str,
        profiler_results: ProfilerResults,
        round_num: int,
    ) -> dict[str, Any] | None:
        """
        Analyze kernel bottlenecks using NCU metrics and Judge LLM.

        Args:
            kernel_code: Current kernel code
            problem_description: Problem description
            ncu_metrics: NCU profiling metrics dictionary
            round_num: Current optimization round number

        Returns:
            Dual-bottleneck analysis dict with bottleneck_1 and bottleneck_2,
            or None if analysis fails
        """
        try:
            ncu_metrics = profiler_results.ncu.metrics

            # Build the judge prompt
            system_prompt, user_prompt = build_judge_optimization_prompt(
                kernel_code=kernel_code,
                problem_description=problem_description,
                ncu_metrics=ncu_metrics,
                gpu_specs=self.gpu_specs,
            )

            # Save prompt for debugging
            if self.logs_dir:
                prompt_content = (
                    "=== SYSTEM PROMPT ===\n"
                    + system_prompt
                    + "\n\n=== USER PROMPT ===\n"
                    + user_prompt
                )
                _save_debug_file(
                    self.logs_dir / f"round{round_num:03d}_judge_prompt.txt",
                    prompt_content,
                    self.logger,
                )

            # Call LLM using shared utility
            # Note: GPT-5 uses reasoning tokens from max_tokens budget, so we need
            # a higher limit to leave room for both reasoning and response content
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response_text = _call_llm(
                provider=self.provider,
                model=self.model,
                messages=messages,
                high_reasoning_effort=self.high_reasoning_effort,
                logger=self.logger,
                max_tokens=40960 if self.model.startswith("gpt-5") else 12288,
            )

            # Save response for debugging
            if self.logs_dir:
                _save_debug_file(
                    self.logs_dir / f"round{round_num:03d}_judge_response.txt",
                    response_text,
                    self.logger,
                )

            # Parse response
            analysis = extract_judge_response(response_text)
            if analysis:
                self.logger.info(
                    f"[{round_num}] Bottleneck analysis complete: "
                    f"primary={analysis.get('bottleneck_1', {}).get('category', 'unknown')}"
                )
                return analysis
            else:
                self.logger.warning(
                    f"[{round_num}] Failed to parse bottleneck analysis. "
                    f"Response saved to: {self.logs_dir / f'round{round_num:03d}_judge_response.txt'}"
                )
                return None

        except Exception as e:
            self.logger.error(f"[{round_num}] Bottleneck analysis failed: {e}")
            return None
