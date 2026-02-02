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

"""
Bottleneck Analyzer - LLM-based NCU profiling analysis.

This module orchestrates LLM calls for bottleneck analysis using:
- judger_prompt.py: Prompt template, parsing, BottleneckResult dataclass
- ncu_roofline.py: Roofline analysis using NCU SOL metrics

Bottleneck Categories:
- memory: Memory bandwidth is the limiting factor
- compute: Compute throughput is the limiting factor
- underutilized: Neither saturated (<60% both), indicating stalls/occupancy issues
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
    BottleneckResult,
    build_bottleneck_prompt,
    parse_bottleneck_response,
)
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
from triton_kernel_agent.worker_util import _call_llm, _save_debug_file
from utils.providers.base import BaseProvider


class BottleneckAnalyzer:
    """LLM-based bottleneck analyzer using NCU metrics."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        gpu_specs: dict[str, Any],
        logs_dir: Path | None = None,
        logger: logging.Logger | None = None,
        num_bottlenecks: int = 1,
        num_causes: int = 2,
        num_fixes: int = 1,
        enable_debug: bool = True,
    ):
        """
        Initialize bottleneck analyzer.

        Args:
            provider: LLM provider instance
            model: Model name for LLM calls
            gpu_specs: GPU hardware specifications
            logs_dir: Directory for saving debug files
            logger: Logger instance
            num_bottlenecks: Number of bottlenecks to request from LLM
            num_causes: Number of root causes per bottleneck
            num_fixes: Number of recommended fixes per bottleneck
            enable_debug: Whether to save debug files (prompts/responses)
        """
        self.provider = provider
        self.model = model
        self.gpu_specs = gpu_specs
        self.logs_dir = logs_dir
        self.logger = logger or logging.getLogger(__name__)
        self.num_bottlenecks = num_bottlenecks
        self.num_causes = num_causes
        self.num_fixes = num_fixes
        self.enable_debug = enable_debug
        self.roofline = RooflineAnalyzer(logger=logger)

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any = None,
    ) -> list[BottleneckResult]:
        """
        Analyze kernel bottlenecks using LLM.

        Args:
            kernel_code: The Triton kernel source code
            ncu_metrics: NCU profiling metrics dictionary
            round_num: Current optimization round (for logging)
            roofline_result: Pre-computed RooflineResult (if None, computed internally)

        Returns:
            List of BottleneckResult (ordered by importance).
            Empty list if analysis fails.
        """
        if roofline_result is None:
            # Filter out PyTorch kernels (at::*) and get Triton kernel metrics
            if ncu_metrics:
                triton_kernels = {
                    name: metrics
                    for name, metrics in ncu_metrics.items()
                    if not name.startswith("at::") and not name.startswith("void at::")
                }
                flat_metrics = (
                    next(iter(triton_kernels.values()))
                    if triton_kernels
                    else next(iter(ncu_metrics.values()), {})
                )
            else:
                flat_metrics = {}
            roofline_result = self.roofline.analyze(flat_metrics)

        prompt = build_bottleneck_prompt(
            kernel_code=kernel_code,
            ncu_metrics=ncu_metrics,
            roofline=roofline_result,
            gpu_specs=self.gpu_specs,
            num_bottlenecks=self.num_bottlenecks,
            num_causes=self.num_causes,
            num_fixes=self.num_fixes,
        )

        response = _call_llm(
            provider=self.provider,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logger=self.logger,
            max_tokens=16384,
        )

        if self.enable_debug and self.logs_dir:
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_bottleneck_prompt.txt",
                prompt,
                self.logger,
            )
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_bottleneck_response.txt",
                response,
                self.logger,
            )

        results = parse_bottleneck_response(response)

        if results:
            categories = [r.category for r in results]
            self.logger.info(f"[{round_num}] Bottlenecks: {', '.join(categories)}")
        else:
            self.logger.warning(f"[{round_num}] Failed to parse bottleneck response")

        return results
