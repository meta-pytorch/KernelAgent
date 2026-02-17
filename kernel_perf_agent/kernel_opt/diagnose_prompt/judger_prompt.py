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
Bottleneck Analysis Prompt Builder

Provides prompt templates and parsing utilities for LLM-based bottleneck analysis
of NCU profiling metrics.

Bottleneck Categories:
- memory: Memory bandwidth is the limiting factor
- compute: Compute throughput is the limiting factor
- underutilized: Neither saturated (<60% both), indicating stalls/occupancy issues

Metric definitions are in metric_schema.py.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.metric_schema import (
    GPU_MEMORY_FIELDS,
    GPU_SPEC_FIELDS,
    NCU_METRIC_SECTIONS,
)
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineResult

BOTTLENECK_CATEGORIES = {"memory", "compute", "underutilized"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BottleneckResult:
    """A single bottleneck analysis."""

    category: str
    summary: str
    reasoning: str
    root_causes: list[dict[str, Any]] = field(default_factory=list)
    recommended_fixes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "summary": self.summary,
            "reasoning": self.reasoning,
            "root_causes": self.root_causes,
            "recommended_fixes": self.recommended_fixes,
        }


# =============================================================================
# Prompt Template
# =============================================================================


BOTTLENECK_PROMPT = """\
You are a GPU performance expert analyzing Triton kernel profiling data.

## Task
Analyze the NCU metrics and identify {num_bottlenecks} performance bottleneck(s). For each, classify as:
- **memory**: Memory bandwidth is the limiting factor
- **compute**: Compute throughput is the limiting factor
- **underutilized**: Neither saturated (<60% both), indicating stalls/occupancy issues

## GPU Specifications
{gpu_specs}

## Roofline Analysis
- Bottleneck: {roofline_bottleneck}
- Compute SOL: {compute_sol:.1f}%
- Memory SOL: {memory_sol:.1f}%
- Efficiency: {efficiency:.1f}%
- Headroom: {headroom:.1f}%
- At Roofline: {at_roofline}
- Tensor Cores: {uses_tc}
- Warnings: {roofline_warnings}

## NCU Metrics
{ncu_metrics}

## Kernel Code
```python
{kernel_code}
```

## Output (JSON array, no markdown fence)
[
    {{
        "category": "memory" | "compute" | "underutilized",
        "summary": "One-line summary",
        "reasoning": "Explanation citing metrics",
        "root_causes": [
            {{
                "cause": "Description",
                "evidence": [{{"metric": "name", "value": 0.0, "interpretation": "meaning"}}],
                "fixes": [
                    {{"fix": "Actionable instruction", "rationale": "Why"}}
                ]
            }}
        ]
    }}
]

Requirements:
- Provide exactly {num_bottlenecks} bottleneck analysis object(s) in the array.
- Order by importance (most critical first).
- Each bottleneck should have exactly {num_causes} root cause(s), each with {num_fixes} fix(es).
- Keep summaries and reasoning concise and grounded in the provided metrics.
"""


# =============================================================================
# Prompt Building
# =============================================================================


def _fmt_value(v: Any) -> str:
    """Format a value for display in prompts."""
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _format_gpu_specs(gpu_specs: dict[str, Any]) -> str:
    """Format GPU specifications using metric_schema definitions."""
    lines = []

    for label, key, unit in GPU_SPEC_FIELDS:
        value = gpu_specs.get(key)
        if value is not None:
            lines.append(f"- {label}: {_fmt_value(value)}{unit}")

    for label, size_key, type_key, unit in GPU_MEMORY_FIELDS:
        size = gpu_specs.get(size_key)
        mem_type = gpu_specs.get(type_key, "")
        if size is not None:
            type_str = f" {mem_type}" if mem_type else ""
            lines.append(f"- {label}: {_fmt_value(size)}{unit}{type_str}")

    return "\n".join(lines) if lines else "N/A"


def _format_ncu_metrics(ncu_metrics: dict[str, Any]) -> str:
    """Format NCU metrics grouped by section using metric_schema definitions."""
    lines = []

    for section_name, metric_defs in NCU_METRIC_SECTIONS.items():
        section_lines = []
        for label, key, unit in metric_defs:
            value = ncu_metrics.get(key)
            if value is not None:
                section_lines.append(f"  - {label}: {_fmt_value(value)}{unit}")

        if section_lines:
            lines.append(f"### {section_name}")
            lines.extend(section_lines)

    schema_keys = {key for _, key, _ in sum(NCU_METRIC_SECTIONS.values(), [])}
    other_keys = sorted(set(ncu_metrics.keys()) - schema_keys)
    if other_keys:
        lines.append("### Other Metrics")
        for key in other_keys:
            value = ncu_metrics[key]
            lines.append(f"  - {key}: {_fmt_value(value)}")

    return "\n".join(lines) if lines else "N/A"


def build_bottleneck_prompt(
    kernel_code: str,
    ncu_metrics: dict[str, Any],
    roofline: RooflineResult,
    gpu_specs: dict[str, Any],
    num_bottlenecks: int = 1,
    num_causes: int = 2,
    num_fixes: int = 1,
) -> str:
    """Build the bottleneck analysis prompt for the LLM.

    Args:
        kernel_code: The Triton kernel source code.
        ncu_metrics: NCU profiling metrics dictionary.
        roofline: Roofline analysis result.
        gpu_specs: GPU hardware specifications.
        num_bottlenecks: Number of bottlenecks to request.
        num_causes: Number of root causes per bottleneck.
        num_fixes: Number of recommended fixes per root cause.

    Returns:
        Formatted prompt string for the LLM.
    """
    return BOTTLENECK_PROMPT.format(
        num_bottlenecks=num_bottlenecks,
        num_causes=num_causes,
        num_fixes=num_fixes,
        gpu_specs=_format_gpu_specs(gpu_specs),
        roofline_bottleneck=roofline.bottleneck,
        compute_sol=roofline.compute_sol_pct,
        memory_sol=roofline.memory_sol_pct,
        efficiency=roofline.efficiency_pct,
        headroom=roofline.headroom_pct,
        at_roofline="Yes" if roofline.at_roofline else "No",
        uses_tc="Yes" if roofline.uses_tensor_cores else "No",
        roofline_warnings="; ".join(roofline.warnings) or "None",
        ncu_metrics=_format_ncu_metrics(ncu_metrics),
        kernel_code=kernel_code,
    )


# =============================================================================
# Response Parsing
# =============================================================================


def parse_bottleneck_response(
    response: str,
    fallback_category: str = "underutilized",
) -> list[BottleneckResult]:
    """Parse LLM response into a list of BottleneckResult.

    Args:
        response: Raw LLM response text.
        fallback_category: Category to use if parsing fails.

    Returns:
        List of BottleneckResult. Empty list if parsing fails completely.
    """
    # Try to find JSON array
    array_match = re.search(r"\[[\s\S]*\]", response)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                return _parse_bottleneck_list(data, fallback_category)
        except json.JSONDecodeError:
            pass

    # Fall back to single object
    obj_match = re.search(r"\{[\s\S]*\}", response)
    if obj_match:
        try:
            data = json.loads(obj_match.group())
            if isinstance(data, dict):
                return _parse_bottleneck_list([data], fallback_category)
        except json.JSONDecodeError:
            pass

    return []


def _parse_bottleneck_list(
    items: list[dict[str, Any]],
    fallback_category: str,
) -> list[BottleneckResult]:
    """Parse a list of bottleneck dicts into BottleneckResult objects."""
    results = []
    for item in items:
        category = item.get("category", fallback_category)
        if category not in BOTTLENECK_CATEGORIES:
            category = fallback_category

        # Parse root causes with nested fixes
        root_causes = []
        all_fixes = []
        for rc in item.get("root_causes", []):
            cause_fixes = [
                {"fix": f.get("fix", ""), "rationale": f.get("rationale", "")}
                for f in rc.get("fixes", [])
            ]
            root_causes.append(
                {
                    "cause": rc.get("cause", "Unknown"),
                    "evidence": rc.get("evidence", []),
                    "fixes": cause_fixes,
                }
            )
            all_fixes.extend(cause_fixes)

        # Also check for legacy top-level recommended_fixes
        for f in item.get("recommended_fixes", []):
            fix_entry = {"fix": f.get("fix", ""), "rationale": f.get("rationale", "")}
            if fix_entry not in all_fixes:
                all_fixes.append(fix_entry)

        results.append(
            BottleneckResult(
                category=category,
                summary=item.get("summary", f"{category}-bound"),
                reasoning=item.get("reasoning", ""),
                root_causes=root_causes,
                recommended_fixes=all_fixes,
            )
        )

    return results
