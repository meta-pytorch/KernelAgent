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
Prompt Builder for Hardware Bottleneck Diagnosis

This module provides prompt templates and builder functions for the Judge LLM
that analyzes NCU profiling metrics to identify performance bottlenecks and
provide specific optimization recommendations.

The Judge uses a dual-bottleneck framework based on NCU hardware profiling:
- bottleneck_1 (Primary): Highest-impact performance issue
- bottleneck_2 (Secondary): Different category issue that also limits performance

Both bottlenecks are selected from NCU hardware profiling categories:
- memory-bound
- compute-bound
- occupancy-limited
- latency-bound

Metric definitions are in metric_schema.py and rendering logic is in section_renderers.py.
"""

from typing import Any, Dict, Optional, Tuple

from .section_renderers import (
    create_metric_getter,
    render_gpu_specs,
    render_kernel_code,
    render_ncu_metrics,
    render_problem_description,
    render_task_instructions,
)


# System prompt for the Judge LLM (Dual-Bottleneck NCU Analysis)
JUDGE_SYSTEM_PROMPT = """You are a senior GPU performance engineer. Analyze the target GPU spec, the current kernel, and the Nsight Compute (NCU) profiling metrics. Identify EXACTLY TWO DISTINCT bottlenecks from the hardware profiling data, and propose specific optimization methods for each. Be surgical and metrics-driven.

## Bottleneck Categories (NCU Hardware Profiling)

Analyze fundamental resource utilization using NCU profiling data:

## Bottleneck Categories (Indicators Only)
- **memory-bound**: High DRAM throughput (>60%), low L1/L2 hit rates (<70%), high memory stalls (>30%)
- **compute-bound**: Low DRAM throughput (<40%), high compute utilization (>60%), low memory stalls (<20%)
- **occupancy-limited**: Low warp active (<50%), high register usage (>100/thread), shared memory pressure (>80%)
- **latency-bound**: High total stalls (>40%), memory dependency stalls dominate, long scoreboard stalls

- Return EXACTLY TWO DISTINCT bottlenecks with DIFFERENT categories
- Both bottlenecks must be from: {memory-bound, compute-bound, occupancy-limited, latency-bound}
- For each bottleneck, cite 3-4 specific NCU metric values that reveal the issue
- Propose ONE actionable optimization method per bottleneck
- Keep fields brief; avoid lists of alternatives, disclaimers, or generic advice

## Output Format (JSON - STRICT)

```json
{
  "bottleneck_1": {
    "category": "<memory-bound|compute-bound|occupancy-limited|latency-bound>",
    "root_cause": "<max 50 words: cite 3-4 specific NCU metric values and explain why they limit performance>",
    "suggestion": "<max 50 words: ONE specific optimization with concrete parameters>",
    "priority_metrics": ["<metric_1>", "<metric_2>", "<metric_3>"],
    "expected_improvement": "<max 40 words: concrete metric targets after optimization>"
  },
  "bottleneck_2": {
    "category": "<memory-bound|compute-bound|occupancy-limited|latency-bound>",
    "root_cause": "<max 50 words: cite 3-4 specific NCU metric values and explain why they limit performance>",
    "suggestion": "<max 50 words: ONE specific optimization with concrete parameters>",
    "priority_metrics": ["<metric_1>", "<metric_2>", "<metric_3>"],
    "expected_improvement": "<max 40 words: concrete metric targets after optimization>"
  }
}
```

## Important Notes

- bottleneck_1 is the PRIMARY (highest-impact) issue
- bottleneck_2 is the SECONDARY issue (different category from bottleneck_1)
- They should be independently addressable (fixing one doesn't automatically fix the other)

Follow the Rules exactly. Return JSON in the specified format.
"""


def build_judge_optimization_prompt(
    kernel_code: str,
    problem_description: str,
    ncu_metrics: Dict[str, Any],
    gpu_specs: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build system and user prompts for Judge to analyze bottleneck.

    This function constructs detailed prompts for the Judge LLM that include:
    - The kernel code being analyzed
    - The original problem description
    - Complete NCU profiling metrics
    - GPU hardware specifications

    Args:
        kernel_code: Current Triton kernel code
        problem_description: Original problem description
        ncu_metrics: NCU profiling metrics as a dictionary (from metrics_to_prompt)
        gpu_specs: GPU specifications (from get_gpu_specs)

    Returns:
        Tuple of (system_prompt, user_prompt)

    Example:
        >>> sys_prompt, user_prompt = build_judge_optimization_prompt(
        ...     kernel_code=kernel_code,
        ...     problem_description=problem_desc,
        ...     ncu_metrics=ncu_metrics,
        ...     gpu_specs=gpu_specs,
        ... )
        >>> response = llm.call([
        ...     {"role": "system", "content": sys_prompt},
        ...     {"role": "user", "content": user_prompt}
        ... ])
    """
    if not ncu_metrics:
        raise ValueError("NCU metrics are empty - cannot build judge prompt")

    # Extract first kernel's metrics for the metric getter
    first_kernel = list(ncu_metrics.values())[0] if ncu_metrics else {}
    get_metric_fn = create_metric_getter(first_kernel)

    # Build user prompt using modular section renderers
    parts: list[str] = []

    # Compose sections using renderers
    parts.extend(render_problem_description(problem_description))
    parts.extend(render_kernel_code(kernel_code))
    parts.extend(render_gpu_specs(gpu_specs))
    parts.extend(render_ncu_metrics(ncu_metrics, get_metric_fn))
    parts.extend(render_task_instructions())

    user_prompt = "\n".join(parts)
    return JUDGE_SYSTEM_PROMPT, user_prompt


def extract_judge_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from Judge LLM response.

    This function handles various response formats and provides fallback strategies
    for robust JSON extraction. Expects dual-bottleneck format with bottleneck_1
    and bottleneck_2 fields.

    Args:
        response_text: Raw text response from Judge LLM

    Returns:
        Parsed JSON dictionary with bottleneck_1 and bottleneck_2,
        or None if extraction fails

    Example:
        >>> response = llm.call(judge_prompts)
        >>> analysis = extract_judge_response(response)
        >>> if analysis:
        ...     print(f"Bottleneck 1: {analysis['bottleneck_1']['category']}")
        ...     print(f"Bottleneck 2: {analysis['bottleneck_2']['category']}")
    """
    import json
    import re

    # Strategy 1: Find JSON in code block
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "bottleneck_1" in data and "bottleneck_2" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find first { ... } block with "bottleneck_1" field
    match = re.search(r'\{[^}]*"bottleneck_1"[^}]*\}', response_text, re.DOTALL)
    if match:
        try:
            # Extract the full JSON object (may be nested)
            start_pos = response_text.find("{", match.start())
            brace_count = 0
            end_pos = start_pos

            for i in range(start_pos, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            json_str = response_text[start_pos:end_pos]
            data = json.loads(json_str)
            if "bottleneck_1" in data and "bottleneck_2" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find any JSON object with dual-bottleneck structure
    match = re.search(
        r'\{\s*"bottleneck_1"\s*:\s*\{.*?\}\s*,\s*"bottleneck_2"\s*:\s*\{.*?\}\s*\}',
        response_text,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Return None if all strategies fail
    return None


def validate_judge_response(analysis: Dict[str, Any]) -> bool:
    """
    Validate that Judge response contains required fields for dual-bottleneck format.

    This function validates the dual-bottleneck format with bottleneck_1 and
    bottleneck_2 fields. Both bottlenecks use NCU hardware profiling categories.

    Args:
        analysis: Parsed JSON from Judge response

    Returns:
        True if response is valid, False otherwise

    Example:
        >>> if validate_judge_response(analysis):
        ...     print("Valid dual-bottleneck response!")
        ... else:
        ...     print("Invalid response - missing required fields")
    """
    # Check for dual-bottleneck format
    if "bottleneck_1" in analysis and "bottleneck_2" in analysis:
        return _validate_bottleneck_entry(
            analysis["bottleneck_1"]
        ) and _validate_bottleneck_entry(analysis["bottleneck_2"])

    # Backward compatibility: Check for old single-bottleneck format
    if "bottleneck" in analysis:
        required_fields = [
            "bottleneck",
            "root_cause",
            "suggestion",
            "priority_metrics",
            "expected_improvement",
        ]

        for field in required_fields:
            if field not in analysis:
                return False

        valid_bottlenecks = [
            "memory-bound",
            "compute-bound",
            "occupancy-limited",
            "latency-bound",
        ]
        if analysis["bottleneck"] not in valid_bottlenecks:
            return False

        if not isinstance(analysis["priority_metrics"], list):
            return False

        for field in ["root_cause", "suggestion", "expected_improvement"]:
            if (
                not isinstance(analysis[field], str)
                or len(analysis[field].strip()) < 10
            ):
                return False

        return True

    return False


def _validate_bottleneck_entry(bottleneck: Dict[str, Any]) -> bool:
    """
    Validate a single bottleneck entry (bottleneck_1 or bottleneck_2).

    Both bottlenecks use NCU hardware profiling categories:
    memory-bound, compute-bound, occupancy-limited, latency-bound

    Args:
        bottleneck: Bottleneck dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "category",
        "root_cause",
        "suggestion",
        "priority_metrics",
        "expected_improvement",
    ]

    for field in required_fields:
        if field not in bottleneck:
            return False

    # NCU hardware profiling categories only
    valid_categories = [
        "memory-bound",
        "compute-bound",
        "occupancy-limited",
        "latency-bound",
    ]

    if bottleneck["category"] not in valid_categories:
        return False

    if not isinstance(bottleneck["priority_metrics"], list):
        return False

    for field in ["root_cause", "suggestion", "expected_improvement"]:
        if not isinstance(bottleneck[field], str) or len(bottleneck[field].strip()) < 5:
            return False

    return True


if __name__ == "__main__":
    print("Judge Prompts Module")
    print("=" * 60)
