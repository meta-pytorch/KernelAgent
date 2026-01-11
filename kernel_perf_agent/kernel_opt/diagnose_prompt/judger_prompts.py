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
"""

from typing import Any, Dict, Optional, Tuple


# System prompt for the Judge LLM (Dual-Bottleneck NCU Analysis)
JUDGE_SYSTEM_PROMPT = """You are a senior GPU performance engineer. Analyze the target GPU spec, the current kernel, and the Nsight Compute (NCU) profiling metrics. Identify EXACTLY TWO DISTINCT bottlenecks from the hardware profiling data, and propose specific optimization methods for each. Be surgical and metrics-driven.

## Bottleneck Categories (NCU Hardware Profiling)

Analyze fundamental resource utilization using NCU profiling data:

- **memory-bound**: DRAM throughput >50% of peak, L1 hit rate <60%, L2 hit rate <70%, memory coalescing <80%, long scoreboard stalls >25%
- **compute-bound**: DRAM throughput <40%, compute/pipe utilization >50%, memory stalls <15%, eligible warps >4/cycle
- **occupancy-limited**: Achieved occupancy <50%, registers/thread >64, shared memory >48KB/block, check launch__occupancy_limit_* for limiter
- **latency-bound**: Total stalls >35%, long scoreboard >20%, short scoreboard >15%, eligible warps <2/cycle, BUT DRAM throughput <50% (latency, not bandwidth)

## Rules (STRICT)

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

    first_kernel = list(ncu_metrics.values())[0] if ncu_metrics else {}

    def get_metric(key: str, default: str = "N/A") -> str:
        val = first_kernel.get(key, default)
        if isinstance(val, (int, float)):
            return f"{val:.2f}"
        return str(val)

    # Build user prompt using list-join pattern (similar to Fuser/prompting.py)
    parts: list[str] = []

    # Problem Description
    parts.append("## Problem Description")
    parts.append("")
    parts.append(problem_description)

    # Current Kernel Code
    parts.append("")
    parts.append("## Current Kernel Code")
    parts.append("")
    parts.append("```python")
    parts.append(kernel_code)
    parts.append("```")

    # GPU Hardware Specifications
    parts.append("")
    parts.append("## GPU Hardware Specifications")
    parts.append("")
    parts.append(f"- **Name:** {gpu_specs.get('name', 'Unknown')}")
    parts.append(f"- **Architecture:** {gpu_specs.get('architecture', 'Unknown')}")
    parts.append(
        f"- **Peak Memory Bandwidth:** {gpu_specs.get('peak_memory_bw_gbps', 'N/A')} GB/s"
    )
    parts.append(
        f"- **Peak FP32 Performance:** {gpu_specs.get('peak_fp32_tflops', 'N/A')} TFLOPS"
    )
    parts.append(
        f"- **Peak FP16 Performance:** {gpu_specs.get('peak_fp16_tflops', 'N/A')} TFLOPS"
    )
    parts.append(f"- **SM Count:** {gpu_specs.get('sm_count', 'N/A')}")
    parts.append(
        f"- **Max Threads per SM:** {gpu_specs.get('max_threads_per_sm', 'N/A')}"
    )
    parts.append(f"- **L1 Cache per SM:** {gpu_specs.get('l1_cache_kb', 'N/A')} KB")
    parts.append(f"- **L2 Cache (Total):** {gpu_specs.get('l2_cache_mb', 'N/A')} MB")
    parts.append(
        f"- **Memory Size:** {gpu_specs.get('memory_gb', 'N/A')} GB {gpu_specs.get('memory_type', '')}"
    )

    # NCU Profiling Metrics
    parts.append("")
    parts.append("## NCU Profiling Metrics")

    # SM & Compute Utilization
    parts.append("")
    parts.append("### SM & Compute Utilization")
    parts.append(f"- **SM Cycles Active:** {get_metric('sm__cycles_active.avg')}")
    parts.append(
        f"- **Warp Active:** {get_metric('sm__warps_active.avg.pct_of_peak_sustained_active')}%"
    )
    parts.append(
        f"- **Total Instructions Executed:** {get_metric('sm__inst_executed.sum')}"
    )
    parts.append(
        f"- **Tensor Core Utilization:** {get_metric('sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active')}%"
    )
    parts.append(
        f"- **Tensor Core Pipeline Active:** {get_metric('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')}%"
    )

    # Memory Bandwidth & Cache
    parts.append("")
    parts.append("### Memory Bandwidth & Cache")
    parts.append(
        f"- **DRAM Throughput:** {get_metric('dram__throughput.avg.pct_of_peak_sustained_elapsed')}%"
    )
    parts.append(
        f"- **DRAM Bandwidth:** {get_metric('dram__bytes.sum.per_second')} bytes/sec"
    )
    parts.append(
        f"- **GPU DRAM Throughput:** {get_metric('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed')}%"
    )
    parts.append(f"- **DRAM Bytes Read:** {get_metric('dram__bytes_read.sum')} bytes")
    parts.append(f"- **DRAM Bytes Write:** {get_metric('dram__bytes_write.sum')} bytes")
    parts.append(
        f"- **L1 Cache Hit Rate:** {get_metric('l1tex__t_sector_hit_rate.pct')}%"
    )
    parts.append(
        f"- **L1 Throughput:** {get_metric('l1tex__throughput.avg.pct_of_peak_sustained_active')}%"
    )
    parts.append(
        f"- **L2 Cache Hit Rate:** {get_metric('lts__t_sector_hit_rate.pct')}%"
    )
    parts.append(
        f"- **L2 Throughput:** {get_metric('lts__throughput.avg.pct_of_peak_sustained_active')}%"
    )

    # Memory Access Patterns
    parts.append("")
    parts.append("### Memory Access Patterns")
    parts.append(
        f"- **Memory Coalescing:** {get_metric('smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct')}%"
    )
    parts.append(
        f"- **Branch Uniformity:** {get_metric('smsp__sass_average_branch_targets_threads_uniform.pct')}%"
    )

    # Occupancy & Resources
    parts.append("")
    parts.append("### Occupancy & Resources")
    parts.append(
        f"- **Occupancy Limited By Blocks:** {get_metric('launch__occupancy_limit_blocks')}"
    )
    parts.append(
        f"- **Occupancy Limited By Registers:** {get_metric('launch__occupancy_limit_registers')}"
    )
    parts.append(
        f"- **Occupancy Limited By Shared Memory:** {get_metric('launch__occupancy_limit_shared_mem')}"
    )
    parts.append(
        f"- **Registers per Thread:** {get_metric('launch__registers_per_thread')}"
    )
    parts.append(
        f"- **Shared Memory per Block:** {get_metric('launch__shared_mem_per_block_allocated')} bytes"
    )

    # Stall Metrics
    parts.append("")
    parts.append("### Stall Metrics (Warp Issue Stalls)")
    parts.append(
        f"- **Short Scoreboard Stalls:** {get_metric('smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct')}%"
    )
    parts.append(
        f"- **Long Scoreboard Stalls:** {get_metric('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct')}%"
    )
    parts.append(
        f"- **Barrier Stalls:** {get_metric('smsp__warp_issue_stalled_barrier_per_warp_active.pct')}%"
    )
    parts.append(
        f"- **Branch Resolving Stalls:** {get_metric('smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct')}%"
    )

    # Task instructions
    parts.append("")
    parts.append("## Your Task")
    parts.append("")
    parts.append(
        "Identify exactly TWO distinct bottlenecks from the NCU profiling metrics above:"
    )
    parts.append("1. **Bottleneck 1 (Primary)**: The highest-impact performance issue")
    parts.append(
        "2. **Bottleneck 2 (Secondary)**: A different category issue that also limits performance"
    )
    parts.append("")
    parts.append(
        "For each bottleneck, cite 3-4 specific metrics that reveal the issue, "
        "and recommend ONE actionable optimization."
    )
    parts.append("")
    parts.append(
        "**Be surgical and metrics-driven.** Return JSON in the format specified in the system prompt."
    )

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

    # Strategy 4: Backward compatibility - single-bottleneck format
    match = re.search(r'\{[^}]*"bottleneck"[^}]*\}', response_text, re.DOTALL)
    if match:
        try:
            old_format = json.loads(match.group(0))
            if "bottleneck" in old_format:
                # Convert old format to dual-bottleneck format
                return {
                    "bottleneck_1": {
                        "category": old_format.get("bottleneck", "unknown"),
                        "root_cause": old_format.get("root_cause", ""),
                        "suggestion": old_format.get("suggestion", ""),
                        "priority_metrics": old_format.get("priority_metrics", []),
                        "expected_improvement": old_format.get(
                            "expected_improvement", ""
                        ),
                    },
                    "bottleneck_2": {
                        "category": "latency-bound",
                        "root_cause": "Secondary bottleneck inferred from single-bottleneck response",
                        "suggestion": "Review stall metrics for additional optimization opportunities",
                        "priority_metrics": [],
                        "expected_improvement": "Requires further profiling analysis",
                    },
                }
        except json.JSONDecodeError:
            pass

    # Strategy 5: Return None if all strategies fail
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
    print("\nThis module provides prompt templates for hardware bottleneck analysis.")
    print("\nExample usage:")
    print(
        """
    from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompts import (
        build_judge_optimization_prompt,
        extract_judge_response,
        validate_judge_response,
    )
    from kernel_perf_agent.kernel_opt.profiler.gpu_specs import get_gpu_specs
    from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import (
        load_ncu_metrics,
        metrics_to_prompt,
    )
    import json

    # Get GPU specs
    gpu_specs = get_gpu_specs()

    # Load NCU metrics
    metrics_df = load_ncu_metrics("ncu_baseline.csv")
    ncu_metrics = json.loads(metrics_to_prompt(metrics_df))

    # Build prompts
    sys_prompt, user_prompt = build_judge_optimization_prompt(
        kernel_code=kernel_code,
        problem_description=problem_description,
        ncu_metrics=ncu_metrics,
        gpu_specs=gpu_specs,
    )

    # Call LLM
    response = llm.call([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ])

    # Extract and validate
    analysis = extract_judge_response(response)
    if analysis and validate_judge_response(analysis):
        print(f"Bottleneck 1: {analysis['bottleneck_1']['category']}")
        print(f"Bottleneck 2: {analysis['bottleneck_2']['category']}")
    """
    )
