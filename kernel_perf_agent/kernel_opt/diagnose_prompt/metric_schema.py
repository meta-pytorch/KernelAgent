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
Metric Schema Definitions for NCU Profiling and GPU Specifications.

This module provides the single source of truth for:
- NCU profiling metric definitions (keys, labels, units)
- GPU specification field definitions

Schema Format: List of tuples (display_label, key, unit_suffix)
- display_label: Human-readable name shown in prompts
- key: NCU metric key or GPU spec dictionary key
- unit_suffix: Unit to append after value (e.g., "%", " GB/s", " bytes")
"""

from typing import Dict, List, Tuple

# Type alias for metric definition: (label, key, unit)
MetricDef = Tuple[str, str, str]

# =============================================================================
# GPU Specification Fields
# =============================================================================

GPU_SPEC_FIELDS: List[MetricDef] = [
    ("Name", "name", ""),
    ("Architecture", "architecture", ""),
    ("Peak Memory Bandwidth", "peak_memory_bw_gbps", " GB/s"),
    ("Peak FP32 Performance", "peak_fp32_tflops", " TFLOPS"),
    ("Peak FP16 Performance", "peak_fp16_tflops", " TFLOPS"),
    ("SM Count", "sm_count", ""),
    ("Max Threads per SM", "max_threads_per_sm", ""),
    ("L1 Cache per SM", "l1_cache_kb", " KB"),
    ("L2 Cache (Total)", "l2_cache_mb", " MB"),
]

# Special case: Memory Size has two fields combined
GPU_MEMORY_FIELDS: List[Tuple[str, str, str, str]] = [
    # (label, size_key, type_key, size_unit)
    ("Memory Size", "memory_gb", "memory_type", " GB"),
]

# =============================================================================
# NCU Profiling Metric Sections
# =============================================================================

NCU_METRIC_SECTIONS: Dict[str, List[MetricDef]] = {
    "SM & Compute Utilization": [
        ("SM Cycles Active", "sm__cycles_active.avg", ""),
        ("Warp Active", "sm__warps_active.avg.pct_of_peak_sustained_active", "%"),
        ("Total Instructions Executed", "sm__inst_executed.sum", ""),
        (
            "Tensor Core Utilization",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
            "%",
        ),
        (
            "Tensor Core Pipeline Active",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "%",
        ),
    ],
    "Memory Bandwidth & Cache": [
        (
            "DRAM Throughput",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "%",
        ),
        ("DRAM Bandwidth", "dram__bytes.sum.per_second", " bytes/sec"),
        (
            "GPU DRAM Throughput",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "%",
        ),
        ("DRAM Bytes Read", "dram__bytes_read.sum", " bytes"),
        ("DRAM Bytes Write", "dram__bytes_write.sum", " bytes"),
        ("L1 Cache Hit Rate", "l1tex__t_sector_hit_rate.pct", "%"),
        (
            "L1 Throughput",
            "l1tex__throughput.avg.pct_of_peak_sustained_active",
            "%",
        ),
        ("L2 Cache Hit Rate", "lts__t_sector_hit_rate.pct", "%"),
        (
            "L2 Throughput",
            "lts__throughput.avg.pct_of_peak_sustained_active",
            "%",
        ),
    ],
    "Memory Access Patterns": [
        (
            "Memory Coalescing",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            "%",
        ),
        (
            "Branch Uniformity",
            "smsp__sass_average_branch_targets_threads_uniform.pct",
            "%",
        ),
    ],
    "Occupancy & Resources": [
        ("Occupancy Limited By Blocks", "launch__occupancy_limit_blocks", ""),
        ("Occupancy Limited By Registers", "launch__occupancy_limit_registers", ""),
        (
            "Occupancy Limited By Shared Memory",
            "launch__occupancy_limit_shared_mem",
            "",
        ),
        ("Registers per Thread", "launch__registers_per_thread", ""),
        (
            "Shared Memory per Block",
            "launch__shared_mem_per_block_allocated",
            " bytes",
        ),
    ],
    "Stall Metrics (Warp Issue Stalls)": [
        (
            "Short Scoreboard Stalls",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "%",
        ),
        (
            "Long Scoreboard Stalls",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "%",
        ),
        (
            "Barrier Stalls",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "%",
        ),
        (
            "Branch Resolving Stalls",
            "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
            "%",
        ),
    ],
}
