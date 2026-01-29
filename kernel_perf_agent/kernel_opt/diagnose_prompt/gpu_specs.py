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
GPU Specifications Database for Bottleneck Analysis

This module provides GPU hardware specifications needed for performance analysis
and bottleneck identification. It includes peak compute performance, memory bandwidth,
cache sizes, and SM counts for common NVIDIA GPUs.

"""

from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs_database import (
    GPU_SPECS_DATABASE,
)

__all__ = ["GPU_SPECS_DATABASE", "get_gpu_specs"]


def get_gpu_specs(gpu_name: str) -> dict[str, Any] | None:
    """
    Get GPU specifications for bottleneck analysis.

    This function returns hardware specifications needed for performance analysis,
    including peak compute performance, memory bandwidth, cache sizes, and SM counts.

    Args:
        gpu_name: GPU name. Must exactly match a key in GPU_SPECS_DATABASE.

    Returns:
        Dictionary with GPU specifications, or None if GPU is not in the database.
        When successful, contains:
        - name: GPU name
        - architecture: GPU architecture (e.g., "Ampere", "Hopper")
        - peak_fp32_tflops: Peak FP32 compute performance in TFLOPS
        - peak_fp16_tflops: Peak FP16 compute performance in TFLOPS
        - peak_bf16_tflops: Peak BF16 compute performance in TFLOPS (0 if not supported)
        - peak_memory_bw_gbps: Peak memory bandwidth in GB/s
        - sm_count: Number of streaming multiprocessors
        - max_threads_per_sm: Maximum threads per SM
        - l1_cache_kb: L1 cache size in KB per SM
        - l2_cache_mb: Total L2 cache size in MB
        - memory_gb: Total GPU memory in GB
        - memory_type: Memory type (e.g., "HBM2e", "GDDR6X")

    Examples:
        >>> specs = get_gpu_specs("NVIDIA A100")
        >>> if specs:
        ...     print(f"SM Count: {specs['sm_count']}")
    """
    if gpu_name in GPU_SPECS_DATABASE:
        return GPU_SPECS_DATABASE[gpu_name].copy()

    print(f"⚠️  Unknown GPU: '{gpu_name}'. Disable Optimization")
    print(f"    Available GPUs: {', '.join(GPU_SPECS_DATABASE.keys())}")
    return None


if __name__ == "__main__":
    print("GPU Specifications Module")
    print("=" * 60)

    # Show all available GPUs
    print("Available GPU specifications in database:")
    for gpu_name in sorted(GPU_SPECS_DATABASE.keys()):
        print(f"  - {gpu_name}")

    # Example usage
    print(f"\n{'=' * 60}")
    example_gpu = "NVIDIA A100"
    specs = get_gpu_specs(example_gpu)
    if specs:
        print(f"\nExample specs for {example_gpu}:")
        print(f"  - Peak Memory Bandwidth: {specs['peak_memory_bw_gbps']} GB/s")
        print(f"  - Peak FP32 Performance: {specs['peak_fp32_tflops']} TFLOPS")
        print(f"  - SM Count: {specs['sm_count']}")
