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

import subprocess
from typing import Any, Dict, Optional

# GPU specifications database
# Sources: NVIDIA official specifications, manufacturer datasheets
GPU_SPECS_DATABASE = {
    "NVIDIA A100": {
        "name": "NVIDIA A100",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_bf16_tflops": 312.0,
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
    },
    "NVIDIA H100": {
        "name": "NVIDIA H100",
        "architecture": "Hopper",
        "peak_fp32_tflops": 51.0,
        "peak_fp16_tflops": 989.0,
        "peak_bf16_tflops": 989.0,
        "peak_memory_bw_gbps": 3352,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM3",
    },
    "NVIDIA RTX 4090": {
        "name": "NVIDIA RTX 4090",
        "architecture": "Ada Lovelace",
        "peak_fp32_tflops": 82.6,
        "peak_fp16_tflops": 165.0,
        "peak_bf16_tflops": 165.0,
        "peak_memory_bw_gbps": 1008,
        "sm_count": 128,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 72,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
    },
    "NVIDIA RTX 5080": {
        "name": "NVIDIA RTX 5080",
        "architecture": "Blackwell",
        "peak_fp32_tflops": 57.0,
        "peak_fp16_tflops": 114.0,
        "peak_bf16_tflops": 114.0,
        "peak_memory_bw_gbps": 960,
        "sm_count": 84,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 64,
        "memory_gb": 16,
        "memory_type": "GDDR7",
    },
}


def query_gpu_name() -> Optional[str]:
    """
    Query GPU name using nvidia-smi.

    Returns:
        GPU name string, or None if query fails
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Take only the first GPU (nvidia-smi returns one line per GPU)
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            return gpu_name
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def get_gpu_specs(gpu_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get GPU specifications for bottleneck analysis.

    This function returns hardware specifications needed for performance analysis,
    including peak compute performance, memory bandwidth, cache sizes, and SM counts.

    Args:
        gpu_name: GPU name (if None, auto-detect with nvidia-smi)

    Returns:
        Dictionary with GPU specifications containing:
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
        >>> specs = get_gpu_specs()  # Auto-detect
        >>> print(f"Peak BW: {specs['peak_memory_bw_gbps']} GB/s")

        >>> specs = get_gpu_specs("NVIDIA A100")
        >>> print(f"SM Count: {specs['sm_count']}")
    """
    # Auto-detect if not provided
    if gpu_name is None:
        gpu_name = query_gpu_name()

    # Return default if detection failed
    if gpu_name is None:
        print("⚠️  GPU auto-detection failed, using A100 specs as fallback")
        return GPU_SPECS_DATABASE["NVIDIA A100"].copy()

    # Try exact match
    if gpu_name in GPU_SPECS_DATABASE:
        return GPU_SPECS_DATABASE[gpu_name].copy()

    # Try fuzzy match (contains or partial match)
    gpu_name_lower = gpu_name.lower()
    for key, specs in GPU_SPECS_DATABASE.items():
        key_lower = key.lower()
        # Check if either name contains the other
        if gpu_name_lower in key_lower or key_lower in gpu_name_lower:
            print(f"ℹ️  Matched '{gpu_name}' to '{key}' (fuzzy match)")
            return specs.copy()

    # Fallback to A100 specs with warning
    print(f"⚠️  Unknown GPU: '{gpu_name}', using A100 specs as fallback")
    print(f"    Available GPUs: {', '.join(GPU_SPECS_DATABASE.keys())}")
    return GPU_SPECS_DATABASE["NVIDIA A100"].copy()


if __name__ == "__main__":
    print("GPU Specifications Module")
    print("=" * 60)

    # Auto-detect GPU
    detected_name = query_gpu_name()
    if detected_name:
        print(f"\nDetected GPU: {detected_name}")
    else:
        print("\nNo GPU detected (nvidia-smi not available)")

    # Get specs
    specs = get_gpu_specs()
    print(
        f"\nUsing specs for: {specs['name']} ({specs.get('architecture', 'Unknown')})"
    )
    print(f"  - Peak Memory Bandwidth: {specs['peak_memory_bw_gbps']} GB/s")
    print(f"  - Peak FP32 Performance: {specs['peak_fp32_tflops']} TFLOPS")
    print(f"  - SM Count: {specs['sm_count']}")

    # Show all available GPUs
    print(f"\n{'=' * 60}")
    print("Available GPU specifications in database:")
    for gpu_name in sorted(GPU_SPECS_DATABASE.keys()):
        print(f"  - {gpu_name}")
