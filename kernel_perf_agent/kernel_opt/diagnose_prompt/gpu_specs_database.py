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
GPU Specifications Database

This module contains the GPU hardware specifications database used for
performance analysis and bottleneck identification. Separated into its
own file to allow easier module overriding.

Sources: NVIDIA official specifications, manufacturer datasheets
"""

GPU_SPECS_DATABASE: dict[str, dict[str, object]] = {
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
