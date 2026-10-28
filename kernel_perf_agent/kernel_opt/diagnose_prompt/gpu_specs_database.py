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
GPU Specifications Database - Updated with Specific SKUs

This module contains the GPU hardware specifications database used for
performance analysis and bottleneck identification. Updated to include
specific SKU variants for multi-SKU GPUs like A100 and H100, and AMD
Instinct GPUs for ROCm support.

Sources:
- NVIDIA official specifications and datasheets
- AMD official specifications and datasheets
- TechPowerUp GPU Database
- Manufacturer datasheets

Last Updated: March 2026
"""

from types import MappingProxyType

_GPU_SPECS_DATABASE: dict[str, dict[str, object]] = {
    # NVIDIA A100 SKUs - SXM4 Variants
    "NVIDIA A100 SXM4 40GB": {
        "name": "NVIDIA A100 SXM4 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    "NVIDIA A100 SXM4 80GB": {
        "name": "NVIDIA A100 SXM4 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 2039,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    # NVIDIA A100 SKUs - PCIe Variants
    "NVIDIA A100 PCIe 40GB": {
        "name": "NVIDIA A100 PCIe 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 250,
    },
    "NVIDIA A100 PCIe 80GB": {
        "name": "NVIDIA A100 PCIe 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,  # Without sparsity
        "peak_bf16_tflops": 312.0,  # Without sparsity
        "peak_memory_bw_gbps": 1935,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 300,
    },
    # NVIDIA H100 SKUs - SXM5 Variant
    "NVIDIA H100 SXM5 80GB": {
        "name": "NVIDIA H100 SXM5 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 1979.0,  # Without sparsity
        "peak_bf16_tflops": 1979.0,  # Without sparsity
        "peak_memory_bw_gbps": 3350,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM3",
        "form_factor": "SXM5",
        "tdp_w": 700,
    },
    # NVIDIA H100 SKUs - PCIe Variant
    "NVIDIA H100 PCIe 80GB": {
        "name": "NVIDIA H100 PCIe 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 51.0,
        "peak_fp16_tflops": 1513.0,  # Without sparsity
        "peak_bf16_tflops": 1513.0,  # Without sparsity
        "peak_memory_bw_gbps": 2000,
        "sm_count": 114,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 350,
    },
    # NVIDIA H100 SKUs - NVL Variant (for LLM inference)
    "NVIDIA H100 NVL 94GB": {
        "name": "NVIDIA H100 NVL 94GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 60.0,
        "peak_fp16_tflops": 1671.0,  # Without sparsity
        "peak_bf16_tflops": 1671.0,  # Without sparsity
        "peak_memory_bw_gbps": 3900,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 94,
        "memory_type": "HBM3",
        "form_factor": "PCIe",
        "tdp_w": 400,
    },
    # NVIDIA RTX 4090
    "NVIDIA RTX 4090": {
        "name": "NVIDIA RTX 4090",
        "architecture": "Ada Lovelace",
        "peak_fp32_tflops": 82.58,
        "peak_fp16_tflops": 82.58,
        "peak_bf16_tflops": 82.58,
        "peak_memory_bw_gbps": 1008,
        "sm_count": 128,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 72,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "form_factor": "PCIe",
        "tdp_w": 450,
    },
    # NVIDIA RTX 5080
    "NVIDIA RTX 5080": {
        "name": "NVIDIA RTX 5080",
        "architecture": "Blackwell",
        "peak_fp32_tflops": 56.28,
        "peak_fp16_tflops": 56.28,
        "peak_bf16_tflops": 56.28,
        "peak_memory_bw_gbps": 960,
        "sm_count": 84,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 64,
        "memory_gb": 16,
        "memory_type": "GDDR7",
        "form_factor": "PCIe",
        "tdp_w": 360,
    },
    # -----------------------------------------------------------------------
    # AMD Instinct GPU SKUs (ROCm / HIP)
    # -----------------------------------------------------------------------
    # AMD Instinct MI300X (CDNA3 / gfx942)
    # Sources: AMD product page, Hot Chips 35 (2023)
    "AMD Instinct MI300X": {
        "name": "AMD Instinct MI300X",
        "architecture": "CDNA3",
        "gfx_target": "gfx942",
        "peak_fp32_tflops": 163.4,
        "peak_fp16_tflops": 1307.4,  # BF16/FP16 matrix (without sparsity)
        "peak_bf16_tflops": 1307.4,
        "peak_memory_bw_gbps": 5300,  # 5.3 TB/s HBM3
        "cu_count": 304,  # Compute Units (AMD equiv of SM)
        "sm_count": 304,  # Alias for compatibility
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,  # Alias for compatibility
        "wavefront_size": 64,
        "l1_cache_kb": 32,  # L1 per CU (vector L1D)
        "l2_cache_mb": 256,  # Total Infinity Cache (across all dies)
        "memory_gb": 192,
        "memory_type": "HBM3",
        "form_factor": "OAM",
        "tdp_w": 750,
    },
    # AMD Instinct MI300A (CDNA3 / gfx942, APU variant)
    "AMD Instinct MI300A": {
        "name": "AMD Instinct MI300A",
        "architecture": "CDNA3",
        "gfx_target": "gfx942",
        "peak_fp32_tflops": 122.6,
        "peak_fp16_tflops": 980.6,
        "peak_bf16_tflops": 980.6,
        "peak_memory_bw_gbps": 3200,  # Unified HBM3 (shared with CPU)
        "cu_count": 228,
        "sm_count": 228,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 32,
        "l2_cache_mb": 192,
        "memory_gb": 128,
        "memory_type": "HBM3 unified",
        "form_factor": "OAM",
        "tdp_w": 550,
    },
    # AMD Instinct MI350X (CDNA4 / gfx950)
    # Sources: AMD press release (Nov 2024), estimated specs
    "AMD Instinct MI350X": {
        "name": "AMD Instinct MI350X",
        "architecture": "CDNA4",
        "gfx_target": "gfx950",
        "peak_fp32_tflops": 288.0,
        "peak_fp16_tflops": 2304.0,  # BF16 matrix estimate
        "peak_bf16_tflops": 2304.0,
        "peak_memory_bw_gbps": 8000,  # ~8 TB/s HBM3E
        "cu_count": 304,
        "sm_count": 304,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 32,
        "l2_cache_mb": 256,
        "memory_gb": 288,
        "memory_type": "HBM3E",
        "form_factor": "OAM",
        "tdp_w": 1000,
        "specs_confidence": "estimated",
    },
    # AMD Instinct MI250X (CDNA2 / gfx90a)
    "AMD Instinct MI250X": {
        "name": "AMD Instinct MI250X",
        "architecture": "CDNA2",
        "gfx_target": "gfx90a",
        "peak_fp32_tflops": 47.9,
        "peak_fp16_tflops": 383.0,
        "peak_bf16_tflops": 383.0,
        "peak_memory_bw_gbps": 3277,  # HBM2e
        "cu_count": 220,
        "sm_count": 220,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 16,
        "l2_cache_mb": 32,
        "memory_gb": 128,
        "memory_type": "HBM2e",
        "form_factor": "OAM",
        "tdp_w": 560,
    },
    # AMD Instinct MI250 (CDNA2 / gfx90a, dual GCD)
    # Same 3.2 TB/s HBM2e as MI250X. Peaks follow the 208/220 CU ratio.
    "AMD Instinct MI250": {
        "name": "AMD Instinct MI250",
        "architecture": "CDNA2",
        "gfx_target": "gfx90a",
        "peak_fp32_tflops": 45.3,
        "peak_fp16_tflops": 362.1,
        "peak_bf16_tflops": 362.1,
        "peak_memory_bw_gbps": 3277,  # 3.2768 TB/s HBM2e
        "cu_count": 208,
        "sm_count": 208,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 16,
        "l2_cache_mb": 16,  # 8 MB per GCD
        "memory_gb": 128,
        "memory_type": "HBM2e",
        "form_factor": "OAM",
        "tdp_w": 500,
    },
    # AMD Instinct MI210 (CDNA2 / gfx90a, single GCD PCIe)
    # gfx90a is shared with MI250 and MI250X. A bare ISA name stays MI250X
    # unless the PCI chip id selects this row.
    # Sources: AMD MI210 product page and brochure (2022), ROCm gpu-arch-specs
    "AMD Instinct MI210": {
        "name": "AMD Instinct MI210",
        "architecture": "CDNA2",
        "gfx_target": "gfx90a",
        "peak_fp32_tflops": 22.6,
        "peak_fp16_tflops": 181.0,
        "peak_bf16_tflops": 181.0,
        "peak_memory_bw_gbps": 1638,  # 1.6384 TB/s HBM2e
        "cu_count": 104,
        "sm_count": 104,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 16,
        "l2_cache_mb": 8,
        "memory_gb": 64,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 300,
    },
    # AMD Instinct MI100 (CDNA / gfx908)
    # Sources: AMD MI100 product page, CDNA whitepaper, ROCm gpu-arch-specs.
    # Matrix BF16 is half the FP16 rate on CDNA1.
    "AMD Instinct MI100": {
        "name": "AMD Instinct MI100",
        "architecture": "CDNA",
        "gfx_target": "gfx908",
        "peak_fp32_tflops": 23.1,
        "peak_fp16_tflops": 184.6,
        "peak_bf16_tflops": 92.3,
        "peak_memory_bw_gbps": 1229,  # 1.2288 TB/s HBM2
        "cu_count": 120,
        "sm_count": 120,
        "max_threads_per_cu": 2048,
        "max_threads_per_sm": 2048,
        "wavefront_size": 64,
        "l1_cache_kb": 16,
        "l2_cache_mb": 8,
        "memory_gb": 32,
        "memory_type": "HBM2",
        "form_factor": "PCIe",
        "tdp_w": 300,
    },
}

# Make database read-only to prevent accidental modification
GPU_SPECS_DATABASE = MappingProxyType(_GPU_SPECS_DATABASE)
