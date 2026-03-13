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
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)

# ROCm/AMD GPU platform constants
_ROCM_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR AMD ROCm:**
- Default tensor allocations to device='cuda' (ROCm exposes HIP as CUDA-compatible via torch.cuda)
- Check availability with: torch.cuda.is_available() (returns True on ROCm/HIP)
- AMD wavefront size is 64 (not 32 like NVIDIA warps) — account for this in tiling
- Do NOT assume NVIDIA-specific ISA features (e.g., warp shuffle semantics differ)
- Use torch.cuda.synchronize() for synchronization (works on ROCm via HIP)
- Preferred block sizes: 64, 128, 256, or 512 (multiples of wavefront size 64)
- triton.language.constexpr BLOCK_SIZE should be a power of 2 >= 64"""

_ROCM_KERNEL_GUIDANCE = """\
## AMD ROCm-Specific Optimizations

You are generating a Triton kernel for AMD GPUs (ROCm/HIP). Follow these guidelines:

1. **Device Context**: Use 'cuda' as the device string (ROCm provides HIP-CUDA compatibility)
2. **Wavefront Size**: AMD GPUs use wavefront size 64 (vs NVIDIA warp size 32)
   - Prefer BLOCK_SIZE multiples of 64 (64, 128, 256, 512)
   - num_warps maps to num_wavefronts on AMD
3. **Memory Hierarchy**: AMD CDNA GPUs have HBM memory with very high bandwidth
   - MI300X: 5.3 TB/s, MI350X: ~8 TB/s
   - Optimize for memory coalescing and avoid strided access patterns
4. **Compute Units**: AMD uses Compute Units (CUs), each with 64-lane SIMD
   - MI300X has 304 CUs, MI350X has 304 CUs
5. **Data Types**: AMD CDNA supports fp32, fp16, bf16, fp8 (gfx942+)
   - BF16 matrix units available on MI300X (CDNA3) and later
6. **Thread Configuration**:
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
   - num_warps: typically 4, 8 for AMD (maps to wavefronts)
7. **Avoid NVIDIA-specific patterns**: Do not use warp-level primitives that assume warp_size=32"""

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "rocm": PlatformConfig(
        name="rocm",
        device_string="cuda",  # ROCm uses torch.cuda (HIP compatibility layer)
        guidance_block=_ROCM_GUIDANCE,
        kernel_guidance=_ROCM_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
