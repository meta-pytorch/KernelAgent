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

import json
import os
from pathlib import Path


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

# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
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

def load_platform_config_from_json(path: str | Path) -> dict[str, PlatformConfig]:
    """Load platform configs from a JSON file."""

    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    loaded: dict[str, PlatformConfig] = {}
    for name, cfg in raw.items():

        device_name = cfg.get("name", "")
        device_string = str(cfg.get("device_string", ""))
        guidance_block = str(cfg.get("guidance_block", ""))
        kernel_guidance = str(cfg.get("kernel_guidance", ""))
        hacks = cfg.get("cuda_hacks_to_strip", [])

        loaded[name] = PlatformConfig(
            name=device_name,
            device_string=device_string,
            guidance_block=guidance_block,
            kernel_guidance=kernel_guidance,
            cuda_hacks_to_strip=tuple(hacks),
        )

    return loaded


def _maybe_load_external_platforms() -> None:
    """Optionally merge JSON-defined platforms into PLATFORMS."""
    from dotenv import load_dotenv
    load_dotenv()

    env_path = os.getenv("KERNELAGENT_PLATFORM_JSON")
    path = Path(env_path) if env_path else None
    if not path or not path.exists():
        return
    try:
        PLATFORMS.update(load_platform_config_from_json(path))
    except Exception:
        raise ValueError(f"Failed to load platforms from JSON at {path}")


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())


# Load external platforms from JSON if specified in environment
_maybe_load_external_platforms()
