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

A backend is described by what it *can do*, not by its name. Templates and
callers read capabilities off a :class:`PlatformConfig`; nothing branches on
``if device == "xpu"``. That matters because an ``if/else`` on two known names
is not an abstraction --- it silently routes every third backend down the CUDA
path --- and because a backend that lives outside this repository cannot add an
arm to an ``if``, but it can register a config.

The five capabilities a generated harness needs are:

``availability_check``
    Code that raises if the device cannot be used. Emitted verbatim into the
    generated test.
``device_setup``
    Any extra setup the backend needs after the device string is bound.
``synchronize_call``
    How to wait for the device, or empty when there is nothing to wait for.
``test_prelude``
    Imports or statements a generated test needs before anything else.
``default_num_workers``
    How many generation workers this backend can usefully run at once.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance

Registering a backend from outside this module:
    from triton_kernel_agent.platform_config import PlatformConfig, register_platform

    register_platform(PlatformConfig(name="mybackend", device_string="mybackend", ...))
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"

# How many generation workers a backend runs by default. Kept as the historical
# value so registering the capability changes no existing behaviour.
DEFAULT_NUM_WORKERS = 4


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend.

    Attributes:
        name: Registry key, and the value a CLI accepts.
        device_string: What ``torch`` calls this device.
        guidance_block: Platform requirements injected into the test prompt.
        kernel_guidance: Platform optimization notes injected into the kernel
            prompt.
        cuda_hacks_to_strip: Literal snippets to remove from model output that
            tried to force a CUDA path.
        availability_check: Code raising if the device is unusable. Rendered
            verbatim into the generated test, so it must be valid Python at zero
            indentation; the template indents it.
        device_setup: Extra setup emitted after the device string is bound.
            Empty for backends that need none.
        synchronize_call: Expression that waits for the device, or empty when
            the backend is synchronous. Empty is a real answer, not a gap.
        test_prelude: Statements a generated test needs before anything else.
        default_num_workers: Generation workers to run concurrently when the
            caller and the environment do not say.
    """

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)
    availability_check: str = ""
    device_setup: str = ""
    synchronize_call: str = ""
    test_prelude: str = ""
    default_num_workers: int = DEFAULT_NUM_WORKERS


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

# Availability checks. These are emitted verbatim into the generated test and
# are byte-for-byte what the `{% if device_string == "xpu" %}` branch used to
# render, so moving them out of the template changes no generated output.
_CUDA_AVAILABILITY = """\
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")"""

_XPU_AVAILABILITY = """\
if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
    raise RuntimeError("Intel XPU not available. Install PyTorch with Intel GPU support.")"""

# The fake backend asserts nothing, because there is nothing to assert: the
# check has to remain a statement so the emitted block is never empty.
_FAKE_AVAILABILITY = """\
# The fake backend has no device; nothing to check.
pass"""

_FAKE_GUIDANCE = """\
**PLATFORM: FAKE BACKEND (no accelerator).**
- This backend exists to exercise the generation pipeline without hardware.
- Nothing generated for it is a performance claim, and no timing it reports is
  meaningful.
- Allocate on device='cpu' and do not call any accelerator API."""


# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
        availability_check=_CUDA_AVAILABILITY,
        synchronize_call="torch.cuda.synchronize()",
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
        availability_check=_XPU_AVAILABILITY,
        synchronize_call="torch.xpu.synchronize()",
    ),
    # A backend with no accelerator behind it, for exercising the pipeline on a
    # host with no device. Mirrors the `noop` implementations in
    # `triton_kernel_agent.platform`, and is named for what it is so nothing
    # downstream mistakes its output for a measurement.
    "fake": PlatformConfig(
        name="fake",
        device_string="cpu",
        guidance_block=_FAKE_GUIDANCE,
        kernel_guidance="",
        cuda_hacks_to_strip=(),
        availability_check=_FAKE_AVAILABILITY,
        # Nothing to synchronize. Empty is the answer, not a missing value.
        synchronize_call="",
        default_num_workers=1,
    ),
}


class UnknownPlatformError(ValueError):
    """Raised for a platform name that is not registered.

    A subclass rather than a bare ``ValueError`` so a caller can distinguish
    "that backend does not exist" from any other bad argument, while remaining
    backward compatible with code that catches ``ValueError``.
    """


def register_platform(config: PlatformConfig, *, replace: bool = False) -> None:
    """Register a backend.

    This is the seam that lets a backend defined outside this repository become
    selectable without editing anything here. It is an explicit call rather than
    an import-time decorator so that registration order is something a caller
    controls and can reason about.

    Args:
        config: The backend to add.
        replace: Whether to overwrite an existing registration.

    Raises:
        ValueError: If the name is blank, or is already registered and
            ``replace`` is not set. Silently overwriting would let one import
            change another backend's behaviour invisibly.
    """
    if not config.name.strip():
        raise ValueError("a platform must have a name")
    if config.name in PLATFORMS and not replace:
        raise ValueError(
            f"platform {config.name!r} is already registered; pass replace=True "
            "to override it deliberately"
        )
    PLATFORMS[config.name] = config


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name.

    Args:
        name: A registered platform name.

    Returns:
        The registered configuration.

    Raises:
        UnknownPlatformError: If the name is not registered. There is no default
            fallback: silently returning CUDA for an unrecognized backend would
            generate CUDA code under another backend's name.
    """
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise UnknownPlatformError(
            f"Unknown platform '{name}'. Available: {available}"
        )
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
