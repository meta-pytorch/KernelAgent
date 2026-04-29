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
KernelAgent-Oink: Blackwell SM10x CuTeDSL kernels + optional vLLM plugin.

This package can be loaded as a vLLM "general plugin" (entrypoint group
`vllm.general_plugins`). In that mode it registers Oink custom ops only when
explicitly enabled via an environment variable (so installing the package does
not change behavior by default).

For standalone usage (outside vLLM), call `kernelagent_oink.register(force=True)`
to register the custom ops explicitly.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_OPS_REGISTERED = False


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _infer_cuda_device_index() -> int:
    local_rank = os.environ.get("LOCAL_RANK", "0").strip()
    try:
        return int(local_rank)
    except ValueError:
        return 0


def _compute_cutedsl_arch(major: int, minor: int) -> str:
    # CuTeDSL uses an "a" suffix for >= Hopper.
    suffix = "a" if major >= 9 else ""
    # Match cutlass/base_dsl/env_manager.py: map sm_110 -> sm_101.
    if major == 11 and minor == 0:
        major, minor = 10, 1
    return f"sm_{major}{minor}{suffix}"


def _check_and_setup() -> bool:
    """Check CUDA availability, SM >= 100, CuTeDSL deps, and set CUTE_DSL_ARCH.

    Returns True if all checks pass, False otherwise. Does not raise.
    """
    try:
        import torch
    except Exception as e:
        logger.debug("Oink plugin: torch import failed: %s", e)
        return False

    try:
        if not torch.cuda.is_available():
            logger.debug("Oink plugin: CUDA not available; skipping")
            return False
        device_index = _infer_cuda_device_index()
        major, minor = torch.cuda.get_device_capability(device_index)
        sm = 10 * int(major) + int(minor)
        if sm < 100:
            return False

        try:
            import cutlass  # noqa: F401
            import cuda.bindings.driver as _cuda  # noqa: F401
        except Exception as e:
            logger.warning(
                "Oink plugin: CuTeDSL deps missing; skipping. "
                "Install `nvidia-cutlass-dsl` + `cuda-python`. Error: %s",
                e,
            )
            return False

        os.environ.setdefault(
            "CUTE_DSL_ARCH", _compute_cutedsl_arch(int(major), int(minor))
        )
        return True
    except Exception as e:
        logger.exception("Oink plugin: setup failed: %s", e)
        return False


def register(*, force: bool = False) -> None:
    """Register Oink torch custom ops (``torch.ops.oink.*``).

    This registers ``torch.ops.oink.rmsnorm`` and
    ``torch.ops.oink.fused_add_rms_norm`` for use by vLLM's direct-call path.
    It does NOT override aten ops — use :func:`register_all_kernels` for that.

    - vLLM plugin mode (default): no-op unless ``VLLM_USE_OINK_RMSNORM`` is truthy.
    - Standalone mode: pass ``force=True`` to register explicitly.
    """
    global _OPS_REGISTERED

    if _OPS_REGISTERED:
        return

    if not force and not _env_truthy("VLLM_USE_OINK_RMSNORM"):
        return

    if not _check_and_setup():
        return

    try:
        from .blackwell import oink_custom_ops  # noqa: F401
    except Exception as e:
        logger.exception("Oink plugin: failed to register custom ops: %s", e)
        return

    _OPS_REGISTERED = True


_ALL_KERNELS_REGISTERED = False


def register_all_kernels(*, force: bool = False) -> None:
    """Override aten ops with Oink's kernels.

    Checks CUDA/Blackwell SM10x/deps, sets up the CuTeDSL environment, then overrides
    ``aten::_fused_rms_norm`` and ``aten::_fused_rms_norm_backward`` on CUDA.

    Does NOT register ``torch.ops.oink.*`` custom ops — use :func:`register`
    separately if those are needed (e.g. for vLLM's direct-call path).

    Args:
        force: If *True*, bypass the ``VLLM_USE_OINK_RMSNORM`` env gate.
    """
    global _ALL_KERNELS_REGISTERED
    if _ALL_KERNELS_REGISTERED:
        return

    if not force and not _env_truthy("VLLM_USE_OINK_RMSNORM"):
        return

    if not _check_and_setup():
        return

    try:
        from .aten_override import override_all_kernels

        override_all_kernels()
    except Exception as e:
        logger.exception("Oink: failed to override aten ops: %s", e)
        return

    _ALL_KERNELS_REGISTERED = True


def unregister_all_kernels() -> None:
    """Remove the aten override. Can be followed by :func:`register_all_kernels`."""
    global _ALL_KERNELS_REGISTERED
    try:
        from .aten_override import restore_all_kernels

        restore_all_kernels()
    except Exception:
        pass
    _ALL_KERNELS_REGISTERED = False


__all__ = ["register", "register_all_kernels", "unregister_all_kernels"]
