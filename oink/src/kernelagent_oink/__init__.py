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
KernelAgent-Oink: SM100 CuTeDSL kernels + optional vLLM plugin.

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
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError:
            pass
    return 0


def _compute_cutedsl_arch(major: int, minor: int) -> str:
    # CuTeDSL uses an "a" suffix for >= Hopper.
    suffix = "a" if major >= 9 else ""
    # Match cutlass/base_dsl/env_manager.py: map sm_110 -> sm_101.
    if major == 11 and minor == 0:
        major, minor = 10, 1
    return f"sm_{major}{minor}{suffix}"


def register(*, force: bool = False) -> None:
    """Register Oink torch custom ops.

    - vLLM plugin mode (default): no-op unless `VLLM_USE_OINK_RMSNORM` is truthy.
    - Standalone mode: pass `force=True` to register explicitly.

    This function must be safe to call multiple times and must not raise. vLLM
    executes it in multiple processes (engine + workers).
    """
    global _OPS_REGISTERED

    if _OPS_REGISTERED:
        return

    # Gate on the vLLM integration flag so installing the package does not
    # change behavior unless explicitly enabled. For standalone usage (outside
    # vLLM), callers can pass force=True to register the ops explicitly.
    if not force and not _env_truthy("VLLM_USE_OINK_RMSNORM"):
        return

    try:
        import torch
    except Exception as e:  # pragma: no cover
        logger.debug("Oink plugin: torch import failed: %s", e)
        return

    try:
        if not torch.cuda.is_available():
            return
        device_index = _infer_cuda_device_index()
        major, minor = torch.cuda.get_device_capability(device_index)
        sm = 10 * int(major) + int(minor)
        if sm < 100:
            return

        # Ensure required deps are importable before registering ops so that vLLM
        # doesn't detect ops that would later fail at first use.
        try:
            import cutlass  # noqa: F401
            import cuda.bindings.driver as _cuda  # noqa: F401
        except Exception as e:
            logger.warning(
                "Oink plugin: CuTeDSL deps missing; skipping op registration. "
                "Install `nvidia-cutlass-dsl` + `cuda-python`. Error: %s",
                e,
            )
            return

        # Ensure CuTeDSL sees a target arch early. If the user has already set it,
        # respect their choice.
        os.environ.setdefault(
            "CUTE_DSL_ARCH", _compute_cutedsl_arch(int(major), int(minor))
        )

        # Import registers the ops via torch.library.custom_op decorators.
        from .blackwell import oink_custom_ops  # noqa: F401
    except Exception as e:  # pragma: no cover
        # Do not raise: vLLM plugin loader does not guard plugin execution.
        logger.exception("Oink plugin: failed to register ops: %s", e)
        return

    _OPS_REGISTERED = True


__all__ = ["register"]
