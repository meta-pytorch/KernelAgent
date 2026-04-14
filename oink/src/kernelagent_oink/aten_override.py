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
Override ``aten::_fused_rms_norm`` with Oink's Blackwell CuTeDSL RMSNorm.

Follows the quack PR pattern (``torch/_native/ops/norm/rmsnorm_impl.py``):
registers with ``with_keyset=True`` so the impl receives a
``DispatchKeySet`` and can call the captured fallback via ``call_boxed``.

Usage::

    import kernelagent_oink
    kernelagent_oink.register_all_kernels(force=True)
    y = torch.nn.functional.rms_norm(x, [N], weight, eps)  # uses Oink
"""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tracing — zero overhead when disabled (OINK_TRACE=1).
# ---------------------------------------------------------------------------

_TRACE_ENABLED: bool = os.environ.get("OINK_TRACE", "0").strip() in ("1", "true")
_call_counts: dict[str, int] = {}


def _trace_call(op_name: str) -> None:
    n = _call_counts.get(op_name, 0) + 1
    _call_counts[op_name] = n
    if n == 1:
        print(f"[OINK] {op_name} override called (first invocation)", flush=True)


# ---------------------------------------------------------------------------
# Device / dtype support check
# ---------------------------------------------------------------------------


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def _is_supported(input: torch.Tensor) -> bool:
    return input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ) and _get_device_major(input.device) in (9, 10)


# =========================================================================
# RMSNorm forward
# =========================================================================


def _fused_rms_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    eps: Optional[float],
    *,
    fallback_kernel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return fallback_kernel.call_boxed(
            dispatch_keys, input, normalized_shape, weight, eps
        )

    if eps is None:
        eps = 1e-6

    if _TRACE_ENABLED:
        _trace_call("_fused_rms_norm")

    orig_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = input.reshape(M, N)

    from kernelagent_oink.blackwell.rmsnorm import rmsnorm_forward

    y, rstd, _ = rmsnorm_forward(
        x, weight=weight, bias=None, residual=None, eps=eps, store_rstd=True,
    )

    y = y.reshape(orig_shape)
    stat_shape = list(orig_shape[: -len(normalized_shape)]) + [1] * len(
        normalized_shape
    )
    rstd = rstd.view(stat_shape)
    return y, rstd


# =========================================================================
# RMSNorm backward
# =========================================================================


def _fused_rms_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: List[int],
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    output_mask: List[bool],
    *,
    fallback_kernel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return fallback_kernel.call_boxed(
            dispatch_keys,
            grad_out, input, normalized_shape, rstd, weight, output_mask,
        )

    if _TRACE_ENABLED:
        _trace_call("_fused_rms_norm_backward")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = input.reshape(M, N).contiguous()
    dout = grad_out.reshape(M, N).contiguous()
    rstd_flat = rstd.reshape(M).contiguous()

    from kernelagent_oink.blackwell.rmsnorm import rmsnorm_backward

    dx, dw, _dbias, _dres = rmsnorm_backward(
        x, weight, dout, rstd_flat,
        dresidual_out=None, has_bias=False, has_residual=False,
    )

    dx = dx.reshape(input.shape)

    if not output_mask[0]:
        dx = torch.zeros_like(input)
    if not output_mask[1] or dw is None:
        dw = torch.zeros_like(weight) if weight is not None else torch.empty(0)

    return dx, dw


# =========================================================================
# Registration
# =========================================================================

_ATEN_LIB: torch.library.Library | None = None


def override_all_aten_kernels() -> None:
    """Override ``aten::_fused_rms_norm`` on CUDA with oink's RMSNorm.

    Uses ``with_keyset=True`` (quack PR pattern) so the override receives
    ``DispatchKeySet`` and can call the original kernel via ``call_boxed``
    for unsupported inputs — no Python wrapper overhead on the fallback path.
    """
    global _ATEN_LIB
    if _ATEN_LIB is not None:
        return

    fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
    bwd_fallback = torch.library.get_kernel(
        "aten::_fused_rms_norm_backward", "CUDA"
    )

    fwd_impl = functools.partial(
        _fused_rms_norm_impl, fallback_kernel=fwd_fallback
    )
    bwd_impl = functools.partial(
        _fused_rms_norm_backward_impl, fallback_kernel=bwd_fallback
    )

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_fused_rms_norm", fwd_impl, "CUDA", with_keyset=True)
    lib.impl("_fused_rms_norm_backward", bwd_impl, "CUDA", with_keyset=True)
    _ATEN_LIB = lib
    logger.info("Oink: overrode aten::_fused_rms_norm on CUDA (with_keyset)")


def restore_all_aten_kernels() -> None:
    """Remove the override and restore PyTorch's native CUDA RMSNorm."""
    global _ATEN_LIB
    if _ATEN_LIB is None:
        return
    _ATEN_LIB = None
    logger.info("Oink: restored aten::_fused_rms_norm to PyTorch default")


# Backward-compatible aliases.
override_aten_rmsnorm = override_all_aten_kernels
restore_aten_rmsnorm = restore_all_aten_kernels


__all__ = [
    "override_all_aten_kernels",
    "restore_all_aten_kernels",
    "override_aten_rmsnorm",
    "restore_aten_rmsnorm",
]
