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
Override Aten kernels with Oink's Blackwell CuTeDSL Kernels.

Currently overrides:
- ``aten::_fused_rms_norm`` → ``rmsnorm_forward``
- ``aten::_fused_rms_norm_backward`` → ``rmsnorm_backward``

Follows the quack PR pattern: ``with_keyset=True``, fallback via ``call_boxed``.
Calls ``rmsnorm_forward`` / ``rmsnorm_backward`` directly to get all kernel
optimizations (ptr fast-launch, atomic dW, _reduce_partial_sum_fp32).
"""

from __future__ import annotations

import importlib
import logging
import math
from functools import cache, partial
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (cached)
# ---------------------------------------------------------------------------


@cache
def _oink_rmsnorm():
    return importlib.import_module("kernelagent_oink.blackwell.rmsnorm")


# ---------------------------------------------------------------------------
# Device support (cached)
# ---------------------------------------------------------------------------


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def _is_supported(input: torch.Tensor) -> bool:
    return (
        input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _get_device_major(input.device) >= 10
        and input.shape[-1] >= 128  # oink kernels require N >= 128
    )


# ---------------------------------------------------------------------------
# Reshape helpers (match quack's norms.py)
# ---------------------------------------------------------------------------


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        return t
    return t.reshape(M, N).contiguous()


def _flatten_rstd(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


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

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = input.reshape(M, N)

    if weight is not None and weight.ndim != 1:
        weight = weight.view(N)

    y, rstd, _ = _oink_rmsnorm().rmsnorm_forward(
        x, weight=weight, bias=None, residual=None, eps=eps, store_rstd=True,
    )

    y = y.reshape(input_shape)
    stat_shape = list(input_shape[: -len(normalized_shape)]) + [1] * len(
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

    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    w = weight if output_mask[1] else None
    dx, dw, _db, _dres = _oink_rmsnorm().rmsnorm_backward(
        x, w, dout, rstd_flat,
        dresidual_out=None, has_bias=False, has_residual=False,
    )

    grad_input: torch.Tensor | None = dx.reshape(input.shape)
    grad_weight: torch.Tensor | None = dw

    # Match native _fused_rms_norm_backward: return None for masked outputs.
    if not output_mask[0]:
        grad_input = None
    if not output_mask[1]:
        grad_weight = None

    return grad_input, grad_weight


# =========================================================================
# Registration
# =========================================================================

_OVERRIDE_LIB: torch.library.Library | None = None


def override_all_kernels() -> None:
    """Override Aten's kernels on CUDA with Oink's kernels."""
    global _OVERRIDE_LIB
    if _OVERRIDE_LIB is not None:
        return

    fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
    bwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm_backward", "CUDA")

    fwd_impl = partial(_fused_rms_norm_impl, fallback_kernel=fwd_fallback)
    bwd_impl = partial(_fused_rms_norm_backward_impl, fallback_kernel=bwd_fallback)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_fused_rms_norm", fwd_impl, "CUDA", with_keyset=True, allow_override=True)
    lib.impl("_fused_rms_norm_backward", bwd_impl, "CUDA", with_keyset=True, allow_override=True)
    _OVERRIDE_LIB = lib
    logger.info("Oink: overrode aten::_fused_rms_norm on CUDA")


def restore_all_kernels() -> None:
    """Remove the override and restore PyTorch's native CUDA kernels."""
    global _OVERRIDE_LIB
    if _OVERRIDE_LIB is None:
        return
    _OVERRIDE_LIB = None


__all__ = [
    "override_all_kernels",
    "restore_all_kernels",
]
