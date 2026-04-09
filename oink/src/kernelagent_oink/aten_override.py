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
Override PyTorch aten operators with Oink's Blackwell CuTeDSL kernels.

Patches the following aten ops at the CUDA dispatch key:

- ``aten::_fused_rms_norm``          → :func:`rmsnorm_forward`
- ``aten::_fused_rms_norm_backward`` → :func:`rmsnorm_backward`
- ``aten::native_layer_norm``        → :func:`layernorm`
- ``aten::native_layer_norm_backward`` → :func:`layernorm_backward`
- ``aten::_softmax``                 → :func:`softmax_forward`
- ``aten::_softmax_backward_data``   → :func:`softmax_backward`

All standard PyTorch APIs (``F.rms_norm``, ``F.layer_norm``, ``F.softmax``,
etc.) transparently route through the Oink kernels on SM100+ CUDA devices
after calling :func:`override_all_aten_kernels`.

Each override captures the original CUDA kernel via
``torch.library.get_kernel`` *before* patching, so unsupported inputs
(wrong dtype, older GPU) fall back to PyTorch's native implementation.

Usage::

    from kernelagent_oink.aten_override import override_all_aten_kernels

    override_all_aten_kernels()
    y = torch.nn.functional.rms_norm(x, [N], weight, eps)   # uses Oink

    restore_all_aten_kernels()                               # restores PyTorch
"""

from __future__ import annotations

import functools
import importlib
import logging
import math
import threading
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy kernel module imports
# ---------------------------------------------------------------------------

_MOD_CACHE: dict[str, object] = {}
_MOD_LOCK = threading.Lock()


def _get_mod(name: str):
    """Thread-safe lazy import of ``kernelagent_oink.blackwell.<name>``."""
    cached = _MOD_CACHE.get(name)
    if cached is not None:
        return cached
    with _MOD_LOCK:
        if name not in _MOD_CACHE:
            _MOD_CACHE[name] = importlib.import_module(
                f"kernelagent_oink.blackwell.{name}"
            )
        return _MOD_CACHE[name]


# ---------------------------------------------------------------------------
# Device capability helpers
# ---------------------------------------------------------------------------


@functools.cache
def _get_device_sm(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return 10 * major + minor


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_supported(t: torch.Tensor) -> bool:
    """True when Oink's SM100 kernel can handle this tensor."""
    return (
        t.is_cuda
        and t.dtype in _SUPPORTED_DTYPES
        and _get_device_sm(t.device) >= 100
    )


# ---------------------------------------------------------------------------
# Fallback kernel capture
# ---------------------------------------------------------------------------

_fallbacks: dict[str, object] = {}


def _capture_fallback(op_name: str, dispatch_key: str = "CUDA") -> None:
    """Snapshot the current CUDA kernel for ``aten::<op_name>`` before we
    overwrite it.  Must be called *before* ``lib.impl``."""
    if op_name in _fallbacks:
        return
    try:
        _fallbacks[op_name] = torch.library.get_kernel(
            f"aten::{op_name}", dispatch_key
        )
    except Exception:
        _fallbacks[op_name] = None


def _call_fallback(op_name: str, *args):
    fb = _fallbacks.get(op_name)
    if fb is not None:
        return fb(*args)
    raise RuntimeError(
        f"Oink: no fallback captured for aten::{op_name} and input is unsupported"
    )


# ---------------------------------------------------------------------------
# Reshape helpers
# ---------------------------------------------------------------------------


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if t.ndim == 2 and t.shape == (M, N) and t.is_contiguous():
        return t
    return t.reshape(M, N).contiguous()


def _flatten_1d(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


def _stat_shape(input_shape, normalized_shape_len: int) -> list[int]:
    """Shape for rstd / mean: ``[*batch_dims, 1, 1, ...]`` with
    ``normalized_shape_len`` trailing ones."""
    return list(input_shape[:-normalized_shape_len]) + [1] * normalized_shape_len


# =========================================================================
# RMSNorm
# =========================================================================


def _oink_fused_rms_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    eps: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return _call_fallback(
            "_fused_rms_norm", input, normalized_shape, weight, eps
        )

    if eps is None:
        eps = 1e-6

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)

    mod = _get_mod("rmsnorm")
    y, rstd, _ = mod.rmsnorm_forward(
        x, weight=weight, bias=None, residual=None, eps=eps, store_rstd=True,
    )

    y = y.reshape(input_shape)
    rstd = rstd.view(_stat_shape(input_shape, len(normalized_shape)))
    return y, rstd


def _oink_fused_rms_norm_backward(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: List[int],
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    output_mask: List[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return _call_fallback(
            "_fused_rms_norm_backward",
            grad_out, input, normalized_shape, rstd, weight, output_mask,
        )

    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_1d(rstd, M)

    mod = _get_mod("rmsnorm")
    dx, dw, _dbias, _dres = mod.rmsnorm_backward(
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
# LayerNorm
# =========================================================================


def _oink_native_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return _call_fallback(
            "native_layer_norm", input, normalized_shape, weight, bias, eps
        )

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)

    mod = _get_mod("layernorm")
    out, rstd, mean = mod.layernorm(
        x, weight, bias=bias, eps=eps, return_rstd=True, return_mean=True,
    )

    out = out.reshape(input_shape)
    stat_sh = _stat_shape(input_shape, len(normalized_shape))
    mean = mean.view(stat_sh)
    rstd = rstd.view(stat_sh)
    return out, mean, rstd


def _oink_native_layer_norm_backward(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: List[int],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    output_mask: List[bool],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return _call_fallback(
            "native_layer_norm_backward",
            grad_out, input, normalized_shape, mean, rstd, weight, bias,
            output_mask,
        )

    N = math.prod(normalized_shape)
    M = input.numel() // N

    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    mean_flat = _flatten_1d(mean, M)
    rstd_flat = _flatten_1d(rstd, M)

    mod = _get_mod("layernorm")
    dx, dw, db = mod.layernorm_backward(
        dout, x, weight, rstd_flat, mean_flat, bias=bias,
    )

    dx = dx.reshape(input.shape) if dx is not None else torch.zeros_like(input)

    if not output_mask[0]:
        dx = torch.zeros_like(input)
    if not output_mask[1] or dw is None:
        dw = torch.zeros_like(weight) if weight is not None else torch.empty(0)
    if not output_mask[2] or db is None:
        db = torch.zeros_like(bias) if bias is not None else torch.empty(0)

    return dx, dw, db


# =========================================================================
# Softmax
# =========================================================================


def _oink_softmax(
    self: torch.Tensor,
    dim: int,
    half_to_float: bool,
) -> torch.Tensor:
    # Oink's softmax only handles the last dimension on 2D inputs.
    # Fall back for other dims or when half_to_float is requested.
    ndim = self.ndim
    actual_dim = dim if dim >= 0 else dim + ndim

    if not _is_supported(self) or actual_dim != ndim - 1 or half_to_float:
        return _call_fallback("_softmax", self, dim, half_to_float)

    input_shape = self.shape
    N = input_shape[-1]
    M = self.numel() // N

    x = _reshape_2d(self, M, N)

    mod = _get_mod("softmax")
    y = mod.softmax_forward(x)

    return y.reshape(input_shape)


def _oink_softmax_backward(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    ndim = output.ndim
    actual_dim = dim if dim >= 0 else dim + ndim

    if (
        not _is_supported(output)
        or actual_dim != ndim - 1
        or input_dtype != output.dtype  # half_to_float case
    ):
        return _call_fallback(
            "_softmax_backward_data", grad_output, output, dim, input_dtype
        )

    input_shape = output.shape
    N = input_shape[-1]
    M = output.numel() // N

    dy = _reshape_2d(grad_output, M, N)
    y = _reshape_2d(output, M, N)

    mod = _get_mod("softmax")
    dx = mod.softmax_backward(dy, y)

    return dx.reshape(input_shape)


# =========================================================================
# Registration
# =========================================================================

_ATEN_LIB: torch.library.Library | None = None

# Mapping: (aten_op_name, impl_function)
_OVERRIDES = [
    ("_fused_rms_norm", _oink_fused_rms_norm),
    ("_fused_rms_norm_backward", _oink_fused_rms_norm_backward),
    ("native_layer_norm", _oink_native_layer_norm),
    ("native_layer_norm_backward", _oink_native_layer_norm_backward),
    ("_softmax", _oink_softmax),
    ("_softmax_backward_data", _oink_softmax_backward),
]


def override_all_aten_kernels() -> None:
    """Patch all supported aten ops on the CUDA dispatch key to use Oink's
    SM100 CuTeDSL kernels.

    Idempotent — safe to call multiple times.  Captures the original CUDA
    kernels before overriding so that unsupported inputs (wrong dtype, older
    GPU) fall back transparently.
    """
    global _ATEN_LIB
    if _ATEN_LIB is not None:
        return

    # Capture original kernels *before* we overwrite them.
    for op_name, _ in _OVERRIDES:
        _capture_fallback(op_name)

    lib = torch.library.Library("aten", "IMPL")
    registered = []

    for op_name, impl_fn in _OVERRIDES:
        try:
            lib.impl(op_name, impl_fn, "CUDA")
            registered.append(op_name)
        except Exception as e:
            logger.warning("Oink: could not override aten::%s: %s", op_name, e)

    _ATEN_LIB = lib
    logger.info("Oink: overrode %d aten ops on CUDA: %s", len(registered), registered)


def restore_all_aten_kernels() -> None:
    """Remove all Oink overrides and restore PyTorch's native CUDA kernels."""
    global _ATEN_LIB
    if _ATEN_LIB is None:
        return
    _ATEN_LIB = None
    logger.info("Oink: restored all aten ops to PyTorch defaults")


# Keep the old single-op API for backward compatibility.
override_aten_rmsnorm = override_all_aten_kernels
restore_aten_rmsnorm = restore_all_aten_kernels


__all__ = [
    "override_all_aten_kernels",
    "restore_all_aten_kernels",
    "override_aten_rmsnorm",
    "restore_aten_rmsnorm",
]
