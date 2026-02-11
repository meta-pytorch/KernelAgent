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
Torch custom ops wrapping Oink's Blackwell RMSNorm kernels.

These ops are designed to be:
- Architecture-aware (use CuTeDSL SM100 kernels when available, fall back
  to a safe reference elsewhere).
- Layout-preserving for 2D row-major inputs, including padded MLA-style
  layouts where stride(0) > N and stride(1) == 1.
- torch.compile-friendly via proper fake implementations that mirror
  runtime shapes and strides.

Public ops (Python signatures):

  torch.ops.oink.rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor
      Functional RMSNorm. Returns a new tensor with the same shape and
      stride as x when using the fast CuTeDSL path.

  torch.ops.oink.fused_add_rms_norm(
      x: Tensor, residual: Tensor, weight: Tensor, eps: float
  ) -> None
      In-place fused residual-add + RMSNorm matching vLLM semantics:
          residual = x + residual   (stored into `residual`)
          x = RMSNorm(residual, w)  (stored into `x`)
      Mutates `x` and `residual` in-place and returns None.
"""

from __future__ import annotations

import importlib
import threading

import torch
from torch.library import custom_op

_RMSNORM_MOD: object | None = None
_RMSNORM_MOD_LOCK = threading.Lock()


def _get_rmsnorm_mod():
    """Lazy import to keep plugin registration lightweight.

    Importing the CuTeDSL kernel stack can be expensive and may require a CUDA
    context. We defer it until the first actual execution of the custom op.
    """
    global _RMSNORM_MOD

    cached = _RMSNORM_MOD
    if cached is not None:
        return cached

    with _RMSNORM_MOD_LOCK:
        if _RMSNORM_MOD is None:
            _RMSNORM_MOD = importlib.import_module("kernelagent_oink.blackwell.rmsnorm")
        return _RMSNORM_MOD


def _get_sm(device: torch.device | None = None) -> int:
    """Return SM version as an int (e.g., 100 for SM100 / Blackwell)."""
    if device is None:
        device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    return 10 * major + minor


#
# RMSNorm (functional)
#


@custom_op("oink::rmsnorm", mutates_args=())
def oink_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Functional RMSNorm entrypoint.

    This op is model-agnostic. It expects a 2D [M, N] view of the input
    where the last dimension is contiguous (stride(1) == 1). The leading
    dimension stride(0) may be larger than N (padded-row layouts), and
    will be preserved on the fast CuTeDSL path.

    On SM100 (and newer), this dispatches to the tuned CuTeDSL Blackwell
    RMSNorm kernel in rmsnorm.rmsnorm_forward, which in turn selects the
    best internal schedule (including DSv3-specific stage-2 kernels where
    applicable) and preserves the input's 2D stride when using the
    pointer-based path.

    On older architectures it falls back to a safe PyTorch reference
    implementation for correctness.
    """
    assert x.is_cuda, "oink::rmsnorm requires CUDA tensors"
    assert x.dim() == 2, "oink::rmsnorm expects a 2D [M, N] tensor view"
    assert weight.dim() == 1, "weight must be 1D [N]"

    sm = _get_sm(x.device)
    _rms = _get_rmsnorm_mod()
    if sm >= 100:
        # Use the tuned CuTeDSL SM100 kernel. The public API already
        # contains all necessary gating and layout checks internally.
        y, _rstd, _res = _rms.rmsnorm_forward(
            x,
            weight=weight,
            bias=None,
            residual=None,
            eps=eps,
            store_rstd=False,
        )
        return y

    # Fallback: reference implementation (correctness-first).
    return _rms.rmsnorm_ref(
        x,
        w=weight,
        b=None,
        residual=None,
        eps=eps,
    )


@oink_rmsnorm.register_fake
def oink_rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Fake (meta) implementation for oink::rmsnorm.

    We must preserve x's logical layout (shape + stride) so that Inductor's
    CUDA graph capture sees the same stride contract as the real kernel.
    """
    # x is a FakeTensor here; x.shape/x.stride()/x.device/x.dtype are defined.
    return torch.empty_strided(
        x.shape,
        x.stride(),
        device=x.device,
        dtype=x.dtype,
    )


#
# Fused residual-add + RMSNorm (in-place, vLLM semantics)
#


@custom_op("oink::fused_add_rms_norm", mutates_args=("x", "residual"))
def oink_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """
    In-place fused residual-add + RMSNorm:

        residual <- x + residual
        x <- RMSNorm(residual, weight, eps)

    Returns:
        None (mutates `x` and `residual` in-place).
    """
    assert x.is_cuda and residual.is_cuda, (
        "oink::fused_add_rms_norm requires CUDA tensors"
    )
    assert x.shape == residual.shape, "x and residual must have the same shape"
    assert x.dtype == residual.dtype, "x and residual must have the same dtype"
    assert weight.dim() == 1, "weight must be 1D [N]"

    sm = _get_sm(x.device)
    _rms = _get_rmsnorm_mod()

    if sm < 100:
        # Non-SM100 fallback: keep semantics in-place (correctness-first).
        residual.add_(x)
        y = _rms.rmsnorm_ref(residual, w=weight, b=None, residual=None, eps=eps)
        x.copy_(y)
        return None

    # SM100+: prefer the lowest-overhead in-place entrypoint (returns None).
    if hasattr(_rms, "fused_add_rmsnorm_inplace_"):
        _rms.fused_add_rmsnorm_inplace_(  # type: ignore[misc]
            x,
            residual,
            weight,
            eps=eps,
        )
        return None

    # Backward-compatible wrapper (returns (x, residual)).
    if hasattr(_rms, "fused_add_rmsnorm_forward_inplace"):
        _rms.fused_add_rmsnorm_forward_inplace(  # type: ignore[misc]
            x,
            residual,
            weight,
            eps=eps,
        )
        return None

    # Extremely defensive fallback if the Oink module doesn't provide
    # the in-place entrypoint.
    y, z = _rms.fused_add_rmsnorm_forward(x, residual, weight, eps=eps)
    x.copy_(y)
    residual.copy_(z)
    return None


@oink_fused_add_rms_norm.register_fake
def oink_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """
    Fake (meta) implementation for oink::fused_add_rms_norm.

    Because this op mutates its inputs in-place, the outputs alias the input
    buffers and therefore have the same shapes and strides.
    """
    return None


__all__ = [
    "oink_rmsnorm",
    "oink_fused_add_rms_norm",
]
