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

"""Tests for Oink's operator overrides.

Verifies that ``register_all_kernels`` / ``override_all_kernels``
properly patches Oink's kernels and their backward, and that the
overridden kernels produce numerically correct results.

Reference values are computed by calling the original aten CUDA kernel
captured via ``torch.library.get_kernel`` before the override is applied,
invoked with ``call_boxed(DispatchKeySet, ...)``.
"""

from __future__ import annotations

import types

import pytest
import torch

TEST_CUDA = torch.cuda.is_available()

_SM = 0
if TEST_CUDA:
    _major, _minor = torch.cuda.get_device_capability(0)
    _SM = 10 * _major + _minor
SM100_OR_LATER = _SM >= 100

requires_cuda = pytest.mark.skipif(not TEST_CUDA, reason="CUDA not available")
requires_sm100 = pytest.mark.skipif(not SM100_OR_LATER, reason="requires SM100+")


_CUDA_KS = None
_orig_kernels: dict[str, object] = {}

if TEST_CUDA:
    try:
        _CUDA_KS = torch.DispatchKeySet(torch.DispatchKey.CUDA)
        for _op in [
            "_fused_rms_norm",
            "_fused_rms_norm_backward",
        ]:
            _orig_kernels[_op] = torch.library.get_kernel(f"aten::{_op}", "CUDA")
    except Exception:
        pass

_OVERRIDE_APPLIED = False
if TEST_CUDA and SM100_OR_LATER:
    try:
        from kernelagent_oink.aten_override import override_all_kernels

        override_all_kernels()
        _OVERRIDE_APPLIED = True
    except Exception:
        pass

requires_override = pytest.mark.skipif(
    not _OVERRIDE_APPLIED, reason="override not applied"
)


SHAPES = [(8, 128), (4, 8, 32), (2, 16, 512), (4, 32, 1024)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
EPS = 1e-5


def _atol_for(dtype):
    if dtype == torch.bfloat16:
        return 1e-1  # bf16 has 8-bit mantissa, larger rounding error
    if dtype == torch.float16:
        return 1e-2  # fp16 has 11-bit mantissa
    return 1e-4      # fp32


@requires_cuda
@requires_sm100
def test_override_sets_library():
    """The Library object should be non-None after override."""
    from kernelagent_oink.aten_override import _OVERRIDE_LIB

    assert _OVERRIDE_LIB is not None, "override_all_kernels did not create Library"


@requires_cuda
@requires_sm100
def test_custom_ops_registered():
    """torch.ops.oink.rmsnorm should be callable after registration."""
    from kernelagent_oink import register_all_kernels

    register_all_kernels(force=True)
    assert hasattr(torch.ops, "oink"), "torch.ops.oink namespace missing"
    assert hasattr(torch.ops.oink, "rmsnorm"), "torch.ops.oink.rmsnorm missing"


def test_oink_availability_checks(monkeypatch: pytest.MonkeyPatch):
    """Probe _is_supported with mocked CUDA."""
    from kernelagent_oink.aten_override import _get_device_major, _is_supported

    fake_tensor = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # SM90 (Hopper) → not supported (SM100+ only).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    _get_device_major.cache_clear()
    assert _is_supported(fake_tensor) is False

    # SM100 → supported.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (10, 0))
    _get_device_major.cache_clear()
    assert _is_supported(fake_tensor) is True

    # float64 → not supported
    fake_f64 = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float64, device=torch.device("cuda:0")
    )
    assert _is_supported(fake_f64) is False

    _get_device_major.cache_clear()


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_fwd(dtype):
    atol = _atol_for(dtype)
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")

        # Oink override.
        y, rstd = torch.ops.aten._fused_rms_norm(x, normalized_shape, w, EPS)

        # Native aten reference.
        y_ref, rstd_ref = _orig_kernels["_fused_rms_norm"].call_boxed(
            _CUDA_KS, x, normalized_shape, w, EPS
        )

        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd y shape={shape} dtype={dtype}"
        )
        torch.testing.assert_close(
            rstd, rstd_ref, atol=_atol_for(rstd.dtype), rtol=0,
            msg=f"fwd rstd shape={shape} rstd_dtype={rstd.dtype}",
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_bwd(dtype):
    atol = _atol_for(dtype)
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        # Get rstd from native aten forward (needed by backward).
        _, rstd_ref = _orig_kernels["_fused_rms_norm"].call_boxed(
            _CUDA_KS, x, normalized_shape, w, EPS
        )

        # Oink override backward.
        dx, dw = torch.ops.aten._fused_rms_norm_backward(
            grad_out, x, normalized_shape, rstd_ref, w, [True, True]
        )

        # Native aten reference backward.
        dx_ref, dw_ref = _orig_kernels["_fused_rms_norm_backward"].call_boxed(
            _CUDA_KS, grad_out, x, normalized_shape, rstd_ref, w, [True, True]
        )

        torch.testing.assert_close(
            dx, dx_ref, atol=atol, rtol=0,
            msg=f"bwd dx shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            dw, dw_ref, atol=atol, rtol=0,
            msg=f"bwd dw shape={shape} dtype={dtype}",
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize(
    "mask", [[True, True], [True, False], [False, True], [False, False]]
)
def test_backward_output_mask(mask):
    """Backward output_mask behavior should match native aten exactly."""
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    grad = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")

    _, rstd = torch.ops.aten._fused_rms_norm(x, [128], w, EPS)

    # Oink override.
    dx, dw = torch.ops.aten._fused_rms_norm_backward(
        grad, x, [128], rstd, w, mask
    )

    # Native aten reference.
    _, rstd_ref = _orig_kernels["_fused_rms_norm"].call_boxed(_CUDA_KS, x, [128], w, EPS)
    dx_ref, dw_ref = _orig_kernels["_fused_rms_norm_backward"].call_boxed(
        _CUDA_KS, grad, x, [128], rstd_ref, w, mask
    )

    assert (dx is None) == (dx_ref is None), (
        f"dx None mismatch: oink={dx is None}, aten={dx_ref is None} for mask={mask}"
    )
    assert (dw is None) == (dw_ref is None), (
        f"dw None mismatch: oink={dw is None}, aten={dw_ref is None} for mask={mask}"
    )


@requires_cuda
@requires_sm100
@requires_override
def test_float64_rmsnorm_falls_back():
    """float64 is not supported by oink — should fall back gracefully."""
    x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
    w = torch.randn(32, dtype=torch.float64, device="cuda")
    y, rstd = torch.ops.aten._fused_rms_norm(x, [32], w, EPS)
    assert y.shape == x.shape
    assert y.dtype == torch.float64


@requires_cuda
@requires_sm100
def test_restore_then_reregister():
    """restore + re-register should work in the same process."""
    from kernelagent_oink import unregister_all_kernels
    from kernelagent_oink.aten_override import override_all_kernels

    unregister_all_kernels()

    # After unregister, re-register should succeed.
    override_all_kernels()

    from kernelagent_oink.aten_override import _OVERRIDE_LIB

    assert _OVERRIDE_LIB is not None
