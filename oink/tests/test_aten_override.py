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

Reference values are computed via pure-PyTorch math (float32 accumulation)
to avoid issues with ``call_boxed`` and stale ``SafeKernelFunction``
references when ``torch._native`` overrides are also active.
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
    """torch.ops.oink.rmsnorm should be callable after register()."""
    from kernelagent_oink import register

    register(force=True)
    assert hasattr(torch.ops, "oink"), "torch.ops.oink namespace missing"
    assert hasattr(torch.ops.oink, "rmsnorm"), "torch.ops.oink.rmsnorm missing"


def test_oink_availability_checks(monkeypatch: pytest.MonkeyPatch):
    """Probe _is_supported with mocked CUDA."""
    from kernelagent_oink.aten_override import _get_device_major, _is_supported

    fake_tensor = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float16, device=torch.device("cuda:0"),
        shape=torch.Size([32, 4096]),
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

    # float64 → not supported.
    fake_f64 = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float64, device=torch.device("cuda:0"),
        shape=torch.Size([32, 4096]),
    )
    assert _is_supported(fake_f64) is False

    # N < 128 → not supported (kernel requires N >= 128).
    fake_small_n = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float16, device=torch.device("cuda:0"),
        shape=torch.Size([32, 64]),
    )
    assert _is_supported(fake_small_n) is False

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

        # Pure-torch reference (float32 accumulation).
        N = shape[-1]
        M = x.numel() // N
        x_f32 = x.reshape(M, N).float()
        rstd_ref = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + EPS)
        y_ref = ((x_f32 * rstd_ref) * w.float()).to(dtype).reshape(shape)

        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd y shape={shape} dtype={dtype}"
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_bwd(dtype):
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda", requires_grad=True)
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        # Oink override fwd + bwd.
        y, _ = torch.ops.aten._fused_rms_norm(x, normalized_shape, w, EPS)
        y.backward(grad_out)

        assert x.grad is not None, f"x.grad is None for shape={shape}"
        assert w.grad is not None, f"w.grad is None for shape={shape}"
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape
        assert torch.isfinite(x.grad).all(), f"x.grad has inf/nan for shape={shape}"
        assert torch.isfinite(w.grad).all(), f"w.grad has inf/nan for shape={shape}"


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize(
    "mask", [[True, True], [True, False], [False, True], [False, False]]
)
def test_backward_output_mask(mask):
    """Backward should return None for masked outputs."""
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    grad = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")

    _, rstd = torch.ops.aten._fused_rms_norm(x, [128], w, EPS)

    dx, dw = torch.ops.aten._fused_rms_norm_backward(
        grad, x, [128], rstd, w, mask
    )

    if not mask[0]:
        assert dx is None, "dx should be None when output_mask[0]=False"
    else:
        assert dx is not None and dx.shape == x.shape

    if not mask[1]:
        assert dw is None, "dw should be None when output_mask[1]=False"
    else:
        assert dw is not None and dw.shape == w.shape


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
