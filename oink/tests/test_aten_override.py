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

"""Tests for Oink's aten operator overrides.

Verifies that ``register_all_kernels`` / ``override_all_aten_kernels``
properly patches PyTorch's aten ops and that the overridden kernels produce
numerically correct results compared to PyTorch's native CUDA kernels.

The correctness tests capture the original CUDA kernel *before* the override
is applied, then compare the override's output against it.
"""

from __future__ import annotations

import types

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

TEST_CUDA = torch.cuda.is_available()

_SM = 0
if TEST_CUDA:
    _major, _minor = torch.cuda.get_device_capability(0)
    _SM = 10 * _major + _minor

SM100_OR_LATER = _SM >= 100

requires_cuda = pytest.mark.skipif(not TEST_CUDA, reason="CUDA not available")
requires_sm100 = pytest.mark.skipif(not SM100_OR_LATER, reason="requires SM100+")

# ---------------------------------------------------------------------------
# Capture original CUDA kernels *before* any override is applied.
# ---------------------------------------------------------------------------

_orig_fused_rms_norm = None
_orig_fused_rms_norm_bwd = None
_orig_native_layer_norm = None
_orig_native_layer_norm_bwd = None
_orig_softmax = None
_orig_softmax_bwd = None

if TEST_CUDA:
    try:
        _orig_fused_rms_norm = torch.library.get_kernel(
            "aten::_fused_rms_norm", "CUDA"
        )
        _orig_fused_rms_norm_bwd = torch.library.get_kernel(
            "aten::_fused_rms_norm_backward", "CUDA"
        )
        _orig_native_layer_norm = torch.library.get_kernel(
            "aten::native_layer_norm", "CUDA"
        )
        _orig_native_layer_norm_bwd = torch.library.get_kernel(
            "aten::native_layer_norm_backward", "CUDA"
        )
        _orig_softmax = torch.library.get_kernel("aten::_softmax", "CUDA")
        _orig_softmax_bwd = torch.library.get_kernel(
            "aten::_softmax_backward_data", "CUDA"
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Apply the override (module-level, happens once).
# ---------------------------------------------------------------------------

_OVERRIDE_APPLIED = False

if TEST_CUDA and SM100_OR_LATER:
    try:
        from kernelagent_oink.aten_override import override_all_aten_kernels

        override_all_aten_kernels()
        _OVERRIDE_APPLIED = True
    except Exception:
        pass

requires_override = pytest.mark.skipif(
    not _OVERRIDE_APPLIED, reason="override not applied"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPES = [(8, 128), (4, 8, 32), (2, 16, 512), (4, 32, 1024)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
EPS = 1e-5


def _atol_for(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-1
    return 1e-5


# =========================================================================
# Registration tests
# =========================================================================


@requires_cuda
@requires_sm100
def test_override_sets_library():
    """The aten Library object should be non-None after override."""
    from kernelagent_oink.aten_override import _ATEN_LIB

    assert _ATEN_LIB is not None, "override_all_aten_kernels did not create Library"


@requires_cuda
@requires_sm100
def test_all_fallbacks_captured():
    """All 6 fallback kernels should have been captured."""
    from kernelagent_oink.aten_override import _fallbacks

    expected_ops = [
        "_fused_rms_norm",
        "_fused_rms_norm_backward",
        "native_layer_norm",
        "native_layer_norm_backward",
        "_softmax",
        "_softmax_backward_data",
    ]
    for op in expected_ops:
        assert op in _fallbacks, f"fallback not captured for {op}"
        assert _fallbacks[op] is not None, f"fallback is None for {op}"


@requires_cuda
@requires_sm100
def test_custom_ops_registered():
    """torch.ops.oink.rmsnorm should be callable after registration."""
    from kernelagent_oink import register_all_kernels

    register_all_kernels(force=True)
    assert hasattr(torch.ops, "oink"), "torch.ops.oink namespace missing"
    assert hasattr(torch.ops.oink, "rmsnorm"), "torch.ops.oink.rmsnorm missing"


# =========================================================================
# Availability / stride-guard tests (no GPU required for some)
# =========================================================================


def test_oink_availability_checks(monkeypatch: pytest.MonkeyPatch):
    """Probe is_oink_available_for_device with mocked CUDA."""
    from kernelagent_oink.aten_override import _is_supported

    # Mock a CUDA tensor with SM90 (below threshold).
    fake_tensor = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float16, device=torch.device("cuda:0")
    )

    # SM90 → not supported.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    # Clear the cached SM value.
    from kernelagent_oink.aten_override import _get_device_sm

    _get_device_sm.cache_clear()
    assert _is_supported(fake_tensor) is False

    # SM100 → supported.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (10, 0))
    _get_device_sm.cache_clear()
    assert _is_supported(fake_tensor) is True

    # float64 → not supported even on SM100.
    fake_f64 = types.SimpleNamespace(
        is_cuda=True, dtype=torch.float64, device=torch.device("cuda:0")
    )
    assert _is_supported(fake_f64) is False

    _get_device_sm.cache_clear()


def test_can_view_as_2d_stride_guard():
    """Verify _can_view_as_2d correctly identifies non-viewable layouts."""
    from kernelagent_oink.aten_override import _can_view_as_2d

    x = torch.zeros((2, 3, 4))
    assert _can_view_as_2d(x) is True

    # Size-1 dims should be ignored by the viewability check.
    base = torch.zeros((2, 10, 4))
    x_singleton = base[:, :1, :]
    assert _can_view_as_2d(x_singleton) is True

    # Middle-dimension stride break: view(-1, hidden) should be invalid.
    x2 = x[:, ::2, :]
    with pytest.raises(RuntimeError):
        x2.view(-1, x2.shape[-1])
    assert _can_view_as_2d(x2) is False


def test_is_oink_stride_compatible_2d():
    """Verify vectorization alignment check."""
    from kernelagent_oink.aten_override import _is_oink_stride_compatible_2d

    # Standard contiguous tensor (stride(0)==N, stride(1)==1) → compatible.
    x = torch.zeros(4, 128, dtype=torch.float16)
    assert _is_oink_stride_compatible_2d(x) is True

    # Padded row: stride(0) % 16 == 0 → compatible.
    base = torch.zeros(4, 256, dtype=torch.float16)
    x_padded = base[:, :128]  # stride(0)=256, stride(1)=1
    assert x_padded.stride(0) == 256
    assert _is_oink_stride_compatible_2d(x_padded) is True

    # 1D tensor → not compatible.
    assert _is_oink_stride_compatible_2d(torch.zeros(128)) is False

    # Wrong dtype → not compatible.
    assert _is_oink_stride_compatible_2d(torch.zeros(4, 128, dtype=torch.float64)) is False


# =========================================================================
# Correctness tests — RMSNorm
# =========================================================================


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

        y, rstd = torch.ops.aten._fused_rms_norm(x, normalized_shape, w, EPS)
        y_ref, rstd_ref = _orig_fused_rms_norm(x, normalized_shape, w, EPS)

        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd y shape={shape} dtype={dtype}"
        )
        torch.testing.assert_close(
            rstd, rstd_ref, atol=1e-5, rtol=0,
            msg=f"fwd rstd shape={shape} dtype={dtype}",
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_bwd(dtype):
    atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        x1 = x.detach().requires_grad_(True)
        w1 = w.detach().requires_grad_(True)
        y1, _ = torch.ops.aten._fused_rms_norm(x1, normalized_shape, w1, EPS)
        y1.backward(grad_out)

        x2 = x.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        y2, _ = _orig_fused_rms_norm(x2, normalized_shape, w2, EPS)
        y2.backward(grad_out)

        torch.testing.assert_close(
            x1.grad, x2.grad, atol=atol, rtol=0,
            msg=f"bwd x_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            w1.grad, w2.grad, atol=atol, rtol=0,
            msg=f"bwd w_grad shape={shape} dtype={dtype}",
        )


# =========================================================================
# Correctness tests — LayerNorm
# =========================================================================


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm_fwd(dtype):
    atol = _atol_for(dtype)
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        b = torch.randn(*normalized_shape, dtype=dtype, device="cuda")

        out, mean, rstd = torch.ops.aten.native_layer_norm(
            x, normalized_shape, w, b, EPS
        )
        out_ref, mean_ref, rstd_ref = _orig_native_layer_norm(
            x, normalized_shape, w, b, EPS
        )

        torch.testing.assert_close(
            out, out_ref, atol=atol, rtol=0, msg=f"fwd shape={shape} dtype={dtype}"
        )
        torch.testing.assert_close(
            mean, mean_ref, atol=1e-5, rtol=0,
            msg=f"fwd mean shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            rstd, rstd_ref, atol=1e-5, rtol=0,
            msg=f"fwd rstd shape={shape} dtype={dtype}",
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm_bwd(dtype):
    atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
    for shape in SHAPES:
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        b = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        x1 = x.detach().requires_grad_(True)
        w1 = w.detach().requires_grad_(True)
        b1 = b.detach().requires_grad_(True)
        out1, _, _ = torch.ops.aten.native_layer_norm(
            x1, normalized_shape, w1, b1, EPS
        )
        out1.backward(grad_out)

        x2 = x.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        b2 = b.detach().requires_grad_(True)
        out2, _, _ = _orig_native_layer_norm(x2, normalized_shape, w2, b2, EPS)
        out2.backward(grad_out)

        torch.testing.assert_close(
            x1.grad, x2.grad, atol=atol, rtol=0,
            msg=f"bwd x_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            w1.grad, w2.grad, atol=atol, rtol=0,
            msg=f"bwd w_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            b1.grad, b2.grad, atol=atol, rtol=0,
            msg=f"bwd b_grad shape={shape} dtype={dtype}",
        )


# =========================================================================
# Correctness tests — Softmax
# =========================================================================


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_fwd(dtype):
    atol = _atol_for(dtype)
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=dtype, device="cuda")

        y = torch.ops.aten._softmax(x, -1, False)
        y_ref = _orig_softmax(x, -1, False)

        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd shape={shape} dtype={dtype}"
        )


@requires_cuda
@requires_sm100
@requires_override
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_bwd(dtype):
    atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
    for shape in SHAPES:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        x1 = x.detach().requires_grad_(True)
        y1 = torch.softmax(x1, dim=-1)
        y1.backward(grad_out)

        x2 = x.detach().requires_grad_(True)
        y2 = _orig_softmax(x2, -1, False)
        dx_ref = _orig_softmax_bwd(grad_out, y2, -1, dtype)

        torch.testing.assert_close(
            x1.grad, dx_ref, atol=atol, rtol=0,
            msg=f"bwd shape={shape} dtype={dtype}",
        )


# =========================================================================
# Fallback tests
# =========================================================================


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
@requires_override
def test_float64_layernorm_falls_back():
    x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
    w = torch.randn(32, dtype=torch.float64, device="cuda")
    b = torch.randn(32, dtype=torch.float64, device="cuda")
    out, mean, rstd = torch.ops.aten.native_layer_norm(x, [32], w, b, EPS)
    assert out.shape == x.shape
    assert out.dtype == torch.float64


@requires_cuda
@requires_sm100
@requires_override
def test_float64_softmax_falls_back():
    x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
    y = torch.ops.aten._softmax(x, -1, False)
    assert y.shape == x.shape
    assert y.dtype == torch.float64


@requires_cuda
@requires_sm100
@requires_override
def test_non_last_dim_softmax_falls_back():
    """Softmax on dim=0 should fall back (oink only handles last dim)."""
    x = torch.randn(4, 32, dtype=torch.float16, device="cuda")
    y = torch.ops.aten._softmax(x, 0, False)
    assert y.shape == x.shape
    y_ref = _orig_softmax(x, 0, False)
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=0)
