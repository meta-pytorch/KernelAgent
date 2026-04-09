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
is applied, then compare the override's output against it.  This mirrors the
approach used in PyTorch's own quack override tests.
"""

from __future__ import annotations

import math
import unittest

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
# Apply the override (module-level so it happens once).
# ---------------------------------------------------------------------------

_OVERRIDE_APPLIED = False

if TEST_CUDA and SM100_OR_LATER:
    try:
        from kernelagent_oink.aten_override import (
            _ATEN_LIB,
            _fallbacks,
            override_all_aten_kernels,
        )

        override_all_aten_kernels()
        _OVERRIDE_APPLIED = True
    except Exception:
        pass


# =========================================================================
# Registration tests
# =========================================================================


class TestAtenOverrideRegistration(unittest.TestCase):
    """Verify that override_all_aten_kernels sets up state correctly."""

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
    def test_override_sets_library(self):
        """The aten Library object should be non-None after override."""
        from kernelagent_oink.aten_override import _ATEN_LIB

        self.assertIsNotNone(
            _ATEN_LIB, "override_all_aten_kernels did not create the Library"
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
    def test_all_fallbacks_captured(self):
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
            self.assertIn(op, _fallbacks, f"fallback not captured for {op}")
            self.assertIsNotNone(_fallbacks[op], f"fallback is None for {op}")

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
    def test_custom_ops_registered(self):
        """torch.ops.oink.rmsnorm should be callable after registration."""
        from kernelagent_oink import register_all_kernels

        register_all_kernels(force=True)
        self.assertTrue(
            hasattr(torch.ops, "oink"), "torch.ops.oink namespace missing"
        )
        self.assertTrue(
            hasattr(torch.ops.oink, "rmsnorm"), "torch.ops.oink.rmsnorm missing"
        )


# =========================================================================
# Correctness tests — RMSNorm
# =========================================================================

SHAPES = [(8, 128), (4, 8, 32), (2, 16, 512), (4, 32, 1024)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
EPS = 1e-5


def _atol_for(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-1
    return 1e-5


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
@unittest.skipIf(not _OVERRIDE_APPLIED, "override not applied")
class TestRMSNormOverride(unittest.TestCase):
    """Compare oink RMSNorm override against the captured ATen fallback."""

    def _run_fwd(self, shape, dtype):
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")

        # Oink (through overridden aten op)
        y, rstd = torch.ops.aten._fused_rms_norm(x, normalized_shape, w, EPS)

        # Reference (captured original kernel)
        y_ref, rstd_ref = _orig_fused_rms_norm(x, normalized_shape, w, EPS)

        atol = _atol_for(dtype)
        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd y shape={shape} dtype={dtype}"
        )
        torch.testing.assert_close(
            rstd,
            rstd_ref,
            atol=1e-5,
            rtol=0,
            msg=f"fwd rstd shape={shape} dtype={dtype}",
        )

    def test_fwd_fp16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float16)

    def test_fwd_bf16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.bfloat16)

    def test_fwd_fp32(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float32)

    def _run_bwd(self, shape, dtype):
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        # Oink
        x1 = x.detach().requires_grad_(True)
        w1 = w.detach().requires_grad_(True)
        y1, _ = torch.ops.aten._fused_rms_norm(x1, normalized_shape, w1, EPS)
        y1.backward(grad_out)

        # Reference
        x2 = x.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        y2, _ = _orig_fused_rms_norm(x2, normalized_shape, w2, EPS)
        y2.backward(grad_out)

        atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
        torch.testing.assert_close(
            x1.grad,
            x2.grad,
            atol=atol,
            rtol=0,
            msg=f"bwd x_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            w1.grad,
            w2.grad,
            atol=atol,
            rtol=0,
            msg=f"bwd w_grad shape={shape} dtype={dtype}",
        )

    def test_bwd_fp16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float16)

    def test_bwd_bf16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.bfloat16)

    def test_bwd_fp32(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float32)


# =========================================================================
# Correctness tests — LayerNorm
# =========================================================================


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
@unittest.skipIf(not _OVERRIDE_APPLIED, "override not applied")
class TestLayerNormOverride(unittest.TestCase):
    """Compare oink LayerNorm override against the captured ATen fallback."""

    def _run_fwd(self, shape, dtype):
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        b = torch.randn(*normalized_shape, dtype=dtype, device="cuda")

        # Oink
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            x, normalized_shape, w, b, EPS
        )

        # Reference
        out_ref, mean_ref, rstd_ref = _orig_native_layer_norm(
            x, normalized_shape, w, b, EPS
        )

        atol = _atol_for(dtype)
        torch.testing.assert_close(
            out, out_ref, atol=atol, rtol=0, msg=f"fwd shape={shape} dtype={dtype}"
        )
        torch.testing.assert_close(
            mean,
            mean_ref,
            atol=1e-5,
            rtol=0,
            msg=f"fwd mean shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            rstd,
            rstd_ref,
            atol=1e-5,
            rtol=0,
            msg=f"fwd rstd shape={shape} dtype={dtype}",
        )

    def test_fwd_fp16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float16)

    def test_fwd_bf16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.bfloat16)

    def test_fwd_fp32(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float32)

    def _run_bwd(self, shape, dtype):
        normalized_shape = [shape[-1]]
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        w = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        b = torch.randn(*normalized_shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        # Oink
        x1 = x.detach().requires_grad_(True)
        w1 = w.detach().requires_grad_(True)
        b1 = b.detach().requires_grad_(True)
        out1, _, _ = torch.ops.aten.native_layer_norm(
            x1, normalized_shape, w1, b1, EPS
        )
        out1.backward(grad_out)

        # Reference
        x2 = x.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        b2 = b.detach().requires_grad_(True)
        out2, _, _ = _orig_native_layer_norm(x2, normalized_shape, w2, b2, EPS)
        out2.backward(grad_out)

        atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
        torch.testing.assert_close(
            x1.grad,
            x2.grad,
            atol=atol,
            rtol=0,
            msg=f"bwd x_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            w1.grad,
            w2.grad,
            atol=atol,
            rtol=0,
            msg=f"bwd w_grad shape={shape} dtype={dtype}",
        )
        torch.testing.assert_close(
            b1.grad,
            b2.grad,
            atol=atol,
            rtol=0,
            msg=f"bwd b_grad shape={shape} dtype={dtype}",
        )

    def test_bwd_fp16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float16)

    def test_bwd_bf16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.bfloat16)

    def test_bwd_fp32(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float32)


# =========================================================================
# Correctness tests — Softmax
# =========================================================================


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
@unittest.skipIf(not _OVERRIDE_APPLIED, "override not applied")
class TestSoftmaxOverride(unittest.TestCase):
    """Compare oink Softmax override against the captured ATen fallback."""

    def _run_fwd(self, shape, dtype):
        x = torch.randn(*shape, dtype=dtype, device="cuda")

        # Oink (dim=-1, half_to_float=False)
        y = torch.ops.aten._softmax(x, -1, False)

        # Reference
        y_ref = _orig_softmax(x, -1, False)

        atol = _atol_for(dtype)
        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=0, msg=f"fwd shape={shape} dtype={dtype}"
        )

    def test_fwd_fp16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float16)

    def test_fwd_bf16(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.bfloat16)

    def test_fwd_fp32(self):
        for shape in SHAPES:
            self._run_fwd(shape, torch.float32)

    def _run_bwd(self, shape, dtype):
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        grad_out = torch.randn(*shape, dtype=dtype, device="cuda")

        # Oink
        x1 = x.detach().requires_grad_(True)
        y1 = torch.softmax(x1, dim=-1)
        y1.backward(grad_out)

        # Reference (manual softmax + bwd to avoid the override)
        x2 = x.detach().requires_grad_(True)
        y2 = _orig_softmax(x2, -1, False)
        dx_ref = _orig_softmax_bwd(grad_out, y2, -1, dtype)

        atol = 3e-1 if dtype == torch.bfloat16 else _atol_for(dtype)
        torch.testing.assert_close(
            x1.grad,
            dx_ref,
            atol=atol,
            rtol=0,
            msg=f"bwd shape={shape} dtype={dtype}",
        )

    def test_bwd_fp16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float16)

    def test_bwd_bf16(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.bfloat16)

    def test_bwd_fp32(self):
        for shape in SHAPES:
            self._run_bwd(shape, torch.float32)


# =========================================================================
# Fallback tests
# =========================================================================


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM100_OR_LATER, "requires SM100+")
@unittest.skipIf(not _OVERRIDE_APPLIED, "override not applied")
class TestFallback(unittest.TestCase):
    """Verify that unsupported inputs fall back to the native CUDA kernel."""

    def test_float64_rmsnorm_falls_back(self):
        """float64 is not supported by oink — should fall back gracefully."""
        x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
        w = torch.randn(32, dtype=torch.float64, device="cuda")
        y, rstd = torch.ops.aten._fused_rms_norm(x, [32], w, EPS)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, torch.float64)

    def test_float64_layernorm_falls_back(self):
        x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
        w = torch.randn(32, dtype=torch.float64, device="cuda")
        b = torch.randn(32, dtype=torch.float64, device="cuda")
        out, mean, rstd = torch.ops.aten.native_layer_norm(x, [32], w, b, EPS)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float64)

    def test_float64_softmax_falls_back(self):
        x = torch.randn(4, 32, dtype=torch.float64, device="cuda")
        y = torch.ops.aten._softmax(x, -1, False)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, torch.float64)

    def test_non_last_dim_softmax_falls_back(self):
        """Softmax on dim=0 should fall back (oink only handles last dim)."""
        x = torch.randn(4, 32, dtype=torch.float16, device="cuda")
        y = torch.ops.aten._softmax(x, 0, False)
        self.assertEqual(y.shape, x.shape)
        # Verify correctness: softmax on dim=0
        y_ref = _orig_softmax(x, 0, False)
        torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
