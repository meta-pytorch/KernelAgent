#!/usr/bin/env python3
"""Tests for XPU path: _auto_patch_common_triton_issues and TARGET PLATFORM prompt block."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernel_agent.platform_config import get_platform


class TestAutoPatchCudaHacksXPU:
    """Tests for CUDA hack stripping on XPU platform."""

    def test_xpu_strips_cuda_is_available_hack(self):
        """XPU should strip torch.cuda.is_available hack."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = (
            "import torch\ntorch.cuda.is_available = lambda: True\nx = torch.randn(10)"
        )
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert "torch.cuda.is_available = lambda: True" not in patched
        assert "torch.randn" in patched

    def test_xpu_strips_fake_torch_device(self):
        """XPU should strip _fake_torch_device function."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = """_real_torch_device = torch.device
def _fake_torch_device(arg):
    return _real_torch_device(arg)

torch.device = _fake_torch_device
x = torch.randn(10)
"""
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert "def _fake_torch_device" not in patched
        assert "torch.randn" in patched

    def test_xpu_strips_triton_backends_env(self):
        """XPU should strip TRITON_BACKENDS environment variable hack."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = 'os.environ["TRITON_BACKENDS"] = "cuda"\nimport triton'
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert 'os.environ["TRITON_BACKENDS"]' not in patched

    def test_cuda_platform_does_not_strip_hacks(self):
        """CUDA platform should NOT strip any hacks."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = "torch.cuda.is_available = lambda: True"
        platform = get_platform("cuda")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert "torch.cuda.is_available = lambda: True" in patched


class TestTargetPlatformPromptBlock:
    """Tests for TARGET PLATFORM block in prompts."""

    def test_cuda_has_empty_guidance(self):
        """CUDA platform should have empty guidance block."""
        platform = get_platform("cuda")
        assert platform.guidance_block == ""
        assert platform.kernel_guidance == ""

    def test_xpu_has_guidance_block(self):
        """XPU platform should have non-empty guidance block."""
        platform = get_platform("xpu")
        assert platform.guidance_block != ""
        assert "xpu" in platform.guidance_block.lower()

    def test_xpu_guidance_warns_against_cuda_device(self):
        """XPU guidance should warn against using cuda device."""
        platform = get_platform("xpu")
        assert "cuda" in platform.guidance_block.lower()
        assert "xpu" in platform.guidance_block.lower()

    def test_xpu_kernel_guidance_exists(self):
        """XPU should have kernel-specific guidance."""
        platform = get_platform("xpu")
        assert platform.kernel_guidance != ""
        assert "Intel" in platform.kernel_guidance or "XPU" in platform.kernel_guidance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
