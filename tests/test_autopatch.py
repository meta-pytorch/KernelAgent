#!/usr/bin/env python3
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

"""Tests for XPU path: _auto_patch_common_triton_issues and platform guidance."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernel_agent.platform_config import get_platform


class TestAutoPatchCudaHacksXPU:
    """Tests for CUDA hack stripping on XPU platform."""

    def test_xpu_strips_cuda_is_available_hack(self):
        """XPU should strip torch.cuda.is_available hack."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = "torch.cuda.is_available = lambda: True"
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert "torch.cuda.is_available = lambda: True" not in patched

    def test_xpu_strips_fake_torch_device(self):
        """XPU should strip _fake_torch_device function."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = """_real_torch_device = torch.device
def _fake_torch_device(arg):
    return _real_torch_device(arg)
torch.device = _fake_torch_device
"""
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert "def _fake_torch_device" not in patched
        assert "_real_torch_device" not in patched

    def test_xpu_strips_triton_backends_env(self):
        """XPU should strip TRITON_BACKENDS environment variable hack."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = 'os.environ["TRITON_BACKENDS"] = "cuda"'
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

    def test_xpu_no_change_when_no_hacks_present(self):
        """XPU should not modify code without CUDA hacks."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = "import torch\nx = torch.randn(10, device='xpu')"
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is False
        assert patched == code

    def test_xpu_strips_multiple_hacks(self):
        """XPU should strip multiple CUDA hacks in same code."""
        from Fuser.compose_end_to_end import _auto_patch_common_triton_issues

        code = """torch.cuda.is_available = lambda: True
os.environ["TRITON_BACKENDS"] = "cuda"
"""
        platform = get_platform("xpu")
        patched, changed = _auto_patch_common_triton_issues(code, platform)

        assert changed is True
        assert "torch.cuda.is_available = lambda: True" not in patched
        assert 'os.environ["TRITON_BACKENDS"]' not in patched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
