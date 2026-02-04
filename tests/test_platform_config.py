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

"""Tests for PlatformConfig module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernel_agent.platform_config import (
    DEFAULT_PLATFORM,
    PLATFORMS,
    get_platform,
    get_platform_choices,
)


class TestGetPlatform:
    """Tests for the get_platform() function."""

    def test_get_platform_xpu(self):
        """get_platform('xpu') should return XPU config."""
        config = get_platform("xpu")
        assert config.name == "xpu"
        assert config.device_string == "xpu"

    def test_get_platform_invalid_raises_valueerror(self):
        """get_platform() with invalid name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_platform("invalid_platform")
        assert "Unknown platform" in str(exc_info.value)
        assert "invalid_platform" in str(exc_info.value)

    def test_get_platform_error_shows_available(self):
        """ValueError message should list available platforms."""
        with pytest.raises(ValueError) as exc_info:
            get_platform("nonexistent")
        error_msg = str(exc_info.value)
        assert "cuda" in error_msg
        assert "xpu" in error_msg

    def test_get_platform_case_sensitive(self):
        """Platform names should be case-sensitive."""
        with pytest.raises(ValueError):
            get_platform("CUDA")
        with pytest.raises(ValueError):
            get_platform("Xpu")


class TestGetPlatformChoices:
    """Tests for the get_platform_choices() function."""

    def test_get_platform_choices_contains_cuda_and_xpu(self):
        """Choices should include both cuda and xpu."""
        choices = get_platform_choices()
        assert "cuda" in choices
        assert "xpu" in choices

    def test_get_platform_choices_is_sorted(self):
        """Choices should be sorted alphabetically."""
        choices = get_platform_choices()
        assert choices == sorted(choices)

    def test_get_platform_choices_matches_platforms_registry(self):
        """Choices should match PLATFORMS registry keys."""
        choices = get_platform_choices()
        assert set(choices) == set(PLATFORMS.keys())


class TestPlatformConfigs:
    """Tests for CUDA and XPU platform configurations."""

    def test_cuda_device_string(self):
        """CUDA device_string should be 'cuda'."""
        config = get_platform("cuda")
        assert config.device_string == "cuda"

    def test_cuda_has_empty_guidance(self):
        """CUDA should have empty guidance (no special instructions needed)."""
        config = get_platform("cuda")
        assert config.guidance_block == ""
        assert config.kernel_guidance == ""

    def test_cuda_has_no_hacks_to_strip(self):
        """CUDA should have no CUDA hacks to strip."""
        config = get_platform("cuda")
        assert config.cuda_hacks_to_strip == ()

    def test_xpu_has_guidance_block(self):
        """XPU should have non-empty guidance block mentioning xpu."""
        config = get_platform("xpu")
        assert config.guidance_block != ""
        assert "xpu" in config.guidance_block.lower()

    def test_xpu_guidance_warns_against_cuda(self):
        """XPU guidance should warn against using cuda."""
        config = get_platform("xpu")
        assert "cuda" in config.guidance_block.lower()

    def test_xpu_has_kernel_guidance(self):
        """XPU should have kernel-specific guidance."""
        config = get_platform("xpu")
        assert config.kernel_guidance != ""

    def test_xpu_has_cuda_hacks_to_strip(self):
        """XPU should have CUDA hacks to strip from generated code."""
        config = get_platform("xpu")
        assert len(config.cuda_hacks_to_strip) > 0
        assert isinstance(config.cuda_hacks_to_strip, tuple)


class TestDefaultPlatform:
    """Tests for default platform behavior."""

    def test_default_platform_is_cuda(self):
        """DEFAULT_PLATFORM should be 'cuda'."""
        assert DEFAULT_PLATFORM == "cuda"

    def test_default_platform_is_valid(self):
        """DEFAULT_PLATFORM should be a valid platform."""
        config = get_platform(DEFAULT_PLATFORM)
        assert config is not None
        assert config.name == DEFAULT_PLATFORM


class TestPlatformRegistry:
    """Tests for the PLATFORMS registry."""

    def test_platform_names_match_keys(self):
        """Each PlatformConfig.name should match its registry key."""
        for key, config in PLATFORMS.items():
            assert config.name == key

    def test_all_platforms_accessible(self):
        """All platforms in registry should be accessible via get_platform."""
        for name in get_platform_choices():
            config = get_platform(name)
            assert config.name == name
            assert config.device_string in ["cuda", "xpu"]


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string_platform_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            get_platform("")

    def test_none_platform_raises(self):
        """None should raise appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            get_platform(None)

    def test_whitespace_platform_raises(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError):
            get_platform("   ")

    def test_platform_with_extra_whitespace_raises(self):
        """Platform name with whitespace should raise ValueError."""
        with pytest.raises(ValueError):
            get_platform(" cuda ")


@pytest.mark.parametrize("platform_name", ["cuda", "xpu"])
def test_all_platforms_have_consistent_structure(platform_name):
    """All platforms should have consistent field types."""
    config = get_platform(platform_name)
    assert isinstance(config.name, str)
    assert isinstance(config.device_string, str)
    assert isinstance(config.guidance_block, str)
    assert isinstance(config.kernel_guidance, str)
    assert isinstance(config.cuda_hacks_to_strip, tuple)


@pytest.mark.parametrize("platform_name", ["cuda", "xpu"])
def test_platform_name_equals_device_string(platform_name):
    """Platform name should equal device string for simplicity."""
    config = get_platform(platform_name)
    assert config.name == config.device_string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
