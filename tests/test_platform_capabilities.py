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

"""Tests for backend capabilities in the generation path.

Host-only: no accelerator, no torch, no network, no model call. Where a test
needs ``torch``, it injects a stub, which is what lets the emitted harness be
*executed* here rather than only inspected.

The snapshot tests compare against the template as it was before capabilities
were introduced, rendered through the same Jinja environment. That is stronger
than a hand-written golden string: it cannot drift from the historical
behaviour it claims to preserve, and it does not depend on anybody reasoning
correctly about ``trim_blocks``.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from triton_kernel_agent.platform_config import (
    DEFAULT_NUM_WORKERS,
    DEFAULT_PLATFORM,
    get_platform,
    get_platform_choices,
    PLATFORMS,
    PlatformConfig,
    register_platform,
    UnknownPlatformError,
)
from triton_kernel_agent.prompt_manager import PromptManager

# The device block exactly as it stood before backend capabilities existed.
# Rendering this and the current template must produce identical text for every
# backend that existed then.
_LEGACY_DEVICE_BLOCK = """\
        # Device setup
        device = "{{ device_string }}"
{% if device_string == "xpu" %}
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU not available. Install PyTorch with Intel GPU support.")
{% else %}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
{% endif %}

        # Create test data
"""

# The same region of the current template, kept in step with test_generation.j2.
_CURRENT_DEVICE_BLOCK = """\
        # Device setup
        device = "{{ device_string }}"
{% if device_setup %}{{ device_setup | indent(8, true) }}
{% endif %}
{{ availability_check | indent(8, true) }}

        # Create test data
"""

_PRE_CAPABILITY_PLATFORMS = ("cuda", "xpu")


def _env() -> Environment:
    """A Jinja environment configured exactly as PromptManager configures its own."""
    return Environment(trim_blocks=True, lstrip_blocks=True)


def _render_block(source: str, platform: PlatformConfig) -> str:
    """Render one template fragment for one backend."""
    return _env().from_string(source).render(
        device_string=platform.device_string,
        availability_check=platform.availability_check,
        device_setup=platform.device_setup,
        test_prelude=platform.test_prelude,
    )


def _stub_torch(*, cuda_available: bool = False, xpu_available: bool = False):
    """A torch stand-in with just enough surface for an availability check."""
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available, synchronize=lambda: None
    )
    torch.cuda = cuda
    if xpu_available:
        torch.xpu = types.SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: None
        )
    return torch


def _exec_check(platform: PlatformConfig, torch_stub) -> None:
    """Execute a backend's availability check against a stubbed torch.

    Raises whatever the emitted code raises, which is the point: this proves the
    check is valid Python *and* that it actually refuses an unavailable device.
    """
    namespace = {"torch": torch_stub}
    exec(compile(platform.availability_check, "<availability_check>", "exec"), namespace)


class SnapshotTest(unittest.TestCase):
    """Existing backends must render byte-identically."""

    def test_the_device_block_is_unchanged_for_every_pre_capability_backend(
        self,
    ) -> None:
        for name in _PRE_CAPABILITY_PLATFORMS:
            platform = get_platform(name)
            legacy = _render_block(_LEGACY_DEVICE_BLOCK, platform)
            current = _render_block(_CURRENT_DEVICE_BLOCK, platform)
            self.assertEqual(current, legacy, f"{name} device block drifted")

    def test_the_snapshot_would_catch_a_drift(self) -> None:
        # A snapshot that cannot fail proves nothing. A backend whose check
        # differs from the legacy if/else must produce different text.
        drifted = PlatformConfig(
            name="drifted",
            device_string="cuda",
            guidance_block="",
            kernel_guidance="",
            availability_check="pass",
        )
        self.assertNotEqual(
            _render_block(_CURRENT_DEVICE_BLOCK, drifted),
            _render_block(_LEGACY_DEVICE_BLOCK, drifted),
        )

    def test_the_full_prompt_still_contains_the_backends_check(self) -> None:
        # Guards against the template fragment above drifting out of step with
        # the real file: this renders test_generation.j2 itself.
        for name in _PRE_CAPABILITY_PLATFORMS:
            platform = get_platform(name)
            prompt = PromptManager(target_platform=platform).render_test_generation_prompt(
                "add two tensors"
            )
            self.assertIn(platform.availability_check.splitlines()[0], prompt)
            self.assertIn(f'device = "{platform.device_string}"', prompt)

    def test_the_template_no_longer_branches_on_a_device_name(self) -> None:
        # The defect being removed: an if/else over two known names routes every
        # third backend down the CUDA path without saying so.
        source = (_templates_dir() / "test_generation.j2").read_text(encoding="utf-8")
        self.assertNotIn('device_string == "xpu"', source)
        self.assertNotIn("torch.cuda.is_available()", source)


def _templates_dir() -> Path:
    """Locate the bundled templates directory."""
    import triton_kernel_agent

    return Path(triton_kernel_agent.__file__).parent / "templates"


class CapabilityTest(unittest.TestCase):
    def test_every_registered_backend_declares_an_availability_check(self) -> None:
        # A backend with no check emits an empty statement block, which is a
        # syntax error inside the generated `try:`.
        for name, platform in PLATFORMS.items():
            self.assertTrue(platform.availability_check.strip(), name)

    def test_every_availability_check_is_valid_python(self) -> None:
        for name, platform in PLATFORMS.items():
            with self.subTest(platform=name):
                compile(platform.availability_check, "<check>", "exec")

    def test_an_accelerator_check_refuses_an_unavailable_device(self) -> None:
        # Proves the emitted check is not vacuous.
        with self.assertRaises(RuntimeError):
            _exec_check(get_platform("cuda"), _stub_torch(cuda_available=False))
        with self.assertRaises(RuntimeError):
            _exec_check(get_platform("xpu"), _stub_torch(cuda_available=True))

    def test_an_accelerator_check_passes_when_the_device_is_there(self) -> None:
        _exec_check(get_platform("cuda"), _stub_torch(cuda_available=True))
        _exec_check(get_platform("xpu"), _stub_torch(xpu_available=True))

    def test_synchronization_is_declared_per_backend(self) -> None:
        self.assertEqual(get_platform("cuda").synchronize_call, "torch.cuda.synchronize()")
        self.assertEqual(get_platform("xpu").synchronize_call, "torch.xpu.synchronize()")
        # Empty is a real answer for a backend with nothing to wait for, not a
        # gap to be filled with the CUDA call.
        self.assertEqual(get_platform("fake").synchronize_call, "")

    def test_concurrency_is_declared_per_backend(self) -> None:
        for name in _PRE_CAPABILITY_PLATFORMS:
            self.assertEqual(
                get_platform(name).default_num_workers, DEFAULT_NUM_WORKERS, name
            )
        self.assertEqual(get_platform("fake").default_num_workers, 1)


class FakeBackendTest(unittest.TestCase):
    """The fake backend must drive the pipeline with no accelerator present."""

    def test_it_is_registered_and_selectable(self) -> None:
        self.assertIn("fake", get_platform_choices())
        self.assertEqual(get_platform("fake").device_string, "cpu")

    def test_its_check_executes_with_no_device_and_no_torch_calls(self) -> None:
        # The stub reports nothing available; the fake backend must still pass.
        _exec_check(get_platform("fake"), _stub_torch(cuda_available=False))

    def test_it_generates_a_harness_that_runs_on_the_host(self) -> None:
        # Take the emitted device setup and availability check, assemble the
        # harness the template describes, and actually execute it.
        platform = get_platform("fake")
        rendered = _render_block(_CURRENT_DEVICE_BLOCK, platform)
        # The template renders into a `def test_kernel(): try:` body, so the
        # emitted block sits at eight spaces. Reproduce that exactly.
        # Jinja drops the template's final newline, so rejoin explicitly or the
        # last rendered line swallows whatever is appended to it.
        harness = (
            "def test_kernel():\n"
            "    try:\n"
            + rendered.rstrip("\n")
            + "\n        return device\n"
            "    except Exception:\n"
            "        raise\n"
        )
        namespace = {"torch": _stub_torch()}
        exec(compile(harness, "<harness>", "exec"), namespace)
        self.assertEqual(namespace["test_kernel"](), "cpu")

    def test_its_guidance_says_its_numbers_are_not_measurements(self) -> None:
        # Naming matters here: a backend that fabricates timings must not be
        # mistaken for one that measures them.
        guidance = get_platform("fake").guidance_block
        self.assertIn("no accelerator", guidance)
        self.assertIn("is a performance claim", guidance)
        self.assertIn("Nothing generated for it", guidance)

    def test_the_full_prompt_renders_for_it(self) -> None:
        prompt = PromptManager(
            target_platform=get_platform("fake")
        ).render_test_generation_prompt("add two tensors")
        self.assertIn('device = "cpu"', prompt)
        self.assertNotIn("torch.cuda.is_available()", prompt)


class RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(PLATFORMS)

    def tearDown(self) -> None:
        PLATFORMS.clear()
        PLATFORMS.update(self._saved)

    def _config(self, name: str = "outside") -> PlatformConfig:
        return PlatformConfig(
            name=name,
            device_string=name,
            guidance_block="",
            kernel_guidance="",
            availability_check="pass",
        )

    def test_a_backend_defined_elsewhere_becomes_selectable(self) -> None:
        # The seam that lets a backend outside this repository be used without
        # editing anything in it.
        register_platform(self._config())
        self.assertIn("outside", get_platform_choices())
        self.assertIs(get_platform("outside"), PLATFORMS["outside"])

    def test_registering_over_an_existing_backend_needs_saying_so(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            register_platform(self._config("cuda"))
        self.assertIn("already registered", str(ctx.exception))
        register_platform(self._config("cuda"), replace=True)
        self.assertEqual(get_platform("cuda").device_string, "cuda")

    def test_a_nameless_backend_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            register_platform(self._config(" "))

    def test_an_unknown_backend_raises_rather_than_defaulting_to_cuda(self) -> None:
        # Returning CUDA for an unrecognized name would generate CUDA code under
        # another backend's name.
        with self.assertRaises(UnknownPlatformError) as ctx:
            get_platform("nonexistent")
        self.assertIn("Available:", str(ctx.exception))
        # Still a ValueError, so existing callers keep working.
        self.assertIsInstance(ctx.exception, ValueError)

    def test_the_default_platform_constant_is_the_one_actually_used(self) -> None:
        # DEFAULT_PLATFORM used to be declared and then ignored in favour of
        # hardcoded "cuda" literals.
        self.assertIn(DEFAULT_PLATFORM, PLATFORMS)
        import triton_kernel_agent.agent as agent_module

        source = Path(agent_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn('get_platform("cuda")', source)


class NoInternalImportsTest(unittest.TestCase):
    def test_the_generic_tree_imports_nothing_meta_internal(self) -> None:
        # The exported package must stay installable outside fbsource.
        import triton_kernel_agent

        root = Path(triton_kernel_agent.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for marker in ("kernelagent.fb", "from fb.", "import fb.", "libfb"):
                if marker in text:
                    offenders.append(f"{path.name}: {marker}")
        self.assertEqual(offenders, [])

    def test_platform_config_needs_only_the_standard_library(self) -> None:
        # It is imported by every entry point, including ones with no jinja2.
        import triton_kernel_agent.platform_config as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in ("import torch", "import jinja2", "import numpy"):
            self.assertNotIn(banned, source)


class ExecutedHarnessIsolationTest(unittest.TestCase):
    def test_executing_a_harness_does_not_import_real_torch(self) -> None:
        # Guards the host-only claim: if these tests ever start importing torch
        # for real, they stop being runnable on a machine without it.
        before = "torch" in sys.modules
        with tempfile.TemporaryDirectory():
            _exec_check(get_platform("fake"), _stub_torch())
        self.assertEqual("torch" in sys.modules, before)


class RegistryConsistencyTest(unittest.TestCase):
    """Mirrors the invariants in the pytest-only test_platform_config.py.

    That file cannot be collected by python_unittest, so the facts it asserts
    are re-asserted here where they are actually run.
    """

    def test_every_config_name_matches_its_registry_key(self) -> None:
        for key, config in PLATFORMS.items():
            self.assertEqual(config.name, key)

    def test_every_registered_backend_is_reachable_and_named(self) -> None:
        for name in get_platform_choices():
            config = get_platform(name)
            self.assertEqual(config.name, name)
            # Not an allow-list of device strings: the registry is extensible.
            self.assertTrue(config.device_string)

    def test_choices_match_the_registry_and_are_sorted(self) -> None:
        choices = get_platform_choices()
        self.assertEqual(set(choices), set(PLATFORMS))
        self.assertEqual(choices, sorted(choices))

    def test_every_backend_has_the_full_capability_surface(self) -> None:
        for name in get_platform_choices():
            config = get_platform(name)
            with self.subTest(platform=name):
                self.assertIsInstance(config.guidance_block, str)
                self.assertIsInstance(config.kernel_guidance, str)
                self.assertIsInstance(config.cuda_hacks_to_strip, tuple)
                self.assertIsInstance(config.availability_check, str)
                self.assertIsInstance(config.device_setup, str)
                self.assertIsInstance(config.synchronize_call, str)
                self.assertIsInstance(config.test_prelude, str)
                self.assertIsInstance(config.default_num_workers, int)
                self.assertGreater(config.default_num_workers, 0)

    def test_a_blank_or_padded_name_is_refused(self) -> None:
        for bad in ("", "   ", " cuda "):
            with self.assertRaises(ValueError):
                get_platform(bad)
