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

"""Basic sanity tests for the optional KernelAgent-Oink plugin.

These tests are written to be safe on CPU-only CI:
- Oink itself is not installed as part of `pip install -e .`, so we import it
  by adding `./oink/src` to sys.path.
- GPU / CuTeDSL correctness tests are skipped unless the environment is
  properly configured (SM100 + torch + cuda-python + nvidia-cutlass-dsl).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _add_oink_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    oink_src = repo_root / "oink" / "src"
    sys.path.insert(0, str(oink_src))


def test_oink_import_and_register_no_raise():
    """Importing and calling register() must not raise on CPU-only envs."""
    _add_oink_to_syspath()

    import kernelagent_oink

    # Default behavior: no-op unless env var enabled.
    kernelagent_oink.register()

    # Standalone mode: still should not raise even if torch/cuda deps are missing.
    kernelagent_oink.register(force=True)


def test_oink_env_helpers(monkeypatch: pytest.MonkeyPatch):
    _add_oink_to_syspath()
    import kernelagent_oink

    monkeypatch.delenv("VLLM_USE_OINK_RMSNORM", raising=False)
    assert kernelagent_oink._env_truthy("VLLM_USE_OINK_RMSNORM") is False

    monkeypatch.setenv("VLLM_USE_OINK_RMSNORM", "1")
    assert kernelagent_oink._env_truthy("VLLM_USE_OINK_RMSNORM") is True

    monkeypatch.setenv("LOCAL_RANK", "7")
    assert kernelagent_oink._infer_cuda_device_index() == 7

    monkeypatch.setenv("LOCAL_RANK", "not_an_int")
    assert kernelagent_oink._infer_cuda_device_index() == 0


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed (CPU-only CI)",
)
def test_oink_rmsnorm_sm100_correctness():
    """Optional correctness check (skips unless SM100 + deps are available)."""
    import torch

    # Some environments may have a partial/namespace `torch` installed without
    # CUDA bindings.
    if not hasattr(torch, "cuda"):
        pytest.skip("torch.cuda not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires SM100 / Blackwell")

    # Optional deps required by the CuTeDSL path.
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("nvidia-cutlass-dsl / cutlass not installed")
    if importlib.util.find_spec("cuda") is None:
        pytest.skip("cuda-python not installed")

    _add_oink_to_syspath()
    import kernelagent_oink

    # Ensure the custom ops are registered (safe if already registered).
    kernelagent_oink.register(force=True)
    if not hasattr(torch.ops, "oink") or not hasattr(torch.ops.oink, "rmsnorm"):
        pytest.skip("oink custom ops not registered (missing deps or non-SM100)")

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    y = torch.ops.oink.rmsnorm(x, w, eps)

    # Reference: pure PyTorch RMSNorm in fp32 accumulation.
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y_ref = (xf * rstd * w.float()).to(x.dtype)

    assert y.shape == x.shape
    assert y.stride() == x.stride()
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
