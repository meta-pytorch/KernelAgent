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

"""Verify that default YAML configs stay consistent with their class __init__ defaults."""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from Fuser.auto_agent import AutoKernelRouter


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Add new (cls, yaml_path) pairs here to cover more configs.
_CONFIG_PAIRS = [
    (AutoKernelRouter, _REPO_ROOT / "Fuser" / "config" / "autoagent_default.yml"),
]


def _get_init_defaults(cls):
    """Extract default values from a class's __init__ signature."""
    init = cls.__init__
    if hasattr(init, "__wrapped__"):
        init = init.__wrapped__
    sig = inspect.signature(init)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if name != "self" and param.default is not param.empty
    }


@pytest.mark.parametrize(
    "cls, yaml_path",
    _CONFIG_PAIRS,
    ids=[cls.__name__ for cls, _ in _CONFIG_PAIRS],
)
def test_yaml_matches_init_defaults(cls, yaml_path):
    """YAML keys and values must exactly match __init__ defaults."""
    yaml_data = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    init_defaults = _get_init_defaults(cls)

    assert set(yaml_data.keys()) == set(init_defaults.keys()), (
        f"Key mismatch — missing from YAML: {set(init_defaults.keys()) - set(yaml_data.keys())}, "
        f"extra in YAML: {set(yaml_data.keys()) - set(init_defaults.keys())}"
    )

    mismatches = {
        k: {"yaml": yaml_data[k], "init": init_defaults[k]}
        for k in init_defaults
        if yaml_data[k] != init_defaults[k]
    }
    assert not mismatches, f"Value mismatches: {mismatches}"
