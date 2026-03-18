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
"""
Generate per-subgraph monkey-patch test scripts.

Each test script loads the fused PyTorch model, patches one subgraph module's
forward() to call the Triton kernel_function from the worker directory, then
runs the user's additional test against the full model.  This lets the
KernelAgent refinement loop catch regressions at the full-model level.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Regex to strip ``from kernel import kernel_function`` (with optional alias)
_IMPORT_KERNEL_RE = re.compile(
    r"^\s*from\s+kernel\s+import\s+kernel_function\b[^\n]*$",
    re.MULTILINE,
)


def _parse_where_parts(where: str) -> list[str]:
    """Split a dotted ``where`` string into navigation tokens.

    Example: ``"stacked.layers[0].input_proj"``
    → ``["stacked", "layers", "[0]", "input_proj"]``
    """
    parts: list[str] = []
    for segment in where.split("."):
        # Handle indexed access like ``layers[0]``
        idx = segment.find("[")
        if idx != -1:
            name = segment[:idx]
            if name:
                parts.append(name)
            # Capture all bracket expressions (e.g. [0][1])
            rest = segment[idx:]
            while rest:
                end = rest.find("]")
                if end == -1:
                    break
                parts.append(rest[: end + 1])
                rest = rest[end + 1 :]
        else:
            if segment:
                parts.append(segment)
    return parts


def _weight_param_names(item: dict[str, Any]) -> list[str]:
    """Return the list of weight parameter attribute names from the subgraph item.

    These correspond to ``nn.Parameter`` / buffer attribute names on the target
    module, and are also the extra parameters (after inputs) expected by the
    subgraph ``kernel_function``.
    """
    wf = item.get("weights_fused") or item.get("weights_original") or item.get("weights") or {}
    if isinstance(wf, dict):
        return list(wf.keys())
    return []


def _count_forward_inputs(item: dict[str, Any]) -> int:
    """Estimate the number of tensor inputs to the subgraph's forward().

    Multi-input subgraphs have an ``inputs`` list; single-input ones have
    ``input_shape``.
    """
    inputs_multi = item.get("inputs")
    if isinstance(inputs_multi, list) and inputs_multi:
        return len(inputs_multi)
    return 1


def build_monkeypatch_test(
    fused_code_path: Path,
    subgraph_item: dict[str, Any],
    user_test_code: str,
    target_platform: str = "cuda",
) -> str:
    """Return a self-contained Python test script string.

    The script:
    1. Imports the subgraph kernel as ``_subgraph_kfn``.
    2. Loads and instantiates the fused PyTorch model.
    3. Patches the target subgraph module's ``forward()`` to call the kernel.
    4. Defines a module-level ``kernel_function`` wrapping the patched model.
    5. Appends the user's test logic (with the kernel import line stripped).
    """
    where = str(subgraph_item.get("where", ""))
    where_parts = _parse_where_parts(where) if where else []
    weight_names = _weight_param_names(subgraph_item)
    n_forward_inputs = _count_forward_inputs(subgraph_item)
    source = subgraph_item.get("source") or {}
    module_class = source.get("module", "")

    # Device string
    device = "cuda" if target_platform == "cuda" else target_platform

    # Strip the ``from kernel import kernel_function`` line from user test code
    user_test_body = _IMPORT_KERNEL_RE.sub("", user_test_code)

    # Build the script
    lines: list[str] = []
    lines.append("import sys, types, inspect")
    lines.append("import torch")
    lines.append("")
    lines.append("# 1. Import subgraph kernel from worker directory")
    lines.append("from kernel import kernel_function as _subgraph_kfn")
    lines.append("")
    lines.append("# 2. Load fused model")
    lines.append("_ns = {}")
    lines.append(f'with open({str(fused_code_path.resolve())!r}, "r") as _f:')
    lines.append("    exec(_f.read(), _ns)")
    lines.append(
        f'_model = _ns["Model"](*_ns.get("get_init_inputs", lambda: ())()).to({device!r}).eval()'
    )
    lines.append("")

    # 3. Navigate to the target module
    lines.append("# 3. Navigate to and patch the target subgraph module")
    if where_parts:
        lines.append("_target = _model")
        for part in where_parts:
            if part.startswith("[") and part.endswith("]"):
                lines.append(f"_target = _target[{part[1:-1]}]")
            else:
                lines.append(f"_target = getattr(_target, {part!r})")
    else:
        # Fallback: search by module class name
        lines.append("# Fallback: locate module by class name")
        lines.append("_target = None")
        if module_class:
            lines.append(f"_module_class_name = {module_class!r}")
        else:
            lines.append("_module_class_name = ''")
        lines.append("for _name, _mod in _model.named_modules():")
        lines.append("    if type(_mod).__name__ == _module_class_name:")
        lines.append("        _target = _mod")
        lines.append("        break")
        lines.append("if _target is None:")
        lines.append("    _target = _model  # last resort: patch model itself")
    lines.append("")

    # 4. Build parameter mapping and patched forward
    lines.append("# 4. Build parameter mapping and patch forward")
    lines.append(f"_weight_param_names = {weight_names!r}")
    lines.append(f"_n_inputs = {n_forward_inputs}")
    lines.append("")
    lines.append("def _patched_forward(self, *args, **kwargs):")
    lines.append("    weight_tensors = [getattr(self, n) for n in _weight_param_names]")
    lines.append("    return _subgraph_kfn(*args, *weight_tensors)")
    lines.append("")
    lines.append("_target.forward = types.MethodType(_patched_forward, _target)")
    lines.append("")

    # 5. Full model wrapper
    lines.append("# 5. Full model wrapper")
    lines.append("def kernel_function(*args):")
    lines.append("    with torch.no_grad():")
    lines.append("        return _model(*args)")
    lines.append("")

    # 6. User's test body
    lines.append("# 6. User's additional test")
    lines.append(user_test_body)
    lines.append("")

    return "\n".join(lines)
