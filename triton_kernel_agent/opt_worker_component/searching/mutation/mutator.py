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

"""Mutation strategies for generating kernel optimization prompts.

Mutators build prompts for the LLM to optimize kernels, including:
- The parent kernel to improve
- History of what was tried before
- Any additional context (bottleneck analysis, inspirations, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..history import AttemptRecord


class Mutator(Protocol):
    """Interface for building optimization prompts."""

    def build_prompt(
        self,
        parent: AttemptRecord,
        history: list[AttemptRecord] | None = None,
    ) -> str:
        """Build a prompt for the LLM to optimize the kernel.

        Args:
            parent: The kernel to optimize.
            history: Previous attempts, ordered oldest-first.
        """
        ...


class SimpleMutator:
    """Minimal mutator: basic prompt with kernel and history."""

    def build_prompt(
        self,
        parent: AttemptRecord,
        history: list[AttemptRecord] | None = None,
    ) -> str:
        lines = [
            "# Optimize this Triton kernel\n",
            f"Current performance: {parent.time_ms:.4f}ms\n",
        ]

        if history:
            lines.append("\n## Recent attempts:\n")
            for a in history[-3:]:
                lines.append(f"- [{a.outcome.value}] {a.time_ms:.4f}ms\n")

        lines.append("\n## Kernel:\n```python\n")
        lines.append(parent.kernel_code)
        lines.append("\n```\n")

        return "".join(lines)
