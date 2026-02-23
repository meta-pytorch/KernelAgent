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

from typing import Protocol

from ..history import AttemptRecord, ProgramDatabase


class Mutator(Protocol):
    """Interface for building optimization prompts."""

    def build_prompt(self, parent: AttemptRecord) -> str:
        """Build a prompt for the LLM to optimize the kernel.

        Args:
            parent: The kernel to optimize.
        """
        ...


class SimpleMutator:
    """Minimal mutator: basic prompt with kernel and history."""

    def __init__(self, store: ProgramDatabase) -> None:
        self.store = store

    def build_prompt(self, parent: AttemptRecord) -> str:
        lines = [
            "# Optimize this Triton kernel\n",
            f"Current performance: {parent.time_ms:.4f}ms\n",
        ]

        history = self.store.get_recent(3)
        if history:
            lines.append("\n## Recent attempts:\n")
            for a in history:
                lines.append(f"- [{a.outcome.value}] {a.time_ms:.4f}ms\n")

        lines.append("\n## Kernel:\n```python\n")
        lines.append(parent.kernel_code)
        lines.append("\n```\n")

        return "".join(lines)
