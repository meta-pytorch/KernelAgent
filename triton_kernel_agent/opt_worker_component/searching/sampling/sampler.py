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

"""Sampling strategies for selecting parents and inspirations from history.

Samplers control how we select:
- Parents: Which kernel to optimize next
- Inspirations: Which kernels to show as few-shot examples to the LLM
"""

from typing import Any, Protocol

from ..history.records import AttemptRecord


class Sampler(Protocol):
    """Interface for sampling from optimization history."""

    def sample_parent(self) -> AttemptRecord | None:
        """Select a parent for the next optimization attempt."""
        ...

    def get_top_inspirations(
        self,
        n: int,
    ) -> list[AttemptRecord]:
        """Get top-performing attempts for few-shot prompting."""
        ...


class BestSampler:
    """Sampler that always returns the best parent and top-k inspirations."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def sample_parent(self) -> AttemptRecord | None:
        return self.store.get_best()

    def get_top_inspirations(
        self,
        n: int,
    ) -> list[AttemptRecord]:
        return self.store.get_top_k(n)
