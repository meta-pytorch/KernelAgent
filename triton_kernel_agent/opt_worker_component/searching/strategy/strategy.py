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

"""Strategies for controlling the optimization search loop.

Strategies decide:
- Which parent to optimize next
- When to stop (convergence, plateau, max rounds)
- How to select the next generation of candidates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..history import AttemptRecord, AttemptStore
    from ..sampling import Sampler


class Strategy(Protocol):
    """Interface for optimization loop control."""

    def next_parent(self) -> AttemptRecord | None:
        """Select the next parent to optimize from."""
        ...

    def record_result(self, attempt: AttemptRecord) -> None:
        """Record an optimization attempt result."""
        ...

    def get_best(self) -> AttemptRecord | None:
        """Get the best attempt so far."""
        ...

    def should_stop(self, round_num: int, max_rounds: int) -> bool:
        """Check if optimization should terminate early."""
        ...


class SimpleStrategy:
    """Example Implementation: always pick best, stop at max rounds."""

    def __init__(self, store: AttemptStore, sampler: Sampler) -> None:
        self.store = store
        self.sampler = sampler

    def next_parent(self) -> AttemptRecord | None:
        return self.sampler.sample_parent()

    def record_result(self, attempt: AttemptRecord) -> None:
        self.store.add(attempt)

    def get_best(self) -> AttemptRecord | None:
        return self.store.get_best()

    def should_stop(self, round_num: int, max_rounds: int) -> bool:
        return round_num >= max_rounds
