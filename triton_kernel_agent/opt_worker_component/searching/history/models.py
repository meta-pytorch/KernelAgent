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

"""Data models for the optimization program database.

This module defines the core data structures used to track kernel programs
and their performance metrics throughout the optimization process.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProgramMetrics:
    """Performance metrics for a program."""

    time_ms: float


@dataclass
class ProgramEntry:
    """A program in the database with metadata."""

    program_id: str
    kernel_code: str
    metrics: ProgramMetrics

    # Lineage
    problem_id: str
    parent_id: str | None = None
    generation: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"ProgramEntry(id={self.program_id}, time={self.metrics.time_ms:.4f}ms, "
            f"gen={self.generation}, parent={self.parent_id})"
        )
