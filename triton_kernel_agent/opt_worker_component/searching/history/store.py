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

"""Storage interface and implementations for optimization attempts.

The attempt store provides persistent storage for kernel optimization
attempts discovered during the search process. This enables:

- Resume: Continue optimization runs after interruption
- History: Track what was tried and what worked/failed
- Learning: Use past attempts to guide future exploration
- Analysis: Understand optimization trajectories post-hoc

Thread/process safety:
- Only the main optimization loop should write to the store
- Workers return results via queue; manager calls add()
"""

import json
from pathlib import Path
from typing import Protocol

from .records import AttemptRecord, Outcome


class AttemptStore(Protocol):
    """Interface for storing and querying optimization attempts.

    Implementations must provide:
    - add(): Store a new attempt
    - get_recent(): Get recent attempts for history context
    - get_top_k(): Get best performers for parent selection
    - get_best(): Get single best attempt
    - count(): Count total attempts
    """

    def add(self, attempt: AttemptRecord) -> None:
        """Store an attempt."""
        ...

    def get_recent(self, n: int) -> list[AttemptRecord]:
        """Get the n most recent attempts (oldest first)."""
        ...

    def get_top_k(self, k: int) -> list[AttemptRecord]:
        """Get the k best attempts by time_ms (fastest first)."""
        ...

    def get_best(self) -> AttemptRecord | None:
        """Get the attempt with the lowest time_ms."""
        ...

    def count(self) -> int:
        """Count total attempts in the store."""
        ...


class JsonAttemptStore:
    """JSON file-based implementation of AttemptStore."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._attempts: list[AttemptRecord] = []
        self._load()

    def _load(self) -> None:
        """Load attempts from JSON file if it exists.

        Falls back to empty store if the file is corrupted (e.g., partial write).
        """
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._attempts = [AttemptRecord.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError) as e:
                import warnings

                warnings.warn(f"Corrupted store at {self.path}, starting fresh: {e}")
                self._attempts = []

    def _save(self) -> None:
        """Save attempts to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([a.to_dict() for a in self._attempts], f, indent=2)

    def add(self, attempt: AttemptRecord) -> None:
        """Store an attempt and persist to disk."""
        self._attempts.append(attempt)
        self._save()

    def get_recent(self, n: int) -> list[AttemptRecord]:
        """Get the n most recent attempts (oldest first)."""
        return self._attempts[-n:]

    def get_top_k(self, k: int) -> list[AttemptRecord]:
        """Get the k best attempts by time_ms (fastest first).

        Ties are broken by created_at (oldest first) for deterministic ordering.
        """
        valid = [a for a in self._attempts if a.outcome != Outcome.FAILED]
        sorted_by_time = sorted(valid, key=lambda a: (a.time_ms, a.created_at))
        return sorted_by_time[:k]

    def get_best(self) -> AttemptRecord | None:
        """Get the attempt with the lowest time_ms (excluding failed)."""
        valid = [a for a in self._attempts if a.outcome != Outcome.FAILED]
        if not valid:
            return None
        return min(valid, key=lambda a: a.time_ms)

    def count(self) -> int:
        """Count total attempts in the store."""
        return len(self._attempts)
