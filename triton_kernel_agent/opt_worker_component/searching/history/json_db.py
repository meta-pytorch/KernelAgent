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

"""JSON file-based program database implementation.

Thread safety: Uses file locking for save operations.
Process safety: Only manager process should write; workers return results via queue.
"""

import fcntl
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .store import ProgramDatabase
from .models import ProgramEntry, ProgramMetrics


class JSONProgramDatabase(ProgramDatabase):
    """JSON file-based program database.

    This implementation stores all programs in a single JSON file with
    file locking for concurrent access safety.
    """

    def __init__(self, path: Path | str):
        """Initialize the JSON database.

        Args:
            path: Path to the JSON file for storage
        """
        self.path = Path(path)
        self.programs: dict[str, ProgramEntry] = {}

        if self.path.exists():
            self.load()

    def add_program(self, entry: ProgramEntry) -> str:
        """Add a program to the database."""
        self.programs[entry.program_id] = entry
        return entry.program_id

    def get_program(self, program_id: str) -> ProgramEntry | None:
        """Get a program by ID."""
        return self.programs.get(program_id)

    def get_top_k(self, k: int, problem_id: str | None = None) -> list[ProgramEntry]:
        """Get top-k programs by performance (lowest time_ms first)."""
        programs = list(self.programs.values())
        if problem_id:
            programs = [p for p in programs if p.problem_id == problem_id]
        programs.sort(key=lambda x: x.metrics.time_ms)
        return programs[:k]

    def get_all(self, problem_id: str | None = None) -> list[ProgramEntry]:
        """Get all programs, optionally filtered by problem."""
        programs = list(self.programs.values())
        if problem_id:
            programs = [p for p in programs if p.problem_id == problem_id]
        return programs

    def sample_inspirations(
        self, n: int, exclude_ids: list[str] | None = None, problem_id: str | None = None
    ) -> list[ProgramEntry]:
        """Sample diverse inspirations - mix of top performers and random."""
        exclude_ids = exclude_ids or []
        programs = [
            p
            for p in self.programs.values()
            if p.program_id not in exclude_ids
            and (problem_id is None or p.problem_id == problem_id)
        ]

        if len(programs) <= n:
            return programs

        # Take half from top performers, half random for diversity
        programs_sorted = sorted(programs, key=lambda x: x.metrics.time_ms)
        top_half = programs_sorted[: n // 2 + 1]
        remaining = [p for p in programs if p not in top_half]
        random_half = random.sample(remaining, min(n - len(top_half), len(remaining)))

        result = top_half[: n // 2] + random_half
        return result[:n]

    def count(self, problem_id: str | None = None) -> int:
        """Count programs in database."""
        if problem_id:
            return sum(1 for p in self.programs.values() if p.problem_id == problem_id)
        return len(self.programs)

    def save(self) -> None:
        """Save to JSON with file locking."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {"programs": [self._entry_to_dict(p) for p in self.programs.values()]}

        with open(self.path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load(self) -> None:
        """Load from JSON."""
        if not self.path.exists():
            return

        with open(self.path, "r") as f:
            data = json.load(f)

        for prog_dict in data.get("programs", []):
            entry = self._dict_to_entry(prog_dict)
            self.programs[entry.program_id] = entry

    def _entry_to_dict(self, entry: ProgramEntry) -> dict[str, Any]:
        """Convert ProgramEntry to dictionary for JSON serialization."""
        return {
            "program_id": entry.program_id,
            "kernel_code": entry.kernel_code,
            "metrics": {
                "time_ms": entry.metrics.time_ms,
            },
            "problem_id": entry.problem_id,
            "parent_id": entry.parent_id,
            "generation": entry.generation,
            "created_at": entry.created_at.isoformat(),
        }

    def _dict_to_entry(self, d: dict[str, Any]) -> ProgramEntry:
        """Convert dictionary to ProgramEntry."""
        metrics_dict = d.get("metrics", {})
        metrics = ProgramMetrics(
            time_ms=metrics_dict.get("time_ms", float("inf")),
        )

        created_at = d.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        return ProgramEntry(
            program_id=d["program_id"],
            kernel_code=d["kernel_code"],
            metrics=metrics,
            problem_id=d["problem_id"],
            parent_id=d.get("parent_id"),
            generation=d.get("generation", 0),
            created_at=created_at,
        )
