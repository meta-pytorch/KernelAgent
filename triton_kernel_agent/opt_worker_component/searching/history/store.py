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

"""Protocol for program database storage.

The program database provides persistent storage for kernel programs
discovered during optimization. This enables:

- Resume: Continue optimization runs after interruption
- History: Track what was tried and what worked/failed
- Learning: Use past attempts to guide future exploration
- Analysis: Understand optimization trajectories post-hoc

Thread/process safety:
- Only the main optimization loop should write to the store
- Workers return results via queue; manager calls add_program()
"""

from typing import Protocol

from .models import ProgramEntry


class ProgramDatabase(Protocol):
    """Interface for storing and querying optimization programs.

    Implementations must provide:
    - add_program(): Store a new program entry
    - get_program(): Retrieve a program by ID
    - get_top_k(): Get best performers for parent selection
    - get_all(): Get all programs, optionally filtered
    - sample_inspirations(): Sample diverse programs for few-shot prompting
    - save() / load(): Persist and restore state
    - count(): Count total programs
    """

    def add_program(self, entry: ProgramEntry) -> str:
        """Add a program to the database.

        Args:
            entry: The program entry to add

        Returns:
            The program ID
        """
        ...

    def get_program(self, program_id: str) -> ProgramEntry | None:
        """Get a program by its ID.

        Args:
            program_id: The ID of the program to retrieve

        Returns:
            The program entry, or None if not found
        """
        ...

    def get_top_k(self, k: int, problem_id: str | None = None) -> list[ProgramEntry]:
        """Get the top-k programs by performance (lowest time_ms first).

        Args:
            k: Number of programs to return
            problem_id: Optional filter by problem ID

        Returns:
            List of top-k program entries sorted by time_ms ascending
        """
        ...

    def get_all(self, problem_id: str | None = None) -> list[ProgramEntry]:
        """Get all programs, optionally filtered by problem.

        Args:
            problem_id: Optional filter by problem ID

        Returns:
            List of all matching program entries
        """
        ...

    def sample_inspirations(
        self,
        n: int,
        exclude_ids: list[str] | None = None,
        problem_id: str | None = None,
    ) -> list[ProgramEntry]:
        """Sample diverse inspirations for few-shot prompting.

        Args:
            n: Number of inspirations to sample
            exclude_ids: Program IDs to exclude from sampling
            problem_id: Optional filter by problem ID

        Returns:
            List of sampled program entries
        """
        ...

    def save(self) -> None:
        """Persist the database to storage."""
        ...

    def load(self) -> None:
        """Load the database from storage."""
        ...

    def count(self, problem_id: str | None = None) -> int:
        """Count programs in the database.

        Args:
            problem_id: Optional filter by problem ID

        Returns:
            Number of programs
        """
        ...
