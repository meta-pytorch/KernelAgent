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

"""Protocol for search strategies controlling the optimization loop.

Strategies decide:
- Which parent to optimize next (candidate selection)
- How many workers to spawn per round
- When to stop (convergence, plateau, max rounds)
"""

import logging
from typing import Any, Protocol

from ..history.models import ProgramEntry


class SearchStrategy(Protocol):
    """Interface for optimization loop control.

    Implementations must provide:
    - initialize(): Set up with starting program
    - select_candidates(): Choose programs to explore this round
    - update_with_results(): Process worker results
    - get_best_program(): Return best found so far
    - should_terminate(): Check for early stopping
    - num_workers_needed: Property for worker count
    """

    logger: logging.Logger
    problem_id: str | None

    def initialize(self, initial_program: ProgramEntry) -> None:
        """Initialize the strategy with a starting program.

        Must set self.problem_id from initial_program.problem_id.

        Args:
            initial_program: The initial kernel program to start from
        """
        ...

    def select_candidates(self, round_num: int) -> list[dict[str, Any]]:
        """Select programs to explore this round.

        Returns list of candidate specs for workers:
        [
            {
                "parent": ProgramEntry,
                "inspirations": list[ProgramEntry],
                "bottleneck_id": int,
            },
            ...
        ]

        Args:
            round_num: Current optimization round number

        Returns:
            List of candidate specifications for workers
        """
        ...

    def update_with_results(
        self, results: list[dict[str, Any]], round_num: int
    ) -> None:
        """Update strategy state with worker results.

        Args:
            results: List of result dicts from workers
            round_num: Current optimization round number
        """
        ...

    def get_best_program(self) -> ProgramEntry | None:
        """Get the best program found so far.

        Returns:
            The best performing ProgramEntry, or None if none found
        """
        ...

    def should_terminate(self, round_num: int, max_rounds: int) -> bool:
        """Check whether optimization should terminate early.

        Args:
            round_num: Current optimization round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if optimization should stop, False to continue
        """
        ...

    @property
    def num_workers_needed(self) -> int:
        """Number of workers this strategy needs per round."""
        ...

    def handle_worker_failure(self, worker_id: int, error: Exception) -> None:
        """Handle a failed worker gracefully.

        Default implementation logs a warning.

        Args:
            worker_id: ID of the failed worker
            error: The exception that occurred
        """
        ...
