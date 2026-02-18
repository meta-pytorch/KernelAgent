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

"""Greedy optimization strategy.

Simple single-best optimization that always explores from the current
best kernel with one worker. Includes early termination when improvement
plateaus.
"""

import logging
from typing import Any

from ..history.models import ProgramEntry, ProgramMetrics
from ..history.store import ProgramDatabase
from .strategy import SearchStrategy


class GreedyStrategy(SearchStrategy):
    """Greedy single-best optimization strategy.

    Always optimizes from the current best kernel using a single worker.
    Terminates early if no improvement for several consecutive rounds.
    """

    def __init__(
        self,
        database: ProgramDatabase | None = None,
        max_no_improvement: int = 5,
        logger: logging.Logger | None = None,
    ):
        """Initialize greedy strategy.

        Args:
            database: Optional program database for persistence
            max_no_improvement: Stop after this many rounds without improvement
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.problem_id: str | None = None
        self.database = database
        self.best_program: ProgramEntry | None = None
        self.no_improvement_count: int = 0
        self.max_no_improvement: int = max_no_improvement

    @property
    def num_workers_needed(self) -> int:
        """Greedy uses a single worker."""
        return 1

    def initialize(self, initial_program: ProgramEntry) -> None:
        """Initialize with starting program.

        Args:
            initial_program: The initial kernel to start optimization from
        """
        self.problem_id = initial_program.problem_id
        self.best_program = initial_program
        self.no_improvement_count = 0
        self.logger.info("GreedyStrategy initialized with 1 worker")

    def select_candidates(self, round_num: int) -> list[dict[str, Any]]:
        """Select the single candidate (current best).

        Args:
            round_num: Current round number

        Returns:
            List with single candidate spec
        """
        return [
            {
                "parent": self.best_program,
                "inspirations": [],
                "bottleneck_id": 1,
            }
        ]

    def update_with_results(
        self, results: list[dict[str, Any]], round_num: int
    ) -> None:
        """Update best program if worker improved.

        Args:
            results: Worker results (single result expected)
            round_num: Current round number
        """
        for result in results:
            if result.get("success"):
                entry = ProgramEntry(
                    program_id=f"r{round_num}_w{result['worker_id']}",
                    kernel_code=result["kernel_code"],
                    metrics=ProgramMetrics(time_ms=result["time_ms"]),
                    problem_id=self.problem_id,
                    parent_id=self.best_program.program_id
                    if self.best_program
                    else None,
                    generation=round_num,
                )

                if self.database:
                    self.database.add_program(entry)

                # Update best if improved
                if (
                    self.best_program is None
                    or entry.metrics.time_ms < self.best_program.metrics.time_ms
                ):
                    improvement = (
                        (self.best_program.metrics.time_ms - entry.metrics.time_ms)
                        / self.best_program.metrics.time_ms
                        * 100
                        if self.best_program
                        else 0
                    )
                    self.logger.info(
                        f"New best: {entry.metrics.time_ms:.4f}ms "
                        f"({improvement:.1f}% improvement)"
                    )
                    self.best_program = entry
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    self.logger.info(
                        f"No improvement "
                        f"({self.no_improvement_count}/{self.max_no_improvement})"
                    )

        if self.database:
            self.database.save()

    def get_best_program(self) -> ProgramEntry | None:
        """Get the current best program."""
        return self.best_program

    def should_terminate(self, round_num: int, max_rounds: int) -> bool:
        """Terminate on plateau or max rounds.

        Args:
            round_num: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if should terminate
        """
        if self.no_improvement_count >= self.max_no_improvement:
            self.logger.info(
                f"Early termination: no improvement for "
                f"{self.max_no_improvement} rounds"
            )
            return True
        return round_num >= max_rounds

    def handle_worker_failure(self, worker_id: int, error: Exception) -> None:
        """Handle a failed worker gracefully."""
        self.logger.warning(f"Worker {worker_id} failed: {error}")
