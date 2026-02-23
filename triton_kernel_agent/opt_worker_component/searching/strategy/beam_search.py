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

"""Beam search optimization strategy.

Maintains top-N kernels and explores M bottlenecks per kernel each round.
Total workers = N × M.
"""

import logging
from typing import Any

from ..history.models import ProgramEntry, ProgramMetrics
from ..history.store import ProgramDatabase
from .strategy import SearchStrategy


class BeamSearchStrategy(SearchStrategy):
    """Beam search strategy for kernel optimization.

    This strategy maintains a beam of top-performing kernels and explores
    multiple bottleneck directions for each. It mirrors the original
    beam search behavior from optimization_manager.py.

    Workers = num_top_kernels × num_bottlenecks
    """

    def __init__(
        self,
        num_top_kernels: int = 2,
        num_bottlenecks: int = 2,
        database: ProgramDatabase | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize beam search strategy.

        Args:
            num_top_kernels: Number of top kernels to maintain in beam
            num_bottlenecks: Number of bottleneck directions to explore per kernel
            database: Optional program database for persistence
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.problem_id: str | None = None
        self.num_top_kernels = num_top_kernels
        self.num_bottlenecks = num_bottlenecks
        self.database = database
        self.top_kernels: list[ProgramEntry] = []

    @property
    def num_workers_needed(self) -> int:
        """Number of workers = top_kernels × bottlenecks."""
        return self.num_top_kernels * self.num_bottlenecks

    def initialize(self, initial_program: ProgramEntry) -> None:
        """Initialize with starting program.

        Args:
            initial_program: The initial kernel to start optimization from
        """
        self.problem_id = initial_program.problem_id
        # Start with N copies of initial (will be deduplicated on first update)
        self.top_kernels = [initial_program] * self.num_top_kernels
        self.logger.info(
            f"BeamSearch initialized: {self.num_top_kernels} kernels × "
            f"{self.num_bottlenecks} bottlenecks = {self.num_workers_needed} workers"
        )

    def select_candidates(self, round_num: int) -> list[dict[str, Any]]:
        """Select candidates for this round.

        Creates one candidate for each (kernel, bottleneck) pair.

        Args:
            round_num: Current round number

        Returns:
            List of candidate specs for workers
        """
        candidates = []
        for rank, kernel in enumerate(self.top_kernels):
            for bottleneck_id in range(1, self.num_bottlenecks + 1):
                candidates.append(
                    {
                        "parent": kernel,
                        "bottleneck_id": bottleneck_id,
                        "kernel_rank": rank,
                    }
                )
        return candidates

    def update_with_results(
        self, results: list[dict[str, Any]], round_num: int
    ) -> None:
        """Update beam with worker results.

        Adds successful results to candidates, sorts by performance,
        and keeps top-k.

        Args:
            results: Worker results
            round_num: Current round number
        """
        # Add successful results
        new_entries = []
        for result in results:
            if result.get("success"):
                entry = ProgramEntry(
                    program_id=f"r{round_num}_w{result['worker_id']}",
                    kernel_code=result["kernel_code"],
                    metrics=ProgramMetrics(time_ms=result["time_ms"]),
                    problem_id=self.problem_id,
                    parent_id=result.get("parent_id"),
                    generation=round_num,
                )
                new_entries.append(entry)
                if self.database:
                    self.database.add_program(entry)

        # Update top-k (combine, sort, truncate)
        all_candidates = self.top_kernels + new_entries
        all_candidates.sort(key=lambda x: x.metrics.time_ms)
        self.top_kernels = all_candidates[: self.num_top_kernels]

        if self.database:
            self.database.save()

        # Log update
        if new_entries:
            best_new = min(e.metrics.time_ms for e in new_entries)
            self.logger.info(
                f"Round {round_num}: {len(new_entries)} successful, "
                f"best new: {best_new:.4f}ms"
            )

    def get_best_program(self) -> ProgramEntry | None:
        """Get the best performing kernel in the beam."""
        if not self.top_kernels:
            return None
        return min(self.top_kernels, key=lambda x: x.metrics.time_ms)

    def should_terminate(self, round_num: int, max_rounds: int) -> bool:
        """Terminate when max rounds reached.

        Beam search doesn't have early termination - it runs all rounds.

        Args:
            round_num: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            True if round_num >= max_rounds
        """
        return round_num >= max_rounds

    def handle_worker_failure(self, worker_id: int, error: Exception) -> None:
        """Handle a failed worker gracefully."""
        self.logger.warning(f"Worker {worker_id} failed: {error}")
