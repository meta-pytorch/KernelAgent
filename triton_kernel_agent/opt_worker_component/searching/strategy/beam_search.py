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

Maintains a candidate pool of top-N kernels and explores M bottlenecks
per kernel each round, optionally fanned out across K distinct LLMs and
C independent samples per prompt.  Every member of the pool is expanded
each round; total workers = candidate_pool_size × M × K × C.
"""

import logging
from typing import Any, Sequence

from ..history.models import ProgramEntry, ProgramMetrics
from ..history.store import ProgramDatabase
from ..technique_vector import (
    TechniqueDefinition,
    classify_many,
    select_diverse_top_k,
)
from .strategy import SearchStrategy


class BeamSearchStrategy(SearchStrategy):
    """Beam search strategy for kernel optimization.

    This strategy maintains a candidate pool of top-performing kernels
    and expands every member of the pool each round.  Expansion can fan
    out across several dimensions for diversity:

    - ``candidate_pool_size`` (N): pool size carried round-to-round; every
      member is expanded.
    - ``num_bottlenecks`` (M): bottleneck directions per parent.
    - ``models`` (K): LLM providers to fan across; each generates its own
      bottleneck analysis and rewrite.
    - ``samples_per_prompt`` (C): independent LLM draws per (parent,
      bottleneck, model) triple, to harvest sampling-level diversity.

    Workers per round = candidate_pool_size × M × K × C.

    After workers return, candidates are deduplicated by PTX fingerprint
    (same normalized compiled PTX ⇒ same kernel) before being ranked and
    truncated to ``candidate_pool_size``.
    """

    def __init__(
        self,
        candidate_pool_size: int = 2,
        num_bottlenecks: int = 2,
        database: ProgramDatabase | None = None,
        logger: logging.Logger | None = None,
        models: list[str] | None = None,
        samples_per_prompt: int = 1,
        techniques: Sequence[TechniqueDefinition] | None = None,
        technique_classifier_provider: Any | None = None,
        technique_classifier_model: str | None = None,
        technique_classifier_concurrency: int = 4,
    ):
        """Initialize beam search strategy.

        Args:
            candidate_pool_size: Number of kernels carried in the pool
                round-to-round.  Every member of the pool is expanded
                each round.
            num_bottlenecks: Number of bottleneck directions to explore per kernel
            database: Optional program database for persistence
            logger: Optional logger
            models: Optional list of LLM model names. When provided, every
                (kernel, bottleneck) pair is expanded once per model.
            samples_per_prompt: Number of independent LLM samples to draw
                per (parent, bottleneck, model) triple.  Values >1 rely on
                the LLM being non-deterministic (temperature >0) to yield
                distinct candidates.  Default 1 preserves prior behavior.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.problem_id: str | None = None
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.num_bottlenecks = num_bottlenecks
        self.database = database
        self.top_kernels: list[ProgramEntry] = []
        self.models = models
        self.samples_per_prompt = max(1, samples_per_prompt)
        # Internal iteration list: [None] means "use runner default".
        self._expansion_models: list[str | None] = list(models) if models else [None]

        # Technique-vector clustering (opt-in): only active when both
        # ``techniques`` and a classifier provider/model are supplied.
        # When inactive, the strategy falls back to the previous
        # PTX-dedup-then-sort-and-truncate behavior.
        self._techniques: tuple[TechniqueDefinition, ...] = (
            tuple(techniques) if techniques else ()
        )
        self._classifier_provider = technique_classifier_provider
        self._classifier_model = technique_classifier_model
        self._classifier_concurrency = max(1, int(technique_classifier_concurrency))

    @property
    def technique_clustering_enabled(self) -> bool:
        return bool(
            self._techniques
            and self._classifier_provider is not None
            and self._classifier_model
        )

    @property
    def num_workers_needed(self) -> int:
        """Workers per round = pool × bottlenecks × models × samples."""
        return (
            self.candidate_pool_size
            * self.num_bottlenecks
            * len(self._expansion_models)
            * self.samples_per_prompt
        )

    def initialize(self, initial_program: ProgramEntry) -> None:
        """Initialize with starting program.

        Args:
            initial_program: The initial kernel to start optimization from
        """
        self.problem_id = initial_program.problem_id
        # Start with N copies of initial (will be deduplicated on first update)
        self.top_kernels = [initial_program] * self.candidate_pool_size
        models_str = (
            ", ".join(str(m) for m in self.models) if self.models else "<default>"
        )
        self.logger.info(
            f"BeamSearch initialized: pool={self.candidate_pool_size} × "
            f"{self.num_bottlenecks} bottlenecks × {len(self._expansion_models)} "
            f"models [{models_str}] × {self.samples_per_prompt} samples "
            f"= {self.num_workers_needed} workers"
        )

    def select_candidates(self, round_num: int) -> list[dict[str, Any]]:
        """Select candidates for this round.

        Creates one candidate for each (parent, bottleneck, model, sample)
        tuple.  Every member of ``self.top_kernels`` is expanded.

        Args:
            round_num: Current round number

        Returns:
            List of candidate specs for workers
        """
        candidates: list[dict[str, Any]] = []
        for rank, kernel in enumerate(self.top_kernels):
            for bottleneck_id in range(1, self.num_bottlenecks + 1):
                for model in self._expansion_models:
                    for sample_idx in range(self.samples_per_prompt):
                        candidates.append(
                            {
                                "parent": kernel,
                                "bottleneck_id": bottleneck_id,
                                "kernel_rank": rank,
                                "openai_model": model,
                                "sample_idx": sample_idx,
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
        # Build a (kernel_code → ptx_hash) lookup from the current beam.
        # If a worker returned an *unchanged* parent (i.e. its LLM did not
        # improve the kernel), the worker reports ptx_hash=None even though
        # the parent's hash is already known to the strategy.  Falling back
        # to that hash lets PTX dedup correctly merge such results with
        # the existing parent entry.
        ptx_by_kernel_code: dict[str, str] = {
            p.kernel_code: p.ptx_hash for p in self.top_kernels if p.ptx_hash
        }

        # Add successful results
        new_entries: list[ProgramEntry] = []
        entry_models: dict[str, str | None] = {}
        for result in results:
            if result.get("success"):
                program_id = f"r{round_num}_w{result['worker_id']}"
                ptx_hash = result.get("ptx_hash") or ptx_by_kernel_code.get(
                    result.get("kernel_code", "")
                )
                entry = ProgramEntry(
                    program_id=program_id,
                    kernel_code=result["kernel_code"],
                    metrics=ProgramMetrics(time_ms=result["time_ms"]),
                    problem_id=self.problem_id,
                    parent_id=result.get("parent_id"),
                    generation=round_num,
                    ptx_hash=ptx_hash,
                )
                new_entries.append(entry)
                entry_models[program_id] = result.get("openai_model")
                if self.database:
                    self.database.add_program(entry)

        # Combine old beam + new results into the candidate pool.
        all_candidates = self.top_kernels + new_entries

        # Dedup by PTX fingerprint: entries with the same normalized PTX are
        # the same kernel at the compiler level, so keep only the fastest
        # per fingerprint.  Entries with ``ptx_hash is None`` (capture
        # failed) are treated as singletons so we never accidentally merge
        # them together.
        dedup_before = len(all_candidates)
        pooled = self._dedup_by_ptx(all_candidates)
        dedup_after = len(pooled)

        # Optional technique-vector clustering: classify newly-arrived
        # survivors via LLM, then pick top-N with diversity preserved
        # across distinct vectors.  When disabled, fall back to the
        # plain sort+truncate path.
        cluster_log = ""
        if self.technique_clustering_enabled:
            self._classify_pooled(pooled)
            self.top_kernels = select_diverse_top_k(pooled, self.candidate_pool_size)
            distinct_clusters = len(
                {
                    tuple(e.technique_vector)
                    if e.technique_vector is not None
                    else ("__none__", e.program_id)
                    for e in pooled
                }
            )
            cluster_log = f", clusters={distinct_clusters}"
        else:
            pooled.sort(key=lambda x: x.metrics.time_ms)
            self.top_kernels = pooled[: self.candidate_pool_size]

        if self.database:
            self.database.save()

        # Log update
        if new_entries:
            best_new_entry = min(new_entries, key=lambda e: e.metrics.time_ms)
            best_new_model = entry_models.get(best_new_entry.program_id)
            model_tag = f" [{best_new_model}]" if best_new_model else ""
            self.logger.info(
                f"Round {round_num}: {len(new_entries)} successful, "
                f"PTX dedup {dedup_before}→{dedup_after}{cluster_log}, "
                f"best new: {best_new_entry.metrics.time_ms:.4f}ms{model_tag}"
            )

    def _classify_pooled(self, pooled: list[ProgramEntry]) -> None:
        """Fill in missing / stale ``technique_vector`` on each entry.

        Entries whose vector is already present and matches the current
        taxonomy length are left alone (cached from prior rounds or
        prior runs via the JSON database).  Everything else gets a
        fresh LLM classification, fanned out via a thread pool.
        """
        if not self.technique_clustering_enabled:
            return
        expected_dim = len(self._techniques)
        # Identify entries that need (re)classification.
        to_classify: list[tuple[str, str]] = []  # (program_id, kernel_code)
        index: dict[str, ProgramEntry] = {}
        for entry in pooled:
            v = entry.technique_vector
            if v is None or len(v) != expected_dim:
                # Stale-length vectors (taxonomy changed) get reclassified.
                if v is not None and len(v) != expected_dim:
                    entry.technique_vector = None
                to_classify.append((entry.program_id, entry.kernel_code))
                index[entry.program_id] = entry
        if not to_classify:
            return

        self.logger.info(
            f"Technique clustering: classifying {len(to_classify)} kernel(s) "
            f"with {self._classifier_model} (taxonomy dim={expected_dim})"
        )
        results = classify_many(
            to_classify,
            self._techniques,
            self._classifier_provider,
            self._classifier_model,
            logger=self.logger,
            max_concurrency=self._classifier_concurrency,
        )
        for pid, vec in results.items():
            entry = index.get(pid)
            if entry is None:
                continue
            entry.technique_vector = vec  # may be None if classification failed
            if self.database:
                # Re-add to persist the updated vector.
                self.database.add_program(entry)

    @staticmethod
    def _dedup_by_ptx(entries: list[ProgramEntry]) -> list[ProgramEntry]:
        """Collapse entries with identical PTX fingerprints to the fastest.

        Entries whose ``ptx_hash`` is ``None`` are kept as-is (treated as
        singletons) so we never merge two kernels whose PTX we couldn't
        capture.
        """
        best_by_hash: dict[str, ProgramEntry] = {}
        singletons: list[ProgramEntry] = []
        for e in entries:
            h = e.ptx_hash
            if h is None:
                singletons.append(e)
                continue
            incumbent = best_by_hash.get(h)
            if incumbent is None or e.metrics.time_ms < incumbent.metrics.time_ms:
                best_by_hash[h] = e
        return list(best_by_hash.values()) + singletons

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
