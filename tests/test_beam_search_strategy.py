#!/usr/bin/env python3
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

"""Tests for BeamSearchStrategy config validation and candidate schema contract."""

import logging
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_metrics(**kw):
    from triton_kernel_agent.opt_worker_component.searching.history.models import ProgramMetrics
    defaults = dict(time_ms=1.0)
    defaults.update(kw)
    return ProgramMetrics(**defaults)


def _make_entry(**kw):
    from triton_kernel_agent.opt_worker_component.searching.history.models import ProgramEntry
    defaults = dict(
        program_id="prog_0",
        kernel_code="def k(): pass",
        metrics=_make_metrics(),
        problem_id="p0",
    )
    defaults.update(kw)
    return ProgramEntry(**defaults)


def _make_strategy(database=None, **kw):
    from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import BeamSearchStrategy
    defaults = dict(
        num_top_kernels=4,
        num_bottlenecks=2,
        models=["claude-sonnet-4-6"],
        samples_per_prompt=1,
        num_expanding_parents=2,
        database=database,
    )
    defaults.update(kw)
    return BeamSearchStrategy(**defaults)


class TestNumExpandingParentsValidation:
    """Tests for num_expanding_parents config validation."""

    def test_zero_parents_clamped_to_one(self, caplog):
        with caplog.at_level(logging.WARNING, logger="BeamSearchStrategy"):
            s = _make_strategy(num_expanding_parents=0)
        assert s.num_expanding_parents == 1
        assert any("clamping to 1" in r.message for r in caplog.records)

    def test_negative_parents_clamped_to_one(self, caplog):
        with caplog.at_level(logging.WARNING, logger="BeamSearchStrategy"):
            s = _make_strategy(num_expanding_parents=-1)
        assert s.num_expanding_parents == 1

    def test_none_parents_not_clamped(self):
        """num_expanding_parents=None is valid and means use all top kernels."""
        s = _make_strategy(num_expanding_parents=None)
        assert s.num_expanding_parents is None

    def test_zero_parents_still_produces_candidates(self):
        s = _make_strategy(num_expanding_parents=0)
        s.initialize(_make_entry())
        candidates = s.select_candidates(round_num=1)
        assert len(candidates) > 0


class TestCandidateSchemaContract:
    """Tests for SearchStrategy Protocol schema compliance in beam search candidates."""

    def test_candidates_contain_inspirations_key(self):
        """Every candidate dict must include the 'inspirations' key per SearchStrategy Protocol."""
        mock_db = MagicMock()
        mock_db.sample_inspirations.return_value = []
        s = _make_strategy(database=mock_db)
        s.initialize(_make_entry())
        for candidate in s.select_candidates(round_num=1):
            assert "inspirations" in candidate
            assert isinstance(candidate["inspirations"], list)

    def test_candidates_satisfy_full_protocol_schema(self):
        """All required Protocol keys must be present in every candidate."""
        mock_db = MagicMock()
        mock_db.sample_inspirations.return_value = []
        s = _make_strategy(database=mock_db)
        s.initialize(_make_entry())
        required_keys = {"parent", "bottleneck_id", "inspirations"}
        for candidate in s.select_candidates(round_num=1):
            assert required_keys.issubset(candidate.keys())

    def test_inspirations_excludes_parent_kernel(self):
        mock_db = MagicMock()
        mock_db.sample_inspirations.return_value = []
        s = _make_strategy(database=mock_db)
        entry = _make_entry(program_id="prog_parent")
        s.initialize(entry)
        s.select_candidates(round_num=1)
        calls = mock_db.sample_inspirations.call_args_list
        assert len(calls) > 0
        for call in calls:
            exclude = call.kwargs.get("exclude_ids") or (
                call.args[1] if len(call.args) > 1 else None
            )
            assert exclude is not None and "prog_parent" in exclude

    def test_no_database_returns_empty_inspirations(self):
        """When database=None, inspirations should be empty list, not an error."""
        s = _make_strategy(database=None)
        s.initialize(_make_entry())
        for candidate in s.select_candidates(round_num=1):
            assert candidate["inspirations"] == []


class TestWorkerCount:
    """Tests for num_workers_needed property."""

    def test_workers_needed_matches_fanout(self):
        """num_workers_needed must equal parents × bottlenecks × models × samples."""
        s = _make_strategy(
            num_expanding_parents=2,
            num_bottlenecks=3,
            models=["a", "b"],
            samples_per_prompt=2,
        )
        assert s.num_workers_needed == 2 * 3 * 2 * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])