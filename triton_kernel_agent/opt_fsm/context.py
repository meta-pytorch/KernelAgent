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

"""Optimization context for the outer (manager-level) FSM."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)


@dataclass
class OptimizationContext:
    """All mutable state for the outer (manager-level) FSM.

    Replaces the local variables scattered across OptimizationManager methods.
    """

    # --- Inputs (set once at start) ---
    initial_kernel: str = ""
    problem_file: Path = field(default_factory=lambda: Path("."))
    test_code: str = ""
    max_rounds: int = 10

    # --- Infrastructure ---
    strategy: SearchStrategy | None = None
    database: JSONProgramDatabase | None = None
    log_dir: Path = field(default_factory=lambda: Path("."))
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("OptFSM")
    )
    benchmark_lock: Any = None  # mp.Lock
    profiling_semaphore: Any = None  # mp.Semaphore

    # --- LLM / worker config ---
    openai_model: str = "claude-opus-4.5"
    high_reasoning_effort: bool = True
    bottleneck_override: str | None = None
    worker_kwargs: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4

    # --- Baselines (populated by BENCHMARK_BASELINES) ---
    pytorch_baseline_ms: float = float("inf")
    pytorch_compile_ms: float = float("inf")
    initial_kernel_time_ms: float = float("inf")

    # --- Round tracking ---
    round_num: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    worker_results: list[dict[str, Any]] = field(default_factory=list)

    # --- Shared history across rounds ---
    shared_history: list[dict] = field(default_factory=list)
    shared_reflexions: list[dict] = field(default_factory=list)
    history_size: int = 10

    # --- Result (populated by FINALIZE) ---
    result: dict[str, Any] = field(default_factory=dict)

    # --- Control flags ---
    initial_verification_passed: bool = False
    terminated: bool = False
