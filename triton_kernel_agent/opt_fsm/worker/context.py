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

"""Worker context for the inner (per-worker) FSM."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationAttempt,
    Reflexion,
)


@dataclass
class WorkerContext:
    """All mutable state for the inner (per-worker) FSM.

    Replaces the local variables in OptimizationOrchestrator.optimize_kernel().
    """

    # --- Components (set by worker_fsm._worker_process) ---
    profiler: Any = None  # KernelProfiler
    benchmarker: Any = None  # Benchmark
    bottleneck_analyzer: Any = None  # BottleneckAnalyzer
    verification_worker: Any = None  # VerificationWorker
    prompt_manager: Any = None  # PromptManager
    roofline_analyzer: RooflineAnalyzer | None = None
    rag_prescriber: Any = None  # RAGPrescriber | None
    provider: Any = None  # BaseProvider
    model: str = ""
    high_reasoning_effort: bool = True

    # --- File paths ---
    kernel_file: Path = field(default_factory=lambda: Path("."))
    artifact_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("."))
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("WorkerFSM")
    )

    # --- Configuration ---
    gpu_specs: dict[str, Any] | None = None
    divergence_threshold: float = 50.0
    sol_improvement_threshold: float = 5.0
    bottleneck_id: int | None = None
    bottleneck_override: str | None = None

    # --- Inputs (set once at start) ---
    initial_kernel: str = ""
    problem_file: Path = field(default_factory=lambda: Path("."))
    test_code: str = ""
    known_kernel_time: float | None = None
    pytorch_baseline_time: float | None = None

    # --- Current kernel state ---
    current_kernel: str = ""
    error_feedback: str = ""
    problem_description: str = ""

    # --- Baseline results (set by BENCHMARK_BASELINE) ---
    best_time: float = float("inf")
    baseline_results: dict[str, float] = field(default_factory=dict)
    baseline_sol: float = 0.0

    # --- Two-kernel tracking ---
    best_runtime_kernel: str = ""
    best_runtime_time: float = float("inf")
    best_runtime_sol: float = 0.0
    best_sol_kernel: str = ""
    best_sol_time: float = float("inf")
    best_sol_sol: float = 0.0

    # --- Profiling / analysis results for current round ---
    bottleneck_results: list | None = None
    roofline_result: Any = None
    ncu_metrics: dict[str, Any] | None = None
    primary_bottleneck: Any = None  # BottleneckResult
    current_attempt: OptimizationAttempt | None = None
    current_config: dict[str, Any] = field(default_factory=dict)

    # --- Generated kernel (output of GENERATE_KERNEL) ---
    optimized_kernel: str | None = None

    # --- Verification state ---
    refinement_attempts: int = 0
    max_refine_attempts: int = 3
    test_stdout: str = ""
    test_stderr: str = ""
    violation_message: str | None = None

    # --- Benchmark results for new kernel ---
    new_time: float = float("inf")
    new_kernel_metrics: dict[str, Any] | None = None
    new_sol: float = 0.0

    # --- History tracking ---
    attempt_history: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )
    reflexions: list[Reflexion] = field(default_factory=list)
    history_size: int = 5

    # --- Metadata tracking ---
    best_ncu_metrics: dict[str, Any] | None = None
    best_bottleneck_category: str | None = None
    best_round_num: int = 0
    early_stop_reason: str = ""
    any_verified: bool = False

    # --- Round tracking ---
    round_num: int = 1
    max_opt_rounds: int = 1

    # --- Result (populated by WORKER_FINALIZE) ---
    success: bool = False
    result_kernel: str = ""
    result_metrics: dict[str, Any] = field(default_factory=dict)
