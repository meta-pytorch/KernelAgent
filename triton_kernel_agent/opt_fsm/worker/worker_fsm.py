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

"""Worker FSM builder and process entry point."""

from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_fsm.engine import FSMEngine
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext
from triton_kernel_agent.opt_fsm.worker.states.analyze_bottleneck import (
    AnalyzeBottleneck,
)
from triton_kernel_agent.opt_fsm.worker.states.benchmark_baseline import (
    BenchmarkBaseline,
)
from triton_kernel_agent.opt_fsm.worker.states.benchmark_new import BenchmarkNew
from triton_kernel_agent.opt_fsm.worker.states.check_attempts import CheckAttempts
from triton_kernel_agent.opt_fsm.worker.states.check_early_stop import CheckEarlyStop
from triton_kernel_agent.opt_fsm.worker.states.check_patterns import CheckPatterns
from triton_kernel_agent.opt_fsm.worker.states.generate_kernel import GenerateKernel
from triton_kernel_agent.opt_fsm.worker.states.generate_reflexion import (
    GenerateReflexion,
)
from triton_kernel_agent.opt_fsm.worker.states.profile_kernel import ProfileKernel
from triton_kernel_agent.opt_fsm.worker.states.profile_sol import ProfileSol
from triton_kernel_agent.opt_fsm.worker.states.record_failure import RecordFailure
from triton_kernel_agent.opt_fsm.worker.states.refine_kernel import RefineKernel
from triton_kernel_agent.opt_fsm.worker.states.run_tests import RunTests
from triton_kernel_agent.opt_fsm.worker.states.update_kernels import UpdateKernels
from triton_kernel_agent.opt_fsm.worker.states.worker_finalize import WorkerFinalize


def build_worker_fsm(logger: logging.Logger) -> FSMEngine:
    """Build and return the inner (per-worker) FSM with all states registered."""
    engine = FSMEngine(logger=logger)
    engine.add_state(BenchmarkBaseline())
    engine.add_state(ProfileKernel())
    engine.add_state(AnalyzeBottleneck())
    engine.add_state(GenerateKernel())
    engine.add_state(CheckPatterns())
    engine.add_state(RunTests())
    engine.add_state(CheckAttempts())
    engine.add_state(RefineKernel())
    engine.add_state(BenchmarkNew())
    engine.add_state(ProfileSol())
    engine.add_state(GenerateReflexion())
    engine.add_state(UpdateKernels())
    engine.add_state(CheckEarlyStop())
    engine.add_state(RecordFailure())
    engine.add_state(WorkerFinalize())
    return engine


def worker_process_entry(
    worker_id: int,
    kernel_code: str,
    known_time: float,
    parent_id: str,
    problem_file: Path,
    test_code: str,
    workdir: Path,
    log_dir: Path,
    result_queue: Any,  # mp.Queue
    benchmark_lock: Any,
    profiling_semaphore: Any,
    pytorch_baseline: float,
    bottleneck_id: int,
    openai_model: str,
    high_reasoning_effort: bool,
    bottleneck_override: str | None,
    worker_kwargs: dict,
    prior_history: list[dict],
    prior_reflexions: list[dict],
) -> None:
    """Worker process entry point — runs the inner FSM.

    Same signature as opt_manager._worker_process so RUN_WORKERS can spawn it.
    """
    import sys

    kernel_agent_path = Path(__file__).parent.parent.parent.parent
    if str(kernel_agent_path) not in sys.path:
        sys.path.insert(0, str(kernel_agent_path))

    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker
        from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
            OptimizationAttempt,
            Reflexion,
        )

        # Ensure directories
        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Copy problem file
        shutil.copy(problem_file, workdir / "problem.py")

        # Create worker to get all components
        worker = OptimizationWorker(
            worker_id=worker_id,
            workdir=workdir,
            log_dir=log_dir,
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            bottleneck_id=bottleneck_id,
            benchmark_lock=benchmark_lock,
            profiling_semaphore=profiling_semaphore,
            pytorch_baseline_time=pytorch_baseline,
            bottleneck_override=bottleneck_override,
            prior_history=prior_history,
            prior_reflexions=prior_reflexions,
            **worker_kwargs,
        )

        # Build worker context from components
        ctx = WorkerContext(
            profiler=worker.profiler,
            benchmarker=worker.benchmarker,
            bottleneck_analyzer=worker.bottleneck_analyzer,
            verification_worker=worker.verification_worker,
            prompt_manager=worker.prompt_manager,
            roofline_analyzer=worker.roofline_analyzer,
            rag_prescriber=worker.rag_prescriber,
            provider=worker.provider,
            model=worker.openai_model,
            high_reasoning_effort=worker.high_reasoning_effort,
            kernel_file=worker.kernel_file,
            artifact_dir=worker.artifact_dir,
            output_dir=worker.output_dir,
            logger=worker.logger,
            gpu_specs=worker.gpu_specs,
            divergence_threshold=worker.divergence_threshold,
            sol_improvement_threshold=worker.sol_improvement_threshold,
            bottleneck_id=worker.bottleneck_id,
            bottleneck_override=worker.bottleneck_override,
            initial_kernel=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            pytorch_baseline_time=pytorch_baseline,
            max_opt_rounds=1,  # Single step per manager round
        )

        # Initialize history from prior rounds
        if prior_history:
            for attempt_dict in prior_history:
                ctx.attempt_history.append(
                    OptimizationAttempt.from_dict(attempt_dict)
                )
        if prior_reflexions:
            for reflexion_dict in prior_reflexions:
                ctx.reflexions.append(Reflexion.from_dict(reflexion_dict))

        # Build and run the worker FSM
        fsm = build_worker_fsm(worker.logger)
        fsm.run("BenchmarkBaseline", ctx)

        # Get attempt/reflexion for shared history
        attempt_data = ctx.result_metrics.get("last_attempt")
        reflexion_data = ctx.result_metrics.get("last_reflexion")

        result_queue.put(
            {
                "success": ctx.success,
                "worker_id": worker_id,
                "kernel_code": ctx.result_kernel,
                "time_ms": ctx.result_metrics.get("best_time_ms", float("inf")),
                "parent_id": parent_id,
                "attempt": attempt_data,
                "reflexion": reflexion_data,
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "worker_id": worker_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
