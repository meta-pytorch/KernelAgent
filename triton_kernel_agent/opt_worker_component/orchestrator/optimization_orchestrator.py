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


"""Main optimization orchestration logic."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import BottleneckResult
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.worker import VerificationWorker
from triton_kernel_agent.worker_util import (
    _call_llm,
    _extract_code_from_response,
    _write_kernel_file,
)
from utils.providers.base import BaseProvider


def extract_triton_config(kernel_code: str) -> dict[str, Any]:
    """
    Extract Triton config from @triton.autotune decorator in kernel code.

    Parses: @triton.autotune(configs=[triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2)])
    Returns: {"num_warps": 4, "num_stages": 2, "BLOCK_M": 64, ...}
    """
    config: dict[str, Any] = {}

    # Match triton.Config(...) blocks
    config_pattern = r"triton\.Config\s*\(\s*\{([^}]*)\}(?:\s*,\s*([^)]*))?\)"
    matches = re.findall(config_pattern, kernel_code)

    if not matches:
        return config

    # Take the first config (primary config)
    block_params, extra_params = matches[0]

    # Parse block parameters: 'BLOCK_M': 64, 'BLOCK_N': 128
    block_matches = re.findall(r"['\"](\w+)['\"]\s*:\s*(\d+)", block_params)
    for name, value in block_matches:
        config[name] = int(value)

    # Parse extra parameters: num_warps=4, num_stages=2
    if extra_params:
        extra_matches = re.findall(r"(\w+)\s*=\s*(\d+)", extra_params)
        for name, value in extra_matches:
            config[name] = int(value)

    return config


@dataclass
class OptimizationAttempt:
    """Records a single optimization attempt with its results."""

    round_num: int
    worker_id: int = 0
    bottleneck_category: str = ""
    root_cause: str = ""
    recommended_fix: str = ""
    config_changes: dict[str, str] = field(default_factory=dict)
    time_before_ms: float = 0.0
    time_after_ms: float = 0.0
    improvement_pct: float = 0.0
    is_improvement: bool = False
    compute_sol_pct: float = 0.0
    memory_sol_pct: float = 0.0
    combined_sol_pct: float = 0.0
    passed_verification: bool = False
    error_message: str = ""

    def format_for_prompt(self) -> str:
        """Format attempt for inclusion in optimization prompt."""
        status = "✓ IMPROVED" if self.is_improvement else "✗ NO IMPROVEMENT"
        if not self.passed_verification:
            status = "✗ FAILED VERIFICATION"

        perf_change = f"{abs(self.improvement_pct):.1f}% {'faster' if self.improvement_pct > 0 else 'slower'}"

        lines = [
            f"#### Attempt {self.round_num} (Worker {self.worker_id}) [{status} {perf_change}]",
            f"**Prescription Applied**: `{self.bottleneck_category}` tier",
        ]

        if self.config_changes:
            config_str = ", ".join(f"{k}={v}" for k, v in self.config_changes.items())
            lines.append(f"**Config**: {config_str}")

        lines.append(f"**Root Cause**: {self.root_cause}")
        lines.append(f"**Fix Applied**: {self.recommended_fix}")

        if self.passed_verification:
            lines.append(
                f"**Performance**: {self.time_before_ms:.4f}ms → {self.time_after_ms:.4f}ms ({perf_change})"
            )
            lines.append(
                f"**NCU SOL (% of Peak)**: Combined: {self.combined_sol_pct:.1f}%, "
                f"Compute: {self.compute_sol_pct:.1f}%, Memory: {self.memory_sol_pct:.1f}%"
            )
        else:
            lines.append(f"**Error**: {self.error_message[:200] if self.error_message else 'Unknown'}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> OptimizationAttempt:
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

@dataclass
class Reflexion:
    """Self-reflection on an optimization attempt."""

    round_num: int
    root_cause_diagnosed: str = ""
    fix_applied: str = ""
    expected_outcome: str = ""
    actual_outcome: str = ""
    performance_delta_pct: float = 0.0
    was_diagnosis_correct: bool = False
    was_fix_effective: bool = False
    reasoning: str = ""
    lessons: list[str] = field(default_factory=list)
    avoid_patterns: list[str] = field(default_factory=list)
    try_patterns: list[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format reflexion for inclusion in optimization prompt."""
        status = "Effective" if self.was_fix_effective else "Ineffective"

        lines = [
            f"#### Round {self.round_num} Reflection [{status}]",
            f"**Diagnosis**: {self.root_cause_diagnosed}",
            f"**Fix Applied**: {self.fix_applied}",
            f"**Expected**: {self.expected_outcome}",
            f"**Actual**: {self.actual_outcome} ({self.performance_delta_pct:+.1f}%)",
            f"**Analysis**: {self.reasoning}",
        ]

        if self.lessons:
            lines.append("**Lessons**:")
            for lesson in self.lessons:
                lines.append(f"  - {lesson}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> Reflexion:
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


def _get_triton_kernel_metrics(ncu_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Extract metrics for the Triton kernel, filtering out PyTorch kernels.

    NCU profiles all CUDA kernels including PyTorch internals (at::*).
    This function finds the actual Triton kernel metrics.

    Args:
        ncu_metrics: Dict keyed by kernel name with metric dicts as values

    Returns:
        Flat metrics dict for the Triton kernel, or first non-PyTorch kernel
    """
    if not ncu_metrics:
        return {}

    # Filter out PyTorch kernels (they start with "at::" or "void at::")
    triton_kernels = {
        name: metrics
        for name, metrics in ncu_metrics.items()
        if not name.startswith("at::") and not name.startswith("void at::")
    }

    if triton_kernels:
        # Return the first Triton kernel's metrics
        return next(iter(triton_kernels.values()))

    # Fallback: return first kernel if no Triton kernel found
    return next(iter(ncu_metrics.values()), {})


class OptimizationOrchestrator:
    """Orchestrates the main optimization loop."""

    def __init__(
        self,
        profiler: Any,
        benchmarker: Any,
        bottleneck_analyzer: Any,
        verification_worker: VerificationWorker,
        prompt_manager: PromptManager,
        provider: BaseProvider,
        model: str,
        high_reasoning_effort: bool,
        kernel_file: Path,
        gpu_specs: dict[str, Any] | None,
        pytorch_baseline_time: float | None,
        artifact_dir: Path,
        output_dir: Path,
        logger: logging.Logger,
        roofline_analyzer: RooflineAnalyzer,
        divergence_threshold: float = 50.0,
        sol_improvement_threshold: float = 5.0,
        bottleneck_id: int | None = None,
        bottleneck_override: str | None = None,
        # Shared history from beam search manager
        prior_history: list[dict] | None = None,
        prior_reflexions: list[dict] | None = None,
        rag_prescriber: Any | None = None,
    ):
        """
        Initialize optimization orchestrator.

        Args:
            profiler: KernelProfiler instance
            benchmarker: Benchmark instance (handles both kernel and PyTorch benchmarking)
            bottleneck_analyzer: BottleneckAnalyzer instance
            verification_worker: VerificationWorker for verify + refine
            prompt_manager: PromptManager for building optimization prompts
            provider: LLM provider instance
            model: Model name for LLM calls
            high_reasoning_effort: Whether to use high reasoning effort
            kernel_file: Path to kernel file for writing
            gpu_specs: GPU specifications for optimization prompt
            pytorch_baseline_time: Pre-computed PyTorch baseline
            artifact_dir: Directory for optimization artifacts
            output_dir: Directory for final output (best_kernel.py)
            logger: Logger instance
            roofline_analyzer: RooflineAnalyzer for optimization guidance and early termination
            divergence_threshold: Max % worse performance before reverting to best kernel
            sol_improvement_threshold: Min SOL % improvement to accept kernel even if runtime doesn't improve
            bottleneck_id: Which bottleneck to use from LLM results (1-indexed, for beam search diversity)
            bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
            rag_prescriber: Optional RAGPrescriber instance for retrieval-augmented optimization
        """
        # Components
        self.profiler = profiler
        self.benchmarker = benchmarker
        self.bottleneck_analyzer = bottleneck_analyzer
        self.verification_worker = verification_worker
        self.prompt_manager = prompt_manager

        # LLM configuration
        self.provider = provider
        self.model = model
        self.high_reasoning_effort = high_reasoning_effort

        # File configuration
        self.kernel_file = kernel_file

        # Configuration
        self.gpu_specs = gpu_specs
        self.pytorch_baseline_time = pytorch_baseline_time
        self.divergence_threshold = divergence_threshold
        self.sol_improvement_threshold = sol_improvement_threshold

        # Paths
        self.artifact_dir = artifact_dir
        self.output_dir = output_dir

        # Logger
        self.logger = logger

        # Optional roofline analyzer
        self.roofline_analyzer = roofline_analyzer

        # Optional RAG prescriber
        self.rag_prescriber = rag_prescriber

        # Bottleneck selection for beam search diversity (1-indexed)
        self.bottleneck_id = bottleneck_id

        # Pre-computed bottleneck override (skip LLM analysis if set)
        self.bottleneck_override = bottleneck_override

        # History tracking for reflexion
        self.attempt_history: deque[OptimizationAttempt] = deque(maxlen=10)
        self.reflexions: list[Reflexion] = []
        self.history_size: int = 5

        # Initialize from prior history if provided (shared from beam search manager)
        if prior_history:
            for attempt_dict in prior_history:
                self.attempt_history.append(OptimizationAttempt.from_dict(attempt_dict))
        if prior_reflexions:
            for reflexion_dict in prior_reflexions:
                self.reflexions.append(Reflexion.from_dict(reflexion_dict))

    def optimize_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
        known_kernel_time: float | None = None,
        max_opt_rounds: int = 5,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Main optimization loop.

        Args:
            kernel_code: Initial kernel code
            problem_file: Path to problem file
            test_code: Test code for verification
            known_kernel_time: Known performance of kernel_code in ms
            max_opt_rounds: Maximum optimization rounds

        Returns:
            Tuple of (success, best_kernel_code, performance_metrics)
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting hardware-guided optimization")
        self.logger.info("=" * 80)

        # Initialize state
        current_kernel = kernel_code
        error_feedback = ""
        best_ncu_metrics: dict[str, Any] | None = None
        best_bottleneck_category: str | None = None
        best_round_num: int = 0
        early_stop_reason = ""
        any_verified = False

        # Reset roofline history for new optimization run
        self.roofline_analyzer.reset_history()

        # Note: We don't reset attempt_history and reflexions here because
        # they may contain prior history from beam search manager passed in __init__

        # Extract problem description
        problem_description = problem_file.read_text()
        self.logger.info(f"Problem: {problem_description[:100]}...")

        # Benchmark baseline and PyTorch (now includes baseline SOL profiling)
        best_time, baseline_results, pytorch_baseline_time, baseline_sol = (
            self._benchmark_baseline(kernel_code, problem_file, known_kernel_time)
        )

        # Two-kernel tracking: track best-by-runtime and best-by-SOL independently
        # This prevents mixing metrics from different kernels
        best_runtime_kernel = kernel_code
        best_runtime_time = best_time
        best_runtime_sol = baseline_sol

        best_sol_kernel = kernel_code
        best_sol_time = best_time
        best_sol_sol = baseline_sol

        # Optimization rounds
        for round_num in range(1, max_opt_rounds + 1):
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"ROUND {round_num}/{max_opt_rounds}")
            self.logger.info("=" * 80)

            # Profile and analyze bottleneck
            bottleneck_results, roofline_result, ncu_metrics = (
                self._profile_and_analyze(current_kernel, problem_file, round_num)
            )

            # Log roofline for the kernel we just profiled
            if ncu_metrics:
                flat_metrics = _get_triton_kernel_metrics(ncu_metrics)
                roofline_check = self.roofline_analyzer.analyze(
                    ncu_metrics=flat_metrics,
                )
                self.logger.info(
                    f"[{round_num}] Roofline (kernel_round_{round_num - 1}): "
                    f"{roofline_check.bottleneck}-bound, {roofline_check.efficiency_pct:.1f}% SOL "
                    f"(Compute: {roofline_check.compute_sol_pct:.1f}%, "
                    f"Memory: {roofline_check.memory_sol_pct:.1f}%)"
                )

            if not bottleneck_results:
                self.logger.warning(
                    f"[{round_num}] No analysis available, skipping round"
                )
                continue

            # Build optimization prompt using PromptManager with correct API
            # Select bottleneck based on bottleneck_id for beam search diversity
            if self.bottleneck_id is not None and len(bottleneck_results) >= self.bottleneck_id:
                primary = bottleneck_results[self.bottleneck_id - 1]  # 1-indexed
            else:
                primary = bottleneck_results[0]

            # Get recent attempts for history (limit to history_size)
            recent_attempts = list(self.attempt_history)[-self.history_size:]

            # Create current attempt (will be completed after benchmarking)
            current_attempt = OptimizationAttempt(
                round_num=round_num,
                worker_id=0,
                bottleneck_category=primary.category,
                root_cause=primary.root_causes[0].get("cause", "") if primary.root_causes else "",
                recommended_fix=primary.recommended_fixes[0].get("fix", "") if primary.recommended_fixes else "",
                time_before_ms=best_runtime_time,
            )

            # Extract config from current kernel for tracking changes
            current_config = extract_triton_config(current_kernel)

            # RAG retrieval
            rag_context = None
            if self.rag_prescriber is not None:
                rag_query = f"{primary.category}: {primary.summary}"
                if primary.recommended_fixes:
                    rag_query += f" {primary.recommended_fixes[0].get('fix', '')}"
                try:
                    opt_node, scores = self.rag_prescriber.retrieve(rag_query)
                    if opt_node is not None:
                        rag_context = self.rag_prescriber.build_context(opt_node)
                        self.logger.info(f"[{round_num}] RAG retrieved pattern (len={len(rag_context)})")
                except Exception as e:
                    self.logger.warning(f"[{round_num}] RAG retrieval failed: {e}")

            opt_prompt = self.prompt_manager.render_kernel_optimization_prompt(
                problem_description=problem_description,
                kernel_code=current_kernel,
                gpu_specs=self.gpu_specs,
                roofline=roofline_result.to_dict() if roofline_result else {},
                category=primary.category,
                summary=primary.summary,
                reasoning=primary.reasoning,
                root_cause=primary.root_causes[0] if primary.root_causes else {},
                recommended_fix=(
                    primary.recommended_fixes[0] if primary.recommended_fixes else {}
                ),
                pytorch_baseline_ms=pytorch_baseline_time,
                current_best_ms=best_runtime_time,
                error_feedback=error_feedback if error_feedback else None,
                recent_attempts=recent_attempts if recent_attempts else None,
                reflexions=self.reflexions[-self.history_size:] if self.reflexions else None,
                rag_context=rag_context,
            )

            # Save prompt
            prompt_file = self.artifact_dir / f"round{round_num:03d}_opt_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(opt_prompt)

            # Generate optimized kernel
            optimized_kernel = self._generate_optimized_kernel(opt_prompt, round_num)
            if not optimized_kernel:
                error_feedback = "Failed to extract valid kernel code. Please provide complete kernel wrapped in ```python blocks."
                continue

            # Verify and refine
            success, optimized_kernel, verify_error = self._verify_and_refine(
                optimized_kernel, test_code, problem_description, round_num
            )
            if not success:
                error_feedback = (
                    verify_error or "Previous attempt failed correctness check."
                )
                # Record failed attempt
                current_attempt.passed_verification = False
                current_attempt.error_message = error_feedback
                self.attempt_history.append(current_attempt)
                continue

            error_feedback = ""

            # Save and benchmark
            kernel_file_round = self.artifact_dir / f"kernel_round_{round_num}.py"
            kernel_file_round.write_text(optimized_kernel)

            bench_results = self.benchmarker.benchmark_kernel(
                kernel_file_round, problem_file
            )
            new_time = bench_results["time_ms"]

            # Profile the NEW kernel to get its SOL metrics
            new_kernel_metrics = self._profile_kernel_for_sol(
                optimized_kernel, problem_file, round_num
            )
            new_sol = (
                new_kernel_metrics.get("efficiency_pct", 0.0)
                if new_kernel_metrics
                else 0.0
            )

            # Complete the attempt with benchmark results
            new_config = extract_triton_config(optimized_kernel)
            config_changes = {}
            for key in set(current_config.keys()) | set(new_config.keys()):
                old_val = current_config.get(key)
                new_val = new_config.get(key)
                if old_val != new_val:
                    config_changes[key] = f"{old_val}→{new_val}"

            improvement_pct = ((best_runtime_time - new_time) / best_runtime_time * 100) if best_runtime_time > 0 else 0
            current_attempt.time_after_ms = new_time
            current_attempt.improvement_pct = improvement_pct
            current_attempt.is_improvement = new_time < best_runtime_time
            current_attempt.passed_verification = True
            any_verified = True
            current_attempt.config_changes = config_changes

            # Add SOL metrics from new kernel profiling
            if new_kernel_metrics:
                current_attempt.compute_sol_pct = new_kernel_metrics.get("compute_sol_pct", 0.0)
                current_attempt.memory_sol_pct = new_kernel_metrics.get("memory_sol_pct", 0.0)
                current_attempt.combined_sol_pct = new_sol

            # Add attempt to history
            self.attempt_history.append(current_attempt)

            # Generate reflexion for this attempt
            reflexion = self._generate_reflexion(current_attempt)
            if reflexion:
                self.reflexions.append(reflexion)

            # Update kernels using two-kernel tracking
            # This keeps best-runtime and best-SOL kernels separate to avoid metric mixing
            (
                current_kernel,
                best_runtime_kernel,
                best_runtime_time,
                best_runtime_sol,
                best_sol_kernel,
                best_sol_time,
                best_sol_sol,
            ) = self._update_kernels(
                optimized_kernel,
                new_time,
                new_sol,
                current_kernel,
                best_runtime_kernel,
                best_runtime_time,
                best_runtime_sol,
                best_sol_kernel,
                best_sol_time,
                best_sol_sol,
                round_num,
            )

            # Track metadata when new best runtime is found
            if new_time < best_runtime_time or new_sol > best_sol_sol:
                best_round_num = round_num
                best_bottleneck_category = primary.category
                if new_kernel_metrics:
                    best_ncu_metrics = new_kernel_metrics.get("ncu_metrics")

            # Roofline check for early termination
            # Use best_runtime kernel's SOL for early termination check
            # We want a kernel that is both fast AND efficient
            if new_kernel_metrics:
                roofline_check = new_kernel_metrics.get("roofline_result")
                if roofline_check:
                    self.logger.info(
                        f"[{round_num}] Roofline: {roofline_check.bottleneck}-bound, "
                        f"{roofline_check.efficiency_pct:.1f}% SOL "
                        f"(Compute: {roofline_check.compute_sol_pct:.1f}%, "
                        f"Memory: {roofline_check.memory_sol_pct:.1f}%)"
                    )

                    # Only early terminate if the best runtime kernel is at roofline
                    # This prevents stopping with a slow but "efficient" kernel
                    if (
                        best_runtime_kernel == optimized_kernel
                        and roofline_check.at_roofline
                    ):
                        should_stop, stop_reason = self.roofline_analyzer.should_stop(
                            roofline_check
                        )
                        if should_stop and self.roofline_analyzer.config.early_stop:
                            self.logger.info(
                                f"[{round_num}] 🎯 Early termination: {stop_reason}"
                            )
                            early_stop_reason = stop_reason
                            break

        # Profile the final best kernel to get its roofline
        if best_round_num > 0:
            final_kernel_file = self.artifact_dir / f"kernel_round_{best_round_num}.py"
            if final_kernel_file.exists():
                self.logger.info(
                    f"Profiling final best kernel (round {best_round_num})..."
                )
                final_profiler_results = self.profiler.profile_kernel(
                    final_kernel_file, problem_file, best_round_num
                )
                if final_profiler_results and final_profiler_results.metrics:
                    best_ncu_metrics = final_profiler_results.metrics
                    final_flat_metrics = _get_triton_kernel_metrics(best_ncu_metrics)
                    final_roofline = self.roofline_analyzer.analyze(
                        ncu_metrics=final_flat_metrics,
                    )
                    self.logger.info(
                        f"Final roofline (kernel_round_{best_round_num}): "
                        f"{final_roofline.bottleneck}-bound, {final_roofline.efficiency_pct:.1f}% SOL "
                        f"(Compute: {final_roofline.compute_sol_pct:.1f}%, "
                        f"Memory: {final_roofline.memory_sol_pct:.1f}%)"
                    )

        # Final results - use best runtime kernel as primary result
        return self._finalize_results(
            best_runtime_kernel,
            best_runtime_time,
            best_runtime_sol,
            best_sol_kernel,
            best_sol_time,
            best_sol_sol,
            baseline_results,
            pytorch_baseline_time,
            max_opt_rounds,
            best_ncu_metrics,
            best_bottleneck_category,
            best_round_num,
            early_stop_reason,
            any_verified,
        )

    def _benchmark_baseline(
        self, kernel_code: str, problem_file: Path, known_kernel_time: float | None
    ) -> tuple[float, dict[str, float], float | None, float]:
        """Benchmark baseline kernel and PyTorch, and profile baseline SOL.

        Returns:
            Tuple of (best_time, baseline_results, pytorch_baseline_time, baseline_sol)
        """
        baseline_sol = 0.0

        if known_kernel_time and known_kernel_time != float("inf"):
            best_time = known_kernel_time
            baseline_results = {"time_ms": known_kernel_time, "speedup": 1.0}
            self.logger.info(f"📊 Using known kernel time: {best_time:.4f} ms")
            # Still need to profile for SOL
            kernel_file_round = self.artifact_dir / "kernel_round_0.py"
            kernel_file_round.write_text(kernel_code)
        else:
            _write_kernel_file(self.kernel_file, kernel_code, self.logger)
            kernel_file_round = self.artifact_dir / "kernel_round_0.py"
            kernel_file_round.write_text(kernel_code)

            baseline_results = self.benchmarker.benchmark_kernel(
                kernel_file_round, problem_file
            )
            best_time = baseline_results["time_ms"]
            self.logger.info(f"📊 Baseline time: {best_time:.4f} ms")

        # Profile baseline kernel for SOL metrics
        baseline_metrics = self._profile_kernel_for_sol(kernel_code, problem_file, 0)
        if baseline_metrics:
            baseline_sol = baseline_metrics.get("efficiency_pct", 0.0)
            bottleneck = baseline_metrics.get("bottleneck", "unknown")
            compute_sol = baseline_metrics.get("compute_sol_pct", 0.0)
            memory_sol = baseline_metrics.get("memory_sol_pct", 0.0)
            self.logger.info(
                f"📊 Baseline SOL: {baseline_sol:.1f}% ({bottleneck}-bound, "
                f"Compute: {compute_sol:.1f}%, Memory: {memory_sol:.1f}%)"
            )

        # PyTorch baseline
        if self.pytorch_baseline_time is not None:
            pytorch_baseline_time = self.pytorch_baseline_time
            if pytorch_baseline_time != float("inf"):
                self.logger.info(
                    f"📊 PyTorch baseline: {pytorch_baseline_time:.4f} ms (pre-computed)"
                )
            else:
                pytorch_baseline_time = None
        else:
            pytorch_results = self.benchmarker.benchmark_pytorch(problem_file)
            pytorch_baseline_time = pytorch_results.get("time_ms", float("inf"))
            if pytorch_baseline_time != float("inf"):
                self.logger.info(f"📊 PyTorch baseline: {pytorch_baseline_time:.4f} ms")
            else:
                pytorch_baseline_time = None

        return best_time, baseline_results, pytorch_baseline_time, baseline_sol

    def _profile_and_analyze(
        self,
        current_kernel: str,
        problem_file: Path,
        round_num: int,
    ) -> tuple[list[BottleneckResult] | None, Any | None, dict[str, Any] | None]:
        """Profile kernel and analyze bottlenecks.

        Returns:
            Tuple of (bottleneck_results, roofline_result, ncu_metrics).
            All can be None if profiling fails.
        """
        self.logger.info(f"[{round_num}] Profiling current kernel with NCU...")
        kernel_file_round = self.artifact_dir / f"kernel_round_{round_num - 1}.py"
        kernel_file_round.write_text(current_kernel)

        profiler_results = self.profiler.profile_kernel(
            kernel_file_round, problem_file, round_num
        )

        if profiler_results is None:
            self.logger.warning(f"[{round_num}] Profiling failed")
            return None, None, None

        ncu_metrics = profiler_results.metrics

        if not ncu_metrics:
            return None, None, ncu_metrics

        # Run roofline analysis
        flat_metrics = next(iter(ncu_metrics.values()), {}) if ncu_metrics else {}
        roofline_result = self.bottleneck_analyzer.roofline.analyze(flat_metrics)

        # Use pre-computed bottleneck if override is set
        if self.bottleneck_override:
            self.logger.info(
                f"[{round_num}] Using pre-computed bottleneck: {self.bottleneck_override}-bound (with LLM analysis for details)"
            )
            # Call LLM to get diverse root_causes and recommended_fixes
            llm_results = self.bottleneck_analyzer.analyze(
                current_kernel, ncu_metrics, round_num, roofline_result
            )

            if llm_results:
                # Create results with pre-computed category but LLM-generated details
                # This preserves diversity across workers (different bottleneck_id = different root causes)
                bottleneck_results = [
                    BottleneckResult(
                        category=self.bottleneck_override,  # Override with pre-computed
                        summary=f"Pre-computed: {self.bottleneck_override}-bound kernel",
                        reasoning=r.reasoning,
                        root_causes=r.root_causes,
                        recommended_fixes=r.recommended_fixes,
                    )
                    for r in llm_results
                ]
            else:
                self.logger.warning(f"[{round_num}] LLM analysis failed, using empty root_causes/fixes")
                bottleneck_results = [
                    BottleneckResult(
                        category=self.bottleneck_override,
                        summary=f"Pre-computed: {self.bottleneck_override}-bound kernel",
                        reasoning="Classification based on operation arithmetic intensity",
                        root_causes=[],
                        recommended_fixes=[],
                    )
                ]
        else:
            # Run bottleneck analysis via LLM
            self.logger.info(f"[{round_num}] Analyzing bottleneck...")
            bottleneck_results = self.bottleneck_analyzer.analyze(
                current_kernel, ncu_metrics, round_num, roofline_result
            )

        if bottleneck_results:
            strategy_file = self.artifact_dir / f"round{round_num:03d}_strategy.json"
            with open(strategy_file, "w") as f:
                json.dump([r.to_dict() for r in bottleneck_results], f, indent=2)
            return bottleneck_results, roofline_result, ncu_metrics

        return None, roofline_result, ncu_metrics

    def _profile_kernel_for_sol(
        self,
        kernel_code: str,
        problem_file: Path,
        round_num: int,
    ) -> dict[str, Any] | None:
        """Profile a kernel to get its SOL metrics.

        This is a lightweight profiling specifically for SOL measurement,
        used to evaluate the new kernel after benchmarking.

        Args:
            kernel_code: Kernel code to profile
            problem_file: Path to problem file
            round_num: Current round number

        Returns:
            Dict with efficiency_pct, roofline_result, ncu_metrics, or None if profiling fails
        """
        try:
            # Write kernel to temp file for profiling
            kernel_file = self.artifact_dir / f"kernel_round_{round_num}_sol.py"
            kernel_file.write_text(kernel_code)

            profiler_results = self.profiler.profile_kernel(
                kernel_file, problem_file, round_num
            )

            if profiler_results is None or not profiler_results.metrics:
                return None

            ncu_metrics = profiler_results.metrics
            flat_metrics = _get_triton_kernel_metrics(ncu_metrics)

            # Run roofline analysis
            roofline_result = self.roofline_analyzer.analyze(ncu_metrics=flat_metrics)

            return {
                "efficiency_pct": roofline_result.efficiency_pct,
                "compute_sol_pct": roofline_result.compute_sol_pct,
                "memory_sol_pct": roofline_result.memory_sol_pct,
                "bottleneck": roofline_result.bottleneck,
                "roofline_result": roofline_result,
                "ncu_metrics": ncu_metrics,
            }

        except Exception as e:
            self.logger.warning(f"[{round_num}] SOL profiling failed: {e}")
            return None

    def _generate_optimized_kernel(self, opt_prompt: str, round_num: int) -> str | None:
        """Generate optimized kernel from LLM."""
        self.logger.info(f"[{round_num}] Generating optimized kernel...")
        try:
            messages = [{"role": "user", "content": opt_prompt}]
            response_text = _call_llm(
                provider=self.provider,
                model=self.model,
                messages=messages,
                high_reasoning_effort=self.high_reasoning_effort,
                logger=self.logger,
                max_tokens=24576,
            )

            # Save response
            response_file = self.artifact_dir / f"round{round_num:03d}_opt_reply.txt"
            with open(response_file, "w") as f:
                f.write(response_text)

            # Extract code
            optimized_kernel = _extract_code_from_response(
                response_text=response_text,
                logger=self.logger,
            )

            if not optimized_kernel or len(optimized_kernel) < 100:
                self.logger.warning(
                    f"[{round_num}] Failed to extract valid kernel code"
                )
                return None

            return optimized_kernel

        except Exception as e:
            self.logger.error(f"[{round_num}] LLM call failed: {e}")
            return None

    def _verify_and_refine(
        self,
        optimized_kernel: str,
        test_code: str,
        problem_description: str,
        round_num: int,
    ) -> tuple[bool, str, str]:
        """
        Verify kernel correctness with refinement attempts.

        Returns:
            Tuple of (success, final_kernel, error_feedback)
        """
        self.logger.info(f"[{round_num}] Verifying correctness...")
        success, final_kernel, error_feedback = (
            self.verification_worker.verify_with_refinement(
                kernel_code=optimized_kernel,
                test_code=test_code,
                problem_description=problem_description,
            )
        )

        if success:
            self.logger.info(f"[{round_num}] ✅ Correctness check passed")
        else:
            self.logger.warning(f"[{round_num}] ❌ Correctness check failed")

        return success, final_kernel, error_feedback

    def _generate_reflexion(self, attempt: OptimizationAttempt) -> Reflexion | None:
        """
        Generate a reflexion for an optimization attempt using LLM.

        Args:
            attempt: The completed optimization attempt to reflect on

        Returns:
            Reflexion object, or None if generation fails
        """
        if not attempt.passed_verification:
            # For failed attempts, create a simple reflexion without LLM
            return Reflexion(
                round_num=attempt.round_num,
                root_cause_diagnosed=attempt.root_cause,
                fix_applied=attempt.recommended_fix,
                expected_outcome="Improve performance",
                actual_outcome="Failed verification",
                performance_delta_pct=0.0,
                was_diagnosis_correct=False,
                was_fix_effective=False,
                reasoning=f"Attempt failed verification: {attempt.error_message[:200] if attempt.error_message else 'Unknown error'}",
                lessons=["Ensure generated code passes correctness checks"],
                avoid_patterns=[f"Similar approach to round {attempt.round_num} that failed verification"],
                try_patterns=[],
            )

        try:
            # Build reflexion prompt
            reflexion_prompt = self.prompt_manager.render_reflexion_prompt(attempt)

            messages = [{"role": "user", "content": reflexion_prompt}]
            response_text = _call_llm(
                provider=self.provider,
                model=self.model,
                messages=messages,
                high_reasoning_effort=False,  # Use standard reasoning for reflexion
                logger=self.logger,
                max_tokens=2048,
            )

            # Save reflexion response
            reflexion_file = self.artifact_dir / f"round{attempt.round_num:03d}_reflexion.txt"
            with open(reflexion_file, "w") as f:
                f.write(response_text)

            # Parse JSON from response
            reflexion_data = self._parse_reflexion_response(response_text, attempt)
            return reflexion_data

        except Exception as e:
            self.logger.warning(f"[{attempt.round_num}] Failed to generate reflexion: {e}")
            return self._fallback_reflexion(attempt)

    def _fallback_reflexion(self, attempt: OptimizationAttempt, reasoning: str | None = None) -> Reflexion:
        """Create a basic reflexion from attempt data when LLM is unavailable."""
        return Reflexion(
            round_num=attempt.round_num,
            root_cause_diagnosed=attempt.root_cause,
            fix_applied=attempt.recommended_fix,
            expected_outcome="Improve performance by addressing bottleneck",
            actual_outcome="Improved" if attempt.is_improvement else "No improvement",
            performance_delta_pct=attempt.improvement_pct,
            was_diagnosis_correct=attempt.is_improvement,
            was_fix_effective=attempt.is_improvement,
            reasoning=reasoning or f"Performance changed by {attempt.improvement_pct:+.1f}%",
            lessons=[],
            avoid_patterns=[],
            try_patterns=[],
        )

    def _parse_reflexion_response(
        self, response_text: str, attempt: OptimizationAttempt
    ) -> Reflexion:
        """Parse LLM response into Reflexion object."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Reflexion(
                    round_num=attempt.round_num,
                    root_cause_diagnosed=attempt.root_cause,
                    fix_applied=attempt.recommended_fix,
                    expected_outcome=data.get("expected_outcome", "Improve performance"),
                    actual_outcome=data.get("actual_outcome", ""),
                    performance_delta_pct=attempt.improvement_pct,
                    was_diagnosis_correct=data.get("was_diagnosis_correct", attempt.is_improvement),
                    was_fix_effective=data.get("was_fix_effective", attempt.is_improvement),
                    reasoning=data.get("reasoning", ""),
                    lessons=data.get("lessons", []),
                    avoid_patterns=data.get("avoid_patterns", []),
                    try_patterns=data.get("try_patterns", []),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: create reflexion from attempt data
        return self._fallback_reflexion(
            attempt,
            reasoning=f"Applied {attempt.recommended_fix}. Performance changed by {attempt.improvement_pct:+.1f}%",
        )

    def _update_kernels(
        self,
        optimized_kernel: str,
        new_time: float,
        new_sol: float,
        current_kernel: str,
        best_runtime_kernel: str,
        best_runtime_time: float,
        best_runtime_sol: float,
        best_sol_kernel: str,
        best_sol_time: float,
        best_sol_sol: float,
        round_num: int,
    ) -> tuple[str, str, float, float, str, float, float]:
        """Update current and best kernels based on performance and SOL.

        Uses two-kernel tracking to avoid mixing metrics from different kernels:
        - best_runtime_kernel: The kernel with the best runtime (primary result)
        - best_sol_kernel: The kernel with the best SOL (for analysis)

        Returns:
            Tuple of (current_kernel,
                      best_runtime_kernel, best_runtime_time, best_runtime_sol,
                      best_sol_kernel, best_sol_time, best_sol_sol)
        """
        updated_runtime_kernel = best_runtime_kernel
        updated_runtime_time = best_runtime_time
        updated_runtime_sol = best_runtime_sol

        updated_sol_kernel = best_sol_kernel
        updated_sol_time = best_sol_time
        updated_sol_sol = best_sol_sol

        # Check for runtime improvement (independent of SOL)
        if new_time < best_runtime_time:
            speedup = best_runtime_time / new_time
            improvement = (best_runtime_time - new_time) / best_runtime_time * 100
            self.logger.info(
                f"[{round_num}] 🎉 NEW BEST RUNTIME! {new_time:.4f} ms "
                f"(speedup: {speedup:.2f}x, improvement: {improvement:.1f}%)"
            )
            if new_sol > 0:
                self.logger.info(f"[{round_num}] 📊 SOL: {new_sol:.1f}%")
            updated_runtime_kernel = optimized_kernel
            updated_runtime_time = new_time
            updated_runtime_sol = new_sol  # This kernel's SOL (consistent!)

        # Check for SOL improvement (independent of runtime)
        if new_sol > best_sol_sol:
            self.logger.info(
                f"[{round_num}] 📈 NEW BEST SOL! {best_sol_sol:.1f}% → {new_sol:.1f}%"
            )
            self.logger.info(
                f"[{round_num}]    Runtime: {new_time:.4f} ms (best runtime: {best_runtime_time:.4f} ms)"
            )
            updated_sol_kernel = optimized_kernel
            updated_sol_time = new_time  # This kernel's time (consistent!)
            updated_sol_sol = new_sol

        # Decide which kernel to continue exploring with
        divergence = (
            (new_time - best_runtime_time) / best_runtime_time * 100
            if best_runtime_time > 0
            else 0
        )

        if divergence > self.divergence_threshold:
            # Too slow - revert to best runtime kernel for next round
            self.logger.warning(
                f"[{round_num}] ⚠️ Excessive divergence ({divergence:.1f}%), "
                f"reverting to best runtime kernel"
            )
            next_kernel = best_runtime_kernel
        else:
            # Continue exploring with new kernel
            if new_time >= best_runtime_time and new_sol <= best_sol_sol:
                self.logger.info(
                    f"[{round_num}] No improvement: {new_time:.4f} ms vs best {best_runtime_time:.4f} ms"
                )
                if new_sol > 0:
                    self.logger.info(
                        f"[{round_num}] SOL: {new_sol:.1f}% (best: {best_sol_sol:.1f}%)"
                    )
            next_kernel = optimized_kernel

        return (
            next_kernel,
            updated_runtime_kernel,
            updated_runtime_time,
            updated_runtime_sol,
            updated_sol_kernel,
            updated_sol_time,
            updated_sol_sol,
        )

    def _finalize_results(
        self,
        best_runtime_kernel: str,
        best_runtime_time: float,
        best_runtime_sol: float,
        best_sol_kernel: str,
        best_sol_time: float,
        best_sol_sol: float,
        baseline_results: dict[str, float],
        pytorch_baseline_time: float | None,
        rounds: int,
        ncu_metrics: dict[str, Any] | None = None,
        bottleneck_category: str | None = None,
        best_round: int = 0,
        early_stop_reason: str = "",
        any_verified: bool = False,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Finalize and log optimization results.

        Reports both best-runtime and best-SOL kernels separately to avoid
        mixing metrics from different kernels. The best-runtime kernel is
        returned as the primary result.
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        if early_stop_reason:
            self.logger.info(f"   (Early termination: {early_stop_reason})")
        self.logger.info("=" * 80)

        baseline_speedup = baseline_results["time_ms"] / best_runtime_time
        improvement_percent = (
            (baseline_results["time_ms"] - best_runtime_time)
            / baseline_results["time_ms"]
            * 100
        )

        self.logger.info("📊 Final Results - BEST BY RUNTIME:")
        self.logger.info(f"   Time: {best_runtime_time:.4f} ms")
        self.logger.info(f"   SOL:  {best_runtime_sol:.1f}%")
        self.logger.info(f"   Baseline time: {baseline_results['time_ms']:.4f} ms")
        self.logger.info(f"   Speedup vs baseline: {baseline_speedup:.2f}x")

        # Report best-SOL kernel if it's different from best-runtime
        if best_sol_kernel != best_runtime_kernel:
            self.logger.info("")
            self.logger.info("📈 BEST BY SOL (different kernel):")
            self.logger.info(f"   Time: {best_sol_time:.4f} ms")
            self.logger.info(f"   SOL:  {best_sol_sol:.1f}%")

        if pytorch_baseline_time and pytorch_baseline_time != float("inf"):
            pytorch_speedup = pytorch_baseline_time / best_runtime_time
            self.logger.info(f"   PyTorch baseline: {pytorch_baseline_time:.4f} ms")
            self.logger.info(f"   Speedup vs PyTorch: {pytorch_speedup:.2f}x")

        self.logger.info(f"   Improvement: {improvement_percent:.1f}%")

        # Save best runtime kernel (primary result)
        best_kernel_file = self.output_dir / "best_kernel.py"
        best_kernel_file.write_text(best_runtime_kernel)

        perf_metrics = {
            "baseline_time_ms": baseline_results["time_ms"],
            "best_time_ms": best_runtime_time,
            "best_runtime_sol_pct": best_runtime_sol,
            "speedup": baseline_speedup,
            "rounds": rounds,
        }

        # Include best-SOL kernel info if different
        if best_sol_kernel != best_runtime_kernel:
            perf_metrics["best_sol_time_ms"] = best_sol_time
            perf_metrics["best_sol_sol_pct"] = best_sol_sol

        if bottleneck_category:
            perf_metrics["bottleneck_addressed"] = bottleneck_category
            perf_metrics["bottleneck_category"] = bottleneck_category

        # Add NCU metrics if available
        if ncu_metrics:
            kernel_metrics = next(iter(ncu_metrics.values()), {})
            perf_metrics["memory_throughput"] = kernel_metrics.get(
                "dram__throughput.avg.pct_of_peak_sustained_elapsed"
            )
            perf_metrics["compute_throughput"] = kernel_metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            )

        if early_stop_reason:
            perf_metrics["early_stop_reason"] = early_stop_reason

        # Include last attempt and reflexion for shared history (beam search)
        if self.attempt_history:
            perf_metrics["last_attempt"] = asdict(self.attempt_history[-1])
        if self.reflexions:
            perf_metrics["last_reflexion"] = asdict(self.reflexions[-1])

        success = best_runtime_time != float("inf") and any_verified
        return success, best_runtime_kernel, perf_metrics
