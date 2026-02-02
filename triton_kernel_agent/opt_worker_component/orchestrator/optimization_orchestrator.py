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

        # Paths
        self.artifact_dir = artifact_dir
        self.output_dir = output_dir

        # Logger
        self.logger = logger

        # Optional roofline analyzer
        self.roofline_analyzer = roofline_analyzer

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
        best_kernel = kernel_code
        best_time = float("inf")
        error_feedback = ""
        best_ncu_metrics: dict[str, Any] | None = None
        best_bottleneck_category: str | None = None
        best_round_num: int = 0
        early_stop_reason = ""

        # Reset roofline history for new optimization run
        self.roofline_analyzer.reset_history()

        # Extract problem description
        problem_description = problem_file.read_text()
        self.logger.info(f"Problem: {problem_description[:100]}...")

        # Benchmark baseline and PyTorch
        best_time, baseline_results, pytorch_baseline_time = self._benchmark_baseline(
            kernel_code, problem_file, known_kernel_time
        )

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

            if not bottleneck_results:
                self.logger.warning(
                    f"[{round_num}] No analysis available, skipping round"
                )
                continue

            # Build optimization prompt using PromptManager with correct API
            primary = bottleneck_results[0]
            opt_prompt = self.prompt_manager.render_kernel_optimization_prompt(
                problem_description=problem_description,
                kernel_code=current_kernel,
                gpu_specs=self.gpu_specs,
                roofline=roofline_result.to_dict() if roofline_result else {},
                category=primary.category,
                summary=primary.summary,
                reasoning=primary.reasoning,
                root_cause=primary.root_causes[0] if primary.root_causes else {},
                recommended_fix=primary.recommended_fixes[0]
                if primary.recommended_fixes
                else {},
                pytorch_baseline_ms=pytorch_baseline_time,
                current_best_ms=best_time,
                error_feedback=error_feedback if error_feedback else None,
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
                continue

            error_feedback = ""

            # Save and benchmark
            kernel_file_round = self.artifact_dir / f"kernel_round_{round_num}.py"
            kernel_file_round.write_text(optimized_kernel)

            bench_results = self.benchmarker.benchmark_kernel(
                kernel_file_round, problem_file
            )
            new_time = bench_results["time_ms"]

            # Update kernels based on performance
            old_best_time = best_time
            current_kernel, best_kernel, best_time = self._update_kernels(
                optimized_kernel,
                new_time,
                current_kernel,
                best_kernel,
                best_time,
                round_num,
            )

            # Track metadata when new best is found
            if best_time < old_best_time:
                best_round_num = round_num
                best_bottleneck_category = primary.category
                if ncu_metrics:
                    best_ncu_metrics = ncu_metrics

            # Roofline check for early termination
            if ncu_metrics:
                # Get Triton kernel metrics (filter out PyTorch kernels)
                flat_metrics = _get_triton_kernel_metrics(ncu_metrics)
                roofline_check = self.roofline_analyzer.analyze(
                    ncu_metrics=flat_metrics,
                )

                self.logger.info(
                    f"[{round_num}] Roofline: {roofline_check.bottleneck}-bound, "
                    f"{roofline_check.efficiency_pct:.1f}% SOL "
                    f"(Compute: {roofline_check.compute_sol_pct:.1f}%, "
                    f"Memory: {roofline_check.memory_sol_pct:.1f}%)"
                )

                should_stop, stop_reason = self.roofline_analyzer.should_stop(
                    roofline_check
                )
                if should_stop and self.roofline_analyzer.config.early_stop:
                    self.logger.info(
                        f"[{round_num}] 🎯 Early termination: {stop_reason}"
                    )
                    early_stop_reason = stop_reason
                    break

        # Final results
        return self._finalize_results(
            best_kernel,
            best_time,
            baseline_results,
            pytorch_baseline_time,
            max_opt_rounds,
            best_ncu_metrics,
            best_bottleneck_category,
            best_round_num,
            early_stop_reason,
        )

    def _benchmark_baseline(
        self, kernel_code: str, problem_file: Path, known_kernel_time: float | None
    ) -> tuple[float, dict[str, float], float | None]:
        """Benchmark baseline kernel and PyTorch."""
        if known_kernel_time and known_kernel_time != float("inf"):
            best_time = known_kernel_time
            baseline_results = {"time_ms": known_kernel_time, "speedup": 1.0}
            self.logger.info(f"📊 Using known kernel time: {best_time:.4f} ms")
        else:
            _write_kernel_file(self.kernel_file, kernel_code, self.logger)
            kernel_file_round = self.artifact_dir / "kernel_round_0.py"
            kernel_file_round.write_text(kernel_code)

            baseline_results = self.benchmarker.benchmark_kernel(
                kernel_file_round, problem_file
            )
            best_time = baseline_results["time_ms"]
            self.logger.info(f"📊 Baseline time: {best_time:.4f} ms")

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

        return best_time, baseline_results, pytorch_baseline_time

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

        # Run bottleneck analysis
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

    def _update_kernels(
        self,
        optimized_kernel: str,
        new_time: float,
        current_kernel: str,
        best_kernel: str,
        best_time: float,
        round_num: int,
    ) -> tuple[str, str, float]:
        """Update current and best kernels based on performance."""
        if new_time < best_time:
            # New best found
            speedup = best_time / new_time
            improvement = (best_time - new_time) / best_time * 100
            self.logger.info(
                f"[{round_num}] 🎉 NEW BEST! {new_time:.4f} ms (speedup: {speedup:.2f}x, improvement: {improvement:.1f}%)"
            )
            return optimized_kernel, optimized_kernel, new_time
        else:
            # Check for excessive divergence
            divergence = (new_time - best_time) / best_time * 100

            if divergence > self.divergence_threshold:
                self.logger.warning(
                    f"[{round_num}] ⚠️  EXCESSIVE DIVERGENCE: {new_time:.4f} ms is {divergence:.1f}% worse"
                )
                self.logger.warning(f"[{round_num}] 🔄 REVERTING to best kernel")
                return best_kernel, best_kernel, best_time
            else:
                self.logger.info(
                    f"[{round_num}] No improvement: {new_time:.4f} ms vs best {best_time:.4f} ms"
                )
                return optimized_kernel, best_kernel, best_time

    def _finalize_results(
        self,
        best_kernel: str,
        best_time: float,
        baseline_results: dict[str, float],
        pytorch_baseline_time: float | None,
        rounds: int,
        ncu_metrics: dict[str, Any] | None = None,
        bottleneck_category: str | None = None,
        best_round: int = 0,
        early_stop_reason: str = "",
    ) -> tuple[bool, str, dict[str, Any]]:
        """Finalize and log optimization results."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        if early_stop_reason:
            self.logger.info(f"   (Early termination: {early_stop_reason})")
        self.logger.info("=" * 80)

        baseline_speedup = baseline_results["time_ms"] / best_time
        improvement_percent = (
            (baseline_results["time_ms"] - best_time)
            / baseline_results["time_ms"]
            * 100
        )

        self.logger.info("📊 Final Results:")
        self.logger.info(f"   Best time:     {best_time:.4f} ms")
        self.logger.info(f"   Baseline time: {baseline_results['time_ms']:.4f} ms")
        self.logger.info(f"   Speedup vs baseline: {baseline_speedup:.2f}x")

        if pytorch_baseline_time and pytorch_baseline_time != float("inf"):
            pytorch_speedup = pytorch_baseline_time / best_time
            self.logger.info(f"   PyTorch baseline: {pytorch_baseline_time:.4f} ms")
            self.logger.info(f"   Speedup vs PyTorch: {pytorch_speedup:.2f}x")

        self.logger.info(f"   Improvement: {improvement_percent:.1f}%")

        # Save best kernel
        best_kernel_file = self.output_dir / "best_kernel.py"
        best_kernel_file.write_text(best_kernel)

        perf_metrics = {
            "baseline_time_ms": baseline_results["time_ms"],
            "best_time_ms": best_time,
            "speedup": baseline_speedup,
            "rounds": rounds,
        }

        if bottleneck_category:
            perf_metrics["bottleneck_addressed"] = bottleneck_category

        # Add NCU metrics if available
        if ncu_metrics:
            kernel_metrics = next(iter(ncu_metrics.values()), {})
            perf_metrics["memory_throughput"] = kernel_metrics.get(
                "dram__throughput.avg.pct_of_peak_sustained_elapsed"
            )
            perf_metrics["compute_throughput"] = kernel_metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            )

        if bottleneck_category:
            perf_metrics["bottleneck_category"] = bottleneck_category

        if early_stop_reason:
            perf_metrics["early_stop_reason"] = early_stop_reason

        success = best_time != float("inf")
        return success, best_kernel, perf_metrics
