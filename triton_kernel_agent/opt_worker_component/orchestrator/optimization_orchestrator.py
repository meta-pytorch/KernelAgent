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

import json
import logging
from pathlib import Path
from typing import Any

from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.worker import VerificationWorker
from triton_kernel_agent.worker_util import (
    _call_llm,
    _extract_code_from_response,
    _write_kernel_file,
)
from utils.providers.base import BaseProvider


class OptimizationOrchestrator:
    """Orchestrates the main optimization loop."""

    def __init__(
        self,
        # Components
        profiler: Any,  # KernelProfiler
        benchmarker: Any,  # Benchmark (handles both kernel and PyTorch benchmarking)
        bottleneck_analyzer: Any,  # BottleneckAnalyzer
        verification_worker: VerificationWorker,  # For verify + refine
        prompt_manager: PromptManager,  # PromptManager for building prompts
        # LLM configuration (replaces llm_client, code_extractor adapters)
        provider: BaseProvider,
        model: str,
        high_reasoning_effort: bool,
        # File configuration (replaces file_writer adapter)
        kernel_file: Path,
        # Configuration
        gpu_specs: dict[str, Any] | None,
        enable_ncu_profiling: bool,
        bottleneck_id: int | None,
        pytorch_baseline_time: float | None,
        divergence_threshold: 50.0,  # Max % worse before reverting
        # Paths
        artifact_dir: Path,
        output_dir: Path,
        # Logger
        logger: logging.Logger,
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
            enable_ncu_profiling: Enable NCU profiling
            bottleneck_id: Which bottleneck to explore (1 or 2)
            pytorch_baseline_time: Pre-computed PyTorch baseline
            divergence_threshold: Max % worse performance before reverting to best kernel (default: 50.0)
            artifact_dir: Directory for optimization artifacts (kernels, prompts, responses per round)
            output_dir: Directory for final output (best_kernel.py)
            logger: Logger instance
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
        self.enable_ncu_profiling = enable_ncu_profiling
        self.bottleneck_id = bottleneck_id
        self.pytorch_baseline_time = pytorch_baseline_time
        self.divergence_threshold = divergence_threshold

        # Paths
        self.artifact_dir = artifact_dir
        self.output_dir = output_dir

        # Logger
        self.logger = logger

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
        current_time = float("inf")
        best_time = float("inf")
        error_feedback = ""

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
            bottleneck_analysis = self._profile_and_analyze(
                current_kernel, problem_description, problem_file, round_num
            )

            if not bottleneck_analysis:
                self.logger.warning(
                    f"[{round_num}] No bottleneck analysis available, skipping round"
                )
                continue

            # Build optimization prompt using PromptManager
            opt_prompt = self.prompt_manager.render_kernel_optimization_prompt(
                kernel_code=current_kernel,
                problem_description=problem_description,
                bottleneck_analysis=bottleneck_analysis,
                bottleneck_id=self.bottleneck_id,
                gpu_specs=self.gpu_specs,
                pytorch_baseline_ms=pytorch_baseline_time,
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
            current_kernel, current_time, best_kernel, best_time = self._update_kernels(
                optimized_kernel,
                new_time,
                current_kernel,
                current_time,
                best_kernel,
                best_time,
                round_num,
            )

        # Final results
        return self._finalize_results(
            best_kernel,
            best_time,
            baseline_results,
            pytorch_baseline_time,
            max_opt_rounds,
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
        problem_description: str,
        problem_file: Path,
        round_num: int,
    ) -> dict[str, Any] | None:
        """Profile kernel and analyze bottlenecks."""
        if not self.enable_ncu_profiling:
            self.logger.warning(f"[{round_num}] NCU profiling disabled")
            return None

        self.logger.info(f"[{round_num}] Profiling current kernel with NCU...")
        kernel_file_round = self.artifact_dir / f"kernel_round_{round_num - 1}.py"
        kernel_file_round.write_text(current_kernel)

        profiler_results = self.profiler.profile_kernel(
            kernel_file_round, problem_file, round_num
        )

        if profiler_results is None:
            self.logger.warning(f"[{round_num}] Profiling failed")
            return None

        if profiler_results:
            self.logger.info(f"[{round_num}] Analyzing bottleneck...")
            bottleneck_analysis = self.bottleneck_analyzer.analyze_bottleneck(
                current_kernel, problem_description, profiler_results, round_num
            )

            if bottleneck_analysis:
                strategy_file = (
                    self.artifact_dir / f"round{round_num:03d}_strategy.json"
                )
                with open(strategy_file, "w") as f:
                    json.dump(bottleneck_analysis, f, indent=2)
                return bottleneck_analysis

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

        Delegates to VerificationWorker.verify_with_refinement().

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
        current_time: float,
        best_kernel: str,
        best_time: float,
        round_num: int,
    ) -> tuple[str, float, str, float]:
        """Update current and best kernels based on performance."""
        if new_time < best_time:
            # New best found
            speedup = best_time / new_time
            improvement = (best_time - new_time) / best_time * 100
            self.logger.info(
                f"[{round_num}] 🎉 NEW BEST! {new_time:.4f} ms (speedup: {speedup:.2f}x, improvement: {improvement:.1f}%)"
            )
            return optimized_kernel, new_time, optimized_kernel, new_time
        else:
            # Check for excessive divergence
            divergence = (new_time - best_time) / best_time * 100

            if divergence > self.divergence_threshold:
                self.logger.warning(
                    f"[{round_num}] ⚠️  EGREGIOUS DIVERGENCE: {new_time:.4f} ms is {divergence:.1f}% worse"
                )
                self.logger.warning(f"[{round_num}] 🔄 REVERTING to best kernel")
                return best_kernel, best_time, best_kernel, best_time
            else:
                self.logger.info(
                    f"[{round_num}] No improvement: {new_time:.4f} ms vs best {best_time:.4f} ms"
                )
                return optimized_kernel, new_time, best_kernel, best_time

    def _finalize_results(
        self,
        best_kernel: str,
        best_time: float,
        baseline_results: dict[str, float],
        pytorch_baseline_time: float | None,
        rounds: int,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Finalize and log optimization results."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
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

        success = best_time != float("inf")
        return success, best_kernel, perf_metrics
