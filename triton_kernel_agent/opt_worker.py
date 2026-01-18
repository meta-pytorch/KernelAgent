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

"""Hardware-Aware Optimization Worker for Triton Kernels.

This module provides the OptimizationWorker class that integrates:
- Hardware-aware optimization (NCU profiling + GPU specs + bottleneck analysis)
- Correctness verification via VerificationWorker
- Performance benchmarking

The worker assembles modular components from opt_worker_component/:
- KernelProfiler: NCU profiling
- Benchmark: Kernel and PyTorch benchmarking
- OptimizationOrchestrator: Main optimization loop
"""

import logging
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt import (
    build_judge_optimization_prompt,
    extract_judge_response,
    get_gpu_specs,
)
from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import Benchmark
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
    KernelProfiler,
)
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager
from triton_kernel_agent.worker import VerificationWorker
from triton_kernel_agent.worker_util import (
    _call_llm,
    _extract_code_from_response,
    _save_debug_file,
    _write_kernel_file,
)
from utils.providers import get_model_provider
from utils.providers.base import BaseProvider


class PyTorchBenchmark:
    """Wrapper for PyTorch baseline benchmarking."""

    def __init__(self, benchmark: Benchmark, logger: logging.Logger | None = None):
        self.benchmark = benchmark
        self.logger = logger or logging.getLogger(__name__)

    def benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark PyTorch baseline and return time in ms."""
        try:
            results = self.benchmark.benchmark_pytorch(problem_file)
            return results.get("time_ms", float("inf"))
        except Exception as e:
            self.logger.error(f"PyTorch benchmark failed: {e}")
            return float("inf")


class BottleneckAnalyzer:
    """Analyzes NCU metrics to identify performance bottlenecks using LLM.

    This class wraps the Judge LLM workflow:
    1. Build prompt from kernel code, problem description, NCU metrics, GPU specs
    2. Call LLM to analyze bottlenecks
    3. Parse response to extract dual-bottleneck analysis
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        gpu_specs: dict[str, Any],
        high_reasoning_effort: bool = True,
        logs_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.model = model
        self.gpu_specs = gpu_specs
        self.high_reasoning_effort = high_reasoning_effort
        self.logs_dir = logs_dir
        self.logger = logger or logging.getLogger(__name__)

    def analyze_bottleneck(
        self,
        kernel_code: str,
        problem_description: str,
        ncu_metrics: dict[str, Any],
        round_num: int,
    ) -> dict[str, Any] | None:
        """
        Analyze kernel bottlenecks using NCU metrics and Judge LLM.

        Args:
            kernel_code: Current kernel code
            problem_description: Problem description
            ncu_metrics: NCU profiling metrics dictionary
            round_num: Current optimization round number

        Returns:
            Dual-bottleneck analysis dict with bottleneck_1 and bottleneck_2,
            or None if analysis fails
        """
        try:
            # Build the judge prompt
            system_prompt, user_prompt = build_judge_optimization_prompt(
                kernel_code=kernel_code,
                problem_description=problem_description,
                ncu_metrics=ncu_metrics,
                gpu_specs=self.gpu_specs,
            )

            # Save prompt for debugging
            if self.logs_dir:
                prompt_content = (
                    "=== SYSTEM PROMPT ===\n"
                    + system_prompt
                    + "\n\n=== USER PROMPT ===\n"
                    + user_prompt
                )
                _save_debug_file(
                    self.logs_dir / f"round{round_num:03d}_judge_prompt.txt",
                    prompt_content,
                    self.logger,
                )

            # Call LLM using shared utility
            # Note: GPT-5 uses reasoning tokens from max_tokens budget, so we need
            # a higher limit to leave room for both reasoning and response content
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response_text = _call_llm(
                provider=self.provider,
                model=self.model,
                messages=messages,
                high_reasoning_effort=self.high_reasoning_effort,
                logger=self.logger,
                max_tokens=40960 if self.model.startswith("gpt-5") else 12288,
            )

            # Save response for debugging
            if self.logs_dir:
                _save_debug_file(
                    self.logs_dir / f"round{round_num:03d}_judge_response.txt",
                    response_text,
                    self.logger,
                )

            # Parse response
            analysis = extract_judge_response(response_text)
            if analysis:
                self.logger.info(
                    f"[{round_num}] Bottleneck analysis complete: "
                    f"primary={analysis.get('bottleneck_1', {}).get('category', 'unknown')}"
                )
                return analysis
            else:
                self.logger.warning(
                    f"[{round_num}] Failed to parse bottleneck analysis. "
                    f"Response saved to: {self.logs_dir / f'round{round_num:03d}_judge_response.txt'}"
                )
                return None

        except Exception as e:
            self.logger.error(f"[{round_num}] Bottleneck analysis failed: {e}")
            return None


class LLMClientAdapter:
    """Adapter class to provide call_llm interface expected by OptimizationOrchestrator."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        high_reasoning_effort: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.model = model
        self.high_reasoning_effort = high_reasoning_effort
        self.logger = logger or logging.getLogger(__name__)

    def call_llm(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call LLM using the shared utility function."""
        return _call_llm(
            provider=self.provider,
            model=self.model,
            messages=messages,
            high_reasoning_effort=self.high_reasoning_effort,
            logger=self.logger,
            **kwargs,
        )


class CodeExtractorAdapter:
    """Adapter class to provide extract_code_from_response interface."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_code_from_response(
        self, response_text: str, language: str = "python"
    ) -> str | None:
        """Extract code from LLM response text using shared utility."""
        return _extract_code_from_response(
            response_text=response_text,
            language=language,
            logger=self.logger,
        )


class KernelFileWriterAdapter:
    """Adapter class to provide write_kernel interface."""

    def __init__(self, kernel_file: Path, logger: logging.Logger | None = None):
        self.kernel_file = kernel_file
        self.logger = logger or logging.getLogger(__name__)

    def write_kernel(self, kernel_code: str) -> None:
        """Write kernel code to file using shared utility."""
        _write_kernel_file(self.kernel_file, kernel_code, self.logger)


class OptimizationWorker:
    """Hardware-aware optimization worker for Triton kernels.

    This worker orchestrates the full optimization pipeline:
    1. Profile kernel with NCU to identify bottlenecks
    2. Analyze bottlenecks and generate optimization strategies
    3. Use LLM to generate optimized kernel variants
    4. Verify correctness and benchmark performance
    5. Iterate until convergence or max rounds reached

    Example:
        >>> worker = OptimizationWorker(
        ...     worker_id=0,
        ...     workdir=Path("./optimization"),
        ...     log_dir=Path("./logs"),
        ...     openai_model="gpt-5",
        ... )
        >>> success, kernel, metrics = worker.optimize_kernel(
        ...     kernel_code="...",
        ...     problem_file=Path("problem.py"),
        ...     test_code="...",
        ... )
    """

    def __init__(
        self,
        worker_id: int,
        workdir: Path,
        log_dir: Path,
        max_rounds: int = 10,
        openai_model: str = "gpt-5",
        high_reasoning_effort: bool = True,
        gpu_name: str | None = None,
        ncu_bin_path: str | None = None,
        enable_ncu_profiling: bool = True,
        benchmark_warmup: int = 25,
        benchmark_repeat: int = 100,
        bottleneck_id: int | None = None,
        benchmark_lock: Any | None = None,
        pytorch_baseline_time: float | None = None,
        divergence_threshold: float = 50.0,
        target_platform: str = "cuda",
    ):
        """
        Initialize the optimization worker.

        Args:
            worker_id: Unique identifier for this worker
            workdir: Working directory for this worker
            log_dir: Directory for logging
            max_rounds: Maximum optimization rounds
            openai_model: Model name for optimization
            high_reasoning_effort: Whether to use high reasoning effort
            gpu_name: GPU name (auto-detect if None)
            ncu_bin_path: Path to NCU binary (auto-detect if None)
            enable_ncu_profiling: Enable NCU profiling
            benchmark_warmup: Number of warmup iterations for benchmarking
            benchmark_repeat: Number of repeat iterations for benchmarking
            bottleneck_id: Which bottleneck to explore (1 or 2)
            benchmark_lock: Shared lock to serialize GPU benchmarking
            pytorch_baseline_time: Pre-computed PyTorch baseline (ms)
            divergence_threshold: Max % worse performance before reverting
            target_platform: Target platform (cuda, rocm, etc.)
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_id = bottleneck_id or 1
        self.pytorch_baseline_time = pytorch_baseline_time
        self.divergence_threshold = divergence_threshold
        self.target_platform = target_platform
        self.enable_ncu_profiling = enable_ncu_profiling
        self.ncu_bin_path = ncu_bin_path
        self.benchmark_warmup = benchmark_warmup
        self.benchmark_repeat = benchmark_repeat

        # Setup files
        self.kernel_file = self.workdir / "kernel.py"
        self.test_file = self.workdir / "test_kernel.py"

        # Create directories
        self.artifact_dir = self.workdir / "artifacts"
        self.output_dir = self.workdir / "output"

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize benchmark lock
        import multiprocessing as mp

        self.benchmark_lock = benchmark_lock or mp.Lock()

        # Get GPU specs
        self.gpu_specs = get_gpu_specs(gpu_name) if gpu_name else get_gpu_specs()
        self.logger.info(
            f"Initialized for GPU: {self.gpu_specs.get('name', 'unknown')}"
        )

        # Initialize LLM provider (like worker.py)
        self.provider = get_model_provider(self.openai_model)

        # Initialize components
        self._init_components()

    def _setup_logging(self) -> None:
        """Setup worker-specific logging."""
        log_file = self.log_dir / f"opt_worker_{self.worker_id}.log"
        self.logger = logging.getLogger(f"opt_worker_{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)

    def _init_components(self) -> None:
        """Initialize all modular components."""
        # Prompt manager
        platform_config = get_platform(self.target_platform)
        self.prompt_manager = PromptManager(target_platform=platform_config)

        # LLM client adapter (for components that expect object.call_llm())
        self.llm_client = LLMClientAdapter(
            provider=self.provider,
            model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            logger=self.logger,
        )

        # Code extractor adapter
        self.code_extractor = CodeExtractorAdapter(logger=self.logger)

        # File writer adapter
        self.file_writer = KernelFileWriterAdapter(self.kernel_file, logger=self.logger)

        # Benchmarking
        self.benchmarker = Benchmark(
            logger=self.logger,
            artifacts_dir=self.artifact_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=self.worker_id,
            warmup=self.benchmark_warmup,
            repeat=self.benchmark_repeat,
        )

        self.pytorch_benchmarker = PyTorchBenchmark(
            benchmark=self.benchmarker,
            logger=self.logger,
        )

        # Profiler
        self.profiler = KernelProfiler(
            logger=self.logger,
            artifacts_dir=self.artifact_dir,
            logs_dir=self.log_dir,
            ncu_bin_path=self.ncu_bin_path,
        )

        # Bottleneck analyzer
        self.bottleneck_analyzer = BottleneckAnalyzer(
            provider=self.provider,
            model=self.openai_model,
            gpu_specs=self.gpu_specs,
            high_reasoning_effort=self.high_reasoning_effort,
            logs_dir=self.log_dir,
            logger=self.logger,
        )

        # Verification worker (for correctness checks)
        self.verification_worker = VerificationWorker(
            worker_id=self.worker_id,
            workdir=self.workdir,
            log_dir=self.log_dir,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            target_platform=self.target_platform,
        )

        self.logger.info("OptimizationWorker components initialized")

    def optimize_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
        known_kernel_time: float | None = None,
        max_opt_rounds: int | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Run hardware-guided optimization on a kernel.

        Args:
            kernel_code: Initial kernel code to optimize
            problem_file: Path to problem file defining Model and get_inputs()
            test_code: Test code for correctness verification
            known_kernel_time: Known baseline time in ms (skip initial benchmark)
            max_opt_rounds: Maximum optimization rounds (defaults to self.max_rounds)

        Returns:
            Tuple of (success, best_kernel_code, performance_metrics)
        """
        if max_opt_rounds is None:
            max_opt_rounds = self.max_rounds

        self.logger.info(f"Starting optimization (worker {self.worker_id})")

        # Create orchestrator with all components
        orchestrator = OptimizationOrchestrator(
            # Components
            profiler=self.profiler,
            benchmarker=self.benchmarker,
            pytorch_benchmarker=self.pytorch_benchmarker,
            bottleneck_analyzer=self.bottleneck_analyzer,
            verification_worker=self.verification_worker,
            llm_client=self.llm_client,
            code_extractor=self.code_extractor,
            file_writer=self.file_writer,
            prompt_manager=self.prompt_manager,
            # Configuration
            gpu_specs=self.gpu_specs,
            enable_ncu_profiling=self.enable_ncu_profiling,
            bottleneck_id=self.bottleneck_id,
            pytorch_baseline_time=self.pytorch_baseline_time,
            divergence_threshold=self.divergence_threshold,
            # Paths
            artifact_dir=self.artifact_dir,
            output_dir=self.output_dir,
            # Logger
            logger=self.logger,
        )

        # Run optimization
        return orchestrator.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_kernel_time,
            max_opt_rounds=max_opt_rounds,
        )
