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

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
    RooflineConfig,
)
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager
from utils.providers import get_model_provider


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
        >>> success, optimized_kernel, metrics = worker.optimize_kernel(
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
        benchmark_warmup: int = 25,
        benchmark_repeat: int = 100,
        benchmark_lock: Any | None = None,
        profiling_semaphore: Any | None = None,
        pytorch_baseline_time: float | None = None,
        divergence_threshold: float = 50.0,
        sol_improvement_threshold: float = 5.0,
        target_platform: str = "cuda",
        roofline_config: RooflineConfig | None = None,
        # BeamSearch parameters (passed by opt_manager)
        bottleneck_id: int | None = None,
        bottleneck_override: str | None = None,
        # Shared history from beam search manager
        prior_history: list[dict] | None = None,
        prior_reflexions: list[dict] | None = None,
        use_rag: bool = False,
        # Platform components resolved by the manager registry ─────
        platform_components: dict[str, Any] | None = None,
        # Registry-driven platform config (string names) ───────────
        platform_config: dict[str, str] | None = None,
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
            benchmark_warmup: Number of warmup iterations for benchmarking
            benchmark_repeat: Number of repeat iterations for benchmarking
            benchmark_lock: Shared lock to serialize GPU benchmarking
            profiling_semaphore: Semaphore to limit concurrent NCU profiling
            pytorch_baseline_time: Pre-computed PyTorch baseline (ms)
            divergence_threshold: Max % worse performance before reverting
            sol_improvement_threshold: Min SOL % improvement to accept kernel even if runtime doesn't improve
            target_platform: Target platform (cuda, rocm, etc.)
            roofline_config: Roofline configuration (uses defaults if None)
            use_rag: Whether to enable RAG-based prescriber for optimization hints
            platform_components: Dict of registry-resolved component instances
                keyed by registry name (e.g. ``{"profiler": <instance>, ...}``).
                Populated by ``OptimizationManager._resolve_platform()``.
            platform_config: Dict of registry component names (e.g.
                ``{"profiler": "nvidia"}``).  Resolved via the platform
                registry on first use.  Takes effect for any component not
                already present in *platform_components*.
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.pytorch_baseline_time = pytorch_baseline_time
        self.divergence_threshold = divergence_threshold
        self.sol_improvement_threshold = sol_improvement_threshold
        self.target_platform = target_platform
        self.gpu_name = gpu_name
        self.ncu_bin_path = ncu_bin_path
        self.benchmark_warmup = benchmark_warmup
        self.benchmark_repeat = benchmark_repeat
        self.roofline_config = roofline_config or RooflineConfig()
        self.use_rag = use_rag

        # Platform components (registry-resolved, may be empty)
        self._platform = platform_components or {}
        self._platform_config = platform_config or {}

        # BeamSearch parameters
        self.bottleneck_id = bottleneck_id
        self.bottleneck_override = bottleneck_override

        # Shared history from beam search manager
        self.prior_history = prior_history or []
        self.prior_reflexions = prior_reflexions or []

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

        # Initialize benchmark lock and profiling semaphore
        import multiprocessing as mp

        self.benchmark_lock = benchmark_lock or mp.Lock()
        self.profiling_semaphore = (
            profiling_semaphore  # Can be None for standalone usage
        )

        # Resolve registry-driven config into platform instances
        # (needs logger + profiling_semaphore, so must come after their setup)
        self._resolve_platform_config()

        # Get GPU specs (via registry-resolved provider or default NVIDIA lookup)
        specs_provider = self._platform.get("specs_provider")
        if specs_provider is not None:
            self.gpu_specs = specs_provider.get_specs(self.gpu_name)
        else:
            from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
                get_gpu_specs,
            )

            if not self.gpu_name:
                raise ValueError("gpu_name is required (e.g. 'NVIDIA H100 NVL 94GB')")
            self.gpu_specs = get_gpu_specs(self.gpu_name)
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

    def _resolve_platform_config(self) -> None:
        """Resolve ``self._platform_config`` string names via the registry.

        Populates ``self._platform`` with component instances for any
        keys not already present (explicit instances take precedence).
        """
        if not self._platform_config:
            return
        from triton_kernel_agent.platform.registry import registry

        resolved = registry.create_from_config(
            self._platform_config,
            logger=self.logger,
            log_dir=self.log_dir,
            artifacts_dir=self.artifact_dir,
            ncu_bin_path=self.ncu_bin_path,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            gpu_name=self.gpu_name,
            roofline_config=self.roofline_config,
        )
        for k, v in resolved.items():
            if k not in self._platform:
                self._platform[k] = v

    def _init_components(self) -> None:
        """Initialize all modular components.

        Each component checks for a platform-override first; when absent
        the default (NVIDIA/CUDA) implementation is constructed.
        """
        # Prompt manager (platform-aware, but not hardware-specific)
        platform_config = get_platform(self.target_platform)
        self.prompt_manager = PromptManager(target_platform=platform_config)

        # Benchmarking
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        self.benchmarker = Benchmark(
            logger=self.logger,
            artifacts_dir=self.artifact_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=self.worker_id,
            warmup=self.benchmark_warmup,
            repeat=self.benchmark_repeat,
        )

        # Profiler
        if "profiler" in self._platform:
            self.profiler = self._platform["profiler"]
        else:
            from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
                KernelProfiler,
            )

            self.profiler = KernelProfiler(
                logger=self.logger,
                artifacts_dir=self.artifact_dir,
                logs_dir=self.log_dir,
                ncu_bin_path=self.ncu_bin_path,
                profiling_semaphore=self.profiling_semaphore,
            )

        # Bottleneck analyzer
        if "bottleneck_analyzer" in self._platform:
            self.bottleneck_analyzer = self._platform["bottleneck_analyzer"]
        else:
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )

            self.bottleneck_analyzer = BottleneckAnalyzer(
                provider=self.provider,
                model=self.openai_model,
                gpu_specs=self.gpu_specs,
                logs_dir=self.log_dir,
                logger=self.logger,
            )

        # Verification worker (for correctness checks)
        from triton_kernel_agent.worker import VerificationWorker

        self.verification_worker = VerificationWorker(
            worker_id=self.worker_id,
            workdir=self.workdir,
            log_dir=self.log_dir,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            target_platform=self.target_platform,
        )

        # Roofline analyzer
        if "roofline_analyzer" in self._platform:
            self.roofline_analyzer = self._platform["roofline_analyzer"]
        else:
            from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
                RooflineAnalyzer,
            )

            self.roofline_analyzer = RooflineAnalyzer(
                config=self.roofline_config,
                logger=self.logger,
            )

        # RAG prescriber
        if "rag_prescriber" in self._platform:
            self.rag_prescriber = self._platform["rag_prescriber"]
        elif self.use_rag:
            try:
                from triton_kernel_agent.opt_worker_component.prescribing.RAG_based_prescriber import (
                    RAGPrescriber,
                )

                self.rag_prescriber = RAGPrescriber(logger=self.logger)
                self.logger.info("RAG prescriber initialized")
            except Exception as e:
                self.logger.warning(f"RAG prescriber init failed: {e}")
                self.rag_prescriber = None
        else:
            self.rag_prescriber = None

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
            bottleneck_analyzer=self.bottleneck_analyzer,
            verification_worker=self.verification_worker,
            prompt_manager=self.prompt_manager,
            # LLM configuration
            provider=self.provider,
            model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            # File configuration
            kernel_file=self.kernel_file,
            # Configuration
            gpu_specs=self.gpu_specs,
            pytorch_baseline_time=self.pytorch_baseline_time,
            artifact_dir=self.artifact_dir,
            output_dir=self.output_dir,
            logger=self.logger,
            roofline_analyzer=self.roofline_analyzer,
            divergence_threshold=self.divergence_threshold,
            sol_improvement_threshold=self.sol_improvement_threshold,
            bottleneck_id=self.bottleneck_id,
            bottleneck_override=self.bottleneck_override,
            rag_prescriber=self.rag_prescriber,
            # Shared history from beam search manager
            prior_history=self.prior_history,
            prior_reflexions=self.prior_reflexions,
        )

        # Run optimization
        return orchestrator.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_kernel_time,
            max_opt_rounds=max_opt_rounds,
        )
