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

"""Optimization Manager for parallel kernel optimization.

This module provides the OptimizationManager class that orchestrates
parallel kernel optimization using pluggable search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization

Example:
    >>> manager = OptimizationManager(
    ...     strategy="beam_search",
    ...     num_workers=4,
    ...     strategy_config={"num_top_kernels": 2, "num_bottlenecks": 2},
    ... )
    >>> result = manager.run_optimization(
    ...     initial_kernel=kernel_code,
    ...     problem_file=Path("problem.py"),
    ...     test_code=test_file.read_text(),
    ...     max_rounds=20,
    ... )
"""

import logging
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.greedy import (
    GreedyStrategy,
)
from triton_kernel_agent.platform.interfaces import (
    KernelBenchmarker,
    KernelVerifier,
    WorkerRunner,
)
from utils.config_injectable import config_injectable

# Maps registry component keys to OptimizationWorker __init__ parameter names.
_WORKER_KWARG_NAMES: dict[str, str] = {
    "specs_provider": "specs_provider",
    "profiler": "profiler_override",
    "roofline_analyzer": "roofline_override",
    "bottleneck_analyzer": "bottleneck_analyzer_override",
    "rag_prescriber": "rag_prescriber_override",
}


def _worker_kwarg_name(registry_key: str) -> str:
    """Map a registry component key to its OptimizationWorker kwarg name."""
    return _WORKER_KWARG_NAMES[registry_key]


@config_injectable
class OptimizationManager:
    """Manages parallel kernel optimization with pluggable strategies.

    Supports:
    - beam_search: Current default (top-N kernels × M bottlenecks)
    - greedy: Simple single-best optimization

    Platform-specific behaviour (verification, benchmarking, worker
    orchestration) is delegated to injectable components that implement
    :class:`KernelVerifier`, :class:`KernelBenchmarker`, and
    :class:`WorkerRunner`.  When these are not supplied the default
    NVIDIA / CUDA implementations are used.
    """

    def __init__(
        self,
        strategy: str = "beam_search",
        num_workers: int = 4,
        max_rounds: int = 10,
        log_dir: Path | str | None = None,
        database_path: Path | str | None = None,
        strategy_config: dict[str, Any] | None = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        bottleneck_override: str | None = None,
        # Platform component overrides ─────────────────────────────
        platform: dict[str, str] | str | None = None,
        verifier: KernelVerifier | None = None,
        benchmarker: KernelBenchmarker | None = None,
        worker_runner: WorkerRunner | None = None,
        **worker_kwargs: Any,
    ):
        """Initialize the optimization manager.

        Args:
            strategy: Search strategy name ("beam_search" or "greedy")
            num_workers: Number of parallel workers
            max_rounds: Maximum optimization rounds
            log_dir: Directory for logs and artifacts
            database_path: Path for program database JSON file
            strategy_config: Strategy-specific configuration
            openai_model: Model name for LLM optimization
            high_reasoning_effort: Whether to use high reasoning effort
            bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
            platform: Platform component selection.  Can be:

                - A **string** — shorthand preset that applies to every
                  component (e.g. ``"noop"`` or ``"nvidia"``).
                - A **dict** mapping component keys to implementation
                  names, e.g. ``{"verifier": "noop", "profiler": "noop"}``.
                  Only the components listed are overridden; the rest
                  fall through to the defaults.
                - ``None`` — use the defaults (NVIDIA for manager-level,
                  concrete classes for worker-level).

                Explicit *verifier* / *benchmarker* / *worker_runner*
                arguments take precedence over anything in *platform*.
            verifier: Optional :class:`KernelVerifier` (default: NVIDIA)
            benchmarker: Optional :class:`KernelBenchmarker` (default: NVIDIA)
            worker_runner: Optional :class:`WorkerRunner` (default: NVIDIA)
            **worker_kwargs: Additional kwargs passed to OptimizationWorker
        """
        self.max_rounds = max_rounds
        self.log_dir = (
            Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="opt_"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize database
        db_path = (
            Path(database_path)
            if database_path
            else self.log_dir / "program_database.json"
        )
        self.database = JSONProgramDatabase(db_path)

        # Initialize strategy
        self.strategy = self._create_strategy(
            strategy, strategy_config or {}, num_workers
        )

        # Validate worker count
        if num_workers != self.strategy.num_workers_needed:
            raise ValueError(
                f"Strategy '{strategy}' requires {self.strategy.num_workers_needed} "
                f"workers, got {num_workers}. Adjust num_workers or strategy_config."
            )

        self.num_workers = num_workers
        self.benchmark_lock = mp.Lock()
        # Semaphore to serialize NCU profiling - NCU requires exclusive GPU access
        # and has high memory overhead, so only one worker should profile at a time
        self.profiling_semaphore = mp.Semaphore(1)

        # Shared history across beam search iterations
        self.shared_history: list[
            dict
        ] = []  # List of serialized OptimizationAttempt dicts
        self.shared_reflexions: list[dict] = []  # List of serialized Reflexion dicts
        self.history_size: int = 10  # Max history entries to pass to workers

        # ── Platform components ──────────────────────────────────
        # Resolve from registry when `platform` is provided, but let
        # explicit instance arguments win.
        resolved = self._resolve_platform(platform)
        self.verifier = verifier or resolved.get("verifier") or self._default_verifier()
        self.benchmarker = (
            benchmarker or resolved.get("benchmarker") or self._default_benchmarker()
        )
        self.worker_runner = (
            worker_runner
            or resolved.get("worker_runner")
            or self._default_worker_runner()
        )

        # Propagate any worker-level component names from the platform
        # config so that OptimizationWorker can pick them up via
        # worker_kwargs (e.g. profiler_override, specs_provider, …).
        worker_level_keys = {
            "specs_provider",
            "profiler",
            "roofline_analyzer",
            "bottleneck_analyzer",
            "rag_prescriber",
        }
        for key in worker_level_keys:
            if key in resolved and key not in self.worker_kwargs:
                self.worker_kwargs[_worker_kwarg_name(key)] = resolved[key]

        self.logger.info(
            f"OptimizationManager initialized: strategy={strategy}, workers={num_workers}"
        )

    # ------------------------------------------------------------------
    # Platform registry resolution
    # ------------------------------------------------------------------

    def _resolve_platform(
        self, platform: dict[str, str] | str | None
    ) -> dict[str, Any]:
        """Resolve *platform* specification into component instances.

        Returns a dict keyed by component name whose values are either
        instantiated objects (manager-level) or objects to pass through
        to the worker (worker-level).  Missing keys simply mean "use
        the default".
        """
        if platform is None:
            return {}

        from triton_kernel_agent.platform.registry import registry

        # Shorthand string → expand to all registered components for
        # that implementation name.
        if isinstance(platform, str):
            platform = {
                comp: platform
                for comp in registry.list_components()
                if registry.has(comp, platform)
            }

        # Build a shared kwargs bag that the registry can pick from
        # when constructing manager-level factories.
        shared_kwargs: dict[str, Any] = {
            "log_dir": self.log_dir,
            "logger": self.logger,
            "benchmark_lock": self.benchmark_lock,
            "profiling_semaphore": self.profiling_semaphore,
            "openai_model": self.openai_model,
            "high_reasoning_effort": self.high_reasoning_effort,
            "bottleneck_override": self.bottleneck_override,
            "worker_kwargs": self.worker_kwargs,
        }

        return registry.create_from_config(platform, **shared_kwargs)

    # ------------------------------------------------------------------
    # Default (NVIDIA) component factories
    # ------------------------------------------------------------------

    def _default_verifier(self) -> KernelVerifier:
        from triton_kernel_agent.platform.nvidia import NvidiaVerifier

        return NvidiaVerifier(log_dir=self.log_dir, logger=self.logger)

    def _default_benchmarker(self) -> KernelBenchmarker:
        from triton_kernel_agent.platform.nvidia import NvidiaBenchmarker

        return NvidiaBenchmarker(
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
        )

    def _default_worker_runner(self) -> WorkerRunner:
        from triton_kernel_agent.platform.nvidia import NvidiaWorkerRunner

        return NvidiaWorkerRunner(
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            bottleneck_override=self.bottleneck_override,
            worker_kwargs=self.worker_kwargs,
        )

    # ------------------------------------------------------------------
    # Logging / strategy helpers (unchanged)
    # ------------------------------------------------------------------

    def _setup_logging(self) -> logging.Logger:
        """Setup manager logging."""
        logger = logging.getLogger("OptimizationManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / "manager.log")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console)

        return logger

    def _create_strategy(
        self, name: str, config: dict[str, Any], num_workers: int
    ) -> SearchStrategy:
        """Create the search strategy.

        Args:
            name: Strategy name
            config: Strategy-specific configuration
            num_workers: Number of workers

        Returns:
            Configured SearchStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        if name == "beam_search":
            return BeamSearchStrategy(
                num_top_kernels=config.get("num_top_kernels", 2),
                num_bottlenecks=config.get("num_bottlenecks", 2),
                database=self.database,
                logger=self.logger,
            )
        elif name == "greedy":
            return GreedyStrategy(
                database=self.database,
                max_no_improvement=config.get("max_no_improvement", 5),
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unknown strategy: {name}. Use 'beam_search' or 'greedy'")

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def run_optimization(
        self,
        initial_kernel: str,
        problem_file: Path | str,
        test_code: str,
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization with the configured strategy.

        Args:
            initial_kernel: Starting kernel code
            problem_file: Path to problem.py defining Model and get_inputs()
            test_code: Test code for correctness verification
            max_rounds: Override max_rounds (optional)
            **kwargs: Additional kwargs (reserved for future use)

        Returns:
            Dict with:
                - success: bool
                - kernel_code: str | None
                - best_time_ms: float
                - total_rounds: int
                - top_kernels: list[dict]
        """
        max_rounds = max_rounds or self.max_rounds
        problem_file = Path(problem_file)

        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZATION")
        self.logger.info("=" * 80)

        # Initialize strategy with starting kernel
        initial_entry = ProgramEntry(
            program_id="initial",
            kernel_code=initial_kernel,
            metrics=ProgramMetrics(time_ms=float("inf")),
            problem_id=str(problem_file),
        )
        self.strategy.initialize(initial_entry)

        # Verify initial kernel correctness before investing in benchmarks/optimization
        if not self._verify_initial_kernel(initial_kernel, problem_file, test_code):
            return {
                "success": False,
                "kernel_code": None,
                "best_time_ms": float("inf"),
                "total_rounds": 0,
                "top_kernels": [],
                "error": "Initial kernel failed correctness verification",
            }

        # Benchmark PyTorch baseline once (before spawning workers)
        pytorch_baseline = self._benchmark_pytorch_baseline(problem_file)

        # Benchmark torch.compile baseline
        pytorch_compile_time = self._benchmark_pytorch_compile(problem_file)

        # Benchmark the initial kernel
        initial_kernel_time = self._benchmark_initial_kernel(
            initial_kernel, problem_file
        )

        # Round loop
        round_num = 0
        for round_num in range(1, max_rounds + 1):
            self.logger.info("")
            self.logger.info(f"{'=' * 20} ROUND {round_num}/{max_rounds} {'=' * 20}")

            # 1. Get candidates from strategy
            candidates = self.strategy.select_candidates(round_num)
            if not candidates:
                self.logger.warning("No candidates to explore, terminating")
                break

            # 2. Spawn workers
            results = self._run_workers(
                candidates, round_num, problem_file, test_code, pytorch_baseline
            )

            # 3. Update strategy with results
            self.strategy.update_with_results(results, round_num)

            # Log per-round winner summary
            successful = [r for r in results if r.get("success")]
            if successful:
                best = min(successful, key=lambda r: r.get("time_ms", float("inf")))
                self.logger.info(
                    f"Round {round_num} best: worker {best['worker_id']} at {best['time_ms']:.4f} ms"
                )
            else:
                self.logger.info(f"Round {round_num}: no successful workers")

            # 4. Check termination
            if self.strategy.should_terminate(round_num, max_rounds):
                self.logger.info("Strategy signaled termination")
                break

        # Return best result
        best = self.strategy.get_best_program()

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("=" * 80)

        if best:
            self.logger.info(f"Best time: {best.metrics.time_ms:.4f}ms")
            if initial_kernel_time != float("inf") and best.metrics.time_ms > 0:
                speedup = initial_kernel_time / best.metrics.time_ms
                self.logger.info(f"Speedup vs initial kernel: {speedup:.2f}x")
            if pytorch_baseline != float("inf") and best.metrics.time_ms > 0:
                speedup_pt = pytorch_baseline / best.metrics.time_ms
                self.logger.info(f"Speedup vs PyTorch eager: {speedup_pt:.2f}x")

        return {
            "success": best is not None and best.metrics.time_ms != float("inf"),
            "kernel_code": best.kernel_code if best else None,
            "best_time_ms": best.metrics.time_ms if best else float("inf"),
            "total_rounds": round_num,
            "pytorch_baseline_ms": pytorch_baseline,
            "pytorch_compile_ms": pytorch_compile_time,
            "initial_kernel_time_ms": initial_kernel_time,
            "top_kernels": [
                {
                    "kernel_code": p.kernel_code,
                    "time_ms": p.metrics.time_ms,
                    "generation": p.generation,
                    "program_id": p.program_id,
                }
                for p in self.database.get_top_k(5)
            ],
        }

    # ------------------------------------------------------------------
    # Thin delegates to platform components
    # ------------------------------------------------------------------

    def _verify_initial_kernel(
        self,
        initial_kernel: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        """Verify the initial kernel passes correctness before optimization."""
        return self.verifier.verify(initial_kernel, problem_file, test_code)

    def _benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark the eager reference implementation."""
        return self.benchmarker.benchmark_reference(problem_file)

    def _benchmark_pytorch_compile(self, problem_file: Path) -> float:
        """Benchmark the compiler-optimized reference."""
        return self.benchmarker.benchmark_reference_compiled(problem_file)

    def _benchmark_initial_kernel(
        self, initial_kernel: str, problem_file: Path
    ) -> float:
        """Benchmark the initial kernel before optimization begins."""
        return self.benchmarker.benchmark_kernel(initial_kernel, problem_file)

    def _run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: str,
        pytorch_baseline: float,
    ) -> list[dict[str, Any]]:
        """Spawn workers for each candidate and collect results."""
        results = self.worker_runner.run_workers(
            candidates=candidates,
            round_num=round_num,
            problem_file=problem_file,
            test_code=test_code,
            pytorch_baseline=pytorch_baseline,
            shared_history=(
                self.shared_history[-self.history_size :]
                if self.shared_history
                else []
            ),
            shared_reflexions=(
                self.shared_reflexions[-self.history_size :]
                if self.shared_reflexions
                else []
            ),
        )

        # Collect history and reflexions from worker results
        for r in results:
            if r.get("attempt"):
                self.shared_history.append(r["attempt"])
            if r.get("reflexion"):
                self.shared_reflexions.append(r["reflexion"])

        # Log errors from failed workers
        for r in results:
            if not r.get("success") and r.get("error"):
                self.logger.error(
                    f"Worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    self.logger.debug(f"Traceback:\n{r.get('traceback')}")

        return results
