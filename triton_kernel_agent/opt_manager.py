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
import shutil
import tempfile
import time
import traceback
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


class OptimizationManager:
    """Manages parallel kernel optimization with pluggable strategies.

    Supports:
    - beam_search: Current default (top-N kernels × M bottlenecks)
    - greedy: Simple single-best optimization
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

        self.logger.info(
            f"OptimizationManager initialized: strategy={strategy}, workers={num_workers}"
        )

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

    def _benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark PyTorch baseline once before spawning workers.

        Args:
            problem_file: Path to problem.py

        Returns:
            PyTorch baseline time in ms
        """
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        benchmarker = Benchmark(
            logger=self.logger,
            artifacts_dir=artifacts_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=-1,
        )

        result = benchmarker.benchmark_pytorch(problem_file)
        pytorch_time = result.get("time_ms", float("inf"))

        if pytorch_time != float("inf"):
            self.logger.info(f"PyTorch baseline: {pytorch_time:.4f}ms")

        return pytorch_time

    def _verify_initial_kernel(
        self,
        initial_kernel: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        """Verify the initial kernel passes correctness before optimization.

        Args:
            initial_kernel: Kernel source code
            problem_file: Path to problem.py
            test_code: Test code for correctness verification

        Returns:
            True if the initial kernel passes verification
        """
        from triton_kernel_agent.worker import VerificationWorker

        verify_dir = self.log_dir / "initial_verify"
        verify_dir.mkdir(parents=True, exist_ok=True)

        # Copy problem file so the test can import it
        shutil.copy(problem_file, verify_dir / "problem.py")

        worker = VerificationWorker(
            worker_id=-1,
            workdir=verify_dir,
            log_dir=verify_dir,
        )

        success, _, error = worker.verify_with_refinement(
            kernel_code=initial_kernel,
            test_code=test_code,
            problem_description=problem_file.read_text(),
            max_refine_attempts=0,
        )

        if not success:
            self.logger.error(
                f"Initial kernel failed correctness verification: {error[:200]}"
            )
        else:
            self.logger.info("Initial kernel passed correctness verification")

        return success

    def _benchmark_initial_kernel(
        self, initial_kernel: str, problem_file: Path
    ) -> float:
        """Benchmark the initial kernel before optimization begins.

        Args:
            initial_kernel: Kernel source code
            problem_file: Path to problem.py

        Returns:
            Initial kernel time in ms
        """
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Write kernel to a temp file
        kernel_file = artifacts_dir / "initial_kernel.py"
        kernel_file.write_text(initial_kernel, encoding="utf-8")

        benchmarker = Benchmark(
            logger=self.logger,
            artifacts_dir=artifacts_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=-1,
        )

        result = benchmarker.benchmark_kernel(kernel_file, problem_file)
        kernel_time = result.get("time_ms", float("inf"))

        if kernel_time != float("inf"):
            self.logger.info(f"Initial kernel time: {kernel_time:.4f}ms")

        return kernel_time

    def _benchmark_pytorch_compile(self, problem_file: Path) -> float:
        """Benchmark torch.compile'd PyTorch baseline.

        Args:
            problem_file: Path to problem.py

        Returns:
            torch.compile baseline time in ms
        """
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        benchmarker = Benchmark(
            logger=self.logger,
            artifacts_dir=artifacts_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=-1,
        )

        result = benchmarker.benchmark_pytorch_compile(problem_file)
        compile_time = result.get("time_ms", float("inf"))

        if compile_time != float("inf"):
            self.logger.info(f"PyTorch compile baseline: {compile_time:.4f}ms")

        return compile_time

    def _run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: str,
        pytorch_baseline: float,
    ) -> list[dict[str, Any]]:
        """Spawn workers for each candidate and collect results.

        Args:
            candidates: List of candidate specs from strategy
            round_num: Current round number
            problem_file: Path to problem.py
            test_code: Test code for verification
            pytorch_baseline: PyTorch baseline time in ms

        Returns:
            List of result dicts from workers
        """
        result_queue = mp.Queue()
        workers = []

        for i, candidate in enumerate(candidates):
            workdir = self.log_dir / "workers" / f"w{i}" / f"r{round_num}"
            workdir.mkdir(parents=True, exist_ok=True)

            args = (
                i,  # worker_id
                candidate["parent"].kernel_code,
                candidate["parent"].metrics.time_ms,
                candidate["parent"].program_id,
                problem_file,
                test_code,
                workdir,
                workdir / "logs",
                result_queue,
                self.benchmark_lock,
                self.profiling_semaphore,
                pytorch_baseline,
                candidate["bottleneck_id"],
                self.openai_model,
                self.high_reasoning_effort,
                self.bottleneck_override,
                self.worker_kwargs,
                # Pass shared history (limited to history_size)
                (
                    self.shared_history[-self.history_size :]
                    if self.shared_history
                    else []
                ),
                (
                    self.shared_reflexions[-self.history_size :]
                    if self.shared_reflexions
                    else []
                ),
            )

            p = mp.Process(target=_worker_process, args=args)
            p.start()
            workers.append(p)

        # Wait for completion with timeout — use a shared deadline so all
        # workers get the full budget regardless of join order.  The per-worker
        # budget must be long enough for serialised NCU profiling across all
        # workers (each NCU ~2-3 min × num_workers × profiles_per_worker).
        worker_timeout = 1800  # 30 minutes
        deadline = time.time() + worker_timeout
        for w in workers:
            remaining = max(0, deadline - time.time())
            w.join(timeout=remaining)
            if w.is_alive():
                self.logger.warning(f"Worker {w.pid} timed out, terminating")
                w.terminate()
                w.join(timeout=5)
                if w.is_alive():
                    self.logger.warning(f"Worker {w.pid} still alive, killing")
                    w.kill()
                    w.join(timeout=2)
            w.close()

        # Collect results
        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        # Clean up queue resources to prevent thread hangs during GC
        result_queue.close()
        result_queue.join_thread()

        successful = sum(1 for r in results if r.get("success"))
        self.logger.info(
            f"Round {round_num}: {successful}/{len(candidates)} workers succeeded "
            f"({len(results)} results received)"
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


def _worker_process(
    worker_id: int,
    kernel_code: str,
    known_time: float,
    parent_id: str,
    problem_file: Path,
    test_code: str,
    workdir: Path,
    log_dir: Path,
    result_queue: mp.Queue,
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
    """Worker process function.

    This runs in a separate process to optimize a single kernel variant.

    Args:
        worker_id: Worker identifier
        kernel_code: Starting kernel code
        known_time: Known baseline time
        parent_id: ID of parent program
        problem_file: Path to problem.py
        test_code: Test code for verification
        workdir: Worker working directory
        log_dir: Worker log directory
        result_queue: Queue for results
        benchmark_lock: Shared benchmark lock
        profiling_semaphore: Semaphore to serialize NCU profiling
        pytorch_baseline: PyTorch baseline time
        bottleneck_id: Which bottleneck to explore
        openai_model: Model name
        high_reasoning_effort: High reasoning flag
        bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
        worker_kwargs: Additional worker kwargs
        prior_history: Shared history from previous rounds (serialized)
        prior_reflexions: Shared reflexions from previous rounds (serialized)
    """
    # Ensure correct path for worker process
    import sys

    kernel_agent_path = Path(__file__).parent.parent
    if str(kernel_agent_path) not in sys.path:
        sys.path.insert(0, str(kernel_agent_path))

    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker

        # Ensure directories exist
        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Copy problem file to workdir
        shutil.copy(problem_file, workdir / "problem.py")

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
            # Pass shared history
            prior_history=prior_history,
            prior_reflexions=prior_reflexions,
            **worker_kwargs,
        )

        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,  # Single step per round
        )

        # Get attempt and reflexion from worker for shared history
        attempt_data = metrics.get("last_attempt")
        reflexion_data = metrics.get("last_reflexion")

        result_queue.put(
            {
                "success": success,
                "worker_id": worker_id,
                "kernel_code": best_kernel,
                "time_ms": metrics.get("best_time_ms", float("inf")),
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
