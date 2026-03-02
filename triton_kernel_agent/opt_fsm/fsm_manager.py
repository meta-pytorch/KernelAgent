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

"""FSM-based Optimization Manager — drop-in replacement for OptimizationManager."""

from __future__ import annotations

import logging
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import FSMEngine
from triton_kernel_agent.opt_fsm.states.benchmark_baselines import BenchmarkBaselines
from triton_kernel_agent.opt_fsm.states.check_termination import CheckTermination
from triton_kernel_agent.opt_fsm.states.finalize import Finalize
from triton_kernel_agent.opt_fsm.states.run_workers import RunWorkers
from triton_kernel_agent.opt_fsm.states.select_candidates import SelectCandidates
from triton_kernel_agent.opt_fsm.states.update_strategy import UpdateStrategy
from triton_kernel_agent.opt_fsm.states.verify_initial import VerifyInitialKernel
from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.greedy import (
    GreedyStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)


class FSMOptimizationManager:
    """FSM-based optimization manager — drop-in replacement for OptimizationManager.

    Accepts the same constructor arguments and run_optimization() returns the
    identical result dict. Internally, the optimization flow is modeled as an
    explicit Finite State Machine.
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
        self.max_rounds = max_rounds
        self.log_dir = (
            Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="opt_"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs

        self.logger = self._setup_logging()

        db_path = (
            Path(database_path)
            if database_path
            else self.log_dir / "program_database.json"
        )
        self.database = JSONProgramDatabase(db_path)

        self.strategy = self._create_strategy(
            strategy, strategy_config or {}, num_workers
        )

        if num_workers != self.strategy.num_workers_needed:
            raise ValueError(
                f"Strategy '{strategy}' requires {self.strategy.num_workers_needed} "
                f"workers, got {num_workers}. Adjust num_workers or strategy_config."
            )

        self.num_workers = num_workers
        self.benchmark_lock = mp.Lock()
        self.profiling_semaphore = mp.Semaphore(1)

        self.logger.info(
            f"FSMOptimizationManager initialized: strategy={strategy}, "
            f"workers={num_workers}"
        )

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("FSMOptimizationManager")
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
            raise ValueError(
                f"Unknown strategy: {name}. Use 'beam_search' or 'greedy'"
            )

    def run_optimization(
        self,
        initial_kernel: str,
        problem_file: Path | str,
        test_code: str,
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization using the FSM.

        Same API as OptimizationManager.run_optimization(). Returns an
        identical result dict.
        """
        max_rounds = max_rounds or self.max_rounds
        problem_file = Path(problem_file)

        # Initialize strategy
        initial_entry = ProgramEntry(
            program_id="initial",
            kernel_code=initial_kernel,
            metrics=ProgramMetrics(time_ms=float("inf")),
            problem_id=str(problem_file),
        )
        self.strategy.initialize(initial_entry)

        # Build context
        ctx = OptimizationContext(
            initial_kernel=initial_kernel,
            problem_file=problem_file,
            test_code=test_code,
            max_rounds=max_rounds,
            strategy=self.strategy,
            database=self.database,
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            bottleneck_override=self.bottleneck_override,
            worker_kwargs=self.worker_kwargs,
            num_workers=self.num_workers,
        )

        # Build and run the outer FSM
        fsm = FSMEngine(logger=self.logger)
        fsm.add_state(VerifyInitialKernel())
        fsm.add_state(BenchmarkBaselines())
        fsm.add_state(SelectCandidates())
        fsm.add_state(RunWorkers())
        fsm.add_state(UpdateStrategy())
        fsm.add_state(CheckTermination())
        fsm.add_state(Finalize())

        fsm.run("VerifyInitialKernel", ctx)

        return ctx.result
