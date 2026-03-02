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

"""RUN_WORKERS state — spawn parallel worker processes."""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class RunWorkers(State):
    """Spawn parallel worker processes for each candidate.

    Each worker process runs the inner (worker) FSM. This state replicates
    the exact parallelism behavior of OptimizationManager._run_workers().

    Transitions:
        always -> UpdateStrategy
    """

    def execute(self, ctx: OptimizationContext) -> str:
        from triton_kernel_agent.opt_fsm.worker.worker_fsm import (
            worker_process_entry,
        )

        result_queue: mp.Queue = mp.Queue()
        workers: list[mp.Process] = []

        for i, candidate in enumerate(ctx.candidates):
            workdir = ctx.log_dir / "workers" / f"w{i}" / f"r{ctx.round_num}"
            workdir.mkdir(parents=True, exist_ok=True)

            args = (
                i,  # worker_id
                candidate["parent"].kernel_code,
                candidate["parent"].metrics.time_ms,
                candidate["parent"].program_id,
                ctx.problem_file,
                ctx.test_code,
                workdir,
                workdir / "logs",
                result_queue,
                ctx.benchmark_lock,
                ctx.profiling_semaphore,
                ctx.pytorch_baseline_ms,
                candidate["bottleneck_id"],
                ctx.openai_model,
                ctx.high_reasoning_effort,
                ctx.bottleneck_override,
                ctx.worker_kwargs,
                (
                    ctx.shared_history[-ctx.history_size :]
                    if ctx.shared_history
                    else []
                ),
                (
                    ctx.shared_reflexions[-ctx.history_size :]
                    if ctx.shared_reflexions
                    else []
                ),
            )

            p = mp.Process(target=worker_process_entry, args=args)
            p.start()
            workers.append(p)

        # Wait with shared deadline (30 min)
        worker_timeout = 1800
        deadline = time.time() + worker_timeout
        for w in workers:
            remaining = max(0, deadline - time.time())
            w.join(timeout=remaining)
            if w.is_alive():
                ctx.logger.warning(f"Worker {w.pid} timed out, terminating")
                w.terminate()
                w.join(timeout=5)
                if w.is_alive():
                    ctx.logger.warning(f"Worker {w.pid} still alive, killing")
                    w.kill()
                    w.join(timeout=2)
            w.close()

        # Collect results
        results: list[dict[str, Any]] = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        result_queue.close()
        result_queue.join_thread()

        ctx.worker_results = results

        successful = sum(1 for r in results if r.get("success"))
        ctx.logger.info(
            f"Round {ctx.round_num}: {successful}/{len(ctx.candidates)} workers succeeded "
            f"({len(results)} results received)"
        )

        return "UpdateStrategy"
