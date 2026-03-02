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

"""UPDATE_STRATEGY state — feed worker results to strategy and collect history."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class UpdateStrategy(State):
    """Update strategy with worker results, collect shared history.

    Transitions:
        always -> CheckTermination
    """

    def execute(self, ctx: OptimizationContext) -> str:
        results = ctx.worker_results

        # Update strategy
        ctx.strategy.update_with_results(results, ctx.round_num)

        # Log per-round winner
        successful = [r for r in results if r.get("success")]
        if successful:
            best = min(successful, key=lambda r: r.get("time_ms", float("inf")))
            ctx.logger.info(
                f"Round {ctx.round_num} best: worker {best['worker_id']} "
                f"at {best['time_ms']:.4f} ms"
            )
        else:
            ctx.logger.info(f"Round {ctx.round_num}: no successful workers")

        # Collect history and reflexions from worker results
        for r in results:
            if r.get("attempt"):
                ctx.shared_history.append(r["attempt"])
            if r.get("reflexion"):
                ctx.shared_reflexions.append(r["reflexion"])

        # Log errors from failed workers
        for r in results:
            if not r.get("success") and r.get("error"):
                ctx.logger.error(
                    f"Worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    ctx.logger.debug(f"Traceback:\n{r.get('traceback')}")

        return "CheckTermination"
