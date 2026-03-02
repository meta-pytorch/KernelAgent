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

"""SELECT_CANDIDATES state — ask strategy for candidates this round."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class SelectCandidates(State):
    """Increment round counter and ask the strategy for candidates.

    Transitions:
        candidates found -> RunWorkers
        no candidates    -> Finalize
    """

    def execute(self, ctx: OptimizationContext) -> str:
        ctx.round_num += 1

        ctx.logger.info("")
        ctx.logger.info(
            f"{'=' * 20} ROUND {ctx.round_num}/{ctx.max_rounds} {'=' * 20}"
        )

        ctx.candidates = ctx.strategy.select_candidates(ctx.round_num)
        if not ctx.candidates:
            ctx.logger.warning("No candidates to explore, terminating")
            return "Finalize"

        return "RunWorkers"
