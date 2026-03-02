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

"""CHECK_TERMINATION state — check if optimization should stop."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class CheckTermination(State):
    """Check if strategy signals termination.

    Transitions:
        terminate -> Finalize
        continue  -> SelectCandidates
    """

    def execute(self, ctx: OptimizationContext) -> str:
        if ctx.strategy.should_terminate(ctx.round_num, ctx.max_rounds):
            ctx.logger.info("Strategy signaled termination")
            return "Finalize"
        return "SelectCandidates"
