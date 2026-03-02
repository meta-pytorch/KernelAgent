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

"""CHECK_EARLY_STOP state — roofline-based early termination."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class CheckEarlyStop(State):
    """Check if optimization should stop because the kernel is at roofline.

    Only entered when the optimized kernel IS the best runtime kernel
    AND the roofline check indicates at_roofline.

    Transitions:
        stop     -> WorkerFinalize (with early_stop_reason set)
        continue -> WorkerFinalize
    """

    def execute(self, ctx: WorkerContext) -> str:
        roofline_check = ctx.new_kernel_metrics.get("roofline_result")

        should_stop, stop_reason = ctx.roofline_analyzer.should_stop(roofline_check)
        if should_stop and ctx.roofline_analyzer.config.early_stop:
            ctx.logger.info(
                f"[{ctx.round_num}] Early termination: {stop_reason}"
            )
            ctx.early_stop_reason = stop_reason

        return "WorkerFinalize"
