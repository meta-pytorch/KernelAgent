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

"""CHECK_DISALLOWED_PATTERNS state — scan kernel for forbidden PyTorch usage."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class CheckPatterns(State):
    """Scan optimized kernel for disallowed PyTorch patterns.

    Transitions:
        violation -> RecordFailure
        clean     -> RunTests
    """

    def execute(self, ctx: WorkerContext) -> str:
        violation = ctx.verification_worker._detect_pytorch_compute(
            ctx.optimized_kernel
        )
        if violation:
            message = f"Disallowed PyTorch usage detected: {violation}"
            ctx.logger.error(f"[{ctx.round_num}] {message}")
            ctx.violation_message = message
            ctx.error_feedback = message
            return "RecordFailure"

        ctx.violation_message = None
        return "RunTests"
