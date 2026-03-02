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

"""CHECK_REFINE_ATTEMPTS state — check if refinement budget remains."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class CheckAttempts(State):
    """Check if refinement attempts remain.

    Transitions:
        attempts remaining -> RefineKernel
        exhausted          -> RecordFailure
    """

    def execute(self, ctx: WorkerContext) -> str:
        ctx.refinement_attempts += 1

        if ctx.refinement_attempts > ctx.max_refine_attempts:
            error_output = (
                ctx.test_stderr if ctx.test_stderr.strip() else ctx.test_stdout
            )
            ctx.error_feedback = (
                f"Verification failed after {ctx.max_refine_attempts} "
                f"refinement attempts:\n{error_output[:2000]}"
            )
            ctx.logger.warning(
                f"[{ctx.round_num}] Verification failed after "
                f"{ctx.max_refine_attempts} refinement attempts"
            )
            return "RecordFailure"

        ctx.logger.info(
            f"[{ctx.round_num}] Refinement attempt "
            f"{ctx.refinement_attempts}/{ctx.max_refine_attempts}..."
        )
        return "RefineKernel"
