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

"""RECORD_FAILURE state — record a failed generation or verification attempt."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class RecordFailure(State):
    """Record a failed optimization attempt.

    Sets the current_attempt's error fields so GenerateReflexion can process it.

    Transitions:
        always -> GenerateReflexion
    """

    def execute(self, ctx: WorkerContext) -> str:
        if ctx.current_attempt is not None:
            ctx.current_attempt.passed_verification = False
            ctx.current_attempt.error_message = ctx.error_feedback

        return "GenerateReflexion"
