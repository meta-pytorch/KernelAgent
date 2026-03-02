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

"""RUN_TESTS state — run test suite against the optimized kernel."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class RunTests(State):
    """Run the test suite against the current optimized kernel.

    Writes kernel and test files, then runs the test subprocess.

    Transitions:
        pass -> BenchmarkNew
        fail -> CheckAttempts
    """

    def execute(self, ctx: WorkerContext) -> str:
        rn = ctx.round_num
        vw = ctx.verification_worker

        # Write kernel and test files
        vw._write_files(ctx.optimized_kernel, ctx.test_code)

        # Run single verification pass
        success, stdout, stderr, violation = vw._single_verification_pass(
            ctx.optimized_kernel
        )

        ctx.test_stdout = stdout
        ctx.test_stderr = stderr

        if violation:
            ctx.violation_message = violation
            ctx.error_feedback = violation
            return "RecordFailure"

        if success:
            ctx.logger.info(f"[{rn}] Correctness check passed")
            ctx.error_feedback = ""
            return "BenchmarkNew"

        ctx.logger.warning(f"[{rn}] Test failed")
        return "CheckAttempts"
