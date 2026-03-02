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

"""VERIFY_INITIAL_KERNEL state — verify the starting kernel is correct."""

from __future__ import annotations

import shutil

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class VerifyInitialKernel(State):
    """Verify that the initial kernel passes correctness tests.

    Transitions:
        pass -> BenchmarkBaselines
        fail -> Finalize (with error result)
    """

    def execute(self, ctx: OptimizationContext) -> str:
        from triton_kernel_agent.worker import VerificationWorker

        ctx.logger.info("=" * 80)
        ctx.logger.info("STARTING OPTIMIZATION")
        ctx.logger.info("=" * 80)

        verify_dir = ctx.log_dir / "initial_verify"
        verify_dir.mkdir(parents=True, exist_ok=True)

        # Copy problem file so the test can import it
        shutil.copy(ctx.problem_file, verify_dir / "problem.py")

        worker = VerificationWorker(
            worker_id=-1,
            workdir=verify_dir,
            log_dir=verify_dir,
        )

        success, _, error = worker.verify_with_refinement(
            kernel_code=ctx.initial_kernel,
            test_code=ctx.test_code,
            problem_description=ctx.problem_file.read_text(),
            max_refine_attempts=0,
        )

        if not success:
            ctx.logger.error(
                f"Initial kernel failed correctness verification: {error[:200]}"
            )
            ctx.result = {
                "success": False,
                "kernel_code": None,
                "best_time_ms": float("inf"),
                "total_rounds": 0,
                "top_kernels": [],
                "error": "Initial kernel failed correctness verification",
            }
            return "Finalize"

        ctx.logger.info("Initial kernel passed correctness verification")
        ctx.initial_verification_passed = True
        return "BenchmarkBaselines"
