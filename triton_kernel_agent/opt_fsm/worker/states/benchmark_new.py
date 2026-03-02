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

"""BENCHMARK_NEW_KERNEL state — benchmark the verified optimized kernel."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.engine import State
from triton_kernel_agent.opt_fsm.worker.context import WorkerContext


class BenchmarkNew(State):
    """Benchmark the newly verified optimized kernel.

    Transitions:
        always -> ProfileSol
    """

    def execute(self, ctx: WorkerContext) -> str:
        rn = ctx.round_num

        kernel_file_round = ctx.artifact_dir / f"kernel_round_{rn}.py"
        kernel_file_round.write_text(ctx.optimized_kernel)

        bench_results = ctx.benchmarker.benchmark_kernel(
            kernel_file_round, ctx.problem_file
        )
        ctx.new_time = bench_results["time_ms"]

        return "ProfileSol"
