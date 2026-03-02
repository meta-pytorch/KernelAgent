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

"""BENCHMARK_BASELINES state — benchmark PyTorch eager, compile, and initial kernel."""

from __future__ import annotations

from triton_kernel_agent.opt_fsm.context import OptimizationContext
from triton_kernel_agent.opt_fsm.engine import State


class BenchmarkBaselines(State):
    """Benchmark all baselines before optimization begins.

    Benchmarks:
    1. PyTorch eager baseline
    2. torch.compile baseline
    3. Initial kernel

    Transitions:
        always -> SelectCandidates
    """

    def execute(self, ctx: OptimizationContext) -> str:
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        artifacts_dir = ctx.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        benchmarker = Benchmark(
            logger=ctx.logger,
            artifacts_dir=artifacts_dir,
            benchmark_lock=ctx.benchmark_lock,
            worker_id=-1,
        )

        # 1. PyTorch eager baseline
        result = benchmarker.benchmark_pytorch(ctx.problem_file)
        ctx.pytorch_baseline_ms = result.get("time_ms", float("inf"))
        if ctx.pytorch_baseline_ms != float("inf"):
            ctx.logger.info(f"PyTorch baseline: {ctx.pytorch_baseline_ms:.4f}ms")

        # 2. torch.compile baseline
        result = benchmarker.benchmark_pytorch_compile(ctx.problem_file)
        ctx.pytorch_compile_ms = result.get("time_ms", float("inf"))
        if ctx.pytorch_compile_ms != float("inf"):
            ctx.logger.info(
                f"PyTorch compile baseline: {ctx.pytorch_compile_ms:.4f}ms"
            )

        # 3. Initial kernel
        kernel_file = artifacts_dir / "initial_kernel.py"
        kernel_file.write_text(ctx.initial_kernel, encoding="utf-8")

        result = benchmarker.benchmark_kernel(kernel_file, ctx.problem_file)
        ctx.initial_kernel_time_ms = result.get("time_ms", float("inf"))
        if ctx.initial_kernel_time_ms != float("inf"):
            ctx.logger.info(
                f"Initial kernel time: {ctx.initial_kernel_time_ms:.4f}ms"
            )

        return "SelectCandidates"
