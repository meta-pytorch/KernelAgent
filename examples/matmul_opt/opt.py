#!/usr/bin/env python3
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


"""Example demonstrating hardware-guided kernel optimization using OptimizationWorker.


This script shows how to use the OptimizationWorker to optimize a Triton kernel
using NCU profiling, bottleneck analysis, and LLM-guided optimization.
"""

from datetime import datetime
from pathlib import Path
from triton_kernel_agent.opt_worker import OptimizationWorker

import os

os.environ["LLM_RELAY_TIMEOUT_S"] = "600"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")


def example_optimization():
    """
    Example demonstrating hardware-guided kernel optimization.


    Optimization Strategy:
    - Uses NCU profiling to identify performance bottlenecks
    - Judge LLM analyzes metrics to identify primary and secondary bottlenecks
    - Optimizer LLM generates improved kernel variants
    - VerificationWorker ensures correctness
    - Iterates until convergence or max rounds reached
    """

    example_dir = Path(__file__).parent
    session_dir = (
        Path(example_dir).parent.parent
        / "triton_kernel_logs"
        / example_dir.name
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    # copy example_dir to session_dir
    for src_path in example_dir.iterdir():
        if src_path.is_file():
            dest_path = session_dir / src_path.name
            dest_path.write_bytes(src_path.read_bytes())

    # copy example files to session dir
    kernel_file = session_dir / "matmul.py"
    problem_file = session_dir / "problem.py"
    test_file = session_dir / "test.py"

    # Create log/work directories
    workdir = session_dir / "opt_workdir"
    log_dir = session_dir / "opt_logs"

    workdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    kernel_code = kernel_file.read_text()
    test_code = test_file.read_text()

    print("=" * 80)
    print("Hardware-Guided Kernel Optimization")
    print("=" * 80)
    print(f"Kernel file: {kernel_file}")
    print(f"Problem file: {problem_file}")
    print(f"Work directory: {workdir}")
    print(f"Log directory: {log_dir}")
    print(f"model: {OPENAI_MODEL}")
    print()

    # Initialize OptimizationWorker
    #
    # Key Parameters:
    # - max_rounds: Maximum optimization iterations
    # - openai_model: LLM model for optimization
    # - enable_ncu_profiling: Enable NCU profiling for bottleneck analysis
    # - gpu_name: GPU name for hardware specs (None for auto-detect)
    # - bottleneck_id: Which bottleneck to focus on (1=primary, 2=secondary)
    # - divergence_threshold: Max % worse before reverting to best kernel
    worker = OptimizationWorker(
        worker_id=0,
        workdir=workdir,
        log_dir=log_dir,
        max_rounds=5,
        openai_model=OPENAI_MODEL,
        high_reasoning_effort=True,
        # Hardware-aware parameters
        gpu_name=None,  # Auto-detect GPU
        enable_ncu_profiling=True,
        bottleneck_id=1,  # Focus on primary bottleneck
        # Benchmarking parameters
        benchmark_warmup=25,
        benchmark_repeat=100,
        # Performance safeguards
        divergence_threshold=50.0,  # Revert if 50% worse
        target_platform="cuda",
        use_triton_mpp=True,
    )

    # Run optimization
    print("Starting optimization...")
    print()

    success, best_kernel, metrics = worker.optimize_kernel(
        kernel_code=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
    )

    if success:
        print()
        print("=" * 80)
        print("OPTIMIZATION SUCCESSFUL!")
        print("=" * 80)
        print(f"Baseline time: {metrics.get('baseline_time_ms', 'N/A'):.4f} ms")
        print(f"Best time: {metrics.get('best_time_ms', 'N/A'):.4f} ms")
        print(f"Speedup: {metrics.get('speedup', 'N/A'):.2f}x")
        print(f"Optimization rounds: {metrics.get('rounds', 'N/A')}")

        # Save final optimized kernel
        final_kernel_path = session_dir / "final_kernel_optimized.py"
        final_kernel_path.write_text(best_kernel)
        print(f"\nSaved optimized kernel to: {final_kernel_path}")

    else:
        print()
        print("=" * 80)
        print("OPTIMIZATION FAILED")
        print("=" * 80)
        print("Check logs for details:", log_dir)


if __name__ == "__main__":
    example_optimization()
