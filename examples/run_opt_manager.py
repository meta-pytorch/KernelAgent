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

"""Example demonstrating multi-strategy kernel optimization using OptimizationManager.

This script shows how to use the OptimizationManager to optimize a Triton kernel
using different search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization with early termination

The OptimizationManager orchestrates parallel workers and persists optimization
history to a JSON database for analysis and resumption.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from triton_kernel_agent.opt_manager import OptimizationManager
from triton_kernel_agent.platform.noop import (
    NoOpBenchmarker,
    NoOpVerifier,
    NoOpWorkerRunner,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def run_beam_search_optimization(
    kernel_code: str,
    problem_file: Path,
    test_code: str,
    log_dir: Path,
    max_rounds: int = 5,
) -> dict:
    """
    Run optimization using beam search strategy.

    Beam search maintains top-N kernels and explores M bottleneck directions
    for each, giving N×M parallel workers per round.

    Args:
        kernel_code: Initial kernel source code
        problem_file: Path to problem.py
        test_code: Test code for verification
        log_dir: Directory for logs and artifacts
        max_rounds: Maximum optimization rounds

    Returns:
        Optimization result dict
    """
    print("\n" + "=" * 80)
    print("BEAM SEARCH OPTIMIZATION")
    print("=" * 80)

    config_path = Path(__file__).parent / "configs" / "beam_search.yaml"
    manager = OptimizationManager(
        max_rounds=max_rounds,
        log_dir=log_dir / "beam_search",
        database_path=log_dir / "beam_search" / "program_db.json",
        config=str(config_path),
    )

    return manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
    )


def run_greedy_optimization(
    kernel_code: str,
    problem_file: Path,
    test_code: str,
    log_dir: Path,
    max_rounds: int = 10,
) -> dict:
    """
    Run optimization using greedy strategy.

    Greedy strategy always optimizes from the current best kernel
    with a single worker. Terminates early if no improvement for
    several consecutive rounds.

    Args:
        kernel_code: Initial kernel source code
        problem_file: Path to problem.py
        test_code: Test code for verification
        log_dir: Directory for logs and artifacts
        max_rounds: Maximum optimization rounds

    Returns:
        Optimization result dict
    """
    print("\n" + "=" * 80)
    print("GREEDY OPTIMIZATION")
    print("=" * 80)

    config_path = Path(__file__).parent / "configs" / "greedy.yaml"
    manager = OptimizationManager(
        max_rounds=max_rounds,
        log_dir=log_dir / "greedy",
        database_path=log_dir / "greedy" / "program_db.json",
        config=str(config_path),
    )

    return manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
    )


def run_noop_optimization(
    kernel_code: str,
    problem_file: Path,
    test_code: str,
    log_dir: Path,
    max_rounds: int = 2,
) -> dict:
    """
    Run optimization with no-op platform components.

    All verification, benchmarking, and worker steps are replaced with
    no-op stubs that print when called and return neutral defaults.
    The result is a pass-through: the initial kernel is returned unchanged.

    Args:
        kernel_code: Initial kernel source code
        problem_file: Path to problem.py
        test_code: Test code for verification
        log_dir: Directory for logs and artifacts
        max_rounds: Maximum optimization rounds

    Returns:
        Optimization result dict (kernel_code == initial kernel)
    """
    print("\n" + "=" * 80)
    print("NO-OP OPTIMIZATION (platform pass-through)")
    print("=" * 80)

    manager = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        max_rounds=max_rounds,
        log_dir=log_dir / "noop",
        database_path=log_dir / "noop" / "program_db.json",
        strategy_config={"max_no_improvement": 1},
        verifier=NoOpVerifier(),
        benchmarker=NoOpBenchmarker(),
        worker_runner=NoOpWorkerRunner(),
    )

    return manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
    )


def print_result(result: dict, strategy_name: str, kernel_dir: Path) -> None:
    """Print optimization result and save the best kernel."""
    if result["success"]:
        print()
        print("=" * 80)
        print(f"{strategy_name} OPTIMIZATION SUCCESSFUL!")
        print("=" * 80)
        print(f"Best time: {result['best_time_ms']:.4f} ms")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Top kernels found: {len(result['top_kernels'])}")

        # Show top kernels summary
        if result["top_kernels"]:
            print("\nTop kernels performance:")
            for i, kernel in enumerate(result["top_kernels"][:3], 1):
                print(f"  {i}. {kernel['time_ms']:.4f}ms (gen {kernel['generation']})")

        # Save the best kernel
        if result["kernel_code"]:
            output_file = kernel_dir / f"optimized_kernel_{strategy_name.lower()}.py"
            output_file.write_text(result["kernel_code"])
            print(f"\nSaved optimized kernel to: {output_file}")
    else:
        print()
        print("=" * 80)
        print(f"{strategy_name} OPTIMIZATION FAILED")
        print("=" * 80)
        print("Check logs for details")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Optimize Triton kernels using different search strategies"
    )
    parser.add_argument(
        "--strategy",
        choices=["beam_search", "greedy", "noop", "all"],
        default="beam_search",
        help="Optimization strategy to use (default: beam_search). "
        "'noop' runs with no-op platform components (pass-through).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum optimization rounds (default: 5)",
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Directory containing kernel.py, problem.py, and test.py",
    )

    args = parser.parse_args()

    # Setup paths
    kernel_dir = args.kernel_dir.resolve()
    kernel_file = kernel_dir / "input.py"  # Use input.py as initial
    problem_file = kernel_dir / "problem.py"
    test_file = kernel_dir / "test.py"
    log_dir = kernel_dir / "opt_manager_logs"

    if not problem_file.exists():
        print(f"ERROR: problem.py not found in {kernel_dir}")
        sys.exit(1)

    if not test_file.exists():
        print(f"ERROR: test.py not found in {kernel_dir}")
        sys.exit(1)

    # Read source files
    kernel_code = kernel_file.read_text()
    test_code = test_file.read_text()

    # Print header
    print("=" * 80)
    print("OptimizationManager - Multi-Strategy Kernel Optimization")
    print("=" * 80)
    print(f"Kernel file: {kernel_file}")
    print(f"Problem file: {problem_file}")
    print(f"Strategy: {args.strategy}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Log directory: {log_dir}")

    # Run selected strategy
    if args.strategy == "beam_search":
        result = run_beam_search_optimization(
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            args.max_rounds,
        )
        print_result(result, "BEAM_SEARCH", kernel_dir)

    elif args.strategy == "greedy":
        result = run_greedy_optimization(
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            args.max_rounds,
        )
        print_result(result, "GREEDY", kernel_dir)

    elif args.strategy == "noop":
        result = run_noop_optimization(
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            args.max_rounds,
        )
        print_result(result, "NOOP", kernel_dir)

    elif args.strategy == "all":
        # Run all strategies and compare
        results = {}

        results["beam_search"] = run_beam_search_optimization(
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            args.max_rounds,
        )
        print_result(results["beam_search"], "BEAM_SEARCH", kernel_dir)

        results["greedy"] = run_greedy_optimization(
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            args.max_rounds,
        )
        print_result(results["greedy"], "GREEDY", kernel_dir)

        # Compare results
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        for name, result in results.items():
            status = "✓" if result["success"] else "✗"
            time_str = f"{result['best_time_ms']:.4f}ms" if result["success"] else "N/A"
            print(f"  {status} {name:15} - Best: {time_str}")


if __name__ == "__main__":
    main()
