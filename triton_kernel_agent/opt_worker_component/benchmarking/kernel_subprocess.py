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

"""
A task-agnostic, profiling-only benchmark script for Triton kernels.
This script ONLY benchmarks candidate kernels without correctness checks.
Assumes correctness has been verified upstream.

Design:
- Skips correctness verification (assumes already verified)
- Only runs candidate kernels
- Fast profiling for iterative optimization loops
- Uses shared utilities from timing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import shared utilities from timing module (avoid duplication)
from timing import (
    import_module,
    load_kernel_function,
    load_problem_interface,
    prepare_inputs,
)
from typing import Any, Callable, Dict, Tuple

import torch
import triton.testing as tt


def _run_once(
    fn: Callable, inputs: Tuple[torch.Tensor, ...], init_inputs: list, name: str
) -> torch.Tensor:
    """Run kernel once to get output shape/dtype info.

    Args:
        fn: Kernel function
        inputs: Input tensors
        init_inputs: Initialization inputs (e.g., features, eps)
        name: Name for logging

    Returns:
        Output tensor

    Raises:
        Exception if kernel fails to run
    """
    try:
        with torch.inference_mode():
            out = fn(*inputs, *init_inputs)
        return out
    except Exception as exc:
        raise RuntimeError(f"{name} failed to execute: {exc}") from exc


def benchmark(
    fn: Callable,
    inputs: Tuple[torch.Tensor, ...],
    init_inputs: list,
    name: str,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark a kernel function using triton.testing.do_bench.

    Args:
        fn: Kernel function to benchmark
        inputs: Input tensors
        init_inputs: Initialization inputs (e.g., features, eps)
        name: Name for logging
        warmup: Number of warmup iterations
        rep: Number of measurement iterations

    Returns:
        Mean latency in milliseconds
    """
    try:
        ms = tt.do_bench(
            lambda: fn(*inputs, *init_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="mean",
        )
        print(f"{name}: {ms:.4f} ms (mean over {rep} runs)")
        return ms
    except Exception as exc:
        print(f"❌ {name}: Benchmark failed: {exc}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Task-agnostic Triton kernel benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # File paths
    parser.add_argument(
        "--problem",
        type=Path,
        required=True,
        help="Path to problem file (must define Model and get_inputs)",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        required=True,
        help="Path to kernel file (must define kernel_function)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include PyTorch reference model in benchmark",
    )

    # Benchmark configuration
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)

    # Problem configuration
    parser.add_argument("--size", type=int, default=4096, help="Problem size N")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )

    # Output options
    parser.add_argument("--json", type=Path, help="Save results to JSON file")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    args = parser.parse_args()

    # Resolve paths
    args.problem = args.problem.resolve()
    args.kernel = args.kernel.resolve()

    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if not args.quiet:
        print("=" * 80)
        print("TRITON KERNEL PROFILING")
        print("=" * 80)
        print(f"Problem: {args.problem.name}")
        print(f"Size: {args.size}")
        print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
        print()

    # Import problem module using shared utility
    try:
        Model, get_inputs, get_init_inputs = load_problem_interface(args.problem)
    except Exception as exc:
        print(f"❌ Failed to import problem file: {exc}")
        sys.exit(1)

    # Check for optional benchmark config
    try:
        problem_mod = import_module(args.problem, "problem")
        get_benchmark_config = getattr(problem_mod, "get_benchmark_config", None)
    except Exception:
        get_benchmark_config = None

    # Override benchmark config if provided by problem
    if get_benchmark_config is not None:
        config = get_benchmark_config()
        args.warmup = config.get("warmup", args.warmup)
        args.repeat = config.get("repeat", args.repeat)
        if not args.quiet:
            cfg_msg = (
                f"Using problem-specific config: "
                f"warmup={args.warmup}, repeat={args.repeat}"
            )
            print(cfg_msg)

    # Generate inputs using shared utility
    try:
        inputs = prepare_inputs(get_inputs, device=device, dtype=dtype)

        # Get initialization inputs (e.g., features, eps for RMSNorm)
        init_inputs = []
        if get_init_inputs is not None:
            init_inputs = get_init_inputs()
            if not isinstance(init_inputs, (tuple, list)):
                init_inputs = [init_inputs]
    except Exception as exc:
        print(f"❌ Failed to generate inputs: {exc}")
        sys.exit(1)

    # Create reference model only if baseline is requested
    model = None
    if args.baseline:
        try:
            # Initialize model with init_inputs if provided
            if init_inputs:
                model = Model(*init_inputs).to(device=device, dtype=dtype)
            else:
                model = Model().to(device=device, dtype=dtype)
            model.eval()
            # Run once to get output shape
            out = _run_once(model, inputs, [], "Reference")
            if not args.quiet:
                print(f"Reference output shape: {out.shape}, dtype: {out.dtype}")
                print()
        except Exception as exc:
            print(f"❌ Failed to create reference model: {exc}")
            sys.exit(1)

    # Results tracking
    results: Dict[str, Any] = {
        "problem": str(args.problem),
        "size": args.size,
        "device": str(device),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "kernels": {},
    }

    baseline_time = None

    # Benchmark PyTorch baseline if requested
    if args.baseline and model is not None:
        if not args.quiet:
            print("1. PyTorch Reference")
        baseline_time = benchmark(
            model, inputs, [], "PyTorch", args.warmup, args.repeat
        )
        results["kernels"]["pytorch_reference"] = {
            "time_ms": baseline_time,
            "speedup": 1.0,
        }
        if not args.quiet:
            print()

    # Benchmark candidate kernel
    kernel_name = args.kernel.stem

    if not args.quiet:
        idx = 2 if args.baseline else 1
        print(f"{idx}. Candidate: {kernel_name}")

    # Import kernel using shared utility
    try:
        kernel_function = load_kernel_function(args.kernel)
    except Exception as exc:
        print(f"❌ Failed to import {kernel_name}: {exc}")
        results["kernels"][kernel_name] = {
            "time_ms": float("inf"),
            "error": str(exc),
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
        sys.exit(1)

    # Check if kernel expects weight/bias parameters (e.g., Conv, Linear)
    # If so, extract them from a Model instance
    import inspect

    needs_model = False
    try:
        sig = inspect.signature(kernel_function)
        params = list(sig.parameters.keys())
        # Check if kernel expects 'weight' parameter (common for Conv, Linear, etc.)
        if "weight" in params:
            needs_model = True
    except Exception:
        pass

    # Prepare kernel arguments
    kernel_args = inputs
    kernel_init_args = init_inputs

    if needs_model and Model is not None:
        try:
            # Initialize model to extract weight and bias
            if init_inputs:
                extract_model = Model(*init_inputs).to(device=device, dtype=dtype)
            else:
                extract_model = Model().to(device=device, dtype=dtype)

            # Extract weight and bias from model layer
            # Check various possible attribute names
            weight = None
            bias = None
            layer = None
            for name, module in extract_model.named_modules():
                if isinstance(
                    module,
                    (
                        torch.nn.Conv1d,
                        torch.nn.Conv2d,
                        torch.nn.Conv3d,
                        torch.nn.Linear,
                    ),
                ):
                    if hasattr(module, "weight") and module.weight is not None:
                        layer = module
                        weight = module.weight
                        bias = getattr(module, "bias", None)
                        break

            if weight is not None and layer is not None:
                # Build kwargs for kernel_function
                kernel_kwargs = {}

                # Add conv/linear-specific parameters if they exist
                if hasattr(layer, "stride"):
                    stride = (
                        layer.stride[0]
                        if isinstance(layer.stride, (tuple, list))
                        else layer.stride
                    )
                    kernel_kwargs["stride"] = stride
                if hasattr(layer, "padding"):
                    padding = (
                        layer.padding[0]
                        if isinstance(layer.padding, (tuple, list))
                        else layer.padding
                    )
                    kernel_kwargs["padding"] = padding
                if hasattr(layer, "dilation"):
                    dilation = (
                        layer.dilation[0]
                        if isinstance(layer.dilation, (tuple, list))
                        else layer.dilation
                    )
                    kernel_kwargs["dilation"] = dilation
                if hasattr(layer, "groups"):
                    kernel_kwargs["groups"] = layer.groups

                # Capture original kernel function to avoid recursion
                original_kernel_function = kernel_function

                # Prepare wrapper function that passes weight/bias
                def kernel_with_model(*args, **kwargs):
                    return original_kernel_function(
                        args[0], weight, bias, **kernel_kwargs
                    )

                # Update kernel function and clear init_inputs (already handled)
                kernel_function = kernel_with_model
                kernel_init_args = []
        except Exception as exc:
            if not args.quiet:
                print(f"⚠️  Warning: Failed to extract model parameters: {exc}")
                print("   Falling back to direct kernel invocation")

    # Run once to verify it executes
    try:
        out = _run_once(kernel_function, kernel_args, kernel_init_args, kernel_name)
        if not args.quiet:
            print(f"✓ {kernel_name} executes successfully")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    except Exception as exc:
        print(f"❌ {kernel_name} failed: {exc}")
        results["kernels"][kernel_name] = {
            "time_ms": float("inf"),
            "error": str(exc),
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
        sys.exit(1)

    # Benchmark
    kernel_time = benchmark(
        kernel_function,
        kernel_args,
        kernel_init_args,
        kernel_name,
        args.warmup,
        args.repeat,
    )

    results["kernels"][kernel_name] = {
        "time_ms": kernel_time,
        "path": str(args.kernel),
    }

    if baseline_time is not None and kernel_time != float("inf"):
        speedup = baseline_time / kernel_time
        results["kernels"][kernel_name]["speedup"] = speedup
        if not args.quiet:
            print(f"Speedup vs PyTorch: {speedup:.2f}x")

    # Save JSON results
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(130)
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}")
        sys.exit(1)
