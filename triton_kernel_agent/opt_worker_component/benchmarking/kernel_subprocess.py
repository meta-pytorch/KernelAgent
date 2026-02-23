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
import inspect
import json
import sys
from pathlib import Path

from timing import (
    import_module,
    load_kernel_function,
    load_problem_interface,
    prepare_inputs,
)
from typing import Any, Callable, Tuple

import torch
import triton.testing as tt


_CONV_TYPES = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
)
_NORM_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
)
_POOL_TYPES = (
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AdaptiveMaxPool1d,
    torch.nn.AdaptiveMaxPool2d,
    torch.nn.AdaptiveMaxPool3d,
)


def _extract_model_params(model: torch.nn.Module) -> tuple[dict[str, Any], list]:
    """Extract all learnable parameters and layer config from a PyTorch model.

    Walks model submodules and extracts weights, biases, and layer-specific
    hyperparameters (stride, padding, etc.) into a flat dict keyed by
    parameter name.  The keys are chosen to match common kernel function
    signatures (``weight``, ``bias``, ``conv_bias``, ``stride``, …).

    Returns:
        Tuple of (flat dict mapping parameter names to values,
                  ordered list of all conv/linear weight tensors).
    """
    params: dict[str, Any] = {}
    all_weights: list = []

    for _, module in model.named_modules():
        if isinstance(module, (*_CONV_TYPES, torch.nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                all_weights.append(module.weight)
                params.setdefault("weight", module.weight)
                params.setdefault("w", module.weight)  # alias
                if getattr(module, "bias", None) is not None:
                    params.setdefault("conv_bias", module.bias)
                    params.setdefault("bias", module.bias)

                # Conv / ConvTranspose hyperparameters
                for attr in ("stride", "padding", "dilation", "output_padding"):
                    val = getattr(module, attr, None)
                    if val is not None:
                        params.setdefault(attr, val)
                if hasattr(module, "groups"):
                    params.setdefault("groups", module.groups)

                # NO break — collect all conv/linear weights

        elif isinstance(module, _NORM_TYPES):
            if getattr(module, "weight", None) is not None:
                params.setdefault("weight", module.weight)
                params.setdefault("w", module.weight)
            if getattr(module, "bias", None) is not None:
                params.setdefault("bias", module.bias)
            if hasattr(module, "eps"):
                params["eps"] = module.eps
            if hasattr(module, "num_groups"):
                params["num_groups"] = module.num_groups
            if hasattr(module, "normalized_shape"):
                params["normalized_shape"] = module.normalized_shape

        elif isinstance(module, _POOL_TYPES):
            for attr in ("kernel_size", "stride", "padding", "dilation"):
                val = getattr(module, attr, None)
                if val is not None:
                    params.setdefault(attr, val)

    # Top-level bias on the model itself (for fusion kernels like Conv+ReLU+BiasAdd)
    if hasattr(model, "bias") and isinstance(
        model.bias, (torch.Tensor, torch.nn.Parameter)
    ):
        params["add_bias"] = model.bias
        params.setdefault("bias", model.bias)

    return params, all_weights


def _run_once(
    fn: Callable, inputs: Tuple[torch.Tensor, ...], init_inputs: list, name: str
) -> torch.Tensor:
    """Run kernel once to verify execution and get output shape/dtype."""
    try:
        with torch.inference_mode():
            return fn(*inputs, *init_inputs)
    except Exception as exc:
        raise RuntimeError(f"{name} failed to execute: {exc}") from exc


def _benchmark(
    fn: Callable,
    inputs: Tuple[torch.Tensor, ...],
    init_inputs: list,
    name: str,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark a kernel function using triton.testing.do_bench."""
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


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Task-agnostic Triton kernel benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

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
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--size", type=int, default=4096, help="Problem size N")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--json", type=Path, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    args.problem = args.problem.resolve()
    args.kernel = args.kernel.resolve()
    return args


def _load_problem(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[type, tuple, list, torch.nn.Module | None]:
    """Load problem interface, prepare inputs, and optionally create baseline model.

    Returns:
        Tuple of (Model class, inputs, init_inputs, baseline_model or None)
    """
    Model, get_inputs, get_init_inputs = load_problem_interface(args.problem)

    # Check for optional benchmark config override
    try:
        problem_mod = import_module(args.problem, "problem")
        get_benchmark_config = getattr(problem_mod, "get_benchmark_config", None)
        if get_benchmark_config is not None:
            config = get_benchmark_config()
            args.warmup = config.get("warmup", args.warmup)
            args.repeat = config.get("repeat", args.repeat)
            if not args.quiet:
                print(
                    f"Using problem-specific config: "
                    f"warmup={args.warmup}, repeat={args.repeat}"
                )
    except Exception:
        pass

    inputs = prepare_inputs(get_inputs, device=device, dtype=dtype)

    init_inputs = get_init_inputs() if get_init_inputs is not None else []
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]

    # Create baseline model if requested
    baseline_model = None
    if args.baseline:
        baseline_model = (
            Model(*init_inputs).to(device=device, dtype=dtype)
            if init_inputs
            else Model().to(device=device, dtype=dtype)
        )
        baseline_model.eval()
        out = _run_once(baseline_model, inputs, [], "Reference")
        if not args.quiet:
            print(f"Reference output shape: {out.shape}, dtype: {out.dtype}")
            print()

    return Model, inputs, init_inputs, baseline_model


def _prepare_kernel(
    kernel_file: Path,
    Model: type,
    baseline_model: torch.nn.Module | None,
    init_inputs: list,
    device: torch.device,
    dtype: torch.dtype,
    quiet: bool = False,
) -> tuple[Callable, list]:
    """Load kernel and wrap it with model parameters if needed.

    Returns:
        Tuple of (kernel_function, kernel_init_args) where kernel_init_args
        is always [] (model params are baked into the wrapper).
    """
    kernel_function = load_kernel_function(kernel_file)

    # Check if kernel expects model-derived parameters:
    # - 'weight' / 'w' for Conv, Linear, Norm layers
    # - pooling scalars (kernel_size, stride, padding, dilation) for Pool layers
    _MODEL_PARAM_NAMES = {"weight", "w", "kernel_size", "stride", "padding", "dilation"}
    needs_model = False
    has_var_positional = False
    has_var_keyword = False
    kernel_params: list[str] = []
    try:
        sig = inspect.signature(kernel_function)
        kernel_params = [
            name
            for name, p in sig.parameters.items()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        param_kinds = [p.kind for p in sig.parameters.values()]
        has_var_positional = any(
            k == inspect.Parameter.VAR_POSITIONAL for k in param_kinds
        )
        has_var_keyword = any(k == inspect.Parameter.VAR_KEYWORD for k in param_kinds)
        if _MODEL_PARAM_NAMES.intersection(kernel_params):
            needs_model = True
        # If kernel uses *args/**kwargs, inspect source for weight-related patterns
        if not needs_model and (has_var_positional or has_var_keyword):
            try:
                src = inspect.getsource(kernel_function)
                needs_model = any(
                    kw in src
                    for kw in (
                        "weight",
                        "is_weight",
                        "w.shape",
                        "w.ndim",
                        "kernel_size",
                        "dilation",
                    )
                )
            except (OSError, TypeError):
                pass
    except Exception:
        pass

    kernel_init_args: list = []

    if needs_model and Model is not None:
        try:
            extract_model = baseline_model
            if extract_model is None:
                extract_model = (
                    Model(*init_inputs).to(device=device, dtype=dtype)
                    if init_inputs
                    else Model().to(device=device, dtype=dtype)
                )

            model_params, all_weights = _extract_model_params(extract_model)

            if model_params:
                original_fn = kernel_function

                if has_var_positional and all_weights:
                    # *args kernel: pass (inputs + weights) positionally,
                    # config as kwargs with uniform tuples collapsed to scalars
                    def kernel_with_model_varargs(*args, **kwargs):
                        pos_args = list(args) + list(all_weights)
                        config_kwargs = {}
                        for k, v in model_params.items():
                            if k not in (
                                "weight",
                                "w",
                                "bias",
                                "conv_bias",
                                "add_bias",
                            ):
                                if (
                                    isinstance(v, (tuple, list))
                                    and len(v) >= 1
                                    and all(e == v[0] for e in v)
                                ):
                                    v = v[0]
                                config_kwargs[k] = v
                        return original_fn(*pos_args, **config_kwargs)

                    kernel_function = kernel_with_model_varargs
                else:
                    # Build a wrapper that maps extracted params into the kernel
                    # signature by name.  Positional args (from *inputs) fill the
                    # leading parameters that are NOT found in model_params.
                    def kernel_with_model(*args, **kwargs):
                        bound: dict[str, Any] = {}
                        positional_idx = 0
                        for pname in kernel_params:
                            if pname in model_params:
                                v = model_params[pname]
                                if (
                                    isinstance(v, (tuple, list))
                                    and len(v) >= 1
                                    and all(e == v[0] for e in v)
                                ):
                                    v = v[0]
                                bound[pname] = v
                            elif positional_idx < len(args):
                                bound[pname] = args[positional_idx]
                                positional_idx += 1
                        return original_fn(**bound)

                    kernel_function = kernel_with_model
        except Exception as exc:
            if not quiet:
                print(f"⚠️  Warning: Failed to extract model parameters: {exc}")
                print("   Falling back to direct kernel invocation")

    # For kernels that don't need model extraction but have init_inputs
    # (e.g., dim parameter for reduction ops), pass them as positional args
    if not needs_model and init_inputs:
        kernel_init_args = init_inputs

    return kernel_function, kernel_init_args


def _save_results(results: dict[str, Any], path: Path) -> None:
    """Save benchmark results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {path}")


def main():
    args = _parse_args()

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # Auto-detect dtype from kernel source (matches NCU wrapper's dtype inference)
    try:
        kernel_source = args.kernel.read_text()
        if "bfloat16" in kernel_source.lower():
            dtype = torch.bfloat16
        elif "float16" in kernel_source.lower() or "half" in kernel_source.lower():
            dtype = torch.float16
        elif "float32" in kernel_source.lower():
            dtype = torch.float32
    except Exception:
        pass

    if not args.quiet:
        print("=" * 80)
        print("TRITON KERNEL PROFILING")
        print("=" * 80)
        print(f"Problem: {args.problem.name}")
        print(f"Size: {args.size}")
        print(f"Device: {device}, Dtype: {dtype}")
        print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
        print()

    # Load problem and prepare inputs
    try:
        Model, inputs, init_inputs, baseline_model = _load_problem(args, device, dtype)
    except Exception as exc:
        print(f"❌ Failed to load problem: {exc}")
        sys.exit(1)

    results: dict[str, Any] = {
        "problem": str(args.problem),
        "size": args.size,
        "device": str(device),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "kernels": {},
    }

    # Benchmark baseline (if requested)
    baseline_time = None
    if baseline_model is not None:
        if not args.quiet:
            print("1. PyTorch Reference")
        baseline_time = _benchmark(
            baseline_model, inputs, [], "PyTorch", args.warmup, args.repeat
        )
        results["kernels"]["pytorch_reference"] = {
            "time_ms": baseline_time,
            "speedup": 1.0,
        }
        if not args.quiet:
            print()

    # Load and prepare kernel
    kernel_name = args.kernel.stem
    if not args.quiet:
        idx = 2 if args.baseline else 1
        print(f"{idx}. Candidate: {kernel_name}")

    try:
        kernel_fn, kernel_init_args = _prepare_kernel(
            args.kernel, Model, baseline_model, init_inputs, device, dtype, args.quiet
        )
    except Exception as exc:
        print(f"❌ Failed to load kernel: {exc}")
        results["kernels"][kernel_name] = {"time_ms": float("inf"), "error": str(exc)}
        if args.json:
            _save_results(results, args.json)
        sys.exit(1)

    # Verify kernel executes
    try:
        out = _run_once(kernel_fn, inputs, kernel_init_args, kernel_name)
        if not args.quiet:
            print(f"✓ {kernel_name} executes successfully")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    except Exception as exc:
        print(f"❌ {kernel_name} failed: {exc}")
        results["kernels"][kernel_name] = {"time_ms": float("inf"), "error": str(exc)}
        if args.json:
            _save_results(results, args.json)
        sys.exit(1)

    # Benchmark kernel
    kernel_time = _benchmark(
        kernel_fn, inputs, kernel_init_args, kernel_name, args.warmup, args.repeat
    )
    results["kernels"][kernel_name] = {"time_ms": kernel_time, "path": str(args.kernel)}

    # Calculate speedup
    if baseline_time is not None and kernel_time != float("inf"):
        speedup = baseline_time / kernel_time
        results["kernels"][kernel_name]["speedup"] = speedup
        if not args.quiet:
            print(f"Speedup vs PyTorch: {speedup:.2f}x")

    if args.json:
        _save_results(results, args.json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(130)
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}")
        sys.exit(1)
