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

"""Core timing and model loading utilities for kernel benchmarking.

This module consolidates:
- Timing functions (CUDA events, do_bench, host timing)
- Model/kernel loading utilities
- Statistics computation

Inspired by KernelBench's timing.py
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Model and Kernel Loading Utilities
# =============================================================================


class CompilationError(RuntimeError):
    """Raised when a kernel or problem file fails to compile/import."""

    pass


def import_module(path: Path, module_name: Optional[str] = None):
    """Dynamically import a Python file.

    Args:
        path: Path to the Python file
        module_name: Optional name for the module (auto-generated if None)

    Returns:
        The imported module

    Raises:
        FileNotFoundError: If path doesn't exist
        CompilationError: If import fails
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if module_name is None:
        module_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Failed to create spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CompilationError(f"Failed to import {path}: {exc}") from exc

    return module


def load_problem_interface(
    problem_file: Path,
) -> Tuple[type, Callable, Optional[Callable]]:
    """Load the standard problem interface from a problem file.

    Args:
        problem_file: Path to problem file

    Returns:
        Tuple of (Model class, get_inputs function, get_init_inputs function)

    Raises:
        CompilationError: If problem file doesn't define required interface
    """
    module = import_module(problem_file, "problem")

    Model = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)

    if Model is None:
        raise CompilationError("Problem file must define 'Model' class")
    if get_inputs is None:
        raise CompilationError("Problem file must define 'get_inputs()' function")

    return Model, get_inputs, get_init_inputs


def prepare_inputs(
    get_inputs: Callable,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, ...]:
    """Prepare inputs by converting to target device and dtype.

    Args:
        get_inputs: Function that returns inputs
        device: Target device
        dtype: Target dtype for floating-point tensors

    Returns:
        Tuple of prepared inputs
    """
    inputs = get_inputs()
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    # Convert inputs to target device and dtype
    # IMPORTANT: Only convert floating-point tensors; preserve integer/bool tensors
    converted_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            # Move to device
            inp = inp.to(device=device)
            # Convert dtype ONLY for floating-point tensors
            # Preserve integer/bool tensors (e.g., targets for classification)
            if inp.is_floating_point():
                inp = inp.to(dtype=dtype)
        converted_inputs.append(inp)

    return tuple(converted_inputs)


def prepare_pytorch_model(
    problem_file: Path,
    device: torch.device | str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """Prepare PyTorch model and inputs for benchmarking.

    This handles the full workflow:
    1. Load problem interface (Model, get_inputs, get_init_inputs)
    2. Initialize model with init inputs
    3. Move model to device
    4. Handle dtype conversion based on whether model has parameters

    Args:
        problem_file: Path to problem file
        device: Target device
        dtype: Target dtype (auto-detected if None)

    Returns:
        Tuple of (model, inputs) ready for benchmarking
    """
    # Load problem interface
    Model, get_inputs, get_init_inputs = load_problem_interface(problem_file)

    # Get initialization inputs (e.g., features, eps for RMSNorm)
    init_inputs = []
    if get_init_inputs is not None:
        init_inputs = get_init_inputs()
        if not isinstance(init_inputs, (tuple, list)):
            init_inputs = [init_inputs]

    # Initialize model
    if init_inputs:
        model = Model(*init_inputs)
    else:
        model = Model()

    # Move model to CUDA
    model = model.cuda()

    # Check if model has trainable parameters
    has_parameters = any(p.numel() > 0 for p in model.parameters())

    # Get inputs
    inputs = get_inputs()
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    # Determine target dtype
    # Default to bfloat16 unless explicitly specified or model is a loss function
    target_dtype = dtype or torch.bfloat16

    # Check if this is actually a loss function
    is_loss_function = isinstance(model, torch.nn.modules.loss._Loss)

    # Handle dtype conversion based on model type
    if has_parameters or not is_loss_function:
        # Models with parameters (Conv, Linear, etc.) OR compute operations (matmul, etc.)
        # → use bfloat16 (or user-specified dtype)
        if has_parameters:
            model = model.to(target_dtype)
        inputs = [
            (
                inp.cuda().to(target_dtype)
                if isinstance(inp, torch.Tensor) and inp.is_floating_point()
                else inp.cuda()
                if isinstance(inp, torch.Tensor)
                else inp
            )
            for inp in inputs
        ]
    else:
        # Loss functions (no parameters) → use float32 for compatibility
        # PyTorch cross_entropy doesn't support bf16 on CUDA
        processed_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                if i == 0 and inp.is_floating_point():
                    # First input (predictions) - convert to float32 for compatibility
                    processed_inputs.append(inp.cuda().to(torch.float32))
                else:
                    # Other inputs (like targets) - just move to CUDA, preserve dtype
                    processed_inputs.append(inp.cuda())
            else:
                processed_inputs.append(inp)
        inputs = processed_inputs

    return model, tuple(inputs)


def load_kernel_function(kernel_file: Path) -> Callable:
    """Load kernel_function from a kernel file.

    Args:
        kernel_file: Path to kernel file

    Returns:
        The kernel_function callable

    Raises:
        CompilationError: If kernel file doesn't define kernel_function
    """
    module = import_module(kernel_file, "kernel")

    kernel_function = getattr(module, "kernel_function", None)
    if kernel_function is None:
        raise CompilationError(
            f"Kernel file {kernel_file.name} must define 'kernel_function'"
        )

    return kernel_function


# =============================================================================
# Timing Utilities
# =============================================================================


def clear_l2_cache(device: torch.device | str = "cuda") -> None:
    """Clear L2 cache by thrashing with a large tensor.

    This ensures we measure cold cache performance, which is more representative
    of real-world scenarios where data isn't already cached.

    Reference: KernelBench timing.py
    L2 cache sizes: A100=40MB, H100=50MB, H200=90MB, RTX4090=72MB, L40S=48MB
    We overwrite >256MB to fully thrash L2 cache.

    Args:
        device: CUDA device to use
    """
    # 32 * 1024 * 1024 * 8B = 256MB - enough to thrash most GPU L2 caches
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)  # Write to tensor to ensure cache thrashing
    del dummy


def time_with_cuda_events(
    kernel_fn: Callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    clear_cache: bool = True,
    discard_first: int = 0,
    verbose: bool = False,
    device: Optional[torch.device | str] = None,
) -> list[float]:
    """Time a CUDA kernel using CUDA events for accurate device-side timing.

    This measures actual GPU execution time without host-side overhead.
    Each trial clears L2 cache to measure cold-cache performance.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials
        clear_cache: Whether to clear L2 cache between trials
        discard_first: Number of initial trials to discard
        verbose: Print per-trial timing info
        device: CUDA device to use (None = current device)

    Returns:
        List of elapsed times in milliseconds (length = num_trials)
    """
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        # Warmup
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()

        if verbose:
            print(
                f"[Timing] Device: {torch.cuda.get_device_name(device)}, "
                f"warmup={num_warmup}, trials={num_trials}"
            )

        elapsed_times: list[float] = []

        # Timing trials
        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            if clear_cache:
                clear_l2_cache(device=device)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)

            if trial >= discard_first:
                if verbose:
                    print(
                        f"  Trial {trial - discard_first + 1}: {elapsed_time_ms:.3f} ms"
                    )
                elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def time_with_inductor_benchmarker(
    kernel_fn: Callable,
    args: list[Any],
    num_warmup: int = 25,
    verbose: bool = False,
) -> float:
    """Time using PyTorch Inductor's benchmarker (simplest approach).

    This is a thin wrapper around torch._inductor.runtime.benchmarking.benchmarker,
    which handles CUDA synchronization and timing internally.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        verbose: Print timing info

    Returns:
        Elapsed time in milliseconds (single value, not a list)

    Note:
        This uses a private PyTorch API (_inductor) which may change without notice.
    """
    from torch._inductor.runtime.benchmarking import benchmarker

    # Warmup
    for _ in range(num_warmup):
        kernel_fn(*args)

    ms = benchmarker.benchmark_gpu(lambda: kernel_fn(*args))

    if verbose:
        print(f"[Timing] Inductor benchmarker: {ms:.4f} ms")

    return ms


def time_with_triton_do_bench(
    kernel_fn: Callable,
    args: list[Any],
    warmup: int = 25,
    rep: int = 100,
    verbose: bool = False,
    device: Optional[torch.device | str] = None,
) -> list[float]:
    """Time using Triton's do_bench with adaptive trial count.

    Triton's do_bench automatically determines the number of trials based on
    warmup/rep time budgets. This is convenient but gives less control.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        warmup: Warmup time budget in milliseconds
        rep: Repetition time budget in milliseconds
        verbose: Print timing info
        device: CUDA device to use

    Returns:
        List of all trial times in milliseconds
    """
    if device is None:
        device = torch.cuda.current_device()

    try:
        from triton import testing as triton_testing
    except ImportError:
        raise ImportError("Triton is required for time_with_triton_do_bench")

    with torch.cuda.device(device):
        if verbose:
            print(
                f"[Timing] Using triton.do_bench on {torch.cuda.get_device_name(device)}"
            )

        def wrapped_fn():
            return kernel_fn(*args)

        times = triton_testing.do_bench(
            fn=wrapped_fn,
            warmup=warmup,
            rep=rep,
            grad_to_none=None,
            quantiles=None,
            return_mode="all",
        )

    return times


def compute_timing_stats(
    elapsed_times: list[float],
    device: Optional[torch.device | str] = None,
) -> dict[str, Any]:
    """Compute essential timing statistics.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device (for recording hardware info)

    Returns:
        Dictionary with timing statistics:
            - mean: Mean time in ms
            - std: Standard deviation in ms
            - min: Minimum time in ms
            - max: Maximum time in ms
            - num_trials: Number of trials
            - all_times: All trial times
            - hardware: GPU name (if device provided)
    """
    times_array = np.array(elapsed_times)

    stats = {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "num_trials": len(elapsed_times),
        "all_times": [float(t) for t in elapsed_times],
    }

    if device is not None:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats
