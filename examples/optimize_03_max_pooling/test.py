"""Correctness test for 3D max pooling kernel."""

import inspect
import sys

import torch
from kernel import kernel_function
from problem import get_init_inputs, get_inputs, Model

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


def _extract_model_params(model):
    """Extract learnable parameters and layer config from a PyTorch model."""
    params = {}

    for _, module in model.named_modules():
        if isinstance(module, (*_CONV_TYPES, torch.nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                params.setdefault("weight", module.weight)
                params.setdefault("w", module.weight)
                if getattr(module, "bias", None) is not None:
                    params.setdefault("conv_bias", module.bias)
                    params.setdefault("bias", module.bias)
                for attr in ("stride", "padding", "dilation", "output_padding"):
                    val = getattr(module, attr, None)
                    if val is not None:
                        params.setdefault(attr, val)
                if hasattr(module, "groups"):
                    params.setdefault("groups", module.groups)

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

    if hasattr(model, "bias") and isinstance(
        model.bias, (torch.Tensor, torch.nn.Parameter)
    ):
        params["add_bias"] = model.bias
        params.setdefault("bias", model.bias)

    # Extract simple scalar attributes stored by Model.__init__
    # (catches dim, negative_slope, min_val, max_val, etc.)
    _INIT_SCALAR_NAMES = {
        "dim",
        "negative_slope",
        "min_val",
        "max_val",
        "beta",
        "threshold",
        "alpha",
        "lambd",
        "upper",
        "lower",
        "p",
    }
    for attr_name in _INIT_SCALAR_NAMES:
        if hasattr(model, attr_name) and not isinstance(
            getattr(model, attr_name), (torch.Tensor, torch.nn.Module)
        ):
            params.setdefault(attr_name, getattr(model, attr_name))

    return params


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    # Setup reference model
    model = Model(*get_init_inputs()).to(device).to(dtype)
    inputs = [
        (
            x.to(device).to(dtype)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else (x.to(device) if isinstance(x, torch.Tensor) else x)
        )
        for x in get_inputs()
    ]

    # Get reference output
    with torch.no_grad():
        ref_output = model(*inputs)

    # Smart parameter binding: detect if kernel needs model params
    sig = inspect.signature(kernel_function)
    kernel_params = list(sig.parameters.keys())
    param_kinds = [p.kind for p in sig.parameters.values()]
    has_var_positional = any(k == inspect.Parameter.VAR_POSITIONAL for k in param_kinds)
    has_var_keyword = any(k == inspect.Parameter.VAR_KEYWORD for k in param_kinds)
    _MODEL_PARAM_NAMES = {
        "weight",
        "w",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "output_padding",
        "groups",
        "bias",
        "conv_bias",
        "eps",
        "num_groups",
        "normalized_shape",
        "dim",
        "negative_slope",
        "min_val",
        "max_val",
        "beta",
        "threshold",
        "alpha",
        "lambd",
        "upper",
        "lower",
        "p",
    }
    needs_model = bool(_MODEL_PARAM_NAMES & set(kernel_params))
    # If kernel uses *args/**kwargs, inspect its source for weight-related hints
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

    if needs_model:
        model_params = _extract_model_params(model)
        has_weight = "weight" in model_params or "w" in model_params
        if has_var_positional and has_weight:
            # *args kernel with weight: pass (input, weight1, weight2, ...) positionally
            pos_args = list(inputs)
            # Collect ALL conv/linear weights from model
            for _, mod in model.named_modules():
                if isinstance(mod, (*_CONV_TYPES, torch.nn.Linear)):
                    if hasattr(mod, "weight") and mod.weight is not None:
                        pos_args.append(mod.weight)
            # Pass config params as kwargs
            config_kwargs = {}
            for k, v in model_params.items():
                if k not in ("weight", "w", "bias", "conv_bias", "add_bias"):
                    # Convert uniform tuples to scalar int for compatibility
                    if (
                        isinstance(v, (tuple, list))
                        and len(v) >= 1
                        and all(e == v[0] for e in v)
                    ):
                        v = v[0]
                    config_kwargs[k] = v
            kernel_output = kernel_function(*pos_args, **config_kwargs)
        else:
            # Bind keyword args, adapting tuple/int form to match defaults
            call_args = {}
            pos_idx = 0
            for pname in kernel_params:
                p = sig.parameters[pname]
                if (
                    p.kind == inspect.Parameter.VAR_POSITIONAL
                    or p.kind == inspect.Parameter.VAR_KEYWORD
                ):
                    continue
                if pname in model_params:
                    val = model_params[pname]
                    # Convert tuple/list to scalar when kernel expects int
                    if isinstance(val, (tuple, list)):
                        if p.default is not inspect.Parameter.empty and isinstance(
                            p.default, int
                        ):
                            val = val[0]
                        elif len(val) == 1:
                            val = val[0]
                    call_args[pname] = val
                elif pos_idx < len(inputs):
                    call_args[pname] = inputs[pos_idx]
                    pos_idx += 1
            kernel_output = kernel_function(**call_args)
    else:
        kernel_output = kernel_function(*inputs)

    # Compare
    # Handle in-place kernels that return None
    if kernel_output is None:
        # Assume in-place modification of first input
        kernel_output = inputs[0]
    # Handle shape mismatch: kernel may return per-sample loss vs reference scalar mean
    if ref_output.dim() == 0 and kernel_output.dim() >= 1:
        kernel_output = kernel_output.mean()
    elif kernel_output.dim() == 0 and ref_output.dim() >= 1:
        ref_output = ref_output.mean()
    # Align dtypes for comparison
    if ref_output.dtype != kernel_output.dtype:
        # If kernel outputs higher precision, recompute reference at that precision
        # using the SAME inputs to ensure fair comparison
        if kernel_output.dtype == torch.float32 and ref_output.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            model_f32 = Model(*get_init_inputs()).to(device).to(torch.float32)
            inputs_f32 = [
                x.to(torch.float32) if x.is_floating_point() else x for x in inputs
            ]
            with torch.no_grad():
                ref_output = model_f32(*inputs_f32)
        else:
            kernel_output = kernel_output.to(ref_output.dtype)
    if torch.allclose(ref_output, kernel_output, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        max_diff = (ref_output - kernel_output).abs().max().item()
        print(f"FAIL: max difference = {max_diff}")
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
