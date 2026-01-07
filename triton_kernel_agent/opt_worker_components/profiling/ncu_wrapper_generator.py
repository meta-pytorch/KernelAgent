"""NCU wrapper script generation for kernel profiling."""

import logging
from pathlib import Path


class NCUWrapperGenerator:
    """Generates NCU wrapper scripts for profiling Triton kernels."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize the NCU wrapper generator.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    def create_ncu_wrapper(self, kernel_file: Path, problem_file: Path, output_dir: Path) -> Path:
        """
        Create NCU wrapper script for profiling.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            output_dir: Directory to write wrapper script

        Returns:
            Path to created wrapper script
        """
        wrapper_file = output_dir / "ncu_wrapper.py"

        wrapper_content = f'''"""NCU profiling wrapper."""
import sys
import torch
import inspect
sys.path.insert(0, str({str(kernel_file.parent)!r}))
sys.path.insert(0, str({str(problem_file.parent)!r}))

from {kernel_file.stem} import kernel_function
from {problem_file.stem} import get_inputs, get_init_inputs

# Try to import Model if it exists (for Conv, Linear, etc.)
try:
    from {problem_file.stem} import Model
    has_model = True
except ImportError:
    has_model = False

# Get inputs
inputs = get_inputs()

# Get additional initialization inputs (e.g., features, eps for RMSNorm)
init_inputs = get_init_inputs()

# Infer required dtype from kernel function signature/docstring
required_dtype = None
try:
    # Try to get dtype from kernel function docstring or source
    kernel_source = inspect.getsource(kernel_function)
    if 'bfloat16' in kernel_source.lower():
        required_dtype = torch.bfloat16
    elif 'float16' in kernel_source.lower() or 'half' in kernel_source.lower():
        required_dtype = torch.float16
    elif 'float32' in kernel_source.lower():
        required_dtype = torch.float32
except:
    pass

# Prepare inputs: move to CUDA and convert dtype if needed
# IMPORTANT: Only convert floating-point tensors; preserve integer tensors (e.g., class labels)
cuda_inputs = []
for inp in inputs:
    if isinstance(inp, torch.Tensor):
        # Move to CUDA if not already
        if not inp.is_cuda:
            inp = inp.cuda()
        # Convert dtype if required, but ONLY for floating-point tensors
        # Preserve integer/bool tensors (e.g., targets for classification)
        if required_dtype is not None and inp.is_floating_point() and inp.dtype != required_dtype:
            inp = inp.to(required_dtype)
        cuda_inputs.append(inp)
    else:
        cuda_inputs.append(inp)

# Check if this is a conv-like kernel that needs a Model to extract weights
needs_model = False
try:
    sig = inspect.signature(kernel_function)
    params = list(sig.parameters.keys())
    # Check if kernel expects 'weight' parameter (common for Conv, Linear, etc.)
    if 'weight' in params:
        needs_model = True
except:
    pass

if needs_model and has_model and init_inputs:
    # Initialize model to extract weight and bias
    model = Model(*init_inputs) if init_inputs else Model()

    # Move model to CUDA and convert dtype
    model = model.cuda()
    if required_dtype is not None:
        model = model.to(required_dtype)

    # Extract weight and bias from model
    # Check various possible attribute names
    weight = None
    bias = None
    layer = None
    for attr_name in ['conv1', 'conv2', 'conv3', 'conv1d', 'conv2d', 'conv', 'conv3d', 'linear', 'fc']:
        if hasattr(model, attr_name):
            layer = getattr(model, attr_name)
            if hasattr(layer, 'weight'):
                weight = layer.weight
                bias = layer.bias if hasattr(layer, 'bias') else None
                break

    if weight is not None and layer is not None:
        # Build arguments for kernel_function using keyword arguments
        # to avoid positional argument misalignment issues
        kernel_kwargs = {{}}

        # Add conv/linear-specific parameters if they exist
        if hasattr(layer, 'stride'):
            stride = layer.stride[0] if isinstance(layer.stride, (tuple, list)) else layer.stride
            kernel_kwargs['stride'] = stride
        if hasattr(layer, 'padding'):
            padding = layer.padding[0] if isinstance(layer.padding, (tuple, list)) else layer.padding
            kernel_kwargs['padding'] = padding
        if hasattr(layer, 'dilation'):
            dilation = layer.dilation[0] if isinstance(layer.dilation, (tuple, list)) else layer.dilation
            kernel_kwargs['dilation'] = dilation
        if hasattr(layer, 'groups'):
            kernel_kwargs['groups'] = layer.groups

        # Call kernel with extracted parameters
        output = kernel_function(cuda_inputs[0], weight, bias, **kernel_kwargs)
    else:
        # Fallback to original behavior
        output = kernel_function(*cuda_inputs, *init_inputs)
else:
    # Run kernel with both tensor inputs and initialization inputs
    # For example: RMSNorm needs kernel_function(x, features, eps)
    # For cross-entropy: kernel_function(predictions, targets)
    # where inputs come from get_inputs() and init_inputs from get_init_inputs()
    output = kernel_function(*cuda_inputs, *init_inputs)

print("Kernel executed successfully, output shape: " + str(output.shape if hasattr(output, 'shape') else type(output)))
'''

        wrapper_file.write_text(wrapper_content)
        self.logger.info(f"Created NCU wrapper: {wrapper_file}")
        return wrapper_file
