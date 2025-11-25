import triton
import triton.language as tl
import torch


@triton.jit
def _relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for ReLU activation function.
    
    Implements: output = max(0, input) element-wise
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor  
        n_elements: Total number of elements in tensor
        BLOCK_SIZE: Number of elements processed per block
    """
    # Get program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate the block's start position and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary protection
    mask = offsets < n_elements
    
    # Load input data using Triton
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute ReLU using Triton operations: max(0, x)
    # This is the core computation - must use Triton, not PyTorch
    output_data = tl.where(input_data > 0, input_data, 0.0)
    
    # Store result using Triton
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function for ReLU activation using Triton kernel.
    
    This function handles:
    - Input validation and tensor preparation
    - Grid configuration and kernel launch
    - Result packaging
    
    All mathematical computation is delegated to the Triton kernel.
    
    Args:
        input_tensor: Input tensor of shape [batch_size, dim] with dtype bfloat16
        
    Returns:
        Output tensor with ReLU activation applied element-wise
    """
    # Input validation
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if input_tensor.device.type != 'cuda':
        raise ValueError("Input tensor must be on CUDA device")
    
    # Create output tensor with same shape and dtype as input
    output_tensor = torch.empty_like(input_tensor)
    
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Choose block size - powers of 2 for optimal performance
    # Using 1024 as it's a good balance for modern GPUs
    BLOCK_SIZE = 1024
    
    # Calculate grid size: number of blocks needed to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch Triton kernel
    # All mathematical computation happens here in the Triton kernel
    _relu_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor