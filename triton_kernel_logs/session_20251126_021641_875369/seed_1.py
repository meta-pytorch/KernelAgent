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
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data from global memory
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute ReLU: max(0, x) using Triton operations
    # For bfloat16 inputs, we need to handle the computation carefully
    # Convert to float32 for precise computation, then convert back
    if input_ptr.dtype.element_ty == tl.bfloat16:
        # Convert bfloat16 to float32 for computation
        input_f32 = input_data.to(tl.float32)
        # Compute ReLU in float32
        output_f32 = tl.where(input_f32 > 0, input_f32, 0.0)
        # Convert back to bfloat16 for storage
        result = output_f32.to(tl.bfloat16)
    else:
        # For other types, compute directly
        result = tl.where(input_data > 0, input_data, 0.0)
    
    # Store result to global memory
    tl.store(output_ptr + offsets, result, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function for ReLU activation using Triton.
    
    This function implements the ReLU activation: output = max(0, input)
    using a fused Triton kernel that handles the entire computation in a single pass.
    
    FUSION ANALYSIS:
    - ReLU is a simple elementwise operation: no fusion opportunities with other ops
    - The entire computation runs in a single Triton kernel pass
    - No intermediate tensors or separate kernel launches needed
    - All mathematical work happens inside the Triton kernel using tl operations
    
    Args:
        input_tensor: Input tensor of shape [batch_size, dim] with dtype bfloat16
        
    Returns:
        Output tensor of same shape and dtype as input, with ReLU applied
    """
    # Input validation
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if input_tensor.device.type != 'cuda':
        raise ValueError("Input tensor must be on CUDA device")
    
    # Allocate output tensor (PyTorch operation allowed in wrapper)
    output_tensor = torch.empty_like(input_tensor)
    
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # If tensor is empty, return immediately
    if n_elements == 0:
        return output_tensor
    
    # Choose optimal block size based on tensor size
    # Use powers of 2 as recommended in Triton guidelines
    if n_elements <= 1024:
        BLOCK_SIZE = 64
    elif n_elements <= 4096:
        BLOCK_SIZE = 128
    elif n_elements <= 16384:
        BLOCK_SIZE = 256
    elif n_elements <= 65536:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks needed)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch Triton kernel - all computation happens here
    _relu_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor