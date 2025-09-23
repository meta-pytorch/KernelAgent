#!/usr/bin/env python3
"""
Example of using Triton Kernel Agent with additional reference code.

This demonstrates how to provide a reference implementation that the agent
will use to understand the algorithm. The generated Triton kernel should:
1. Be numerically correct (match the reference)
2. Be faster than PyTorch native operations
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_kernel_agent import TritonKernelAgent


def main():
    """Example of generating a kernel with reference implementation."""

    # Load environment variables
    load_dotenv()

    # Create agent
    print("Creating Triton Kernel Agent...")
    agent = TritonKernelAgent(num_workers=2, max_rounds=5)

    # Define the problem
    problem_description = """
    Create a Triton kernel for fused dropout + residual addition.

    The kernel should:
    1. Apply dropout to input tensor with probability p
    2. Add the result to a residual tensor
    3. Return the final output

    Input tensors shape: (batch_size, seq_len, hidden_dim)
    All tensors are float32 on CUDA device.
    """

    # Provide a reference implementation
    additional_code = '''
def reference_dropout_residual(x, residual, p=0.1, training=True):
    """
    Reference implementation of dropout + residual.

    This is a correct but potentially slow implementation using PyTorch.
    The Triton kernel should produce the same results but run faster.
    """
    import torch
    import torch.nn.functional as F

    if training:
        # Apply dropout
        dropout_out = F.dropout(x, p=p, training=True)
        # Add residual
        output = dropout_out + residual
    else:
        # No dropout during inference
        output = x + residual

    return output
'''

    # Generate the kernel
    print("\nGenerating optimized Triton kernel...")
    print("Problem:", problem_description.strip()[:100] + "...")
    print("\nReference implementation provided: Yes")
    print("Goal: Generate a kernel that is both correct AND faster than PyTorch\n")

    result = agent.generate_kernel(
        problem_description=problem_description, additional_code=additional_code
    )

    # Display results
    if result["success"]:
        print(f"✅ SUCCESS! Generated kernel in {result['rounds']} rounds")
        print(f"Worker {result['worker_id']} found the solution")
        print(f"\nSession directory: {result['session_dir']}")
        print("\nGenerated kernel saved to:")
        print(f"  {result['session_dir']}/final_kernel.py")

        # Show a snippet of the kernel
        kernel_lines = result["kernel_code"].split("\n")
        print("\nKernel preview (first 20 lines):")
        print("-" * 60)
        for i, line in enumerate(kernel_lines[:20]):
            print(line)
        if len(kernel_lines) > 20:
            print("... (truncated)")
        print("-" * 60)

        print("\nThe generated kernel should:")
        print("1. ✅ Produce the same results as the reference implementation")
        print("2. ✅ Run faster than the PyTorch operations used in the TEST")
        print("3. ✅ Handle edge cases and different input sizes")
        print(
            "\nNote: The performance target is the test's PyTorch code, NOT the reference implementation!"
        )

    else:
        print(f"❌ FAILED: {result['message']}")
        print(f"Session directory: {result['session_dir']}")
        print("\nCheck the logs in the session directory for more details.")

    # Cleanup
    agent.cleanup()


if __name__ == "__main__":
    main()
