#!/usr/bin/env python3
"""
Fused dot-compress operation example for KernelAgent.
"""

import sys
import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from triton_kernel_agent import TritonKernelAgent


def main():
    """Generate and test a fused dot-compress kernel."""
    # Load environment
    load_dotenv()

    # Create agent
    agent = TritonKernelAgent()

    print("=" * 80)
    print("Fused Dot-Compress Operation Kernel Generation")
    print("Batch size: 8, Matrix dimensions: 128x64")
    print("Operation: X^T @ Y, then X @ (X^T @ Y + Z)")
    print("=" * 80)

    # Define the problem
    problem_description = """
Write a Triton kernel for fused dot-compress operation:

# NOTE: You should only write ONE kernel to do the entire operation, instead of writing multiple kernels

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        # Fused dot-compress operation
        # Step 1: Compute X^T @ Y (batch matrix multiplication)
        xty = torch.bmm(x.permute(0, 2, 1), y)
        
        # Step 2: Add Z and multiply with X
        out = torch.bmm(x, xty + z)
        
        return out

# Define input dimensions and parameters

B=2048
NUM_EMB, DIM, NUM_COMPRESS_EMB = 512, 128, 32

def generate_dot_compress_inputs(
    B: int,  # batch size
    NUM_EMB: int,  # num embeddings
    DIM: int,  # embedding dimension
    NUM_COMPRESS_EMB: int,  # num compressed embeddings
    dtype: torch.dtype = torch.float16,  # dtype of the inputs
    requires_grad: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x = torch.randn(
        (B, NUM_EMB, DIM),
        device="cuda",
        dtype=dtype,
    ).requires_grad_(requires_grad)
    y = torch.randn(
        (B, NUM_EMB, NUM_COMPRESS_EMB),
        device="cuda",
        dtype=dtype,
    ).requires_grad_(requires_grad)
    z = torch.randn(
        (B, DIM, NUM_COMPRESS_EMB),
        device="cuda",
        dtype=dtype,
    ).requires_grad_(requires_grad)

    return x, y, z


def get_init_inputs():
    return []
    """

    print("\nGenerating fused dot-compress kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(problem_description, test_code=None)  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated fused dot-compress kernel!")
        print(f"  Worker {result['worker_id']} found solution in {result['rounds']} rounds")
        print(f"  Session directory: {result['session_dir']}")

        print("\n" + "=" * 80)
        print("Generated Kernel Code:")
        print("=" * 80)
        print(result["kernel_code"])
        print("=" * 80)

        # Save the kernel to a file for future use
        kernel_file = "fused_dcpp_kernel.py"
        with open(kernel_file, "w") as f:
            f.write(result["kernel_code"])
        print(f"\n✓ Kernel saved to: {kernel_file}")

        # Run the generated test to show performance
        print("\nRunning the generated test...")

        # Read the generated test code
        import os

        test_file = os.path.join(result["session_dir"], "test.py")
        with open(test_file, "r") as f:
            test_code = f.read()

        print("\nGenerated Test Code:")
        print("=" * 80)
        print(test_code)
        print("=" * 80)

        # Create a test script that uses the generated kernel
        # First, copy the kernel to kernel.py so the test can import it
        with open("kernel.py", "w") as f:
            f.write(result["kernel_code"])

        final_test_script = test_code

        with open("final_test.py", "w") as f:
            f.write(final_test_script)

        print("\nExecuting kernel test...")
        os.system("python final_test.py")

        # Cleanup kernel.py
        if os.path.exists("kernel.py"):
            os.remove("kernel.py")

        # Cleanup temporary test file
        os.remove("final_test.py")

    else:
        print("\n✗ Failed to generate kernel")
        print(f"  Message: {result['message']}")
        print(f"  Session directory: {result['session_dir']}")
        
        # Still show what was attempted
        if result.get('session_dir'):
            import os
            problem_file = os.path.join(result['session_dir'], 'problem.txt')
            if os.path.exists(problem_file):
                with open(problem_file, 'r') as f:
                    print(f"\n  Problem attempted:\n{f.read()}")
        
        sys.exit(1)

    # Cleanup
    agent.cleanup()
    print("\n✓ Fused dot-compress example completed!")


if __name__ == "__main__":
    main()