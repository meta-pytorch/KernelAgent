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

"""Fused matrix multiplication with block normalization example."""

import sys
import time
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from triton_kernel_agent import TritonKernelAgent


def main():
    """Generate and test a fused matmul + block normalization kernel."""
    # Load environment
    load_dotenv()

    # Create agent
    agent = TritonKernelAgent()

    print("=" * 80)
    print("Fused Matrix Multiplication + Block Normalization Kernel Generation")
    print("Matrix dimensions: A(128, 64) @ B(64, 256)")
    print("Block dimension: 32")
    print("Operation: Fused matmul + optional residual + block normalization")
    print("=" * 80)

    # Define the problem
    problem_description = """
Write a Triton kernel for fused matrix multiplication with block normalization:

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.norm = nn.LayerNorm(32)

    def forward(self, a, b, residual=None, use_residual=False):
        # Matrix multiplication
        x = a @ b
        
        # Optional residual addition
        if use_residual and residual is not None:
            x = x + residual
            
        # Block-wise normalization
        m, n = x.shape
        x = x.reshape(m, -1, 32)  # block_dim = 32
        x = self.norm(x)
        x = x.reshape(m, n)
        return x

# Define input dimensions and parameters
M, K, N = 128, 64, 256
block_dim = 32
dtype = torch.float32

def get_inputs():
    # Generate matrices and optional residual
    a = torch.randn(M, K, dtype=dtype, device='cuda')
    b = torch.randn(K, N, dtype=dtype, device='cuda') 
    residual = torch.randn(M, N, dtype=dtype, device='cuda')
    return [a, b, residual, True]  # use_residual=True

def get_init_inputs():
    return []
    """

    print("\nGenerating fused matmul + block norm kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(
        problem_description, test_code=None
    )  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated fused matmul + block norm kernel!")
        print(
            f"  Worker {result['worker_id']} found solution in {result['rounds']} rounds"
        )
        print(f"  Session directory: {result['session_dir']}")

        print("\n" + "=" * 80)
        print("Generated Kernel Code:")
        print("=" * 80)
        print(result["kernel_code"])
        print("=" * 80)

        # Save the kernel to a file for future use
        kernel_file = "fused_reduction_gemm_kernel.py"
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
        if result.get("session_dir"):
            import os

            problem_file = os.path.join(result["session_dir"], "problem.txt")
            if os.path.exists(problem_file):
                with open(problem_file, "r") as f:
                    print(f"\n  Problem attempted:\n{f.read()}")

        sys.exit(1)

    # Cleanup
    agent.cleanup()
    print("\n✓ Fused matmul + block norm example completed!")


if __name__ == "__main__":
    main()
