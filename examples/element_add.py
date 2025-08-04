#!/usr/bin/env python3
"""
Element-wise addition example for KernelAgent.
"""

import sys
import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from triton_kernel_agent import TritonKernelAgent


def main():
    """Generate and test an element-wise addition kernel."""
    # Load environment
    load_dotenv()

    # Create agent
    agent = TritonKernelAgent()

    print("=" * 80)
    print("Element-wise Addition Kernel Generation")
    print("Vector size: 1024 elements")
    print("Operation: C = A + B (element-wise)")
    print("=" * 80)

    # Define the problem
    problem_description = """
Write a Triton kernel for element-wise addition of two vectors:

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b):
        # Element-wise addition: c = a + b
        return a + b

# Define input dimensions and parameters
vector_size = 1024
dtype = torch.float32

def get_inputs():
    # Generate two random vectors for addition
    a = torch.randn(vector_size, dtype=dtype, device='cuda')
    b = torch.randn(vector_size, dtype=dtype, device='cuda')
    return [a, b]

def get_init_inputs():
    return []
    """

    print("\nGenerating element-wise addition kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(problem_description, test_code=None)  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated element-wise addition kernel!")
        print(f"  Worker {result['worker_id']} found solution in {result['rounds']} rounds")
        print(f"  Session directory: {result['session_dir']}")

        print("\n" + "=" * 80)
        print("Generated Kernel Code:")
        print("=" * 80)
        print(result["kernel_code"])
        print("=" * 80)

        # Save the kernel to a file for future use
        kernel_file = "element_add_kernel.py"
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
    print("\n✓ Element-wise addition example completed!")


if __name__ == "__main__":
    main()