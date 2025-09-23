#!/usr/bin/env python3
"""
Example of using Triton Kernel Agent with additional reference code.

This demonstrates how to provide a reference implementation that the agent
will use to understand the algorithm. The generated Triton kernel should:
1. Be numerically correct (match the reference)
2. Be faster than PyTorch native operations
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_kernel_agent import TritonKernelAgent


def get_parser():
    parser = argparse.ArgumentParser(description="Generate and test a Triton kernel.")
    parser.add_argument("--working-dir", type=str)
    return parser


def get_inputs():
    base_dir = get_parser().parse_args().working_dir
    problem_description = None
    additional_code = None

    print(f"Working directory: {base_dir}")
    assert os.path.exists(base_dir), f"Directory {base_dir} does not exist"

    path = os.path.join(base_dir, "problem_description.md")
    assert os.path.exists(path), f"{path} does not exist"
    with open(os.path.join(base_dir, "problem_description.md"), "r") as f:
        problem_description = f.read()

    path = os.path.join(base_dir, "additional_code.py")
    if os.path.exists(path):
        with open(path, "r") as f:
            additional_code = f.read()

    return problem_description, additional_code


def main():
    """Example of generating a kernel with reference implementation."""

    # Get inputs
    (problem_description, additional_code) = get_inputs()

    print("Problem:", problem_description.strip()[:200] + "...")
    print(f"\nReference implementation provided: {'Yes' if additional_code else 'No'}")

    # Load environment variables
    load_dotenv()

    # Create agent
    print("Creating Triton Kernel Agent...")
    agent = TritonKernelAgent(num_workers=2, max_rounds=5)

    # Generate the kernel
    print("\nGenerating optimized Triton kernel...")
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
