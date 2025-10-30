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

"""Fused dropout-residual operation example for KernelAgent."""

import argparse
import sys
import time
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from triton_kernel_agent import TritonKernelAgent


def main():
    """Generate and test a fused dropout-residual kernel."""

    # Load additional code if working directory provided
    parser = argparse.ArgumentParser(
        description="Generate fused dropout-residual kernel with optional additional code."
    )
    parser.add_argument(
        "--additional-code", type=str, help="Path to additional reference code file"
    )
    args = parser.parse_args()

    additional_code = None
    if args.additional_code:
        if not os.path.exists(args.additional_code):
            print(f"Error: File {args.additional_code} does not exist")
            sys.exit(1)

        try:
            with open(args.additional_code, "r") as f:
                additional_code = f.read()
            print(f"Loaded additional code from: {args.additional_code}")
        except Exception as e:
            print(f"Error: Failed to read additional code: {e}")
            sys.exit(1)
    else:
        print("No --additional-code provided, skipping additional_code")

    # Load environment
    load_dotenv()

    # Create agent
    agent = TritonKernelAgent()

    print("=" * 80)
    print("Fused dropout-residual Operation Kernel Generation")
    print("Dropout rate: 0.1")
    print(
        "Input tensors shape: batch size: 32, sequence length: 2048, hidden dimension: 4096"
    )
    print("Operation: dropout + residual")
    print("=" * 80)

    # Define the problem
    problem_description = """
Create a Triton kernel for fused dropout-residual addition.

The kernel should:
1. Apply dropout to input tensor with probability p
2. Add the result to a residual tensor
3. Return the final output

Input tensors shape: batch size: 32, sequence length: 2048, hidden dimension: 4096

# NOTE: You should only write ONE kernel to do the entire operation, instead of writing multiple kernels
    """
    print("\nGenerating fused dot-compress kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(
        problem_description, test_code=None, additional_code=additional_code
    )  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated fused dropout-residual kernel!")
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
        kernel_file = "fused_dropout_residual_kernel.py"
        with open(kernel_file, "w") as f:
            f.write(result["kernel_code"])
        print(f"\n✓ Kernel saved to: {kernel_file}")

        # Run the generated test to show performance
        print("\nRunning the generated test...")

        # Read the generated test code
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
            problem_file = os.path.join(result["session_dir"], "problem.txt")
            if os.path.exists(problem_file):
                with open(problem_file, "r") as f:
                    print(f"\n  Problem attempted:\n{f.read()}")

        sys.exit(1)

    # Cleanup
    agent.cleanup()
    print("\n✓ Fused dropout_residual example completed!")


if __name__ == "__main__":
    main()
