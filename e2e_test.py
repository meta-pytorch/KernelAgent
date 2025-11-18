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

"""End-to-end BF16 matmul+sigmoid test harness."""

from pathlib import Path
import sys
import time
from dotenv import load_dotenv
from hydra import main as hydra_main
from omegaconf import DictConfig
from triton_kernel_agent import TritonKernelAgent


@hydra_main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "configs"),
    config_name="e2e_test",
)
def main(cfg: DictConfig) -> None:
    """Generate and test a BF16 matmul kernel with fused sigmoid activation."""
    # Load environment
    load_dotenv()

    # Create agent with config parameters
    agent = TritonKernelAgent(
        num_workers=cfg.num_workers,
        max_rounds=cfg.max_rounds,
        log_dir=cfg.log_dir,
        model_name=cfg.model_name,
        high_reasoning_effort=cfg.high_reasoning_effort,
    )

    print("=" * 80)
    print("BF16 Matmul with Fused Sigmoid Activation")
    print("Matrix dimensions: M=1024, N=2058, K=4096")
    print("=" * 80)

    # Define the problem
    problem_description = """
Write a fused Triton kernel for the following problem:

import torch
import torch.nn as nn

class Model(nn.Module):
def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.bfloat16))

    def forward(self, x):
        # Perform matmul and apply sigmoid activation
        output = torch.matmul(x, self.weight)
        output = torch.sigmoid(output)
        return output

# Define input dimensions and parameters
batch_size = 1024
in_features = 4096
out_features = 2058

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.bfloat16)]

def get_init_inputs():
    return [in_features, out_features]
    """

    # Let the agent generate the test code
    print("\nGenerating kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(
        problem_description, test_code=None
    )  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated BF16 matmul + sigmoid kernel!")
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
        kernel_file = "bf16_matmul_sigmoid_kernel.py"
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
        sys.exit(1)

    # Cleanup
    agent.cleanup()
    print("\n✓ E2E test completed successfully!")


if __name__ == "__main__":
    main()
