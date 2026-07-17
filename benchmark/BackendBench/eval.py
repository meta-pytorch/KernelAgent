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

"""
Evaluation script for KernelAgent using BackendBench infrastructure with a two-phase workflow:

1. Kernel-generation Phase: Generate kernels using KernelAgent → save to generated_kernels/
2. Evaluation Phase: Run BackendBench main.py with DirectoryBackend to evaluate

"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

import torch
from BackendBench.suite import OpInfoTestSuite
from BackendBench.utils import extract_operator_name, op_name_to_folder_name
from triton_kernel_agent import TritonKernelAgent


def generate_kernels(
    suite_name: str = "torchbench",
    num_operators: int = None,
    num_workers: int = 4,
    max_rounds: int = 10,
    workflow: str = "base",
    verbose: bool = False,
    test_cases: bool = True,
) -> str:
    """
    Phase 1: Generate kernels using TritonKernelAgent.

    Args:
        suite_name: Test suite name (opinfo/smoke/torchbench)
        num_operators: Number of operators to generate (None = all)
        num_workers: Number of parallel workers for KernelAgent
        max_rounds: Max refinement rounds per worker
        verbose: Verbose logging

    Returns:
        Path to generated_kernels directory
    """
    logger = logging.getLogger("generate_kernels")
    logger.info("Phase 1: Generating kernels with TritonKernelAgent")

    # Use standard BackendBench directory structure: generated_kernels/
    kernels_dir = "generated_kernels"
    os.makedirs(kernels_dir, exist_ok=True)
    logger.info(f"Kernels will be saved to: {kernels_dir}/<op_name>/")

    # Initialize TritonKernelAgent with timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_log_dir = f"agent_logs/run_{timestamp}"
    os.makedirs(agent_log_dir, exist_ok=True)

    agent = TritonKernelAgent(
        log_dir=agent_log_dir,
        num_workers=num_workers,
        max_rounds=max_rounds,
    )
    logger.info(
        f"TritonKernelAgent initialized (workers={num_workers}, max_rounds={max_rounds})"
    )
    logger.info(f"Agent logs: {agent_log_dir}")

    # Create test suite
    if suite_name == "opinfo":
        test_suite = OpInfoTestSuite("opinfo", "cuda", torch.bfloat16)
    elif suite_name == "smoke":
        from BackendBench.suite import SmokeTestSuite

        test_suite = SmokeTestSuite("smoke", "cuda")
    elif suite_name == "torchbench":
        from BackendBench.suite import TorchBenchTestSuite

        test_suite = TorchBenchTestSuite("torchbench", None)
    else:
        raise ValueError(f"Unknown suite: {suite_name}")

    # Get operators to generate
    operators = list(test_suite)
    if num_operators:
        operators = operators[:num_operators]

    logger.info(f"Generating kernels for {len(operators)} operators")

    # Generate kernels
    successful = 0
    for idx, optest in enumerate(operators, 1):
        op = optest.op
        op_name = extract_operator_name(str(op))

        logger.info(f"[{idx}/{len(operators)}] Generating {op_name}")

        try:
            # Create problem description for the operator
            folder_name = op_name_to_folder_name(op_name)
            problem_description = _create_problem_description_from_op(op, op_name)

            # Create test code from BackendBench tests if provided
            test_code = None
            if test_cases:
                test_code = _create_test_code_from_backendbench(
                    op=op,
                    op_name=op_name,
                    test_cases=optest.correctness_tests,
                    logger=logger,
                )

            # Generate kernel using TritonKernelAgent
            result = agent.generate_kernel(
                problem_description=problem_description,
                test_code=test_code,
            )

            if result["success"]:
                kernel_code = result["kernel_code"]

                # Automatically fix function name to match BackendBench's expectations
                # Replace generic function names with the required name
                import re

                expected_func_name = f"{folder_name}_kernel_impl"
                kernel_code = re.sub(
                    r"\bdef\s+(kernel_function)\s*\(",
                    f"def {expected_func_name}(",
                    kernel_code,
                )
                logger.debug(f"    Ensured function name is: {expected_func_name}")

                # Create operator directory (e.g., generated_kernels/abs__default/)
                folder_name = op_name_to_folder_name(op_name)
                op_dir = os.path.join(kernels_dir, folder_name)
                os.makedirs(op_dir, exist_ok=True)

                # Save kernel with DirectoryBackend's expected naming: {op_name}_implementation.py
                kernel_file = os.path.join(op_dir, f"{folder_name}_implementation.py")
                with open(kernel_file, "w") as f:
                    f.write(kernel_code)

                successful += 1
                logger.info(
                    f"  ✓ Success ({successful}/{idx}) - saved to {folder_name}/{folder_name}_implementation.py"
                )
            else:
                logger.warning(f"  ✗ Failed: {result.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")

    logger.info(f"\nGeneration complete: {successful}/{len(operators)} successful")
    logger.info(f"Kernels saved to: {kernels_dir}")

    return kernels_dir


def evaluate_kernels(
    kernels_dir: str,
    suite_name: str = "opinfo",
    verbose: bool = False,
) -> int:
    """
    Phase 2: Evaluate kernels using BackendBench's main.py script.

    Args:
        kernels_dir: Path to generated_kernels directory
        suite_name: Test suite name
        verbose: Verbose logging

    Returns:
        Exit code from BackendBench evaluation
    """
    logger = logging.getLogger("evaluate_kernels")
    logger.info("Phase 2: Evaluating kernels with BackendBench")
    logger.info(f"Loading kernels from: {kernels_dir}")

    # Create separate timestamped log directory for BackendBench evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log_BackendBench/run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {log_dir}")

    # Build command to run BackendBench main.py
    cmd = [
        sys.executable,
        "-m",
        "BackendBench.scripts.main",
        "--backend",
        "directory",
        "--suite",
        suite_name,
        "--ops-directory",
        kernels_dir,
        "--log-dir",
        log_dir,  # Save evaluation results to separate log directory
    ]

    if verbose:
        cmd.append("--log-level=DEBUG")

    logger.info(f"Running: {' '.join(cmd)}")

    # Run BackendBench evaluation
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        logger.info(f"\nEvaluation results saved to: {log_dir}/")
        logger.info(f"  - {log_dir}/OVERALL_SUMMARY.md")
        logger.info(f"  - {log_dir}/operator_summary.csv")
        logger.info(f"  - {log_dir}/full_results.json")

    return result.returncode


def _create_problem_description_from_op(op, op_name: str) -> str:
    """
    Create a problem description for KernelAgent based on the PyTorch operation.

    Args:
        op: PyTorch operation
        op_name: Operation name extracted from op

    Returns:
        Problem description string for KernelAgent
    """
    # Create a comprehensive problem description that KernelAgent can understand
    problem_description = f"""
Task: Implement a high-performance Triton kernel for the PyTorch operation: {op_name}

Requirements:
1. The kernel must be functionally equivalent to the PyTorch operation
2. Implement using Triton language primitives (tl.load, tl.store, etc.)
3. Handle all tensor shapes and data types that the original operation supports
4. Optimize for GPU performance with proper memory coalescing
5. Include proper boundary condition handling
6. Follow Triton best practices for kernel design



The generated kernel should:
- Take the same input arguments as the PyTorch operation
- Return outputs with identical shapes, dtypes, and numerical values
- Be optimized for common tensor shapes and memory layouts
- Handle edge cases gracefully

Please generate a complete, production-ready Triton kernel implementation.
"""
    return problem_description


def _create_test_code_from_backendbench(op, op_name: str, test_cases, logger) -> str:
    """
    Convert BackendBench test cases to KernelAgent-compatible test code.
    Args:
        op: PyTorch operation
        op_name: Operation name
        test_cases: BackendBench test cases
    Returns:
        Test code string for KernelAgent, or None if no test cases
    """
    test_list = list(test_cases) if test_cases else []
    if not test_list:
        return None

    logger.debug(f"    Using {len(test_list)} BackendBench test cases")

    # Use a few representative test cases (not all, to avoid overwhelming the LLM)
    max_tests = min(5, len(test_list))

    # Import the serialization utility
    from BackendBench.utils import serialize_args

    test_code = f'''import torch
import torch.nn.functional as F
import re
def _deserialize_tensor(match):
    """Convert T([shape], dtype) to appropriate torch tensor creation"""
    # Parse the T(...) format
    content = match.group(1)
    parts = [p.strip() for p in content.split(', ')]

    # Extract shape (first part)
    shape_str = parts[0]

    # Extract dtype (second part)
    dtype_str = parts[1]

    # Handle stride if present (third part)
    # For now, we ignore stride and create contiguous tensors

    # Convert dtype abbreviations to torch dtypes
    dtype_map = {{
        'bf16': 'torch.bfloat16',
        'f64': 'torch.float64',
        'f32': 'torch.float32',
        'f16': 'torch.float16',
        'c32': 'torch.complex32',
        'c64': 'torch.complex64',
        'c128': 'torch.complex128',
        'i8': 'torch.int8',
        'i16': 'torch.int16',
        'i32': 'torch.int32',
        'i64': 'torch.int64',
        'b8': 'torch.bool',
        'u8': 'torch.uint8',
    }}

    torch_dtype = dtype_map.get(dtype_str, 'torch.float32')

    # Choose appropriate tensor creation based on dtype
    if dtype_str in ['b8']:  # Boolean
        return f"torch.randint(0, 2, {{shape_str}}, dtype={{torch_dtype}}, device='cuda').bool()"
    elif dtype_str in ['i8', 'i16', 'i32', 'i64', 'u8']:  # Integer types
        return f"torch.randint(0, 10, {{shape_str}}, dtype={{torch_dtype}}, device='cuda')"
    elif dtype_str in ['c32', 'c64', 'c128']:  # Complex types
        return f"torch.randn({{shape_str}}, dtype={{torch_dtype}}, device='cuda')"
    else:  # Float types
        return f"torch.randn({{shape_str}}, dtype={{torch_dtype}}, device='cuda')"
def deserialize_test_args(serialized_str):
    """Convert serialized args string to actual args and kwargs"""
    # Replace T(...) with torch.randn(...)
    pattern = r'T\(([^)]+)\)'
    deserialized = re.sub(pattern, _deserialize_tensor, serialized_str)

    # The serialized format is: (args_tuple, kwargs_dict)
    # Evaluate to get the tuple
    full_data = eval(deserialized)

    # Extract args and kwargs
    if isinstance(full_data, tuple) and len(full_data) == 2:
        args, kwargs = full_data
        return list(args), kwargs
    else:
        # Handle case where there's only args
        return list(full_data), {{}}
def test_kernel():
    """Test the {op_name} kernel using BackendBench test cases."""
    from kernel import kernel_function

    all_passed = True
    failed_tests = []

'''

    for i, test in enumerate(test_list[:max_tests]):
        # Use BackendBench's serialization format
        serialized_args = serialize_args(test.args, test.kwargs)

        test_code += f"    # Test case {i + 1} from BackendBench\n"
        test_code += "    try:\n"
        test_code += "        # Deserialize the test arguments\n"
        test_code += f'        serialized = """{serialized_args}"""\n'
        test_code += "        args, kwargs = deserialize_test_args(serialized)\n"

        # Test execution
        op_str = str(op).replace("OpOverload", "").replace("OpOverloadPacket", "")
        test_code += f"""
        # Get reference result from PyTorch
        ref_result = torch.ops.{op_str}(*args, **kwargs)

        # Get result from our kernel
        kernel_result = kernel_function(*args, **kwargs)

        # Compare results
        torch.testing.assert_close(ref_result, kernel_result, rtol=1e-2, atol=1e-2)
        print(f"Test case {i + 1} passed!")

    except Exception as e:
        print(f"Test case {i + 1} failed: {{e}}")
        failed_tests.append({i + 1})
        all_passed = False
"""

    test_code += """
    if all_passed:
        print("All BackendBench tests passed!")
    else:
        print(f"Failed tests: {failed_tests}")

    return all_passed
if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
"""

    return test_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate KernelAgent using BackendBench infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and evaluate 5 operators
  python eval.py --num-operators 5

  # Generate and evaluate all operators
  python eval.py

  # Only generate (no evaluation)
  python eval.py --generate-only --num-operators 10

  # Only evaluate (use existing kernels)
  python eval.py --evaluate-only --kernels-dir generated_kernels/kernel_agent_run_20241106_123456
        """,
    )

    parser.add_argument(
        "--suite",
        choices=["opinfo", "smoke", "torchbench"],
        default="opinfo",
        help="Test suite to use (opinfo=correctness only, torchbench=correctness+performance)",
    )
    parser.add_argument(
        "--num-operators",
        type=int,
        default=None,
        help="Number of operators to generate (None = all)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for KernelAgent",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum refinement rounds for KernelAgent",
    )
    parser.add_argument(
        "--workflows",
        choices=["base", "fuser"],
        default="base",
        help="Workflow type: 'base' (KernelAgent only) or 'fuser' (KernelAgent + Fuser)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate kernels (no evaluation)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate (use existing kernels)",
    )
    parser.add_argument(
        "--kernels-dir",
        type=str,
        default=None,
        help="Directory with generated kernels (for --evaluate-only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("main")
    logger.info("Starting KernelAgent evaluation")

    # Validate arguments
    if args.evaluate_only and not args.kernels_dir:
        logger.error("--evaluate-only requires --kernels-dir")
        return 1

    if args.generate_only and args.evaluate_only:
        logger.error("Cannot use both --generate-only and --evaluate-only")
        return 1

    try:
        kernels_dir = args.kernels_dir

        # Phase 1: Generate kernels
        if not args.evaluate_only:
            kernels_dir = generate_kernels(
                suite_name=args.suite,
                num_operators=args.num_operators,
                num_workers=args.num_workers,
                max_rounds=args.max_rounds,
                workflow=args.workflows,
                verbose=args.verbose,
            )

            if args.generate_only:
                logger.info(f"Generation complete. Kernels saved to: {kernels_dir}")
                return 0

        # Phase 2: Evaluate kernels
        if not args.generate_only:
            exit_code = evaluate_kernels(
                kernels_dir=kernels_dir,
                suite_name=args.suite,
                verbose=args.verbose,
            )

            if exit_code == 0:
                logger.info("Evaluation complete!")
                logger.info(f"Results saved to: {kernels_dir}")
            else:
                logger.error(f"Evaluation failed with exit code {exit_code}")

            return exit_code

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
