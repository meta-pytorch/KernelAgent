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
Call the setup script to create directory structure for PyTorch operators in op_map.
This creates directories for operators that are actually used in evaluation suites
(opinfo, torchbench).

This wrapper calls BackendBench's setup_operator_directories and creates
a dedicated folder for generated operator kernels.
"""

import argparse
from pathlib import Path


def setup_backendbench_operators(
    base_dir: str = "generated_kernels", verbose: bool = False, suite: str = "all"
):
    """
    Setup operator directories for BackendBench integration.

    Creates a new directory structure for storing generated kernels.

    Args:
        base_dir: Base directory name for operator implementations (default: "generated_kernels")
                  Will be created relative to current working directory
        verbose: Show verbose output for each directory created/skipped
        suite: Which operators to include ('torchbench', 'opinfo', 'all')
    """
    try:
        from BackendBench.scripts.setup_operator_directories import (
            setup_operator_directories,
        )

        # Convert to absolute path and create base directory
        abs_base_dir = Path(base_dir).resolve()

        # Show message if using default directory
        if base_dir == "generated_kernels":
            print(f"ℹ️  Using default directory: {abs_base_dir}")
            print("   (Specify --base-dir to use a different location)\n")

        abs_base_dir.mkdir(parents=True, exist_ok=True)

        print("Setting up BackendBench operator directories...")
        print(f"Base directory: {abs_base_dir}")
        print(f"Suite: {suite}")
        print()

        setup_operator_directories(
            base_dir=str(abs_base_dir), verbose=verbose, suite=suite
        )

        print("\n✓ BackendBench setup complete!")
        print(f"✓ Operator directories created in: {abs_base_dir}")

    except ImportError as e:
        print("✗ Error: Could not import BackendBench")
        print(f"  {e}")
        print("\nMake sure BackendBench is installed:")
        print("  pip install git+ssh://git@github.com/meta-pytorch/BackendBench.git")
        return False
    except Exception as e:
        print(f"✗ Error during setup: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Command-line interface for BackendBench setup."""
    parser = argparse.ArgumentParser(
        description="Setup BackendBench operator directories for KernelAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default 'generated_kernels' folder with all operators
  python setup.py

  # Create custom folder with opinfo operators only
  python setup.py --base-dir my_kernels --suite opinfo

  # Use absolute path
  python setup.py --base-dir /home/user/kernels --suite all --verbose
        """,
    )
    parser.add_argument(
        "--base-dir",
        default="generated_kernels",
        help="Base directory name for operator implementations (default: generated_kernels)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output for each directory created/skipped",
    )
    parser.add_argument(
        "--suite",
        choices=["torchbench", "opinfo", "all"],
        default="torchbench",
        help="Which test suite operators to include (default: torchbench)",
    )

    args = parser.parse_args()

    success = setup_backendbench_operators(
        base_dir=args.base_dir, verbose=args.verbose, suite=args.suite
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
