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

"""NCU wrapper script generation for kernel profiling."""

import logging
from functools import cached_property
from pathlib import Path

from jinja2 import Template


class NCUWrapperFactory:
    """Factory for creating NCU wrapper scripts for profiling Triton kernels."""

    # Template file path (relative to this file)
    WRAPPER_TEMPLATE = Path(__file__).parent / "ncu_wrapper_template.j2"

    def __init__(self, logger: logging.Logger):
        """
        Initialize the NCU wrapper factory.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    @cached_property
    def template(self) -> Template:
        """
        Jinja2 template for wrapper script generation.

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if not self.WRAPPER_TEMPLATE.exists():
            raise FileNotFoundError(f"Template not found: {self.WRAPPER_TEMPLATE}")
        return Template(self.WRAPPER_TEMPLATE.read_text())

    def create_ncu_wrapper(
        self,
        kernel_file: Path,
        problem_file: Path,
        output_dir: Path,
        dtype_inference: bool = True,
        model_extraction: bool = True,
    ) -> Path:
        """
        Create NCU wrapper script for profiling.

        The wrapper handles multiple kernel types:
        - Standard kernels: kernel_function(*inputs)
        - Conv/Linear kernels: Extracts weights from Model
        - RMSNorm kernels: Passes init_inputs (features, eps)

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            output_dir: Directory to write wrapper script
            dtype_inference: Enable automatic dtype inference from kernel source (default: True)
            model_extraction: Enable model weight extraction for Conv/Linear kernels (default: True)

        Returns:
            Path to created wrapper script

        Raises:
            FileNotFoundError: If kernel_file or problem_file doesn't exist
            OSError: If output_dir is not writable
        """
        # Validate inputs
        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")
        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        # Ensure output directory exists
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        wrapper_file = output_dir / "ncu_wrapper.py"

        # Render template
        wrapper_content = self.template.render(
            kernel_file_parent=repr(str(kernel_file.parent)),
            problem_file_parent=repr(str(problem_file.parent)),
            kernel_module=kernel_file.stem,
            problem_module=problem_file.stem,
            dtype_inference=dtype_inference,
            model_extraction=model_extraction,
        )

        # Write wrapper file
        wrapper_file.write_text(wrapper_content)
        self.logger.info(f"Created NCU wrapper: {wrapper_file}")
        return wrapper_file
