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

"""rocprof wrapper script generation for ROCm kernel profiling."""

import logging
from functools import cached_property
from pathlib import Path

from jinja2 import Template


class ROCmWrapperFactory:
    """Factory for creating rocprof wrapper scripts for profiling Triton kernels on AMD GPUs.

    Mirrors :class:`triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory.NCUWrapperFactory`
    but targets ROCm / HIP instead of NVIDIA NCU.
    """

    WRAPPER_TEMPLATE = Path(__file__).parent / "rocprof_wrapper_template.j2"

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @cached_property
    def template(self) -> Template:
        if not self.WRAPPER_TEMPLATE.exists():
            raise FileNotFoundError(f"Template not found: {self.WRAPPER_TEMPLATE}")
        return Template(self.WRAPPER_TEMPLATE.read_text())

    def create_rocprof_wrapper(
        self,
        kernel_file: Path,
        problem_file: Path,
        output_dir: Path,
        dtype_inference: bool = True,
        model_extraction: bool = True,
    ) -> Path:
        """Create a rocprof wrapper script for profiling.

        Args:
            kernel_file: Path to kernel file.
            problem_file: Path to problem file.
            output_dir: Directory to write wrapper script.
            dtype_inference: Enable automatic dtype inference from kernel source.
            model_extraction: Enable model weight extraction for Conv/Linear kernels.

        Returns:
            Path to created wrapper script.
        """
        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")
        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        output_dir.mkdir(parents=True, exist_ok=True)

        wrapper_file = output_dir / "rocprof_wrapper.py"
        wrapper_content = self.template.render(
            kernel_file_parent=repr(str(kernel_file.parent)),
            problem_file_parent=repr(str(problem_file.parent)),
            kernel_module=kernel_file.stem,
            problem_module=problem_file.stem,
            dtype_inference=dtype_inference,
            model_extraction=model_extraction,
        )

        wrapper_file.write_text(wrapper_content)
        self.logger.info(f"Created rocprof wrapper: {wrapper_file}")
        return wrapper_file
