"""NCU wrapper script generation for kernel profiling."""

import logging
from pathlib import Path
from typing import Optional

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class NCUWrapperGenerator:
    """Generates NCU wrapper scripts for profiling Triton kernels."""

    # Template file path (relative to this file)
    WRAPPER_TEMPLATE = Path(__file__).parent / "ncu_wrapper_template.j2"

    def __init__(self, logger: logging.Logger):
        """
        Initialize the NCU wrapper generator.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self._template_cache: Optional[Template] = None

    def _load_template(self) -> Template:
        """
        Load the Jinja2 template (cached).

        Returns:
            Jinja2 Template object

        Raises:
            ImportError: If Jinja2 is not installed
            FileNotFoundError: If template file doesn't exist
        """
        if self._template_cache is not None:
            return self._template_cache

        if not HAS_JINJA2:
            raise ImportError(
                "Jinja2 is required for wrapper generation. "
                "Install it with: pip install jinja2"
            )

        if not self.WRAPPER_TEMPLATE.exists():
            raise FileNotFoundError(f"Template not found: {self.WRAPPER_TEMPLATE}")

        self._template_cache = Template(self.WRAPPER_TEMPLATE.read_text())
        return self._template_cache

    def create_ncu_wrapper(
        self,
        kernel_file: Path,
        problem_file: Path,
        output_dir: Path,
        dtype_inference: bool = True,
        model_extraction: bool = True,
        use_cache: bool = True,
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
            use_cache: Reuse existing wrapper if files haven't changed (default: True)

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

        # Check cache: reuse wrapper if it's fresh
        if use_cache and wrapper_file.exists():
            wrapper_mtime = wrapper_file.stat().st_mtime
            kernel_mtime = kernel_file.stat().st_mtime
            problem_mtime = problem_file.stat().st_mtime

            if wrapper_mtime > kernel_mtime and wrapper_mtime > problem_mtime:
                self.logger.info(
                    f"Reusing cached NCU wrapper (fresher than source files): {wrapper_file}"
                )
                return wrapper_file

        # Load template and render
        template = self._load_template()
        wrapper_content = template.render(
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
