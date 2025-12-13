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

"""Prompt Manager for handling Jinja2 templates."""

from pathlib import Path
from typing import Dict, Optional
from triton_kernel_agent.platform_config import PlatformConfig, get_platform

try:
    from jinja2 import Environment, FileSystemLoader, Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    FileSystemLoader = None
    Template = None


class PromptManager:
    """
    Manages Jinja2 templates for prompt generation.

    This class provides a centralized way to load and render templates
    for test generation, kernel generation, and kernel refinement.
    """

    def __init__(
        self,
        templates_dir: Optional[str] = None,
        target_platform: Optional[PlatformConfig] = None,
    ):
        """
        Initialize the prompt manager.

        Args:
            templates_dir: Path to the templates directory. If None, uses default.
            target_platform: Target platform PlatformConfig
        """
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "Jinja2 is not available. Please install it with: pip install jinja2"
            )

        if target_platform is None:
            target_platform = get_platform("cuda")
        self.target_platform = target_platform
        # Set up templates directory
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            # Default to bundled templates directory within the package
            self.templates_dir = Path(__file__).parent / "templates"

        if not self.templates_dir.exists():
            raise FileNotFoundError(
                f"Templates directory not found: {self.templates_dir}"
            )

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load templates
        self._load_templates()

    def _load_templates(self):
        """Load all available templates."""
        self.templates = {}

        # Define template mappings
        template_files = {
            "test_generation": "test_generation.j2",
            "kernel_generation": "kernel_generation.j2",
            "kernel_refinement": "kernel_refinement.j2",
            "triton_guidelines": "triton_guidelines.j2",
        }

        # Load each template
        for template_name, template_file in template_files.items():
            template_path = self.templates_dir / template_file
            if template_path.exists():
                self.templates[template_name] = self.env.get_template(template_file)
            else:
                raise FileNotFoundError(f"Template file not found: {template_path}")

    def render_test_generation_prompt(
        self, problem_description: str, provided_test_code: Optional[str] = None
    ) -> str:
        """
        Render the test generation prompt.

        Args:
            problem_description: Description of the problem to generate tests for
            provided_test_code: Optional reference test code provided by user

        Returns:
            Rendered prompt string
        """
        template = self.templates["test_generation"]
        return template.render(
            problem_description=problem_description,
            provided_test_code=provided_test_code,
            device_string=self.target_platform.device_string,
        )

    def render_kernel_generation_prompt(
        self,
        problem_description: str,
        test_code: str,
        triton_guidelines: Optional[str] = None,
    ) -> str:
        """
        Render the kernel generation prompt.

        Args:
            problem_description: Description of the kernel to generate
            test_code: Test code that the kernel must pass
            triton_guidelines: Optional guidelines (if None, loads from template)

        Returns:
            Rendered prompt string
        """
        template = self.templates["kernel_generation"]

        # Load triton guidelines if not provided
        if triton_guidelines is None:
            triton_guidelines = self.render_triton_guidelines()

        return template.render(
            problem_description=problem_description,
            test_code=test_code,
            triton_guidelines=triton_guidelines,
            kernel_guidance=self.target_platform.kernel_guidance,
        )

    def render_kernel_refinement_prompt(
        self,
        problem_description: str,
        test_code: str,
        kernel_code: str,
        error_info: Dict[str, str],
        history_context: Optional[str] = None,
        triton_guidelines: Optional[str] = None,
    ) -> str:
        """
        Render the kernel refinement prompt.

        Args:
            problem_description: Description of the problem
            test_code: Test code that the kernel must pass
            kernel_code: Current kernel implementation
            error_info: Dictionary with error information (stdout, stderr)
            history_context: Optional context from previous attempts
            triton_guidelines: Optional guidelines (if None, loads from template)

        Returns:
            Rendered prompt string
        """
        template = self.templates["kernel_refinement"]

        # Load triton guidelines if not provided
        if triton_guidelines is None:
            triton_guidelines = self.render_triton_guidelines()

        return template.render(
            problem_description=problem_description,
            test_code=test_code,
            kernel_code=kernel_code,
            error_info=error_info,
            history_context=history_context,
            triton_guidelines=triton_guidelines,
            kernel_guidance=self.target_platform.kernel_guidance,
        )

    def render_triton_guidelines(self) -> str:
        """
        Render the Triton guidelines.

        Returns:
            Rendered guidelines string
        """
        template = self.templates["triton_guidelines"]
        return template.render()

    def get_template(self, template_name: str) -> Template:
        """
        Get a template by name for custom rendering.

        Args:
            template_name: Name of the template to get

        Returns:
            Jinja2 Template object
        """
        if template_name not in self.templates:
            raise ValueError(
                f"Template '{template_name}' not found. Available: {list(self.templates.keys())}"
            )

        return self.templates[template_name]

    def render_custom_template(self, template_name: str, **kwargs) -> str:
        """
        Render a custom template with provided variables.

        Args:
            template_name: Name of the template to render
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template string
        """
        template = self.get_template(template_name)
        return template.render(**kwargs)

    def list_templates(self) -> list:
        """
        List all available templates.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def add_template(self, template_name: str, template_content: str):
        """
        Add a new template from string content.

        Args:
            template_name: Name for the new template
            template_content: Jinja2 template content
        """
        template = self.env.from_string(template_content)
        self.templates[template_name] = template

    def reload_templates(self):
        """Reload all templates from disk."""
        self._load_templates()
