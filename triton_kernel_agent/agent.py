"""
Main Triton Kernel Generation Agent.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from dotenv import load_dotenv

from .manager import WorkerManager
from .prompt_manager import PromptManager
from .providers import get_model_provider


class TritonKernelAgent:
    """Main agent for generating and optimizing Triton kernels."""

    def __init__(
        self,
        num_workers: Optional[int] = None,
        max_rounds: Optional[int] = None,
        log_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        high_reasoning_effort: bool = True,
    ):
        """
        Initialize the Triton Kernel Agent.

        Args:
            num_workers: Number of parallel workers for verification (loaded from .env if None)
            max_rounds: Maximum refinement rounds per worker (loaded from .env if None)
            log_dir: Directory for logs (creates temp if None)
            model_name: OpenAI model to use (loaded from .env if None)
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
        """
        # Load environment variables
        load_dotenv()

        # Load configuration from environment
        self.num_workers = num_workers or int(os.getenv("NUM_KERNEL_SEEDS", "4"))
        self.max_rounds = max_rounds or int(os.getenv("MAX_REFINEMENT_ROUNDS", "10"))
        self.model_name = model_name or os.getenv(
            "OPENAI_MODEL", "claude-sonnet-4-20250514"
        )
        self.high_reasoning_effort = high_reasoning_effort

        # Initialize provider
        self.provider = None
        try:
            self.provider = get_model_provider(self.model_name)
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(
                f"Initialized provider '{self.provider.name}' for model '{self.model_name}'"
            )
        except ValueError as e:
            # Will be handled in setup_logging, just store the error for now
            self._provider_error = str(e)

        # Setup logging
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path.cwd() / "triton_kernel_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Setup main logger
        self._setup_logging()

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize worker manager
        self.manager = WorkerManager(
            num_workers=self.num_workers,
            max_rounds=self.max_rounds,
            log_dir=self.log_dir,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=self.model_name,
            high_reasoning_effort=self.high_reasoning_effort,
        )

    def _setup_logging(self):
        """Setup agent logging."""
        log_file = (
            self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("TritonKernelAgent")

    def _extract_code_from_response(
        self, response_text: str, language: str = "python"
    ) -> Optional[str]:
        """
        Extract code from LLM response text.

        Args:
            response_text: The full LLM response text
            language: The expected language (default: python)

        Returns:
            Extracted code or None if no valid code block found
        """
        if not response_text:
            return None

        # First, try to find code blocks with language markers
        # Pattern matches ```python or ```language_name
        pattern = rf"```{language}\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Return the first match (largest code block)
            return matches[0].strip()

        # Try generic code blocks without language marker
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Return the first match
            return matches[0].strip()

        # If no code blocks found, check if the entire response looks like code
        # This is a fallback for cases where LLM doesn't use code blocks
        lines = response_text.strip().split("\n")

        # Simple heuristic: if response contains import statements or function definitions
        code_indicators = ["import ", "from ", "def ", "class ", "@", '"""', "'''"]
        if any(
            line.strip().startswith(indicator)
            for line in lines
            for indicator in code_indicators
        ):
            # Likely the entire response is code
            return response_text.strip()

        # No code found
        self.logger.warning("No code block found in LLM response")
        return None

    def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.model_name}")

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.model_name, messages, **kwargs)
        return response.content

    def _generate_test(
        self, problem_description: str, provided_test_code: Optional[str] = None
    ) -> str:
        """
        Generate test code for the problem using OpenAI API.

        The test must import from 'kernel' module since each worker writes
        the kernel to 'kernel.py' in their working directory.

        Args:
            problem_description: Description of the problem
            provided_test_code: Optional reference test code provided by user

        Returns:
            Generated test code in standardized format
        """
        # Use LLM provider if available; no mock fallback allowed
        if not self.provider:
            raise RuntimeError(
                "Unable to generate test code: no LLM provider available and mock fallback disabled"
            )
        # Use LLM provider if available
        if self.provider:
            try:
                self.logger.info(f"Generating test code using {self.model_name}")

                # Create prompt for test generation using template
                prompt = self.prompt_manager.render_test_generation_prompt(
                    problem_description=problem_description,
                    provided_test_code=provided_test_code,
                )

                # Call LLM API
                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_llm(messages, max_tokens=24000)
                self.logger.info("Raw test generation response:\n%s", response_text)

                # Extract test code from response
                test_code = self._extract_code_from_response(response_text)

                if test_code:
                    self.logger.info(
                        f"Successfully generated test code using {self.model_name}"
                    )
                    return test_code
                else:
                    self.logger.error("Failed to extract valid code from LLM response")
                    raise ValueError("No valid code found in LLM response")

            except Exception as e:
                self.logger.error(f"Error generating test with LLM API: {e}")
                raise

        # Mock test generation (fallback)
        self.logger.info("Generating test code (mock implementation)")

        # If provided test code exists, create a basic wrapper
        if provided_test_code:
            test_code = '''"""
Test for kernel implementation (adapted from provided test).
"""
import torch

def test_kernel():
    """Test the kernel implementation."""
    from kernel import kernel_function

    # Adapted from provided test code
    try:
        # Create test data (standardized format)
        test_input = torch.randn(1024, device='cuda')

        # Call kernel_function as a normal Python function
        result = kernel_function(test_input)

        # Basic validation
        if result is not None:
            print("Test passed!")
            return True
        else:
            print("Test failed: No result returned")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
'''
        else:
            test_code = '''"""
Test for kernel implementation.
"""
import torch

def test_kernel():
    """Test the kernel implementation."""
    from kernel import kernel_function

    # Mock test - replace with actual test logic
    try:
        # Create test data
        test_input = torch.randn(1024, device='cuda')

        # Call kernel_function as a normal Python function
        # (kernel launch logic is handled inside kernel.py)
        result = kernel_function(test_input)

        print("Test passed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
'''
        return test_code

    def _generate_kernel_seeds(
        self, problem_description: str, test_code: str, num_seeds: Optional[int] = None
    ) -> List[str]:
        """
        Generate initial kernel implementations using OpenAI API.

        Args:
            problem_description: Description of the kernel to generate
            test_code: Test code that the kernel must pass
            num_seeds: Number of kernel variations to generate

        Returns:
            List of kernel implementation strings
        """
        if num_seeds is None:
            num_seeds = self.num_workers

        # Use LLM provider if available
        if self.provider:
            try:
                self.logger.info(
                    f"Generating {num_seeds} kernel seeds using {self.model_name}"
                )

                # Create prompt with Triton guidelines using template
                prompt = self.prompt_manager.render_kernel_generation_prompt(
                    problem_description=problem_description, test_code=test_code
                )

                kernels = []
                messages = [{"role": "user", "content": prompt}]

                # Use provider's multiple response capability
                max_completion_tokens = 20000

                if self.provider.supports_multiple_completions():
                    # Provider supports native multiple completions
                    responses = self.provider.get_multiple_responses(
                        self.model_name,
                        messages,
                        n=num_seeds,
                        temperature=0.8,
                        max_tokens=max_completion_tokens,
                        high_reasoning_effort=self.high_reasoning_effort,
                    )

                    for i, response in enumerate(responses):
                        kernel_code = self._extract_code_from_response(response.content)
                        if kernel_code:
                            kernels.append(kernel_code)
                        else:
                            self.logger.warning(
                                f"Failed to extract code from kernel seed {i}"
                            )
                else:
                    # Provider doesn't support multiple completions, make individual calls
                    for i in range(num_seeds):
                        response_text = self._call_llm(
                            messages,
                            max_tokens=max_completion_tokens,
                            temperature=0.8 + (i * 0.1),
                        )
                        kernel_code = self._extract_code_from_response(response_text)

                        if kernel_code:
                            kernels.append(kernel_code)
                        else:
                            self.logger.warning(
                                f"Failed to extract code from kernel seed {i}"
                            )

                if kernels:
                    self.logger.info(
                        f"Successfully generated {len(kernels)} kernel seeds"
                    )
                    return kernels
                else:
                    self.logger.error(
                        "Failed to extract any valid kernels from LLM responses"
                    )
                    raise ValueError("No valid kernel code found in any LLM response")

            except Exception as e:
                self.logger.error(f"Error generating kernels with LLM API: {e}")
                # Fall back to mock implementation

        # Mock kernel generation (fallback)
        self.logger.info(f"Generating {num_seeds} kernel seeds (mock implementation)")

        kernels = []
        for i in range(num_seeds):
            # Simpler mock that still demonstrates the wrapper pattern
            if i == 2:  # Third kernel will pass
                kernel = '''"""
Kernel implementation - working version.
"""

def kernel_function(*args, **kwargs):
    """Wrapper function that handles kernel launch."""
    # Mock implementation that passes tests
    # In real kernels, this would launch a Triton kernel
    return True
'''
            else:
                kernel = f'''"""
Kernel implementation {i + 1}.
"""

def kernel_function(*args, **kwargs):
    """Wrapper function that handles kernel launch."""
    # Mock implementation that fails
    raise NotImplementedError('Mock kernel not implemented')
'''
            kernels.append(kernel)

        return kernels

    def generate_kernel(
        self, problem_description: str, test_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an optimized Triton kernel for the given problem.

        Args:
            problem_description: Description of the kernel to generate
            test_code: Optional test code (generated if not provided)
                      The test code should:
                      1. Import the kernel function: from kernel import kernel_function
                      2. Test the kernel and return True/False
                      3. Exit with code 0 on success, 1 on failure

        Returns:
            Dictionary with results including successful kernel
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting kernel generation")
        self.logger.info(f"Problem: {problem_description[:100]}...")

        # Always generate test code using LLM (even if test is provided as reference)
        generated_test_code = self._generate_test(problem_description, test_code)
        self.logger.info(
            "Generated test code using LLM" + (" with reference" if test_code else "")
        )

        # Use the generated test code in standardized format
        test_code = generated_test_code

        # Log inputs
        import time

        # Add microseconds to ensure unique directory names
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{int(time.time() * 1000000) % 1000000}"
        )
        session_dir = self.log_dir / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        with open(session_dir / "problem.txt", "w") as f:
            f.write(problem_description)
        with open(session_dir / "test.py", "w") as f:
            f.write(test_code)

        # Generate kernel seeds
        kernel_seeds = self._generate_kernel_seeds(problem_description, test_code)

        # Save seeds
        for i, kernel in enumerate(kernel_seeds):
            with open(session_dir / f"seed_{i}.py", "w") as f:
                f.write(kernel)

        # Run parallel verification with session directory for worker logs
        result = self.manager.run_verification(
            kernel_seeds=kernel_seeds,
            test_code=test_code,
            problem_description=problem_description,
            session_log_dir=session_dir,
        )

        # Process results
        if result and result["success"]:
            self.logger.info(f"Success! Worker {result['worker_id']} found solution")

            # Save successful kernel
            with open(session_dir / "final_kernel.py", "w") as f:
                f.write(result["kernel_code"])

            # Save full result
            with open(session_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            return {
                "success": True,
                "kernel_code": result["kernel_code"],
                "worker_id": result["worker_id"],
                "rounds": result["rounds"],
                "session_dir": str(session_dir),
            }
        else:
            self.logger.warning("No worker found a successful solution")
            return {
                "success": False,
                "message": "Failed to generate working kernel",
                "session_dir": str(session_dir),
            }

    def cleanup(self):
        """Clean up resources."""
        self.manager.cleanup()
