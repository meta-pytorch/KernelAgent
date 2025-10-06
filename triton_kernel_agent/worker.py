"""Verification Worker for testing and refining individual kernels."""

import json
import logging
import multiprocessing as mp
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .prompt_manager import PromptManager
from .providers import get_model_provider


DISALLOWED_TORCH_PATTERNS = [
    (
        re.compile(r"\bimport\s+torch\.nn(\b|\s+as\b)"),
        "importing torch.nn modules is not allowed",
    ),
    (
        re.compile(r"\bfrom\s+torch\s+import\s+nn\b"),
        "importing torch.nn modules is not allowed",
    ),
    (
        re.compile(r"\bimport\s+torch\.nn\.functional\s+as\s+F\b"),
        "aliasing torch.nn.functional as F is not allowed",
    ),
    (re.compile(r"\btorch\.nn\."), "torch.nn module usage is not allowed"),
    (
        re.compile(r"\btorch\.nn\.functional\b"),
        "torch.nn.functional usage is not allowed",
    ),
    (
        re.compile(r"\bF\.[A-Za-z_]+\("),
        "torch.nn.functional alias calls (F.*) are not allowed",
    ),
    (re.compile(r"\btorch\.conv"), "torch convolution helpers are not allowed"),
    (
        re.compile(
            r"\btorch\.(relu|sigmoid|tanh|softmax|gelu|mish|hardtanh|max_pool|avg_pool)[A-Za-z0-9_]*\("
        ),
        "PyTorch activation/pooling helpers are not allowed",
    ),
    (
        re.compile(r"\bclass\s+\w+\s*\(\s*nn\.Module"),
        "Subclassing torch.nn.Module is not allowed",
    ),
    (
        re.compile(r"\.forward\("),
        "Calling .forward() indicates torch.nn module usage and is not allowed",
    ),
]


class VerificationWorker:
    """Worker that verifies and refines a single kernel implementation."""

    def __init__(
        self,
        worker_id: int,
        workdir: Path,
        log_dir: Path,
        max_rounds: int = 10,
        history_size: int = 8,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-5",
        high_reasoning_effort: bool = True,
    ):
        """
        Initialize a verification worker.

        Args:
            worker_id: Unique identifier for this worker
            workdir: Working directory for this worker
            log_dir: Directory for logging
            max_rounds: Maximum refinement rounds
            history_size: Number of recent rounds to keep
            openai_api_key: OpenAI API key for refinement
            openai_model: Model name for refinement
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.history_size = history_size
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort

        # Setup files
        self.kernel_file = self.workdir / "kernel.py"
        self.test_file = self.workdir / "test_kernel.py"

        # History for LLM context
        self.history = deque(maxlen=history_size)

        # Setup logging early so it is available for any error paths
        self._setup_logging()

        # Initialize provider (may be unavailable in offline/test environments)
        self.provider = None
        try:
            self.provider = get_model_provider(self.openai_model)
        except ValueError as e:
            # Provider not available, will use mock mode
            self.logger.warning(f"Provider not available: {e}")

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

    def _setup_logging(self):
        """Setup worker-specific logging."""
        log_file = self.log_dir / f"worker_{self.worker_id}.log"
        self.logger = logging.getLogger(f"worker_{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

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

    def _write_kernel(self, kernel_code: str):
        """Write only the kernel code to file."""
        self.kernel_file.write_text(kernel_code)
        self.logger.info("Updated kernel file")

    def _write_files(self, kernel_code: str, test_code: str):
        """Write kernel and test code to files.

        Note: The test code should import the kernel function from the kernel file:
            from kernel import kernel_function

        Both files are written to the same directory (workdir).
        """
        self.kernel_file.write_text(kernel_code)
        self.test_file.write_text(test_code)
        self.logger.info("Wrote kernel and test files")

    def _strip_comments_and_strings(self, code: str) -> str:
        """Remove comments and docstrings to avoid false positives when scanning code."""
        pattern = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#.*)')
        return re.sub(pattern, "", code)

    def _detect_pytorch_compute(self, kernel_code: str) -> Optional[str]:
        """Detect disallowed PyTorch usage inside the kernel wrapper."""
        sanitized = self._strip_comments_and_strings(kernel_code)
        for pattern, message in DISALLOWED_TORCH_PATTERNS:
            if pattern.search(sanitized):
                return message
        return None

    def _run_test(self) -> Tuple[bool, str, str]:
        """
        Run the test script and capture results.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [sys.executable, str(self.test_file)]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            success = result.returncode == 0
            self.logger.info(
                f"Test {'passed' if success else 'failed'} with code {result.returncode}"
            )

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error("Test timed out")
            return False, "", "Test execution timed out after 30 seconds"
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            return False, "", str(e)

    def _call_llm(self, messages: list, **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.openai_model}")

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.openai_model, messages, **kwargs)
        return response.content

    def _refine_kernel(
        self,
        kernel_code: str,
        error_info: Dict[str, str],
        problem_description: str,
        test_code: str,
    ) -> str:
        """
        Refine kernel based on error information using OpenAI API.

        Uses multi-turn dialogue by incorporating history of previous attempts.
        """
        if self.provider:
            try:
                self.logger.info(f"Refining kernel using {self.openai_model}")

                # Build context from history
                history_context = ""
                if self.history:
                    history_context = "\n\nPREVIOUS ATTEMPTS:\n"
                    for i, round_data in enumerate(self.history):
                        history_context += f"\nAttempt {i + 1}:\n"
                        history_context += f"Kernel code:\n```python\n{round_data['kernel_code'][:500]}...\n```\n"
                        if round_data.get("stderr"):
                            history_context += f"Error: {round_data['stderr'][:200]}\n"
                        if round_data.get("stdout"):
                            history_context += f"Output: {round_data['stdout'][:200]}\n"

                # Create refinement prompt using template
                prompt = self.prompt_manager.render_kernel_refinement_prompt(
                    problem_description=problem_description,
                    test_code=test_code,
                    kernel_code=kernel_code,
                    error_info=error_info,
                    history_context=history_context,
                )

                # Call LLM API
                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_llm(messages, max_tokens=20000)

                # Extract refined kernel from response
                refined_kernel = self._extract_code_from_response(response_text)

                if refined_kernel:
                    self.logger.info(
                        f"Successfully refined kernel using {self.openai_model}"
                    )
                    return refined_kernel
                else:
                    self.logger.error("Failed to extract valid code from LLM response")
                    # Return original kernel if extraction fails
                    return kernel_code

            except Exception as e:
                self.logger.error(f"Error refining kernel with LLM API: {e}")
                # Fall back to mock refinement

        # Mock refinement (fallback)
        self.logger.info("Refining kernel (mock implementation)")

        # For testing, make a simple modification
        if "error" in error_info.get("stderr", "").lower():
            # Add a comment to show refinement happened
            return f"# Refinement attempt {len(self.history) + 1}\n{kernel_code}"

        return kernel_code

    def _log_round(
        self, round_num: int, success: bool, kernel_code: str, stdout: str, stderr: str
    ):
        """Log the results of a verification round."""
        round_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "kernel_code": kernel_code,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Save to log file
        round_log_file = self.log_dir / f"round_{round_num}.json"
        with open(round_log_file, "w") as f:
            json.dump(round_data, f, indent=2)

        # Add to history
        self.history.append(round_data)

    def run(
        self,
        kernel_code: str,
        test_code: str,
        problem_description: str,
        success_event: mp.Event,
    ) -> Dict[str, Any]:
        """
        Run verification and refinement loop.

        Args:
            kernel_code: Initial kernel implementation
            test_code: Test code to verify kernel
            problem_description: Problem description for context
            success_event: Shared event to check if another worker succeeded

        Returns:
            Dictionary with results
        """
        self.logger.info(f"Starting verification for worker {self.worker_id}")

        current_kernel = kernel_code

        for round_num in range(self.max_rounds):
            # Check if another worker has succeeded
            if success_event.is_set():
                self.logger.info("Another worker succeeded, stopping")
                return {
                    "worker_id": self.worker_id,
                    "success": False,
                    "stopped_early": True,
                    "rounds": round_num,
                }

            self.logger.info(f"Round {round_num + 1}/{self.max_rounds}")

            # Write files - test only on first round, kernel every round
            if round_num == 0:
                # First round: write both kernel and test
                self._write_files(current_kernel, test_code)
            else:
                # Subsequent rounds: only update kernel, test remains unchanged
                self._write_kernel(current_kernel)

            violation = self._detect_pytorch_compute(current_kernel)
            if violation:
                message = f"Disallowed PyTorch usage detected: {violation}"
                self.logger.error(message)
                self._log_round(round_num + 1, False, current_kernel, "", message)
                error_info = {
                    "stdout": "",
                    "stderr": message,
                    "history": list(self.history),
                }
                current_kernel = self._refine_kernel(
                    current_kernel, error_info, problem_description, test_code
                )
                continue

            # Run test
            success, stdout, stderr = self._run_test()

            # Log round
            self._log_round(round_num + 1, success, current_kernel, stdout, stderr)

            if success:
                self.logger.info(
                    f"Success! Kernel passed test in round {round_num + 1}"
                )
                return {
                    "worker_id": self.worker_id,
                    "success": True,
                    "kernel_code": current_kernel,
                    "rounds": round_num + 1,
                    "history": list(self.history),
                }

            # Refine kernel for next round
            error_info = {
                "stdout": stdout,
                "stderr": stderr,
                "history": list(self.history),
            }

            current_kernel = self._refine_kernel(
                current_kernel, error_info, problem_description, test_code
            )

        # Max rounds reached without success
        self.logger.warning(f"Max rounds ({self.max_rounds}) reached without success")
        return {
            "worker_id": self.worker_id,
            "success": False,
            "max_rounds_reached": True,
            "rounds": self.max_rounds,
            "history": list(self.history),
        }
