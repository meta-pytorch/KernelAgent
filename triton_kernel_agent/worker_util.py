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

"""Utility functions for the verification and optimization workers."""

import logging
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any

from logging import Logger

from utils.providers import get_model_provider
from utils.providers.base import BaseProvider


# =============================================================================
# LLM Utilities
# =============================================================================


def call_llm(
    provider: BaseProvider,
    model: str,
    messages: list[dict[str, str]],
    high_reasoning_effort: bool = True,
    logger: Logger | None = None,
    **kwargs,
) -> str:
    """
    Call the LLM provider and return response text.

    Args:
        provider: The LLM provider instance
        model: Model name to use
        messages: List of message dicts with 'role' and 'content'
        high_reasoning_effort: Whether to use high reasoning effort
        logger: Optional logger for debugging
        **kwargs: Additional parameters for the API call

    Returns:
        Generated response text (empty string if None)

    Raises:
        RuntimeError: If provider is not available
    """
    if not provider:
        raise RuntimeError(f"No provider available for model {model}")

    log = logger or logging.getLogger(__name__)

    # Add high_reasoning_effort to kwargs if set
    if high_reasoning_effort:
        kwargs["high_reasoning_effort"] = True

    try:
        log.info(f"Calling LLM: model={model}, provider={provider.name}, messages={len(messages)}")
        response = provider.get_response(model, messages, **kwargs)

        # Log raw response for debugging
        log.info(f"LLM response object: {response}")

        content = response.content
        if content is None:
            log.warning(f"LLM returned None content. Full response: {response}")
            # Try to get more info from the response
            if hasattr(response, '__dict__'):
                log.warning(f"Response attributes: {response.__dict__}")
            return ""

        if len(content) == 0:
            log.warning("LLM returned empty string content")
            return ""

        log.info(f"LLM response received: {len(content)} chars")
        return content

    except Exception as e:
        log.error(f"LLM call failed: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


def create_llm_provider(
    model: str, logger: Logger | None = None
) -> tuple[BaseProvider, str]:
    """
    Create an LLM provider for the given model.

    Args:
        model: Model name
        logger: Optional logger

    Returns:
        Tuple of (provider, model_name)
    """
    log = logger or logging.getLogger(__name__)
    provider = get_model_provider(model)
    log.info(f"Created LLM provider: model={model}, provider={provider.name}")
    return provider, model


# =============================================================================
# Code Extraction Utilities
# =============================================================================


def extract_code_from_response(
    response_text: str,
    language: str = "python",
    logger: Logger | None = None,
) -> str | None:
    """
    Extract code from LLM response text.

    Args:
        response_text: The full LLM response text
        language: The expected language (default: python)
        logger: Optional logger for debugging

    Returns:
        Extracted code or None if no valid code block found
    """
    log = logger or logging.getLogger(__name__)

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
    log.warning("No code block found in LLM response")
    return None


# =============================================================================
# File I/O Utilities
# =============================================================================


def write_kernel_file(kernel_file: Path, kernel_code: str, logger: Logger | None = None) -> None:
    """Write kernel code to file."""
    kernel_file.write_text(kernel_code)
    if logger:
        logger.debug(f"Wrote kernel to {kernel_file}")


def save_debug_file(
    filepath: Path,
    content: str,
    logger: Logger | None = None,
) -> None:
    """Save content to a file for debugging purposes."""
    try:
        filepath.write_text(content)
        if logger:
            logger.debug(f"Saved debug file: {filepath}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save debug file {filepath}: {e}")


# =============================================================================
# Test Execution Utilities
# =============================================================================


def _run_test_process(test_file: Path, workdir: Path, result_queue: mp.Queue) -> None:
    """
    Helper function to run test in a separate process.
    Captures stdout/stderr and sends results via queue.

    Args:
        test_file: Path to the test file
        workdir: Working directory
        result_queue: Queue to send results back
    """
    import sys
    import traceback
    from io import StringIO

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        # Redirect stdout/stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        # Add workdir to sys.path so imports from the same directory work
        os.chdir(str(workdir))
        if str(workdir) not in sys.path:
            sys.path.insert(0, str(workdir))

        with open(test_file) as f:
            code = f.read()

        exec_globals = {
            "__name__": "__main__",
            "__file__": str(test_file),
        }
        exec(code, exec_globals)

        # Template enforces testsfiles to call sys.exit(0)
        # This should NOT be reached
        result_queue.put(
            (
                False,
                stdout_buffer.getvalue(),
                "Test misformatted; did not call sys.exit(#) "
                + stderr_buffer.getvalue(),
            )
        )

    except SystemExit as e:
        # Test code template calls sys.exit()
        if e.code == 0 or e.code is None:
            result_queue.put((True, stdout_buffer.getvalue(), stderr_buffer.getvalue()))
        else:
            # Non-zero exit code means failure
            result_queue.put(
                (
                    False,
                    stdout_buffer.getvalue(),
                    f"Test exited with code {e.code} " + stderr_buffer.getvalue(),
                )
            )

    except BaseException:
        result_queue.put(
            (
                False,
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue() + traceback.format_exc(),
            )
        )

    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _run_test_multiprocess(
    logger: Logger, workdir: Path, test_file: Path
) -> tuple[bool, str, str]:
    """
    Run the test script and capture results using multiprocessing.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    # Create process to run the test
    result_queue = mp.Queue()
    process = mp.Process(
        target=_run_test_process,
        args=(test_file, workdir, result_queue),
    )
    process.start()
    process.join(timeout=30)

    # Check if process is still alive (timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()

        logger.error("Test timed out")
        return False, "", "Test execution timed out after 30 seconds"

    # Get result from queue
    try:
        success, stdout, stderr = result_queue.get_nowait()
        if success:
            logger.info("Test passed")
        else:
            logger.error(
                "Test failed. Exit code: %s, stderr: %s", process.exitcode, stderr[:500]
            )
        return success, stdout, stderr
    except mp.queues.Empty:
        error_msg = (
            f"Test process ended without result. Exit code: {process.exitcode}. "
        )
        logger.error(error_msg)
        return False, "", error_msg
