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

import multiprocessing as mp
import os
import re
from logging import Logger
from pathlib import Path

# ------------------------
# LLM Utilities
# ------------------------


def _extract_history_usage_from_response(
    response_text: str,
    logger: Logger | None = None,
) -> dict[str, str | int | None] | None:
    """
    Extract history usage metadata from LLM response.

    Looks for structured comments indicating how the LLM used history.

    Returns:
        Dict with history_usage, based_on_attempt, evolution_rationale, or None.
    """
    if not response_text:
        return None

    result: dict[str, str | int | None] = {}

    # Look for history usage patterns
    usage_match = re.search(r"History usage:\s*(\w+)", response_text, re.IGNORECASE)
    if usage_match:
        result["history_usage"] = usage_match.group(1)

    # Look for "based on attempt N"
    attempt_match = re.search(r"based on attempt\s*(\d+)", response_text, re.IGNORECASE)
    if attempt_match:
        result["based_on_attempt"] = int(attempt_match.group(1))

    # Look for evolution rationale
    rationale_match = re.search(
        r"Evolution rationale:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE
    )
    if rationale_match:
        result["evolution_rationale"] = rationale_match.group(1).strip()

    return result if result else None


# ------------------------
# File I/O Utilities
# ------------------------


def _write_kernel_file(
    kernel_file: Path, kernel_code: str, logger: Logger | None = None
) -> None:
    """Write kernel code to file."""
    kernel_file.write_text(kernel_code)
    if logger:
        logger.debug(f"Wrote kernel to {kernel_file}")


def _save_debug_file(
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


# ------------------------
# Test Execution Utilities
# ------------------------


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
