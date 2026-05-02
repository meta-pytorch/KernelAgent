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

"""Profiles Triton kernels on AMD GPUs using rocprof."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Dict

from kernel_perf_agent.kernel_opt.profiler.rocprof_profiler import (
    load_rocm_metrics,
    profile_triton_kernel_rocm,
)
from triton_kernel_agent.opt_worker_component.profiling.rocprof_wrapper_factory import (
    ROCmWrapperFactory,
)

# Default timeout for rocprof profiling in seconds (two passes = 2x NVIDIA time)
DEFAULT_ROCPROF_TIMEOUT_SECONDS = 600

# Default timeout for profiling semaphore (15 minutes)
DEFAULT_SEMAPHORE_TIMEOUT_SECONDS = 900


@dataclass
class ROCmProfilerMetadata:
    """Metadata about a ROCm profiling run."""

    kernel_file: str
    problem_file: str
    round_num: int
    timestamp: str
    rocprof_bin: str | None


@dataclass
class ROCmProfilerResults:
    """Results from a ROCm kernel profiling run.

    Designed to be a drop-in replacement for
    :class:`triton_kernel_agent.opt_worker_component.profiling.kernel_profiler.ProfilerResults`
    in the optimization pipeline.
    """

    metrics: Dict[str, Any]
    metadata: ROCmProfilerMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "metadata": asdict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ROCmKernelProfiler:
    """Profiles Triton kernels on AMD GPUs using rocprof.

    Drop-in replacement for
    :class:`triton_kernel_agent.opt_worker_component.profiling.kernel_profiler.KernelProfiler`
    for the ROCm/HIP platform.
    """

    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        logs_dir: Path,
        rocprof_bin_path: str | None = None,
        rocprof_timeout_seconds: int = DEFAULT_ROCPROF_TIMEOUT_SECONDS,
        profiling_semaphore: Any | None = None,
    ) -> None:
        """
        Initialize the ROCm kernel profiler.

        Args:
            logger: Logger instance.
            artifacts_dir: Directory for optimization artifacts.
            logs_dir: Directory for saving profiling logs.
            rocprof_bin_path: Path to rocprof binary (auto-detect if None).
            rocprof_timeout_seconds: Timeout per rocprof invocation.
            profiling_semaphore: Semaphore to limit concurrent rocprof runs.
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.rocprof_bin_path = rocprof_bin_path
        self.rocprof_timeout_seconds = rocprof_timeout_seconds
        self.profiling_semaphore = profiling_semaphore
        self.wrapper_factory = ROCmWrapperFactory(logger)

    @cached_property
    def rocprof_bin(self) -> str | None:
        """Resolved rocprof binary path (cached)."""
        import shutil

        if self.rocprof_bin_path:
            return self.rocprof_bin_path
        for candidate in ("rocprofv3", "rocprof"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def _wait_with_backoff(self, attempt: int) -> None:
        wait_time = 2**attempt
        self.logger.warning(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> ROCmProfilerResults | None:
        """Profile a Triton kernel with rocprof (with retry logic).

        Args:
            kernel_file: Path to kernel file.
            problem_file: Path to problem file.
            round_num: Current optimization round number.
            max_retries: Maximum number of retry attempts.

        Returns:
            :class:`ROCmProfilerResults` or ``None`` on failure.
        """
        wrapper_file = self.wrapper_factory.create_rocprof_wrapper(
            kernel_file, problem_file, self.artifacts_dir
        )

        semaphore_acquired = False
        if self.profiling_semaphore is not None:
            self.logger.info(f"[Round {round_num}] Waiting for profiling semaphore...")
            semaphore_acquired = self.profiling_semaphore.acquire(
                timeout=DEFAULT_SEMAPHORE_TIMEOUT_SECONDS
            )
            if not semaphore_acquired:
                self.logger.warning(
                    f"[Round {round_num}] Semaphore timeout after "
                    f"{DEFAULT_SEMAPHORE_TIMEOUT_SECONDS}s, skipping profiling"
                )
                return None
            self.logger.info(f"[Round {round_num}] Acquired profiling semaphore")

        try:
            return self._profile_kernel_impl(
                wrapper_file, kernel_file, problem_file, round_num, max_retries
            )
        finally:
            if semaphore_acquired:
                self.profiling_semaphore.release()
                self.logger.debug(f"[Round {round_num}] Released profiling semaphore")

    def _profile_kernel_impl(
        self,
        wrapper_file: Path,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int,
    ) -> ROCmProfilerResults | None:
        """Internal profiling implementation (called with semaphore held)."""

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    f"[Round {round_num}] rocprof profiling attempt {attempt}/{max_retries}..."
                )

                metrics_json = profile_triton_kernel_rocm(
                    benchmark_script=wrapper_file,
                    workdir=self.artifacts_dir,
                    out_prefix=f"rocprof_round_{round_num}",
                    rocprof_bin=self.rocprof_bin,
                    timeout=self.rocprof_timeout_seconds,
                )

                metrics = load_rocm_metrics(metrics_json)

                results = ROCmProfilerResults(
                    metrics=metrics,
                    metadata=ROCmProfilerMetadata(
                        kernel_file=str(kernel_file),
                        problem_file=str(problem_file),
                        round_num=round_num,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        rocprof_bin=self.rocprof_bin,
                    ),
                )

                self._save_profiler_results(results)
                self.logger.info(
                    f"✅ rocprof profiling completed for round {round_num}"
                )
                return results

            except FileNotFoundError as e:
                self.logger.error(f"❌ File not found during profiling: {e}")
                return None

            except subprocess.TimeoutExpired:
                is_final = attempt >= max_retries
                if is_final:
                    self.logger.error(
                        f"❌ rocprof timed out after {self.rocprof_timeout_seconds}s "
                        f"(final attempt {attempt}/{max_retries})"
                    )
                    return None
                self.logger.debug(
                    f"rocprof timed out (attempt {attempt}/{max_retries})"
                )
                self._wait_with_backoff(attempt)

            except Exception as e:
                is_final = attempt >= max_retries
                err_str = str(e)
                if (
                    "signal" in err_str.lower()
                    or "segfault" in err_str.lower()
                    or "sigsegv" in err_str.lower()
                ):
                    self.logger.error(
                        "❌ rocprof crashed with segfault (likely ROCm/PyTorch version mismatch). "
                        "Profiling unavailable — optimization will use timing-only fallback."
                    )
                    return None
                if is_final:
                    self.logger.error(
                        f"❌ Unexpected error during profiling (final attempt): {e}",
                        exc_info=True,
                    )
                    return None
                self.logger.debug(
                    f"Unexpected error (attempt {attempt}/{max_retries}): {e}"
                )
                self._wait_with_backoff(attempt)

        self.logger.error(
            f"❌ rocprof profiling failed after {max_retries} attempts for round {round_num}"
        )
        return None

    def _save_profiler_results(self, results: ROCmProfilerResults) -> None:
        metrics_file = (
            self.logs_dir
            / f"round{results.metadata.round_num:03d}_rocprof_metrics.json"
        )
        with open(metrics_file, "w") as f:
            f.write(results.to_json())
        self.logger.debug(f"Saved ROCm metrics: {metrics_file}")
