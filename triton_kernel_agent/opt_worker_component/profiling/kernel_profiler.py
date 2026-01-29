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

"""Profiles Triton kernels using NVIDIA Nsight Compute (NCU)."""

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import (
    profile_triton_kernel,
)
from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (
    NCUWrapperFactory,
)
from triton_kernel_agent.opt_worker_component.profiling.ncu_metric_parser import (
    load_and_filter_metrics,
)

# Default timeout for NCU profiling in seconds
DEFAULT_NCU_TIMEOUT_SECONDS = 120


@dataclass
class ProfilerMetadata:
    """Metadata about a profiling run."""

    kernel_file: str
    problem_file: str
    round_num: int
    timestamp: str
    ncu_version: str | None


@dataclass
class NcuProfilerResults:
    """
    Results from an NCU kernel profiling run.

    This dataclass encapsulates both the metrics DataFrame and the parsed
    metrics dictionary, along with metadata about the profiling run.
    """

    metrics_df: pd.DataFrame
    metrics: Dict[str, Any]
    metadata: ProfilerMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for JSON serialization."""
        return {
            "metrics": self.metrics,
            "metadata": asdict(self.metadata),
        }

    def to_json(self) -> str:
        """Convert results to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ProfilerResults:
    """
    Results from a kernel profiling run.
    """

    ncu: NcuProfilerResults | None = None
    sm_occupancy: str | None = None


class KernelProfiler:
    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        logs_dir: Path,
        ncu_bin_path: str | None = None,
        ncu_timeout_seconds: int = DEFAULT_NCU_TIMEOUT_SECONDS,
    ):
        """
        Initialize the kernel profiler.

        Args:
            logger: Logger instance
            artifacts_dir: Directory for optimization artifacts
            logs_dir: Directory for saving profiling logs
            ncu_bin_path: Path to NCU binary (auto-detect if None)
            ncu_timeout_seconds: Timeout for NCU profiling in seconds
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.ncu_bin_path = ncu_bin_path
        self.ncu_timeout_seconds = ncu_timeout_seconds
        self.wrapper_factory = NCUWrapperFactory(logger)

    @cached_property
    def ncu_version(self) -> str | None:
        """
        NCU version string (cached).

        Returns:
            Version string like "2025.2.1.0" or None if unavailable
        """
        try:
            ncu_cmd = self.ncu_bin_path or "ncu"
            result = subprocess.run(
                [ncu_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Extract version from output
                # Example: "Version 2025.2.1.0 (build 35987062) (public-release)"
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Version "):
                        return line.split()[1]
                return None
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get NCU version: {e}")
            return None

    def _wait_with_backoff(self, attempt: int) -> None:
        """
        Wait with exponential backoff before retrying.

        Args:
            attempt: Current attempt number (1-indexed)
        """
        wait_time = 2**attempt  # Exponential backoff: 2, 4, 8 seconds
        self.logger.warning(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> ProfilerResults | None:
        """
        Profile kernel with NCU (with retry logic).

        NCU profiling can fail due to GPU contention or transient issues.
        This method automatically retries with exponential backoff.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            round_num: Current optimization round number (used for file naming
                and tracking which optimization iteration this profiling belongs to)
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            ProfilerResults containing metrics and metadata, or None on failure

        Example:
            >>> profiler = KernelProfiler(logger, artifacts_dir, logs_dir)
            >>> results = profiler.profile_kernel(
            ...     Path("kernel.py"), Path("problem.py"), round_num=1
            ... )
              >>> if results and results.ncu:
            ...     print(f"DRAM throughput: {results.ncu.metrics['dram__throughput']}")
        """
        # Create NCU wrapper script
        wrapper_file = self.wrapper_factory.create_ncu_wrapper(
            kernel_file, problem_file, self.artifacts_dir
        )

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    f"[Round {round_num}] NCU profiling attempt {attempt}/{max_retries}..."
                )

                # Profile with NCU
                csv_file = f"ncu_round_{round_num}.csv"
                csv_path = profile_triton_kernel(
                    benchmark_script=wrapper_file,
                    workdir=self.artifacts_dir,
                    out_csv=csv_file,
                    ncu_bin=self.ncu_bin_path,
                    launch_count=20,
                    timeout=self.ncu_timeout_seconds,
                )

                # Load and parse metrics
                # Filter by Triton kernel names to exclude PyTorch fill/copy kernels
                metrics_df, metrics = load_and_filter_metrics(csv_path)

                # Build NcuProfilerResults
                ncu_results = NcuProfilerResults(
                    metrics_df=metrics_df,
                    metrics=metrics,
                    metadata=ProfilerMetadata(
                        kernel_file=str(kernel_file),
                        problem_file=str(problem_file),
                        round_num=round_num,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        ncu_version=self.ncu_version,
                    ),
                )

                # Save metrics with metadata
                self._save_profiler_results(ncu_results)

                self.logger.info(f"✅ NCU profiling completed for round {round_num}")
                return ProfilerResults(ncu=ncu_results)

            except FileNotFoundError as e:
                self.logger.error(f"❌ File not found during profiling: {e}")
                return None

            except subprocess.TimeoutExpired:
                is_final_attempt = attempt >= max_retries
                if is_final_attempt:
                    self.logger.error(
                        f"❌ NCU profiling timed out after {self.ncu_timeout_seconds}s "
                        f"(final attempt {attempt}/{max_retries})"
                    )
                    return None
                else:
                    self.logger.debug(
                        f"NCU profiling timed out after {self.ncu_timeout_seconds}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    self._wait_with_backoff(attempt)
                    continue

            except json.JSONDecodeError as e:
                is_final_attempt = attempt >= max_retries
                if is_final_attempt:
                    self.logger.error(
                        f"❌ Failed to parse NCU metrics (final attempt): {e}"
                    )
                    return None
                else:
                    self.logger.debug(
                        f"Failed to parse NCU metrics (attempt {attempt}/{max_retries}): {e}"
                    )
                    self._wait_with_backoff(attempt)
                    continue

            except Exception as e:
                is_final_attempt = attempt >= max_retries
                if is_final_attempt:
                    self.logger.error(
                        f"❌ Unexpected error during profiling (final attempt): {e}",
                        exc_info=True,
                    )
                    return None
                else:
                    self.logger.debug(
                        f"Unexpected error during profiling (attempt {attempt}/{max_retries}): {e}"
                    )
                    self._wait_with_backoff(attempt)
                    continue

        # All retries exhausted
        self.logger.error(
            f"❌ NCU profiling failed after {max_retries} attempts for round {round_num}"
        )
        return None

    def _save_profiler_results(self, results: NcuProfilerResults) -> None:
        """
        Save profiling results with metadata to a JSON file.

        Args:
            results: NcuProfilerResults to save
        """
        metrics_file = (
            self.logs_dir / f"round{results.metadata.round_num:03d}_ncu_metrics.json"
        )

        with open(metrics_file, "w") as f:
            f.write(results.to_json())

        self.logger.debug(f"Saved metrics with metadata: {metrics_file}")
