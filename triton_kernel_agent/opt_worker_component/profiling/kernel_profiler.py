"""Kernel profiling with NCU."""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from kernel_perf_util.kernel_opt.profiler.ncu_profiler import (
    load_ncu_metrics,
    metrics_to_prompt,
    profile_triton_kernel,
)

from .ncu_wrapper_generator import NCUWrapperGenerator


class KernelProfiler:
    """Profiles Triton kernels using NVIDIA Nsight Compute (NCU)."""

    def __init__(
        self,
        logger: logging.Logger,
        temp_dir: Path,
        logs_dir: Path,
        ncu_bin_path: Optional[str] = None,
    ):
        """
        Initialize the kernel profiler.

        Args:
            logger: Logger instance
            temp_dir: Temporary directory for profiling artifacts
            logs_dir: Directory for saving profiling logs
            ncu_bin_path: Path to NCU binary (auto-detect if None)
        """
        self.logger = logger
        self.temp_dir = temp_dir
        self.logs_dir = logs_dir
        self.ncu_bin_path = ncu_bin_path
        self.wrapper_generator = NCUWrapperGenerator(logger)

    def _get_ncu_version(self) -> Optional[str]:
        """
        Get NCU version string.

        Returns:
            NCU version string or None if failed
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
                # Extract version from output (e.g., "NVIDIA Nsight Compute 2024.3.1")
                version_line = result.stdout.strip().split("\n")[0]
                return version_line
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get NCU version: {e}")
            return None

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 3,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Profile kernel with NCU (with retry logic).

        NCU profiling can fail due to GPU contention or transient issues.
        This method automatically retries with exponential backoff.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            round_num: Current optimization round number
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Tuple of (metrics_df, metrics_json) or (None, None) on failure

        Example:
            >>> profiler = KernelProfiler(logger, temp_dir, logs_dir)
            >>> metrics_df, metrics_json = profiler.profile_kernel(
            ...     Path("kernel.py"), Path("problem.py"), round_num=1
            ... )
            >>> if metrics_json:
            ...     print(f"DRAM throughput: {metrics_json['dram__throughput']}")
        """
        wrapper_file = None

        for attempt in range(1, max_retries + 1):
            try:
                # Create NCU wrapper script (cached if unchanged)
                if wrapper_file is None:
                    wrapper_file = self.wrapper_generator.create_ncu_wrapper(
                        kernel_file, problem_file, self.temp_dir
                    )

                self.logger.info(
                    f"[Round {round_num}] NCU profiling attempt {attempt}/{max_retries}..."
                )

                # Profile with NCU
                csv_file = f"ncu_round_{round_num}.csv"
                csv_path = profile_triton_kernel(
                    benchmark_script=wrapper_file,
                    workdir=self.temp_dir,
                    out_csv=csv_file,
                    ncu_bin=self.ncu_bin_path,
                    launch_count=20,
                    timeout=120,
                )

                # Load and parse metrics
                metrics_df = load_ncu_metrics(csv_path, select="last")
                metrics_json = json.loads(metrics_to_prompt(metrics_df))

                # Save metrics with metadata
                self._save_metrics_with_metadata(
                    metrics_json, kernel_file, problem_file, round_num
                )

                self.logger.info(f"✅ NCU profiling completed for round {round_num}")
                return metrics_df, metrics_json

            except FileNotFoundError as e:
                self.logger.error(f"❌ File not found during profiling: {e}")
                return None, None

            except subprocess.TimeoutExpired:
                self.logger.error(
                    f"❌ NCU profiling timed out after 120s (attempt {attempt}/{max_retries})"
                )
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff: 2, 4, 8 seconds
                    self.logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None, None

            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse NCU metrics: {e}")
                if attempt < max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None, None

            except Exception as e:
                self.logger.error(
                    f"❌ Unexpected error during profiling (attempt {attempt}/{max_retries}): {e}",
                    exc_info=True,
                )
                if attempt < max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None, None

        # All retries exhausted
        self.logger.error(
            f"❌ NCU profiling failed after {max_retries} attempts for round {round_num}"
        )
        return None, None

    def _save_metrics_with_metadata(
        self,
        metrics_json: Dict[str, Any],
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
    ) -> None:
        """
        Save profiling metrics with additional metadata.

        Args:
            metrics_json: NCU metrics as JSON dict
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            round_num: Current optimization round number
        """
        metrics_file = self.logs_dir / f"round{round_num:03d}_ncu_metrics.json"

        # Build metadata
        metadata = {
            "metrics": metrics_json,
            "metadata": {
                "kernel_file": str(kernel_file),
                "problem_file": str(problem_file),
                "round_num": round_num,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "ncu_version": self._get_ncu_version(),
            },
        }

        with open(metrics_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.debug(f"Saved metrics with metadata: {metrics_file}")
