"""Kernel profiling with NCU."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import (
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

    def profile_kernel(
        self, kernel_file: Path, problem_file: Path, round_num: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Profile kernel with NCU.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            round_num: Current optimization round number

        Returns:
            Tuple of (metrics_df, metrics_json) or (None, None) on failure
        """
        try:
            # Create NCU wrapper script
            wrapper_file = self.wrapper_generator.create_ncu_wrapper(
                kernel_file, problem_file, self.temp_dir
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

            # Save metrics JSON
            metrics_file = self.logs_dir / f"round{round_num:03d}_ncu_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_json, f, indent=2)

            self.logger.info(f"✅ NCU profiling completed for round {round_num}")
            return metrics_df, metrics_json

        except Exception as e:
            self.logger.error(f"❌ NCU profiling failed: {e}")
            return None, None
