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

import logging
from datetime import datetime
from pathlib import Path

from mpp.frontend.script_executor import PicklableScriptExecutor
from mpp.frontend.mpp_frontend import MppFrontend
from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
    NcuProfilerResults,
    ProfilerResults,
    ProfilerMetadata,
)
from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (
    NCUWrapperFactory,
)
from triton_kernel_agent.opt_worker_component.profiling.ncu_metric_parser import (
    load_and_filter_metrics,
)


# Default timeout for NCU profiling in seconds
DEFAULT_NCU_TIMEOUT_SECONDS = 600


class TritonMppProfiler:
    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        logs_dir: Path,
        ncu_timeout_seconds: int = DEFAULT_NCU_TIMEOUT_SECONDS,
    ):
        """
        Initialize the kernel profiler.
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.ncu_timeout_seconds = ncu_timeout_seconds
        self.wrapper_factory = NCUWrapperFactory(logger)

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> ProfilerResults | None:
        self.logger.info(f"[TritonMppProfiler][Round {round_num}] Profiling kernel...")

        wrapper_file = self.wrapper_factory.create_ncu_wrapper(
            kernel_file, problem_file, self.artifacts_dir
        )

        ncu_result = self._do_ncu(
            wrapper_file, kernel_file, problem_file, round_num, max_retries
        )

        sm_occupancy_result = self._do_sm_occupancy(round_num, wrapper_file)

        if not ncu_result and not sm_occupancy_result:
            return None

        return ProfilerResults(ncu=ncu_result, sm_occupancy=sm_occupancy_result)

    def _do_ncu(
        self,
        wrapper_file: Path,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> NcuProfilerResults | None:
        """
        Profile kernel with NCU using MppFrontend.profiler() API (with retry logic).
        """

        self.logger.info(f"[Round {round_num}] NCU profiling...")

        from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import METRICS

        frontend = MppFrontend.create_from_script(wrapper_file)
        profiler = frontend.profiler()

        prefix = f"ncu_round_{round_num}"
        csv_path = self.artifacts_dir / f"{prefix}.csv"
        mpp_report_path = self.artifacts_dir / f"ncu_mpp_report_round_{round_num}.txt"
        ncu_rep_path = self.artifacts_dir / f"ncu_rep_round_{round_num}.ncu-rep"

        ncu_args = " ".join(
            [
                "--csv",
                f"--log-file {csv_path}",
                "--page=raw",
                "--set=roofline",
                f"--metrics={METRICS}",
                "--launch-skip=1",
                "--launch-count=20",
            ]
        )

        profiler.ncu(
            output_file=str(mpp_report_path),
            include_source_analysis=False,
            include_detailed_metrics=True,
            ncu_report_file=str(ncu_rep_path),
            additional_args=ncu_args,
        )

        if not csv_path.exists():
            self.logger.error(f"❌ NCU did not create output CSV: {csv_path}")
            return None

        metrics_df, metrics = load_and_filter_metrics(csv_path)

        results = NcuProfilerResults(
            metrics_df=metrics_df,
            metrics=metrics,
            metadata=ProfilerMetadata(
                kernel_file=str(kernel_file),
                problem_file=str(problem_file),
                round_num=round_num,
                timestamp=datetime.utcnow().isoformat() + "Z",
                ncu_version=None,
            ),
        )

        self._save_profiler_results(results)
        self.logger.info(f"✅ NCU profiling completed for round {round_num}")
        return results

    def _do_sm_occupancy(
        self,
        round_num: int,
        wrapper_file: Path,
    ) -> str | None:
        """Generate SM occupancy plot using MppFrontend.profiler() API.

        Args:
            round_num: Current optimization round number
            wrapper_file: Path to the wrapper script

        Returns:
            Path to the generated plot file, or None on failure
        """
        plot_file = self.artifacts_dir / f"round{round_num:03d}_sm_occupancy.png"

        self.logger.info(f"[Round {round_num}] Generating SM occupancy plot...")

        try:
            # Create profiler using MppFrontend API
            frontend = MppFrontend(PicklableScriptExecutor(str(wrapper_file)))
            profiler = frontend.profiler()

            # Generate SM occupancy plot using the new Profiler API
            profiler.sm_occupancy(plot_file=str(plot_file))

            if not plot_file.exists():
                self.logger.error(f"❌ SM occupancy plot file not created: {plot_file}")
                return None

            self.logger.info(f"✅ SM occupancy plot saved to: {plot_file}")
            return str(plot_file)

        except Exception as e:
            self.logger.error(f"❌ Failed to generate SM occupancy plot: {e}")
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
