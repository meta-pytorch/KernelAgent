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

"""NVIDIA (CUDA / NCU) implementations of platform interfaces.

These wrap the existing NVIDIA-specific code that was previously inlined
in ``OptimizationManager``.  When no explicit platform components are
provided, these are used as the default.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RAGPrescriberBase,
    RooflineAnalyzerBase,
)
from triton_kernel_agent.platform.worker_runner import NvidiaWorkerRunner  # noqa: F401

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class NvidiaVerifier(KernelVerifier):
    """Verifies kernel correctness using ``VerificationWorker`` on CUDA."""

    def __init__(self, log_dir: Path, logger: logging.Logger) -> None:
        self.log_dir = log_dir
        self.logger = logger

    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        from triton_kernel_agent.worker import VerificationWorker

        verify_dir = self.log_dir / "initial_verify"
        verify_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(problem_file, verify_dir / "problem.py")

        worker = VerificationWorker(
            worker_id=-1,
            workdir=verify_dir,
            log_dir=verify_dir,
        )

        success, _, error = worker.verify_with_refinement(
            kernel_code=kernel_code,
            test_code=test_code,
            problem_description=problem_file.read_text(),
            max_refine_attempts=0,
        )

        if not success:
            self.logger.error(
                f"Initial kernel failed correctness verification: {error[:200]}"
            )
        else:
            self.logger.info("Initial kernel passed correctness verification")

        return success


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------


class NvidiaBenchmarker(KernelBenchmarker):
    """Benchmarks kernels and baselines using CUDA events / ``triton.testing``."""

    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        warmup: int = 25,
        repeat: int = 100,
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self.benchmark_lock = benchmark_lock
        self.warmup = warmup
        self.repeat = repeat

    def _get_benchmarker(self):
        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            Benchmark,
        )

        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return Benchmark(
            logger=self.logger,
            artifacts_dir=artifacts_dir,
            benchmark_lock=self.benchmark_lock,
            worker_id=-1,
            warmup=self.warmup,
            repeat=self.repeat,
        )

    def benchmark_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
    ) -> float:
        benchmarker = self._get_benchmarker()
        artifacts_dir = self.log_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        kernel_file = artifacts_dir / "initial_kernel.py"
        kernel_file.write_text(kernel_code, encoding="utf-8")

        result = benchmarker.benchmark_kernel(kernel_file, problem_file)
        kernel_time = result.get("time_ms", float("inf"))

        if kernel_time != float("inf"):
            self.logger.info(f"Initial kernel time: {kernel_time:.4f}ms")

        return kernel_time

    def benchmark_reference(
        self,
        problem_file: Path,
    ) -> float:
        benchmarker = self._get_benchmarker()
        result = benchmarker.benchmark_pytorch(problem_file)
        pytorch_time = result.get("time_ms", float("inf"))

        if pytorch_time != float("inf"):
            self.logger.info(f"PyTorch baseline: {pytorch_time:.4f}ms")

        return pytorch_time

    def benchmark_reference_compiled(
        self,
        problem_file: Path,
    ) -> float:
        benchmarker = self._get_benchmarker()
        result = benchmarker.benchmark_pytorch_compile(problem_file)
        compile_time = result.get("time_ms", float("inf"))

        if compile_time != float("inf"):
            self.logger.info(f"PyTorch compile baseline: {compile_time:.4f}ms")

        return compile_time


# ---------------------------------------------------------------------------
# Worker-level NVIDIA implementations
#
# These wrap the concrete NVIDIA/CUDA classes that live deeper in the
# codebase.  They use lazy delegation so that heavy imports (NCU, GPU
# detection, OpenAI embeddings) only happen on first use, and so that
# instances can be created at manager time with a subset of kwargs.
# ---------------------------------------------------------------------------


class NvidiaAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Looks up NVIDIA GPU specs via ``get_gpu_specs``."""

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )

        if not device_name:
            raise ValueError("device_name is required (e.g. 'NVIDIA H100 NVL 94GB')")
        return get_gpu_specs(device_name)


class NvidiaKernelProfiler(KernelProfilerBase):
    """Wraps :class:`KernelProfiler` (NCU-based) with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        ncu_bin_path: str | None = None,
        ncu_timeout_seconds: int | None = None,
        profiling_semaphore: Any | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else Path(".")
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._ncu_bin_path = ncu_bin_path
        self._ncu_timeout_seconds = ncu_timeout_seconds
        self._profiling_semaphore = profiling_semaphore
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
                KernelProfiler,
            )

            artifacts_dir = self._artifacts_dir or self._log_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "logger": self._logger,
                "artifacts_dir": artifacts_dir,
                "logs_dir": self._log_dir,
                "ncu_bin_path": self._ncu_bin_path,
                "profiling_semaphore": self._profiling_semaphore,
            }
            if self._ncu_timeout_seconds is not None:
                kwargs["ncu_timeout_seconds"] = self._ncu_timeout_seconds
            self._delegate = KernelProfiler(**kwargs)
        return self._delegate

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        return self._get_delegate().profile_kernel(
            kernel_file, problem_file, round_num, max_retries
        )


class NvidiaRooflineAnalyzer(RooflineAnalyzerBase):
    """Wraps :class:`RooflineAnalyzer` (NCU SOL metrics) with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        roofline_config: Any | None = None,
    ) -> None:
        self._logger = logger
        self._roofline_config = roofline_config
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
                RooflineAnalyzer,
            )

            kwargs: dict[str, Any] = {"logger": self._logger}
            if self._roofline_config is not None:
                kwargs["config"] = self._roofline_config
            self._delegate = RooflineAnalyzer(**kwargs)
        return self._delegate

    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        return self._get_delegate().analyze(ncu_metrics)

    def should_stop(self, result: Any) -> tuple[bool, str]:
        return self._get_delegate().should_stop(result)

    @property
    def config(self) -> Any:
        # The orchestrator reads early_stop on the platform analyzer after
        # should_stop(). The flag lives on the inner NCU analyzer.
        return self._get_delegate().config

    def reset_history(self) -> None:
        self._get_delegate().reset_history()


class NvidiaBottleneckAnalyzer(BottleneckAnalyzerBase):
    """Wraps :class:`BottleneckAnalyzer` (LLM-based) with lazy construction.

    The ``provider`` and ``gpu_specs`` dependencies are resolved lazily
    on first use so this class can be instantiated at manager time when
    only ``logger`` and ``openai_model`` are available.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "gpt-5",
        gpu_name: str | None = None,
        roofline_config: Any | None = None,
        num_bottlenecks: int = 1,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else None
        self._openai_model = openai_model
        self._gpu_name = gpu_name
        self._num_bottlenecks = max(1, int(num_bottlenecks))
        self._delegate: Any | None = None
        # Orchestrator accesses ``bottleneck_analyzer.roofline`` directly.
        self.roofline = NvidiaRooflineAnalyzer(
            logger=self._logger, roofline_config=roofline_config
        )

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
                get_gpu_specs,
            )
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            if not self._gpu_name:
                raise ValueError("gpu_name is required for NvidiaBottleneckAnalyzer")
            provider = get_model_provider(self._openai_model)
            gpu_specs = get_gpu_specs(self._gpu_name)

            self._delegate = BottleneckAnalyzer(
                provider=provider,
                model=self._openai_model,
                gpu_specs=gpu_specs,
                logs_dir=self._log_dir,
                logger=self._logger,
                num_bottlenecks=self._num_bottlenecks,
            )
        return self._delegate

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze(
            kernel_code, ncu_metrics, round_num, roofline_result
        )


class NvidiaRAGPrescriber(RAGPrescriberBase):
    """Wraps :class:`RAGPrescriber` (OpenAI-embedding RAG) with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        database_path: Path | None = None,
    ) -> None:
        self._logger = logger
        self._database_path = database_path
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.RAG_based_prescriber import (
                RAGPrescriber,
            )

            kwargs: dict[str, Any] = {"logger": self._logger}
            if self._database_path is not None:
                kwargs["database_path"] = self._database_path
            self._delegate = RAGPrescriber(**kwargs)
        return self._delegate

    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        return self._get_delegate().retrieve(query)

    def build_context(self, opt_node: Any, **kwargs: Any) -> str:
        return self._get_delegate().build_context(opt_node, **kwargs)
