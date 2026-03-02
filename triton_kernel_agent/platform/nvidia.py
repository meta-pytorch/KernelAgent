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
import multiprocessing as mp
import shutil
import time
import traceback
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
    WorkerRunner,
)


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
        test_code: str,
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
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self.benchmark_lock = benchmark_lock

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
# Worker runner
# ---------------------------------------------------------------------------


class NvidiaWorkerRunner(WorkerRunner):
    """Spawns ``OptimizationWorker`` processes on NVIDIA GPUs."""

    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        profiling_semaphore: Any,
        openai_model: str,
        high_reasoning_effort: bool,
        bottleneck_override: str | None,
        worker_kwargs: dict[str, Any],
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self.benchmark_lock = benchmark_lock
        self.profiling_semaphore = profiling_semaphore
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs

    def run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: str,
        pytorch_baseline: float,
        shared_history: list[dict],
        shared_reflexions: list[dict],
    ) -> list[dict[str, Any]]:
        result_queue = mp.Queue()
        workers = []

        for i, candidate in enumerate(candidates):
            workdir = self.log_dir / "workers" / f"w{i}" / f"r{round_num}"
            workdir.mkdir(parents=True, exist_ok=True)

            args = (
                i,  # worker_id
                candidate["parent"].kernel_code,
                candidate["parent"].metrics.time_ms,
                candidate["parent"].program_id,
                problem_file,
                test_code,
                workdir,
                workdir / "logs",
                result_queue,
                self.benchmark_lock,
                self.profiling_semaphore,
                pytorch_baseline,
                candidate["bottleneck_id"],
                self.openai_model,
                self.high_reasoning_effort,
                self.bottleneck_override,
                self.worker_kwargs,
                shared_history,
                shared_reflexions,
            )

            p = mp.Process(target=_nvidia_worker_process, args=args)
            p.start()
            workers.append(p)

        # Wait for completion with timeout
        worker_timeout = 1800  # 30 minutes
        deadline = time.time() + worker_timeout
        for w in workers:
            remaining = max(0, deadline - time.time())
            w.join(timeout=remaining)
            if w.is_alive():
                self.logger.warning(f"Worker {w.pid} timed out, terminating")
                w.terminate()
                w.join(timeout=5)
                if w.is_alive():
                    self.logger.warning(f"Worker {w.pid} still alive, killing")
                    w.kill()
                    w.join(timeout=2)
            w.close()

        # Collect results
        results: list[dict[str, Any]] = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        # Clean up queue resources to prevent thread hangs during GC
        result_queue.close()
        result_queue.join_thread()

        successful = sum(1 for r in results if r.get("success"))
        self.logger.info(
            f"Round {round_num}: {successful}/{len(candidates)} workers succeeded "
            f"({len(results)} results received)"
        )

        return results


# ---------------------------------------------------------------------------
# Module-level worker process target (must be picklable)
# ---------------------------------------------------------------------------


def _nvidia_worker_process(
    worker_id: int,
    kernel_code: str,
    known_time: float,
    parent_id: str,
    problem_file: Path,
    test_code: str,
    workdir: Path,
    log_dir: Path,
    result_queue: mp.Queue,
    benchmark_lock: Any,
    profiling_semaphore: Any,
    pytorch_baseline: float,
    bottleneck_id: int,
    openai_model: str,
    high_reasoning_effort: bool,
    bottleneck_override: str | None,
    worker_kwargs: dict,
    prior_history: list[dict],
    prior_reflexions: list[dict],
) -> None:
    """Worker process function for NVIDIA GPUs.

    Runs in a separate process to optimise a single kernel variant using
    NCU profiling and CUDA benchmarking.
    """
    import sys

    kernel_agent_path = Path(__file__).parent.parent.parent
    if str(kernel_agent_path) not in sys.path:
        sys.path.insert(0, str(kernel_agent_path))

    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker

        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(problem_file, workdir / "problem.py")

        worker = OptimizationWorker(
            worker_id=worker_id,
            workdir=workdir,
            log_dir=log_dir,
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            bottleneck_id=bottleneck_id,
            benchmark_lock=benchmark_lock,
            profiling_semaphore=profiling_semaphore,
            pytorch_baseline_time=pytorch_baseline,
            bottleneck_override=bottleneck_override,
            prior_history=prior_history,
            prior_reflexions=prior_reflexions,
            **worker_kwargs,
        )

        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,
        )

        attempt_data = metrics.get("last_attempt")
        reflexion_data = metrics.get("last_reflexion")

        result_queue.put(
            {
                "success": success,
                "worker_id": worker_id,
                "kernel_code": best_kernel,
                "time_ms": metrics.get("best_time_ms", float("inf")),
                "parent_id": parent_id,
                "attempt": attempt_data,
                "reflexion": reflexion_data,
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "worker_id": worker_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


# ---------------------------------------------------------------------------
# Worker-level NVIDIA implementations
#
# These wrap the concrete NVIDIA/CUDA classes that live deeper in the
# codebase.  They use lazy delegation so that heavy imports (NCU, GPU
# detection, OpenAI embeddings) only happen on first use, and so that
# the registry can instantiate them at manager time with the shared
# kwargs bag.
# ---------------------------------------------------------------------------


class NvidiaAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Looks up NVIDIA GPU specs via ``get_gpu_specs``."""

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )

        return get_gpu_specs(device_name) if device_name else get_gpu_specs()


class NvidiaKernelProfiler(KernelProfilerBase):
    """Wraps :class:`KernelProfiler` (NCU-based) with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        profiling_semaphore: Any | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else Path(".")
        self._profiling_semaphore = profiling_semaphore
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
                KernelProfiler,
            )

            artifacts_dir = self._log_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._delegate = KernelProfiler(
                logger=self._logger,
                artifacts_dir=artifacts_dir,
                logs_dir=self._log_dir,
                profiling_semaphore=self._profiling_semaphore,
            )
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

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
                RooflineAnalyzer,
            )

            self._delegate = RooflineAnalyzer(logger=self._logger)
        return self._delegate

    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        return self._get_delegate().analyze(ncu_metrics)

    def should_stop(self, result: Any) -> tuple[bool, str]:
        return self._get_delegate().should_stop(result)

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
        openai_model: str = "claude-opus-4.5",
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._openai_model = openai_model
        self._delegate: Any | None = None
        # Orchestrator accesses ``bottleneck_analyzer.roofline`` directly.
        self.roofline = NvidiaRooflineAnalyzer(logger=self._logger)

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
                get_gpu_specs,
            )
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            provider = get_model_provider(self._openai_model)
            gpu_specs = get_gpu_specs()

            self._delegate = BottleneckAnalyzer(
                provider=provider,
                model=self._openai_model,
                gpu_specs=gpu_specs,
                logger=self._logger,
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

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.RAG_based_prescriber import (
                RAGPrescriber,
            )

            self._delegate = RAGPrescriber(logger=self._logger)
        return self._delegate

    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        return self._get_delegate().retrieve(query)

    def build_context(self, opt_node: Any) -> str:
        return self._get_delegate().build_context(opt_node)
