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

"""AMD ROCm / HIP implementations of platform interfaces.

These wrap the ROCm-specific code (rocprof profiling, AMD GPU specs,
ROCm roofline analysis) behind the same abstract interfaces used by the
NVIDIA CUDA path in ``triton_kernel_agent/platform/nvidia.py``.

ROCm-specific notes
-------------------
- ``torch.cuda`` works on ROCm via the HIP compatibility layer, so CUDA
  event timing, ``torch.cuda.synchronize()``, etc. all work unchanged.
- Profiling uses ``rocprof`` instead of NVIDIA NCU.
- GPU names are detected via ``torch.cuda.get_device_name()``; AMD Instinct
  GPUs return strings like ``"AMD Instinct MI300X"``.
- Wavefront size is 64 (vs 32 for NVIDIA warps).
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
# GPU name detection helper
# ---------------------------------------------------------------------------


def detect_amd_gpu_name() -> str | None:
    """Detect the AMD GPU name via ``torch.cuda.get_device_name()``.

    On ROCm, PyTorch's CUDA layer is backed by HIP, and
    ``torch.cuda.get_device_name()`` returns the real AMD GPU name
    (e.g., ``"AMD Instinct MI300X"``).

    Returns:
        GPU name string, or ``None`` if no GPU is found or the GPU is NVIDIA.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        if "AMD" in name or "Instinct" in name or "Radeon" in name:
            return name
        return None
    except Exception:
        return None


def _normalize_amd_gpu_name(raw_name: str) -> str:
    """Normalize the raw torch GPU name to a key in the GPU specs database.

    ``torch.cuda.get_device_name()`` may return strings like
    ``"AMD Instinct MI300X OAM"`` or ``"Instinct MI300X"``.
    We strip the OAM/PCIe suffix and ensure the ``"AMD Instinct"`` prefix.

    Args:
        raw_name: Raw name from ``torch.cuda.get_device_name()``.

    Returns:
        Normalized name string.
    """
    # Strip trailing form-factor suffixes
    for suffix in (" OAM", " PCIe", " SXM", " NVL"):
        if raw_name.endswith(suffix):
            raw_name = raw_name[: -len(suffix)].strip()

    # Ensure "AMD Instinct" prefix
    if not raw_name.startswith("AMD "):
        raw_name = "AMD " + raw_name

    return raw_name


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ROCmVerifier(KernelVerifier):
    """Verifies kernel correctness on AMD GPUs.

    Uses the same ``VerificationWorker`` as the NVIDIA path — ROCm exposes
    ``torch.cuda`` via HIP so no changes are needed at the verification level.
    """

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
            target_platform="rocm",
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


class ROCmBenchmarker(KernelBenchmarker):
    """Benchmarks Triton kernels on AMD GPUs.

    Uses ``triton.testing.do_bench`` / ``torch.cuda.Event`` timing, which
    works on ROCm via the HIP compatibility layer.
    """

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

    def benchmark_kernel(self, kernel_code: str, problem_file: Path) -> float:
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

    def benchmark_reference(self, problem_file: Path) -> float:
        benchmarker = self._get_benchmarker()
        result = benchmarker.benchmark_pytorch(problem_file)
        pytorch_time = result.get("time_ms", float("inf"))

        if pytorch_time != float("inf"):
            self.logger.info(f"PyTorch baseline: {pytorch_time:.4f}ms")

        return pytorch_time

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        benchmarker = self._get_benchmarker()
        result = benchmarker.benchmark_pytorch_compile(problem_file)
        compile_time = result.get("time_ms", float("inf"))

        if compile_time != float("inf"):
            self.logger.info(f"PyTorch compile baseline: {compile_time:.4f}ms")

        return compile_time


# ---------------------------------------------------------------------------
# Worker runner
# ---------------------------------------------------------------------------


class ROCmWorkerRunner(WorkerRunner):
    """Spawns ``OptimizationWorker`` processes on AMD GPUs."""

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
        test_code: list[str],
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
                i,
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

            p = mp.Process(target=_rocm_worker_process, args=args)
            p.start()
            workers.append(p)

        worker_timeout = 1800  # 30 minutes (longer due to two rocprof passes)
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

        results: list[dict[str, Any]] = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        result_queue.close()
        result_queue.join_thread()

        successful = sum(1 for r in results if r.get("success"))
        self.logger.info(
            f"Round {round_num}: {successful}/{len(candidates)} workers succeeded "
            f"({len(results)} results received)"
        )
        return results


def _rocm_worker_process(
    worker_id: int,
    kernel_code: str,
    known_time: float,
    parent_id: str,
    problem_file: Path,
    test_code: list[str],
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
    """Worker process function for AMD ROCm GPUs."""
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
            target_platform="rocm",
            **worker_kwargs,
        )

        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,
        )

        result_queue.put(
            {
                "success": success,
                "worker_id": worker_id,
                "kernel_code": best_kernel,
                "time_ms": metrics.get("best_time_ms", float("inf")),
                "parent_id": parent_id,
                "attempt": metrics.get("last_attempt"),
                "reflexion": metrics.get("last_reflexion"),
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
# Worker-level ROCm implementations
# ---------------------------------------------------------------------------


class ROCmAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Looks up AMD GPU specs from the GPU specs database.

    Auto-detects the GPU name via ``torch.cuda.get_device_name()`` when
    ``device_name`` is not provided.
    """

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
            get_gpu_specs,
        )

        if not device_name:
            raw = detect_amd_gpu_name()
            if raw is None:
                raise ValueError(
                    "Could not detect AMD GPU name. Provide gpu_name explicitly."
                )
            device_name = _normalize_amd_gpu_name(raw)

        specs = get_gpu_specs(device_name)
        if specs is None:
            # Try normalization in case the raw name was passed directly
            normalized = _normalize_amd_gpu_name(device_name)
            specs = get_gpu_specs(normalized)

        if specs is None:
            raise ValueError(
                f"AMD GPU '{device_name}' not found in specs database. "
                "Add it to kernel_perf_agent/kernel_opt/diagnose_prompt/gpu_specs_database.py"
            )
        return specs


class ROCmKernelProfilerWrapper(KernelProfilerBase):
    """Wraps :class:`ROCmKernelProfiler` with lazy construction."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        rocprof_bin_path: str | None = None,
        rocprof_timeout_seconds: int | None = None,
        profiling_semaphore: Any | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else Path(".")
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._rocprof_bin_path = rocprof_bin_path
        self._rocprof_timeout_seconds = rocprof_timeout_seconds
        self._profiling_semaphore = profiling_semaphore
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.profiling.rocm_kernel_profiler import (
                ROCmKernelProfiler,
            )

            artifacts_dir = self._artifacts_dir or self._log_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "logger": self._logger,
                "artifacts_dir": artifacts_dir,
                "logs_dir": self._log_dir,
                "rocprof_bin_path": self._rocprof_bin_path,
                "profiling_semaphore": self._profiling_semaphore,
            }
            if self._rocprof_timeout_seconds is not None:
                kwargs["rocprof_timeout_seconds"] = self._rocprof_timeout_seconds
            self._delegate = ROCmKernelProfiler(**kwargs)
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


class ROCmRooflineAnalyzerWrapper(RooflineAnalyzerBase):
    """Wraps :class:`ROCmRooflineAnalyzer` with lazy construction."""

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
            from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import (
                ROCmRooflineAnalyzer,
                ROCmRooflineConfig,
            )

            kwargs: dict[str, Any] = {"logger": self._logger}
            if self._roofline_config is not None:
                # Allow passing either a ROCmRooflineConfig or dict
                if isinstance(self._roofline_config, dict):
                    kwargs["config"] = ROCmRooflineConfig(**self._roofline_config)
                else:
                    kwargs["config"] = self._roofline_config
            self._delegate = ROCmRooflineAnalyzer(**kwargs)
        return self._delegate

    def analyze(self, rocm_metrics: dict[str, Any]) -> Any:
        return self._get_delegate().analyze(rocm_metrics)

    def should_stop(self, result: Any) -> tuple[bool, str]:
        return self._get_delegate().should_stop(result)

    def reset_history(self) -> None:
        self._get_delegate().reset_history()


class ROCmBottleneckAnalyzer(BottleneckAnalyzerBase):
    """Bottleneck analyzer for AMD GPUs.

    Uses the same LLM-based ``BottleneckAnalyzer`` as the NVIDIA path but
    passes AMD GPU specs and ROCm-specific counter names in the context.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "gpt-5",
        gpu_name: str | None = None,
        roofline_config: Any | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else None
        self._openai_model = openai_model
        self._gpu_name = gpu_name
        self._delegate: Any | None = None
        self.roofline = ROCmRooflineAnalyzerWrapper(
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

            gpu_name = self._gpu_name
            if not gpu_name:
                raw = detect_amd_gpu_name()
                if raw:
                    gpu_name = _normalize_amd_gpu_name(raw)
            if not gpu_name:
                raise ValueError("gpu_name is required for ROCmBottleneckAnalyzer")

            provider = get_model_provider(self._openai_model)
            gpu_specs = get_gpu_specs(gpu_name)

            self._delegate = BottleneckAnalyzer(
                provider=provider,
                model=self._openai_model,
                gpu_specs=gpu_specs,
                logs_dir=self._log_dir,
                logger=self._logger,
            )
        return self._delegate

    def analyze(
        self,
        kernel_code: str,
        rocm_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze(
            kernel_code, rocm_metrics, round_num, roofline_result
        )


class ROCmRAGPrescriber(RAGPrescriberBase):
    """RAG prescriber for ROCm — delegates to the same implementation as NVIDIA."""

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
