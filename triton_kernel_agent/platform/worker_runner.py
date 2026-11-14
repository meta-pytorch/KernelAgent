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

"""Shared optimization worker runner.

NVIDIA and ROCm use the same process pool. Each vendor supplies a pin
function that runs in the child before torch is imported. ROCm also
forces ``target_platform`` so a CUDA value in the shared kwargs cannot
override the worker.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.interfaces import WorkerRunner


def pin_nvidia_worker(worker_id: int, gpu_id: int) -> None:
    """Pin a worker to one NVIDIA device. The child inherits the variable."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(
        f"[worker {worker_id}] pinned to GPU {gpu_id} "
        f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})",
        flush=True,
    )


def pin_rocm_worker(worker_id: int, gpu_id: int) -> None:
    """Pin a worker to one HIP id and drop the CUDA and ROCR masks."""
    from triton_kernel_agent.platform.rocm import _pin_rocm_visible_devices

    _pin_rocm_visible_devices(gpu_id)
    print(
        f"[worker {worker_id}] pinned to GPU {gpu_id} "
        f"(HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']})",
        flush=True,
    )


def _optimization_worker_process(
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
    baseline_metrics: dict[str, Any] | None,
    gpu_id: int,
    pin: Callable[[int, int], None],
    target_platform: str | None,
) -> None:
    """Run one OptimizationWorker. ``pin`` runs before torch is imported."""
    import sys

    pin(worker_id, gpu_id)

    kernel_agent_path = Path(__file__).parent.parent.parent
    if str(kernel_agent_path) not in sys.path:
        sys.path.insert(0, str(kernel_agent_path))

    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker

        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(problem_file, workdir / "problem.py")

        # Copy before popping so later workers still see the shared dict.
        kwargs = dict(worker_kwargs)
        platform_arg: dict[str, Any] = {}
        if target_platform is not None:
            kwargs.pop("target_platform", None)
            platform_arg["target_platform"] = target_platform
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
            **platform_arg,
            **kwargs,
        )

        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,
            baseline_metrics=baseline_metrics,
        )

        result_queue.put(
            {
                "success": success,
                "worker_id": worker_id,
                "kernel_code": best_kernel,
                "time_ms": metrics.get("best_time_ms", float("inf")),
                "parent_id": parent_id,
                "openai_model": openai_model,
                "ptx_hash": metrics.get("best_ptx_hash"),
                "attempt": metrics.get("last_attempt"),
                "reflexion": metrics.get("last_reflexion"),
            }
        )

    except Exception as e:  # noqa: BLE001
        result_queue.put(
            {
                "success": False,
                "worker_id": worker_id,
                "openai_model": openai_model,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def nvidia_worker_process(*args: Any) -> None:
    """Picklable NVIDIA worker target. The pin is not part of the arg tuple."""
    _optimization_worker_process(
        *args, pin=pin_nvidia_worker, target_platform=None
    )


def rocm_worker_process(*args: Any) -> None:
    """Picklable ROCm worker target. ``target_platform`` stays ``rocm``."""
    _optimization_worker_process(
        *args, pin=pin_rocm_worker, target_platform="rocm"
    )


class DeviceWorkerRunner(WorkerRunner):
    """Spawn one process per candidate and drain the result queue while joining.

    ``worker_target`` is a module-level function so ``mp.Process`` can pickle it.
    """

    worker_target: Callable[..., None]

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
        gpu_ids: list[int] | None = None,
        gpu_locks: dict[int, Any] | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self.benchmark_lock = benchmark_lock
        self.profiling_semaphore = profiling_semaphore
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs
        self.gpu_ids: list[int] = list(gpu_ids) if gpu_ids else [0]
        self.gpu_locks: dict[int, Any] = (
            dict(gpu_locks) if gpu_locks else {0: benchmark_lock}
        )

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
        import queue as queue_mod

        result_queue = mp.Queue()
        workers = []

        for i, candidate in enumerate(candidates):
            workdir = self.log_dir / "workers" / f"w{i}" / f"r{round_num}"
            workdir.mkdir(parents=True, exist_ok=True)

            worker_model = candidate.get("openai_model") or self.openai_model
            baseline_metrics = candidate.get("baseline_metrics")
            # The same per-GPU lock is both the benchmark lock and the
            # profiling lock, so one GPU runs one of those at a time.
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            gpu_lock = self.gpu_locks[gpu_id]

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
                gpu_lock,
                gpu_lock,
                pytorch_baseline,
                candidate["bottleneck_id"],
                worker_model,
                self.high_reasoning_effort,
                self.bottleneck_override,
                self.worker_kwargs,
                shared_history,
                shared_reflexions,
                baseline_metrics,
                gpu_id,
            )

            process = mp.Process(target=self.worker_target, args=args)
            process.start()
            workers.append(process)

        # Drain while joining. A full mp.Queue pipe blocks the worker's
        # feeder thread, and join-then-drain deadlocks. 30 minutes covers
        # an NCU session and a multi-pass rocprof session.
        worker_timeout = 1800
        deadline = time.time() + worker_timeout
        results: list[dict[str, Any]] = []
        remaining_workers = list(workers)

        while remaining_workers and time.time() < deadline:
            while True:
                try:
                    results.append(result_queue.get_nowait())
                except queue_mod.Empty:
                    break
                except Exception:  # noqa: BLE001
                    break
            still_alive = []
            for worker in remaining_workers:
                worker.join(timeout=0.5)
                if worker.is_alive():
                    still_alive.append(worker)
                else:
                    worker.close()
            remaining_workers = still_alive

        for worker in remaining_workers:
            self.logger.warning(f"Worker {worker.pid} timed out, terminating")
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                self.logger.warning(f"Worker {worker.pid} still alive, killing")
                worker.kill()
                worker.join(timeout=2)
            worker.close()

        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue_mod.Empty:
                break
            except Exception:  # noqa: BLE001
                break

        result_queue.close()
        result_queue.join_thread()

        successful = sum(1 for result in results if result.get("success"))
        self.logger.info(
            f"Round {round_num}: {successful}/{len(candidates)} workers succeeded "
            f"({len(results)} results received)"
        )
        return results


class NvidiaWorkerRunner(DeviceWorkerRunner):
    """Spawns ``OptimizationWorker`` processes on NVIDIA GPUs."""

    # staticmethod: instance access must not bind the runner as the first
    # process argument. That extra value lands on ``pin`` and the keyword
    # pin then raises TypeError.
    worker_target = staticmethod(nvidia_worker_process)


class ROCmWorkerRunner(DeviceWorkerRunner):
    """Spawns ``OptimizationWorker`` processes on AMD GPUs.

    ``gpu_ids`` are HIP ids. The worker pin drops ``CUDA_VISIBLE_DEVICES``
    and ``ROCR_VISIBLE_DEVICES``, then sets ``HIP_VISIBLE_DEVICES``.
    """

    worker_target = staticmethod(rocm_worker_process)
