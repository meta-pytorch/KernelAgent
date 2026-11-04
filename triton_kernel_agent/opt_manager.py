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

"""Optimization Manager for parallel kernel optimization.

This module provides the OptimizationManager class that orchestrates
parallel kernel optimization using pluggable search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization

Example:
    >>> manager = OptimizationManager(
    ...     strategy="beam_search",
    ...     num_workers=4,
    ...     strategy_config={"num_top_kernels": 2, "num_bottlenecks": 2},
    ... )
    >>> result = manager.run_optimization(
    ...     initial_kernel=kernel_code,
    ...     problem_file=Path("problem.py"),
    ...     test_code=test_file.read_text(),
    ...     max_rounds=20,
    ... )
"""

import logging
import multiprocessing as mp
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.greedy import (
    GreedyStrategy,
)
from utils.config_injectable import config_injectable

# Manager-level component keys resolved by the registry
_MANAGER_LEVEL_KEYS = {"verifier", "benchmarker", "worker_runner"}


def _specs_for_baseline(
    gpu_name: str | None, *, rocm: bool
) -> dict[str, Any] | None:
    """Peak-bandwidth specs for the manager's baseline roofline.

    ROCm uses the specs provider so a missing name is detected and a raw
    name such as ``gfx950`` is normalized. An exact ``get_gpu_specs`` lookup
    misses both, and the cached baseline SOL would stay at 0.
    """
    if rocm:
        from triton_kernel_agent.platform.rocm import ROCmAcceleratorSpecsProvider

        return ROCmAcceleratorSpecsProvider().get_specs(gpu_name)
    if not gpu_name:
        return None
    from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import get_gpu_specs

    return get_gpu_specs(gpu_name)


def _query_smi(argv: list[str]) -> str | None:
    """Run a GPU listing tool. Return stdout, or None when it is unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _parse_amd_smi_gpu_ids(text: str) -> list[int]:
    """HIP ordinals from ``amd-smi list``. Prefer ``HIP_ID`` over the tool index."""
    hip_ids = [int(match) for match in re.findall(r"(?m)^\s*HIP_ID:\s*(\d+)\s*$", text)]
    if hip_ids:
        return hip_ids
    return [int(match) for match in re.findall(r"(?m)^GPU:\s*(\d+)\s*$", text)]


def _parse_nvidia_smi_gpu_ids(text: str) -> list[int]:
    """GPU indexes from ``nvidia-smi --query-gpu=index``."""
    ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids


def _uses_rocm_gpu_list(platform: dict[str, str] | str | None) -> bool:
    """True when the caller asked for a ROCm platform."""
    if platform == "rocm":
        return True
    if isinstance(platform, dict):
        if platform.get("worker_runner") == "rocm":
            return True
        return bool(platform) and all(value == "rocm" for value in platform.values())
    return False


def _uses_nvidia_gpu_list(platform: dict[str, str] | str | None) -> bool:
    """True when the caller asked for an NVIDIA platform."""
    if platform == "nvidia":
        return True
    if isinstance(platform, dict):
        if platform.get("worker_runner") == "nvidia":
            return True
        return bool(platform) and all(value == "nvidia" for value in platform.values())
    return False


def _amd_smi_hip_ids() -> list[int] | None:
    """HIP ids from ``amd-smi``. None when the binary cannot be found."""
    from triton_kernel_agent.platform.rocm import _find_amd_smi

    amd_smi = _find_amd_smi()
    if not amd_smi:
        return None
    for args in (["list", "-e"], ["list"]):
        listed = _query_smi([amd_smi, *args])
        if not listed:
            continue
        hip_ids = _parse_amd_smi_gpu_ids(listed)
        if hip_ids:
            return hip_ids
    return []


def _nvidia_gpu_ids() -> list[int]:
    """NVIDIA indexes from ``nvidia-smi``, narrowed by ``CUDA_VISIBLE_DEVICES``.

    An empty ``CUDA_VISIBLE_DEVICES`` disables NVIDIA and does not call
    ``nvidia-smi``. A non-empty value is the index list only when
    ``nvidia-smi`` lists a GPU. The variable alone does not mean the
    machine has an NVIDIA GPU.
    """
    import os

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and not cvd.strip():
        return []
    nvidia = _query_smi(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    )
    listed = _parse_nvidia_smi_gpu_ids(nvidia) if nvidia else []
    if not listed:
        return []
    if cvd:
        try:
            ids = [int(x.strip()) for x in cvd.split(",") if x.strip()]
            if ids:
                return ids
        except ValueError:
            pass
    return listed


def _select_gpus(requested: str | None = None) -> tuple[str, list[int]]:
    """Return ``("nvidia"|"rocm", gpu ids)`` for this process.

    Crucially this MUST NOT initialize the CUDA context in the manager
    process — workers are spawned via ``mp.Process`` (fork on Linux), and
    if the manager has already touched ``torch.cuda`` the children inherit
    that locked-in device list and ignore any post-fork visibility change.
    Listing goes through subprocess tools instead of ``torch.cuda``.

    A run uses NVIDIA or AMD, not both. NVIDIA is the default when no
    platform was requested. AMD is used when no NVIDIA GPU is listed, or
    when ``HIP_VISIBLE_DEVICES`` or ``ROCR_VISIBLE_DEVICES`` leaves an
    amd-smi HIP id. A requested ROCm platform reads that HIP list even
    when both masks are unset, and a mask only restricts the list. An
    empty value or an index outside that list leaves the ROCm result
    empty. An empty visibility variable disables that vendor and may be
    combined with the other vendor. A non-empty ``CUDA_VISIBLE_DEVICES``
    together with a non-empty ``HIP_VISIBLE_DEVICES`` or
    ``ROCR_VISIBLE_DEVICES`` is an error. ``CUDA_VISIBLE_DEVICES`` selects
    NVIDIA indexes only after ``nvidia-smi`` lists a GPU. The caller runs
    that vendor's platform components, so the worker matches the id list.
    Nothing listed is an empty result, and the manager stops.
    """
    import os

    # Empty is not a device list. It disables that vendor.
    cuda_names = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    rocm_names = bool(os.environ.get("HIP_VISIBLE_DEVICES")) or bool(
        os.environ.get("ROCR_VISIBLE_DEVICES")
    )
    if cuda_names and rocm_names:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES conflicts with HIP_VISIBLE_DEVICES"
            " or ROCR_VISIBLE_DEVICES"
        )

    nvidia_ids = _nvidia_gpu_ids()
    # No requested platform and no ROCm mask: NVIDIA wins without amd-smi.
    if requested != "rocm" and nvidia_ids and not rocm_names:
        return "nvidia", nvidia_ids
    hip_ids = _amd_smi_hip_ids()
    if hip_ids:
        from triton_kernel_agent.platform.rocm import visible_hip_ids

        ids = visible_hip_ids(hip_ids)
        if ids:
            return "rocm", ids
        # The ROCm mask did not match an amd-smi device. A requested ROCm
        # platform stays on that empty list.
        if requested != "rocm" and nvidia_ids:
            return "nvidia", nvidia_ids
        return "rocm", []
    if requested != "rocm" and nvidia_ids:
        return "nvidia", nvidia_ids
    if rocm_names or requested == "rocm":
        return "rocm", []
    return "nvidia", []


def _requested_platform_error(requested: str, vendor: str, ids: list[int]) -> str:
    """Error for an explicit platform that did not get its own devices."""
    if not ids:
        return f"{requested} platform was requested but no devices are visible"
    selected = "NVIDIA" if vendor == "nvidia" else "ROCm"
    return f"{requested} platform was requested but {selected} devices were selected"


def _detect_gpus() -> list[int]:
    """GPU ids from :func:`_select_gpus`."""
    return _select_gpus()[1]


@config_injectable
class OptimizationManager:
    """Manages parallel kernel optimization with pluggable strategies.

    Supports:
    - beam_search: Current default (top-N kernels × M bottlenecks)
    - greedy: Simple single-best optimization

    Platform-specific behaviour (verification, benchmarking, worker
    orchestration) is delegated to injectable components that implement
    :class:`KernelVerifier`, :class:`KernelBenchmarker`, and
    :class:`WorkerRunner`.  When these are not supplied the default
    NVIDIA / CUDA implementations are used.
    """

    def __init__(
        self,
        strategy: str = "beam_search",
        num_workers: int = 4,
        max_rounds: int = 10,
        log_dir: Path | str | None = None,
        database_path: Path | str | None = None,
        strategy_config: dict[str, Any] | None = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        bottleneck_override: str | None = None,
        platform: dict[str, str] | str | None = None,
        gpu_ids: list[int] | None = None,
        **worker_kwargs: Any,
    ):
        """Initialize the optimization manager.

        Args:
            strategy: Search strategy name ("beam_search" or "greedy")
            num_workers: Number of parallel workers
            max_rounds: Maximum optimization rounds
            log_dir: Directory for logs and artifacts
            database_path: Path for program database JSON file
            strategy_config: Strategy-specific configuration
            openai_model: Model name for LLM optimization
            high_reasoning_effort: Whether to use high reasoning effort
            bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
            platform: Platform component config.  Can be:
                - ``None`` — use ``"nvidia"`` for all components (default)
                - a string like ``"nvidia"`` — shorthand for all components
                - a dict like ``{"verifier": "nvidia", ...}`` — per-component
              When ``gpu_ids`` is omitted, the visibility rules choose the
              vendor and every component uses that vendor. A requested ROCm
              platform is not replaced with NVIDIA, and a requested NVIDIA
              platform is not replaced with ROCm; either mismatch raises.
            **worker_kwargs: Additional kwargs passed to OptimizationWorker
        """
        self.max_rounds = max_rounds
        self.log_dir = (
            Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="opt_"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs

        # Store template overrides (also stays in worker_kwargs for forwarding)
        self.templates_config = worker_kwargs.get("templates")

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize database
        db_path = (
            Path(database_path)
            if database_path
            else self.log_dir / "program_database.json"
        )
        self.database = JSONProgramDatabase(db_path)

        # Initialize strategy
        self.strategy = self._create_strategy(
            strategy, strategy_config or {}, num_workers
        )

        # Forward the strategy's bottleneck-fanout knob to workers so the
        # ``BottleneckAnalyzer`` actually requests that many ranked
        # bottlenecks from the LLM.  Without this, the strategy spawns N
        # workers per parent (with bottleneck_id ∈ {1..N}) but the analyzer
        # always returns 1, and workers with id>1 silently fall back to id=1.
        if (
            strategy_config
            and "num_bottlenecks" in strategy_config
            and "num_bottlenecks_to_request" not in self.worker_kwargs
        ):
            self.worker_kwargs["num_bottlenecks_to_request"] = strategy_config[
                "num_bottlenecks"
            ]

        # Validate worker count
        if num_workers != self.strategy.num_workers_needed:
            raise ValueError(
                f"Strategy '{strategy}' requires {self.strategy.num_workers_needed} "
                f"workers, got {num_workers}. Adjust num_workers or strategy_config."
            )

        self.num_workers = num_workers

        # Per-GPU lock pool — one lock per GPU does double duty for both
        # benchmarking and NCU profiling (the two GPU-serialized
        # operations).  Workers running on different GPUs proceed in
        # parallel; workers on the same GPU serialize.  This is the
        # "collapsed" multi-GPU design: a single shared object per GPU
        # acts as both ``benchmark_lock`` and ``profiling_semaphore``.
        if gpu_ids:
            self.gpu_ids = list(gpu_ids)
        elif platform == "noop":
            # A dry run does not need a listed GPU. gpu 0 only names the lock.
            self.gpu_ids = [0]
        else:
            # The visibility rules choose the vendor. A requested ROCm
            # platform reads the HIP ids, and the mask restricts that list.
            # An explicit platform or component dict is kept. The detected
            # vendor replaces the platform only when the caller omitted it.
            requested = "rocm" if _uses_rocm_gpu_list(platform) else None
            vendor, detected = _select_gpus(requested)
            if _uses_rocm_gpu_list(platform) and (vendor != "rocm" or not detected):
                raise RuntimeError(
                    _requested_platform_error("ROCm", vendor, detected)
                )
            if _uses_nvidia_gpu_list(platform) and (
                vendor != "nvidia" or not detected
            ):
                raise RuntimeError(
                    _requested_platform_error("NVIDIA", vendor, detected)
                )
            self.gpu_ids = detected
            if platform is None and detected:
                platform = vendor
        if not self.gpu_ids:
            raise RuntimeError("no GPU visible")
        # An explicit string or component dict stays as the caller passed it.
        # The worker still needs target_platform for the timeout and the
        # kernel guidance. Do not overwrite a value the caller set.
        if "target_platform" not in self.worker_kwargs:
            if platform == "rocm" or _uses_rocm_gpu_list(platform):
                self.worker_kwargs["target_platform"] = "rocm"
            elif platform == "nvidia" or _uses_nvidia_gpu_list(platform):
                self.worker_kwargs["target_platform"] = "cuda"
        # The manager verifies and profiles in this process. Drop a caller
        # CUDA mask only for that work, then put the variable back.
        self._hide_caller_cuda_mask = platform == "rocm" or _uses_rocm_gpu_list(
            platform
        )
        self.gpu_locks: dict[int, Any] = {g: mp.Lock() for g in self.gpu_ids}

        # Manager-level GPU work (initial-kernel verify, PyTorch baselines,
        # baseline NCU cache) runs in this process — it pins to the first
        # GPU and uses that GPU's lock as both benchmark_lock and
        # profiling_semaphore for back-compat with components that take
        # those names.
        _first_gpu = self.gpu_ids[0]
        self.benchmark_lock = self.gpu_locks[_first_gpu]
        self.profiling_semaphore = self.gpu_locks[_first_gpu]

        # Shared history across beam search iterations
        self.shared_history: list[
            dict
        ] = []  # List of serialized OptimizationAttempt dicts
        self.shared_reflexions: list[dict] = []  # List of serialized Reflexion dicts
        self.history_size: int = 10  # Max history entries to pass to workers

        # Per-parent baseline NCU/roofline cache, keyed by program_id.
        # Populated in _run_workers; shared across all sibling workers of the
        # same round (and surviving across rounds for beam members that stick).
        self._baseline_profile_cache: dict[str, dict[str, Any] | None] = {}

        # ── Platform components (resolved from registry) ─────────
        self._resolve_platform(platform)

        self.logger.info(
            f"OptimizationManager initialized: strategy={strategy}, "
            f"workers={num_workers}, gpus={self.gpu_ids}"
        )

    # ------------------------------------------------------------------
    # Platform resolution
    # ------------------------------------------------------------------

    def _resolve_platform(self, platform: dict[str, str] | str | None) -> None:
        """Resolve platform components from the :mod:`platform.registry`.

        Manager-level components (``verifier``, ``benchmarker``,
        ``worker_runner``) are instantiated and stored on *self*.
        Worker-level component names are forwarded to worker processes
        via ``self.worker_kwargs["platform_config"]`` so each worker
        can resolve its own instances from the registry.

        Additionally, the manager resolves its *own* ``profiler`` and
        ``roofline_analyzer`` instances (without removing them from the
        worker config) so it can profile baseline kernels once per round
        and share the result across sibling workers.
        """
        from triton_kernel_agent.platform.registry import registry

        # Expand shorthand → full per-component dict
        if platform is None or isinstance(platform, str):
            impl = platform or "nvidia"
            config = {k: impl for k in registry.list_components()}
        else:
            config = dict(platform)

        # Split manager vs worker keys
        mgr_config = {k: v for k, v in config.items() if k in _MANAGER_LEVEL_KEYS}
        worker_config = {
            k: v for k, v in config.items() if k not in _MANAGER_LEVEL_KEYS
        }

        # Resolve manager-level components (shared kwargs bag is
        # filtered per-factory by the registry)
        components = registry.create_from_config(
            mgr_config,
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            bottleneck_override=self.bottleneck_override,
            worker_kwargs=self.worker_kwargs,
            gpu_ids=self.gpu_ids,
            gpu_locks=self.gpu_locks,
        )
        self.verifier = components["verifier"]
        self.benchmarker = components["benchmarker"]
        self.worker_runner = components["worker_runner"]

        # Resolve a manager-owned profiler + roofline_analyzer for the
        # baseline-caching step.  These coexist with the worker-level
        # instances (workers still build their own from ``worker_config``).
        self._mgr_profiler: Any | None = None
        self._mgr_roofline: Any | None = None
        for key, setter_attr in (
            ("profiler", "_mgr_profiler"),
            ("roofline_analyzer", "_mgr_roofline"),
        ):
            impl_name = config.get(key)
            if impl_name and registry.has(key, impl_name):
                try:
                    baseline_kwargs: dict[str, Any] = {
                        "logger": self.logger,
                        "log_dir": self.log_dir,
                        "artifacts_dir": self.log_dir / "baseline_profiles",
                        "profiling_semaphore": self.profiling_semaphore,
                    }
                    if self.worker_kwargs.get("ncu_bin_path"):
                        baseline_kwargs["ncu_bin_path"] = self.worker_kwargs[
                            "ncu_bin_path"
                        ]
                    # Same bag the workers unpack, so a YAML rocprof_bin_path
                    # applies to the cached baseline profile too.
                    baseline_kwargs.update(
                        self.worker_kwargs.get("platform_kwargs") or {}
                    )
                    setattr(
                        self,
                        setter_attr,
                        registry.create(key, impl_name, **baseline_kwargs),
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to create manager-level {key}: {e}. "
                        f"Baseline NCU caching disabled; workers will profile "
                        f"their baselines individually."
                    )

        # Propagate worker-level config (string names) to worker
        # processes — each worker resolves its own instances via the
        # registry so there are no pickling issues.
        if worker_config:
            self.worker_kwargs["platform_config"] = worker_config

    # ------------------------------------------------------------------
    # Logging / strategy helpers (unchanged)
    # ------------------------------------------------------------------

    def _setup_logging(self) -> logging.Logger:
        """Setup manager logging."""
        logger = logging.getLogger("OptimizationManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / "manager.log")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console)

        return logger

    def _create_strategy(
        self, name: str, config: dict[str, Any], num_workers: int
    ) -> SearchStrategy:
        """Create the search strategy.

        Args:
            name: Strategy name
            config: Strategy-specific configuration
            num_workers: Number of workers

        Returns:
            Configured SearchStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        if name == "beam_search":
            return BeamSearchStrategy(
                num_top_kernels=config.get("num_top_kernels", 2),
                num_bottlenecks=config.get("num_bottlenecks", 2),
                database=self.database,
                logger=self.logger,
                models=config.get("models"),
                samples_per_prompt=config.get("samples_per_prompt", 1),
                num_expanding_parents=config.get("num_expanding_parents"),
            )
        elif name == "greedy":
            return GreedyStrategy(
                database=self.database,
                max_no_improvement=config.get("max_no_improvement", 5),
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unknown strategy: {name}. Use 'beam_search' or 'greedy'")

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def run_optimization(
        self,
        initial_kernel: str,
        problem_file: Path | str,
        test_code: str | list[str],
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization with the configured strategy.

        Args:
            initial_kernel: Starting kernel code
            problem_file: Path to problem.py defining Model and get_inputs()
            test_code: Test code for correctness verification. Can be a single
                string or a list.
            max_rounds: Override max_rounds (optional)
            **kwargs: Additional kwargs (reserved for future use)

        Returns:
            Dict with:
                - success: bool
                - kernel_code: str | None
                - best_time_ms: float
                - total_rounds: int
                - top_kernels: list[dict]
        """
        max_rounds = max_rounds or self.max_rounds
        problem_file = Path(problem_file)

        # Normalize test_code to list
        if isinstance(test_code, str):
            test_code = [test_code]

        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZATION")
        self.logger.info("=" * 80)

        # Initialize strategy with starting kernel
        initial_entry = ProgramEntry(
            program_id="initial",
            kernel_code=initial_kernel,
            metrics=ProgramMetrics(time_ms=float("inf")),
            problem_id=str(problem_file),
        )
        self.strategy.initialize(initial_entry)

        # Verify initial kernel correctness before investing in benchmarks/optimization
        if not self._verify_initial_kernel(initial_kernel, problem_file, test_code):
            return {
                "success": False,
                "kernel_code": None,
                "best_time_ms": float("inf"),
                "total_rounds": 0,
                "top_kernels": [],
                "error": "Initial kernel failed correctness verification",
            }

        # Benchmark PyTorch baseline once (before spawning workers)
        pytorch_baseline = self._benchmark_pytorch_baseline(problem_file)

        # Benchmark torch.compile baseline
        pytorch_compile_time = self._benchmark_pytorch_compile(problem_file)

        # Benchmark the initial kernel
        initial_kernel_time = self._benchmark_initial_kernel(
            initial_kernel, problem_file
        )

        # Round loop
        round_num = 0
        for round_num in range(1, max_rounds + 1):
            self.logger.info("")
            self.logger.info(f"{'=' * 20} ROUND {round_num}/{max_rounds} {'=' * 20}")

            # 1. Get candidates from strategy
            candidates = self.strategy.select_candidates(round_num)
            if not candidates:
                self.logger.warning("No candidates to explore, terminating")
                break

            # 2. Spawn workers
            results = self._run_workers(
                candidates,
                round_num,
                problem_file,
                test_code,
                pytorch_baseline,
            )

            # 3. Update strategy with results
            self.strategy.update_with_results(results, round_num)

            # Log per-round winner summary
            successful = [r for r in results if r.get("success")]
            if successful:
                best = min(successful, key=lambda r: r.get("time_ms", float("inf")))
                self.logger.info(
                    f"Round {round_num} best: worker {best['worker_id']} at {best['time_ms']:.4f} ms"
                )
            else:
                self.logger.info(f"Round {round_num}: no successful workers")

            # 4. Check termination
            if self.strategy.should_terminate(round_num, max_rounds):
                self.logger.info("Strategy signaled termination")
                break

        # Return best result
        best = self.strategy.get_best_program()

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("=" * 80)

        if best:
            self.logger.info(f"Best time: {best.metrics.time_ms:.4f}ms")
            if initial_kernel_time != float("inf") and best.metrics.time_ms > 0:
                speedup = initial_kernel_time / best.metrics.time_ms
                self.logger.info(f"Speedup vs initial kernel: {speedup:.2f}x")
            if pytorch_baseline != float("inf") and best.metrics.time_ms > 0:
                speedup_pt = pytorch_baseline / best.metrics.time_ms
                self.logger.info(f"Speedup vs PyTorch eager: {speedup_pt:.2f}x")

        return {
            "success": best is not None and best.metrics.time_ms != float("inf"),
            "kernel_code": best.kernel_code if best else None,
            "best_time_ms": best.metrics.time_ms if best else float("inf"),
            "total_rounds": round_num,
            "pytorch_baseline_ms": pytorch_baseline,
            "pytorch_compile_ms": pytorch_compile_time,
            "initial_kernel_time_ms": initial_kernel_time,
            "top_kernels": [
                {
                    "kernel_code": p.kernel_code,
                    "time_ms": p.metrics.time_ms,
                    "generation": p.generation,
                    "program_id": p.program_id,
                }
                for p in self.database.get_top_k(5)
            ],
        }

    # ------------------------------------------------------------------
    # Thin delegates to platform components
    # ------------------------------------------------------------------

    @contextmanager
    def _without_caller_cuda_mask(self) -> Iterator[None]:
        """Hide ``CUDA_VISIBLE_DEVICES`` while this process uses a ROCm GPU.

        Workers pin their own environment. Restoring the variable here leaves
        the process that constructed the manager unchanged after the call.
        """
        import os

        if not self._hide_caller_cuda_mask:
            yield
            return
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            yield
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def _benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark the eager reference implementation."""
        with self._without_caller_cuda_mask():
            return self.benchmarker.benchmark_reference(problem_file)

    def _verify_initial_kernel(
        self,
        initial_kernel: str,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        """Verify the initial kernel passes correctness before optimization."""
        with self._without_caller_cuda_mask():
            return self.verifier.verify(initial_kernel, problem_file, test_code)

    def _benchmark_initial_kernel(
        self, initial_kernel: str, problem_file: Path
    ) -> float:
        """Benchmark the initial kernel before optimization begins."""
        with self._without_caller_cuda_mask():
            return self.benchmarker.benchmark_kernel(initial_kernel, problem_file)

    def _benchmark_pytorch_compile(self, problem_file: Path) -> float:
        """Benchmark the compiler-optimized reference."""
        with self._without_caller_cuda_mask():
            return self.benchmarker.benchmark_reference_compiled(problem_file)

    def _run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: list[str],
        pytorch_baseline: float,
    ) -> list[dict[str, Any]]:
        """Spawn workers for each candidate and collect results."""
        # Profile each distinct parent kernel once and share the NCU/roofline
        # result across sibling workers.  This avoids repeating an expensive,
        # semaphore-serialized NCU run for every (bottleneck, model) fanout.
        self._populate_baseline_cache(candidates, problem_file, round_num)
        for cand in candidates:
            parent_id = cand["parent"].program_id
            cand["baseline_metrics"] = self._baseline_profile_cache.get(parent_id)

        results = self.worker_runner.run_workers(
            candidates=candidates,
            round_num=round_num,
            problem_file=problem_file,
            test_code=test_code,
            pytorch_baseline=pytorch_baseline,
            shared_history=(
                self.shared_history[-self.history_size :] if self.shared_history else []
            ),
            shared_reflexions=(
                self.shared_reflexions[-self.history_size :]
                if self.shared_reflexions
                else []
            ),
        )

        # Collect history and reflexions from worker results
        for r in results:
            if r.get("attempt"):
                self.shared_history.append(r["attempt"])
            if r.get("reflexion"):
                self.shared_reflexions.append(r["reflexion"])

        # Log errors from failed workers
        for r in results:
            if not r.get("success") and r.get("error"):
                self.logger.error(
                    f"Worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    self.logger.debug(f"Traceback:\n{r.get('traceback')}")

        return results

    def _populate_baseline_cache(
        self,
        candidates: list[dict[str, Any]],
        problem_file: Path,
        round_num: int,
    ) -> None:
        """Profile each distinct parent kernel once and cache the result.

        The cache is keyed by ``parent.program_id`` and persists across
        rounds, so a beam member that survives multiple rounds is profiled
        at most once.  If the manager-level profiler or roofline analyzer
        is unavailable, this is a no-op and workers fall back to profiling
        their own baselines.
        """
        if self._mgr_profiler is None or self._mgr_roofline is None:
            return
        # The profiled process inherits this environment. Hide the caller's
        # CUDA mask for the profile, then restore it.
        with self._without_caller_cuda_mask():
            self._populate_baseline_cache_uncached(candidates, problem_file, round_num)

    def _populate_baseline_cache_uncached(
        self,
        candidates: list[dict[str, Any]],
        problem_file: Path,
        round_num: int,
    ) -> None:
        from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
            _get_triton_kernel_metrics,
            _sol_log_clause,
        )

        baseline_dir = self.log_dir / "baseline_profiles"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Collect (program_id, kernel_code) for parents we haven't cached yet.
        # De-dup by program_id since many candidates share the same parent.
        unseen: dict[str, str] = {}
        for cand in candidates:
            parent = cand["parent"]
            pid = parent.program_id
            if pid not in self._baseline_profile_cache and pid not in unseen:
                unseen[pid] = parent.kernel_code

        for pid, kernel_code in unseen.items():
            try:
                kernel_file = baseline_dir / f"{pid}.py"
                kernel_file.write_text(kernel_code)

                profiler_results = self._mgr_profiler.profile_kernel(
                    kernel_file, problem_file, round_num
                )
                if profiler_results is None or not getattr(
                    profiler_results, "metrics", None
                ):
                    self._baseline_profile_cache[pid] = None
                    self.logger.warning(
                        f"Baseline profile failed for parent {pid}; "
                        f"workers will profile their own baselines."
                    )
                    continue

                ncu_metrics = profiler_results.metrics
                flat_metrics = _get_triton_kernel_metrics(ncu_metrics)
                from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import (
                    with_peak_bandwidth,
                )

                rocm_baseline = self._hide_caller_cuda_mask or (
                    self.worker_kwargs.get("target_platform") == "rocm"
                )
                specs = _specs_for_baseline(
                    self.worker_kwargs.get("gpu_name"), rocm=rocm_baseline
                )
                flat_metrics = with_peak_bandwidth(flat_metrics, specs)
                roofline_result = self._mgr_roofline.analyze(ncu_metrics=flat_metrics)

                self._baseline_profile_cache[pid] = {
                    "efficiency_pct": roofline_result.efficiency_pct,
                    "compute_sol_pct": roofline_result.compute_sol_pct,
                    "compute_sol_measured": getattr(
                        roofline_result, "compute_sol_measured", True
                    ),
                    "memory_sol_pct": roofline_result.memory_sol_pct,
                    "memory_sol_measured": getattr(
                        roofline_result, "memory_sol_measured", True
                    ),
                    "bottleneck": roofline_result.bottleneck,
                    "roofline_result": roofline_result,
                    "ncu_metrics": ncu_metrics,
                }
                self.logger.info(
                    f"Baseline profiled for parent {pid}: "
                    f"{_sol_log_clause(roofline_result)}"
                )
            except Exception as e:
                self._baseline_profile_cache[pid] = None
                self.logger.warning(
                    f"Baseline profile errored for parent {pid}: {e}; "
                    f"workers will profile their own baselines."
                )
