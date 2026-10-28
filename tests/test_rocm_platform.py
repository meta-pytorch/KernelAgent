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

"""ROCm platform wiring that must hold without an AMD GPU."""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import get_gpu_specs
from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
    _format_gpu_specs,
)
from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import ROCmRooflineAnalyzer
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    _get_triton_kernel_metrics,
    profiler_is_rocm,
    should_disable_profiling,
)
from triton_kernel_agent.platform.registry import registry
from triton_kernel_agent.platform.rocm import (
    ROCmBottleneckAnalyzer,
    ROCmKernelProfilerWrapper,
    ROCmRooflineAnalyzerWrapper,
    ROCmWorkerRunner,
    _normalize_amd_gpu_name,
    _rocm_worker_process,
)
from triton_kernel_agent.platform_config import get_platform, resolve_test_timeout_s

_ROCM_COMPONENTS = (
    "verifier",
    "benchmarker",
    "worker_runner",
    "specs_provider",
    "profiler",
    "roofline_analyzer",
    "bottleneck_analyzer",
    "rag_prescriber",
)

_FLAT_ROCM_METRICS = {
    "compute_sol_pct": 70.0,
    "memory_sol_pct": 20.0,
    "valu_utilization_pct": 70.0,
    "memory_bound_pct": 20.0,
    "tcc_cache_hit_rate_pct": 90.0,
    "_raw_counters": {"SQ_INSTS_VALU": 7},
}


def test_registry_lists_rocm_for_every_component():
    for component in _ROCM_COMPONENTS:
        assert "rocm" in registry.list_implementations(component)


def test_registry_profiler_is_the_wrapper():
    profiler = registry.create("profiler", "rocm")
    assert isinstance(profiler, ROCmKernelProfilerWrapper)
    assert profiler.platform == "rocm"
    assert profiler_is_rocm(profiler)
    assert profiler_is_rocm(SimpleNamespace(platform="cuda")) is False


def test_ncu_bin_path_is_accepted_as_rocprof_binary():
    profiler = ROCmKernelProfilerWrapper(ncu_bin_path="/opt/rocm/bin/rocprofv3")
    assert profiler._rocprof_bin_path == "/opt/rocm/bin/rocprofv3"
    explicit = ROCmKernelProfilerWrapper(
        rocprof_bin_path="/usr/bin/rocprof",
        ncu_bin_path="/usr/bin/ncu",
    )
    assert explicit._rocprof_bin_path == "/usr/bin/rocprof"


def test_roofline_accepts_shared_ncu_metrics_keyword():
    wrapper = ROCmRooflineAnalyzerWrapper()
    result = wrapper.analyze(ncu_metrics=_FLAT_ROCM_METRICS)
    assert result.efficiency_pct == 70.0
    assert result.bottleneck == "compute"

    direct = ROCmRooflineAnalyzer()
    assert direct.config.early_stop is False
    stop, _reason = direct.should_stop(result)
    assert stop is False


def test_roofline_rejects_scalar_metrics():
    analyzer = ROCmRooflineAnalyzer()
    with pytest.raises(TypeError):
        analyzer.analyze(ncu_metrics=70.0)  # type: ignore[arg-type]


def test_flat_rocprof_metrics_are_not_unwrapped_to_a_scalar():
    flat = _get_triton_kernel_metrics(_FLAT_ROCM_METRICS)
    assert isinstance(flat, dict)
    assert flat["compute_sol_pct"] == 70.0
    assert "_raw_counters" not in flat

    nested = _get_triton_kernel_metrics(
        {
            "void at::add": {"compute_sol_pct": 1.0},
            "triton_kernel": {"compute_sol_pct": 40.0},
        }
    )
    assert nested == {"compute_sol_pct": 40.0}


def test_synthetic_mode_is_rocm_only():
    rocm = SimpleNamespace(platform="rocm")
    cuda = SimpleNamespace(platform="nvidia")
    assert should_disable_profiling(rocm, profiling_available=False)
    assert should_disable_profiling(rocm, profiling_available=True) is False
    assert should_disable_profiling(cuda, profiling_available=False) is False


def test_timeout_default_is_platform_specific():
    assert resolve_test_timeout_s(None, "cuda") == 30
    assert resolve_test_timeout_s(None, "xpu") == 30
    assert resolve_test_timeout_s(None, "rocm") == 300
    assert resolve_test_timeout_s(30, "rocm") == 30
    assert resolve_test_timeout_s(None, get_platform("rocm")) == 300


def test_gfx_isa_names_map_onto_spec_database_keys():
    assert _normalize_amd_gpu_name("gfx942") == "AMD Instinct MI300X"
    assert _normalize_amd_gpu_name("gfx950") == "AMD Instinct MI350X"
    assert _normalize_amd_gpu_name("Instinct MI300X OAM") == "AMD Instinct MI300X"
    specs = get_gpu_specs(_normalize_amd_gpu_name("gfx942"))
    assert specs is not None
    assert specs["cu_count"] == 304


def test_prompt_uses_cu_labels_and_marks_estimated_specs():
    mi300a = get_gpu_specs("AMD Instinct MI300A")
    assert mi300a is not None
    text = _format_gpu_specs(mi300a)
    assert "CU Count" in text
    assert "SM Count" not in text
    assert "HBM3 unified" in text

    mi350 = get_gpu_specs("AMD Instinct MI350X")
    assert mi350 is not None
    assert "Specs confidence: estimated" in _format_gpu_specs(mi350)


def test_bottleneck_analyzer_accepts_num_bottlenecks():
    params = inspect.signature(ROCmBottleneckAnalyzer.__init__).parameters
    assert "num_bottlenecks" in params


def test_yaml_platform_kwargs_select_rocprof_binary(tmp_path, monkeypatch):
    gpu_name = "GPU_NAME"
    rocprof_bin_path = "/path/to/rocprofv3"
    monkeypatch.setattr(
        "triton_kernel_agent.opt_worker.get_model_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.ROCmAcceleratorSpecsProvider.get_specs",
        lambda _self, device_name=None: {"name": device_name},
    )
    cfg = tmp_path / "amd.yaml"
    cfg.write_text(
        f"""\
strategy: greedy
num_workers: 1
gpu_name: {gpu_name}
target_platform: rocm
platform:
  verifier: noop
  benchmarker: noop
  worker_runner: noop
  profiler: rocm
  specs_provider: rocm
platform_kwargs:
  rocprof_bin_path: {rocprof_bin_path}
"""
    )
    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.opt_worker import OptimizationWorker

    manager = OptimizationManager(
        config=str(cfg),
        log_dir=tmp_path / "logs",
        gpu_ids=[0],
    )
    assert manager.worker_kwargs["gpu_name"] == gpu_name
    assert manager._mgr_profiler._rocprof_bin_path == rocprof_bin_path

    # OptimizationWorker always constructs Benchmark, and that module imports
    # torch. CI does not install torch, so stand in for the class.
    benchmark_mod = ModuleType(
        "triton_kernel_agent.opt_worker_component.benchmarking.benchmark"
    )

    class _Benchmark:
        def __init__(self, *_args, **_kwargs):
            pass

    benchmark_mod.Benchmark = _Benchmark
    monkeypatch.setitem(
        sys.modules,
        "triton_kernel_agent.opt_worker_component.benchmarking.benchmark",
        benchmark_mod,
    )

    worker = OptimizationWorker(
        worker_id=0,
        workdir=tmp_path / "worker",
        log_dir=tmp_path / "worker-logs",
        **manager.worker_kwargs,
    )
    assert worker.gpu_name == gpu_name
    assert worker.profiler._rocprof_bin_path == rocprof_bin_path


def test_worker_runner_matches_multi_gpu_contract():
    params = inspect.signature(ROCmWorkerRunner.__init__).parameters
    assert "gpu_ids" in params
    assert "gpu_locks" in params
    source = inspect.getsource(ROCmWorkerRunner.run_workers)
    assert "remaining_workers" in source
    assert "baseline_metrics" in source
    process_source = inspect.getsource(_rocm_worker_process)
    assert "CUDA_VISIBLE_DEVICES" in process_source
    assert "baseline_metrics=baseline_metrics" in process_source
    assert 'worker_kwargs.pop("target_platform"' not in process_source
