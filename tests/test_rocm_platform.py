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
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import get_gpu_specs
from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
    _format_gpu_specs,
)
from kernel_perf_agent.kernel_opt.profiler.rocprof_profiler import (
    _merge_rocprof_outputs,
    _run_rocprof_pmc,
    _run_rocprof_stats,
    _run_rocprofv3_pmc_passes,
)
from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import ROCmRooflineAnalyzer
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    _get_triton_kernel_metrics,
    profiler_is_rocm,
    should_disable_profiling,
    should_profile_final_kernel,
)
from triton_kernel_agent.platform.nvidia import NvidiaRooflineAnalyzer
from triton_kernel_agent.platform.registry import registry
from triton_kernel_agent.platform.rocm import (
    ROCmBottleneckAnalyzer,
    ROCmKernelProfilerWrapper,
    ROCmRooflineAnalyzerWrapper,
    _amd_smi_gpu_record,
    _normalize_amd_gpu_name,
    _sku_for_torch_name,
    sku_name_for_gfx90a_chip,
    sku_name_for_gfx942_chip,
)
from triton_kernel_agent.platform_config import get_platform, resolve_test_timeout_s
from triton_kernel_agent.prompt_manager import PromptManager

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
    assert _normalize_amd_gpu_name("gfx908") == "AMD Instinct MI100"
    assert _normalize_amd_gpu_name("gfx90a") == "AMD Instinct MI250X"
    assert _normalize_amd_gpu_name("gfx90a:sramecc+:xnack-") == "AMD Instinct MI250X"
    assert _normalize_amd_gpu_name("Instinct MI300X OAM") == "AMD Instinct MI300X"
    assert _normalize_amd_gpu_name("Instinct MI210 PCIe") == "AMD Instinct MI210"
    assert (
        _normalize_amd_gpu_name("AMD Instinct MI210 (gfx90a)") == "AMD Instinct MI210"
    )
    assert _normalize_amd_gpu_name("Instinct MI250 OAM") == "AMD Instinct MI250"
    assert _normalize_amd_gpu_name("AMD Instinct MI250X (gfx90a)") == (
        "AMD Instinct MI250X"
    )
    specs = get_gpu_specs(_normalize_amd_gpu_name("gfx942"))
    assert specs is not None
    assert specs["cu_count"] == 304

    mi100 = get_gpu_specs("AMD Instinct MI100")
    assert mi100 is not None
    assert mi100["cu_count"] == 120
    assert mi100["gfx_target"] == "gfx908"
    assert mi100["peak_bf16_tflops"] == 92.3

    mi210 = get_gpu_specs("AMD Instinct MI210")
    assert mi210 is not None
    assert mi210["cu_count"] == 104
    assert mi210["memory_gb"] == 64
    assert mi210["gfx_target"] == "gfx90a"

    mi250 = get_gpu_specs("AMD Instinct MI250")
    assert mi250 is not None
    assert mi250["cu_count"] == 208
    assert mi250["memory_gb"] == 128


def test_gfx90a_chip_id_separates_mi210_and_mi250():
    assert sku_name_for_gfx90a_chip(0x740F, 104) == "AMD Instinct MI210"
    assert sku_name_for_gfx90a_chip(0x7410, 104) == "AMD Instinct MI210"
    assert sku_name_for_gfx90a_chip(0x7408, 110) == "AMD Instinct MI250X"
    assert sku_name_for_gfx90a_chip(0x740C, 104) == "AMD Instinct MI250"
    assert sku_name_for_gfx90a_chip(0x740C, 208) == "AMD Instinct MI250"
    assert sku_name_for_gfx90a_chip(0x740C, 110) == "AMD Instinct MI250X"
    assert sku_name_for_gfx90a_chip(0x740C, None) == "AMD Instinct MI250X"
    assert sku_name_for_gfx90a_chip(None, 104) is None


def test_combined_mi250_name_uses_chip_id():
    combined = "AMD Instinct MI250X / MI250"
    no_space = "AMD Instinct MI250X/MI250"
    assert _sku_for_torch_name(combined, 0x740C, 104) == "AMD Instinct MI250"
    assert _sku_for_torch_name(combined, 0x740C, 208) == "AMD Instinct MI250"
    assert _sku_for_torch_name(combined, 0x740C, 110) == "AMD Instinct MI250X"
    assert _sku_for_torch_name(no_space, 0x740C, 104) == "AMD Instinct MI250"
    assert _sku_for_torch_name("AMD Instinct MI250X", 0x740C, 104) == (
        "AMD Instinct MI250X"
    )
    # No chip id and no CU count: the string-only path keeps the longer token.
    assert _normalize_amd_gpu_name(combined) == "AMD Instinct MI250X"
    assert _normalize_amd_gpu_name(no_space) == "AMD Instinct MI250X"


def test_amd_smi_asic_follows_hip_id():
    listed = "GPU: 0\n    HIP_ID: 2\nGPU: 1\n    HIP_ID: 0\n"
    asic = (
        "GPU: 0\n    DEVICE_ID: 0x7408\n    NUM_COMPUTE_UNITS: 110\n"
        "GPU: 1\n    DEVICE_ID: 0x740c\n    NUM_COMPUTE_UNITS: 104\n"
    )
    assert _amd_smi_gpu_record(listed, asic, 0) == (0x740C, 104)
    assert _amd_smi_gpu_record(listed, asic, 2) == (0x7408, 110)


def test_gfx942_chip_id_separates_mi300a_and_mi300x():
    assert sku_name_for_gfx942_chip(0x74A0) == "AMD Instinct MI300A"
    assert sku_name_for_gfx942_chip(0x74A1) == "AMD Instinct MI300X"
    assert sku_name_for_gfx942_chip(0x74A9) == "AMD Instinct MI300X"
    assert sku_name_for_gfx942_chip(0x74A2) is None
    assert sku_name_for_gfx942_chip(None) is None
    assert _sku_for_torch_name("gfx942", 0x74A0, 228) == "AMD Instinct MI300A"
    assert _sku_for_torch_name("gfx942", 0x74A9, 304) == "AMD Instinct MI300X"
    assert _sku_for_torch_name("gfx942", None, None) == "gfx942"
    assert _sku_for_torch_name("AMD Instinct MI300A", 0x74A1, 304) == (
        "AMD Instinct MI300A"
    )


def test_prompt_uses_cu_labels():
    mi300a = get_gpu_specs("AMD Instinct MI300A")
    assert mi300a is not None
    text = _format_gpu_specs(mi300a)
    assert "CU Count" in text
    assert "SM Count" not in text
    assert "HBM3 unified" in text

    mi350 = get_gpu_specs("AMD Instinct MI350X")
    assert mi350 is not None
    assert mi350["cu_count"] == 256
    assert "Specs confidence" not in _format_gpu_specs(mi350)


def _render_optimization_prompt(gpu_specs: dict) -> str:
    manager = PromptManager(target_platform=get_platform("rocm"))
    return manager.render_kernel_optimization_prompt(
        problem_description="problem",
        kernel_code="kernel",
        gpu_specs=gpu_specs,
        roofline={},
        category="memory",
        summary="summary",
        reasoning="reasoning",
        root_cause={"cause": "cause"},
        recommended_fix={"fix": "fix"},
        pytorch_baseline_ms=1.0,
        current_best_ms=1.0,
    )


def test_rocm_kernel_guidance_skips_sku_numbers():
    text = get_platform("rocm").kernel_guidance
    assert "MI300X" not in text
    assert "MI350X" not in text
    assert "wavefront size 64" in text
    assert "warp_size=32" in text
    assert "Do not use tensor descriptors or TMA" in text
    assert "Do not pass warp_specialize" in text
    assert (
        "Where this section conflicts with the Triton programming guidelines, "
        "follow this section."
    ) in text
    assert "fp8 is available on gfx942" in text
    assert "BF16 matrix units are available on CDNA" in text


def test_optimization_prompt_uses_cu_labels_for_cdna():
    mi300 = get_gpu_specs("AMD Instinct MI300X")
    assert mi300 is not None
    text = _render_optimization_prompt(mi300)
    assert "CU Count: 304" in text
    assert "Max Threads per CU" in text
    assert "L1 Cache per CU" in text
    assert "SM Count" not in text
    assert "Do not use tensor descriptors or TMA" in text
    assert "Do not pass warp_specialize" in text

    cuda = PromptManager(target_platform=get_platform("cuda"))
    cuda_text = cuda.render_kernel_optimization_prompt(
        problem_description="problem",
        kernel_code="kernel",
        gpu_specs=mi300,
        roofline={},
        category="memory",
        summary="summary",
        reasoning="reasoning",
        root_cause={"cause": "cause"},
        recommended_fix={"fix": "fix"},
        pytorch_baseline_ms=1.0,
        current_best_ms=1.0,
    )
    assert "tensor descriptors" not in cuda_text
    assert "warp_specialize" not in cuda_text

    h100 = get_gpu_specs("NVIDIA H100 SXM5 80GB")
    assert h100 is not None
    nvidia = _render_optimization_prompt(h100)
    assert "SM Count:" in nvidia
    assert "CU Count" not in nvidia


def test_refinement_prompt_includes_rocm_kernel_guidance():
    rocm = PromptManager(target_platform=get_platform("rocm"))
    text = rocm.render_kernel_refinement_prompt(
        problem_description="problem",
        test_code="test",
        kernel_code="kernel",
        error_info={"stderr": "", "stdout": ""},
    )
    assert "AMD ROCm-Specific Optimizations" in text
    assert "Do not pass warp_specialize" in text

    cuda = PromptManager(target_platform=get_platform("cuda"))
    cuda_text = cuda.render_kernel_refinement_prompt(
        problem_description="problem",
        test_code="test",
        kernel_code="kernel",
        error_info={"stderr": "", "stdout": ""},
    )
    assert "AMD ROCm-Specific Optimizations" not in cuda_text


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


def test_missing_rocprof_csv_is_not_a_profile(tmp_path):
    stats = tmp_path / "rocprof_round_1_stats.csv"
    pmc = tmp_path / "rocprof_round_1_pmc.csv"
    with pytest.raises(RuntimeError, match="without a usable CSV"):
        _merge_rocprof_outputs(stats, pmc)

    stats.write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "kernel,1,2000000,2000000,100\n"
    )
    with pytest.raises(RuntimeError, match="without a usable CSV"):
        _merge_rocprof_outputs(stats, pmc)


def test_collected_rocprof_csvs_keep_measured_counters(tmp_path):
    stats = tmp_path / "stats.csv"
    pmc = tmp_path / "pmc.csv"
    stats.write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "kernel,1,2000000,2000000,100\n"
    )
    Path(str(stats) + "_kernel_trace.csv").write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "KERNEL_DISPATCH,kernel,0,2000000\n"
    )
    pmc.write_text(
        "KernelName,SQ_INSTS_VALU,SQ_INSTS_SALU,TCC_HIT_sum,TCC_MISS_sum\n"
        "kernel,900,100,80,20\n"
    )
    metrics = _merge_rocprof_outputs(stats, pmc)
    assert metrics["duration_ms"] == 2.0
    assert metrics["sq_insts_valu"] == 900.0
    assert metrics["valu_utilization_pct"] == 90.0
    assert "compute_sol_pct" not in metrics
    assert "memory_sol_pct" not in metrics
    assert metrics["tcc_hit"] == 80.0
    assert metrics["tcc_miss"] == 20.0
    assert "sq_waves" not in metrics
    assert "tcc_ea_rdreq" not in metrics
    assert "fetch_size_kb" not in metrics


def test_duration_is_the_last_triton_dispatch(tmp_path):
    stats = tmp_path / "stats.csv"
    pmc = tmp_path / "pmc.csv"
    stats.write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "at::native::vectorized_elementwise,2,3000,1500,10\n"
        "triton_add,4,204960,51240,90\n"
        "_reduce_kernel,4,2000,500,1\n"
    )
    Path(str(stats) + "_kernel_trace.csv").write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "KERNEL_DISPATCH,at::native::fill,0,1000\n"
        "KERNEL_DISPATCH,triton_add,1000,35000\n"
        "KERNEL_DISPATCH,triton_add,35000,55000\n"
        "KERNEL_DISPATCH,triton_add,55000,80000\n"
        "KERNEL_DISPATCH,triton_add,80000,100000\n"
        "KERNEL_DISPATCH,_reduce_kernel,100000,100500\n"
        "KERNEL_DISPATCH,at::native::fill,100500,103000\n"
    )
    pmc.write_text("Counter_Name,Counter_Value\nSQ_INSTS_VALU,100\n")
    metrics = _merge_rocprof_outputs(stats, pmc)
    # Last triton_add is 20000 ns. The later _reduce_kernel is not the duration.
    assert metrics["duration_ms"] == pytest.approx(0.02)


def test_pmc_keeps_the_last_dispatch_of_the_profiled_kernel(tmp_path):
    stats = tmp_path / "stats.csv"
    pmc = tmp_path / "pmc.csv"
    stats.write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "at::native::vectorized_elementwise,1,3000,3000,1\n"
        "triton_kernel,2,200000,100000,99\n"
    )
    Path(str(stats) + "_kernel_trace.csv").write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "KERNEL_DISPATCH,triton_kernel,0,100000\n"
        "KERNEL_DISPATCH,triton_kernel,100000,200000\n"
    )
    pmc.write_text(
        "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value\n"
        "1,triton_kernel,SQ_INSTS_VALU,100\n"
        "2,triton_kernel,SQ_INSTS_VALU,250\n"
        "3,at::native::vectorized_elementwise,SQ_INSTS_VALU,900\n"
    )
    metrics = _merge_rocprof_outputs(stats, pmc)
    assert metrics["sq_insts_valu"] == 250.0


def test_stats_csv_without_kernel_trace_is_not_a_profile(tmp_path):
    stats = tmp_path / "stats.csv"
    pmc = tmp_path / "pmc.csv"
    stats.write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "triton_add,4,204960,51240,100\n"
    )
    pmc.write_text("Counter_Name,Counter_Value\nSQ_INSTS_VALU,100\n")
    with pytest.raises(RuntimeError, match="kernel trace has no dispatch"):
        _merge_rocprof_outputs(stats, pmc)


def test_stats_csv_without_averagens_is_not_a_profile(tmp_path):
    stats = tmp_path / "stats.csv"
    pmc = tmp_path / "pmc.csv"
    stats.write_text("Name,Calls,Duration(ns)\nkernel,1,2000000\n")
    pmc.write_text("Counter_Name,Counter_Value\nSQ_INSTS_VALU,900\nSQ_INSTS_SALU,100\n")
    with pytest.raises(RuntimeError, match="no AverageNs"):
        _merge_rocprof_outputs(stats, pmc)


def _fake_rocprof(path, source: str) -> str:
    path.write_text("#!/usr/bin/env python3\n" + source)
    path.chmod(0o755)
    return str(path)


def test_rocprofv3_pass_without_csv_is_not_collected(tmp_path):
    bin_path = _fake_rocprof(
        tmp_path / "rocprofv3",
        "import sys\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "if out.endswith('_pass0_0'):\n"
        "    open(out + '_counter_collection.csv', 'w').write(\n"
        "        'Counter_Name,Counter_Value\\nSQ_INSTS_VALU,900\\n'\n"
        "    )\n",
    )
    out_csv = tmp_path / "pmc.csv"
    with pytest.raises(RuntimeError, match="wrote no"):
        _run_rocprofv3_pmc_passes(
            rocprof_bin=bin_path,
            benchmark_script=tmp_path / "bench.py",
            workdir=tmp_path,
            out_csv=out_csv,
            python_executable=sys.executable,
            env={},
            timeout=30,
        )
    assert not Path(str(out_csv) + "_counter_collection.csv").exists()


def test_rocprof_exit_0_does_not_keep_the_previous_kernel(tmp_path):
    bin_path = _fake_rocprof(tmp_path / "rocprofv3", "import sys\nraise SystemExit(0)\n")
    stats = tmp_path / "rocprof_round_1_stats.csv"
    pmc = tmp_path / "rocprof_round_1_pmc.csv"
    stats.with_name(stats.name + "_kernel_stats.csv").write_text(
        "Name,Calls,TotalDurationNs,AverageNs,Percentage\n"
        "old_kernel,1,5000000,5000000,100\n"
    )
    Path(str(stats) + "_kernel_trace.csv").write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "KERNEL_DISPATCH,old_kernel,0,5000000\n"
    )
    # Pass 2 choice 0 is TCC_EA0_RDREQ. Pass 4 is FETCH_SIZE.
    Path(str(pmc) + "_pass2_0_counter_collection.csv").write_text(
        "Counter_Name,Counter_Value\nTCC_EA0_RDREQ_sum,99\n"
    )
    Path(str(pmc) + "_pass4_0_counter_collection.csv").write_text(
        "Counter_Name,Counter_Value\nFETCH_SIZE,4194304\n"
    )
    shared = {
        "rocprof_bin": bin_path,
        "benchmark_script": tmp_path / "bench.py",
        "workdir": tmp_path,
        "python_executable": sys.executable,
        "env": {},
        "timeout": 30,
    }
    with pytest.raises(RuntimeError, match="did not write"):
        _run_rocprof_stats(out_csv=stats, **shared)
    with pytest.raises(RuntimeError, match="wrote no"):
        _run_rocprofv3_pmc_passes(out_csv=pmc, **shared)
    with pytest.raises(RuntimeError, match="without a usable CSV"):
        _merge_rocprof_outputs(stats, pmc)


def test_rocprof_stale_ea_file_does_not_skip_the_backup_name(tmp_path):
    bin_path = _fake_rocprof(
        tmp_path / "rocprofv3",
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "out = Path(args[args.index('-o') + 1])\n"
        "text = Path(args[args.index('-i') + 1]).read_text()\n"
        "if 'TCC_EA0_' in text:\n"
        "    raise SystemExit(0)\n"
        "name, value = 'SQ_WAVES', '1'\n"
        "if 'TCC_EA_RDREQ_sum' in text:\n"
        "    name, value = 'TCC_EA_RDREQ_sum', '11'\n"
        "elif 'TCC_EA_WRREQ_sum' in text:\n"
        "    name, value = 'TCC_EA_WRREQ_sum', '7'\n"
        "elif 'FETCH_SIZE' in text:\n"
        "    name, value = 'FETCH_SIZE', '8'\n"
        "out.with_name(out.name + '_counter_collection.csv').write_text(\n"
        "    f'Counter_Name,Counter_Value\\n{name},{value}\\n'\n"
        ")\n",
    )
    pmc = tmp_path / "pmc.csv"
    Path(str(pmc) + "_pass2_0_counter_collection.csv").write_text(
        "Counter_Name,Counter_Value\nTCC_EA0_RDREQ_sum,99\n"
    )
    _run_rocprofv3_pmc_passes(
        rocprof_bin=bin_path,
        benchmark_script=tmp_path / "bench.py",
        workdir=tmp_path,
        out_csv=pmc,
        python_executable=sys.executable,
        env={},
        timeout=30,
    )
    text = Path(str(pmc) + "_counter_collection.csv").read_text()
    assert "TCC_EA_RDREQ_sum,11" in text
    assert "TCC_EA0_RDREQ_sum,99" not in text
    assert "FETCH_SIZE,8" in text


def test_rocprofv3_uses_long_output_format(tmp_path):
    bin_path = _fake_rocprof(
        tmp_path / "rocprofv3",
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if '-f' in args or '--output-format' not in args:\n"
        "    raise SystemExit(2)\n"
        "out = Path(args[args.index('-o') + 1])\n"
        "if '--stats' in args:\n"
        "    out.with_name(out.name + '_kernel_stats.csv').write_text(\n"
        "        'Name,Calls,TotalDurationNs,AverageNs,Percentage\\n'\n"
        "        'gemm,1,100000,100000,100\\n'\n"
        "    )\n"
        "    out.with_name(out.name + '_kernel_trace.csv').write_text(\n"
        "        'Kind,Kernel_Name,Start_Timestamp,End_Timestamp\\n'\n"
        "        'KERNEL_DISPATCH,gemm,0,100000\\n'\n"
        "    )\n"
        "elif '-i' in args:\n"
        "    text = Path(args[args.index('-i') + 1]).read_text()\n"
        "    if 'TCC_EA0_' in text:\n"
        "        raise SystemExit(0)\n"
        "    name, value = 'SQ_WAVES', '4'\n"
        "    if 'TCC_EA_RDREQ_sum' in text:\n"
        "        name, value = 'TCC_EA_RDREQ_sum', '11'\n"
        "    elif 'TCC_EA_WRREQ_sum' in text:\n"
        "        name, value = 'TCC_EA_WRREQ_sum', '7'\n"
        "    out.with_name(out.name + '_counter_collection.csv').write_text(\n"
        "        f'Counter_Name,Counter_Value\\n{name},{value}\\n'\n"
        "    )\n"
        "else:\n"
        "    raise SystemExit(2)\n",
    )
    stats = tmp_path / "rocprof_v3_stats.csv"
    pmc = tmp_path / "rocprof_v3_pmc.csv"
    shared = {
        "rocprof_bin": bin_path,
        "benchmark_script": tmp_path / "bench.py",
        "workdir": tmp_path,
        "python_executable": sys.executable,
        "env": {},
        "timeout": 30,
    }
    _run_rocprof_stats(out_csv=stats, **shared)
    _run_rocprof_pmc(out_csv=pmc, **shared)
    metrics = _merge_rocprof_outputs(stats, pmc)
    assert metrics["duration_ms"] == pytest.approx(0.1)
    # Four passes report SQ_WAVES. The gfx942 EA names write nothing, so the
    # gfx90a names are collected instead.
    assert metrics["sq_waves"] == 16.0
    assert metrics["tcc_ea_rdreq"] == 11.0
    assert metrics["tcc_ea_wrreq"] == 7.0


def test_rocprofv3_retries_ea_counter_after_nonzero_exit(tmp_path):
    bin_path = _fake_rocprof(
        tmp_path / "rocprofv3",
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "out = Path(args[args.index('-o') + 1])\n"
        "text = Path(args[args.index('-i') + 1]).read_text()\n"
        "if 'TCC_EA0_' in text:\n"
        "    raise SystemExit(1)\n"
        "name, value = 'SQ_WAVES', '1'\n"
        "if 'TCC_EA_RDREQ_sum' in text:\n"
        "    name, value = 'TCC_EA_RDREQ_sum', '11'\n"
        "elif 'TCC_EA_WRREQ_sum' in text:\n"
        "    name, value = 'TCC_EA_WRREQ_sum', '7'\n"
        "out.with_name(out.name + '_counter_collection.csv').write_text(\n"
        "    f'Counter_Name,Counter_Value\\n{name},{value}\\n'\n"
        ")\n",
    )
    pmc = tmp_path / "pmc.csv"
    _run_rocprofv3_pmc_passes(
        rocprof_bin=bin_path,
        benchmark_script=tmp_path / "bench.py",
        workdir=tmp_path,
        out_csv=pmc,
        python_executable=sys.executable,
        env={},
        timeout=30,
    )
    assert (tmp_path / "rocprof_input_pass2_1.txt").read_text() == (
        "pmc: TCC_EA_RDREQ_sum\n"
    )
    assert (tmp_path / "rocprof_input_pass3_1.txt").read_text() == (
        "pmc: TCC_EA_WRREQ_sum\n"
    )


def test_roofline_wrappers_expose_early_stop():
    wrapper = ROCmRooflineAnalyzerWrapper()
    result = wrapper.analyze(ncu_metrics=_FLAT_ROCM_METRICS)
    assert wrapper.config.early_stop is False
    stop = False
    for _ in range(wrapper.config.convergence_rounds):
        stop, _reason = wrapper.should_stop(result)
    assert stop is True
    # The orchestrator breaks only when both are true. ROCm leaves the flag
    # off, so a converged multi-round run continues.
    assert not (stop and wrapper.config.early_stop)

    assert NvidiaRooflineAnalyzer().config.early_stop is True


def _hide_rocm_masks(monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)


def test_find_amd_smi_beside_python_before_resolved_path(monkeypatch, tmp_path):
    from triton_kernel_agent.platform.rocm import _find_amd_smi

    link_dir = tmp_path / "venv" / "bin"
    real_dir = tmp_path / "real"
    link_dir.mkdir(parents=True)
    real_dir.mkdir()
    real_python = real_dir / "python3"
    real_python.write_text("")
    python = link_dir / "python"
    python.symlink_to(real_python)
    sibling = link_dir / "amd-smi"
    sibling.write_text("")
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.shutil.which", lambda _name: None
    )
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.sys.executable", str(python)
    )
    assert _find_amd_smi() == str(sibling)


def test_find_amd_smi_uses_opt_rocm_bin(monkeypatch, tmp_path):
    from triton_kernel_agent.platform.rocm import _find_amd_smi

    python = tmp_path / "python"
    python.write_text("")
    bundled = Path("/opt/rocm/bin/amd-smi")
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.shutil.which", lambda _name: None
    )
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.sys.executable", str(python)
    )
    real_is_file = Path.is_file

    def is_file(self):
        if self == bundled:
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    assert _find_amd_smi() == str(bundled)


def test_mixed_node_uses_nvidia_unless_hip_visible_devices(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    launched: list[str] = []

    def query(argv):
        launched.append(argv[0])
        if argv[0] == "nvidia-smi":
            return "0\n1\n"
        if argv == ["amd-smi", "list", "-e"]:
            return (
                "GPU: 0\n    HIP_ID: 1\n"
                "GPU: 1\n    HIP_ID: 2\n"
                "GPU: 2\n    HIP_ID: 3\n"
                "GPU: 3\n    HIP_ID: 4\n"
            )
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: "amd-smi"
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)

    def use(cuda=None, hip=None, rocr=None):
        for name, value in (
            ("CUDA_VISIBLE_DEVICES", cuda),
            ("HIP_VISIBLE_DEVICES", hip),
            ("ROCR_VISIBLE_DEVICES", rocr),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        launched.clear()

    # No visibility variables: NVIDIA is the default.
    use()
    assert _detect_gpus() == [0, 1]
    assert launched == ["nvidia-smi"]

    # Only CUDA_VISIBLE_DEVICES is set, so its indexes are used.
    use(cuda="3")
    assert _detect_gpus() == [3]

    # HIP id 4 is listed, so AMD is selected.
    use(hip="4")
    assert _detect_gpus() == [4]
    assert "amd-smi" in launched

    # 9 is not a listed HIP id. amd-smi must run to see that.
    use(hip="9")
    assert _detect_gpus() == [0, 1]
    assert launched == ["nvidia-smi", "amd-smi"]

    # An empty HIP variable disables AMD.
    use(hip="")
    assert _detect_gpus() == [0, 1]
    assert launched == ["nvidia-smi"]

    # Empty HIP disables AMD. The CUDA list supplies the NVIDIA id.
    use(cuda="0", hip="")
    assert _detect_gpus() == [0]

    # Empty CUDA disables NVIDIA, so a listed HIP id selects AMD.
    use(cuda="", hip="4")
    assert _detect_gpus() == [4]

    # Empty CUDA disables NVIDIA. HIP index 2 is the third ROCR device.
    use(cuda="", rocr="1,2,3", hip="2")
    assert _detect_gpus() == [3]

    # A CUDA list and a HIP list both name devices.
    use(cuda="0", hip="4")
    with pytest.raises(RuntimeError, match="conflicts"):
        _detect_gpus()

def test_path_binaries_use_nvidia_unless_hip_visible_devices(monkeypatch, tmp_path):
    """Run fake ``nvidia-smi`` and ``amd-smi`` from ``PATH``.

    The NVIDIA listing is the live output of
    ``nvidia-smi --query-gpu=index --format=csv,noheader`` on
    rocm-framework-h100-pcie.amd.com (two NVIDIA H100 PCIe GPUs). That host
    has no ``amd-smi``, so the HIP listing is synthetic and distinct.
    """
    from triton_kernel_agent.opt_manager import _detect_gpus

    bindir = tmp_path / "bin"
    bindir.mkdir()
    # Shebang is this test's interpreter. PATH below contains only bindir, so
    # ``#!/usr/bin/env python3`` cannot find Python. A bash wrapper is also
    # unsafe: the NVIDIA container sets BASH_ENV to a file that runs
    # ``nvidia-smi -q -d COMPUTE`` for every bash script, and a script of
    # that name then calls itself.
    shebang = f"#!{sys.executable}\n"
    nvidia = bindir / "nvidia-smi"
    nvidia.write_text(
        shebang
        + "import sys\n"
        + "if sys.argv[1:] == ['--query-gpu=index', '--format=csv,noheader']:\n"
        + "    sys.stdout.write('0\\n1\\n')\n"
        + "    raise SystemExit(0)\n"
        + "raise SystemExit(2)\n"
    )
    amd = bindir / "amd-smi"
    amd.write_text(
        shebang
        + "import sys\n"
        + "if sys.argv[1:] == ['list', '-e']:\n"
        + "    sys.stdout.write("
        + "'GPU: 0\\n    HIP_ID: 0\\n"
        + "GPU: 1\\n    HIP_ID: 1\\n"
        + "GPU: 2\\n    HIP_ID: 2\\n')\n"
        + "    raise SystemExit(0)\n"
        + "raise SystemExit(2)\n"
    )
    nvidia.chmod(0o755)
    amd.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _hide_rocm_masks(monkeypatch)

    assert _detect_gpus() == [0, 1]
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    assert _detect_gpus() == [2]


def test_no_listed_gpu_stops(monkeypatch, tmp_path):
    from triton_kernel_agent.opt_manager import OptimizationManager

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _hide_rocm_masks(monkeypatch)
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: None
    )
    monkeypatch.setattr(
        "triton_kernel_agent.opt_manager._query_smi", lambda argv: None
    )
    with pytest.raises(RuntimeError, match="no GPU visible"):
        OptimizationManager(
            strategy="greedy",
            num_workers=1,
            log_dir=tmp_path / "no-gpu",
        )


def _amd_enum(*hip_ids: int) -> str:
    lines = []
    for index, hip_id in enumerate(hip_ids):
        lines.append(f"GPU: {index}\n    HIP_ID: {hip_id}\n")
    return "".join(lines)


def test_amd_smi_print_order_is_kept_when_nvidia_is_absent(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _hide_rocm_masks(monkeypatch)
    binary = "amd-smi"
    enumerated = _amd_enum(2, 0, 1)

    def query(argv):
        if argv == [binary, "list", "-e"]:
            return enumerated
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: binary
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)
    assert _detect_gpus() == [2, 0, 1]


def test_unmatched_hip_visible_devices_disables_amd(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    binary = "amd-smi"

    def amd_only(argv):
        if argv == [binary, "list", "-e"]:
            return _amd_enum(0)
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: binary
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", amd_only)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    assert _detect_gpus() == []


def test_rocm_worker_process_pins_and_forces_target_platform(monkeypatch, tmp_path):
    from triton_kernel_agent.platform.worker_runner import rocm_worker_process

    constructed: dict = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def optimize_kernel(self, **kwargs):
            return True, "kernel", {"best_time_ms": 1.0}

    monkeypatch.setattr(
        "triton_kernel_agent.opt_worker.OptimizationWorker", FakeWorker
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,0")
    shared = {"target_platform": "cuda", "gpu_name": "AMD Instinct MI300X"}
    problem = tmp_path / "problem.py"
    problem.write_text("def ref():\n    return 0\n")
    workdir = tmp_path / "work"
    queued: list = []

    rocm_worker_process(
        0,
        "def kernel():\n    return 0\n",
        1.0,
        "parent",
        problem,
        [],
        workdir,
        workdir / "logs",
        SimpleNamespace(put=queued.append),
        None,
        None,
        1.0,
        0,
        "model",
        False,
        None,
        shared,
        [],
        [],
        None,
        3,
    )

    assert shared["target_platform"] == "cuda"
    assert constructed["target_platform"] == "rocm"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert os.environ["HIP_VISIBLE_DEVICES"] == "3"
    assert queued and queued[0]["success"] is True


def test_baseline_cache_normalizes_rocm_gpu_name(tmp_path, monkeypatch):
    import logging

    from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import ROCmRooflineAnalyzer
    from triton_kernel_agent.opt_manager import (
        OptimizationManager,
        _specs_for_baseline,
    )

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm.detect_amd_gpu_name",
        lambda: "gfx950",
    )
    detected = _specs_for_baseline(None, rocm=True)
    assert detected is not None
    assert detected["name"] == "AMD Instinct MI350X"
    assert detected["peak_memory_bw_gbps"] == 8000

    class Profiler:
        def profile_kernel(self, kernel_file, problem_file, round_num):
            return SimpleNamespace(
                metrics={
                    "fetch_size_kb": 4194304,
                    "duration_ms": 2.4,
                    "valu_utilization_pct": 20.0,
                    "memory_bound_pct": 80.0,
                }
            )

    manager = SimpleNamespace(
        worker_kwargs={"gpu_name": "gfx942", "target_platform": "rocm"},
        _hide_caller_cuda_mask=True,
        log_dir=tmp_path,
        _baseline_profile_cache={},
        logger=logging.getLogger("baseline-cache-test"),
        _mgr_profiler=Profiler(),
        _mgr_roofline=ROCmRooflineAnalyzer(),
    )
    parent = SimpleNamespace(program_id="parent", kernel_code="def kernel():\n    pass\n")
    OptimizationManager._populate_baseline_cache_uncached(
        manager,
        [{"parent": parent}],
        tmp_path / "problem.py",
        1,
    )
    cached = manager._baseline_profile_cache["parent"]
    assert cached["memory_sol_pct"] == pytest.approx(33.77, abs=0.01)
    assert cached["compute_sol_measured"] is False
    assert cached["memory_sol_measured"] is True

    class MixProfiler:
        def profile_kernel(self, kernel_file, problem_file, round_num):
            return SimpleNamespace(
                metrics={
                    "valu_utilization_pct": 20.0,
                    "memory_bound_pct": 80.0,
                }
            )

    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Capture()
    manager.logger.addHandler(handler)
    manager.logger.setLevel(logging.INFO)
    manager._mgr_profiler = MixProfiler()
    try:
        OptimizationManager._populate_baseline_cache_uncached(
            manager,
            [
                {
                    "parent": SimpleNamespace(
                        program_id="unmeasured",
                        kernel_code="def kernel():\n    pass\n",
                    )
                }
            ],
            tmp_path / "problem.py",
            1,
        )
    finally:
        manager.logger.removeHandler(handler)
    assert manager._baseline_profile_cache["unmeasured"]["bottleneck"] == "unknown"
    assert any("SOL not measured" in message for message in messages)
    assert not any("0.0% SOL" in message for message in messages)


def test_pin_rocm_worker_unsets_cuda_visible_devices(monkeypatch):
    from triton_kernel_agent.platform.rocm import _pin_rocm_visible_devices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    _pin_rocm_visible_devices(2)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"


def test_visible_hip_id_uses_the_detected_list(monkeypatch):
    from triton_kernel_agent.platform.rocm import _visible_hip_listing

    def visible_hip_id() -> int:
        return _visible_hip_listing()[1]

    listing = "GPU: 0\n    HIP_ID: 2\nGPU: 1\n    HIP_ID: 0\nGPU: 2\n    HIP_ID: 1\n"

    def run(args):
        if args == ["list", "-e"]:
            return listing
        return None

    monkeypatch.setattr("triton_kernel_agent.platform.rocm._run_amd_smi", run)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert visible_hip_id() == 2

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    with pytest.raises(RuntimeError, match="no GPU visible"):
        visible_hip_id()

    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._run_amd_smi", lambda _args: None
    )
    with pytest.raises(RuntimeError, match="no GPU visible"):
        visible_hip_id()


def test_chip_lookup_uses_the_listing_that_produced_the_hip_id(monkeypatch):
    from triton_kernel_agent.platform.rocm import _amd_smi_chip_id_and_cu_count

    listed = "GPU: 0\n    HIP_ID: 2\nGPU: 1\n    HIP_ID: 0\n"
    asic = (
        "GPU: 0\n    DEVICE_ID: 0x7408\n    NUM_COMPUTE_UNITS: 110\n"
        "GPU: 1\n    DEVICE_ID: 0x740c\n    NUM_COMPUTE_UNITS: 104\n"
    )

    def run(args):
        if args == ["list", "-e"]:
            return None
        if args == ["list"]:
            return listed
        if args == ["static", "--asic"]:
            return asic
        return None

    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("triton_kernel_agent.platform.rocm._run_amd_smi", run)
    # HIP id 2 is GPU 0. Using 2 as an ASIC index would miss both records.
    assert _amd_smi_chip_id_and_cu_count() == (0x7408, 110)


def test_manager_runs_the_vendor_the_visibility_rules_choose(monkeypatch, tmp_path):
    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.platform.rocm import ROCmWorkerRunner

    def query(argv):
        if argv[0] == "nvidia-smi":
            return "0\n1\n"
        if argv == ["amd-smi", "list", "-e"]:
            return "GPU: 0\n    HIP_ID: 4\n"
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: "amd-smi"
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4")

    # NVIDIA was requested and HIP id 4 matches. Do not run ROCm.
    with pytest.raises(
        RuntimeError,
        match="NVIDIA platform was requested but ROCm devices were selected",
    ):
        OptimizationManager(
            strategy="greedy",
            num_workers=1,
            platform="nvidia",
            log_dir=tmp_path / "nvidia-requested",
        )

    # No platform was requested. HIP id 4 is listed, so the run is ROCm.
    rocm_manager = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        log_dir=tmp_path / "rocm",
    )
    assert isinstance(rocm_manager.worker_runner, ROCmWorkerRunner)
    assert rocm_manager.gpu_ids == [4]
    assert rocm_manager.worker_kwargs["target_platform"] == "rocm"

    # No ROCm mask. A requested ROCm platform uses the amd-smi HIP ids.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    requested_open = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform="rocm",
        target_platform="rocm",
        log_dir=tmp_path / "rocm-open",
    )
    assert isinstance(requested_open.worker_runner, ROCmWorkerRunner)
    assert requested_open.gpu_ids == [4]

    # The caller already chose the worker target. Detection must not replace it.
    kept_target = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform="rocm",
        target_platform="cuda",
        log_dir=tmp_path / "rocm-keep-target",
    )
    assert isinstance(kept_target.worker_runner, ROCmWorkerRunner)
    assert kept_target.gpu_ids == [4]
    assert kept_target.worker_kwargs["target_platform"] == "cuda"

    # 9 matches nothing. The request stays on ROCm and does not use NVIDIA.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    with pytest.raises(RuntimeError, match="no devices are visible"):
        OptimizationManager(
            strategy="greedy",
            num_workers=1,
            platform="rocm",
            log_dir=tmp_path / "rocm-miss",
        )


def test_cuda_visible_devices_without_nvidia_smi_keeps_amd(monkeypatch, tmp_path):
    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.platform.rocm import ROCmWorkerRunner

    enumerated = _amd_enum(1, 3)

    def query(argv):
        if argv == ["amd-smi", "list", "-e"]:
            return enumerated
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: "amd-smi"
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _hide_rocm_masks(monkeypatch)

    manager = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform="rocm",
        log_dir=tmp_path / "amd-only",
    )
    assert isinstance(manager.worker_runner, ROCmWorkerRunner)
    assert manager.gpu_ids == [1, 3]
    assert manager.worker_kwargs["target_platform"] == "rocm"
    # Construction leaves the caller's mask alone.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    with manager._without_caller_cuda_mask():
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_explicit_nvidia_without_listed_gpus_does_not_invent_one(
    monkeypatch, tmp_path
):
    from triton_kernel_agent.opt_manager import OptimizationManager

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _hide_rocm_masks(monkeypatch)
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: None
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", lambda argv: None)
    with pytest.raises(
        RuntimeError, match="NVIDIA platform was requested but no devices are visible"
    ):
        OptimizationManager(
            strategy="greedy",
            num_workers=1,
            platform="nvidia",
            log_dir=tmp_path / "no-gpu",
        )


def test_noop_platform_does_not_need_a_gpu(monkeypatch, tmp_path):
    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.platform.noop import (
        NoOpProfiler,
        NoOpVerifier,
        NoOpWorkerRunner,
    )
    from triton_kernel_agent.platform.nvidia import NvidiaVerifier

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _hide_rocm_masks(monkeypatch)
    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: None
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", lambda argv: None)

    dry = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform="noop",
        log_dir=tmp_path / "noop",
    )
    assert isinstance(dry.verifier, NoOpVerifier)
    assert isinstance(dry.worker_runner, NoOpWorkerRunner)
    assert isinstance(dry._mgr_profiler, NoOpProfiler)
    assert dry.gpu_ids == [0]

    def query(argv):
        if argv[0] == "nvidia-smi":
            return "0\n1\n"
        return None

    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)
    listed = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform="noop",
        log_dir=tmp_path / "noop-listed",
    )
    assert isinstance(listed.verifier, NoOpVerifier)
    assert isinstance(listed._mgr_profiler, NoOpProfiler)
    assert listed.gpu_ids == [0]

    mixed = {
        "verifier": "nvidia",
        "benchmarker": "nvidia",
        "worker_runner": "nvidia",
        "specs_provider": "nvidia",
        "profiler": "noop",
        "roofline_analyzer": "noop",
        "bottleneck_analyzer": "noop",
        "rag_prescriber": "noop",
    }
    kept = OptimizationManager(
        strategy="greedy",
        num_workers=1,
        platform=mixed,
        log_dir=tmp_path / "mixed",
    )
    assert isinstance(kept.verifier, NvidiaVerifier)
    assert isinstance(kept._mgr_profiler, NoOpProfiler)
    assert kept.worker_kwargs["platform_config"]["profiler"] == "noop"
    assert kept.worker_kwargs["target_platform"] == "cuda"
    assert kept.gpu_ids == [0, 1]


def test_memory_sol_is_traffic_divided_by_peak():
    analyzer = ROCmRooflineAnalyzer()
    slow = analyzer.analyze(
        ncu_metrics={
            "fetch_size_kb": 4194304,
            "duration_ms": 2.4,
            "peak_memory_bw_gbps": 5300,
            "valu_utilization_pct": 20.0,
            "memory_bound_pct": 80.0,
            "tcc_cache_hit_rate_pct": 10.0,
        }
    )
    assert slow.memory_sol_pct == pytest.approx(33.77, abs=0.01)
    assert slow.efficiency_pct == slow.memory_sol_pct
    assert slow.compute_sol_pct == 0.0
    assert slow.compute_sol_measured is False
    assert slow.at_roofline is False
    assert slow.bottleneck == "memory"
    from kernel_perf_agent.kernel_opt.roofline.rocm_roofline import (
        format_rocm_roofline_summary,
    )

    summary = format_rocm_roofline_summary(slow)
    assert slow.memory_sol_measured is True
    assert slow.memory_sol_from_traffic is True
    assert "Compute SOL:" not in summary
    assert "Memory SOL:" in summary
    assert "traffic / peak bandwidth" in summary
    assert "underutilized" not in summary

    both_low = analyzer.analyze(
        ncu_metrics={"compute_sol_pct": 10.0, "memory_sol_pct": 20.0}
    )
    assert both_low.compute_sol_measured is True
    assert both_low.memory_sol_measured is True
    assert both_low.bottleneck == "underutilized"

    saturated = analyzer.analyze(
        ncu_metrics={
            "fetch_size_kb": 4194304,
            "duration_ms": 0.4,
            "peak_memory_bw_gbps": 5300,
        }
    )
    assert saturated.at_roofline is True
    saturated_summary = format_rocm_roofline_summary(saturated)
    assert "Status: AT ROOFLINE" in saturated_summary
    assert "heuristic" not in saturated_summary

    mix_only = analyzer.analyze(
        ncu_metrics={
            "valu_utilization_pct": 20.0,
            "memory_bound_pct": 80.0,
        }
    )
    assert mix_only.efficiency_pct == 0.0
    assert mix_only.memory_sol_measured is False
    assert mix_only.compute_sol_measured is False
    assert mix_only.bottleneck == "unknown"
    assert mix_only.at_roofline is False
    assert mix_only.valu_utilization_pct == 20.0
    assert mix_only.memory_bound_pct == 80.0
    mix_summary = format_rocm_roofline_summary(mix_only)
    assert "Memory SOL:" not in mix_summary
    assert "Compute SOL:" not in mix_summary
    assert "SOL Efficiency:" not in mix_summary
    assert "underutilized" not in mix_summary

    from types import SimpleNamespace

    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader("triton_kernel_agent/templates"),
    )
    bottleneck = SimpleNamespace(
        category="memory",
        summary="s",
        reasoning="r",
        root_cause=SimpleNamespace(cause="c", evidence=[]),
        recommended_fix=SimpleNamespace(fix="f", rationale=""),
    )
    prompt = env.get_template("kernel_optimization.j2").render(
        pytorch_baseline_ms=1.0,
        roofline=mix_only.to_dict(),
        kernel_code="pass",
        bottleneck=bottleneck,
        problem_description="p",
    )
    roofline_section = prompt.split("## ROOFLINE ANALYSIS")[1].split("## BOTTLENECK")[0]
    assert "Primary Bottleneck: UNKNOWN" in roofline_section
    assert "Efficiency:" not in roofline_section
    assert "Memory SOL:" not in roofline_section
    assert "Compute SOL:" not in roofline_section

    attempt = SimpleNamespace(
        round_num=1,
        bottleneck_category="underutilized",
        root_cause="c",
        recommended_fix="f",
        config_changes={},
        time_before_ms=1.0,
        time_after_ms=1.0,
        improvement_pct=0.0,
        compute_sol_pct=0.0,
        compute_sol_measured=False,
        memory_sol_pct=0.0,
        memory_sol_measured=False,
        passed_verification=True,
        error_message=None,
    )
    reflexion = env.get_template("reflexion_prompt.j2").render(attempt=attempt)
    assert "NCU SOL: not measured" in reflexion

    nvidia = {
        "bottleneck": "compute",
        "compute_sol_pct": 70.0,
        "memory_sol_pct": 20.0,
        "efficiency_pct": 70.0,
        "headroom_pct": 30.0,
        "at_roofline": False,
        "uses_tensor_cores": True,
        "warnings": [],
    }
    nvidia_prompt = env.get_template("kernel_optimization.j2").render(
        pytorch_baseline_ms=1.0,
        roofline=nvidia,
        kernel_code="pass",
        bottleneck=bottleneck,
        problem_description="p",
    )
    nvidia_section = nvidia_prompt.split("## ROOFLINE ANALYSIS")[1].split(
        "## BOTTLENECK"
    )[0]
    assert "Compute SOL: 70.0%" in nvidia_section
    assert "Memory SOL: 20.0%" in nvidia_section
    assert "Efficiency: 70.0%" in nvidia_section

    supplied = analyzer.analyze(
        ncu_metrics={"compute_sol_pct": 70.0, "memory_sol_pct": 20.0}
    )
    assert supplied.memory_sol_measured is True
    assert supplied.memory_sol_from_traffic is False
    supplied_summary = format_rocm_roofline_summary(supplied)
    assert "Memory SOL:" in supplied_summary
    assert "traffic / peak bandwidth" not in supplied_summary


def test_vendor_runners_share_the_process_pool(tmp_path):
    import logging

    from triton_kernel_agent.platform.nvidia import NvidiaWorkerRunner
    from triton_kernel_agent.platform.rocm import ROCmWorkerRunner
    from triton_kernel_agent.platform.worker_runner import (
        nvidia_worker_process,
        rocm_worker_process,
    )

    def runner(cls):
        return cls(
            log_dir=tmp_path,
            logger=logging.getLogger("worker-target"),
            benchmark_lock=None,
            profiling_semaphore=None,
            openai_model="model",
            high_reasoning_effort=False,
            bottleneck_override=None,
            worker_kwargs={},
        )

    assert runner(ROCmWorkerRunner).worker_target is rocm_worker_process
    assert runner(NvidiaWorkerRunner).worker_target is nvidia_worker_process


def test_final_profile_runs_on_nvidia_when_sol_profile_failed():
    nvidia = SimpleNamespace(platform="nvidia")
    rocm = SimpleNamespace(platform="rocm")
    assert should_profile_final_kernel(nvidia, 1, True)
    assert should_profile_final_kernel(rocm, 1, True)
    assert should_profile_final_kernel(rocm, 1, False) is False
    assert should_profile_final_kernel(nvidia, 0, True) is False


def test_create_kwargs_keep_temperature_in_extra_body():
    from utils.providers.anthropic_provider import AnthropicProvider

    params = AnthropicProvider._create_kwargs(
        SimpleNamespace(get_max_tokens_limit=lambda _name: 8192),
        "claude-opus-4-6",
        [{"role": "user", "content": "hi"}],
        0.0,
        high_reasoning_effort=True,
    )
    assert params["extra_body"] == {"temperature": 0.0}
    assert "output_config" not in params


def test_get_response_forwards_temperature_once():
    from utils.providers.anthropic_provider import AnthropicProvider

    captured = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.client = SimpleNamespace(messages=_Messages())
    provider.is_available = lambda: True
    provider.get_max_tokens_limit = lambda _name: 8192
    response = provider.get_response(
        "claude-opus-4-6",
        [{"role": "user", "content": "hi"}],
        temperature=0.8,
        high_reasoning_effort=True,
    )
    assert response.content == "ok"
    assert captured["extra_body"] == {"temperature": 0.8}
    assert "temperature" not in captured
    assert "output_config" not in captured
