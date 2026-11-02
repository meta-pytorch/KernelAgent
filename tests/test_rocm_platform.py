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
    pmc.write_text(
        "KernelName,SQ_INSTS_VALU,SQ_INSTS_SALU,TCC_HIT_sum,TCC_MISS_sum\n"
        "kernel,900,100,80,20\n"
    )
    metrics = _merge_rocprof_outputs(stats, pmc)
    assert metrics["duration_ms"] == 2.0
    assert metrics["sq_insts_valu"] == 900.0
    assert metrics["tcc_hit"] == 80.0
    assert metrics["tcc_miss"] == 20.0


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


def test_detect_gpus_uses_amd_smi_hip_ids_when_nvidia_smi_is_absent(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    enumerated = "GPU: 0\n    HIP_ID: 1\nGPU: 1\n    HIP_ID: 3\n"
    binary = "/opt/python-3.11-venv/bin/amd-smi"

    def query(argv):
        if argv == [binary, "list", "-e"]:
            return enumerated
        return None

    monkeypatch.setattr(
        "triton_kernel_agent.platform.rocm._find_amd_smi", lambda: binary
    )
    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", query)
    assert _detect_gpus() == [1, 3]


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


def test_detect_gpus_prefers_cuda_visible_devices_and_nvidia_smi(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    def fail_query(argv):
        raise AssertionError(argv)

    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", fail_query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert _detect_gpus() == [4, 5]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")

    def nvidia_query(argv):
        if argv[0] == "nvidia-smi":
            return "0\n1\n"
        raise AssertionError(argv)

    monkeypatch.setattr("triton_kernel_agent.opt_manager._query_smi", nvidia_query)
    assert _detect_gpus() == [0, 1]


def test_detect_gpus_falls_back_to_one_gpu(monkeypatch):
    from triton_kernel_agent.opt_manager import _detect_gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "triton_kernel_agent.opt_manager._query_smi", lambda argv: None
    )
    assert _detect_gpus() == [0]
