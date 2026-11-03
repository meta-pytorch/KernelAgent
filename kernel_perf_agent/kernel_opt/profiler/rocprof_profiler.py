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

"""
rocprof Profiling Module for Triton Kernels on AMD GPUs.

This module wraps ``rocprofv3`` to collect hardware performance counters for
Triton kernels on AMD GPUs (ROCm 10.1 or later). Profiling uses
``--kernel-trace --stats --output-format csv`` and ``--`` before the
application.

Counter Categories
------------------
- **SQ counters**: Shader processor activity (SQ_WAVES, SQ_INSTS_VALU, etc.)
- **TCC counters**: L2 cache / memory controller (TCC_HIT, TCC_MISS, etc.)
- **EA / GRBM counters**: Memory bandwidth and bus activity

The collected counters are mapped to a metric dictionary compatible with the
rest of the KernelAgent optimization pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hardware counter sets collected via rocprof -i <input_file>
# ---------------------------------------------------------------------------

# PMC counters requested from the hardware.  Each line in the input file
# specifies one pass of the hardware; rocprof runs the kernel once per pass.
# Keep pass count low to avoid excessive re-runs.

# Pass 1: Wavefront / shader utilisation
_PMC_PASS1 = [
    "SQ_WAVES",  # total wavefronts launched
    "SQ_INSTS_VALU",  # VALU (vector ALU) instructions executed
    "SQ_INSTS_SALU",  # SALU (scalar ALU) instructions executed
    "SQ_INSTS_VMEM_RD",  # vector memory read instructions
    "SQ_INSTS_VMEM_WR",  # vector memory write instructions
    "SQ_INSTS_LDS",  # LDS instructions
    "SQ_WAIT_INST_ANY",  # stall cycles waiting on any instruction
]

# Pass 3: Memory bandwidth (derived counters)
# FETCH_SIZE and WRITE_SIZE use the same internal hardware PMC block and
# cannot be collected together in a single pass on rocprofv3 / gfx942.
# Collecting them on separate lines avoids error 38 ("Request exceeds the
# capabilities of the hardware to collect") and the associated hang.
_PMC_PASS3 = [
    "FETCH_SIZE",  # bytes fetched from memory (in KB, vendor metric)
]

_PMC_PASS4 = [
    "WRITE_SIZE",  # bytes written to memory (in KB, vendor metric)
]

# rocprofv3 reruns the app once per pmc group and overwrites the CSV, so
# each group is collected on its own. A tuple is a list of alternative
# names: gfx942 exposes TCC_EA0_*, gfx90a exposes TCC_EA_RDREQ_sum /
# TCC_EA_WRREQ_sum. A non-zero exit or an empty CSV tries the next name.
_PMC_V3_PASSES: list[list[str] | tuple[list[str], ...]] = [
    _PMC_PASS1,
    ["TCC_HIT_sum", "TCC_MISS_sum"],
    (["TCC_EA0_RDREQ_sum"], ["TCC_EA_RDREQ_sum"]),
    (["TCC_EA0_WRREQ_sum"], ["TCC_EA_WRREQ_sum"]),
    _PMC_PASS3,
    _PMC_PASS4,
]

# Derived metric keys returned in the normalized metrics dict
ROCM_METRIC_KEYS = [
    "sq_waves",
    "sq_insts_valu",
    "sq_insts_salu",
    "sq_insts_vmem_rd",
    "sq_insts_vmem_wr",
    "sq_insts_lds",
    "sq_wait_inst_any",
    "tcc_hit",
    "tcc_miss",
    "tcc_cache_hit_rate_pct",
    "tcc_ea_rdreq",
    "tcc_ea_wrreq",
    "fetch_size_kb",
    "write_size_kb",
    # Derived
    "valu_utilization_pct",  # SQ_INSTS_VALU / (SQ_WAVES * max_valu_per_wave)
    "memory_bound_pct",  # heuristic: vmem instructions / total instructions
    "compute_sol_pct",  # heuristic compute SOL for roofline compat
    "memory_sol_pct",  # heuristic memory SOL for roofline compat
]


# ---------------------------------------------------------------------------
# rocprof input file generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# rocprof invocation
# ---------------------------------------------------------------------------


def _find_rocprof() -> str:
    """Find ``rocprofv3`` on PATH, beside Python, or under ``/opt/rocm/bin``."""
    found = shutil.which("rocprofv3")
    if found:
        return found
    # A venv python is often a symlink. Look beside the symlink first.
    for executable in (Path(sys.executable), Path(sys.executable).resolve()):
        sibling = executable.with_name("rocprofv3")
        if sibling.is_file():
            return str(sibling)
    bundled = Path("/opt/rocm/bin/rocprofv3")
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError(
        "rocprofv3 not found. Install ROCm 10.1 or later."
    )


def profile_triton_kernel_rocm(
    benchmark_script: Path,
    workdir: Path,
    out_prefix: str = "rocprof_output",
    python_executable: str | None = None,
    rocprof_bin: str | None = None,
    timeout: int = 360,
) -> Path:
    """Profile a Triton kernel with rocprofv3.

    Runs the benchmark script twice:
    1. ``rocprofv3 --kernel-trace --stats`` — kernel timing.
    2. One ``rocprofv3 -i`` pass per counter group.

    The two CSV outputs are merged into a single JSON metrics file.

    Args:
        benchmark_script: Path to the Python script that calls the kernel.
        workdir: Working directory for execution and output files.
        out_prefix: Prefix for output filenames.
        python_executable: Python executable (default: sys.executable).
        rocprof_bin: Path to rocprofv3 (auto-detect if None).
        timeout: Timeout in seconds per rocprof invocation.

    Returns:
        Path to the merged JSON metrics file.

    Raises:
        FileNotFoundError: If rocprof binary or benchmark script not found.
        RuntimeError: If rocprof profiling fails.
    """
    if python_executable is None:
        python_executable = sys.executable

    if rocprof_bin is None:
        rocprof_bin = _find_rocprof()

    benchmark_script = benchmark_script.resolve()
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(workdir / ".triton_cache")

    stats_csv = workdir / f"{out_prefix}_stats.csv"
    pmc_csv = workdir / f"{out_prefix}_pmc.csv"
    metrics_json = workdir / f"{out_prefix}_metrics.json"

    # ------------------------------------------------------------------
    # Pass 1: timing stats (--stats)
    # ------------------------------------------------------------------
    _run_rocprof_stats(
        rocprof_bin=rocprof_bin,
        benchmark_script=benchmark_script,
        workdir=workdir,
        out_csv=stats_csv,
        python_executable=python_executable,
        env=env,
        timeout=timeout,
    )

    # ------------------------------------------------------------------
    # Pass 2: hardware PMC counters, one rocprofv3 run per group
    # ------------------------------------------------------------------
    _run_rocprof_pmc(
        rocprof_bin=rocprof_bin,
        benchmark_script=benchmark_script,
        workdir=workdir,
        out_csv=pmc_csv,
        python_executable=python_executable,
        env=env,
        timeout=timeout,
    )

    # ------------------------------------------------------------------
    # Merge outputs into metrics JSON
    # ------------------------------------------------------------------
    metrics = _merge_rocprof_outputs(stats_csv, pmc_csv)
    metrics_json.write_text(json.dumps(metrics, indent=2))

    print(f"[rocprof] Metrics written: {metrics_json}")
    return metrics_json


def _app_command(python_executable: str, benchmark_script: Path) -> list[str]:
    return [python_executable, str(benchmark_script)]


def _run_rocprof_stats(
    rocprof_bin: str,
    benchmark_script: Path,
    workdir: Path,
    out_csv: Path,
    python_executable: str,
    env: dict,
    timeout: int,
) -> None:
    """Collect kernel timing with rocprofv3.

    ``--output-format csv`` writes ``{prefix}_kernel_stats.csv``.
    """
    cmd = [
        rocprof_bin,
        "--kernel-trace",
        "--stats",
        "--output-format",
        "csv",
        "-o",
        str(out_csv),
        "--",
        *_app_command(python_executable, benchmark_script),
    ]
    print(f"[rocprof] Running stats: {' '.join(cmd[:6])}...")
    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rocprof --stats failed (rc={result.returncode}):\n"
            f"{(result.stderr or result.stdout)[:500]}"
        )
    print("[rocprof] Stats pass completed")


def _run_rocprof_pmc(
    rocprof_bin: str,
    benchmark_script: Path,
    workdir: Path,
    out_csv: Path,
    python_executable: str,
    env: dict,
    timeout: int,
) -> None:
    """Collect PMC counters with one rocprofv3 run per counter group."""
    _run_rocprofv3_pmc_passes(
        rocprof_bin=rocprof_bin,
        benchmark_script=benchmark_script,
        workdir=workdir,
        out_csv=out_csv,
        python_executable=python_executable,
        env=env,
        timeout=timeout,
    )


def _run_rocprofv3_pmc_passes(
    rocprof_bin: str,
    benchmark_script: Path,
    workdir: Path,
    out_csv: Path,
    python_executable: str,
    env: dict,
    timeout: int,
) -> None:
    """Collect each rocprofv3 counter group into its own CSV, then merge.

    A multi-job input rewrites ``{prefix}_counter_collection.csv`` on every
    pass, so only the last group would survive.
    """
    import csv

    combined_header: list[str] | None = None
    combined_rows: list[list[str]] = []
    for index, entry in enumerate(_PMC_V3_PASSES):
        choices = entry if isinstance(entry, tuple) else (entry,)
        collected: list[list[str]] | None = None
        for choice_index, counters in enumerate(choices):
            pass_input = workdir / f"rocprof_input_pass{index}_{choice_index}.txt"
            pass_input.write_text("pmc: " + " ".join(counters) + "\n")
            pass_out = workdir / f"{out_csv.name}_pass{index}_{choice_index}"
            cmd = [
                rocprof_bin,
                "-i",
                str(pass_input),
                "--output-format",
                "csv",
                "-o",
                str(pass_out),
                "--",
                python_executable,
                str(benchmark_script),
            ]
            print(f"[rocprof] Running PMC pass {index + 1}: {' '.join(counters)}")
            result = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            another_choice = choice_index + 1 < len(choices)
            if result.returncode != 0:
                if another_choice:
                    continue
                raise RuntimeError(
                    f"rocprof PMC pass {index + 1} failed (rc={result.returncode}):\n"
                    f"{(result.stderr or result.stdout)[:500]}"
                )
            csv_path = Path(str(pass_out) + "_counter_collection.csv")
            if not csv_path.exists():
                if another_choice:
                    continue
                raise RuntimeError(
                    f"rocprof PMC pass {index + 1} exited 0 but wrote no {csv_path.name}"
                )
            with csv_path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            if len(rows) < 2:
                if another_choice:
                    continue
                raise RuntimeError(
                    f"rocprof PMC pass {index + 1} wrote no counter rows in {csv_path.name}"
                )
            collected = rows
            break
        if collected is None:
            raise RuntimeError(f"rocprof PMC pass {index + 1} produced no counter rows")
        if combined_header is None:
            combined_header = collected[0]
        combined_rows.extend(collected[1:])

    if combined_header is None:
        raise RuntimeError("rocprofv3 PMC produced no counter rows")

    dest = Path(str(out_csv) + "_counter_collection.csv")
    with dest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(combined_header)
        writer.writerows(combined_rows)
    print("[rocprof] PMC pass completed")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _resolve_collected_csv(csv_path: Path, suffix: str) -> Path:
    """Return the CSV rocprof actually wrote.

    rocprofv3 ``--output-format csv`` appends a suffix to the ``-o`` path, for
    example ``stats.csv_kernel_stats.csv``.
    """
    if csv_path.exists():
        return csv_path
    sibling = csv_path.with_name(csv_path.name + suffix)
    if sibling.exists():
        return sibling
    return csv_path


def _kernel_trace_csv(stats_out: Path) -> Path:
    """Kernel trace rocprofv3 writes for a ``--stats`` ``-o`` path."""
    return Path(str(stats_out) + "_kernel_trace.csv")


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    import csv

    if not csv_path.exists():
        return []
    try:
        with csv_path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _busiest_kernel_name(rows: list[dict[str, str]]) -> str | None:
    """Non-PyTorch stats row with the largest ``TotalDurationNs``."""
    best_name: str | None = None
    best_total = -1.0
    for row in rows:
        name = (row.get("Name") or "").strip()
        if not name or "at::" in name:
            continue
        try:
            total = float(row.get("TotalDurationNs") or "")
        except (TypeError, ValueError):
            continue
        if total >= best_total:
            best_total = total
            best_name = name
    return best_name


def _last_profiled_dispatch_ns(trace_path: Path, kernel_name: str) -> float | None:
    """Duration of the last trace dispatch of ``kernel_name``.

    The profiled process runs the kernel three times and then once more, and
    ``rocprofv3`` records every dispatch. ``AverageNs`` is the mean of those
    launches. A later helper kernel is not this duration.
    """
    chosen_end: float | None = None
    chosen_duration: float | None = None
    for row in _read_csv_rows(trace_path):
        name = (row.get("Kernel_Name") or row.get("Name") or "").strip()
        if name != kernel_name:
            continue
        kind = (row.get("Kind") or "KERNEL_DISPATCH").strip()
        if kind != "KERNEL_DISPATCH":
            continue
        try:
            start = float(row["Start_Timestamp"])
            end = float(row["End_Timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            continue
        if chosen_end is None or end >= chosen_end:
            chosen_end = end
            chosen_duration = end - start
    return chosen_duration


def _parse_stats_csv(csv_path: Path) -> dict[str, Any]:
    """Parse rocprofv3 stats and the kernel trace written beside them.

    ``duration_ms`` is the last dispatch of the busiest non-PyTorch kernel.
    A stats file with no ``AverageNs`` is not a completed profile.
    """
    resolved = _resolve_collected_csv(csv_path, "_kernel_stats.csv")
    rows = _read_csv_rows(resolved)
    has_averagens = False
    for row in rows:
        raw = row.get("AverageNs")
        if raw is None or raw == "":
            continue
        try:
            float(raw)
        except (ValueError, TypeError):
            continue
        has_averagens = True
        break
    if not has_averagens:
        return {}

    kernel_name = _busiest_kernel_name(rows)
    if kernel_name is None:
        return {"has_averagens": True}
    duration_ns = _last_profiled_dispatch_ns(_kernel_trace_csv(csv_path), kernel_name)
    if duration_ns is None:
        return {"has_averagens": True}
    return {
        "has_averagens": True,
        "kernel_name": kernel_name,
        "duration_ns": duration_ns,
        "duration_ms": duration_ns / 1e6,
    }


def _pmc_row_kernel_name(row: dict[str, str]) -> str:
    return (row.get("Kernel_Name") or row.get("KernelName") or "").strip()


def _pmc_dispatch_rank(row: dict[str, str], index: int) -> float:
    """Dispatch order. A missing id keeps file order, and a later row wins.

    rocprofv3 writes ``Dispatch_Id``. ``Dispatch_ID`` is accepted too.
    """
    raw = (row.get("Dispatch_Id") or row.get("Dispatch_ID") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(index)


def _pmc_rows_for_kernel(
    rows: list[dict[str, str]], kernel_name: str
) -> list[dict[str, str]] | None:
    """Rows for ``kernel_name``, or None when the CSV names no kernel.

    A counter file that only has counter columns cannot tell dispatches
    apart. A file that names kernels keeps one kernel, so ``at::`` rows
    and other launches drop out.
    """
    if not any(_pmc_row_kernel_name(row) for row in rows):
        return None
    return [row for row in rows if _pmc_row_kernel_name(row) == kernel_name]


def _parse_pmc_csv(csv_path: Path, kernel_name: str) -> dict[str, Any]:
    """Parse rocprof PMC CSV to extract hardware counter values.

    When the CSV names kernels, each counter keeps the value from the last
    dispatch of ``kernel_name``. Passes appended into one file number
    ``Dispatch_ID`` separately, so the last dispatch is per counter.
    """
    csv_path = _resolve_collected_csv(csv_path, "_counter_collection.csv")
    if not csv_path.exists():
        return {}

    import csv

    rows: list[dict[str, str]] = []
    try:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {}

    if not rows:
        return {}

    chosen = _pmc_rows_for_kernel(rows, kernel_name)
    if chosen is not None:
        rows = chosen
    if not rows:
        return {}

    # rocprofv3 writes one row per counter: Counter_Name, Counter_Value.
    if "Counter_Name" in rows[0] and "Counter_Value" in rows[0]:
        if chosen is None:
            aggregated: dict[str, float] = {}
            for row in rows:
                name = (row.get("Counter_Name") or "").strip()
                if not name:
                    continue
                try:
                    aggregated[name] = aggregated.get(name, 0.0) + float(
                        row["Counter_Value"]
                    )
                except (ValueError, TypeError, KeyError):
                    pass
            return aggregated
        last: dict[str, tuple[float, float]] = {}
        for index, row in enumerate(rows):
            name = (row.get("Counter_Name") or "").strip()
            if not name:
                continue
            try:
                value = float(row["Counter_Value"])
            except (ValueError, TypeError, KeyError):
                continue
            rank = _pmc_dispatch_rank(row, index)
            previous = last.get(name)
            if previous is None or rank >= previous[0]:
                last[name] = (rank, value)
        return {name: value for name, (_rank, value) in last.items()}

    # rocprof v2 PMC CSV columns: Index, KernelName, gpu-id, queue-id,
    # queue-index, pid, tid, grd, wgr, lds, scr, arch_vgpr, accum_vgpr,
    # sgpr, wave_size, sig, obj, <counter1>, <counter2>, ...
    # Aggregate all rows by summing numeric counter columns
    if chosen is not None:
        best_rank = -1.0
        best_index = -1
        for index, row in enumerate(rows):
            rank = _pmc_dispatch_rank(row, index)
            if rank >= best_rank:
                best_rank = rank
                best_index = index
        rows = [rows[best_index]] if best_index >= 0 else []

    aggregated: dict[str, float] = {}
    counter_cols = [
        k
        for k in rows[0].keys()
        if k
        not in (
            "Index",
            "KernelName",
            "Kernel_Name",
            "gpu-id",
            "GPU_ID",
            "queue-id",
            "Queue_ID",
            "queue-index",
            "Queue_Index",
            "pid",
            "PID",
            "tid",
            "TID",
            "Dispatch_ID",
            "Dispatch_Id",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Scratch_Per_Workitem",
            "Arch_VGPR",
            "Accum_VGPR",
            "SGPR",
            "Wave_Size",
            "Start_Timestamp",
            "End_Timestamp",
            "Correlation_ID",
            "sig",
            "obj",
        )
    ]

    for row in rows:
        for col in counter_cols:
            try:
                aggregated[col] = aggregated.get(col, 0.0) + float(row[col])
            except (ValueError, TypeError, KeyError):
                pass

    return aggregated


def _merge_rocprof_outputs(
    stats_csv: Path,
    pmc_csv: Path,
) -> dict[str, Any]:
    """Merge stats and PMC outputs into a normalized metrics dictionary.

    The returned dict is compatible with the rest of the KernelAgent pipeline
    (same structure expected by RooflineAnalyzer and BottleneckAnalyzer).
    """
    stats_path = _resolve_collected_csv(stats_csv, "_kernel_stats.csv")
    pmc_path = _resolve_collected_csv(pmc_csv, "_counter_collection.csv")
    missing = [path for path in (stats_path, pmc_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            "rocprof completed without a usable CSV: "
            + ", ".join(str(path) for path in missing)
        )

    stats = _parse_stats_csv(stats_csv)
    if not stats.get("has_averagens"):
        raise RuntimeError(f"rocprof stats CSV has no AverageNs: {stats_path}")
    if "duration_ms" not in stats:
        trace_path = _kernel_trace_csv(stats_csv)
        raise RuntimeError(f"rocprof kernel trace has no dispatch: {trace_path}")
    pmc = _parse_pmc_csv(pmc_csv, stats["kernel_name"])
    if not pmc:
        raise RuntimeError(f"rocprof PMC CSV has no counter rows: {pmc_path}")

    # Raw counters (lower-cased for consistency)
    raw: dict[str, Any] = {}
    for k, v in pmc.items():
        raw[k.lower()] = v

    # Copy only counters the CSV actually contained. A missing column is an
    # uncollected counter, not a measured zero.
    fields = (
        ("sq_waves", ("sq_waves",)),
        ("sq_insts_valu", ("sq_insts_valu",)),
        ("sq_insts_salu", ("sq_insts_salu",)),
        ("sq_insts_vmem_rd", ("sq_insts_vmem_rd",)),
        ("sq_insts_vmem_wr", ("sq_insts_vmem_wr",)),
        ("sq_insts_lds", ("sq_insts_lds",)),
        ("sq_wait_inst_any", ("sq_wait_inst_any",)),
        ("tcc_hit", ("tcc_hit_sum",)),
        ("tcc_miss", ("tcc_miss_sum",)),
        ("tcc_ea_rdreq", ("tcc_ea_rdreq_sum", "tcc_ea0_rdreq_sum")),
        ("tcc_ea_wrreq", ("tcc_ea_wrreq_sum", "tcc_ea0_wrreq_sum")),
        ("fetch_size_kb", ("fetch_size",)),
        ("write_size_kb", ("write_size",)),
    )
    metrics: dict[str, Any] = {"duration_ms": stats["duration_ms"]}
    for key, names in fields:
        value = next((raw[name] for name in names if name in raw), None)
        if value is not None:
            metrics[key] = value

    # Derived: L2 cache hit rate
    if "tcc_hit" in metrics and "tcc_miss" in metrics:
        tcc_total = metrics["tcc_hit"] + metrics["tcc_miss"]
        metrics["tcc_cache_hit_rate_pct"] = (
            100.0 * metrics["tcc_hit"] / tcc_total if tcc_total > 0 else 0.0
        )

    # Derived: VALU share of counted shader instructions, not cycles and
    # not percent of peak FLOPs. Counters the CSV omitted stay out of the sum.
    inst_keys = (
        "sq_insts_valu",
        "sq_insts_salu",
        "sq_insts_vmem_rd",
        "sq_insts_vmem_wr",
        "sq_insts_lds",
    )
    present_insts = [metrics[key] for key in inst_keys if key in metrics]
    total_insts = sum(present_insts)
    if total_insts > 0 and "sq_insts_valu" in metrics:
        metrics["valu_utilization_pct"] = 100.0 * metrics["sq_insts_valu"] / total_insts
        metrics["compute_sol_pct"] = min(metrics["valu_utilization_pct"], 100.0)
    if total_insts > 0 and (
        "sq_insts_vmem_rd" in metrics or "sq_insts_vmem_wr" in metrics
    ):
        vmem_insts = metrics.get("sq_insts_vmem_rd", 0.0) + metrics.get(
            "sq_insts_vmem_wr", 0.0
        )
        metrics["memory_bound_pct"] = 100.0 * vmem_insts / total_insts
        metrics["memory_sol_pct"] = min(metrics["memory_bound_pct"], 100.0)

    # Include raw counter dump for debugging / LLM consumption
    metrics["_raw_counters"] = {k: v for k, v in raw.items()}

    return metrics


# ---------------------------------------------------------------------------
# Public helper: load metrics from a previously written JSON file
# ---------------------------------------------------------------------------


def load_rocm_metrics(json_path: Path) -> dict[str, Any]:
    """Load rocprof metrics from a JSON file produced by :func:`profile_triton_kernel_rocm`.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Metrics dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"ROCm metrics JSON not found: {json_path}")
    return json.loads(json_path.read_text())


def metrics_to_prompt_rocm(metrics: dict[str, Any]) -> str:
    """Convert ROCm metrics dict to a JSON string for LLM prompts.

    Args:
        metrics: Metrics dictionary from :func:`load_rocm_metrics`.

    Returns:
        JSON string (drops internal ``_raw_counters`` key for brevity).
    """
    # Exclude raw counter dump from LLM context (too verbose)
    filtered = {k: v for k, v in metrics.items() if k != "_raw_counters"}
    return json.dumps(filtered, indent=2)
