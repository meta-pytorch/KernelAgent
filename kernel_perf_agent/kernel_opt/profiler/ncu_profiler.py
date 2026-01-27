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
NCU Profiling Module for Triton Kernels

This module wraps three tasks:
1) Collect core metrics for Triton kernels with Nsight Compute into CSV (`profile_triton_kernel`).
2) Extract and clean those metrics into a DataFrame from the CSV (`load_ncu_metrics`).
3) Convert the metrics table into a string suitable for inclusion in an LLM prompt (`metrics_to_prompt`).


"""

import json
import math
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# This selection of metrics is inspried by the CudaForge team (https://github.com/OptimAI-Lab/CudaForge/tree/main)

METRICS = ",".join(
    [
        "sm__cycles_active.avg",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__registers_per_thread",
        "sm__inst_executed.sum",
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum.per_second",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "launch__shared_mem_per_block_allocated",
        "l1tex__t_sector_hit_rate.pct",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "lts__t_sector_hit_rate.pct",
        "lts__throughput.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    ]
)

# List version for convenient header selection
METRIC_COLUMNS: List[str] = [s.strip() for s in METRICS.split(",")]


class MetricSelectionPolicy(Enum):
    """Policy for selecting rows when multiple rows exist for a kernel."""

    FIRST = "first"
    LAST = "last"
    MAX_CYCLES = "max_cycles"


def profile_triton_kernel(
    benchmark_script: Path,
    workdir: Path,
    out_csv: str = "ncu_output.csv",
    python_executable: Optional[str] = None,
    ncu_bin: Optional[str] = None,
    launch_count: int = 20,
    timeout: int = 120,
    use_sudo: bool = False,
) -> Path:
    """
    Profile a Triton kernel using NCU.

    Args:
        benchmark_script: Path to benchmark script that calls the kernel
        workdir: Working directory for execution
        out_csv: Output CSV filename
        python_executable: Python executable to use (default: sys.executable)
        ncu_bin: Path to NCU binary (default: auto-detect)
        launch_count: Number of kernel launches to profile
        timeout: Timeout in seconds for NCU execution
        use_sudo: Whether to run NCU with sudo. Can also be enabled via
            KERNELAGENT_NCU_USE_SUDO=1 environment variable. Default: False

    Returns:
        Path to output CSV file

    Raises:
        RuntimeError: If NCU profiling fails
        FileNotFoundError: If NCU binary or output CSV not found
    """
    # Check for environment variable override
    use_sudo = use_sudo or os.environ.get("KERNELAGENT_NCU_USE_SUDO", "0") == "1"

    # Resolve paths
    if python_executable is None:
        python_executable = sys.executable

    if ncu_bin is None:
        ncu_bin = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"

    if not Path(ncu_bin).exists():
        raise FileNotFoundError(f"NCU binary not found: {ncu_bin}")

    csv_path = (workdir / out_csv).resolve()
    benchmark_script = benchmark_script.resolve()

    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    # Preserve important environment variables
    env = os.environ.copy()

    # Add Triton-specific environment variables
    env["TRITON_CACHE_DIR"] = str(workdir / ".triton_cache")

    preserve = ",".join(
        [
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        ]
    )

    # Build NCU command
    cmd = []
    if use_sudo:
        cmd.extend(["sudo", "-E", f"--preserve-env={preserve}"])

    cmd.extend(
        [
            ncu_bin,
            "--csv",
            "--page=raw",
            "--kernel-name-base=demangled",
            "--target-processes=all",
            "--replay-mode=kernel",
            "--profile-from-start=on",
            f"--log-file={str(csv_path)}",
            f"--metrics={METRICS}",
            "--launch-skip=0",
            f"--launch-count={launch_count}",
            python_executable,
            str(benchmark_script),
        ]
    )

    print("[NCU] Running profiling...")
    print(f"[NCU] Benchmark: {benchmark_script.name}")
    print(f"[NCU] Output: {csv_path}")
    print(f"[NCU] Using sudo: {use_sudo}")
    print(f"[NCU] Command: {' '.join(cmd[:10])}... (truncated)")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout

            # Check for common permission errors
            permission_errors = [
                "permission",
                "ERR_NVGPUCTRPERM",
                "profiling permissions",
                "requires root",
                "access denied",
            ]
            is_permission_error = any(
                err.lower() in error_msg.lower() for err in permission_errors
            )

            if is_permission_error and not use_sudo:
                raise RuntimeError(
                    f"NCU requires elevated permissions on this system.\n"
                    f"Options to resolve:\n"
                    f"  1. Rerun with use_sudo=True\n"
                    f"  2. Set environment variable: KERNELAGENT_NCU_USE_SUDO=1\n"
                    f"  3. Configure driver permissions: "
                    f"Original error:\n{error_msg[:500]}"
                )

            raise RuntimeError(
                f"NCU profiling failed with return code {result.returncode}:\n{error_msg[:500]}"
            )

        if not csv_path.exists():
            raise FileNotFoundError(f"NCU did not create output CSV: {csv_path}")

        # Check if CSV has content
        csv_size = csv_path.stat().st_size
        if csv_size < 100:
            raise RuntimeError(
                f"NCU CSV file is too small ({csv_size} bytes), likely empty"
            )

        print("[NCU] ✓ Profiling completed successfully")
        print(f"[NCU] ✓ CSV written: {csv_path} ({csv_size} bytes)")
        return csv_path

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"NCU profiling timed out after {timeout} seconds")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"NCU profiling failed: {e}")


def _apply_selection_policy(
    df: pd.DataFrame,
    policy: MetricSelectionPolicy,
) -> pd.DataFrame:
    """
    Apply selection policy to choose a single row from a DataFrame.

    Args:
        df: DataFrame with one or more rows
        policy: Selection policy to apply

    Returns:
        DataFrame with a single row based on the policy
    """
    if df.empty:
        return df

    if len(df) == 1:
        return df

    if policy == MetricSelectionPolicy.FIRST:
        return df.iloc[[0]]
    elif policy == MetricSelectionPolicy.LAST:
        return df.iloc[[-1]]
    elif policy == MetricSelectionPolicy.MAX_CYCLES:
        if "sm__cycles_active.avg" in df.columns:
            return df.sort_values("sm__cycles_active.avg", ascending=False).head(1)
        # Fallback to last if cycles column not available
        return df.iloc[[-1]]
    else:
        # Fallback to last for unknown policies
        return df.iloc[[-1]]


def _filter_by_kernel_names(
    df: pd.DataFrame,
    name_list: Sequence[str],
    policy: MetricSelectionPolicy,
    keep_cols: List[str],
) -> pd.DataFrame:
    """
    Filter DataFrame by kernel names with substring matching.

    Args:
        df: DataFrame with NCU metrics
        name_list: List of kernel name substrings to match
        policy: Selection policy when multiple rows match
        keep_cols: Columns to preserve in empty result

    Returns:
        Filtered DataFrame with one row per matched kernel name
    """
    results = []
    for name in name_list:
        # Use contains match instead of exact equality (for Triton's long kernel names)
        matched = df[
            df["Kernel Name"].astype(str).str.contains(name, regex=False, na=False)
        ]
        if matched.empty:
            continue

        row = _apply_selection_policy(matched, policy)
        results.append(row)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=keep_cols)


def load_ncu_metrics(
    csv_path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = ("Kernel Name",),
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,
    select: Union[str, MetricSelectionPolicy] = MetricSelectionPolicy.LAST,
) -> pd.DataFrame:
    """
    Load and parse NCU metrics from CSV file.

    Args:
        csv_path: Path to NCU CSV output
        columns: Specific metric columns to load (default: all METRIC_COLUMNS)
        extra_keep: Additional columns to keep (e.g., "Kernel Name")
        coerce_numeric: Convert metric values to numeric
        name_list: Filter by kernel names (substring match)
        select: Selection policy when multiple rows per name.
                Can be MetricSelectionPolicy enum or string ("first", "last", "max_cycles")

    Returns:
        DataFrame with parsed metrics

    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If no requested columns found in CSV or invalid select value
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Convert string to enum if needed
    if isinstance(select, str):
        try:
            policy = MetricSelectionPolicy(select)
        except ValueError:
            raise ValueError(
                f"Invalid select value: {select}. "
                f"Must be one of: {[p.value for p in MetricSelectionPolicy]}"
            )
    else:
        policy = select

    df = pd.read_csv(csv_path, comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else METRIC_COLUMNS
    keep_cols: List[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    keep_cols.extend([c for c in metric_cols if c in df.columns])

    if not keep_cols:
        raise ValueError("No requested columns found in the CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row (first row often contains units like "%", "inst", etc.)
    if len(sub) > 0:
        first_row_str = sub.iloc[0].astype(str).str.lower()
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        if first_row_str.apply(lambda x: any(tok in x for tok in unit_tokens)).any():
            sub = sub.iloc[1:].reset_index(drop=True)

    # Coerce metrics to numeric
    if coerce_numeric:
        metric_in_sub = [c for c in metric_cols if c in sub.columns]
        sub[metric_in_sub] = (
            sub[metric_in_sub]
            .replace({",": "", "%": ""}, regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )

    # Filter by kernel name list if provided
    if name_list:
        sub = _filter_by_kernel_names(sub, name_list, policy, keep_cols)
    else:
        # Apply selection to all rows if no name filter
        sub = _apply_selection_policy(sub, policy)

    return sub


def metrics_to_prompt(
    df: pd.DataFrame,
    title: str = "GPU Profiling Metrics:",
    key_by: str = "Kernel Name",
    round_digits: Optional[int] = 3,
    compact: bool = False,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Convert NCU metrics DataFrame to JSON string for LLM prompts.

    Returns JSON in format:
    {
      "<kernel_name>": { "<metric>": <value>, ... }
    }
    Args:
        df: DataFrame with NCU metrics
        title: Title for the metrics section (not included in output)
        key_by: Column to use as key (usually "Kernel Name")
        round_digits: Number of decimal places for rounding
        compact: If True, use compact JSON (no indentation)
        keep_cols: Specific columns to include in output

    Returns:
        JSON string with metrics
    """

    def _safe(v: Any) -> Any:
        """Convert values to JSON-safe format."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    # Empty table
    if df is None or df.empty:
        return "{}"

    cols = list(df.columns)

    # Round numeric columns
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    # If key column is missing, return a list of rows
    if key_by not in cols:
        rows = [
            {k: _safe(v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ]
        return json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)

    # Determine value columns
    value_cols = [c for c in cols if c != key_by]
    if keep_cols is not None:
        value_cols = [c for c in value_cols if c in keep_cols]

    data: Dict[str, Any] = {}
    for rec in df[[key_by] + value_cols].to_dict(orient="records"):
        k = str(rec.pop(key_by))
        val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
        if k in data:
            # Multiple rows for same key - convert to list
            if isinstance(data[k], list):
                data[k].append(val_obj)
            else:
                data[k] = [data[k], val_obj]
        else:
            data[k] = val_obj

    return json.dumps(data, ensure_ascii=False, indent=None if compact else 2)


if __name__ == "__main__":
    print("ncu_profiler module loaded.")
    print("Import its functions in your scripts:")
    print(
        "  from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import profile_triton_kernel, load_ncu_metrics, metrics_to_prompt"
    )
