#!/usr/bin/env python3
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
"""Gradio UI for the hardware-guided kernel optimization pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Ensure project root is importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _list_kernelbench_problems(base: Path) -> list[tuple[str, str]]:
    """Return list of (label, absolute_path) pairs for KernelBench problems."""
    problems: list[tuple[str, str]] = []
    if not base.exists():
        return problems
    for level_dir in sorted(base.glob("level*")):
        if not level_dir.is_dir():
            continue
        if level_dir.name.lower() == "level4":
            continue
        for problem in sorted(level_dir.glob("*.py")):
            label = f"{level_dir.name}/{problem.name}"
            problems.append((label, str(problem.resolve())))
    return problems


def _discover_problems() -> list[tuple[str, str]]:
    """Find KernelBench problems from common locations."""
    candidate_roots = [
        Path.cwd() / "external" / "KernelBench" / "KernelBench",
        Path.cwd() / "KernelBench" / "KernelBench",
        Path.cwd().parent / "KernelBench" / "KernelBench",
        Path.cwd().parent.parent / "KernelBench" / "KernelBench",
    ]
    seen: set[str] = set()
    problems: list[tuple[str, str]] = []
    for root in candidate_roots:
        for label, path in _list_kernelbench_problems(root):
            if path not in seen:
                seen.add(path)
                problems.append((label, path))
    return problems


_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

_CUSTOM_OPTION = "-- Custom (paste below) --"


def _discover_examples() -> list[tuple[str, str]]:
    """Find optimization examples from the examples/ directory.

    Returns list of (label, directory_path) for dirs matching ``optimize_*``
    that contain ``input.py`` and ``test.py``.
    """
    examples: list[tuple[str, str]] = []
    if not _EXAMPLES_DIR.is_dir():
        return examples
    for d in sorted(_EXAMPLES_DIR.glob("optimize_*")):
        if not d.is_dir():
            continue
        if (d / "input.py").exists() and (d / "test.py").exists():
            # Turn "optimize_01_matvec" into "MatVec"
            label = d.name.split("_", 2)[-1].replace("_", " ").title()
            examples.append((label, str(d)))
    return examples


def _build_input_choices() -> list[str]:
    """Build the dropdown choices: examples + custom."""
    choices: list[str] = []
    for label, _ in _discover_examples():
        choices.append(f"Example: {label}")
    choices.append(_CUSTOM_OPTION)
    return choices


def _get_gpu_choices() -> list[str]:
    """Return GPU names from the specs database."""
    from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs_database import (
        GPU_SPECS_DATABASE,
    )

    return sorted(GPU_SPECS_DATABASE.keys())


def _env_var_for_model(model_name: str) -> str:
    """Determine which API key env var a model needs."""
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def _load_sibling_file(problem_path: str, filename: str) -> str:
    """Load a sibling file (input.py, test.py) next to a problem file."""
    if not problem_path:
        return ""
    parent = Path(problem_path).parent
    candidate = parent / filename
    if candidate.exists():
        try:
            return candidate.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""


def run_optimization(
    problem_label: str,
    kernel_code: str,
    test_code: str,
    model_name: str,
    gpu_name: str,
    max_rounds: int,
    high_reasoning: bool,
    platform: str,
    api_key: str | None,
    strategy: str = "greedy",
    num_workers: int = 1,
    strategy_config: dict | None = None,
    problem_file_override: str | None = None,
    log_capture: _LogCapture | None = None,
) -> tuple[str, str, str, str | None, str]:
    """Run the optimization pipeline and return (status_md, best_kernel, log, download_path, per_round_html)."""
    from triton_kernel_agent.opt_manager import OptimizationManager

    if not kernel_code or not kernel_code.strip():
        return "**Error:** No kernel code provided.", "", "", None, ""
    if not test_code or not test_code.strip():
        return "**Error:** No test code provided.", "", "", None, ""

    # Resolve API key
    env_var = _env_var_for_model(model_name)
    user_key = api_key.strip() if api_key else None
    original_env_key = os.environ.get(env_var)
    temp_key_set = False
    if user_key:
        os.environ[env_var] = user_key
        temp_key_set = True

    try:
        # Set up run directory
        ts = int(time.time())
        run_dir = Path.cwd() / ".optimize" / f"optimization_{ts}"
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve problem file: explicit override > KB label lookup > stub
        if problem_file_override and Path(problem_file_override).exists():
            problem_file = Path(problem_file_override)
        else:
            problem_mapping = {label: path for label, path in _discover_problems()}
            source_problem = problem_mapping.get(problem_label, "")
            if source_problem and Path(source_problem).exists():
                problem_file = Path(source_problem)
            else:
                # Write a stub problem file from kernel code context
                problem_file = run_dir / "problem.py"
                problem_file.parent.mkdir(parents=True, exist_ok=True)
                problem_file.write_text(
                    "# Auto-generated problem stub\n"
                    "import torch\nimport torch.nn as nn\n\n"
                    "class Model(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "    def forward(self, x):\n"
                    "        return x\n",
                    encoding="utf-8",
                )

        # Set up log capture on the OptimizationManager logger
        if log_capture is None:
            log_capture = _LogCapture()
        log_capture.metadata["log_dir"] = str(run_dir)

        mgr_logger = logging.getLogger("OptimizationManager")
        stream_handler = logging.StreamHandler(log_capture)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        mgr_logger.addHandler(stream_handler)

        try:
            manager = OptimizationManager(
                strategy=strategy,
                num_workers=num_workers,
                max_rounds=max_rounds,
                log_dir=run_dir,
                strategy_config=strategy_config,
                openai_model=model_name,
                high_reasoning_effort=high_reasoning,
                gpu_name=gpu_name,
                target_platform=platform,
            )

            result = manager.run_optimization(
                initial_kernel=kernel_code,
                problem_file=problem_file,
                test_code=test_code,
            )
        finally:
            mgr_logger.removeHandler(stream_handler)

        # Build status markdown
        status_md = _build_status_markdown(result, strategy, num_workers)

        # Build per-round best data from database
        per_round_best: dict[int, dict] = {}
        try:
            all_entries = manager.database.get_all()
            for entry in all_entries:
                gen = entry.generation
                if gen is None or gen < 1:
                    continue
                time_ms = entry.metrics.time_ms
                if (
                    gen not in per_round_best
                    or time_ms < per_round_best[gen]["time_ms"]
                ):
                    per_round_best[gen] = {
                        "time_ms": time_ms,
                        "program_id": entry.program_id,
                        "kernel_code": entry.kernel_code,
                    }
        except Exception:
            pass
        round_html = _build_per_round_html(per_round_best)

        # Save best kernel for download
        best_kernel = result.get("kernel_code") or ""
        download_path = None
        if best_kernel:
            best_file = output_dir / "best_kernel.py"
            best_file.write_text(best_kernel, encoding="utf-8")
            download_path = str(best_file)

        log_text = log_capture.getvalue()
        return status_md, best_kernel, log_text, download_path, round_html

    except Exception as e:
        tb = traceback.format_exc()
        return f"## Error\n\n```\n{e}\n```\n\n```\n{tb}\n```", "", "", None, ""
    finally:
        if temp_key_set:
            if original_env_key is not None:
                os.environ[env_var] = original_env_key
            else:
                os.environ.pop(env_var, None)


def _build_status_markdown(result: dict, strategy: str, num_workers: int) -> str:
    """Build the final status markdown from an OptimizationManager result dict."""
    if not result.get("success"):
        return f"## Optimization Failed\n\n{result.get('error', '') or 'No improvement found.'}"

    best_time = result.get("best_time_ms", 0)
    total_rounds = result.get("total_rounds", 0)
    top_kernels = result.get("top_kernels", [])
    initial_kernel_time = result.get("initial_kernel_time_ms", float("inf"))
    pytorch_baseline = result.get("pytorch_baseline_ms", float("inf"))
    pytorch_compile = result.get("pytorch_compile_ms", float("inf"))

    strategy_label = (
        f"Beam Search ({num_workers} workers)"
        if strategy == "beam_search"
        else f"Greedy ({num_workers} worker)"
    )

    status_md = "## Optimization Complete\n\n"
    status_md += "| Metric | Value |\n|---|---|\n"
    status_md += f"| Best Time | {best_time:.4f} ms |\n"
    if initial_kernel_time != float("inf"):
        status_md += f"| Initial Kernel | {initial_kernel_time:.4f} ms |\n"
    if pytorch_baseline != float("inf"):
        status_md += f"| PyTorch Eager | {pytorch_baseline:.4f} ms |\n"
    if pytorch_compile != float("inf"):
        status_md += f"| PyTorch Compile | {pytorch_compile:.4f} ms |\n"
    if initial_kernel_time != float("inf") and best_time > 0:
        speedup = initial_kernel_time / best_time
        status_md += f"| Speedup vs Initial | {speedup:.2f}x |\n"
    status_md += f"| Rounds | {total_rounds} |\n"
    status_md += f"| Strategy | {strategy_label} |\n"

    if len(top_kernels) > 1:
        status_md += f"| Top Kernels | {len(top_kernels)} found |\n"
        status_md += "\n### Top Kernels\n"
        status_md += "| # | Time (ms) | Generation |\n|---|---|---|\n"
        for i, k in enumerate(top_kernels, 1):
            status_md += f"| {i} | {k['time_ms']:.4f} | {k.get('generation', '-')} |\n"

    return status_md


def _build_per_round_html(per_round_best: dict[int, dict]) -> str:
    """Render per-round best results as collapsible HTML sections.

    Args:
        per_round_best: Mapping of round number to best entry dict
            with keys: time_ms, program_id, kernel_code (snippet).

    Returns:
        HTML string with <details>/<summary> sections, last round open.
    """
    if not per_round_best:
        return ""
    parts = ["<h3>Per-Round Results</h3>"]
    max_round = max(per_round_best)
    for rnd in sorted(per_round_best):
        entry = per_round_best[rnd]
        time_ms = entry.get("time_ms", float("inf"))
        prog_id = entry.get("program_id", "?")
        # Last round is open by default
        open_attr = " open" if rnd == max_round else ""
        parts.append(f"<details{open_attr}>")
        parts.append(
            f"<summary>Round {rnd}: {time_ms:.4f} ms (ID: {prog_id})</summary>"
        )
        code = entry.get("kernel_code", "")
        if code:
            # Show first 30 lines as preview
            lines = code.splitlines()[:30]
            preview = "\n".join(lines)
            if len(code.splitlines()) > 30:
                preview += "\n# ... (truncated)"
            parts.append(f"<pre><code>{preview}</code></pre>")
        parts.append("</details>")
    return "\n".join(parts)


class _LogCapture:
    """Thread-safe stream-like object that captures log messages."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._lock = threading.Lock()
        self._read_index: int = 0
        self.metadata: dict = {}

    def write(self, msg: str) -> None:
        with self._lock:
            self._parts.append(msg)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)

    def get_new_lines(self) -> str:
        """Return log content appended since the last call."""
        with self._lock:
            new = self._parts[self._read_index :]
            self._read_index = len(self._parts)
        return "".join(new)


# Patterns matched against the *message* portion of each log line (after the
# ``asctime - LEVEL - `` prefix).  Order matters: first match wins per line.
_LOG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Round boundary
    (re.compile(r"ROUND\s+(\d+)/(\d+)"), "round"),
    # Orchestrator phase transitions (exact prefixes to avoid duplicates)
    (re.compile(r"\[\d+\] Profiling current kernel with NCU"), "phase_profile"),
    (re.compile(r"\[\d+\] Analyzing bottleneck"), "phase_analyze"),
    (re.compile(r"\[\d+\] Using pre-computed bottleneck"), "phase_analyze"),
    (re.compile(r"\[\d+\] Generating optimized kernel"), "phase_generate"),
    (re.compile(r"\[\d+\] Verifying correctness"), "phase_verify"),
    # Verification result
    (re.compile(r"\[\d+\].*Correctness check passed"), "verify_pass"),
    (re.compile(r"\[\d+\].*Correctness check failed"), "verify_fail"),
    # Performance results
    (
        re.compile(r"NEW BEST RUNTIME.*?(\d+\.?\d*)\s*ms.*?speedup:\s*(\d+\.?\d*)x"),
        "new_best",
    ),
    (
        re.compile(
            r"\[\d+\] No improvement:\s*(\d+\.?\d*)\s*ms.*?best\s+(\d+\.?\d*)\s*ms"
        ),
        "no_improve",
    ),
    # Manager-level baselines (must precede worker-level "Baseline time:")
    (re.compile(r"PyTorch baseline:\s*(\d+\.?\d*)ms"), "pytorch_eager"),
    (re.compile(r"PyTorch compile baseline:\s*(\d+\.?\d*)ms"), "pytorch_compile"),
    (re.compile(r"Initial kernel time:\s*(\d+\.?\d*)ms"), "initial_kernel_time"),
    (re.compile(r"Speedup vs initial kernel:\s*(\d+\.?\d*)x"), "final_speedup_initial"),
    (re.compile(r"Speedup vs PyTorch eager:\s*(\d+\.?\d*)x"), "final_speedup_pytorch"),
    # Worker-level baseline
    (re.compile(r"Baseline time:\s*(\d+\.?\d*)\s*ms"), "baseline"),
    (re.compile(r"Using known kernel time:\s*(\d+\.?\d*)\s*ms"), "baseline"),
    # Roofline / SOL (orchestrator-level, has "-bound" context)
    (re.compile(r"Baseline SOL:\s*(\d+\.?\d*)%.*?(\w+)-bound"), "baseline_sol"),
    (re.compile(r"\[\d+\] Roofline.*?(\w+)-bound.*?(\d+\.?\d*)% SOL"), "roofline"),
    # Per-round best from manager
    (re.compile(r"Round (\d+) best: worker (\d+) at (\d+\.?\d*) ms"), "round_best"),
    (re.compile(r"Round (\d+): no successful workers"), "round_no_success"),
    # Early termination
    (re.compile(r"\[\d+\].*Early termination:\s*(.+)"), "early_stop"),
    # Final summary
    (re.compile(r"OPTIMIZATION COMPLETE"), "done"),
    (re.compile(r"Speedup vs baseline:\s*(\d+\.?\d*)x"), "final_speedup"),
    # Errors
    (re.compile(r"timeout|timed?\s*out", re.IGNORECASE), "error"),
    (re.compile(r"LLM.*?failed", re.IGNORECASE), "error"),
]

_WORKER_DIR_RE = re.compile(r"/w(\d+)/")


def _tail_worker_logs(log_dir: str, offsets: dict[str, int]) -> dict[int, str]:
    """Read new content from worker log files since last poll.

    Args:
        log_dir: Root log directory (same as run_dir).
        offsets: Mutable dict mapping log file path -> last read position.

    Returns:
        Dict mapping worker_id (int) to new log content for that worker.
    """
    per_worker: dict[int, list[str]] = {}
    workers_dir = Path(log_dir) / "workers"
    if not workers_dir.exists():
        return {}
    for log_file in sorted(workers_dir.glob("w*/r*/logs/*.log")):
        path_str = str(log_file)
        prev = offsets.get(path_str, 0)
        wid_match = _WORKER_DIR_RE.search(path_str)
        wid = int(wid_match.group(1)) if wid_match else 0
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                f.seek(prev)
                chunk = f.read()
                if chunk:
                    offsets[path_str] = prev + len(chunk)
                    per_worker.setdefault(wid, []).append(chunk)
        except OSError:
            pass
    return {wid: "".join(parts) for wid, parts in per_worker.items()}


_TIMESTAMP_RE = re.compile(r"(\d{2}:\d{2}:\d{2})")


def _parse_log_for_status(raw_lines: str, manager_round: str = "") -> str:
    """Extract curated status lines from raw log output, prefixed with timestamps.

    Args:
        raw_lines: Raw log text.
        manager_round: If set (e.g. "3/5"), worker-level "Round 1/1" lines
            are rewritten to show the real manager round instead.
    """
    if not raw_lines:
        return ""
    curated: list[str] = []
    for line in raw_lines.splitlines():
        # Extract HH:MM:SS from the log prefix
        ts_match = _TIMESTAMP_RE.search(line)
        ts = ts_match.group(1) if ts_match else ""

        for pattern, kind in _LOG_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            prefix = f"[{ts}] " if ts else ""
            if kind == "round":
                round_label = (
                    manager_round if manager_round else f"{m.group(1)}/{m.group(2)}"
                )
                curated.append(f"\n{prefix}=== Round {round_label} ===")
            elif kind == "phase_profile":
                curated.append(f"{prefix}  Profiling kernel (NCU)...")
            elif kind == "phase_analyze":
                curated.append(f"{prefix}  Analyzing bottleneck...")
            elif kind == "phase_generate":
                curated.append(f"{prefix}  Generating optimized kernel...")
            elif kind == "phase_verify":
                curated.append(f"{prefix}  Verifying correctness...")
            elif kind == "verify_pass":
                curated.append(f"{prefix}  Correctness: PASSED")
            elif kind == "verify_fail":
                curated.append(f"{prefix}  Correctness: FAILED")
            elif kind == "new_best":
                time_val = m.group(1)
                speedup_val = float(m.group(2))
                if speedup_val > 1.0:
                    curated.append(
                        f"{prefix}  \U0001f389 SPEEDUP {speedup_val:.2f}x \u2014 NEW BEST: {time_val} ms"
                    )
                else:
                    curated.append(
                        f"{prefix}  NEW BEST: {time_val} ms (speedup {m.group(2)}x)"
                    )
            elif kind == "no_improve":
                curated.append(
                    f"{prefix}  No improvement ({m.group(1)} ms, best {m.group(2)} ms)"
                )
            elif kind == "pytorch_eager":
                curated.append(f"{prefix}PyTorch eager baseline: {m.group(1)} ms")
            elif kind == "pytorch_compile":
                curated.append(f"{prefix}PyTorch compile baseline: {m.group(1)} ms")
            elif kind == "initial_kernel_time":
                curated.append(f"{prefix}Initial kernel: {m.group(1)} ms")
            elif kind == "final_speedup_initial":
                curated.append(f"{prefix}  Speedup vs initial kernel: {m.group(1)}x")
            elif kind == "final_speedup_pytorch":
                curated.append(f"{prefix}  Speedup vs PyTorch eager: {m.group(1)}x")
            elif kind == "baseline":
                curated.append(f"{prefix}  Worker baseline: {m.group(1)} ms")
            elif kind == "baseline_sol":
                curated.append(
                    f"{prefix}Baseline SOL: {m.group(1)}% ({m.group(2)}-bound)"
                )
            elif kind == "roofline":
                curated.append(
                    f"{prefix}  Roofline: {m.group(1)}-bound, {m.group(2)}% SOL"
                )
            elif kind == "round_best":
                curated.append(
                    f"{prefix}  Round {m.group(1)} winner: worker {m.group(2)} at {m.group(3)} ms"
                )
            elif kind == "round_no_success":
                curated.append(f"{prefix}  Round {m.group(1)}: no successful workers")
            elif kind == "early_stop":
                curated.append(f"{prefix}  Early stop: {m.group(1).strip()}")
            elif kind == "done":
                curated.append(f"\n{prefix}OPTIMIZATION COMPLETE")
            elif kind == "final_speedup":
                curated.append(f"{prefix}  Final speedup: {m.group(1)}x")
            elif kind == "error":
                curated.append(f"{prefix}  [ERROR] {m.group(0)}")
            break  # first matching pattern per line
    return "\n".join(curated)


def build_interface() -> gr.Blocks:
    from utils.providers.models import _get_model_name_to_config

    # Build dropdown: examples + custom
    input_choices = _build_input_choices()
    default_input = input_choices[0] if input_choices else _CUSTOM_OPTION

    # Pre-load default example content so fields aren't empty on launch
    _examples = _discover_examples()
    _example_map_init: dict[str, str] = {
        f"Example: {label}": dirpath for label, dirpath in _examples
    }
    default_kernel = ""
    default_test = ""
    if default_input in _example_map_init:
        _d = Path(_example_map_init[default_input])
        try:
            default_kernel = (_d / "input.py").read_text(encoding="utf-8")
        except OSError:
            pass
        try:
            default_test = (_d / "test.py").read_text(encoding="utf-8")
        except OSError:
            pass

    model_names = sorted(_get_model_name_to_config().keys()) or ["gpt-5"]
    default_model = "gpt-5" if "gpt-5" in model_names else model_names[0]

    gpu_choices = _get_gpu_choices()
    default_gpu = gpu_choices[0] if gpu_choices else ""

    with gr.Blocks(
        title="KernelAgent — Optimization UI",
        theme=gr.themes.Soft(),
        css=".worker-log textarea { background-color: #f5f5f5 !important; }",
    ) as app:
        gr.Markdown(
            "# KernelAgent — Kernel Optimization\n\n"
            "Hardware-guided optimization: NCU profiling, roofline analysis, "
            "LLM bottleneck diagnosis, and iterative refinement.\n\n"
            "We have prepared **three examples** to get started — pick one "
            "from the dropdown, or paste your own kernel and test code.\n\n"
            "**Note:** 5 rounds of optimization can take about 30 minutes."
        )

        with gr.Row():
            # Left column: configuration
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")

                api_key_input = gr.Textbox(
                    label="API Key (optional)",
                    placeholder="sk-... or sk-ant-...",
                    type="password",
                    value="",
                    info="Session-only. Uses env var from .env if empty.",
                )

                input_dropdown = gr.Dropdown(
                    choices=input_choices,
                    label="Input Source",
                    value=default_input,
                    interactive=True,
                    info="Pick an example to get started, or select Custom to paste your own.",
                )

                kernel_input = gr.Textbox(
                    label="Kernel Code",
                    placeholder="Paste a verified Triton kernel here...",
                    lines=12,
                    max_lines=30,
                    value=default_kernel,
                )

                test_input = gr.Textbox(
                    label="Test Code",
                    placeholder="Paste test code here...",
                    lines=8,
                    max_lines=20,
                    value=default_test,
                )

                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    label="Model",
                    value=default_model,
                    interactive=True,
                )

                strategy_radio = gr.Radio(
                    choices=["Greedy (1 worker)", "Beam Search (4 workers)"],
                    value="Greedy (1 worker)",
                    label="Search Strategy",
                )

                gpu_dropdown = gr.Dropdown(
                    choices=gpu_choices,
                    label="GPU",
                    value=default_gpu,
                    interactive=True,
                    info="Select the GPU on your machine.",
                )

                max_rounds_slider = gr.Slider(
                    1, 10, value=5, step=1, label="Max Optimization Rounds"
                )

                high_reasoning_cb = gr.Checkbox(
                    label="High Reasoning Effort",
                    value=True,
                    info="Use high reasoning for better quality (o4-mini/o3 series).",
                )

                optimize_button = gr.Button("Optimize Kernel", variant="primary")

            # Right column: results with tabs
            with gr.Column(scale=2):
                gr.Markdown("## Results")

                status_output = gr.Markdown(
                    value="*Ready — select a problem and paste a kernel to optimize.*"
                )

                with gr.Tab("Log"):
                    manager_log_output = gr.Textbox(
                        label="Manager",
                        interactive=False,
                        lines=8,
                        max_lines=20,
                    )
                    with gr.Row():
                        with gr.Column() as w0_col:
                            w0_log = gr.Textbox(
                                label="Worker 0",
                                interactive=False,
                                lines=18,
                                max_lines=40,
                                elem_classes=["worker-log"],
                            )
                        with gr.Column(visible=False) as w1_col:
                            w1_log = gr.Textbox(
                                label="Worker 1",
                                interactive=False,
                                lines=18,
                                max_lines=40,
                                elem_classes=["worker-log"],
                            )
                        with gr.Column(visible=False) as w2_col:
                            w2_log = gr.Textbox(
                                label="Worker 2",
                                interactive=False,
                                lines=18,
                                max_lines=40,
                                elem_classes=["worker-log"],
                            )
                        with gr.Column(visible=False) as w3_col:
                            w3_log = gr.Textbox(
                                label="Worker 3",
                                interactive=False,
                                lines=18,
                                max_lines=40,
                                elem_classes=["worker-log"],
                            )

                with gr.Tab("Best Kernel"):
                    kernel_output = gr.Code(
                        label="Optimized Kernel",
                        language="python",
                        interactive=False,
                        lines=25,
                    )
                    per_round_html = gr.HTML(
                        value="",
                        label="Per-Round Results",
                    )

                with gr.Tab("Download"):
                    download_output = gr.File(
                        label="Download best kernel",
                        interactive=False,
                    )

        # Wire input dropdown to auto-load kernel and test code
        _example_map = _example_map_init

        def _read_file(path: Path) -> str:
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""

        def on_input_selected(label: str) -> tuple[str, str]:
            if label == _CUSTOM_OPTION or not label:
                return "", ""
            if label in _example_map:
                d = Path(_example_map[label])
                return _read_file(d / "input.py"), _read_file(d / "test.py")
            return "", ""

        input_dropdown.change(
            fn=on_input_selected,
            inputs=input_dropdown,
            outputs=[kernel_input, test_input],
        )

        # Toggle worker column visibility based on strategy
        def on_strategy_change(choice: str):
            is_beam = choice == "Beam Search (4 workers)"
            return (
                gr.update(visible=True),
                gr.update(visible=is_beam),
                gr.update(visible=is_beam),
                gr.update(visible=is_beam),
            )

        strategy_radio.change(
            fn=on_strategy_change,
            inputs=strategy_radio,
            outputs=[w0_col, w1_col, w2_col, w3_col],
        )

        # Wire optimize button
        def _parse_strategy(choice: str) -> tuple[str, int, dict]:
            """Map strategy radio label to (strategy, num_workers, strategy_config)."""
            if choice == "Beam Search (4 workers)":
                return "beam_search", 4, {"num_top_kernels": 2, "num_bottlenecks": 2}
            return "greedy", 1, {"max_no_improvement": 3}

        def on_optimize(
            input_label: str,
            kernel_code: str,
            test_code: str,
            model_name: str,
            strategy_choice: str,
            gpu_name: str,
            max_rounds: int,
            high_reasoning: bool,
            api_key: str | None,
        ):
            strategy, num_workers, strategy_config = _parse_strategy(strategy_choice)

            # Resolve problem_label and problem_file_override from input source
            problem_label = ""
            problem_file_override = None
            if input_label.startswith("KB: "):
                problem_label = input_label[4:]
            elif input_label in _example_map:
                problem_file_override = str(
                    Path(_example_map[input_label]) / "problem.py"
                )

            log_capture = _LogCapture()
            result: list[tuple[str, str, str, str | None]] = []
            error: list[BaseException] = []

            def _worker() -> None:
                try:
                    result.append(
                        run_optimization(
                            problem_label=problem_label,
                            kernel_code=kernel_code,
                            test_code=test_code,
                            model_name=model_name,
                            gpu_name=gpu_name,
                            max_rounds=int(max_rounds),
                            high_reasoning=high_reasoning,
                            platform="cuda",
                            api_key=api_key,
                            strategy=strategy,
                            num_workers=num_workers,
                            strategy_config=strategy_config,
                            problem_file_override=problem_file_override,
                            log_capture=log_capture,
                        )
                    )
                except BaseException as exc:
                    error.append(exc)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            # Accumulated curated logs: manager + per-worker
            mgr_curated = ""
            worker_curated: dict[int, str] = {i: "" for i in range(4)}
            # Track live status from log lines
            current_round = ""
            current_phase = ""
            best_info = ""
            _round_re = re.compile(r"Round (\d+/\d+)")
            _best_re = re.compile(r"NEW BEST: (.+)")
            worker_log_offsets: dict[str, int] = {}

            def _poll_logs() -> None:
                nonlocal mgr_curated, current_round, current_phase, best_info
                # Manager-level log
                mgr_new = log_capture.get_new_lines()
                if mgr_new:
                    parsed = _parse_log_for_status(mgr_new)
                    if parsed:
                        mgr_curated += parsed + "\n"
                    for cline in (parsed or "").splitlines():
                        rm = _round_re.search(cline)
                        if rm:
                            current_round = rm.group(1)
                            current_phase = ""
                        for kw in ("Profiling", "Analyzing", "Generating", "Verifying"):
                            if kw in cline:
                                current_phase = kw.lower()
                        bm = _best_re.search(cline)
                        if bm:
                            best_info = bm.group(1)
                # Worker-level logs (per-worker)
                log_dir = log_capture.metadata.get("log_dir", "")
                if log_dir:
                    per_worker = _tail_worker_logs(log_dir, worker_log_offsets)
                    for wid, raw in per_worker.items():
                        parsed = _parse_log_for_status(raw, manager_round=current_round)
                        if parsed:
                            worker_curated[wid] = (
                                worker_curated.get(wid, "") + parsed + "\n"
                            )

            round_html_val: list[str] = []

            def _make_yield(status, kernel_code, download):
                return (
                    status,
                    kernel_code,
                    mgr_curated.rstrip(),
                    worker_curated.get(0, "").rstrip(),
                    worker_curated.get(1, "").rstrip(),
                    worker_curated.get(2, "").rstrip(),
                    worker_curated.get(3, "").rstrip(),
                    download,
                    round_html_val[-1] if round_html_val else "",
                )

            # Poll with a hard timeout so the generator always terminates.
            # 30 min per round × max_rounds + extra margin for baselines.
            poll_deadline = time.time() + int(max_rounds) * 1800 + 600
            while thread.is_alive():
                thread.join(timeout=2)
                _poll_logs()
                status_parts = ["**Optimizing…**"]
                if current_round:
                    status_parts.append(f"Round {current_round}")
                if current_phase:
                    status_parts.append(f"({current_phase})")
                if best_info:
                    status_parts.append(f"| Best so far: {best_info}")
                yield _make_yield(" ".join(status_parts), "", None)
                if time.time() > poll_deadline:
                    error.append(
                        TimeoutError("Optimization exceeded maximum wall time")
                    )
                    break

            # Drain remaining logs
            _poll_logs()

            if error:
                tb = "".join(traceback.format_exception(error[0]))
                yield _make_yield(
                    f"## Error\n\n```\n{error[0]}\n```\n\n```\n{tb}\n```",
                    "",
                    None,
                )
            elif result:
                status, best_kernel, raw_log, download_path, rh = result[0]
                round_html_val.append(rh)
                # If no curated manager log, fall back to raw
                if not mgr_curated.strip():
                    mgr_curated = raw_log
                yield _make_yield(status, best_kernel, download_path)
            else:
                yield _make_yield(
                    "## Error\n\nOptimization thread finished without result.",
                    "",
                    None,
                )

        optimize_button.click(
            fn=on_optimize,
            inputs=[
                input_dropdown,
                kernel_input,
                test_input,
                model_dropdown,
                strategy_radio,
                gpu_dropdown,
                max_rounds_slider,
                high_reasoning_cb,
                api_key_input,
            ],
            outputs=[
                status_output,
                kernel_output,
                manager_log_output,
                w0_log,
                w1_log,
                w2_log,
                w3_log,
                download_output,
                per_round_html,
            ],
            show_progress="hidden",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization UI")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    load_dotenv()
    app = build_interface()

    print("Starting Optimization UI...")

    meta_keyfile = Path("/var/facebook/x509_identities/server.pem")
    is_meta_devserver = meta_keyfile.exists()

    if is_meta_devserver:
        server_name = os.uname()[1]
        print(f"Meta devserver detected. Visit https://{server_name}:{args.port}/")
        app.launch(
            share=False,
            show_error=True,
            server_name=server_name,
            server_port=args.port,
            ssl_keyfile=str(meta_keyfile),
            ssl_certfile=str(meta_keyfile),
            ssl_verify=False,
            inbrowser=False,
        )
    else:
        print(f"Visit http://{args.host}:{args.port}/")
        app.launch(
            share=False,
            show_error=True,
            server_name=args.host,
            server_port=args.port,
            inbrowser=True,
        )


if __name__ == "__main__":
    main()
