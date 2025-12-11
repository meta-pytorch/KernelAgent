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
"""Gradio UI for FuserAgent (KernelFalcon project)."""

from __future__ import annotations

import argparse
import ast
import os
import tarfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Optional imports from our own toolkit
try:
    # direct import when running inside repo
    from Fuser.subgraph_extractor import extract_subgraphs_to_json  # type: ignore
except Exception:  # pragma: no cover
    extract_subgraphs_to_json = None  # type: ignore

from Fuser.config import OrchestratorConfig, new_run_id
from Fuser.orchestrator import Orchestrator
from Fuser.paths import ensure_abs_regular_file, make_run_dirs, PathSafetyError


@dataclass
class RunArtifacts:
    status_md: str
    summary_md: str
    code_text: str
    run_info_md: str
    zip_path: Optional[Path]


def _list_kernelbench_problems(base: Path) -> List[Tuple[str, str]]:
    """Return list of (label, absolute_path) pairs for KernelBench problems."""
    problems: List[Tuple[str, str]] = []
    if not base.exists():
        return problems
    for level_dir in sorted(base.glob("level*")):
        if not level_dir.is_dir():
            continue
        if level_dir.name.lower() == "level4":
            continue  # Skip Level 4 problems for the current UI
        for problem in sorted(level_dir.glob("*.py")):
            label = f"{level_dir.name}/{problem.name}"
            problems.append((label, str(problem.resolve())))
    return problems


def _format_classes_summary(code_text: str) -> str:
    """Produce a markdown summary of classes and functions defined in code_text."""
    if not code_text.strip():
        return "*No code generated.*"
    try:
        tree = ast.parse(code_text)
    except SyntaxError as exc:
        return f"*Unable to parse generated code: {exc}*"

    def base_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            cur: Optional[ast.AST] = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return ast.dump(node, include_attributes=False)

    class_lines: List[str] = ["## 🧩 Fusion Module Summary"]
    classes: List[ast.ClassDef] = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    functions: List[ast.FunctionDef] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef)
    ]

    if classes:
        for cls in classes:
            bases = [base_name(b) for b in cls.bases] or ["(no explicit bases)"]
            doc = ast.get_docstring(cls) or ""
            doc_first = doc.strip().splitlines()[0] if doc else "No docstring provided."
            methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
            class_lines.append(f"- **`{cls.name}`** — {doc_first}")
            class_lines.append(f"  - Bases: {', '.join(bases)}")
            if methods:
                class_lines.append(f"  - Methods: {', '.join(methods)}")
    else:
        class_lines.append("- *(No classes found in generated code.)*")

    if functions:
        class_lines.append("\n### Top-level Functions")
        for fn in functions:
            doc = ast.get_docstring(fn) or ""
            doc_first = doc.strip().splitlines()[0] if doc else "No docstring provided."
            class_lines.append(f"- **`{fn.name}`** — {doc_first}")

    return "\n".join(class_lines)


def _load_code_from_tar(artifact_path: Path) -> str:
    if not artifact_path.is_file():
        return ""
    with tarfile.open(artifact_path, "r:gz") as tf:
        try:
            member = tf.getmember("code.py")
        except KeyError:
            return ""
        extracted = tf.extractfile(member)
        if extracted is None:
            return ""
        return extracted.read().decode("utf-8")


def _create_zip_from_tar(artifact_path: Path, zip_path: Path) -> Optional[Path]:
    if not artifact_path.is_file():
        return None
    with (
        tarfile.open(artifact_path, "r:gz") as tf,
        zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf,
    ):
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            zf.writestr(member.name, data)
    return zip_path if zip_path.exists() else None


def _compose_run_info(
    run_dir: Path, summary_reason: str, elapsed: float, winner: Optional[str]
) -> str:
    lines = ["## 📁 Run Information"]
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Winner: `{winner or 'None'}`")
    lines.append(f"- Reason: {summary_reason}")
    lines.append(f"- Runtime: {elapsed:.2f} s")
    stream_path = run_dir / "orchestrator" / "stream.log"
    if stream_path.exists():
        lines.append(f"- Stream log: `{stream_path}`")
    return "\n".join(lines)


def run_fuser_problem(
    problem_path: str,
    model_name: str,
    workers: int,
    max_iters: int,
    llm_timeout: int,
    run_timeout: int,
    enable_reasoning: bool,
    user_api_key: Optional[str] = None,
) -> RunArtifacts:
    """Execute the Fuser orchestrator and collect artifacts."""
    if not problem_path:
        return RunArtifacts(
            status_md="❌ Please select a problem file.",
            summary_md="*No summary available.*",
            code_text="",
            run_info_md="",
            zip_path=None,
        )

    try:
        abs_path = ensure_abs_regular_file(problem_path)
    except PathSafetyError as exc:
        return RunArtifacts(
            status_md=f"❌ Invalid problem path: {exc}",
            summary_md="*No summary available.*",
            code_text="",
            run_info_md="",
            zip_path=None,
        )

    original_env_key = os.environ.get("OPENAI_API_KEY")
    temp_key_set = False
    if user_api_key and user_api_key.strip():
        os.environ["OPENAI_API_KEY"] = user_api_key.strip()
        temp_key_set = True
    elif not original_env_key:
        return RunArtifacts(
            status_md="❌ Provide an OpenAI API key (UI input or environment variable).",
            summary_md="*No summary available.*",
            code_text="",
            run_info_md="",
            zip_path=None,
        )

    start_time = time.time()
    try:
        cfg = OrchestratorConfig(
            problem_path=abs_path,
            model=model_name,
            workers=workers,
            max_iters=max_iters,
            llm_timeout_s=llm_timeout,
            run_timeout_s=run_timeout,
            stream_mode="winner",
            store_responses=False,
            isolated=False,
            deny_network=False,
            enable_reasoning_extras=enable_reasoning,
        )

        run_id = new_run_id()
        base_dir = Path.cwd() / ".fuse"
        base_dir.mkdir(exist_ok=True)
        dirs = make_run_dirs(base_dir, run_id)

        # Multiprocessing spawn mode for safety inside UI context
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        orch = Orchestrator(
            cfg,
            run_dir=dirs["run_dir"],
            workers_dir=dirs["workers"],
            orchestrator_dir=dirs["orchestrator"],
        )
        summary = orch.run()
        elapsed = time.time() - start_time

        if summary.winner_worker_id is None or not summary.artifact_path:
            status = f"❌ No passing solution. Reason: {summary.reason}"
            run_info = _compose_run_info(
                dirs["run_dir"], summary.reason, elapsed, summary.winner_worker_id
            )
            return RunArtifacts(
                status_md=status,
                summary_md="*No summary available.*",
                code_text="",
                run_info_md=run_info,
                zip_path=None,
            )

        artifact_path = Path(summary.artifact_path)
        code_text = _load_code_from_tar(artifact_path)
        summary_md = _format_classes_summary(code_text)
        zip_path = _create_zip_from_tar(
            artifact_path, dirs["run_dir"] / "fused_modules.zip"
        )

        status = (
            "✅ **Success!** "
            f"Worker `{summary.winner_worker_id}` passed via {summary.reason}; elapsed {elapsed:.2f}s"
        )
        run_info = _compose_run_info(
            dirs["run_dir"], summary.reason, elapsed, summary.winner_worker_id
        )
        return RunArtifacts(
            status_md=status,
            summary_md=summary_md,
            code_text=code_text,
            run_info_md=run_info,
            zip_path=zip_path,
        )

    except Exception as exc:
        elapsed = time.time() - start_time
        tb = traceback.format_exc()
        status = "❌ **Error during run**"
        summary_md = f"```\n{tb}\n```"
        run_info = _compose_run_info(
            dirs["run_dir"] if "dirs" in locals() else Path("."),
            f"Exception: {exc}",
            elapsed,
            None,
        )
        return RunArtifacts(
            status_md=status,
            summary_md=summary_md,
            code_text="",
            run_info_md=run_info,
            zip_path=None,
        )
    finally:
        if temp_key_set:
            if original_env_key is not None:
                os.environ["OPENAI_API_KEY"] = original_env_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)


class FuserAgentUI:
    """Stateful wrapper for the Gradio UI."""

    def __init__(self) -> None:
        load_dotenv()
        # Try common KernelBench locations and aggregate unique problems
        repo_root = Path(__file__).resolve().parents[1]
        candidate_roots = [
            repo_root / "external" / "KernelBench" / "KernelBench",
            Path.cwd() / "external" / "KernelBench" / "KernelBench",
            Path.cwd() / "KernelBench" / "KernelBench",
            Path.cwd().parent / "KernelBench" / "KernelBench",
        ]
        seen: set[str] = set()
        collected: list[tuple[str, str]] = []
        for base in candidate_roots:
            for label, abspath in _list_kernelbench_problems(base):
                if abspath not in seen:
                    collected.append((label, abspath))
                    seen.add(abspath)
        self.problem_choices = collected

    # ---------- Subgraph helpers ----------
    def _format_fuser_subgraphs_markdown(self, items: list[dict]) -> str:
        lines: list[str] = []
        for i, sg in enumerate(items, 1):
            title = sg.get("type") or "subgraph"
            sid = sg.get("id") or f"sg_{i}"
            lines.append(f"- Subgraph {i}: {title}")
            lines.append(f"  - id: {sid}")
            # ops line
            ops = sg.get("ops") or []
            if isinstance(ops, list):

                def op_str(op: dict) -> str:
                    if not isinstance(op, dict):
                        return str(op)
                    name = op.get("op", "op")
                    params = []
                    # Render common fields succinctly
                    for k in [
                        "in_channels",
                        "out_channels",
                        "num_groups",
                        "num_channels",
                        "kernel_size",
                        "stride",
                        "padding",
                        "dilation",
                        "output_padding",
                        "groups",
                        "affine",
                        "approximate",
                        "eps",
                    ]:
                        if k in op and op[k] is not None:
                            params.append(f"{k}={op[k]}")
                    return f"{name}(" + ", ".join(params) + ")"

                ops_line = ", ".join(op_str(o) for o in ops)
                lines.append(f"  - ops: {ops_line}")
            # shapes
            inp = sg.get("input_shape") or sg.get("inputs")
            out = sg.get("output_shape")
            if inp is not None and out is not None:
                if isinstance(inp, list) and inp and isinstance(inp[0], list):
                    lines.append(f"  - inputs: {inp}")
                else:
                    lines.append(f"  - input: {inp}, output: {out}")
            # weights
            wf = sg.get("weights_fused") or sg.get("weights")
            if wf:
                # flatten to simple summary
                parts = []
                if isinstance(wf, dict):
                    for k, v in wf.items():
                        parts.append(f"{k} {v}")
                lines.append("  - weights_fused: " + ", ".join(parts))
        return "\n".join(lines) if lines else "*No subgraphs.*"

    def _compute_fuser_subgraphs(
        self,
        problem_path: Path,
        model: str,
        workers: int,
        max_iters: int,
        llm_timeout: int,
        run_timeout: int,
    ) -> str:
        try:
            if extract_subgraphs_to_json is None:
                return "*Subgraph extractor not available in this environment.*"
            run_dir, json_path = extract_subgraphs_to_json(
                problem_path=problem_path,
                model_name=model,
                workers=workers,
                max_iters=max_iters,
                llm_timeout_s=llm_timeout,
                run_timeout_s=run_timeout,
            )
            import json

            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return "*Invalid subgraph JSON produced.*"
            return self._format_fuser_subgraphs_markdown(data)
        except Exception as e:
            return f"*Failed to extract subgraphs: {e}*"

    def _compute_torchcompile_subgraphs(
        self, problem_path: Path, strict: bool = False
    ) -> str:
        """Best-effort torch.compile view in FuserAgent style.

        For now, handle common patterns (conv/conv_transpose + activation) + GroupNorm.
        """
        try:
            import importlib.util
            import torch
            import torch.nn.functional as F  # noqa: F401
            import torch._dynamo as dynamo
            from torch.profiler import profile, ProfilerActivity

            spec = importlib.util.spec_from_file_location("kb_dyn", str(problem_path))
            if spec is None or spec.loader is None:
                return "*Unable to import problem file.*"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore

            model = mod.Model(*mod.get_init_inputs()).eval()
            x = mod.get_inputs()[0]
            # Probe attributes
            has_ct = hasattr(model, "conv_transpose")
            has_conv = hasattr(model, "conv")
            has_gn = hasattr(model, "group_norm") or hasattr(model, "gn")
            act_name = (
                "gelu"
                if "gelu" in mod.Model.forward.__code__.co_names
                else (
                    "tanh"
                    if "tanh" in mod.Model.forward.__code__.co_names
                    else "activation"
                )
            )

            subgraphs: list[dict] = []
            if has_ct:
                ct = model.conv_transpose
                sg1 = {
                    "id": f"convtranspose2d_{act_name}_N{x.shape[0]}_C{x.shape[1]}_{int(x.shape[2])}x{int(x.shape[3])}_to_N{x.shape[0]}_C{ct.out_channels}",
                    "type": f"conv_transpose2d+{act_name}",
                    "data_layout": "NCHW",
                    "dtype": str(x.dtype).split(".")[-1],
                    "ops": [
                        {
                            "op": "conv_transpose2d",
                            "in_channels": int(ct.in_channels),
                            "out_channels": int(ct.out_channels),
                            "kernel_size": list(map(int, ct.kernel_size)),
                            "stride": list(map(int, ct.stride)),
                            "padding": list(map(int, ct.padding)),
                            "dilation": list(map(int, ct.dilation)),
                            "output_padding": list(map(int, ct.output_padding)),
                            "groups": int(ct.groups),
                            "bias": ct.bias is not None,
                        },
                        {"op": act_name},
                    ],
                    "input_shape": list(map(int, x.shape)),
                    "output_shape": None,
                    "weights_fused": {
                        "weight": list(map(int, ct.weight.shape)),
                        "bias": list(map(int, ct.bias.shape))
                        if ct.bias is not None
                        else None,
                    },
                }
                subgraphs.append(sg1)
            elif has_conv:
                cv = model.conv
                subgraphs.append(
                    {
                        "id": f"conv2d_{act_name}",
                        "type": f"conv2d+{act_name}",
                        "data_layout": "NCHW",
                        "dtype": str(x.dtype).split(".")[-1],
                        "ops": [
                            {
                                "op": "conv2d",
                                "in_channels": int(cv.in_channels),
                                "out_channels": int(cv.out_channels),
                                "kernel_size": list(map(int, cv.kernel_size)),
                                "stride": list(map(int, cv.stride)),
                                "padding": list(map(int, cv.padding)),
                                "dilation": list(map(int, cv.dilation)),
                                "groups": int(cv.groups),
                                "bias": cv.bias is not None,
                            },
                            {"op": act_name},
                        ],
                        "input_shape": list(map(int, x.shape)),
                        "output_shape": None,
                        "weights_fused": {
                            "weight": list(map(int, cv.weight.shape)),
                            "bias": list(map(int, cv.bias.shape))
                            if cv.bias is not None
                            else None,
                        },
                    }
                )

            if has_gn:
                gn = getattr(model, "group_norm", getattr(model, "gn"))
                subgraphs.append(
                    {
                        "id": f"group_norm_N{x.shape[0]}_C{int(gn.num_channels)}",
                        "type": "group_norm",
                        "data_layout": "NCHW",
                        "dtype": str(x.dtype).split(".")[-1],
                        "ops": [
                            {
                                "op": "group_norm",
                                "num_groups": int(gn.num_groups),
                                "num_channels": int(gn.num_channels),
                                "eps": float(gn.eps),
                                "affine": bool(gn.affine),
                            }
                        ],
                        "input_shape": None,
                        "output_shape": None,
                        "weights_fused": {
                            "weight": list(map(int, gn.weight.shape))
                            if gn.weight is not None
                            else None,
                            "bias": list(map(int, gn.bias.shape))
                            if gn.bias is not None
                            else None,
                        },
                    }
                )

            notes: list[str] = []
            if strict:
                # Use dynamo.explain to capture partitioning and profiler for kernel counts
                try:
                    ex = dynamo.explain(model)(x)
                    gcount = int(getattr(ex, "graph_count", len(ex.graphs)))
                    gbreak = int(getattr(ex, "graph_break_count", 0))
                    notes.append(f"graphs={gcount}, graph_breaks={gbreak}")
                except Exception as _e:
                    notes.append(f"explain failed: {_e}")

                # Profile compiled model kernel launches
                try:
                    compiled = torch.compile(model, backend="inductor")
                    acts = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        acts.append(ProfilerActivity.CUDA)
                    with profile(activities=acts, record_shapes=False) as prof:
                        with torch.no_grad():
                            _ = compiled(x)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                    events = prof.key_averages()
                    # Count and sample names
                    kde = [
                        e
                        for e in events
                        if hasattr(e, "device_type")
                        and e.device_type == torch.device("cuda").type
                    ]
                    if not kde:
                        kde = [
                            e
                            for e in events
                            if "triton" in (e.key or "").lower()
                            or "cuda" in (e.key or "").lower()
                        ]
                    names = []
                    for ev in kde:
                        name = ev.key or "?"
                        if any(
                            s in name.lower()
                            for s in [
                                "triton",
                                "cudnn",
                                "conv",
                                "group_norm",
                                "batch_norm",
                                "gelu",
                                "tanh",
                            ]
                        ):
                            names.append(f"{name} x{int(ev.count)}")
                            if len(names) >= 8:
                                break
                    notes.append(
                        f"cuda_kernels={len(kde)}; sample: "
                        + (", ".join(names) or "(none)")
                    )
                except Exception as _e:
                    notes.append(f"profiler failed: {_e}")

            # Fill shapes by running pieces when available (kept lightweight)
            try:
                with torch.no_grad():
                    if has_ct:
                        y1 = model.conv_transpose(x)
                        y2 = (
                            getattr(torch.nn.functional, act_name)(y1)
                            if hasattr(torch.nn.functional, act_name)
                            else torch.tanh(y1)
                        )
                        subgraphs[0]["output_shape"] = list(map(int, y2.shape))
                        if len(subgraphs) > 1 and (has_gn):
                            subgraphs[1]["input_shape"] = list(map(int, y2.shape))
                            y3 = (
                                model.group_norm
                                if hasattr(model, "group_norm")
                                else model.gn
                            )(y2)
                            subgraphs[1]["output_shape"] = list(map(int, y3.shape))
                    elif has_conv:
                        y1 = model.conv(x)
                        y2 = (
                            getattr(torch.nn.functional, act_name)(y1)
                            if hasattr(torch.nn.functional, act_name)
                            else torch.tanh(y1)
                        )
                        subgraphs[0]["output_shape"] = list(map(int, y2.shape))
                        if len(subgraphs) > 1 and has_gn:
                            subgraphs[1]["input_shape"] = list(map(int, y2.shape))
                            y3 = (
                                model.group_norm
                                if hasattr(model, "group_norm")
                                else model.gn
                            )(y2)
                            subgraphs[1]["output_shape"] = list(map(int, y3.shape))
            except Exception as _e:
                notes.append(f"shape-probe failed: {_e}")

            md = self._format_fuser_subgraphs_markdown(subgraphs)
            if notes:
                md = md + "\n\n> " + "\n> ".join(notes)
            return md
        except Exception as e:
            return f"*Failed to derive torch.compile-style subgraphs: {e}*"

    def run(
        self,
        selected_problem: str,
        custom_problem: str,
        model_name: str,
        workers: int,
        max_iters: int,
        llm_timeout: int,
        run_timeout: int,
        enable_reasoning: bool,
        user_api_key: Optional[str],
    ) -> Tuple[str, str, str, str, Optional[str]]:
        problem_path = custom_problem.strip() or selected_problem
        artifacts = run_fuser_problem(
            problem_path=problem_path,
            model_name=model_name,
            workers=workers,
            max_iters=max_iters,
            llm_timeout=llm_timeout,
            run_timeout=run_timeout,
            enable_reasoning=enable_reasoning,
            user_api_key=user_api_key,
        )
        zip_str = str(artifacts.zip_path) if artifacts.zip_path else None
        return (
            artifacts.status_md,
            artifacts.summary_md,
            artifacts.code_text,
            artifacts.run_info_md,
            zip_str,
        )


def build_interface() -> gr.Blocks:
    ui = FuserAgentUI()
    default_problem = ui.problem_choices[0][1] if ui.problem_choices else ""

    with gr.Blocks(title="KernelFalcon FuserAgent", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
# 🦅 KernelFalcon — FuserAgent

Select a KernelBench problem, generate fusion-ready PyTorch subgraphs, and download the results.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")

                api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key (optional)",
                    placeholder="sk-...",
                    type="password",
                    value="",
                    info="Used only for this session; falls back to environment variable.",
                )

                problem_dropdown = gr.Dropdown(
                    choices=[label for label, _ in ui.problem_choices],
                    label="📁 KernelBench Problem",
                    value=ui.problem_choices[0][0] if ui.problem_choices else None,
                    interactive=True,
                )

                problem_mapping = {label: path for label, path in ui.problem_choices}

                def _problem_label_to_path(label: str) -> str:
                    return problem_mapping.get(label, "")

                problem_path_display = gr.Textbox(
                    label="Selected problem path",
                    value=default_problem,
                    interactive=False,
                )

                def update_problem_path(label: str) -> str:
                    return _problem_label_to_path(label)

                problem_dropdown.change(
                    fn=update_problem_path,
                    inputs=problem_dropdown,
                    outputs=problem_path_display,
                )

                custom_path_input = gr.Textbox(
                    label="Override problem path (optional)",
                    placeholder="/abs/path/to/problem.py",
                )

                model_input = gr.Textbox(
                    label="Model name",
                    value="gpt-5",
                    info="Model passed to Fuser workers",
                )

                workers_slider = gr.Slider(1, 8, value=4, step=1, label="Workers")
                max_iters_slider = gr.Slider(
                    1, 10, value=5, step=1, label="Max iterations"
                )
                llm_timeout_slider = gr.Slider(
                    60, 3600, value=2400, step=60, label="LLM timeout (s)"
                )
                run_timeout_slider = gr.Slider(
                    60, 3600, value=2400, step=60, label="Run timeout (s)"
                )
                reasoning_checkbox = gr.Checkbox(
                    label="Enable reasoning extras",
                    value=True,
                )
                strict_compile_checkbox = gr.Checkbox(
                    label="Strict (compile/profiler)",
                    value=False,
                    info="Run torch.compile + profiler to derive subgraphs and kernel counts",
                )

                generate_button = gr.Button("🚀 Run FuserAgent", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## Results")
                status_output = gr.Markdown(value="*Awaiting run...*")
                with gr.Tabs():
                    with gr.TabItem("Fused Code"):
                        summary_output = gr.Markdown(value="")
                        code_output = gr.Code(language="python", value="", lines=25)
                    with gr.TabItem("FuserAgent Subgraphs"):
                        fuser_subgraphs_output = gr.Markdown(
                            value="*Run first to populate.*"
                        )
                    with gr.TabItem("torch.compile Subgraphs"):
                        tc_subgraphs_output = gr.Markdown(
                            value="*Run first to populate.*"
                        )
                run_info_output = gr.Markdown(value="")
                download_output = gr.File(
                    label="Download fused modules", interactive=False
                )

        def generate(
            selected_label: str,
            custom_path: str,
            model: str,
            workers: int,
            max_iters: int,
            llm_timeout: int,
            run_timeout: int,
            reasoning: bool,
            strict_compile: bool,
            api_key: Optional[str],
        ):
            selected_path = problem_mapping.get(selected_label, default_problem)
            status, summary, code_text, run_info, zip_path = ui.run(
                selected_problem=selected_path,
                custom_problem=custom_path,
                model_name=model,
                workers=workers,
                max_iters=max_iters,
                llm_timeout=llm_timeout,
                run_timeout=run_timeout,
                enable_reasoning=reasoning,
                user_api_key=api_key,
            )
            # Compute additional tabs
            try:
                from pathlib import Path as _P

                problem_path = _P(custom_path.strip() or selected_path)
                fuser_md = ui._compute_fuser_subgraphs(
                    problem_path,
                    model,
                    max(1, workers // 2),
                    max(1, max_iters // 2),
                    llm_timeout,
                    run_timeout,
                )
                tc_md = ui._compute_torchcompile_subgraphs(
                    problem_path, strict=strict_compile
                )
            except Exception as _e:
                fuser_md = f"*Subgraph extraction error: {_e}*"
                tc_md = f"*torch.compile subgraph error: {_e}*"
            return status, summary, code_text, fuser_md, tc_md, run_info, zip_path

        generate_button.click(
            fn=generate,
            inputs=[
                problem_dropdown,
                custom_path_input,
                model_input,
                workers_slider,
                max_iters_slider,
                llm_timeout_slider,
                run_timeout_slider,
                reasoning_checkbox,
                api_key_input,
                strict_compile_checkbox,
            ],
            outputs=[
                status_output,
                summary_output,
                code_output,
                fuser_subgraphs_output,
                tc_subgraphs_output,
                run_info_output,
                download_output,
            ],
            show_progress=True,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FuserAgent UI")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    app = build_interface()

    print("🚀 Starting FuserAgent UI...")

    meta_keyfile = Path("/var/facebook/x509_identities/server.pem")
    is_meta_devserver = meta_keyfile.exists()

    if is_meta_devserver:
        server_name = os.uname()[1]
        print(f"🌐 Meta devserver detected. Visit https://{server_name}:{args.port}/")
        print("💡 Ensure you're on the Meta VPN.")
        app.launch(
            share=False,
            show_error=True,
            server_name=server_name,
            server_port=args.port,
            ssl_keyfile=str(meta_keyfile),
            ssl_certfile=str(meta_keyfile),
            ssl_verify=False,
            show_api=False,
            inbrowser=False,
        )
    else:
        print(f"🌐 Visit http://{args.host}:{args.port}/")
        app.launch(
            share=False,
            show_error=True,
            server_name=args.host,
            server_port=args.port,
            show_api=False,
            inbrowser=True,
        )


if __name__ == "__main__":
    main()
