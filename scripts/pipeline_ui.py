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
"""Gradio UI for the end-to-end pipeline (extract → dispatch → compose)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from Fuser.pipeline import run_pipeline
from Fuser.auto_agent import AutoKernelRouter
from Fuser.code_extractor import extract_single_python_file
from triton_kernel_agent.providers.models import (
    get_model_provider,
    MODEL_NAME_TO_CONFIG,
)


def _list_kernelbench_problems(base: Path) -> List[Tuple[str, str]]:
    """Return list of (label, absolute_path) pairs for KernelBench problems."""
    problems: List[Tuple[str, str]] = []
    if not base.exists():
        return problems
    for level_dir in sorted(base.glob("level*")):
        if not level_dir.is_dir():
            continue
        if level_dir.name.lower() == "level4":
            continue  # optional skip as in fuser_ui
        for problem in sorted(level_dir.glob("*.py")):
            label = f"{level_dir.name}/{problem.name}"
            problems.append((label, str(problem.resolve())))
    return problems


def _zip_dir(src_dir: Path, zip_path: Path) -> Optional[Path]:
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(src_dir):
                for fn in files:
                    p = Path(root) / fn
                    # store relative to src_dir
                    arc = p.relative_to(src_dir)
                    zf.write(p, arcname=str(arc))
        return zip_path if zip_path.exists() else None
    except Exception:
        return None


def _read_text_bounded(p: Path, max_bytes: int = 512 * 1024) -> str:
    try:
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


@dataclass
class PipelineArtifacts:
    status_md: str
    details_md: str
    code_text: str
    run_info_md: str
    zip_path: Optional[Path]


def _write_temp_problem(code: str) -> Path:
    base = Path.cwd() / ".fuse" / "custom_problems"
    base.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    p = base / f"problem_{ts}.py"
    p.write_text(code, encoding="utf-8")
    return p


def _maybe_synthesize_problem(desc: str, model_name: str) -> Path:
    """Create a problem file from a textual description.

    Heuristics:
    - If desc looks like Python (contains 'class Model' or 'def get_inputs'), use it as-is.
    - Else, call LLM to synthesize a KernelBench-compatible problem file.
    """
    txt = desc.strip()
    if not txt:
        raise ValueError("Empty problem description")
    looks_like_py = (
        ("class Model" in txt) or ("def get_inputs" in txt) or ("import torch" in txt)
    )
    if looks_like_py:
        # If it's a full file or a fenced block, try to extract clean python
        try:
            code = extract_single_python_file(txt).code
        except Exception:
            code = txt  # accept raw code
        return _write_temp_problem(code)

    # LLM synthesis path
    provider = get_model_provider(model_name)
    prompt = (
        "You are given a natural language description of a PyTorch problem.\n"
        "Produce ONE KernelBench-style problem file as a single ```python block that contains:\n"
        "- a Model(nn.Module) implementing the described computation in forward(x, ...),\n"
        "- get_inputs() returning a list of input tensors with concrete shapes,\n"
        "- get_init_inputs() returning constructor args for Model(...) if needed.\n"
        "Constraints:\n"
        "- Use only torch and torch.nn / torch.nn.functional.\n"
        "- Ensure shapes are consistent and moderate in size to run quickly on a single GPU.\n"
        "- Do not include training logic; inference path only.\n"
        "- No file I/O or network.\n\n"
        f"Problem description:\n{txt}\n\n"
        "Return only the single Python file in a code fence."
    )
    resp = provider.get_response(
        model_name, [{"role": "user", "content": prompt}], max_tokens=6000
    )
    code = extract_single_python_file(resp.content or "").code
    return _write_temp_problem(code)


def run_pipeline_ui(
    problem_path: str,
    problem_description: str,
    extract_model: str,
    dispatch_model: str,
    compose_model: str,
    dispatch_jobs: str,
    workers: int,
    max_iters: int,
    llm_timeout_s: int,
    run_timeout_s: int,
    compose_max_iters: int,
    verify: bool,
    auto_route: bool = False,
    router_model: Optional[str] = None,
    router_high_reasoning: bool = True,
    user_api_key: Optional[str] = None,
) -> PipelineArtifacts:
    if not problem_path:
        # If no path given, try description
        if not problem_description.strip():
            return PipelineArtifacts(
                status_md="❌ Select a problem or provide a description.",
                details_md="*No details.*",
                code_text="",
                run_info_md="",
                zip_path=None,
            )

    # Set API key precedence like fuser_ui
    original_env_key = os.environ.get("OPENAI_API_KEY")
    temp_key_set = False
    if user_api_key and user_api_key.strip():
        os.environ["OPENAI_API_KEY"] = user_api_key.strip()
        temp_key_set = True
    elif not original_env_key:
        return PipelineArtifacts(
            status_md="❌ Provide an OpenAI API key (UI input or environment variable).",
            details_md="*No details.*",
            code_text="",
            run_info_md="",
            zip_path=None,
        )

    start_time = time.time()
    try:
        # If a description is provided, synthesize or accept as code
        problem_file: Path
        if problem_description.strip():
            try:
                problem_file = _maybe_synthesize_problem(
                    problem_description, extract_model
                )
            except Exception as e:
                return PipelineArtifacts(
                    status_md=f"❌ Failed to use description: {e}",
                    details_md="*No details.*",
                    code_text="",
                    run_info_md="",
                    zip_path=None,
                )
        else:
            problem_file = Path(problem_path)

        # If auto-router enabled, let the router decide between KernelAgent and Fuser pipeline
        if auto_route:
            router = AutoKernelRouter(
                ka_model=None,  # use env default unless a KA-specific UI control is added
                ka_num_workers=workers,
                ka_max_rounds=10,
                ka_high_reasoning=True,
                router_model=router_model or "gpt-5",
                router_high_reasoning=router_high_reasoning,
                extract_model=extract_model,
                dispatch_model=dispatch_model,
                compose_model=compose_model,
                workers=workers,
                max_iters=max_iters,
                llm_timeout_s=llm_timeout_s,
                run_timeout_s=run_timeout_s,
                compose_max_iters=compose_max_iters,
                verify=verify,
                dispatch_jobs=(dispatch_jobs if dispatch_jobs else "1"),
                allow_fallback=True,
            )
            rr = router.solve(problem_file)
            elapsed = time.time() - start_time

            # Build artifacts from router result
            if rr.route == "kernelagent":
                status = (
                    "✅ **Success (KernelAgent route)**"
                    if rr.success
                    else "❌ **KernelAgent route failed**"
                ) + f" — elapsed {elapsed:.2f}s"
                details_lines = [
                    "## Router Result",
                    f"- Route: `{rr.route}`",
                    f"- Success: `{rr.success}`",
                ]
                det = rr.details or {}
                if det:
                    for k in ["worker_id", "rounds", "session_dir", "message"]:
                        if k in det and det[k] is not None:
                            details_lines.append(f"- {k}: `{det[k]}`")
                details_md = "\n".join(details_lines)
                code_text = rr.kernel_code or ""
                run_info_md = "## 📁 Run Information\n- Route: KernelAgent"
                return PipelineArtifacts(
                    status_md=status,
                    details_md=details_md,
                    code_text=code_text,
                    run_info_md=run_info_md,
                    zip_path=None,
                )
            else:
                # Fuser path: details contains pipeline result dict
                res = rr.details or {}
                elapsed = time.time() - start_time
                run_dir = Path(res.get("run_dir", ".")).resolve()
                comp = res.get("composition", {}) or {}
                composed_path = Path(comp.get("composed_path", ""))
                code_text = _read_text_bounded(composed_path)

                passed = comp.get("verify_passed", False) if verify else True
                status = (
                    "✅ **Success (Fuser route)** PASS"
                    if passed
                    else "❌ **Fuser route completed with verification failure**"
                ) + f" — elapsed {elapsed:.2f}s"

                details_lines = [
                    "## Pipeline Outputs",
                    f"- Run dir: `{run_dir}`",
                    f"- Problem file: `{problem_file}`",
                    f"- Subgraphs: `{res.get('subgraphs', '')}`",
                    f"- Kernels summary: `{res.get('kernels_summary', '')}`",
                    f"- Composed: `{composed_path}`",
                ]
                if verify:
                    details_lines.append(
                        f"- Verify rc: {comp.get('verify_rc')} — validator: {comp.get('validator')}"
                    )
                    details_lines.append(f"- Stdout: `{comp.get('stdout_path', '')}`")
                    details_lines.append(f"- Stderr: `{comp.get('stderr_path', '')}`")
                details_md = "\n".join(details_lines)

                run_info_lines = ["## 📁 Run Information"]
                run_info_lines.append("- Route: Fuser")
                run_info_lines.append(f"- Run directory: `{run_dir}`")
                run_info_lines.append(f"- Compose rounds: {comp.get('rounds', 1)}")
                if verify:
                    run_info_lines.append(f"- Result: {'PASS' if passed else 'FAIL'}")
                run_info_md = "\n".join(run_info_lines)

                compose_out_dir = run_dir / "compose_out"
                zip_path = (
                    _zip_dir(compose_out_dir, run_dir / "compose_artifacts.zip")
                    if compose_out_dir.exists()
                    else None
                )

                return PipelineArtifacts(
                    status_md=status,
                    details_md=details_md,
                    code_text=code_text,
                    run_info_md=run_info_md,
                    zip_path=zip_path,
                )

        # Default: plain pipeline path
        res = run_pipeline(
            problem_path=problem_file,
            extract_model=extract_model,
            dispatch_model=dispatch_model,
            compose_model=compose_model,
            dispatch_jobs=dispatch_jobs,
            workers=workers,
            max_iters=max_iters,
            llm_timeout_s=llm_timeout_s,
            run_timeout_s=run_timeout_s,
            out_root=None,
            verify=verify,
            compose_max_iters=compose_max_iters,
        )
        elapsed = time.time() - start_time
        run_dir = Path(res.get("run_dir", ".")).resolve()
        comp = res.get("composition", {}) or {}
        composed_path = Path(comp.get("composed_path", ""))
        code_text = _read_text_bounded(composed_path)

        # Status and details
        passed = comp.get("verify_passed", False) if verify else True
        status = (
            "✅ **Success!** PASS"
            if passed
            else "❌ **Pipeline completed with verification failure**"
        ) + f" — elapsed {elapsed:.2f}s"

        details_lines = []
        details_lines.append("## Pipeline Outputs")
        details_lines.append(f"- Run dir: `{run_dir}`")
        details_lines.append(f"- Problem file: `{problem_file}`")
        details_lines.append(f"- Subgraphs: `{res.get('subgraphs', '')}`")
        details_lines.append(f"- Kernels summary: `{res.get('kernels_summary', '')}`")
        details_lines.append(f"- Composed: `{composed_path}`")
        if verify:
            details_lines.append(
                f"- Verify rc: {comp.get('verify_rc')} — validator: {comp.get('validator')}"
            )
            details_lines.append(f"- Stdout: `{comp.get('stdout_path', '')}`")
            details_lines.append(f"- Stderr: `{comp.get('stderr_path', '')}`")
        details_md = "\n".join(details_lines)

        # Run info summary
        run_info_lines = ["## 📁 Run Information"]
        run_info_lines.append(f"- Run directory: `{run_dir}`")
        run_info_lines.append(f"- Compose rounds: {comp.get('rounds', 1)}")
        if verify:
            run_info_lines.append(f"- Result: {'PASS' if passed else 'FAIL'}")
        run_info_md = "\n".join(run_info_lines)

        # Zip compose_out for download
        compose_out_dir = run_dir / "compose_out"
        zip_path = (
            _zip_dir(compose_out_dir, run_dir / "compose_artifacts.zip")
            if compose_out_dir.exists()
            else None
        )

        return PipelineArtifacts(
            status_md=status,
            details_md=details_md,
            code_text=code_text,
            run_info_md=run_info_md,
            zip_path=zip_path,
        )
    except Exception:
        elapsed = time.time() - start_time
        tb = traceback.format_exc()
        status = "❌ **Error during pipeline run**"
        details_md = f"""```
{tb}
```"""
        run_info_md = f"- Runtime: {elapsed:.2f}s"
        return PipelineArtifacts(
            status_md=status,
            details_md=details_md,
            code_text="",
            run_info_md=run_info_md,
            zip_path=None,
        )
    finally:
        if temp_key_set:
            if original_env_key is not None:
                os.environ["OPENAI_API_KEY"] = original_env_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)


class PipelineUI:
    def __init__(self) -> None:
        load_dotenv()
        candidate_roots = [
            Path.cwd() / "external" / "KernelBench" / "KernelBench",
            Path.cwd() / "KernelBench" / "KernelBench",
            Path.cwd().parent / "KernelBench" / "KernelBench",
        ]
        seen: set[str] = set()
        collected: list[tuple[str, str]] = []
        for base in candidate_roots:
            print(base, file=sys.stderr)
            for label, abspath in _list_kernelbench_problems(base):
                if abspath not in seen:
                    collected.append((label, abspath))
                    seen.add(abspath)
        self.problem_choices = collected

        control_flow_path = (
            Path(__file__).resolve().parent.parent.parent
            / "external"
            / "control_flow.py"
        )
        if control_flow_path.exists():
            self.problem_choices.append(
                ("external/control_flow.py", str(control_flow_path))
            )

    def run(
        self,
        selected_problem_label: str,
        problem_description: str,
        extract_model: str,
        dispatch_model: str,
        compose_model: str,
        dispatch_jobs: str,
        auto_route: bool,
        router_model: str,
        router_high_reasoning: bool,
        workers: int,
        max_iters: int,
        llm_timeout: int,
        run_timeout: int,
        compose_max_iters: int,
        verify: bool,
        user_api_key: Optional[str],
    ) -> Tuple[str, str, str, str, Optional[str]]:
        problem_mapping = {label: path for label, path in self.problem_choices}
        selected_path = problem_mapping.get(selected_problem_label, "")
        # Use description override if present; otherwise selected path
        problem_path = selected_path
        arts = run_pipeline_ui(
            problem_path=problem_path,
            problem_description=problem_description,
            extract_model=extract_model,
            dispatch_model=dispatch_model,
            compose_model=compose_model,
            dispatch_jobs=dispatch_jobs,
            workers=workers,
            max_iters=max_iters,
            llm_timeout_s=llm_timeout,
            run_timeout_s=run_timeout,
            compose_max_iters=compose_max_iters,
            verify=verify,
            auto_route=auto_route,
            router_model=router_model,
            router_high_reasoning=router_high_reasoning,
            user_api_key=user_api_key,
        )
        return (
            arts.status_md,
            arts.details_md,
            arts.code_text,
            arts.run_info_md,
            str(arts.zip_path) if arts.zip_path else None,
        )


def build_interface() -> gr.Blocks:
    ui = PipelineUI()
    default_problem = ui.problem_choices[0][0] if ui.problem_choices else ""
    default_problem_path = ui.problem_choices[0][1] if ui.problem_choices else ""

    with gr.Blocks(title="KernelFalcon — Pipeline UI", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
# 🦅 KernelFalcon — End-to-End Pipeline

Run the extract → dispatch → compose pipeline on KernelBench problems and download composed artifacts.
""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")

                api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key (optional)",
                    placeholder="sk-...",
                    type="password",
                    value="",
                )

                problem_dropdown = gr.Dropdown(
                    choices=[label for label, _ in ui.problem_choices],
                    label="📁 KernelBench Problem",
                    value=default_problem,
                    interactive=True,
                )

                problem_path_display = gr.Textbox(
                    label="Selected problem path",
                    value=default_problem_path,
                    interactive=False,
                )

                problem_mapping = {label: path for label, path in ui.problem_choices}

                def _label_to_path(label: str) -> str:
                    return problem_mapping.get(label, "")

                problem_dropdown.change(
                    fn=_label_to_path,
                    inputs=problem_dropdown,
                    outputs=problem_path_display,
                )

                problem_desc_input = gr.Textbox(
                    label="Override problem description (optional)",
                    placeholder="Paste KernelBench-style Python problem file or a natural language description to synthesize one.",
                    lines=12,
                )

                # Model selectors
                openai_extract_models = ["gpt-5", "o4-mini"]
                registry_models = sorted(list(MODEL_NAME_TO_CONFIG.keys())) or [
                    "gpt-5",
                    "o4-mini",
                ]
                with gr.Row():
                    extract_model_in = gr.Dropdown(
                        choices=openai_extract_models,
                        label="Extract model (OpenAI only)",
                        value="gpt-5",
                        interactive=True,
                    )
                    dispatch_model_in = gr.Dropdown(
                        choices=registry_models,
                        label="Dispatch model",
                        value="gpt-5"
                        if "gpt-5" in registry_models
                        else registry_models[0],
                        interactive=True,
                    )
                    compose_model_in = gr.Dropdown(
                        choices=registry_models,
                        label="Compose model",
                        value="gpt-5"
                        if "gpt-5" in registry_models
                        else registry_models[0],
                        interactive=True,
                    )
                # Auto-router controls
                with gr.Row():
                    auto_route_cb = gr.Checkbox(
                        label="🔀 Auto-route (KernelAgent vs Fuser)", value=True
                    )
                    router_model_in = gr.Dropdown(
                        choices=registry_models,
                        label="Router model",
                        value="gpt-5"
                        if "gpt-5" in registry_models
                        else registry_models[0],
                        interactive=True,
                    )
                    router_high_reasoning_cb = gr.Checkbox(
                        label="Router high reasoning", value=True
                    )

                # Dispatcher jobs (supports 'auto')
                dispatch_jobs_in = gr.Dropdown(
                    choices=["1", "2", "3", "4", "8", "auto"],
                    value="1",
                    label="Parallel dispatcher jobs",
                    interactive=True,
                    info="Use 'auto' to match subgraph count",
                )

                workers_slider = gr.Slider(
                    1,
                    8,
                    value=4,
                    step=1,
                    label="Workers (extract/dispatch; used for KA if auto-route)",
                )
                max_iters_slider = gr.Slider(
                    1, 20, value=5, step=1, label="Extract max iterations"
                )
                llm_timeout_slider = gr.Slider(
                    60, 7200, value=1800, step=60, label="LLM timeout (s)"
                )
                run_timeout_slider = gr.Slider(
                    60, 7200, value=1800, step=60, label="Run timeout (s)"
                )
                compose_iters_slider = gr.Slider(
                    1, 10, value=5, step=1, label="Compose max refinement rounds"
                )
                verify_checkbox = gr.Checkbox(
                    label="Verify composed kernel", value=True
                )

                run_button = gr.Button("🚀 Run Pipeline", variant="primary")

            with gr.Column(scale=1.5):
                gr.Markdown("## Results")
                status_out = gr.Markdown(value="*Awaiting run...*")
                details_out = gr.Markdown(value="")
                code_out = gr.Code(language="python", value="", lines=25)
                run_info_out = gr.Markdown(value="")
                download_out = gr.File(
                    label="Download composed artifacts", interactive=False
                )

        def on_run(
            selected_label: str,
            problem_desc: str,
            extract_model: str,
            dispatch_model: str,
            compose_model: str,
            dispatch_jobs: str,
            auto_route: bool,
            router_model: str,
            router_high_reasoning: bool,
            workers: int,
            max_iters: int,
            llm_timeout: int,
            run_timeout: int,
            compose_max_iters: int,
            verify: bool,
            api_key: Optional[str],
        ):
            return ui.run(
                selected_problem_label=selected_label,
                problem_description=problem_desc,
                extract_model=extract_model,
                dispatch_model=dispatch_model,
                compose_model=compose_model,
                dispatch_jobs=dispatch_jobs,
                auto_route=auto_route,
                router_model=router_model,
                router_high_reasoning=router_high_reasoning,
                workers=workers,
                max_iters=max_iters,
                llm_timeout=llm_timeout,
                run_timeout=run_timeout,
                compose_max_iters=compose_max_iters,
                verify=verify,
                user_api_key=api_key,
            )

        run_button.click(
            fn=on_run,
            inputs=[
                problem_dropdown,
                problem_desc_input,
                extract_model_in,
                dispatch_model_in,
                compose_model_in,
                dispatch_jobs_in,
                auto_route_cb,
                router_model_in,
                router_high_reasoning_cb,
                workers_slider,
                max_iters_slider,
                llm_timeout_slider,
                run_timeout_slider,
                compose_iters_slider,
                verify_checkbox,
                api_key_input,
            ],
            outputs=[status_out, details_out, code_out, run_info_out, download_out],
            show_progress=True,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline UI")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    load_dotenv()
    app = build_interface()

    print("🚀 Starting Pipeline UI...")

    # Mirror fuser_ui devserver behavior for Meta VPN environments
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
