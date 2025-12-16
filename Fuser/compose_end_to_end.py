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
"""
Compose an end-to-end Triton kernel for the original KernelBench problem by
leveraging:
  1) Fuser's subgraphs JSON (decomposition + shapes)
  2) KernelAgent-generated Triton kernels for those subgraphs

We call an LLM to synthesize a final composed kernel that matches the original
problem semantics, returning one complete Python file that exposes
`kernel_function(...)` and includes a minimal self-test that checks numerical
equivalence against a PyTorch reference derived from the original problem.

Usage:
  python -m Fuser.compose_end_to_end \
      --problem /abs/path/to/kernelbench_problem.py \
      --subgraphs /abs/path/to/subgraphs.json \
      --kernels-summary /abs/path/to/kernels_out/summary.json \
      [--model gpt-5] [--out-dir ./compose_out] [--verify]

Notes:
- Requires an available LLM provider configured via KernelAgent providers
  (e.g., OPENAI_API_KEY for OpenAI models).
- Writes composed Python file to <out-dir>/composed_kernel.py and a
composition summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from triton_kernel_agent.platform_config import (
    get_platform,
    get_platform_choices,
    PlatformConfig,
)

# Reuse KernelAgent provider stack for LLM calls
try:
    from utils.providers.models import get_model_provider
except Exception:
    get_model_provider = None  # type: ignore

# Reuse extractor to capture clean python code blocks
from .code_extractor import extract_single_python_file

# Reuse Fuser runner for optional verification
from .runner import run_candidate


@dataclass
class KernelItem:
    subgraph_id: str
    kernel_path: Path
    code: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_kernels_from_summary(summary_path: Path) -> List[KernelItem]:
    data = json.loads(_read_text(summary_path))
    if not isinstance(data, list):
        raise SystemExit("kernels summary must be a JSON array (from dispatch step)")
    items: List[KernelItem] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        if not it.get("success"):
            # Skip failed generations
            continue
        sid = str(it.get("id", ""))
        kpath_str = it.get("kernel_path") or ""
        if not sid or not kpath_str:
            continue
        kpath = Path(kpath_str)
        if not kpath.is_file():
            continue
        code = _read_text(kpath)
        items.append(KernelItem(subgraph_id=sid, kernel_path=kpath, code=code))
    if not items:
        raise SystemExit("no successful kernels in summary.json")
    return items


def _summarize_subgraphs_for_prompt(subgraphs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for it in subgraphs:
        sid = str(it.get("id", "unknown"))
        typ = str(it.get("type", ""))
        layout = it.get("data_layout") or "NCHW"
        dtype = it.get("dtype") or "float32"
        inputs = it.get("inputs")
        in_shape = it.get("input_shape")
        out_shape = it.get("output_shape")
        ops = it.get("ops") or []
        # Compact rendering
        shapes_line = (
            f"inputs={inputs if inputs is not None else in_shape}, output={out_shape}"
        )
        lines.append(
            f"- ID={sid} type={typ} layout={layout} dtype={dtype} {shapes_line}"
        )
        try:
            ops_short = json.dumps(ops)[:400]
        except Exception:
            ops_short = str(ops)[:400]
        lines.append(f"  ops={ops_short}")
    return "\n".join(lines)


def _build_composition_prompt(
    problem_code: str,
    subgraphs: List[Dict[str, Any]],
    kernel_items: List[KernelItem],
    target_platform: PlatformConfig,
) -> str:
    """Create a single user message to instruct composition by the LLM."""
    # Provide a succinct summary of subgraphs up front
    sg_summary = _summarize_subgraphs_for_prompt(subgraphs)

    # Include only essential snippets from each kernel to keep token usage sane
    # We include full files for now; callers can trim by model limits.
    kernels_section_parts: List[str] = []
    for ki in kernel_items:
        kernels_section_parts.append(
            f"### Subgraph {ki.subgraph_id}\n```python\n" + ki.code + "\n```\n"
        )
    kernels_section = "\n".join(kernels_section_parts)
    # Platform-specific guidance
    platform_guidance = target_platform.guidance_block

    guidance = textwrap.dedent(
        f"""
        You are given:
        - The original problem file (PyTorch module and helpers).
        - A decomposition of the model into fusable subgraphs with exact shapes.
        - Working Triton kernels generated for some subgraphs.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}
        {platform_guidance}

        Task:
        - Compose an end-to-end Triton implementation that matches the original
          model's forward pass for the provided shapes. You may inline, adapt,
          or reuse the given subgraph kernels. Prefer fusing into as few kernel
          launches as possible while preserving exact numerical semantics.

        Hard requirements:
        - Return ONE complete Python file only, fenced as a single ```python block.
        - Allocate inputs, weights, intermediates, and outputs on device='{target_platform.device_string}' and keep them there throughout forward/verification.
        - CPU is acceptable only for metadata, scalars, and export serialization—avoid `.cpu()` or `.to('cpu')` on compute tensors.
        - Provide at least one @triton.jit kernel and a top-level Python wrapper
          named kernel_function(...). This wrapper must accept the same primary
          input tensor(s) as the model and any required weights/biases with shapes
          implied by the problem; it should orchestrate Triton kernel(s) and
          return the final output tensor.
        - No PyTorch math path: kernel_function MUST compute the final outputs
          using your Triton kernels only. Do NOT implement or fall back to
          torch.nn / torch.nn.functional / torch.* ops
          sigmoid, etc.) for producing the final result. Using PyTorch for
          reference comparisons is allowed only inside the self-test.
        - Use the data layout and dtype semantics indicated by subgraphs, defaulting
          to NCHW + float32 if unspecified. Respect stride/padding/dilation/groups,
          and exact op order.
        - Numerical equivalence: include a self-test (test_kernel or run_tests)
          that compares your Triton-based result to a PyTorch reference computed
          from the original problem code below (use get_init_inputs() and
          get_inputs() if present to instantiate the Model). The test must print
          'PASS' on success and exit with code 0. Use allclose with rtol<=1e-3,
          atol<=1e-3 for fp32; for fp16/bf16 allow up to 2e-2.
        - No imports beyond torch, triton, triton.language as tl, and stdlib. No I/O.
        - Do NOT monkey-patch PyTorch device functions or torch.cuda.is_available()
        - Do NOT manipulate TRITON_BACKENDS environment variable
        - Do NOT disable or mock XPU/CUDA drivers

        Implementation tips:
        - If merging multiple subgraphs, ensure intermediate tensor shapes match.
        - Hoist constant weights or parameters to avoid reloading per block.
        - Use tl.load/tl.store with masks for boundary conditions.
        - Favor coalesced memory access; tile by blocks; compute grid from shape.
        - Common Triton pitfalls to avoid:
          * Do NOT call tl.broadcast on Python scalars; tl.maximum(x, 0.0) works.
          * Prefer scalar constants directly in elementwise ops (no explicit broadcast needed).
          * Keep BLOCK_SIZE power-of-two; mask stores at tail.
        """
    ).strip()

    user_lines: List[str] = []
    user_lines.append(guidance)
    user_lines.append("")
    user_lines.append("SUBGRAPHS (summary):")
    user_lines.append(sg_summary)
    user_lines.append("")
    user_lines.append("ORIGINAL PROBLEM FILE:")
    user_lines.append("```python")
    user_lines.append(problem_code)
    user_lines.append("```")
    user_lines.append("")
    user_lines.append("SUBGRAPH KERNELS (reference implementations):")
    user_lines.append(kernels_section)
    user_lines.append("")
    user_lines.append(
        "Return only one fenced Python code block with your final composed implementation."
    )
    return "\n".join(user_lines)


def _build_refinement_prompt(
    problem_code: str,
    subgraphs: List[Dict[str, Any]],
    kernel_items: List[KernelItem],
    previous_code: str,
    error_info: Dict[str, str],
    target_platform: PlatformConfig,
) -> str:
    """Prompt the LLM to refine the previously produced code based on errors."""
    err_tail = error_info.get("stderr_tail", "")
    out_tail = error_info.get("stdout_tail", "")

    guidance = textwrap.dedent(
        f"""
        You previously produced a composed Triton implementation, but it failed
        to run/compile. Analyze the ERROR_CONTEXT below and re-emit the entire
        corrected single-file implementation as one ```python block.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}

        Requirements remain the same. Additionally:
        - Fix any Triton compilation/runtime errors. For scalar constants in
          elementwise ops (e.g., ReLU), do not use tl.broadcast. Use direct
          scalars like 0.0 in tl.maximum(x, 0.0).
        - Keep function name kernel_function(...) unchanged and retain the
          self-test that prints PASS on success and exits 0.
        - Do NOT reintroduce any PyTorch math path in kernel_function. The final
          outputs must be computed via your Triton kernels only (no fallback to
          torch.nn / torch.nn.functional ops).
        - Return the complete corrected file; do not send diffs.
        """
    ).strip()

    lines: List[str] = []
    lines.append(guidance)
    lines.append("")
    lines.append("ERROR_CONTEXT (stderr tail):\n```\n" + err_tail + "\n```")
    if out_tail.strip():
        lines.append("STDOUT tail:\n```\n" + out_tail + "\n```")
    lines.append("")
    lines.append("ORIGINAL PROBLEM FILE:\n```python\n" + problem_code + "\n```")
    lines.append("")
    lines.append("SUBGRAPHS (summary):\n" + _summarize_subgraphs_for_prompt(subgraphs))
    lines.append("")
    # Keep previous attempt for reference
    lines.append("PREVIOUS_ATTEMPT:\n```python\n" + previous_code + "\n```")
    lines.append("")
    lines.append(
        "Return only one fenced Python code block with the corrected implementation."
    )
    return "\n".join(lines)


def _auto_patch_common_triton_issues(
    code: str, target_platform: PlatformConfig
) -> Tuple[str, bool]:
    """Apply tiny safe textual patches for known Triton pitfalls.

    - Replace tl.broadcast(0.0, ...) or tl.broadcast(1.0, ...) with scalar constants.
    Returns (patched_code, changed).
    """
    patched = code
    changed = False
    # Simple heuristics; keep conservative
    patterns = [
        ("tl.broadcast(0.0", "0.0"),
        ("tl.broadcast(1.0", "1.0"),
        ("tl.broadcast(0,", "0.0"),
        ("tl.broadcast(1,", "1.0"),
    ]
    for old, new in patterns:
        if old in patched:
            patched = patched.replace(old, new)
            changed = True
    # Remove cuda paterns
    cuda_hacks = target_platform.cuda_hacks_to_strip
    if cuda_hacks:
        lines = patched.split("\n")
        filtered_lines = []
        skip_until_blank = False
        for line in lines:
            # Check if we're in a block to skip
            if skip_until_blank:
                if line.strip() == "":
                    skip_until_blank = False
                continue
            # Check if line contains any CUDA hack pattern
            if any(hack in line for hack in cuda_hacks):
                changed = True
                # Start skipping if this is a function definition we need to remove entirely
                if "def _fake_torch_device" in line:
                    skip_until_blank = True
                continue
            filtered_lines.append(line)
        patched = "\n".join(filtered_lines)
    return patched, changed


def compose(
    problem_path: Path,
    subgraphs_path: Path,
    kernels_summary_path: Path,
    out_dir: Path,
    model_name: str,
    verify: bool = False,
    max_iters: int = 5,
    target_platform: str = "cuda",
) -> Dict[str, Any]:
    if get_model_provider is None:
        raise SystemExit(
            "KernelAgent providers unavailable; ensure package import and dependencies"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    provider = get_model_provider(model_name)

    # Platform
    platform = get_platform(target_platform)

    # Load inputs
    problem_code = _read_text(problem_path)
    subgraphs = json.loads(_read_text(subgraphs_path))
    if not isinstance(subgraphs, list):
        raise SystemExit("subgraphs.json must be a JSON array")
    kernels = _load_kernels_from_summary(kernels_summary_path)

    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    last_usage = None
    last_code = None
    verify_info: Dict[str, Any] = {}

    for i in range(1, max_iters + 1):
        if i == 1 or last_code is None:
            prompt = _build_composition_prompt(
                problem_code, subgraphs, kernels, target_platform=platform
            )
        else:
            # Build refinement using previous error info
            stderr_tail = ""
            stdout_tail = ""
            try:
                if verify_info.get("stderr_path"):
                    with open(
                        verify_info["stderr_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stderr_tail = f.read()[-2000:]
                if verify_info.get("stdout_path"):
                    with open(
                        verify_info["stdout_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stdout_tail = f.read()[-2000:]
            except Exception:
                pass
            prompt = _build_refinement_prompt(
                problem_code,
                subgraphs,
                kernels,
                previous_code=last_code,
                error_info={"stderr_tail": stderr_tail, "stdout_tail": stdout_tail},
                target_platform=platform,
            )

        (attempts_dir / f"attempt_{i}.prompt.txt").write_text(prompt, encoding="utf-8")
        response = provider.get_response(
            model_name, [{"role": "user", "content": prompt}], max_tokens=50000
        )
        last_usage = response.usage
        raw_text = response.content or ""

        # Extract code
        extracted = extract_single_python_file(raw_text)
        code = extracted.code
        # Auto-patch trivial Triton pitfalls before running
        code, changed = _auto_patch_common_triton_issues(code, platform)
        (attempts_dir / f"attempt_{i}.py").write_text(code, encoding="utf-8")
        last_code = code

        # Verify each attempt if requested
        if verify:
            rr = run_candidate(
                artifacts_code_path=attempts_dir / f"attempt_{i}.py",
                run_root=out_dir / "runs",
                timeout_s=2400,
                isolated=False,
                deny_network=False,
            )
            verify_info = {
                "verify_rc": rr.rc,
                "verify_passed": rr.passed,
                "verify_reason": rr.reason,
                "validator": rr.validator_used,
                "stdout_path": str(rr.stdout_path),
                "stderr_path": str(rr.stderr_path),
            }
            if rr.passed:
                break
        else:
            # If not verifying, stop after first attempt
            break

    # Write final composed file as the last attempt
    composed_path = out_dir / "composed_kernel.py"
    composed_path.write_text(last_code or "", encoding="utf-8")

    result: Dict[str, Any] = {
        "success": bool(verify_info.get("verify_passed", not verify)),
        "composed_path": str(composed_path.resolve()),
        "model": model_name,
        "usage": last_usage,
        "rounds": i,
        "target_platform": target_platform,
    }
    result.update(verify_info)

    # Persist a small summary
    (out_dir / "composition_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Compose end-to-end Triton kernel from subgraphs + generated kernels"
    )
    p.add_argument(
        "--problem", required=True, help="Absolute path to KernelBench problem file"
    )
    p.add_argument(
        "--subgraphs", required=True, help="Path to subgraphs.json from Fuser"
    )
    p.add_argument(
        "--kernels-summary",
        required=True,
        help="Path to summary.json from dispatch step",
    )
    p.add_argument(
        "--out-dir",
        default="compose_out",
        help="Output directory for composed artifacts",
    )
    p.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL") or "gpt-5", help="LLM model name"
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Execute generated file and check PASS sentinel",
    )
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform (default: cuda)",
    )
    p.add_argument("--max-iters", type=int, default=5, help="Max LLM refinement rounds")
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    subgraphs_path = Path(args.subgraphs).resolve()
    kernels_summary_path = Path(args.kernels_summary).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not problem_path.is_file():
        print(f"problem file not found: {problem_path}")
        return 2
    if not subgraphs_path.is_file():
        print(f"subgraphs file not found: {subgraphs_path}")
        return 2
    if not kernels_summary_path.is_file():
        print(f"kernels summary not found: {kernels_summary_path}")
        return 2

    try:
        res = compose(
            problem_path=problem_path,
            subgraphs_path=subgraphs_path,
            kernels_summary_path=kernels_summary_path,
            out_dir=out_dir,
            model_name=args.model,
            verify=args.verify,
            max_iters=args.max_iters,
            target_platform=args.target_platform,
        )
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        print(f"compose failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
