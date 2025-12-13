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
Dispatch subgraphs (from subgraphs.json) to KernelAgent to generate Triton kernels.

For each JSON item (a unique, shape-specific subgraph), we synthesize a clear
problem description that enumerates ops, exact shapes, and a PyTorch reference
function. We then invoke TritonKernelAgent to generate and verify a kernel.

Outputs:
- A per-subgraph directory containing the agent session artifacts
- A summary JSON mapping subgraph ids to generation results

Usage:
  python -m Fuser.dispatch_kernel_agent --subgraphs /abs/path/to/subgraphs.json \
      [--agent-model gpt-5] [--out-dir ./kernels_out] [--jobs 1]

Requirements:
- OPENAI_API_KEY (.env in CWD or environment)
- jinja2 installed (used by KernelAgent prompt manager)
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple
import concurrent.futures as _futures

from dotenv import load_dotenv

try:
    from triton_kernel_agent import TritonKernelAgent
except Exception:  # pragma: no cover - import-time dependency
    TritonKernelAgent = None  # type: ignore

from triton_kernel_agent.platform_config import (
    get_platform,
    get_platform_choices,
    PlatformConfig,
)


def _shape_list(shape: Any) -> List[str]:
    if isinstance(shape, list):
        return [str(x) for x in shape]
    return [str(shape)] if shape is not None else []


def _fmt_shape(shape: Any) -> str:
    return "[" + ", ".join(_shape_list(shape)) + "]"


def _py_tuple(arr: Any) -> str:
    vals = _shape_list(arr)
    if not vals:
        return "()"
    if len(vals) == 1:
        return f"({vals[0]},)"
    return f"({', '.join(vals)})"


def _pick_weights(item: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    # Prefer explicit fused/original dicts; fallback to generic 'weights'
    ws = (
        item.get("weights_fused")
        or item.get("weights_original")
        or item.get("weights")
        or {}
    )
    if not isinstance(ws, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in keys:
        if k in ws:
            out[k] = ws[k]
    # include any remaining for completeness
    for k, v in ws.items():
        out.setdefault(k, v)
    return out


def _build_reference_code(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (reference_code_str, param_names) implementing the subgraph.

    param_names are additional parameters to reference() beyond the first input(s).
    """
    ops: List[Dict[str, Any]] = [
        op for op in (item.get("ops") or []) if isinstance(op, dict)
    ]
    lines: List[str] = ["import torch", "import torch.nn.functional as F", ""]
    params: List[str] = []

    # Determine if multi-input
    inputs_multi = item.get("inputs")
    input_names: List[str]
    if isinstance(inputs_multi, list) and inputs_multi:
        input_names = [f"x{i}" for i in range(len(inputs_multi))]
        header = f"def reference({', '.join(input_names)}"  # weights appended later
    else:
        input_names = ["x"]
        header = "def reference(x"

    body: List[str] = []
    cur = input_names[0] if input_names else "x"

    for op in ops:
        kind = str(op.get("op"))
        if kind == "conv2d":
            wmap = _pick_weights(item, ["conv_weight", "weight", "bias"])
            w = (
                "conv_weight"
                if "conv_weight" in wmap
                else ("weight" if "weight" in wmap else "conv_weight")
            )
            b = "bias" if "bias" in wmap else None
            args: List[str] = [cur, w]
            if b:
                args.append(b)
            stride = _py_tuple(op.get("stride", (1, 1)))
            padding = _py_tuple(op.get("padding", (0, 0)))
            dilation = _py_tuple(op.get("dilation", (1, 1)))
            groups = str(op.get("groups", 1))
            body.append(
                f"{cur} = F.conv2d({', '.join(args)}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"
            )
            params.extend([p for p in [w, b] if p])
        elif kind == "conv_transpose2d":
            wmap = _pick_weights(item, ["conv_transpose.weight", "weight", "bias"])
            w = (
                "conv_transpose_weight"
                if "conv_transpose.weight" in wmap
                else ("weight" if "weight" in wmap else "conv_transpose_weight")
            )
            b = "bias" if "bias" in wmap else None
            args: List[str] = [cur, w]
            if b:
                args.append(b)
            stride = _py_tuple(op.get("stride", (1, 1)))
            padding = _py_tuple(op.get("padding", (0, 0)))
            dilation = _py_tuple(op.get("dilation", (1, 1)))
            outpad = _py_tuple(op.get("output_padding", (0, 0)))
            groups = str(op.get("groups", 1))
            body.append(
                f"{cur} = F.conv_transpose2d({', '.join(args)}, stride={stride}, padding={padding}, dilation={dilation}, output_padding={outpad}, groups={groups})"
            )
            params.extend([p for p in [w, b] if p])
        elif kind in ("relu", "tanh", "sigmoid"):  # common activations
            body.append(f"{cur} = torch.{kind}({cur})")
        elif kind == "max_pool2d":
            k = _py_tuple(op.get("kernel_size", (2, 2)))
            s = _py_tuple(op.get("stride", (2, 2)))
            p = _py_tuple(op.get("padding", (0, 0)))
            d = _py_tuple(op.get("dilation", (1, 1)))
            ceil = bool(op.get("ceil_mode", False))
            body.append(
                f"{cur} = F.max_pool2d({cur}, kernel_size={k}, stride={s}, padding={p}, dilation={d}, ceil_mode={str(ceil).lower()})"
            )
        elif kind == "avg_pool2d":
            k = _py_tuple(op.get("kernel_size", (2, 2)))
            s = _py_tuple(op.get("stride", (2, 2)))
            p = _py_tuple(op.get("padding", (0, 0)))
            body.append(
                f"{cur} = F.avg_pool2d({cur}, kernel_size={k}, stride={s}, padding={p})"
            )
        elif kind == "batch_norm":
            # BN parameters
            wmap = _pick_weights(
                item,
                [
                    "batch_norm.weight",
                    "batch_norm.bias",
                    "batch_norm.running_mean",
                    "batch_norm.running_var",
                    "weight",
                    "bias",
                    "running_mean",
                    "running_var",
                ],
            )
            w = (
                "bn_weight"
                if ("batch_norm.weight" in wmap or "weight" in wmap)
                else "bn_weight"
            )
            b = (
                "bn_bias"
                if ("batch_norm.bias" in wmap or "bias" in wmap)
                else "bn_bias"
            )
            rm = "bn_running_mean"
            rv = "bn_running_var"
            eps = op.get("eps", 1e-5)
            momentum = op.get("momentum", 0.1)
            body.append(
                f"{cur} = F.batch_norm({cur}, {rm}, {rv}, {w}, {b}, training=False, momentum={momentum}, eps={eps})"
            )
            params.extend([w, b, rm, rv])
        elif kind == "group_norm":
            wmap = _pick_weights(
                item, ["group_norm.weight", "group_norm.bias", "weight", "bias"]
            )
            w = (
                "gn_weight"
                if ("group_norm.weight" in wmap or "weight" in wmap)
                else "gn_weight"
            )
            b = (
                "gn_bias"
                if ("group_norm.bias" in wmap or "bias" in wmap)
                else "gn_bias"
            )
            num_groups = int(op.get("num_groups", 1))
            eps = op.get("eps", 1e-5)
            body.append(
                f"{cur} = F.group_norm({cur}, {num_groups}, {w}, {b}, eps={eps})"
            )
            params.extend([w, b])
        elif kind in ("add", "sum"):
            # binary elementwise add (assume x0 + x1)
            if len(input_names) >= 2:
                body.append(f"{cur} = {input_names[0]} + {input_names[1]}")
            else:
                body.append(f"{cur} = {cur} + {cur}")  # fallback
        elif kind == "gemm":
            wmap = _pick_weights(item, ["weight", "bias"])  # linear: y = x @ W.T + b
            w = "linear_weight"
            b = "linear_bias"
            body.append(f"{cur} = torch.nn.functional.linear({cur}, {w}, {b})")
            params.extend([w, b])
        else:
            body.append(
                f"# TODO: op '{kind}' not explicitly handled; update generator if needed"
            )

    header += "):\n"
    lines.append(header)
    if not body:
        body = ["return x"]
    indented = ["    " + ln for ln in body]
    indented.append("    return " + cur)
    lines.extend(indented)
    return "\n".join(lines) + "\n", params


def _synthesize_problem_description(
    item: Dict[str, Any], target_platform: PlatformConfig
) -> str:
    id_ = str(item.get("id", "unknown"))
    type_ = str(item.get("type", ""))
    layout = item.get("data_layout") or "NCHW"
    dtype = item.get("dtype") or "float32"
    input_shape = item.get("input_shape")
    output_shape = item.get("output_shape")
    inputs_multi = item.get("inputs")
    weights_fused = item.get("weights_fused")
    weights_orig = item.get("weights_original")
    source = item.get("source") or {}

    ref_code, _ = _build_reference_code(item)

    # Get device string for the platform
    header = textwrap.dedent(
        f"""
        Implement a Triton kernel that computes the following subgraph end-to-end.

        Subgraph ID: {id_}
        Type: {type_}
        Data layout: {layout}
        DType: {dtype}
        Target Platform: {target_platform.name}
        Device String: {target_platform.device_string}

        Shapes:
        - input: {_fmt_shape(inputs_multi[0]) if isinstance(inputs_multi, list) else _fmt_shape(input_shape)}
        {("- input2: " + _fmt_shape(inputs_multi[1])) if isinstance(inputs_multi, list) and len(inputs_multi) > 1 else ""}
        - output: {_fmt_shape(output_shape)}

        Weights (fused): {json.dumps(weights_fused, indent=2) if isinstance(weights_fused, dict) else "null"}
        Weights (original): {json.dumps(weights_orig, indent=2) if isinstance(weights_orig, dict) else "null"}

        Operations in order (with parameters):
        {json.dumps(item.get("ops", []), indent=2)}

        Requirements:
        - Return a complete Python file with a @triton.jit kernel and a wrapper function named kernel_function(...).
        - kernel_function must accept input tensor(s) and any required weights/bias parameters (match shapes above).
        - Implement the exact semantics of the listed ops in the given order for the provided shapes.
        - Use {layout} layout and {dtype} dtype semantics.
        - Allocate inputs, weights, intermediates, and outputs on device='{target_platform.device_string}' and keep them there throughout forward/verification.
        - CPU is acceptable only for metadata, scalars, and export serialization—avoid `.cpu()` or `.to('cpu')` on compute tensors.
        - The test will import kernel_function and compare to the reference implementation below.

        Test tolerance policy (enforced in generated tests):
        - Default tolerances: rtol=1e-3, atol=1e-3.
        - Absolute cap: NEVER exceed rtol=1e-2 or atol=1e-2 in torch.allclose.
        - For float16/bfloat16 inputs: use rtol=1e-2, atol=1e-2 at most (do not go higher).
        - Include a one-line comment if you relax from default; never exceed the cap.

        Reference PyTorch implementation (exact semantics to match):
        """
    ).strip()

    src_code_block = ""  # optional original snippet for context
    if isinstance(source, dict) and source.get("code"):
        mod = source.get("module", "Model")
        code = str(source.get("code"))
        src_code_block = f"\nOriginal source snippet ({mod}):\n```python\n{code}\n```\n"

    problem = header + "\n\n```python\n" + ref_code + "```\n" + src_code_block
    return problem


def run(
    subgraphs_path: Path,
    out_dir: Path,
    agent_model: str | None = None,
    jobs: int = 1,
    target_platform: str = "cuda",
) -> Path:
    """Dispatch subgraphs to KernelAgent with optional parallelism.

    jobs controls the number of concurrent subgraph generations. Default=1
    preserves previous behavior and avoids GPU/LLM contention.
    """
    if TritonKernelAgent is None:
        raise SystemExit(
            "TritonKernelAgent not available. Ensure the package is importable."
        )

    with subgraphs_path.open("r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    if not isinstance(items, list):
        raise SystemExit("subgraphs.json must be a JSON array")

    out_dir.mkdir(parents=True, exist_ok=True)

    platform = get_platform(target_platform)

    # Worker function: create a dedicated agent instance per subgraph to avoid
    # cross-thread state interactions inside the agent/manager.
    def _handle_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, item = idx_item
        sid = str(item.get("id", f"subgraph_{idx}"))
        pdesc = _synthesize_problem_description(item, target_platform=platform)
        sg_dir = out_dir / sid
        sg_dir.mkdir(parents=True, exist_ok=True)
        (sg_dir / "problem.txt").write_text(pdesc, encoding="utf-8")

        # Pin KernelAgent concurrency defaults: 4 workers, 10 rounds
        local_agent = TritonKernelAgent(
            num_workers=4,
            max_rounds=10,
            model_name=agent_model,
            target_platform=platform,
        )
        try:
            result = local_agent.generate_kernel(
                problem_description=pdesc, test_code=None
            )
        except Exception as exc:
            try:
                local_agent.cleanup()
            except Exception:
                pass
            return idx, {"id": sid, "success": False, "error": str(exc)}

        try:
            local_agent.cleanup()
        except Exception:
            pass

        if result.get("success"):
            kernel_code = result.get("kernel_code", "")
            (sg_dir / "kernel.py").write_text(kernel_code, encoding="utf-8")
            return idx, {
                "id": sid,
                "success": True,
                "worker_id": result.get("worker_id"),
                "rounds": result.get("rounds"),
                "session_dir": result.get("session_dir"),
                "kernel_path": str((sg_dir / "kernel.py").resolve()),
            }
        else:
            return idx, {
                "id": sid,
                "success": False,
                "message": result.get("message"),
                "session_dir": result.get("session_dir"),
            }

    # Submit tasks with bounded concurrency
    jobs = max(1, int(jobs or 1))
    ordered_inputs: List[Tuple[int, Dict[str, Any]]] = list(enumerate(items, start=1))
    results: Dict[int, Dict[str, Any]] = {}
    if jobs == 1:
        for pair in ordered_inputs:
            i, res = _handle_one(pair)
            results[i] = res
    else:
        with _futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            future_map = {
                ex.submit(_handle_one, pair): pair[0] for pair in ordered_inputs
            }
            for fut in _futures.as_completed(future_map):
                i, res = fut.result()
                results[i] = res

    # Preserve input order in summary output
    summary: List[Dict[str, Any]] = [results[i] for i in sorted(results.keys())]
    out_summary = out_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_summary


def main(argv: List[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Generate Triton kernels for subgraphs via KernelAgent"
    )
    p.add_argument(
        "--subgraphs", required=True, help="Path to subgraphs.json produced by Fuser"
    )
    p.add_argument(
        "--out-dir",
        default="kernels_out",
        help="Output directory for per-subgraph artifacts",
    )
    p.add_argument(
        "--agent-model", default=None, help="Override KernelAgent model name (optional)"
    )
    p.add_argument(
        "--jobs",
        type=str,
        default="2",
        help="Max concurrent subgraphs to dispatch (default: 2); use 'auto' to match subgraph count",
    )
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform (default: cuda)",
    )
    args = p.parse_args(argv)

    subgraphs_path = Path(args.subgraphs).resolve()
    if not subgraphs_path.is_file():
        print(f"subgraphs file not found: {subgraphs_path}", file=os.sys.stderr)
        return 2
    out_dir = Path(args.out_dir).resolve()

    # Resolve jobs (support "auto")
    try:
        if isinstance(args.jobs, str) and args.jobs.strip().lower() == "auto":
            try:
                with subgraphs_path.open("r", encoding="utf-8") as f:
                    _items = json.load(f)
                jobs_val = max(1, int(len(_items))) if isinstance(_items, list) else 1
            except Exception:
                jobs_val = 1
        else:
            jobs_val = max(1, int(args.jobs))
    except Exception:
        jobs_val = 1

    summary_path = run(
        subgraphs_path,
        out_dir,
        agent_model=args.agent_model,
        jobs=jobs_val,
        target_platform=args.target_platform,
    )
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
