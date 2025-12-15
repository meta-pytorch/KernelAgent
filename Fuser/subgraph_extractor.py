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
Extract unique fusable subgraphs and their exact shape signatures from a fused
model produced by Fuser. Produces a JSON array written to the run directory.

Flow:
1) Run Fuser orchestrator against a KernelBench problem to produce code.py.
2) Ask the LLM to analyze the fused code and emit a JSON array of subgraphs with
   exact input/output/weight shapes. Differences in shapes => different subgraph.
3) Parse the JSON from the LLM output, deduplicate by shape signature, and save
   to <run_dir>/subgraphs.json.

Usage:
  python -m Fuser.subgraph_extractor --problem /abs/path.py [--model gpt-5]
      [--workers 4] [--max-iters 5] [--llm-timeout-s 2400] [--run-timeout-s 2400]

This script loads .env in CWD (same behavior as Fuser CLI) for OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

from .cli import _load_dotenv_if_present  # reuse env loader
from .config import OrchestratorConfig, new_run_id
from .orchestrator import Orchestrator
from .paths import ensure_abs_regular_file, make_run_dirs, PathSafetyError
from .event_adapter import EventAdapter
from triton_kernel_agent.platform_config import get_platform_choices

from utils.providers import get_model_provider


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


_JSON_BLOCK_RE = re.compile(
    r"^```[ \t]*(json)?[ \t]*\n([\s\S]*?)^```[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_json_block(text: str) -> str:
    """Extract the last fenced JSON block or fallback to best-effort slice."""
    matches = list(_JSON_BLOCK_RE.finditer(text))
    chosen: Optional[re.Match[str]] = None
    for m in reversed(matches):
        lang = (m.group(1) or "").strip().lower()
        if lang == "json":
            chosen = m
            break
    if chosen is None and matches:
        # take the last fenced block even if not tagged json
        chosen = matches[-1]
    if chosen is not None:
        return chosen.group(2)
    # fallback: attempt to slice between first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def _dedup_by_shape_signature(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate items by a stable shape signature.

    The signature is based on sorted lists of input/weight/output shapes content,
    ignoring names but preserving dimensions and dtypes.
    """

    def norm_shapes(arr: Any) -> Any:
        if not isinstance(arr, list):
            return []
        normed: list[Any] = []
        for e in arr:
            if isinstance(e, dict):
                # normalize keys commonly used in schema
                shape = e.get("shape") or e.get("dims") or e.get("size")
                dtype = e.get("dtype")
                kind = e.get("kind") or e.get("role")
                # prefer int/str for dims; stringify others
                if isinstance(shape, list):
                    dims = [str(x) for x in shape]
                elif isinstance(shape, (int, str)):
                    dims = [str(shape)]
                else:
                    dims = [str(shape)] if shape is not None else []
                normed.append(
                    {"dims": dims, "dtype": str(dtype) if dtype else None, "k": kind}
                )
            else:
                normed.append(str(e))
        # sort for stable signature
        return sorted(normed, key=lambda x: json.dumps(x, sort_keys=True))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        sig_obj = {
            "inputs": norm_shapes(it.get("input_shapes")),
            "weights": norm_shapes(it.get("weight_shapes") or it.get("weights")),
            "outputs": norm_shapes(it.get("output_shapes")),
        }
        sig = json.dumps(sig_obj, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out


def _build_llm_prompt_for_shapes(fused_code: str, problem_code: str) -> Tuple[str, str]:
    system = "Return a single JSON array only."
    user_lines: list[str] = []
    user_lines.append(
        "You are given:\n- The original problem (PyTorch).\n- A fused refactor produced by Fuser (PyTorch subgraph modules)."
    )
    user_lines.append(
        "Task: Identify every unique subgraph by exact shape signature and emit a JSON array matching this schema (and only this schema):"
    )
    user_lines.append(
        "{\n"
        '  "id": <string>,\n'
        '  "type": <string>,\n'
        '  "data_layout": <\\"NCHW\\"|\\"NHWC\\"|null>,\n'
        '  "dtype": <string|null>,\n'
        '  "ops": [ {"op": <string>, ... op-specific fields ... } ],\n'
        '  "input_shape": [<int|sym>, ...]  // OR \\"inputs\\": [[...], [...]] for multi-input\n'
        '  "output_shape": [<int|sym>, ...],\n'
        '  "weights_fused": { <name>: [<int|sym>, ...], ... } | null,\n'
        '  "weights_original": { <name>: [<int|sym>, ...], ... } | null,\n'
        '  "count": <int>,\n'
        '  "where": <string>,\n'
        '  "source": { "module": <string>, "code": <string> }\n'
        "}"
    )
    user_lines.append("Notes:")
    user_lines.append(
        "- Treat any shape difference (inputs/outputs/weights) as a distinct subgraph. Count occurrences."
    )
    user_lines.append(
        "- Populate op-specific fields for conv/pool/linear, e.g., kernel_size/stride/padding/groups, bn_fused, output_size, start_dim."
    )
    user_lines.append(
        "- Include both weights_original (pre-fusion params like BN gamma/beta/running stats) and weights_fused (post-fusion conv/bias). Use null if not applicable."
    )
    user_lines.append(
        "- Provide a short \"where\" string (e.g., 'Model.forward stem' or 'layer2.block3.conv')."
    )
    user_lines.append(
        '- Provide "source" with the smallest contiguous code snippet implementing the subgraph.'
    )
    user_lines.append(
        "- Use data_layout and dtype when clear (default conv layout is NCHW)."
    )
    user_lines.append(
        '- For binary ops like residual add, use "inputs": [[...],[...]].'
    )
    user_lines.append(
        "- Prefer concrete integers from get_inputs() shapes in the problem; otherwise use symbols like B, H, W."
    )
    user_lines.append("")
    user_lines.append("PROBLEM_FILE:\n```python")
    user_lines.append(problem_code)
    user_lines.append("```")
    user_lines.append("")
    user_lines.append("FUSED_CODE:")
    user_lines.append("""```python""")
    user_lines.append(fused_code)
    user_lines.append("```")
    user_lines.append("")
    user_lines.append(
        "Now return only one fenced JSON block containing the array. No prose."
    )
    return system, "\n".join(user_lines)


def extract_subgraphs_to_json(
    problem_path: Path,
    model_name: str,
    workers: int,
    max_iters: int,
    llm_timeout_s: int,
    run_timeout_s: int,
    target_platform: str = "cuda",
) -> Tuple[Path, Path]:
    """Run Fuser to produce fused code, then use LLM to emit subgraphs JSON.

    Returns (run_dir, json_path).
    """
    # Run orchestrator
    cfg = OrchestratorConfig(
        problem_path=problem_path,
        model=model_name,
        workers=workers,
        max_iters=max_iters,
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
        stream_mode="winner",
        store_responses=False,
        isolated=False,
        deny_network=False,
        enable_reasoning_extras=True,
        target_platform=target_platform,
    )
    run_id = new_run_id()
    base_dir = Path.cwd() / ".fuse"
    base_dir.mkdir(exist_ok=True)
    dirs = make_run_dirs(base_dir, run_id)

    orch = Orchestrator(
        cfg,
        run_dir=dirs["run_dir"],
        workers_dir=dirs["workers"],
        orchestrator_dir=dirs["orchestrator"],
    )
    summary = orch.run()
    if summary.winner_worker_id is None or not summary.artifact_path:
        raise SystemExit(f"No passing fused code: {summary.reason}")

    fused_code = _load_code_from_tar(Path(summary.artifact_path))
    if not fused_code.strip():
        raise SystemExit("Winner artifact missing code.py or empty")
    problem_code = problem_path.read_text(encoding="utf-8")

    # Ask LLM for shapes JSON
    system, user = _build_llm_prompt_for_shapes(fused_code, problem_code)

    """
    Temporary MUX to support Relay while we migrate to OpenAI Responses API.

    Uses EventAdapter for OpenAI, otherwise Provider inferface
    """
    provider = get_model_provider(model_name)
    if provider.name != "openai":
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = provider.get_response(
            model_name,
            messages,
            max_tokens=16000,
            text={"format": {"type": "text"}},
        )
        output_text = result.content or ""
    else:
        jsonl_path = dirs["orchestrator"] / "subgraphs.stream.jsonl"
        adapter = EventAdapter(
            model=model_name,
            store_responses=False,
            timeout_s=llm_timeout_s,
            jsonl_path=jsonl_path,
        )
        result = adapter.stream(
            system_prompt=system,
            user_prompt=user,
            extras={"text": {"format": {"type": "text"}}},
        )
        output_text = result.get("output_text", "")

    raw_json = _extract_json_block(output_text)
    try:
        data = json.loads(raw_json)
    except Exception as e:
        # Write diagnostic and re-raise
        diag = dirs["orchestrator"] / "subgraphs.raw.txt"
        diag.write_text(output_text, encoding="utf-8")
        raise SystemExit(f"Failed to parse LLM JSON: {e}")

    if not isinstance(data, list):
        raise SystemExit("LLM output JSON is not a list")

    # Merge duplicates by signature and sum counts
    grouped: Dict[str, Dict[str, Any]] = {}

    def sig_of(it: Dict[str, Any]) -> str:
        # Build a robust signature from ops + shapes + weights
        ops = it.get("ops") or []
        # normalize ops by sorting keys of each op dict
        ops_norm = []
        for op in ops:
            if isinstance(op, dict):
                ops_norm.append(json.loads(json.dumps(op, sort_keys=True)))
            else:
                ops_norm.append(op)
        inputs_single = it.get("input_shape")
        inputs_multi = it.get("inputs")
        outputs = it.get("output_shape")
        weights = it.get("weights") or {}
        weights_fused = it.get("weights_fused") or {}
        weights_original = it.get("weights_original") or {}

        # sort weight dicts by key for stability
        def sort_w(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                return {k: obj[k] for k in sorted(obj.keys())}
            return {}

        weights_norm = sort_w(weights)
        wf_norm = sort_w(weights_fused)
        wo_norm = sort_w(weights_original)
        sig_obj = {
            "ops": ops_norm,
            "in": inputs_multi if inputs_multi is not None else inputs_single,
            "out": outputs,
            "w": weights_norm,
            "wf": wf_norm,
            "wo": wo_norm,
            "layout": it.get("data_layout"),
            "dtype": it.get("dtype"),
        }
        return json.dumps(sig_obj, sort_keys=True)

    for it in data:
        s = sig_of(it)
        if s not in grouped:
            # ensure id exists; fallback to hash fragment
            if not it.get("id"):
                it["id"] = f"sg_{hash(s) & 0xFFFFFFFF:08x}"
            # normalize count
            c = it.get("count")
            try:
                count_val = int(c) if c is not None else 1
            except Exception:
                count_val = 1
            it["count"] = count_val
            grouped[s] = it
        else:
            # sum counts
            try:
                grouped[s]["count"] += int(it.get("count") or 1)
            except Exception:
                grouped[s]["count"] += 1

    deduped = list(grouped.values())
    out_path = dirs["run_dir"] / "subgraphs.json"
    out_path.write_text(json.dumps(deduped, indent=2), encoding="utf-8")
    return dirs["run_dir"], out_path


def main(argv: Optional[list[str]] = None) -> int:
    _load_dotenv_if_present()
    p = argparse.ArgumentParser(
        description="Extract unique subgraphs with shapes (JSON)"
    )
    p.add_argument(
        "--problem", required=True, help="Absolute path to KernelBench problem file"
    )
    p.add_argument("--model", default="gpt-5", help="OpenAI model name (Responses API)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--llm-timeout-s", type=int, default=2400)
    p.add_argument("--run-timeout-s", type=int, default=2400)
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform",
    )
    args = p.parse_args(argv)

    try:
        problem_path = ensure_abs_regular_file(args.problem)
    except PathSafetyError as e:
        print(str(e), file=sys.stderr)
        return 2

    run_dir, json_path = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=args.model,
        workers=args.workers,
        max_iters=args.max_iters,
        llm_timeout_s=args.llm_timeout_s,
        run_timeout_s=args.run_timeout_s,
        target_platform=args.target_platform,
    )
    print(str(json_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
