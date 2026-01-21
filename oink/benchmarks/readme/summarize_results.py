from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _fmt_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isfinite(v):
            av = abs(v)
            # Use scientific notation for very small values so we don't render
            # meaningful error stats as "0.0000".
            if av != 0.0 and av < 1e-3:
                return f"{v:.2e}"
            return f"{v:.4f}"
        return str(v)
    return str(v)


def _md_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt_cell(r.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _pick_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "M",
        "N",
        "dtype",
        "weight_dtype",
        "mode",
        "eps",
        "store_rstd",
        "return_rstd",
        "return_mean",
        "ignore_index",
        "ours_ms",
        "ours_tbps",
        "ours_hbm_frac",
        "quack_ms",
        "quack_tbps",
        "speedup_vs_quack",
    ]
    present = set().union(*(r.keys() for r in rows)) if rows else set()
    cols = [c for c in preferred if c in present]
    # Fall back to a stable sorted view if we missed everything (shouldn't happen).
    return cols or sorted(present)


def _geomean(values: Iterable[float]) -> Optional[float]:
    logs: List[float] = []
    for v in values:
        if v <= 0 or not math.isfinite(v):
            continue
        logs.append(math.log(v))
    if not logs:
        return None
    return math.exp(sum(logs) / len(logs))


def _collect_error_prefixes(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Infer error-stat prefixes like `ours_err_dx` from row keys."""
    prefixes: set[str] = set()
    for r in rows:
        for k in r.keys():
            if not isinstance(k, str):
                continue
            if not k.endswith("_max_abs"):
                continue
            if "err_" not in k:
                continue
            prefixes.add(k[: -len("_max_abs")])
    return sorted(prefixes)


def _summarize_error_stats(rows: Sequence[Dict[str, Any]]) -> str:
    prefixes = _collect_error_prefixes(rows)
    if not prefixes:
        return ""

    out_rows: List[Dict[str, Any]] = []
    for pfx in prefixes:
        # Per-prefix worst-case across rows.
        max_abs_vals = [float(r[pfx + "_max_abs"]) for r in rows if (pfx + "_max_abs") in r]
        p99_abs_vals = [float(r[pfx + "_p99_abs"]) for r in rows if (pfx + "_p99_abs") in r]
        rel_l2_vals = [float(r[pfx + "_rel_l2"]) for r in rows if (pfx + "_rel_l2") in r]
        if not max_abs_vals and not p99_abs_vals and not rel_l2_vals:
            continue
        out_rows.append(
            {
                "metric": pfx,
                "max_abs (max over shapes)": max(max_abs_vals) if max_abs_vals else None,
                "p99_abs (max over shapes)": max(p99_abs_vals) if p99_abs_vals else None,
                "rel_l2 (max over shapes)": max(rel_l2_vals) if rel_l2_vals else None,
            }
        )

    if not out_rows:
        return ""

    cols = ["metric", "max_abs (max over shapes)", "p99_abs (max over shapes)", "rel_l2 (max over shapes)"]
    return "\n".join(["", "### Error Stats (vs PyTorch ref)", "", _md_table(out_rows, cols), ""])


def summarize_one(path: str) -> str:
    payload = _load_json(path)
    meta = payload.get("meta", {})
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []

    cols = _pick_columns(rows)
    parts: List[str] = []

    base = os.path.basename(path)
    parts.append(f"## `{base}`")
    if meta:
        device = meta.get("device")
        cap = meta.get("capability")
        torch_ver = meta.get("torch")
        cuda_ver = meta.get("cuda")
        git_sha = meta.get("git_sha")
        ts = meta.get("timestamp")
        parts.append("")
        parts.append(
            f"- device: `{device}` | capability: `{cap}` | torch: `{torch_ver}` | cuda: `{cuda_ver}` | git_sha: `{git_sha}` | timestamp: `{ts}`"
        )
        method = meta.get("method")
        if method is not None:
            parts.append(f"- method: `{method}`")
        if meta.get("warmup_ms") is not None and meta.get("rep_ms") is not None:
            parts.append(f"- warmup_ms: `{meta.get('warmup_ms')}` | rep_ms: `{meta.get('rep_ms')}`")

    if rows:
        parts.append("")
        parts.append(_md_table(rows, cols))

        speeds = [float(r["speedup_vs_quack"]) for r in rows if "speedup_vs_quack" in r]
        gm = _geomean(speeds)
        if gm is not None:
            parts.append("")
            parts.append(f"- geomean speedup vs Quack: `{gm:.3f}x` (over {len(speeds)} shapes)")

        err_block = _summarize_error_stats(rows)
        if err_block:
            parts.append(err_block.rstrip())
    else:
        parts.append("")
        parts.append("_No rows found in JSON._")

    parts.append("")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize KernelAgent-Oink benchmark JSONs into Markdown tables.")
    p.add_argument("--in-dir", type=str, required=True, help="Directory containing benchmark JSON files")
    p.add_argument("--out", type=str, default=None, help="Optional output markdown path (default: stdout)")
    args = p.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    if not os.path.isdir(in_dir):
        raise SystemExit(f"--in-dir is not a directory: {in_dir}")

    json_paths = sorted(
        os.path.join(in_dir, name) for name in os.listdir(in_dir) if name.endswith(".json")
    )
    if not json_paths:
        raise SystemExit(f"No .json files found under: {in_dir}")

    out_parts: List[str] = []
    out_parts.append("# KernelAgent-Oink SM100 Benchmark Summary")
    out_parts.append("")
    out_parts.append(f"Input directory: `{in_dir}`")
    out_parts.append("")
    for path in json_paths:
        out_parts.append(summarize_one(path))

    text = "\n".join(out_parts).rstrip() + "\n"
    if args.out is None:
        print(text, end="")
        return

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
