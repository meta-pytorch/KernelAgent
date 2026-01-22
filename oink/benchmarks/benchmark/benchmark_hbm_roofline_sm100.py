"""
HBM roofline microbenchmark for SM100 (GB200 / Blackwell).

This script measures a STREAM-like bandwidth ceiling using a simple Triton kernel
that performs a large contiguous copy (read + write) and/or triad (read + read + write)
over a large buffer.

Why this exists:
- The benchmark harnesses for Oink ops report an "ours_tbps" derived from an IO model.
- For roofline discussions, comparing against a *measured* device bandwidth ceiling
  is often more meaningful than quoting a marketing/theoretical spec.

Example:
  CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_hbm_roofline_sm100.py --dtype bf16 --op copy --gb 2
  CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_hbm_roofline_sm100.py --dtype fp16 --op triad --gb 2
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
import triton
import triton.language as tl

# Reduce fragmentation pressure on busy GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from bench_utils import (  # noqa: E402
    collect_device_meta,
    do_bench_triton,
    parse_dtype,
    write_json,
)


@triton.jit
def _copy_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, x, mask=mask)


@triton.jit
def _triad_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, x + y, mask=mask)


def _bytes_moved(n_elements: int, elem_size: int, *, op: str) -> int:
    if op == "copy":
        return int(2 * n_elements * elem_size)  # read x + write y
    if op == "triad":
        return int(3 * n_elements * elem_size)  # read x + read y + write y
    raise ValueError(f"Unsupported op: {op}")


def bench_one(
    *,
    n_elements: int,
    dtype: torch.dtype,
    op: str,
    block: int,
    num_warps: int,
    warmup_ms: int,
    iters_ms: int,
) -> Tuple[float, float]:
    device = torch.device("cuda")
    x = torch.empty((n_elements,), device=device, dtype=dtype)
    y = torch.empty_like(x)
    # Avoid pathological compression-friendly patterns (e.g. all-zeros) that can
    # artificially inflate apparent bandwidth on some GPUs. Random-ish data is
    # a closer match to ML workloads.
    x.uniform_(-1, 1)
    y.uniform_(-1, 1)

    grid = (triton.cdiv(n_elements, block),)

    if op == "copy":

        def launch():
            _copy_kernel[grid](
                x,
                y,
                n_elements,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=4,
            )

    elif op == "triad":

        def launch():
            _triad_kernel[grid](
                x,
                y,
                n_elements,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=4,
            )

    else:
        raise ValueError(f"Unsupported op: {op}")

    # Force compilation out of the timed region.
    launch()
    torch.cuda.synchronize()

    ms = do_bench_triton(launch, warmup_ms=warmup_ms, rep_ms=iters_ms)
    moved = _bytes_moved(n_elements, x.element_size(), op=op)
    tbps = moved / (ms * 1e-3) / 1e12
    return ms, tbps


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["tbps"]))
    print("\nSummary (STREAM-like):")
    print(
        f"- best_tbps: {best['tbps']:.3f} TB/s  ({best['op']}, BLOCK={best['block']}, warps={best['num_warps']})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    p.add_argument("--op", type=str, default="copy", choices=["copy", "triad", "both"])
    p.add_argument(
        "--gb", type=float, default=2.0, help="Size per tensor in GB (default: 2)"
    )
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument(
        "--iters", type=int, default=100, help="rep_ms for do_bench (default: 100)"
    )
    p.add_argument(
        "--json", type=str, default=None, help="Write JSON results to this path"
    )
    p.add_argument(
        "--no-sweep",
        action="store_true",
        help="Disable tuning sweep; run a single config",
    )
    p.add_argument(
        "--block", type=int, default=2048, help="BLOCK size when --no-sweep is set"
    )
    p.add_argument(
        "--warps", type=int, default=8, help="num_warps when --no-sweep is set"
    )
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    cap = (int(props.major), int(props.minor))
    if cap != (10, 0):
        raise RuntimeError(f"Expected SM100 (10,0), got {cap} ({props.name})")

    elem_size = torch.tensor(0, dtype=dtype).element_size()
    bytes_per_tensor = int(args.gb * (1024**3))
    n_elements = max(1, bytes_per_tensor // elem_size)

    ops: List[str]
    if args.op == "both":
        ops = ["copy", "triad"]
    else:
        ops = [args.op]

    if args.no_sweep:
        sweep: List[Tuple[int, int]] = [(int(args.block), int(args.warps))]
    else:
        # A tiny hand-tuned sweep that keeps compile overhead reasonable.
        sweep = [
            (1024, 4),
            (1024, 8),
            (2048, 4),
            (2048, 8),
            (4096, 8),
        ]

    print(f"Running on {props.name} (SM{props.major}{props.minor})")
    print(f"- dtype: {args.dtype} (elem={elem_size}B)")
    print(
        f"- n_elements: {n_elements:,}  (~{(n_elements * elem_size) / (1024**3):.2f} GiB per tensor)"
    )
    print(f"- ops: {ops}")
    print(f"- sweep: {sweep}")

    meta = collect_device_meta(device)
    rows: List[Dict[str, Any]] = []
    for op in ops:
        for block, warps in sweep:
            ms, tbps = bench_one(
                n_elements=n_elements,
                dtype=dtype,
                op=op,
                block=block,
                num_warps=warps,
                warmup_ms=int(args.warmup_ms),
                iters_ms=int(args.iters),
            )
            rows.append(
                dict(
                    op=op,
                    dtype=str(args.dtype),
                    n_elements=int(n_elements),
                    elem_size_B=int(elem_size),
                    block=int(block),
                    num_warps=int(warps),
                    warmup_ms=int(args.warmup_ms),
                    rep_ms=int(args.iters),
                    ms=float(ms),
                    tbps=float(tbps),
                )
            )
            print(
                f"- {op:5s} BLOCK={block:4d} warps={warps}: {tbps:.3f} TB/s  ({ms:.4f} ms)"
            )

    _print_summary(rows)

    if args.json:
        # Write meta + detailed rows for reproducibility.
        extra = dict(
            bytes_model="copy:2*N*elem, triad:3*N*elem",
            bytes_per_tensor=int(bytes_per_tensor),
            gb_per_tensor=float(args.gb),
        )
        write_json(args.json, meta, rows, extra=extra)


if __name__ == "__main__":
    main()
