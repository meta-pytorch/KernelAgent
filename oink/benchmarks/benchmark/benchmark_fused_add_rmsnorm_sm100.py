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
Benchmark fused_add_rmsnorm (in-place) on SM100.

This matches vLLM's fused_add_rms_norm semantics:
  z = x + residual   (stored into residual)
  y = RMSNorm(z, w)  (stored into x)

Why this exists:
- It is a common inference hot path (vLLM).
- It is strongly memory-bound (reads/writes two MxN tensors), making it a good
  roofline case study for Blackwell.

Example:
  CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py --dtype bf16 --M 65536 --N 4096 \\
    --json /tmp/fused_add_rmsnorm_sm100_bf16.json

DSv3 suite (Oink vs Quack, multi-shape):
  CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py --dtype bf16 --dsv3 \\
    --json /tmp/kernelagent_oink_sm100_suite_bf16/fused_add_rmsnorm_dsv3.json

Quack baseline note:
- Oink exposes an **in-place** fused op (writes `x` and `residual` in-place).
- Quack provides an equivalent fused kernel, but typically returns `out` and
  `residual_out` (out-of-place) and does not expose a public "update my input
  buffers in-place" API.
- For integration realism (vLLM-style semantics) we default to timing:
    Quack fused kernel + 2 explicit copies to apply the in-place updates
  so the benchmark covers the full semantic cost.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch

# Reduce fragmentation pressure on busy GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Ensure SM100 (GB200) architecture is recognized by CuTeDSL when running outside vLLM.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

from bench_utils import (  # noqa: E402
    ErrorStatsAccumulator,
    collect_device_meta,
    detect_hbm_peak_gbps,
    do_bench_triton,
    error_stats_to_row,
    ensure_oink_src_on_path,
    iter_row_blocks,
    parse_dtype,
    write_json,
)

ensure_oink_src_on_path()

from kernelagent_oink.blackwell import rmsnorm as oink_rmsnorm  # noqa: E402

_VERIFY_TOL = {
    # Align with Quack's RMSNorm unit-test defaults (tests/test_rmsnorm.py).
    torch.float32: dict(atol=1e-4, rtol=1e-3),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-2),
}

try:
    # Use the low-level mutating custom op to avoid per-iteration allocations
    # (critical for fair comparisons on small/medium M).
    from quack.rmsnorm import _rmsnorm_fwd as quack_rmsnorm_fwd_mut  # type: ignore
except Exception:
    quack_rmsnorm_fwd_mut = None


def dsv3_configs() -> List[Tuple[int, int]]:
    Ms = [4096, 16384, 65536]
    Ns = [6144, 7168, 8192]
    return [(m, n) for m in Ms for n in Ns]


def bytes_io_model_fused_add_rmsnorm_inplace(M: int, N: int, dtype: torch.dtype) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    # Read x + read residual + write x + write residual + read weight
    return int((4 * M * N + N) * elem)


def _verify_parity(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> dict[str, object]:
    tol = _VERIFY_TOL[x.dtype]
    ref_block_rows = 4096
    M = int(x.shape[0])
    N = int(x.shape[1])

    y_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    z_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    y_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if quack_rmsnorm_fwd_mut is not None
        else None
    )
    z_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if quack_rmsnorm_fwd_mut is not None
        else None
    )

    x_o = x.clone()
    r_o = residual.clone()
    out_q = None
    res_out_q = None
    with torch.no_grad():
        oink_rmsnorm.fused_add_rmsnorm_inplace_(x_o, r_o, w, eps=eps)

        if quack_rmsnorm_fwd_mut is not None:
            out_q = torch.empty_like(x)
            res_out_q = torch.empty_like(residual)
            quack_rmsnorm_fwd_mut(
                x,
                w,
                out_q,
                None,  # bias
                None,  # rstd
                None,  # mean
                residual,
                res_out_q,
                eps,
                False,  # is_layernorm
            )

    # Pure-PyTorch reference (float32 accumulation), chunked over rows.
    M = int(x.shape[0])
    w_f32 = w.float()
    for start, end in iter_row_blocks(M, ref_block_rows):
        z = x[start:end] + residual[start:end]
        zf = z.float()
        rstd = torch.rsqrt(zf.square().mean(dim=-1, keepdim=True) + eps)
        y_ref = ((zf * rstd) * w_f32).to(x.dtype)

        torch.testing.assert_close(x_o[start:end], y_ref, **tol)
        torch.testing.assert_close(r_o[start:end], z, **tol)
        y_acc_ours.update(x_o[start:end], y_ref)
        z_acc_ours.update(r_o[start:end], z)
        if out_q is not None and res_out_q is not None:
            torch.testing.assert_close(out_q[start:end], y_ref, **tol)
            torch.testing.assert_close(res_out_q[start:end], z, **tol)
            assert y_acc_quack is not None and z_acc_quack is not None
            y_acc_quack.update(out_q[start:end], y_ref)
            z_acc_quack.update(res_out_q[start:end], z)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_y", y_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_residual_out", z_acc_ours.finalize()))
    if y_acc_quack is not None and z_acc_quack is not None:
        stats.update(error_stats_to_row("quack_err_y", y_acc_quack.finalize()))
        stats.update(
            error_stats_to_row("quack_err_residual_out", z_acc_quack.finalize())
        )
    return stats


def bench_one(
    *,
    M: int,
    N: int,
    dtype: torch.dtype,
    warmup_ms: int,
    iters_ms: int,
    verify: bool,
    quack_baseline: str,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    x = torch.randn((M, N), device=device, dtype=dtype)
    residual = torch.randn_like(x)
    w = torch.randn((N,), device=device, dtype=dtype)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x=x, residual=residual, w=w, eps=1e-6)

    bytes_io = bytes_io_model_fused_add_rmsnorm_inplace(M, N, dtype)

    def fn():
        oink_rmsnorm.fused_add_rmsnorm_inplace_(x, residual, w, eps=1e-6)

    ms = do_bench_triton(fn, warmup_ms=warmup_ms, rep_ms=iters_ms)

    gbps = bytes_io / (ms * 1e-3) / 1e9
    tbps = gbps / 1000.0
    hbm_frac = gbps / detect_hbm_peak_gbps(device)

    row: Dict[str, Any] = dict(
        M=int(M),
        N=int(N),
        dtype="bf16"
        if dtype is torch.bfloat16
        else ("fp16" if dtype is torch.float16 else "fp32"),
        ours_ms=float(ms),
        ours_gbps=float(gbps),
        ours_tbps=float(tbps),
        ours_hbm_frac=float(hbm_frac),
    )
    row.update(stats)

    if quack_rmsnorm_fwd_mut is not None:
        x_q = x.clone()
        residual_q = residual.clone()
        out_q = torch.empty_like(x_q)
        res_out_q = torch.empty_like(residual_q)

        def fn_q_kernel():
            quack_rmsnorm_fwd_mut(
                x_q,
                w,
                out_q,
                None,  # bias
                None,  # rstd
                None,  # mean
                residual_q,
                res_out_q,
                1e-6,
                False,  # is_layernorm
            )

        if quack_baseline == "kernel":
            fn_q = fn_q_kernel
        elif quack_baseline == "kernel_inplace":

            def fn_q():
                fn_q_kernel()
                # Apply the same in-place semantics as vLLM expects:
                # - x is overwritten with y
                # - residual is overwritten with z = x + residual
                x_q.copy_(out_q)
                residual_q.copy_(res_out_q)

        else:
            raise ValueError(f"Unknown quack_baseline: {quack_baseline}")

        ms_q = do_bench_triton(fn_q, warmup_ms=warmup_ms, rep_ms=iters_ms)
        gbps_q = bytes_io / (ms_q * 1e-3) / 1e9
        row.update(
            dict(
                quack_ms=float(ms_q),
                quack_gbps=float(gbps_q),
                quack_tbps=float(gbps_q / 1000.0),
                speedup_vs_quack=float(ms_q / ms),
            )
        )

    return row


def _dtype_label(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return "fp32"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = ["M", "N", "ours_ms", "ours_tbps"]
    has_quack = any("quack_ms" in r for r in rows)
    if has_quack:
        headers += ["quack_ms", "quack_tbps", "speedup_vs_quack"]
    print("\nSummary:")
    print(" ".join(h.rjust(14) for h in headers))
    for r in rows:
        parts: List[str] = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                parts.append(f"{v:14.4f}")
            else:
                parts.append(f"{str(v):>14}")
        print(" ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    p.add_argument("--M", type=int, default=65536)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument(
        "--dsv3",
        action="store_true",
        help="Run DSv3 set: M in {4096,16384,65536}, N in {6144,7168,8192}",
    )
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument(
        "--iters", type=int, default=200, help="rep_ms for do_bench (default: 200)"
    )
    p.add_argument(
        "--quack-baseline",
        type=str,
        default="kernel_inplace",
        choices=["kernel", "kernel_inplace"],
        help=(
            "How to time Quack for the in-place fused op.\n"
            "- kernel: Quack fused kernel only (preallocated out/residual_out).\n"
            "- kernel_inplace: Quack fused kernel + 2 explicit copies to apply "
            "in-place semantics (integration-realistic)."
        ),
    )
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    meta = collect_device_meta(torch.device("cuda"))

    cfgs = dsv3_configs() if bool(args.dsv3) else [(int(args.M), int(args.N))]
    rows: List[Dict[str, Any]] = []
    for M, N in cfgs:
        print(
            f"bench M={M:<8d} N={N:<6d} dtype={_dtype_label(dtype)} fused_add_rmsnorm ...",
            flush=True,
        )
        rows.append(
            bench_one(
                M=int(M),
                N=int(N),
                dtype=dtype,
                warmup_ms=int(args.warmup_ms),
                iters_ms=int(args.iters),
                verify=not bool(args.skip_verify),
                quack_baseline=str(args.quack_baseline),
            )
        )

    _print_table(rows)

    if args.json:
        write_json(
            args.json,
            meta,
            rows,
            extra=dict(
                io_model_bytes="(4*M*N + N)*elem_size",
                warmup_ms=int(args.warmup_ms),
                rep_ms=int(args.iters),
                method="triton.testing.do_bench(mean)",
                note=(
                    "Oink fused_add_rmsnorm_inplace_ vs Quack baseline "
                    f"({args.quack_baseline}) when available"
                ),
            ),
        )


if __name__ == "__main__":
    main()
