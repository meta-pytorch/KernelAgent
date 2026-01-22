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

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from triton.testing import do_bench as triton_do_bench

# Reduce fragmentation pressure on busy GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Ensure SM100 (GB200) architecture is recognized by CuTeDSL when running outside vLLM.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

# Make the in-repo KernelAgent Oink package importable without an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_OINK_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _OINK_SRC not in sys.path:
    sys.path.insert(0, _OINK_SRC)

from bench_utils import (  # noqa: E402
    ErrorStatsAccumulator,
    collect_device_meta,
    error_stats_to_row,
    iter_row_blocks,
    write_json,
)
from kernelagent_oink.blackwell import rmsnorm as oink_rmsnorm  # noqa: E402

try:
    from quack.rmsnorm import rmsnorm_bwd as quack_rmsnorm_bwd  # type: ignore
except Exception:
    quack_rmsnorm_bwd = None

_VERIFY_TOL_DX = {
    # Match Quack's unit-test defaults (tests/test_rmsnorm.py).
    torch.float32: dict(atol=1e-4, rtol=1e-3),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-2),
}


def detect_hbm_peak_gbps(device: Optional[torch.device] = None) -> float:
    """Approximate HBM peak bandwidth in GB/s for roofline fractions."""
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= 100:
        return 8000.0
    return 2000.0


@dataclass
class Result:
    ms: float
    gbps: float


def do_bench_triton(fn, warmup_ms: int = 25, rep_ms: int = 100) -> float:
    # Kernel-only timing consistent with the existing Oink forward harness.
    return float(triton_do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"))


def bytes_io_model_bwd(
    M: int, N: int, dtype: torch.dtype, *, weight_dtype: torch.dtype = torch.float32
) -> int:
    """A simple IO model for RMSNorm backward.

    This intentionally ignores partial-reduction scratch buffers (`dw_partial` /
    `db_partial`) since those are highly implementation-specific and depend on
    sm_count; we still report speedups and times regardless.
    """
    elem = torch.tensor(0, dtype=dtype).element_size()
    w_elem = torch.tensor(0, dtype=weight_dtype).element_size()
    # Read x + dout + write dx
    total = 3 * M * N * elem
    # Read weight + write dw
    total += 2 * N * w_elem
    # Read rstd (fp32 per row)
    total += M * 4
    return int(total)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def parse_configs(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in s.split(","):
        m, n = part.lower().split("x")
        out.append((int(m), int(n)))
    return out


def quack_suite_configs() -> List[Tuple[int, int, int]]:
    """Return (batch, seq, hidden) triples following Quack's grid (hidden=4096)."""
    batch_sizes = [1, 4, 8, 16, 32]
    seq_lengths = [8192, 16384, 32768, 65536, 131072]
    hidden = 4096
    cfgs: List[Tuple[int, int, int]] = []
    for bs in batch_sizes:
        for sl in seq_lengths:
            M = bs * sl
            if M * hidden > (2**31):
                continue
            cfgs.append((bs, sl, hidden))
    return cfgs


def dsv3_configs() -> List[Tuple[int, int]]:
    Ms = [4096, 16384, 65536]
    Ns = [6144, 7168, 8192]
    return [(m, n) for m in Ms for n in Ns]


def _verify_parity(
    x: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
    *,
    has_bias: bool,
    has_residual: bool,
) -> dict[str, object]:
    tol_dx = _VERIFY_TOL_DX[x.dtype]
    ref_block_rows = 1024
    M, N = int(x.shape[0]), int(x.shape[1])

    dx_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    dx_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if quack_rmsnorm_bwd is not None
        else None
    )

    with torch.no_grad():
        dx_oink, dw_oink, db_oink, dres_oink = oink_rmsnorm.rmsnorm_backward(
            x,
            w,
            dout,
            rstd,
            dresidual_out=None,
            has_bias=has_bias,
            has_residual=has_residual,
        )

        dx_quack = None
        dw_quack = None
        db_quack = None
        dres_quack = None
        if quack_rmsnorm_bwd is not None:
            dx_quack, dw_quack, db_quack, dres_quack = quack_rmsnorm_bwd(
                x,
                w,
                dout,
                rstd,
                dresidual_out=None,
                has_bias=has_bias,
                has_residual=has_residual,
            )
    torch.cuda.synchronize()

    # Pure-PyTorch reference, matching Quack's rmsnorm_bwd_ref (float32 math for x_hat).
    # Chunk over rows to avoid materializing an (M, N) float32 tensor for large shapes.
    dw_accum = torch.zeros((N,), device=x.device, dtype=torch.float32)
    w_f32 = w.float()
    for start, end in iter_row_blocks(M, ref_block_rows):
        x_f32 = x[start:end].float()
        rstd_blk = rstd[start:end]
        x_hat = x_f32 * rstd_blk.unsqueeze(1)
        # Match Quack/PyTorch reference behavior: gradient math uses float32
        # intermediates even when (x, w, dout) are bf16/fp16.
        dout_f32 = dout[start:end].float()
        wdy = dout_f32 * w_f32
        c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
        dx_ref = ((wdy - x_hat * c1) * rstd_blk.unsqueeze(1)).to(x.dtype)

        torch.testing.assert_close(dx_oink[start:end], dx_ref, **tol_dx)
        dx_acc_ours.update(dx_oink[start:end], dx_ref)
        if dx_quack is not None:
            torch.testing.assert_close(dx_quack[start:end], dx_ref, **tol_dx)
            assert dx_acc_quack is not None
            dx_acc_quack.update(dx_quack[start:end], dx_ref)

        if dw_oink is not None:
            dw_accum += (dout_f32 * x_hat).sum(dim=0)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_dx", dx_acc_ours.finalize()))
    if dx_acc_quack is not None:
        stats.update(error_stats_to_row("quack_err_dx", dx_acc_quack.finalize()))

    if dw_oink is not None:
        dw_ref = dw_accum.to(w.dtype)
        if w.dtype == torch.float32:
            # Weight grad is sensitive to reduction order; use a slightly larger
            # absolute tolerance in the suite harness (Quack's unit tests use
            # smaller M, where dw is typically tighter).
            dw_tol = dict(atol=2e-3, rtol=1e-3)
        else:
            # For fp16/bf16 weights, `dw` is low-precision and grows with M; use an
            # ulp/magnitude-aware tolerance rather than a fixed epsilon.
            dw_ref_f32 = dw_ref.to(torch.float32)
            dw_oink_f32 = dw_oink.to(torch.float32)
            scale = float(dw_ref_f32.abs().max().item())
            dw_atol = max(2.0 * torch.finfo(w.dtype).eps * scale, 1e-3)
            dw_tol = dict(atol=dw_atol, rtol=1e-3)
            torch.testing.assert_close(dw_oink_f32, dw_ref_f32, **dw_tol)
            if dw_quack is not None:
                torch.testing.assert_close(
                    dw_quack.to(torch.float32), dw_ref_f32, **dw_tol
                )
            dw_tol = None  # handled above
        if dw_tol is not None:
            torch.testing.assert_close(dw_oink, dw_ref, **dw_tol)
            if dw_quack is not None:
                torch.testing.assert_close(dw_quack, dw_ref, **dw_tol)

        # Record weight-grad error stats (small, so exact p99 over the full vector).
        dw_acc_ours = ErrorStatsAccumulator(
            total_elems=int(dw_ref.numel()), p99_target_samples=int(dw_ref.numel())
        )
        dw_acc_ours.update(dw_oink, dw_ref)
        stats.update(error_stats_to_row("ours_err_dw", dw_acc_ours.finalize()))
        if dw_quack is not None:
            dw_acc_quack = ErrorStatsAccumulator(
                total_elems=int(dw_ref.numel()), p99_target_samples=int(dw_ref.numel())
            )
            dw_acc_quack.update(dw_quack, dw_ref)
            stats.update(error_stats_to_row("quack_err_dw", dw_acc_quack.finalize()))

    assert db_oink is None and db_quack is None
    assert dres_oink is None and dres_quack is None
    return stats


def bench_single(
    M: int,
    N: int,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    iters_ms: int,
    eps: float,
    warmup_ms: int,
    verify: bool,
) -> Tuple[Result, Result | None, dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=weight_dtype)
    dout = torch.randn(M, N, device=device, dtype=dtype)
    # rstd is fp32 per row; compute once outside the timed region.
    with torch.no_grad():
        xf = x.float()
        rstd = torch.rsqrt(xf.square().mean(dim=-1) + eps).to(torch.float32)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x, w, dout, rstd, has_bias=False, has_residual=False)

    def fn_oink():
        return oink_rmsnorm.rmsnorm_backward(
            x,
            w,
            dout,
            rstd,
            dresidual_out=None,
            has_bias=False,
            has_residual=False,
        )

    ms_oink = do_bench_triton(fn_oink, warmup_ms=warmup_ms, rep_ms=iters_ms)
    bytes_io = bytes_io_model_bwd(M, N, dtype, weight_dtype=w.dtype)
    gbps_oink = bytes_io / (ms_oink * 1e-3) / 1e9
    ours = Result(ms=ms_oink, gbps=gbps_oink)

    if quack_rmsnorm_bwd is None:
        return ours, None, stats

    def fn_quack():
        return quack_rmsnorm_bwd(
            x,
            w,
            dout,
            rstd,
            dresidual_out=None,
            has_bias=False,
            has_residual=False,
        )

    ms_quack = do_bench_triton(fn_quack, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_quack = bytes_io / (ms_quack * 1e-3) / 1e9
    return ours, Result(ms=ms_quack, gbps=gbps_quack), stats


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    print(f"Running on {torch.cuda.get_device_name(device)} (SM{sm})")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    p.add_argument(
        "--weight-dtype",
        type=str,
        default="fp32",
        choices=["same", "fp16", "bf16", "fp32"],
        help="RMSNorm weight dtype. `same` matches activation dtype.",
    )
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Triton do_bench rep_ms (kernel-only).",
    )
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument(
        "--csv", type=str, default=None, help="Optional CSV output path; appends rows"
    )
    p.add_argument(
        "--json", type=str, default=None, help="Optional JSON output path (meta + rows)"
    )
    p.add_argument("--configs", type=str, default="1024x4096,8192x4096")
    p.add_argument(
        "--quack-suite", action="store_true", help="Run Quack-style batch/seq grid"
    )
    p.add_argument(
        "--dsv3",
        action="store_true",
        help="Run DSv3 set: M in {4096,16384,65536}, N in {6144,7168,8192}",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip correctness checks (Oink/Quack vs a pure-PyTorch RMSNorm backward reference)",
    )
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    if args.weight_dtype == "same":
        weight_dtype = dtype
    else:
        weight_dtype = parse_dtype(args.weight_dtype)
    eps = float(args.eps)

    if args.quack_suite:
        cfgs = [(bs * sl, hidden) for (bs, sl, hidden) in quack_suite_configs()]
    elif args.dsv3:
        cfgs = dsv3_configs()
    else:
        cfgs = parse_configs(args.configs)

    hbm_peak = detect_hbm_peak_gbps(device)

    rows_out: list[dict[str, object]] = []

    for M, N in cfgs:
        print(f"bench M={M:<8d} N={N:<6d} dtype={args.dtype} ...", flush=True)
        ours, quack, stats = bench_single(
            M=M,
            N=N,
            dtype=dtype,
            weight_dtype=weight_dtype,
            iters_ms=int(args.iters),
            eps=eps,
            warmup_ms=int(args.warmup_ms),
            verify=not args.skip_verify,
        )

        row: dict[str, object] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            "ours_ms": ours.ms,
            "ours_gbps": ours.gbps,
            "ours_tbps": ours.gbps / 1000.0,
            "ours_hbm_frac": ours.gbps / hbm_peak,
        }
        if quack is not None:
            row.update(
                {
                    "quack_ms": quack.ms,
                    "quack_gbps": quack.gbps,
                    "quack_tbps": quack.gbps / 1000.0,
                    "speedup_vs_quack": quack.ms / ours.ms,
                }
            )
        row.update(stats)
        rows_out.append(row)

        if args.csv is not None:
            file_exists = os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    if args.json is not None:
        meta = collect_device_meta(device)
        write_json(
            args.json,
            meta,
            rows_out,
            extra={
                "method": "triton.testing.do_bench(mean)",
                "warmup_ms": int(args.warmup_ms),
                "rep_ms": int(args.iters),
                "io_model_bytes": "see bytes_io_model_bwd in script",
                "weight_dtype": str(args.weight_dtype),
            },
        )

    # Print a small summary table.
    headers = ["M", "N", "dtype", "ours_ms", "ours_tbps", "ours_hbm_frac"]
    if quack_rmsnorm_bwd is not None:
        headers += ["quack_ms", "quack_tbps", "speedup_vs_quack"]
    print("\nSummary:")
    print(" ".join(h.rjust(14) for h in headers))
    for r in rows_out:
        parts: list[str] = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                parts.append(f"{v:14.4f}")
            else:
                parts.append(f"{str(v):>14}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
