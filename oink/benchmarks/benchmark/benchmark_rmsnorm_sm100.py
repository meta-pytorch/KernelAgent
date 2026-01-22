from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

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
    parse_configs,
    parse_dtype,
    quack_suite_configs,
    write_csv,
    write_json,
)

ensure_oink_src_on_path()

from kernelagent_oink.blackwell import rmsnorm as oink_rmsnorm  # noqa: E402

try:
    from quack.rmsnorm import rmsnorm_fwd as quack_rmsnorm_fwd  # type: ignore
except Exception:
    quack_rmsnorm_fwd = None

_VERIFY_TOL_Y = {
    # Match Quack's unit-test defaults (tests/test_rmsnorm.py).
    torch.float32: dict(atol=1e-4, rtol=1e-3),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
    # NOTE: bf16 ulp grows with magnitude; a slightly larger rtol is more robust
    # for the large-M suite shapes (and fused paths that can see larger values).
    torch.bfloat16: dict(atol=1e-1, rtol=1e-2),
}

_VERIFY_TOL_RSTD = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-3, rtol=1e-3),
}


def bytes_io_model_fwd(
    M: int, N: int, dtype: torch.dtype, *, weight_dtype: torch.dtype = torch.float32
) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    w_elem = torch.tensor(0, dtype=weight_dtype).element_size()
    # Read x + write y
    total = 2 * M * N * elem
    # Read weight
    total += N * w_elem
    return int(total)


def dsv3_configs() -> List[Tuple[int, int]]:
    # DSv3-ish hidden sizes used throughout the Oink/Quack SM100 suite tables.
    Ms = [4096, 16384, 65536]
    Ns = [6144, 7168, 8192]
    return [(m, n) for m in Ms for n in Ns]


def _verify_parity(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    eps: float,
    store_rstd: bool,
) -> dict[str, object]:
    tol_y = _VERIFY_TOL_Y[x.dtype]
    tol_rstd = _VERIFY_TOL_RSTD[x.dtype]
    ref_block_rows = 4096
    M = int(x.shape[0])
    N = int(x.shape[1])

    y_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    y_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if quack_rmsnorm_fwd is not None
        else None
    )

    with torch.no_grad():
        y_o, rstd_o, res_o = oink_rmsnorm.rmsnorm_forward(
            x,
            weight=w,
            bias=None,
            residual=None,
            eps=eps,
            store_rstd=store_rstd,
        )
        y_q = None
        rstd_q = None
        if quack_rmsnorm_fwd is not None:
            # Quack returns (out, residual_out, rstd).
            y_q, res_q, rstd_q = quack_rmsnorm_fwd(
                x,
                w,
                bias=None,
                residual=None,
                out_dtype=None,
                residual_dtype=None,
                eps=eps,
                store_rstd=store_rstd,
            )

    # Pure-PyTorch reference (float32 accumulation), chunked over rows to avoid
    # materializing an (M, N) float32 tensor for large Quack-suite shapes.
    w_f32 = w.float()
    rstd_ref = torch.empty((M,), device=x.device, dtype=torch.float32)
    for start, end in iter_row_blocks(M, ref_block_rows):
        x_f32 = x[start:end].float()
        rstd_blk = torch.rsqrt(x_f32.square().mean(dim=-1) + eps)
        rstd_ref[start:end] = rstd_blk

        y_ref_blk_f32 = (x_f32 * rstd_blk.unsqueeze(1)) * w_f32
        y_ref_blk = y_ref_blk_f32.to(x.dtype)
        torch.testing.assert_close(y_o[start:end], y_ref_blk, **tol_y)
        y_acc_ours.update(y_o[start:end], y_ref_blk)
        if y_q is not None:
            torch.testing.assert_close(y_q[start:end], y_ref_blk, **tol_y)
            assert y_acc_quack is not None
            y_acc_quack.update(y_q[start:end], y_ref_blk)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_y", y_acc_ours.finalize()))
    if y_acc_quack is not None:
        stats.update(error_stats_to_row("quack_err_y", y_acc_quack.finalize()))

    if store_rstd:
        assert rstd_o is not None
        torch.testing.assert_close(rstd_o, rstd_ref, **tol_rstd)
        if y_q is not None:
            assert rstd_q is not None
            torch.testing.assert_close(rstd_q, rstd_ref, **tol_rstd)
        # Stats for rstd are cheap (M elements); compute exact p99 over all rows.
        rstd_acc_ours = ErrorStatsAccumulator(
            total_elems=int(rstd_ref.numel()), p99_target_samples=int(rstd_ref.numel())
        )
        rstd_acc_ours.update(rstd_o, rstd_ref)
        stats.update(error_stats_to_row("ours_err_rstd", rstd_acc_ours.finalize()))
        if rstd_q is not None:
            rstd_acc_quack = ErrorStatsAccumulator(
                total_elems=int(rstd_ref.numel()),
                p99_target_samples=int(rstd_ref.numel()),
            )
            rstd_acc_quack.update(rstd_q, rstd_ref)
            stats.update(
                error_stats_to_row("quack_err_rstd", rstd_acc_quack.finalize())
            )
    # Residual output semantics differ slightly across implementations:
    # - Oink returns `None` when residual is None.
    # - Quack returns `x` as a safe alias in that case.
    #
    # For parity we focus on `y` (and optional `rstd`) for the residual=None path.
    assert res_o is None
    if quack_rmsnorm_fwd is not None:
        assert res_q is x
    return stats


def bench_single(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    weight_dtype: torch.dtype,
    eps: float,
    warmup_ms: int,
    iters_ms: int,
    verify: bool,
    store_rstd: bool,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=weight_dtype)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x, w, eps=eps, store_rstd=store_rstd)

    bytes_io = bytes_io_model_fwd(M, N, dtype, weight_dtype=w.dtype)

    def fn_oink():
        return oink_rmsnorm.rmsnorm_forward(
            x,
            weight=w,
            bias=None,
            residual=None,
            eps=eps,
            store_rstd=store_rstd,
        )

    ms_oink = do_bench_triton(fn_oink, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_oink = bytes_io / (ms_oink * 1e-3) / 1e9

    if quack_rmsnorm_fwd is None:
        return (ms_oink, gbps_oink), None, stats

    def fn_quack():
        return quack_rmsnorm_fwd(
            x,
            w,
            bias=None,
            residual=None,
            out_dtype=None,
            residual_dtype=None,
            eps=eps,
            store_rstd=store_rstd,
        )

    ms_quack = do_bench_triton(fn_quack, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_quack = bytes_io / (ms_quack * 1e-3) / 1e9
    return (ms_oink, gbps_oink), (ms_quack, gbps_quack), stats


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
        help="RMSNorm weight dtype. `same` matches activation dtype (vLLM-style inference).",
    )
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument(
        "--store-rstd", action="store_true", help="Also write rstd (fp32 per row)"
    )
    p.add_argument(
        "--iters", type=int, default=100, help="Triton do_bench rep_ms (kernel-only)."
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
        help="Skip correctness checks (Oink/Quack vs a pure-PyTorch reference)",
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
    meta = collect_device_meta(device)

    rows_out: List[Dict[str, Any]] = []
    for M, N in cfgs:
        print(f"bench M={M:<8d} N={N:<6d} dtype={args.dtype} ...", flush=True)
        (ms_oink, gbps_oink), quack, stats = bench_single(
            M=M,
            N=N,
            dtype=dtype,
            weight_dtype=weight_dtype,
            eps=eps,
            warmup_ms=int(args.warmup_ms),
            iters_ms=int(args.iters),
            verify=not args.skip_verify,
            store_rstd=bool(args.store_rstd),
        )
        row: Dict[str, Any] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            "eps": eps,
            "store_rstd": bool(args.store_rstd),
            "ours_ms": ms_oink,
            "ours_gbps": gbps_oink,
            "ours_tbps": gbps_oink / 1000.0,
            "ours_hbm_frac": gbps_oink / hbm_peak,
        }
        if quack is not None:
            ms_q, gbps_q = quack
            row.update(
                {
                    "quack_ms": ms_q,
                    "quack_gbps": gbps_q,
                    "quack_tbps": gbps_q / 1000.0,
                    "speedup_vs_quack": ms_q / ms_oink,
                }
            )
        row.update(stats)
        rows_out.append(row)

    if args.csv is not None:
        write_csv(args.csv, rows_out)
    if args.json is not None:
        write_json(
            args.json,
            meta,
            rows_out,
            extra={
                "method": "triton.testing.do_bench(mean)",
                "warmup_ms": int(args.warmup_ms),
                "rep_ms": int(args.iters),
                "io_model_bytes": "(2*M*N)*elem_size + N*weight_elem_size",
                "store_rstd": bool(args.store_rstd),
                "weight_dtype": str(args.weight_dtype),
            },
        )

    # Print a compact summary table.
    headers = ["M", "N", "ours_ms", "ours_tbps"]
    if quack_rmsnorm_fwd is not None:
        headers += ["quack_ms", "quack_tbps", "speedup_vs_quack"]
    print("\nSummary:")
    print(" ".join(h.rjust(14) for h in headers))
    for r in rows_out:
        parts: List[str] = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                parts.append(f"{v:14.4f}")
            else:
                parts.append(f"{str(v):>14}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
