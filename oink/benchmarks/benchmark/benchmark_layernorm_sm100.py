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

from kernelagent_oink.blackwell import layernorm as oink_ln  # noqa: E402

try:
    # Quack exposes LayerNorm through the RMSNorm module (is_layernorm=True path).
    from quack.rmsnorm import layernorm_fwd as quack_layernorm  # type: ignore
except Exception:
    quack_layernorm = None

_VERIFY_TOL_Y = {
    # Match Quack's unit-test defaults (tests/test_layernorm.py).
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}

# Quack checks rstd/mean (fp32) with a tighter fixed tolerance.
_VERIFY_TOL_STATS = dict(atol=6e-4, rtol=6e-4)


def bytes_io_model_layernorm(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    has_bias: bool,
    return_rstd: bool,
    return_mean: bool,
    weight_dtype: torch.dtype = torch.float32,
) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    w_elem = torch.tensor(0, dtype=weight_dtype).element_size()
    total = 0
    # Read x + write y
    total += 2 * M * N * elem
    # Read weight (+ optional bias) along feature dim
    total += N * w_elem
    if has_bias:
        total += N * w_elem
    # Optional per-row stats (fp32)
    if return_rstd:
        total += M * 4
    if return_mean:
        total += M * 4
    return int(total)


def dsv3_configs() -> List[Tuple[int, int]]:
    Ms = [4096, 16384, 65536]
    Ns = [6144, 7168, 8192]
    return [(m, n) for m in Ms for n in Ns]


def _verify_parity(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    *,
    eps: float,
    return_rstd: bool,
    return_mean: bool,
) -> dict[str, object]:
    tol_y = _VERIFY_TOL_Y[x.dtype]
    ref_block_rows = 4096
    M = int(x.shape[0])
    N = int(x.shape[1])

    y_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    y_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N) if (quack_layernorm is not None and b is None) else None
    )
    with torch.no_grad():
        ours = oink_ln.layernorm(
            x,
            w,
            bias=b,
            eps=eps,
            return_rstd=return_rstd,
            return_mean=return_mean,
        )
        quack = None
        if quack_layernorm is not None and b is None:
            quack = quack_layernorm(
                x,
                w,
                eps=eps,
                return_rstd=return_rstd,
                return_mean=return_mean,
            )
    torch.cuda.synchronize()

    def _unpack(out):
        if return_rstd and return_mean:
            y, rstd, mean = out
        elif return_rstd and not return_mean:
            y, rstd = out
            mean = None
        elif return_mean and not return_rstd:
            y, mean = out
            rstd = None
        else:
            y, rstd, mean = out, None, None
        return y, rstd, mean

    y_o, rstd_o, mean_o = _unpack(ours)
    y_q, rstd_q, mean_q = _unpack(quack) if quack is not None else (None, None, None)

    # Pure-PyTorch reference (float32 accumulation), matching Quack's unit tests:
    # - compute ref output via F.layer_norm on float32
    # - compute mean/rstd from float32 input
    rstd_ref_all = torch.empty((M,), device=x.device, dtype=torch.float32) if return_rstd else None
    mean_ref_all = torch.empty((M,), device=x.device, dtype=torch.float32) if return_mean else None

    for start, end in iter_row_blocks(M, ref_block_rows):
        x_f32 = x[start:end].float()
        y_ref_f32 = torch.nn.functional.layer_norm(x_f32, w.shape, w, b, eps)
        y_ref = y_ref_f32.to(x.dtype)
        torch.testing.assert_close(y_o[start:end], y_ref, **tol_y)
        y_acc_ours.update(y_o[start:end], y_ref)
        if y_q is not None:
            torch.testing.assert_close(y_q[start:end], y_ref, **tol_y)
            assert y_acc_quack is not None
            y_acc_quack.update(y_q[start:end], y_ref)

        # Per-row stats in fp32, as in Quack's tests.
        if return_rstd or return_mean:
            mean_f32 = x_f32.mean(dim=-1)
            if return_mean:
                assert mean_ref_all is not None
                mean_ref_all[start:end] = mean_f32
            if return_rstd:
                var_f32 = ((x_f32 - mean_f32.unsqueeze(1)) ** 2).mean(dim=-1)
                rstd_ref = 1.0 / torch.sqrt(var_f32 + eps)
                assert rstd_ref_all is not None
                rstd_ref_all[start:end] = rstd_ref

                assert rstd_o is not None
                torch.testing.assert_close(rstd_o[start:end], rstd_ref, **_VERIFY_TOL_STATS)
                if rstd_q is not None:
                    torch.testing.assert_close(rstd_q[start:end], rstd_ref, **_VERIFY_TOL_STATS)

            if return_mean:
                mean_ref = mean_f32
                assert mean_o is not None
                torch.testing.assert_close(mean_o[start:end], mean_ref, **_VERIFY_TOL_STATS)
                if mean_q is not None:
                    torch.testing.assert_close(mean_q[start:end], mean_ref, **_VERIFY_TOL_STATS)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_y", y_acc_ours.finalize()))
    if y_acc_quack is not None:
        stats.update(error_stats_to_row("quack_err_y", y_acc_quack.finalize()))

    if return_rstd:
        assert rstd_o is not None and rstd_ref_all is not None
        rstd_acc_ours = ErrorStatsAccumulator(
            total_elems=int(rstd_ref_all.numel()), p99_target_samples=int(rstd_ref_all.numel())
        )
        rstd_acc_ours.update(rstd_o, rstd_ref_all)
        stats.update(error_stats_to_row("ours_err_rstd", rstd_acc_ours.finalize()))
        if rstd_q is not None:
            rstd_acc_quack = ErrorStatsAccumulator(
                total_elems=int(rstd_ref_all.numel()), p99_target_samples=int(rstd_ref_all.numel())
            )
            rstd_acc_quack.update(rstd_q, rstd_ref_all)
            stats.update(error_stats_to_row("quack_err_rstd", rstd_acc_quack.finalize()))

    if return_mean:
        assert mean_o is not None and mean_ref_all is not None
        mean_acc_ours = ErrorStatsAccumulator(
            total_elems=int(mean_ref_all.numel()), p99_target_samples=int(mean_ref_all.numel())
        )
        mean_acc_ours.update(mean_o, mean_ref_all)
        stats.update(error_stats_to_row("ours_err_mean", mean_acc_ours.finalize()))
        if mean_q is not None:
            mean_acc_quack = ErrorStatsAccumulator(
                total_elems=int(mean_ref_all.numel()), p99_target_samples=int(mean_ref_all.numel())
            )
            mean_acc_quack.update(mean_q, mean_ref_all)
            stats.update(error_stats_to_row("quack_err_mean", mean_acc_quack.finalize()))

    return stats


def bench_single(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    eps: float,
    warmup_ms: int,
    iters_ms: int,
    verify: bool,
    return_rstd: bool,
    return_mean: bool,
    has_bias: bool,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32) if has_bias else None

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x, w, b, eps=eps, return_rstd=return_rstd, return_mean=return_mean)

    bytes_io = bytes_io_model_layernorm(
        M,
        N,
        dtype,
        has_bias=has_bias,
        return_rstd=return_rstd,
        return_mean=return_mean,
        weight_dtype=w.dtype,
    )

    def fn_oink():
        return oink_ln.layernorm(
            x,
            w,
            bias=b,
            eps=eps,
            return_rstd=return_rstd,
            return_mean=return_mean,
        )

    ms_oink = do_bench_triton(fn_oink, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_oink = bytes_io / (ms_oink * 1e-3) / 1e9

    if quack_layernorm is None or has_bias:
        return (ms_oink, gbps_oink), None, stats

    def fn_quack():
        return quack_layernorm(
            x,
            w,
            eps=eps,
            return_rstd=return_rstd,
            return_mean=return_mean,
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
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--return-rstd", action="store_true")
    p.add_argument("--return-mean", action="store_true")
    p.add_argument("--with-bias", action="store_true", help="Benchmark bias path (Quack compare skipped)")
    p.add_argument("--iters", type=int, default=100, help="Triton do_bench rep_ms (kernel-only).")
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path; appends rows")
    p.add_argument("--json", type=str, default=None, help="Optional JSON output path (meta + rows)")
    p.add_argument("--configs", type=str, default="1024x4096,8192x4096")
    p.add_argument("--quack-suite", action="store_true", help="Run Quack-style batch/seq grid (hidden=4096)")
    p.add_argument(
        "--dsv3",
        action="store_true",
        help="Run DSv3 set: M in {4096,16384,65536}, N in {6144,7168,8192}",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip correctness checks (Oink/Quack vs a pure-PyTorch reference; Quack compare skipped when bias is enabled)",
    )
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
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
    for (M, N) in cfgs:
        print(f"bench M={M:<8d} N={N:<6d} dtype={args.dtype} ...", flush=True)
        (ms_oink, gbps_oink), quack, stats = bench_single(
            M=M,
            N=N,
            dtype=dtype,
            eps=eps,
            warmup_ms=int(args.warmup_ms),
            iters_ms=int(args.iters),
            verify=not args.skip_verify,
            return_rstd=bool(args.return_rstd),
            return_mean=bool(args.return_mean),
            has_bias=bool(args.with_bias),
        )
        row: Dict[str, Any] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "eps": eps,
            "return_rstd": bool(args.return_rstd),
            "return_mean": bool(args.return_mean),
            "with_bias": bool(args.with_bias),
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
                "io_model_bytes": "see bytes_io_model_layernorm in script",
            },
        )

    headers = ["M", "N", "ours_ms", "ours_tbps"]
    if quack_layernorm is not None and (not args.with_bias):
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
