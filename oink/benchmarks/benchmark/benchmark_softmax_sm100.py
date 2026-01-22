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

from kernelagent_oink.blackwell import softmax as oink_softmax  # noqa: E402

try:
    from quack.softmax import softmax_bwd as quack_softmax_bwd  # type: ignore
    from quack.softmax import softmax_fwd as quack_softmax_fwd  # type: ignore
except Exception:
    quack_softmax_fwd = None
    quack_softmax_bwd = None

_VERIFY_TOL = {
    # Match Quack's unit-test defaults (tests/test_softmax.py).
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


def bytes_io_model_softmax(M: int, N: int, dtype: torch.dtype, *, mode: str) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    if mode == "fwd":
        return int(2 * M * N * elem)  # read x + write y
    if mode == "bwd":
        return int(3 * M * N * elem)  # read dy + read y + write dx
    if mode == "fwd_bwd":
        # Logical IO for dx given (x, dy): read x + read dy + write dx.
        # (The intermediate y=softmax(x) is an implementation detail and is
        # intentionally not counted here.)
        return int(3 * M * N * elem)
    raise ValueError(f"Unsupported mode: {mode}")


def dsv3_configs() -> List[Tuple[int, int]]:
    Ms = [4096, 16384, 65536]
    Ns = [6144, 7168, 8192]
    return [(m, n) for m in Ms for n in Ns]


def _verify_parity(x: torch.Tensor) -> dict[str, object]:
    tol = _VERIFY_TOL[x.dtype]
    ref_block_rows = 4096
    dy = torch.randn_like(x)  # upstream grad

    with torch.no_grad():
        y_o = oink_softmax.softmax_forward(x)
        dx_o = oink_softmax.softmax_backward(dy, y_o)
        dx_fused_o = oink_softmax.softmax_fwd_bwd(dy, x)

        y_q = None
        dx_q = None
        if quack_softmax_fwd is not None and quack_softmax_bwd is not None:
            y_q = quack_softmax_fwd(x)
            dx_q = quack_softmax_bwd(dy, y_q)

    M = int(x.shape[0])
    N = int(x.shape[1])
    y_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    dx_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    dx_fused_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    y_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if (quack_softmax_fwd is not None and quack_softmax_bwd is not None)
        else None
    )
    dx_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if (quack_softmax_fwd is not None and quack_softmax_bwd is not None)
        else None
    )

    # Match Quack tests: compare to PyTorch softmax refs (fwd+bwd), chunked.
    for start, end in iter_row_blocks(M, ref_block_rows):
        x_blk = x[start:end]
        dy_blk = dy[start:end]
        y_ref_blk = torch.softmax(x_blk, dim=-1)
        dot = torch.sum(dy_blk * y_ref_blk, dim=-1, keepdim=True, dtype=torch.float32)
        dx_ref_blk = (dy_blk - dot.to(dy_blk.dtype)) * y_ref_blk

        torch.testing.assert_close(y_o[start:end], y_ref_blk, **tol)
        torch.testing.assert_close(dx_o[start:end], dx_ref_blk, **tol)
        torch.testing.assert_close(dx_fused_o[start:end], dx_ref_blk, **tol)
        y_acc_ours.update(y_o[start:end], y_ref_blk)
        dx_acc_ours.update(dx_o[start:end], dx_ref_blk)
        dx_fused_acc_ours.update(dx_fused_o[start:end], dx_ref_blk)
        if y_q is not None and dx_q is not None:
            torch.testing.assert_close(y_q[start:end], y_ref_blk, **tol)
            torch.testing.assert_close(dx_q[start:end], dx_ref_blk, **tol)
            assert y_acc_quack is not None and dx_acc_quack is not None
            y_acc_quack.update(y_q[start:end], y_ref_blk)
            dx_acc_quack.update(dx_q[start:end], dx_ref_blk)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_y", y_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_dx", dx_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_dx_fused", dx_fused_acc_ours.finalize()))
    if y_acc_quack is not None and dx_acc_quack is not None:
        stats.update(error_stats_to_row("quack_err_y", y_acc_quack.finalize()))
        stats.update(error_stats_to_row("quack_err_dx", dx_acc_quack.finalize()))
    return stats


def bench_single(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    warmup_ms: int,
    iters_ms: int,
    mode: str,
    verify: bool,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    dy = torch.randn_like(x)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x)

    bytes_io = bytes_io_model_softmax(M, N, dtype, mode=mode)

    if mode == "fwd":

        def fn_oink():
            return oink_softmax.softmax_forward(x)

        fn_quack = None
        if quack_softmax_fwd is not None:

            def fn_quack():
                return quack_softmax_fwd(x)

    elif mode == "bwd":
        with torch.no_grad():
            y_o = oink_softmax.softmax_forward(x)
            y_q = quack_softmax_fwd(x) if quack_softmax_fwd is not None else None

        def fn_oink():
            return oink_softmax.softmax_backward(dy, y_o)

        fn_quack = None
        if quack_softmax_bwd is not None and y_q is not None:

            def fn_quack():
                return quack_softmax_bwd(dy, y_q)

    elif mode == "fwd_bwd":

        def fn_oink():
            return oink_softmax.softmax_fwd_bwd(dy, x)

        fn_quack = None
        if quack_softmax_fwd is not None and quack_softmax_bwd is not None:

            def fn_quack():
                return quack_softmax_bwd(dy, quack_softmax_fwd(x))

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    ms_oink = do_bench_triton(fn_oink, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_oink = bytes_io / (ms_oink * 1e-3) / 1e9

    if fn_quack is None:
        return (ms_oink, gbps_oink), None, stats

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
        "--mode", type=str, default="fwd_bwd", choices=["fwd", "bwd", "fwd_bwd"]
    )
    p.add_argument(
        "--iters", type=int, default=50, help="Triton do_bench rep_ms (kernel-only)."
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
        help="Skip correctness checks (Oink/Quack vs PyTorch softmax)",
    )
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)

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
        print(
            f"bench M={M:<8d} N={N:<6d} dtype={args.dtype} mode={args.mode} ...",
            flush=True,
        )
        (ms_oink, gbps_oink), quack, stats = bench_single(
            M=M,
            N=N,
            dtype=dtype,
            warmup_ms=int(args.warmup_ms),
            iters_ms=int(args.iters),
            mode=str(args.mode),
            verify=not args.skip_verify,
        )
        row: Dict[str, Any] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "mode": args.mode,
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
                "io_model_bytes": "mode-dependent: fwd=2*M*N, bwd=3*M*N, fwd_bwd=3*M*N (all * elem_size; fwd_bwd counts logical x+dy+dx)",
            },
        )

    headers = ["M", "N", "mode", "ours_ms", "ours_tbps"]
    if quack_softmax_fwd is not None and quack_softmax_bwd is not None:
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
