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

from kernelagent_oink.blackwell import cross_entropy as oink_ce  # noqa: E402

try:
    from quack.cross_entropy import cross_entropy_bwd as quack_ce_bwd  # type: ignore
    from quack.cross_entropy import cross_entropy_fwd as quack_ce_fwd  # type: ignore
except Exception:
    quack_ce_fwd = None
    quack_ce_bwd = None


# Match Quack's unit-test defaults (tests/test_cross_entropy.py).
_VERIFY_TOL_LOSS = dict(atol=5e-5, rtol=1e-5)  # float32 outputs (loss/lse)
_VERIFY_TOL_DX = {
    torch.float32: dict(atol=5e-5, rtol=1e-5),
    # FP16 `dx` is low-precision; allow ~1 ulp at typical magnitudes.
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    # BF16 `dx` is low-precision; allow ~1 ulp at typical magnitudes.
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


def bytes_io_model_ce(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    target_dtype: torch.dtype = torch.int64,
    mode: str,
) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    t_elem = torch.tensor(0, dtype=target_dtype).element_size()
    # Forward:
    #   read logits (M*N) + read target (M) + write loss (M fp32) + write lse (M fp32)
    fwd = M * N * elem + M * t_elem + 2 * M * 4
    # Backward (reduction="none" path):
    #   read logits (M*N) + read target (M) + read dloss (M fp32) + read lse (M fp32) + write dx (M*N)
    bwd = 2 * M * N * elem + M * t_elem + 2 * M * 4

    if mode == "fwd":
        return int(fwd)
    if mode == "bwd":
        return int(bwd)
    if mode == "fwd_bwd":
        # Logical IO for dx given (logits, target, dloss): read logits + read target
        # + read dloss + write dx. (Intermediate lse/loss are implementation details.)
        return int(2 * M * N * elem + M * t_elem + M * 4)
    raise ValueError(f"Unsupported mode: {mode}")


def dsv3_configs() -> List[Tuple[int, int]]:
    Ms = [4096, 16384, 65536]
    Ns = [3072, 6144, 8192, 12288]
    return [(m, n) for m in Ms for n in Ns]


def _verify_parity(
    logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int
) -> dict[str, object]:
    dtype = logits.dtype
    ref_block_rows = 512
    dloss = torch.randn(
        logits.size(0), device=logits.device, dtype=torch.float32
    )  # upstream grad

    with torch.no_grad():
        loss_o, lse_o = oink_ce.cross_entropy_forward(
            logits, target, ignore_index=ignore_index, reduction="none"
        )
        dx_o = oink_ce.cross_entropy_backward(
            dloss, logits, target, lse_o, ignore_index=ignore_index
        )
        dx_fused_o = oink_ce.cross_entropy_fwd_bwd(
            dloss,
            logits,
            target,
            ignore_index=ignore_index,
        )

        loss_q = None
        lse_q = None
        dx_q = None
        if quack_ce_fwd is not None and quack_ce_bwd is not None:
            loss_q, lse_q = quack_ce_fwd(
                logits,
                target,
                target_logit=None,
                ignore_index=ignore_index,
                return_lse=True,
                return_dx=False,
                inplace_backward=False,
            )
            dx_q = quack_ce_bwd(
                logits,
                target,
                dloss,
                lse_q,
                ignore_index=ignore_index,
                inplace_backward=False,
            )

    M = int(logits.shape[0])
    N = int(logits.shape[1])
    loss_acc_ours = ErrorStatsAccumulator(
        total_elems=M, p99_target_samples=min(M, 1_000_000)
    )
    lse_acc_ours = ErrorStatsAccumulator(
        total_elems=M, p99_target_samples=min(M, 1_000_000)
    )
    dx_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    dx_fused_acc_ours = ErrorStatsAccumulator(total_elems=M * N)
    loss_acc_quack = (
        ErrorStatsAccumulator(total_elems=M, p99_target_samples=min(M, 1_000_000))
        if (quack_ce_fwd is not None and quack_ce_bwd is not None)
        else None
    )
    lse_acc_quack = (
        ErrorStatsAccumulator(total_elems=M, p99_target_samples=min(M, 1_000_000))
        if (quack_ce_fwd is not None and quack_ce_bwd is not None)
        else None
    )
    dx_acc_quack = (
        ErrorStatsAccumulator(total_elems=M * N)
        if (quack_ce_fwd is not None and quack_ce_bwd is not None)
        else None
    )

    # Match Quack tests: compare to a PyTorch reference computed on float32 logits.
    # Chunk over rows so we don't materialize a full (M, N) float32 tensor.
    for start, end in iter_row_blocks(M, ref_block_rows):
        logits_f32 = logits[start:end].float().requires_grad_(True)
        target_blk = target[start:end]
        dloss_blk = dloss[start:end]

        loss_ref = torch.nn.functional.cross_entropy(
            logits_f32,
            target_blk,
            reduction="none",
            ignore_index=ignore_index,
        )
        lse_ref = torch.logsumexp(logits_f32, dim=-1)
        (dx_ref_f32,) = torch.autograd.grad(
            loss_ref, logits_f32, grad_outputs=dloss_blk
        )
        dx_ref = dx_ref_f32.to(dtype)

        torch.testing.assert_close(
            loss_o[start:end], loss_ref.detach(), **_VERIFY_TOL_LOSS
        )
        torch.testing.assert_close(
            lse_o[start:end], lse_ref.detach(), **_VERIFY_TOL_LOSS
        )
        torch.testing.assert_close(dx_o[start:end], dx_ref, **_VERIFY_TOL_DX[dtype])
        torch.testing.assert_close(
            dx_fused_o[start:end], dx_ref, **_VERIFY_TOL_DX[dtype]
        )
        loss_acc_ours.update(loss_o[start:end], loss_ref.detach())
        lse_acc_ours.update(lse_o[start:end], lse_ref.detach())
        dx_acc_ours.update(dx_o[start:end], dx_ref)
        dx_fused_acc_ours.update(dx_fused_o[start:end], dx_ref)

        if loss_q is not None and lse_q is not None and dx_q is not None:
            torch.testing.assert_close(
                loss_q[start:end], loss_ref.detach(), **_VERIFY_TOL_LOSS
            )
            torch.testing.assert_close(
                lse_q[start:end], lse_ref.detach(), **_VERIFY_TOL_LOSS
            )
            torch.testing.assert_close(dx_q[start:end], dx_ref, **_VERIFY_TOL_DX[dtype])
            assert (
                loss_acc_quack is not None
                and lse_acc_quack is not None
                and dx_acc_quack is not None
            )
            loss_acc_quack.update(loss_q[start:end], loss_ref.detach())
            lse_acc_quack.update(lse_q[start:end], lse_ref.detach())
            dx_acc_quack.update(dx_q[start:end], dx_ref)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_loss", loss_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_lse", lse_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_dx", dx_acc_ours.finalize()))
    stats.update(error_stats_to_row("ours_err_dx_fused", dx_fused_acc_ours.finalize()))
    if (
        loss_acc_quack is not None
        and lse_acc_quack is not None
        and dx_acc_quack is not None
    ):
        stats.update(error_stats_to_row("quack_err_loss", loss_acc_quack.finalize()))
        stats.update(error_stats_to_row("quack_err_lse", lse_acc_quack.finalize()))
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
    ignore_index: int,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]], dict[str, object]]:
    device = torch.device("cuda")
    logits = 0.1 * torch.randn(M, N, device=device, dtype=dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    # Sprinkle some ignore_index entries for robustness (and to match reduction semantics).
    if ignore_index is not None:
        mask = torch.rand(M, device=device) < 0.01
        target[mask] = int(ignore_index)
    dloss = torch.randn(M, device=device, dtype=torch.float32)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(logits, target, ignore_index=int(ignore_index))

    bytes_io = bytes_io_model_ce(M, N, dtype, target_dtype=target.dtype, mode=mode)

    if mode == "fwd":

        def fn_oink():
            return oink_ce.cross_entropy_forward(
                logits, target, ignore_index=int(ignore_index), reduction="none"
            )

        fn_quack = None
        if quack_ce_fwd is not None:

            def fn_quack():
                return quack_ce_fwd(
                    logits,
                    target,
                    target_logit=None,
                    ignore_index=int(ignore_index),
                    return_lse=True,
                    return_dx=False,
                    inplace_backward=False,
                )

    elif mode == "bwd":
        with torch.no_grad():
            _loss_o, lse_o = oink_ce.cross_entropy_forward(
                logits, target, ignore_index=int(ignore_index), reduction="none"
            )
            if quack_ce_fwd is not None:
                _loss_q, lse_q = quack_ce_fwd(
                    logits,
                    target,
                    target_logit=None,
                    ignore_index=int(ignore_index),
                    return_lse=True,
                    return_dx=False,
                    inplace_backward=False,
                )
            else:
                lse_q = None

        def fn_oink():
            return oink_ce.cross_entropy_backward(
                dloss, logits, target, lse_o, ignore_index=int(ignore_index)
            )

        fn_quack = None
        if quack_ce_bwd is not None and lse_q is not None:

            def fn_quack():
                return quack_ce_bwd(
                    logits,
                    target,
                    dloss,
                    lse_q,
                    ignore_index=int(ignore_index),
                    inplace_backward=False,
                )

    elif mode == "fwd_bwd":

        def fn_oink():
            return oink_ce.cross_entropy_fwd_bwd(
                dloss,
                logits,
                target,
                ignore_index=int(ignore_index),
            )

        fn_quack = None
        if quack_ce_fwd is not None and quack_ce_bwd is not None:

            def fn_quack():
                _loss_q, lse_q = quack_ce_fwd(
                    logits,
                    target,
                    target_logit=None,
                    ignore_index=int(ignore_index),
                    return_lse=True,
                    return_dx=False,
                    inplace_backward=False,
                )
                return quack_ce_bwd(
                    logits,
                    target,
                    dloss,
                    lse_q,
                    ignore_index=int(ignore_index),
                    inplace_backward=False,
                )

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
    p.add_argument("--ignore-index", type=int, default=-100)
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
        "--quack-suite",
        action="store_true",
        help="Run Quack-style batch/seq grid (vocab=4096)",
    )
    p.add_argument(
        "--dsv3",
        action="store_true",
        help="Run DSv3 set: M in {4096,16384,65536}, N in {3072,6144,8192,12288}",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip correctness checks (Oink/Quack vs PyTorch float32-logits cross entropy)",
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
            ignore_index=int(args.ignore_index),
        )
        row: Dict[str, Any] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "mode": args.mode,
            "ignore_index": int(args.ignore_index),
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
                "io_model_bytes": "mode-dependent; see bytes_io_model_ce in script",
            },
        )

    headers = ["M", "N", "mode", "ours_ms", "ours_tbps"]
    if quack_ce_fwd is not None and quack_ce_bwd is not None:
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
