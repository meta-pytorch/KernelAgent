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
Benchmark Oink CuTeDSL RMSNorm vs PyTorch Aten RMSNorm (aten::_fused_rms_norm).

Based on benchmark_rmsnorm_sm100.py — replaces quack with torch's native aten
kernel so that oink is compared directly against PyTorch's built-in CUDA
implementation.

Both kernels are called at the same level: direct function call, no aten
override dispatch layer.  This isolates kernel performance from Python/dispatch
overhead.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from bench_utils import (  # noqa: E402
    ErrorStatsAccumulator,
    collect_device_meta,
    detect_hbm_peak_gbps,
    do_bench_triton,
    ensure_blackwell_arch_env,
    error_stats_to_row,
    ensure_oink_src_on_path,
    iter_row_blocks,
    parse_configs,
    parse_dtype,
    quack_suite_configs,
    write_csv,
    write_json,
)

ensure_blackwell_arch_env()
ensure_oink_src_on_path()

from kernelagent_oink.blackwell import rmsnorm as oink_rmsnorm  # noqa: E402

# PyTorch aten _fused_rms_norm — called directly to avoid any override layer.
_aten_fused_rms_norm = torch.ops.aten._fused_rms_norm

_VERIFY_TOL_Y = {
    torch.float32: dict(atol=1e-4, rtol=1e-3),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
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
    total = 2 * M * N * elem  # read x + write y
    total += N * w_elem  # read weight
    return int(total)


def dsv3_configs() -> List[Tuple[int, int]]:
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
    y_acc_aten = ErrorStatsAccumulator(total_elems=M * N)

    with torch.no_grad():
        y_o, rstd_o, res_o = oink_rmsnorm.rmsnorm_forward(
            x, weight=w, bias=None, residual=None, eps=eps, store_rstd=store_rstd,
        )
        y_a, rstd_a = _aten_fused_rms_norm(x, [N], w, eps)

    # Pure-PyTorch reference (float32 accumulation), chunked.
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
        torch.testing.assert_close(y_a[start:end], y_ref_blk, **tol_y)
        y_acc_aten.update(y_a[start:end], y_ref_blk)

    stats: dict[str, object] = {}
    stats.update(error_stats_to_row("ours_err_y", y_acc_ours.finalize()))
    stats.update(error_stats_to_row("aten_err_y", y_acc_aten.finalize()))

    if store_rstd:
        assert rstd_o is not None
        torch.testing.assert_close(rstd_o, rstd_ref, **tol_rstd)
        rstd_acc_ours = ErrorStatsAccumulator(
            total_elems=int(rstd_ref.numel()),
            p99_target_samples=int(rstd_ref.numel()),
        )
        rstd_acc_ours.update(rstd_o, rstd_ref)
        stats.update(error_stats_to_row("ours_err_rstd", rstd_acc_ours.finalize()))

    assert res_o is None
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
) -> Tuple[Tuple[float, float], Tuple[float, float], dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=weight_dtype)

    stats: dict[str, object] = {}
    if verify:
        stats = _verify_parity(x, w, eps=eps, store_rstd=store_rstd)

    bytes_io = bytes_io_model_fwd(M, N, dtype, weight_dtype=w.dtype)

    # Oink: call rmsnorm_forward directly (same as benchmark_rmsnorm_sm100.py).
    def fn_oink():
        return oink_rmsnorm.rmsnorm_forward(
            x, weight=w, bias=None, residual=None, eps=eps, store_rstd=store_rstd,
        )

    ms_oink = do_bench_triton(fn_oink, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_oink = bytes_io / (ms_oink * 1e-3) / 1e9

    # Aten: call _fused_rms_norm directly (no Python override layer).
    def fn_aten():
        return _aten_fused_rms_norm(x, [N], w, eps)

    ms_aten = do_bench_triton(fn_aten, warmup_ms=warmup_ms, rep_ms=iters_ms)
    gbps_aten = bytes_io / (ms_aten * 1e-3) / 1e9

    return (ms_oink, gbps_oink), (ms_aten, gbps_aten), stats


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
    p.add_argument(
        "--weight-dtype", type=str, default="fp32",
        choices=["same", "fp16", "bf16", "fp32"],
    )
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--store-rstd", action="store_true")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--configs", type=str, default="1024x4096,8192x4096")
    p.add_argument("--quack-suite", action="store_true")
    p.add_argument("--dsv3", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    weight_dtype = dtype if args.weight_dtype == "same" else parse_dtype(args.weight_dtype)
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
        (ms_oink, gbps_oink), (ms_aten, gbps_aten), stats = bench_single(
            M=M, N=N, dtype=dtype, weight_dtype=weight_dtype, eps=eps,
            warmup_ms=int(args.warmup_ms), iters_ms=int(args.iters),
            verify=not args.skip_verify, store_rstd=bool(args.store_rstd),
        )
        row: Dict[str, Any] = {
            "M": M, "N": N, "dtype": args.dtype,
            "weight_dtype": args.weight_dtype, "eps": eps,
            "store_rstd": bool(args.store_rstd),
            "oink_ms": ms_oink, "oink_gbps": gbps_oink,
            "oink_tbps": gbps_oink / 1000.0,
            "oink_hbm_frac": gbps_oink / hbm_peak,
            "aten_ms": ms_aten, "aten_gbps": gbps_aten,
            "aten_tbps": gbps_aten / 1000.0,
            "speedup_vs_aten": ms_aten / ms_oink,
        }
        row.update(stats)
        rows_out.append(row)

    if args.csv is not None:
        write_csv(args.csv, rows_out)
    if args.json is not None:
        write_json(args.json, meta, rows_out, extra={
            "method": "triton.testing.do_bench(mean)",
            "warmup_ms": int(args.warmup_ms), "rep_ms": int(args.iters),
            "io_model_bytes": "(2*M*N)*elem_size + N*weight_elem_size",
            "store_rstd": bool(args.store_rstd),
            "weight_dtype": str(args.weight_dtype),
        })

    # Compact summary table.
    headers = ["M", "N", "oink_ms", "oink_tbps", "aten_ms", "aten_tbps", "speedup_vs_aten"]
    print("\nSummary:")
    print(" ".join(h.rjust(16) for h in headers))
    for r in rows_out:
        parts: List[str] = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                parts.append(f"{v:16.4f}")
            else:
                parts.append(f"{str(v):>16}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
