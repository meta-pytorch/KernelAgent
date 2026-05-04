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
import importlib
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from bench_utils import (
    ErrorStatsAccumulator,
    collect_device_meta,
    detect_hbm_peak_gbps,
    do_bench_cuda_graph,
    do_bench_triton,
    dsv4_hidden_norm_configs,
    ensure_blackwell_arch_env,
    ensure_oink_src_on_path,
    error_stats_to_row,
    iter_row_blocks,
    parse_configs,
    parse_dtype,
    quack_suite_configs,
    write_csv,
    write_json,
)

# Reduce fragmentation pressure on busy GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ensure_blackwell_arch_env()
ensure_oink_src_on_path()

_oink_ln: ModuleType | None = None
quack_layernorm_bwd: Callable[..., Any] | None = None
_QUACK_LAYERNORM_BWD_STATUS = "uninitialized"


def _load_optional_quack_layernorm_bwd() -> None:
    global quack_layernorm_bwd, _QUACK_LAYERNORM_BWD_STATUS
    try:
        module = importlib.import_module("quack.rmsnorm")
        quack_layernorm_bwd = getattr(module, "layernorm_bwd")
        _QUACK_LAYERNORM_BWD_STATUS = "available: quack.rmsnorm.layernorm_bwd"
    except Exception as e:
        quack_layernorm_bwd = None
        _QUACK_LAYERNORM_BWD_STATUS = f"unavailable: {type(e).__name__}: {e}"


_load_optional_quack_layernorm_bwd()


def _get_oink_layernorm() -> ModuleType:
    global _oink_ln
    if _oink_ln is None:
        _oink_ln = importlib.import_module("kernelagent_oink.blackwell.layernorm")
    return _oink_ln


_VERIFY_TOL_DX = {
    # Match the existing Oink backward benchmark tolerance style.
    torch.float32: dict(atol=1e-4, rtol=1e-3),
    torch.float16: dict(atol=1e-2, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-2),
}


@dataclass(frozen=True)
class BenchResult:
    ms: float
    gbps: float

    @property
    def tbps(self) -> float:
        return self.gbps / 1000.0


BackendFn = Callable[
    [], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
]


def dsv3_configs() -> List[Tuple[int, int]]:
    # CuteDSL kernel workflow default for RMSNorm/LayerNorm when shapes are unspecified.
    Ms = [4096, 16384, 65536]
    Ns = [6144, 8192]
    return [(m, n) for m in Ms for n in Ns]


def _call_quack_layernorm_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if quack_layernorm_bwd is None:
        raise RuntimeError(_QUACK_LAYERNORM_BWD_STATUS)
    if bias is not None:
        raise RuntimeError("Quack LayerNorm backward bias path is not benchmarked")

    # Quack has changed public naming across releases. Prefer the common
    # cute-kernels-style positional API, then try keyword spellings used by
    # adjacent norm wrappers.
    call_errors: list[str] = []
    for args, kwargs in (
        ((dout, x, weight, mean, rstd), {}),
        ((), {"dout": dout, "x": x, "weight": weight, "mean": mean, "rstd": rstd}),
        ((), {"dy": dout, "x": x, "w": weight, "mean": mean, "rstd": rstd}),
    ):
        try:
            out = quack_layernorm_bwd(*args, **kwargs)
            break
        except TypeError as e:
            call_errors.append(str(e))
    else:
        raise TypeError(
            "Unable to call Quack LayerNorm backward: " + " | ".join(call_errors)
        )

    if not isinstance(out, tuple):
        raise TypeError(
            f"Expected Quack LayerNorm backward to return a tuple, got {type(out)}"
        )
    if len(out) == 2:
        dx, dw = out
        db = None
    elif len(out) >= 3:
        dx, dw, db = out[:3]
    else:
        raise TypeError(
            f"Expected Quack LayerNorm backward tuple with >=2 values, got {len(out)}"
        )
    return dx, dw, None if bias is None else db


def parse_weight_dtype(arg: str, activation_dtype: torch.dtype) -> torch.dtype:
    if arg == "same":
        return activation_dtype
    return parse_dtype(arg)


def bytes_io_model_layernorm_bwd(
    M: int,
    N: int,
    dtype: torch.dtype,
    *,
    weight_dtype: torch.dtype,
    has_bias: bool,
) -> int:
    """Useful logical IO model for LayerNorm backward.

    The model intentionally excludes implementation-specific partial-gradient
    scratch traffic so Oink and PyTorch can be compared on the same useful
    read/write work.
    """
    elem = torch.tensor(0, dtype=dtype).element_size()
    w_elem = torch.tensor(0, dtype=weight_dtype).element_size()

    # Read x + dout, write dx.
    total = 3 * M * N * elem
    # Read gamma, write dgamma.
    total += 2 * N * w_elem
    # Read mean + rstd (fp32 per row).
    total += 2 * M * 4
    if has_bias:
        # Write dbias. Bias reads are not needed for LayerNorm backward.
        total += N * w_elem
    return int(total)


def _compute_stats(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    mean = xf.mean(dim=-1).to(torch.float32)
    var = ((xf - mean.unsqueeze(1)) ** 2).mean(dim=-1)
    rstd = torch.rsqrt(var + eps).to(torch.float32)
    return mean, rstd


def _call_oink(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    return _get_oink_layernorm().layernorm_backward(
        dout, x, weight, rstd, mean, bias=bias
    )


def _call_ref(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Native ATen backward is the fastest available PyTorch reference that can
    # reuse the same precomputed mean/rstd as the Oink/cute backends.
    return torch.ops.aten.native_layer_norm_backward.default(
        dout,
        x,
        [int(x.shape[-1])],
        mean,
        rstd,
        weight,
        bias,
        [True, True, bias is not None],
    )


def _available_backend_fns(
    dout: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Dict[str, BackendFn]:
    fns: Dict[str, BackendFn] = {
        "ours": lambda: _call_oink(dout, x, weight, mean, rstd, bias),
        "ref": lambda: _call_ref(dout, x, weight, mean, rstd, bias),
    }
    if quack_layernorm_bwd is not None and bias is None:
        fns["quack"] = lambda: _call_quack_layernorm_bwd(
            dout, x, weight, mean, rstd, bias
        )
    return fns


def _dweight_tolerance(
    dtype: torch.dtype, dw_ref: torch.Tensor
) -> Optional[Dict[str, float]]:
    if dtype == torch.float32:
        return dict(atol=2e-3, rtol=1e-3)
    dw_ref_f32 = dw_ref.to(torch.float32)
    scale = float(dw_ref_f32.abs().max().item())
    atol = max(2.0 * torch.finfo(dtype).eps * scale, 1e-3)
    return dict(atol=float(atol), rtol=1e-3)


def _unpack_backend_output(
    out: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(out, tuple) or len(out) < 2:
        raise TypeError(
            f"Expected a tuple containing at least dx and dweight, got {type(out)}"
        )
    dx = out[0]
    dw = out[1]
    db = out[2] if len(out) > 2 else None
    return dx, dw, db


def _verify_parity(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    backend_fns: Dict[str, BackendFn],
) -> Dict[str, object]:
    tol_dx = _VERIFY_TOL_DX[x.dtype]
    M, N = int(x.shape[0]), int(x.shape[1])
    ref_block_rows = 1024

    outputs: Dict[
        str, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ] = {}
    with torch.no_grad():
        for name, fn in backend_fns.items():
            outputs[name] = _unpack_backend_output(fn())
    torch.cuda.synchronize()

    dx_accs = {
        name: ErrorStatsAccumulator(total_elems=M * N) for name in outputs.keys()
    }
    dw_accum = torch.zeros((N,), device=x.device, dtype=torch.float32)
    db_accum = (
        torch.zeros((N,), device=x.device, dtype=torch.float32)
        if bias is not None
        else None
    )
    weight_f32 = weight.float()

    for start, end in iter_row_blocks(M, ref_block_rows):
        x_f32 = x[start:end].float()
        dout_f32 = dout[start:end].float()
        mean_blk = mean[start:end].unsqueeze(1)
        rstd_blk = rstd[start:end].unsqueeze(1)
        x_hat = (x_f32 - mean_blk) * rstd_blk
        wdy = dout_f32 * weight_f32
        mean_wdy = wdy.mean(dim=-1, keepdim=True)
        mean_xhat_wdy = (x_hat * wdy).mean(dim=-1, keepdim=True)
        dx_ref = ((wdy - mean_wdy - x_hat * mean_xhat_wdy) * rstd_blk).to(x.dtype)

        for name, (dx, _, _) in outputs.items():
            torch.testing.assert_close(
                dx[start:end],
                dx_ref,
                **tol_dx,
                msg=f"{name} dx mismatch M={M} N={N} rows={start}:{end}",
            )
            dx_accs[name].update(dx[start:end], dx_ref)

        dw_accum += (dout_f32 * x_hat).sum(dim=0)
        if db_accum is not None:
            db_accum += dout_f32.sum(dim=0)

    stats: Dict[str, object] = {}
    for name, acc in dx_accs.items():
        stats.update(error_stats_to_row(f"{name}_err_dx", acc.finalize()))

    dw_ref = dw_accum.to(weight.dtype)
    dw_tol = _dweight_tolerance(weight.dtype, dw_ref)
    for name, (_, dw, _) in outputs.items():
        if dw is None:
            raise AssertionError(f"{name} did not return dweight for M={M} N={N}")
        torch.testing.assert_close(
            dw,
            dw_ref,
            **dw_tol,
            msg=f"{name} dweight mismatch M={M} N={N}",
        )
        dw_acc = ErrorStatsAccumulator(
            total_elems=int(dw_ref.numel()), p99_target_samples=int(dw_ref.numel())
        )
        dw_acc.update(dw, dw_ref)
        stats.update(error_stats_to_row(f"{name}_err_dw", dw_acc.finalize()))

    if bias is not None:
        assert db_accum is not None
        db_ref = db_accum.to(bias.dtype)
        db_tol = _dweight_tolerance(bias.dtype, db_ref)
        for name, (_, _, db) in outputs.items():
            if db is None:
                raise AssertionError(f"{name} did not return dbias for M={M} N={N}")
            torch.testing.assert_close(
                db,
                db_ref,
                **db_tol,
                msg=f"{name} dbias mismatch M={M} N={N}",
            )
            db_acc = ErrorStatsAccumulator(
                total_elems=int(db_ref.numel()), p99_target_samples=int(db_ref.numel())
            )
            db_acc.update(db, db_ref)
            stats.update(error_stats_to_row(f"{name}_err_db", db_acc.finalize()))

    return stats


def bench_single(
    M: int,
    N: int,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    *,
    eps: float,
    warmup_ms: int,
    iters_ms: int,
    verify: bool,
    has_bias: bool,
    cuda_graph: bool,
) -> Tuple[Dict[str, BenchResult], Dict[str, object]]:
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=weight_dtype)
    bias = torch.randn(N, device=device, dtype=weight_dtype) if has_bias else None
    dout = torch.randn(M, N, device=device, dtype=dtype)
    mean, rstd = _compute_stats(x, eps)

    backend_fns = _available_backend_fns(dout, x, weight, mean, rstd, bias)
    stats: Dict[str, object] = {}
    if verify:
        stats = _verify_parity(
            x=x,
            weight=weight,
            bias=bias,
            dout=dout,
            mean=mean,
            rstd=rstd,
            backend_fns=backend_fns,
        )

    bytes_io = bytes_io_model_layernorm_bwd(
        M, N, dtype, weight_dtype=weight_dtype, has_bias=has_bias
    )
    results: Dict[str, BenchResult] = {}
    for name, fn in backend_fns.items():
        if cuda_graph:
            # Warm outside graph so CuTeDSL compile/cache and workspace allocation are
            # not captured in the measured replay. Keep the capture-time outputs
            # alive until after replay timing; this avoids tearing down graph-owned
            # allocations while the captured graph is still being measured.
            graph_outputs: list[object] = [None]

            def graph_fn() -> object:
                graph_outputs[0] = fn()
                return graph_outputs[0]

            graph_fn()
            torch.cuda.synchronize()
            ms = do_bench_cuda_graph(graph_fn, rep_ms=iters_ms)
        else:
            ms = do_bench_triton(fn, warmup_ms=warmup_ms, rep_ms=iters_ms)
        gbps = bytes_io / (ms * 1e-3) / 1e9
        results[name] = BenchResult(ms=ms, gbps=gbps)

    return results, stats


def _add_backend_result(row: Dict[str, object], name: str, result: BenchResult) -> None:
    prefix = "ours" if name == "ours" else name
    row[f"{prefix}_ms"] = result.ms
    row[f"{prefix}_gbps"] = result.gbps
    row[f"{prefix}_tbps"] = result.tbps


def _append_speedups(row: Dict[str, object], results: Dict[str, BenchResult]) -> None:
    ours = results.get("ours")
    if ours is None:
        return
    for name, result in results.items():
        if name == "ours":
            continue
        row[f"speedup_vs_{name}"] = result.ms / ours.ms


def _print_summary(rows: List[Dict[str, object]]) -> None:
    base_headers = ["M", "N", "dtype", "weight_dtype", "ours_ms", "ours_tbps"]
    optional_headers = [
        "ref_ms",
        "ref_tbps",
        "speedup_vs_ref",
        "quack_ms",
        "quack_tbps",
        "speedup_vs_quack",
    ]
    headers = base_headers + [h for h in optional_headers if any(h in r for r in rows)]

    print("\nSummary:")
    print(" ".join(h.rjust(22) for h in headers))
    for row in rows:
        parts: List[str] = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                parts.append(f"{value:22.4f}")
            else:
                parts.append(f"{str(value):>22}")
        print(" ".join(parts))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    print(f"Running on {torch.cuda.get_device_name(device)} (SM{sm})")
    print(f"Quack LayerNorm backward: {_QUACK_LAYERNORM_BWD_STATUS}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default="same",
        choices=["same", "fp16", "bf16", "fp32"],
        help="LayerNorm weight dtype. `same` matches activation dtype.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--with-bias", action="store_true")
    parser.add_argument(
        "--iters", type=int, default=100, help="Triton do_bench rep_ms (kernel-only)."
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument(
        "--csv", type=str, default=None, help="Optional CSV output path; appends rows"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Optional JSON output path (meta + rows)"
    )
    parser.add_argument("--configs", type=str, default="1024x4096,8192x4096")
    parser.add_argument(
        "--quack-suite", action="store_true", help="Run Quack-style batch/seq grid"
    )
    parser.add_argument(
        "--dsv3",
        action="store_true",
        help="Run DSv3 set: M in {4096,16384,65536}, N in {6144,8192}",
    )
    parser.add_argument(
        "--dsv4",
        action="store_true",
        help="Run DeepSeek-V4-Flash hidden LayerNorm set: M in {4096,16384,65536}, N=7168",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip correctness checks before timing",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Time warm CUDA-graph replay instead of eager do_bench calls.",
    )
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    weight_dtype = parse_weight_dtype(args.weight_dtype, dtype)
    eps = float(args.eps)

    if args.quack_suite:
        cfgs = [(bs * sl, hidden) for (bs, sl, hidden) in quack_suite_configs()]
    elif args.dsv3:
        cfgs = dsv3_configs()
    elif args.dsv4:
        cfgs = dsv4_hidden_norm_configs()
    else:
        cfgs = parse_configs(args.configs)

    hbm_peak = detect_hbm_peak_gbps(device)
    rows_out: List[Dict[str, object]] = []
    for M, N in cfgs:
        print(
            f"bench M={M:<8d} N={N:<6d} dtype={args.dtype} "
            f"weight_dtype={args.weight_dtype} ...",
            flush=True,
        )
        results, stats = bench_single(
            M=M,
            N=N,
            dtype=dtype,
            weight_dtype=weight_dtype,
            eps=eps,
            warmup_ms=int(args.warmup_ms),
            iters_ms=int(args.iters),
            verify=not args.skip_verify,
            has_bias=bool(args.with_bias),
            cuda_graph=bool(args.cuda_graph),
        )

        row: Dict[str, object] = {
            "M": M,
            "N": N,
            "dtype": args.dtype,
            "weight_dtype": args.weight_dtype,
            "eps": eps,
            "with_bias": bool(args.with_bias),
            "bytes_io": bytes_io_model_layernorm_bwd(
                M, N, dtype, weight_dtype=weight_dtype, has_bias=bool(args.with_bias)
            ),
        }
        for name, result in results.items():
            _add_backend_result(row, name, result)
        if "ours" in results:
            row["ours_hbm_frac"] = results["ours"].gbps / hbm_peak
        _append_speedups(row, results)
        row.update(stats)
        rows_out.append(row)

    if args.csv is not None:
        write_csv(args.csv, rows_out)
    if args.json is not None:
        meta = collect_device_meta(device)
        write_json(
            args.json,
            meta,
            rows_out,
            extra={
                "method": (
                    "triton.testing.do_bench_cudagraph(mean)"
                    if args.cuda_graph
                    else "triton.testing.do_bench(mean)"
                ),
                "cuda_graph": bool(args.cuda_graph),
                "warmup_ms": int(args.warmup_ms),
                "rep_ms": int(args.iters),
                "io_model_bytes": "see bytes_io_model_layernorm_bwd in script",
                "quack_layernorm_bwd_status": _QUACK_LAYERNORM_BWD_STATUS,
                "reference_backend": "torch.ops.aten.native_layer_norm_backward.default",
            },
        )

    _print_summary(rows_out)

    if args.cuda_graph:
        # Some torch/CUDAGraph allocator combinations can segfault during Python
        # finalization after captured allocation-heavy benchmark functions have
        # already written valid results. Exit directly after flushing benchmark
        # output so graph replay CLI runs return success deterministically.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
