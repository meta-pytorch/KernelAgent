from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from bench_utils import collect_device_meta, detect_hbm_peak_gbps, write_csv, write_json


def _bench_oink_smallm_noweight(M: int, N: int) -> float:
    import sys

    from triton.testing import do_bench_cudagraph

    repo_src = Path(__file__).resolve().parents[2] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from kernelagent_oink.blackwell import _rmsnorm_impl as impl

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    return float(
        do_bench_cudagraph(
            lambda: impl._rmsnorm_forward_ptr_into(
                x, None, None, None, out, None, None, 1e-6
            ),
            rep=100,
            return_mode="mean",
        )
    )


def bytes_io_model_fwd(M: int, N: int, dtype: torch.dtype) -> int:
    elem = torch.tensor(0, dtype=dtype).element_size()
    return int(2 * M * N * elem)


def _cuda_13_nvcc() -> Path:
    nvcc = Path("/usr/local/cuda-13.0/bin/nvcc")
    if not nvcc.is_file():
        raise FileNotFoundError(f"CUDA 13.0 nvcc not found at {nvcc}")
    return nvcc


def _build_paulius_binary(src_dir: Path) -> Path:
    nvcc = _cuda_13_nvcc()
    out = src_dir / "r.out"
    cmd = [
        str(nvcc),
        "-arch=sm_100",
        "-Xptxas",
        "-v",
        "-O3",
        "RmsNorm.cu",
        "-I../../../",
        "-o",
        str(out),
        "-lnvidia-ml",
    ]
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda-13.0"
    env["PATH"] = f"/usr/local/cuda-13.0/bin:{env.get('PATH', '')}"
    subprocess.run(
        cmd,
        cwd=src_dir,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def _parse_paulius_output(text: str) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    pattern = re.compile(r"BF16\s+\d+:\s+([0-9.eE+-]+)\s+ms\s+([0-9.eE+-]+)\s+GB/s")
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        rows.append((float(match.group(1)), float(match.group(2))))
    return rows


def _run_paulius(
    binary: Path,
    *,
    M: int,
    N: int,
    cta_dim_y: int,
    warmup_reps: int,
    timing_reps: int,
    gpu_id: int,
) -> Tuple[float, float, Dict[str, Any]]:
    cmd = [
        str(binary),
        str(M),
        str(N),
        str(cta_dim_y),
        str(warmup_reps),
        str(timing_reps),
        str(gpu_id),
        "0",
        "5",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        cwd=binary.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    parsed = _parse_paulius_output(proc.stdout)
    if not parsed:
        raise RuntimeError(
            f"Failed to parse Paulius output:\n{proc.stdout}\n{proc.stderr}"
        )
    ms, gbps = min(parsed, key=lambda row: row[0])
    return ms, gbps, {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}


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
        "--paulius-dir",
        type=str,
        default=os.path.expanduser("~/fbsource/fbcode/scripts/paulius/rmsnorm"),
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--warmup-reps", type=int, default=10)
    p.add_argument("--timing-reps", type=int, default=100)
    p.add_argument("--configs", type=str, default="4096x4096,65536x4096")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    src_dir = Path(args.paulius_dir)
    binary = _build_paulius_binary(src_dir)

    cfgs: List[Tuple[int, int]] = []
    for part in args.configs.split(","):
        m, n = part.lower().split("x")
        cfgs.append((int(m), int(n)))

    meta = collect_device_meta(device)
    hbm_peak = detect_hbm_peak_gbps(device)
    rows_out: List[Dict[str, Any]] = []
    for M, N in cfgs:
        if N != 4096:
            raise SystemExit("Paulius benchmark only supports N=4096")
        best_ms = float("inf")
        best_gbps = 0.0
        best_cta_dim_y = -1
        debug_runs: List[Dict[str, Any]] = []
        for cta_dim_y in (1, 2, 4, 8):
            ms, gbps, debug = _run_paulius(
                binary,
                M=M,
                N=N,
                cta_dim_y=cta_dim_y,
                warmup_reps=int(args.warmup_reps),
                timing_reps=int(args.timing_reps),
                gpu_id=int(args.gpu_id),
            )
            debug_runs.append({"cta_dim_y": cta_dim_y, "ms": ms, "gbps": gbps})
            if ms < best_ms:
                best_ms = ms
                best_gbps = gbps
                best_cta_dim_y = cta_dim_y
        row: Dict[str, Any] = {
            "M": M,
            "N": N,
            "dtype": "bf16",
            "paulius_ms": best_ms,
            "paulius_gbps": best_gbps,
            "paulius_tbps": best_gbps / 1000.0,
            "paulius_hbm_frac": best_gbps / hbm_peak,
            "best_cta_dim_y": best_cta_dim_y,
            "io_model_bytes": bytes_io_model_fwd(M, N, torch.bfloat16),
            "cta_dim_y_candidates": debug_runs,
        }
        if M == 4096:
            oink_ms = _bench_oink_smallm_noweight(M, N)
            oink_gbps = (
                bytes_io_model_fwd(M, N, torch.bfloat16) / (oink_ms * 1e-3) / 1e9
            )
            row.update(
                {
                    "oink_kernel_ms": oink_ms,
                    "oink_kernel_tbps": oink_gbps / 1000.0,
                    "oink_speedup_vs_paulius": best_ms / oink_ms,
                }
            )
        rows_out.append(row)

    if args.csv is not None:
        write_csv(args.csv, rows_out)
    if args.json is not None:
        write_json(
            args.json,
            meta,
            rows_out,
            extra={
                "method": "Paulius CUDA benchmark binary",
                "warmup_reps": int(args.warmup_reps),
                "timing_reps": int(args.timing_reps),
                "paulius_dir": str(src_dir),
            },
        )

    print("\nSummary:")
    print(
        f"{'M':>14} {'N':>14} {'paulius_ms':>14} {'paulius_tbps':>14}"
        f" {'ctaDimY':>14} {'oink_ms':>14} {'oink/paulius':>14}"
    )
    for r in rows_out:
        oink_ms = float(r.get("oink_kernel_ms", float("nan")))
        speedup = float(r.get("oink_speedup_vs_paulius", float("nan")))
        print(
            f"{int(r['M']):>14} {int(r['N']):>14} {float(r['paulius_ms']):14.4f}"
            f" {float(r['paulius_tbps']):14.4f} {int(r['best_cta_dim_y']):>14}"
            f" {oink_ms:14.4f} {speedup:14.4f}"
        )


if __name__ == "__main__":
    main()
