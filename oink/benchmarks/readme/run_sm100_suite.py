from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write JSON outputs (default: /tmp/kernelagent_oink_sm100_suite_<timestamp>)",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip correctness checks (Oink/Quack vs PyTorch / pure-PyTorch references)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    args = p.parse_args()

    # Standardize env for standalone runs outside the vLLM plugin.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

    out_dir = args.out_dir or f"/tmp/kernelagent_oink_sm100_suite_{_ts()}"
    os.makedirs(out_dir, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.abspath(os.path.join(here, "..", "benchmark"))
    py = sys.executable

    def script(name: str) -> str:
        return os.path.join(bench_dir, name)

    common = ["--dtype", args.dtype]
    if args.skip_verify:
        common = [*common, "--skip-verify"]

    runs: List[Tuple[str, List[str]]] = [
        (
            "rmsnorm_fwd_quack_suite_wfp32",
            [
                py,
                script("benchmark_rmsnorm_sm100.py"),
                *common,
                "--weight-dtype",
                "fp32",
                "--quack-suite",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_fwd_quack_suite_wfp32.json"),
            ],
        ),
        (
            "rmsnorm_fwd_dsv3_wfp32",
            [
                py,
                script("benchmark_rmsnorm_sm100.py"),
                *common,
                "--weight-dtype",
                "fp32",
                "--dsv3",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_fwd_dsv3_wfp32.json"),
            ],
        ),
        (
            "rmsnorm_bwd_quack_suite_wfp32",
            [
                py,
                script("benchmark_rmsnorm_bwd_sm100.py"),
                *common,
                "--weight-dtype",
                "fp32",
                "--quack-suite",
                "--iters",
                "100",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_bwd_quack_suite_wfp32.json"),
            ],
        ),
        (
            "rmsnorm_bwd_dsv3_wfp32",
            [
                py,
                script("benchmark_rmsnorm_bwd_sm100.py"),
                *common,
                "--weight-dtype",
                "fp32",
                "--dsv3",
                "--iters",
                "100",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_bwd_dsv3_wfp32.json"),
            ],
        ),
        # vLLM inference-style RMSNorm (weight dtype == activation dtype).
        (
            "rmsnorm_fwd_quack_suite_wsame",
            [
                py,
                script("benchmark_rmsnorm_sm100.py"),
                *common,
                "--weight-dtype",
                "same",
                "--quack-suite",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_fwd_quack_suite_wsame.json"),
            ],
        ),
        (
            "rmsnorm_fwd_dsv3_wsame",
            [
                py,
                script("benchmark_rmsnorm_sm100.py"),
                *common,
                "--weight-dtype",
                "same",
                "--dsv3",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_fwd_dsv3_wsame.json"),
            ],
        ),
        (
            "rmsnorm_bwd_quack_suite_wsame",
            [
                py,
                script("benchmark_rmsnorm_bwd_sm100.py"),
                *common,
                "--weight-dtype",
                "same",
                "--quack-suite",
                "--iters",
                "100",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_bwd_quack_suite_wsame.json"),
            ],
        ),
        (
            "rmsnorm_bwd_dsv3_wsame",
            [
                py,
                script("benchmark_rmsnorm_bwd_sm100.py"),
                *common,
                "--weight-dtype",
                "same",
                "--dsv3",
                "--iters",
                "100",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "rmsnorm_bwd_dsv3_wsame.json"),
            ],
        ),
        (
            "softmax_fwd_bwd_quack_suite",
            [
                py,
                script("benchmark_softmax_sm100.py"),
                *common,
                "--mode",
                "fwd_bwd",
                "--quack-suite",
                "--iters",
                "50",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "softmax_fwd_bwd_quack_suite.json"),
            ],
        ),
        (
            "softmax_fwd_bwd_dsv3",
            [
                py,
                script("benchmark_softmax_sm100.py"),
                *common,
                "--mode",
                "fwd_bwd",
                "--dsv3",
                "--iters",
                "50",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "softmax_fwd_bwd_dsv3.json"),
            ],
        ),
        (
            "cross_entropy_fwd_bwd_quack_suite",
            [
                py,
                script("benchmark_cross_entropy_sm100.py"),
                *common,
                "--mode",
                "fwd_bwd",
                "--quack-suite",
                "--iters",
                "50",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "cross_entropy_fwd_bwd_quack_suite.json"),
            ],
        ),
        (
            "cross_entropy_fwd_bwd_dsv3",
            [
                py,
                script("benchmark_cross_entropy_sm100.py"),
                *common,
                "--mode",
                "fwd_bwd",
                "--dsv3",
                "--iters",
                "50",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "cross_entropy_fwd_bwd_dsv3.json"),
            ],
        ),
        (
            "layernorm_fwd_quack_suite",
            [
                py,
                script("benchmark_layernorm_sm100.py"),
                *common,
                "--quack-suite",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "layernorm_fwd_quack_suite.json"),
            ],
        ),
        (
            "layernorm_fwd_dsv3",
            [
                py,
                script("benchmark_layernorm_sm100.py"),
                *common,
                "--dsv3",
                "--iters",
                "200",
                "--warmup-ms",
                "25",
                "--json",
                os.path.join(out_dir, "layernorm_fwd_dsv3.json"),
            ],
        ),
    ]

    print(f"Writing results to: {out_dir}", flush=True)
    for name, cmd in runs:
        print(f"\n== {name} ==", flush=True)
        _run(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
