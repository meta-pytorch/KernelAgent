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
Benchmark aten vs quack vs oink RMSNorm: normal dispatch + CUDA graph.

All calls go through ``torch.ops.aten._fused_rms_norm``.
Quack is registered via ``torch._native`` (quack PR pattern).
Oink is registered via ``kernelagent_oink.register_all_kernels()``.

Produces four tables:
  - Forward (normal dispatch)
  - Forward + Backward (normal dispatch)
  - Forward (CUDA graph)
  - Forward + Backward (CUDA graph)

Usage::

    python oink/benchmarks/benchmark/benchmark_rmsnorm_all.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("TORCH_NATIVE_SKIP_VERSION_CHECK", "1")


# ---------------------------------------------------------------------------
# Worker code: runs in a subprocess per mode to avoid cross-contamination.
# ---------------------------------------------------------------------------

WORKER_CODE = r"""
import json, os, sys
os.environ.setdefault("TORCH_NATIVE_SKIP_VERSION_CHECK", "1")

import torch
from triton.testing import do_bench

DTYPE = torch.bfloat16

def bench_normal(fn, warmup=50, rep=200):
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="median")

def bench_cudagraph(fn, warmup=50, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return do_bench(lambda: g.replay(), warmup=10, rep=rep, return_mode="median")

mode = sys.argv[1]
shapes_json = sys.argv[2]
SHAPES = json.loads(shapes_json)

if mode == "oink":
    import kernelagent_oink
    kernelagent_oink.register_all_kernels(force=True)

# Warm up
for M, N in SHAPES:
    x = torch.randn(M, N, dtype=DTYPE, device="cuda")
    w = torch.randn(N, dtype=DTYPE, device="cuda")
    torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)
torch.cuda.synchronize()

results = {}
for M, N in SHAPES:
    x = torch.randn(M, N, dtype=DTYPE, device="cuda", requires_grad=True)
    w = torch.randn(N, dtype=DTYPE, device="cuda", requires_grad=True)
    grad = torch.randn(M, N, dtype=DTYPE, device="cuda")

    # Forward (normal)
    def fn_fwd(x=x, w=w, N=N):
        return torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)
    fwd_ms = bench_normal(fn_fwd)

    # Forward + Backward (normal)
    x_ = x.detach().requires_grad_(True)
    w_ = w.detach().requires_grad_(True)
    def fn_fwdbwd(x_=x_, w_=w_, N=N, grad=grad):
        y, _ = torch.ops.aten._fused_rms_norm(x_, [N], w_, 1e-5)
        y.backward(grad)
    fwdbwd_ms = bench_normal(fn_fwdbwd)

    # Forward (CUDA graph)
    x_g = torch.randn(M, N, dtype=DTYPE, device="cuda")
    w_g = torch.randn(N, dtype=DTYPE, device="cuda")
    def fn_fwd_g(x=x_g, w=w_g, N=N):
        return torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)
    try:
        fwd_graph_ms = bench_cudagraph(fn_fwd_g)
    except Exception:
        fwd_graph_ms = -1.0

    # Forward + Backward (CUDA graph)
    x_gb = torch.randn(M, N, dtype=DTYPE, device="cuda", requires_grad=True)
    w_gb = torch.randn(N, dtype=DTYPE, device="cuda", requires_grad=True)
    grad_gb = torch.randn(M, N, dtype=DTYPE, device="cuda")
    def fn_fwdbwd_g(x=x_gb, w=w_gb, N=N, grad=grad_gb):
        y, _ = torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)
        y.backward(grad)
    try:
        fwdbwd_graph_ms = bench_cudagraph(fn_fwdbwd_g)
    except Exception:
        fwdbwd_graph_ms = -1.0

    results[f"{M}x{N}"] = {
        "fwd": fwd_ms,
        "fwdbwd": fwdbwd_ms,
        "fwd_graph": fwd_graph_ms,
        "fwdbwd_graph": fwdbwd_graph_ms,
    }

print(json.dumps({"mode": mode, "results": results}))
"""


# ---------------------------------------------------------------------------
# Main: orchestrates subprocesses and prints tables.
# ---------------------------------------------------------------------------

SHAPES = [
    [1, 4096],
    [1, 8192],
    [32, 4096],
    [32, 8192],
    [256, 4096],
    [256, 8192],
    [1024, 4096],
    [1024, 8192],
    [4096, 4096],
    [4096, 8192],
    [16384, 4096],
    [16384, 8192],
    [65536, 4096],
    [65536, 8192],
]

COL_W = {  # column widths
    "shape": 14,
    "ms": 10,
    "ratio": 8,
}


def find_norm_dir():
    import torch
    from pathlib import Path

    d = Path(torch.__file__).parent / "_native" / "ops" / "norm"
    return str(d) if d.is_dir() else None


def run_mode(mode, norm_dir, shapes):
    init_file = os.path.join(norm_dir, "__init__.py")

    if mode in ("aten", "oink"):
        with open(init_file, "w") as f:
            f.write("")
    elif mode == "quack":
        with open(init_file, "w") as f:
            f.write("from . import rmsnorm_impl  # noqa: F401\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(WORKER_CODE)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path, mode, json.dumps(shapes)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  [{mode}] FAILED: {result.stderr[-300:]}", file=sys.stderr)
            return None
        return json.loads(result.stdout.strip())["results"]
    finally:
        os.unlink(tmp_path)


def _fmt_ms(v):
    return f"{v:>{COL_W['ms']}.4f}" if v > 0 else "FAIL".rjust(COL_W["ms"])


def _fmt_ratio(n, d):
    if d <= 0 or n <= 0:
        return "N/A".rjust(COL_W["ratio"])
    return f"{f'{n / d:.2f}x':>{COL_W['ratio']}}"


def print_table(title, subtitle, aten, quack, oink, key):
    sw, mw, rw = COL_W["shape"], COL_W["ms"], COL_W["ratio"]
    w = [sw, mw, mw, mw, rw, rw, rw]

    def hr(left, mid, right):
        return left + mid.join("─" * (c + 2) for c in w) + right

    hdr = (
        f"│ {'Shape (M,N)':^{sw}} "
        f"│ {'Aten (ms)':^{mw}} "
        f"│ {'Quack (ms)':^{mw}} "
        f"│ {'Oink (ms)':^{mw}} "
        f"│ {'Q/A':^{rw}} "
        f"│ {'O/A':^{rw}} "
        f"│ {'O/Q':^{rw}} │"
    )

    print()
    print(f"  {title}")
    print(f"  {subtitle}")
    print(hr("┌", "┬", "┐"))
    print(hdr)
    print(hr("├", "┼", "┤"))

    for shape in aten:
        M, N = shape.split("x")
        a, q, o = aten[shape][key], quack[shape][key], oink[shape][key]
        row = (
            f"│ {f'({M},{N})':>{sw}} "
            f"│ {_fmt_ms(a)} "
            f"│ {_fmt_ms(q)} "
            f"│ {_fmt_ms(o)} "
            f"│ {_fmt_ratio(a, q)} "
            f"│ {_fmt_ratio(a, o)} "
            f"│ {_fmt_ratio(q, o)} │"
        )
        print(row)

    print(hr("└", "┴", "┘"))


def main():
    import torch

    print("=" * 72)
    print("  RMSNorm Kernel Benchmark: Aten vs Quack vs Oink")
    print("=" * 72)
    print(f"  Device : {torch.cuda.get_device_name(0)}")
    print(f"  Torch  : {torch.__version__}")
    print("  Dtype  : bfloat16")
    print("  Quack  : registered via torch._native (quack PR)")
    print("  Oink   : registered via kernelagent_oink.register_all_kernels()")
    print("  Bench  : triton.testing.do_bench (median, 200 reps)")

    norm_dir = find_norm_dir()
    if norm_dir is None:
        print("ERROR: torch._native/ops/norm/ not found.", file=sys.stderr)
        sys.exit(1)

    print()
    print("Running aten...")
    aten = run_mode("aten", norm_dir, SHAPES)
    print("Running quack...")
    quack = run_mode("quack", norm_dir, SHAPES)
    print("Running oink...")
    oink = run_mode("oink", norm_dir, SHAPES)

    # Restore
    with open(os.path.join(norm_dir, "__init__.py"), "w") as f:
        f.write("from . import rmsnorm_impl  # noqa: F401\n")

    if not all([aten, quack, oink]):
        print("ERROR: one or more modes failed.", file=sys.stderr)
        sys.exit(1)

    print_table(
        "Forward — Normal Dispatch",
        "Standard Python dispatch through torch.ops.aten._fused_rms_norm.",
        aten,
        quack,
        oink,
        "fwd",
    )
    print_table(
        "Forward + Backward — Normal Dispatch",
        "Fwd + autograd backward, standard Python dispatch.",
        aten,
        quack,
        oink,
        "fwdbwd",
    )
    print_table(
        "Forward — CUDA Graph (zero Python overhead)",
        "Kernel captured once, replayed without re-entering Python.",
        aten,
        quack,
        oink,
        "fwd_graph",
    )
    print_table(
        "Forward + Backward — CUDA Graph (zero Python overhead)",
        "Fwd + bwd captured once, replayed without re-entering Python.",
        aten,
        quack,
        oink,
        "fwdbwd_graph",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
