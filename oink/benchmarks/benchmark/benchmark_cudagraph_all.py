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
CUDA graph benchmark: aten vs quack vs oink RMSNorm.

All calls go through torch.ops.aten._fused_rms_norm. CUDA graphs
eliminate Python dispatch overhead, isolating pure kernel performance.

Usage::

    bash oink/benchmarks/benchmark/run_benchmark_cudagraph_all.sh
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCH_NATIVE_SKIP_VERSION_CHECK", "1")

import torch
from triton.testing import do_bench

SHAPES = [
    (1, 4096),
    (1, 8192),
    (32, 4096),
    (32, 8192),
    (256, 4096),
    (256, 8192),
    (1024, 4096),
    (1024, 8192),
    (4096, 4096),
    (4096, 8192),
    (16384, 4096),
    (16384, 8192),
    (65536, 4096),
    (65536, 8192),
]
DTYPE = torch.bfloat16


def bench_cudagraph(fn, warmup=50, rep=200):
    """Capture fn into a CUDA graph, then benchmark replay."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    return do_bench(lambda: g.replay(), warmup=10, rep=rep, return_mode="median")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["aten", "quack", "oink"], required=True)
    args = p.parse_args()

    if args.mode == "oink":
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
        x = torch.randn(M, N, dtype=DTYPE, device="cuda")
        w = torch.randn(N, dtype=DTYPE, device="cuda")

        def fn_fwd(x=x, w=w, N=N):
            return torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)

        try:
            fwd_ms = bench_cudagraph(fn_fwd)
        except Exception:
            fwd_ms = -1.0

        results[f"{M}x{N}"] = {"fwd": fwd_ms}

    print(json.dumps({"mode": args.mode, "results": results}))


if __name__ == "__main__":
    main()
