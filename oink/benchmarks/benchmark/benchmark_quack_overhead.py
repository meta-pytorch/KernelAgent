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
Overhead analysis for the quack RMSNorm PR (pytorch#178326).

Measures aten vs quack through the same ``torch.ops.aten._fused_rms_norm``
API at various shapes. Quack is registered via ``torch._native`` (the PR
pattern). Run with ``--mode=aten`` or ``--mode=quack`` in separate processes
to avoid cross-contamination.

Usage::

    bash oink/benchmarks/benchmark/run_benchmark_quack_overhead.sh
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCH_NATIVE_SKIP_VERSION_CHECK", "1")

import torch
from triton.testing import do_bench

# Comprehensive shape grid: small → large M, production N values.
SHAPES = [
    # Small M (dispatch overhead dominates)
    (1, 4096),
    (1, 8192),
    (32, 4096),
    (32, 8192),
    (256, 4096),
    (256, 8192),
    # Medium M (crossover region)
    (1024, 4096),
    (1024, 8192),
    (4096, 4096),
    (4096, 8192),
    # Large M (kernel compute dominates)
    (16384, 4096),
    (16384, 8192),
    (65536, 4096),
    (65536, 8192),
]
DTYPE = torch.bfloat16


def bench(fn, warmup=50, rep=200):
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["aten", "quack"], required=True)
    args = p.parse_args()

    # Warm up (triggers JIT compilation for quack).
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

        def fn_fwd(x=x, w=w, N=N):
            return torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)

        fwd_ms = bench(fn_fwd)

        x_ = x.detach().requires_grad_(True)
        w_ = w.detach().requires_grad_(True)

        def fn_fwdbwd(x_=x_, w_=w_, N=N, grad=grad):
            y, _ = torch.ops.aten._fused_rms_norm(x_, [N], w_, 1e-5)
            y.backward(grad)

        fwdbwd_ms = bench(fn_fwdbwd)
        results[f"{M}x{N}"] = {"fwd": fwd_ms, "fwdbwd": fwdbwd_ms}

    print(json.dumps({"mode": args.mode, "results": results}))


if __name__ == "__main__":
    main()
