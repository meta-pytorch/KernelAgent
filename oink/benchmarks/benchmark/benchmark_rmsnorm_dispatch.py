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
Benchmark aten vs quack vs oink RMSNorm through the same PyTorch API.

All three are called via ``torch.ops.aten._fused_rms_norm``. Quack is
registered via ``torch._native`` (requires the quack PR in pytorch).
Oink is registered via ``kernelagent_oink.register_all_kernels``.
Aten is the unoverridden baseline.

This script must be invoked three times with ``--mode={aten,quack,oink}``
by the companion ``run_benchmark_rmsnorm_dispatch.sh`` script, which
swaps ``torch._native/ops/norm/__init__.py`` between runs.

Usage::

    bash oink/benchmarks/benchmark/run_benchmark_rmsnorm_dispatch.sh
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCH_NATIVE_SKIP_VERSION_CHECK", "1")

import torch
from triton.testing import do_bench

SHAPES = [
    (4096, 4096),
    (4096, 8192),
    (16384, 4096),
    (16384, 8192),
    (65536, 4096),
    (65536, 8192),
]
DTYPE = torch.bfloat16


def bench(fn, warmup=50, rep=200):
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark aten/quack/oink RMSNorm through aten API."
    )
    p.add_argument("--mode", choices=["aten", "quack", "oink"], required=True)
    args = p.parse_args()

    if args.mode == "oink":
        import kernelagent_oink

        kernelagent_oink.register_all_kernels(force=True)

    # Warm up (triggers JIT compilation for quack/oink).
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

        # Forward only.
        def fn_fwd(x=x, w=w, N=N):
            return torch.ops.aten._fused_rms_norm(x, [N], w, 1e-5)

        fwd_ms = bench(fn_fwd)

        # Forward + backward.
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
