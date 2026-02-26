# SM100 Benchmarks (KernelAgent-Oink vs Quack)

This folder contains SM100 (GB200 / Blackwell) microbenchmarks for the Oink
CuTeDSL kernels vendored into KernelAgent, comparing against Quack’s SM100
kernels where Quack provides an equivalent API.

## Prereqs

- GPU: **SM100** (`torch.cuda.get_device_capability() == (10, 0)`).
- Python deps in your environment:
  - `torch`
  - `nvidia-cutlass-dsl` (CuTeDSL)
  - `cuda-python`
  - `triton` (only for `triton.testing.do_bench`)
  - `quack` (optional; only needed for Oink-vs-Quack comparisons)

Recommended env vars:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUTE_DSL_ARCH=sm_100a
```

## Shape suites

- **Quack-suite**: `(batch, seq) ∈ {1,4,8,16,32} × {8192,16384,32768,65536,131072}`,
  with `hidden = 4096` so `M = batch * seq`, `N = 4096`.
- **DeepSeek-V3-like (DSv3)**
  - RMSNorm / LayerNorm / Softmax: `M ∈ {4096, 16384, 65536}`, `N ∈ {6144, 7168, 8192}`
  - Cross-entropy: `M ∈ {4096, 16384, 65536}`, `N ∈ {3072, 6144, 8192, 12288}`

## Correctness gates

By default, each script runs a per-shape `torch.testing.assert_close` check
vs a **pure-PyTorch reference** **before** emitting timing numbers. When Quack
is available for that op/path, the script also validates Quack vs the *same*
reference (so speedups can’t come from looser numerics).

Disable with `--skip-verify` only for quick smoke tests.

## Running benchmarks

All scripts support:

- `--quack-suite` or `--dsv3` (or `--configs MxN,...`)
- `--dtype {bf16,fp16,fp32}`
- `--iters <ms>` and `--warmup-ms <ms>` for kernel-only timing
- `--json <path>` and/or `--csv <path>` outputs (meta + rows)

### One-command suite

Run the full Quack-suite + DSv3 set (Oink vs Quack) and write all JSON artifacts
to a timestamped directory:

```bash
python oink/benchmarks/readme/run_sm100_suite.py --dtype bf16
```

Turn the JSON artifacts into Markdown tables (with geomean speedups):

```bash
python oink/benchmarks/readme/summarize_results.py --in-dir /tmp/kernelagent_oink_sm100_suite_<timestamp> \
  --out /tmp/kernelagent_oink_sm100_suite_summary.md
```

### Measured HBM roofline (STREAM-like)

To contextualize the `*_tbps` numbers as a fraction of a *measured* bandwidth
ceiling (rather than a theoretical spec), run:

```bash
CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_hbm_roofline_sm100.py --dtype bf16 --op both --gb 2 \
  --json /tmp/hbm_roofline_sm100_bf16.json
```

### RMSNorm forward

```bash
python oink/benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite.json

python oink/benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_dsv3.json

# vLLM-style inference weights (weight dtype == activation dtype)
python oink/benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype same --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite_wsame.json
```

### Fused Add + RMSNorm (vLLM-style, in-place)

This is a good "roofline case study" kernel (heavy read/write traffic, very little extra math):

```bash
CUDA_VISIBLE_DEVICES=0 python oink/benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py --dtype bf16 --M 65536 --N 4096 \
  --json /tmp/fused_add_rmsnorm_sm100_bf16.json
```

Note on the Quack baseline: Oink exposes an **in-place** fused op (updates `x` and `residual`).
Quack’s fused kernel produces `out` and `residual_out` out-of-place, so by default the benchmark
times `quack::_rmsnorm_fwd` **plus** two explicit copies (`x.copy_(out)`, `residual.copy_(residual_out)`)
to match the in-place semantics (integration-realistic). Use `--quack-baseline kernel` to time only
the Quack fused kernel with preallocated outputs.

### RMSNorm backward

```bash
python oink/benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_quack_suite.csv

python oink/benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_dsv3.csv
```

### Softmax (forward + backward)

```bash
python oink/benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_quack_suite.json

python oink/benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_dsv3.json
```

### Cross-entropy (forward + backward)

```bash
python oink/benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_quack_suite.json

python oink/benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_dsv3.json
```

### LayerNorm forward

```bash
python oink/benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_quack_suite.json

python oink/benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_dsv3.json
```

## Notes

- These scripts intentionally avoid importing any external Oink checkout so the
  results reflect the in-tree KernelAgent Oink kernels.
- For RMSNorm, the `rmsnorm_with_stage2` implementation is a **fallback** that
  is only used when the pointer-based fast path cannot be used (e.g. when
  `weight.dtype != x.dtype`, or when layouts/alignments are incompatible). You
  can force it for A/B testing via `KERNELAGENT_OINK_FORCE_RMSNORM_STAGE2=1`.
