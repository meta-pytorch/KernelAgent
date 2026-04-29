# Blackwell SM10x Benchmarks (KernelAgent-Oink vs Quack)

This folder contains SM10x (GB200 / GB300 / Blackwell) microbenchmarks for the
Oink CuTeDSL kernels, comparing against Quack’s SM100 kernels where Quack
provides an equivalent API.

## Prereqs

- GPU: **SM10x / Blackwell** (`torch.cuda.get_device_capability()[0] == 10`).
- Python deps in your environment:
  - `torch`
  - `nvidia-cutlass-dsl>=4.4.2` (CuTeDSL)
  - `cuda-python`
  - `triton` (only for `triton.testing.do_bench`)
  - `quack` / `quack-kernels` (optional; only needed for Oink-vs-Quack comparisons)

Recommended env vars:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
# GB300 / SM103:
export CUTE_DSL_ARCH=sm_103a
# GB200/B200 / SM100 historical runs:
# export CUTE_DSL_ARCH=sm_100a
```

For the pinned GB300 / SM103 benchmark environment used by the current README
numbers:

```bash
conda create -y -n cute python=3.12
conda run -n cute python -m pip install --upgrade pip setuptools wheel packaging ninja
conda run -n cute python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch
conda run -n cute python -m pip install 'nvidia-cutlass-dsl==4.4.2' cuda-python triton matplotlib pytest pytest-cov
conda run -n cute python -m pip install -e '.[bench]'
conda run -n cute python -m pip install 'git+https://github.com/Dao-AILab/quack.git'  # optional comparison baseline
```

## Shape suites

- **Quack-suite**: `(batch, seq) ∈ {1,4,8,16,32} × {8192,16384,32768,65536,131072}`,
  with `hidden = 4096` so `M = batch * seq`, `N = 4096`.
- **DeepSeek-V3-like (DSv3)**
  - RMSNorm / LayerNorm / Softmax: `M ∈ {4096, 16384, 65536}`, `N ∈ {6144, 7168, 8192}`
  - Cross-entropy: `M ∈ {4096, 16384, 65536}`, `N ∈ {3072, 6144, 8192, 12288}`
- **DeepSeek-V4-Flash norm shapes (DSv4)** from `deepseek-ai/DeepSeek-V4-Flash/inference/model.py`
  - hidden-state RMSNorm / LayerNorm: `M ∈ {4096, 16384, 65536}`, `N = 7168`
  - q_lora RMSNorm: `M ∈ {4096, 16384, 65536}`, `N = 1536`
  - kv latent / per-head RMSNorm: `M ∈ {4096, 16384, 65536}`, `N = 512`

## Correctness gates

By default, each script runs a per-shape `torch.testing.assert_close` check vs a
**pure-PyTorch reference** **before** emitting timing numbers. When Quack is
available for that op/path, the script also validates Quack vs the *same*
reference (so speedups can’t come from looser numerics).

Disable with `--skip-verify` only for quick smoke tests. Do not use
`--skip-verify` for README or release performance numbers.

## Roofline reporting

Most benchmark JSONs include `*_hbm_frac` using `bench_utils.detect_hbm_peak_gbps()`.
That helper is a coarse fallback (`8000 GB/s` for SM10x) so old JSONs can be
compared consistently. For GB300/SM103 published results, use a measured roofline
run instead.

Current measured GB300 BF16 STREAM-like roof used in the README:

- **7.140 TB/s** (triad, `BLOCK=2048`, `warps=8`)
- 90% target: **6.426 TB/s**

Regenerate on the current machine:

```bash
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/benchmark/benchmark_hbm_roofline_sm100.py --dtype bf16 --op both --gb 1 \
  --json /tmp/oink_sm103_hbm_roofline_bf16_current.json'
```

## Running benchmarks

All primary scripts support:

- `--quack-suite` or `--dsv3` (and `--dsv4` where applicable)
- `--configs MxN,...`
- `--dtype {bf16,fp16,fp32}`
- `--iters <ms>` and `--warmup-ms <ms>` for kernel-only timing
- `--json <path>` and/or `--csv <path>` outputs (meta + rows)

### One-command suite

Run the full Quack-suite + DSv3 set (Oink vs Quack) and write all JSON artifacts
to a timestamped directory:

```bash
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/readme/run_sm100_suite.py --dtype bf16'

# Include DeepSeek-V4-Flash norm workloads:
conda run -n cute bash -lc 'PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/readme/run_sm100_suite.py --dtype bf16 --include-dsv4 \
  --out-dir /tmp/oink_sm103_suite_bf16_current'
```

Turn JSON artifacts into Markdown tables (with geomean speedups):

```bash
conda run -n cute bash -lc 'python benchmarks/readme/summarize_results.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --out /tmp/oink_sm103_suite_bf16_current_summary.md'
```

Generate SM103 SVGs from current JSONs and measured roofline:

```bash
conda run -n cute bash -lc 'python benchmarks/readme/plot_quack_style_svg.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --suite quack_suite --include-layernorm \
  --roofline-json /tmp/oink_sm103_hbm_roofline_bf16_current.json \
  --arch-label "SM103 / GB300" \
  --out benchmarks/media/sm103_bf16_oink_vs_quack_with_layernorm.svg'

conda run -n cute bash -lc 'python benchmarks/readme/plot_quack_style_svg.py \
  --in-dir /tmp/oink_sm103_suite_bf16_current \
  --suite dsv3_all --shape-policy first \
  --roofline-json /tmp/oink_sm103_hbm_roofline_bf16_current.json \
  --arch-label "SM103 / GB300" \
  --out benchmarks/media/sm103_bf16_oink_vs_quack_dsv3_all.svg'
```

The existing `sm100_*` SVGs in `benchmarks/media/` are historical SM100/B200
plots. Do not use them as GB300 evidence.

### RMSNorm forward

```bash
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite.json

python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_dsv3.json

# vLLM-style inference weights (weight dtype == activation dtype)
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype same --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_quack_suite_wsame.json

# DeepSeek-V4-Flash norm grid
python benchmarks/benchmark/benchmark_rmsnorm_sm100.py --dtype bf16 --weight-dtype same --dsv4 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_rmsnorm_fwd_dsv4_wsame.json
```

### Fused Add + RMSNorm (vLLM-style, in-place)

This is a good roofline case study kernel (heavy read/write traffic, very little
extra math). Oink exposes an **in-place** fused op that updates `x` and
`residual`. Quack's fused kernel writes separate `out` and `residual_out`
buffers, so the default benchmark baseline (`--quack-baseline kernel_inplace`)
times Quack plus the copies needed to match Oink's in-place semantics. Use
`--quack-baseline kernel` to time only the Quack kernel with preallocated
outputs.

```bash
# DeepSeek-V3 hidden-size sweep
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py \
    --dtype bf16 --dsv3 --iters 80 --warmup-ms 15 \
    --quack-baseline kernel_inplace \
    --json /tmp/oink_sm103_fused_add_rmsnorm_dsv3_bf16.json

# DeepSeek-V4-Flash hidden-state sweep (N=7168)
PYTHONNOUSERSITE=1 CUTE_DSL_ARCH=sm_103a \
  python benchmarks/benchmark/benchmark_fused_add_rmsnorm_sm100.py \
    --dtype bf16 --dsv4 --iters 80 --warmup-ms 15 \
    --quack-baseline kernel_inplace \
    --json /tmp/oink_sm103_fused_add_rmsnorm_dsv4_bf16.json
```

Current GB300 / SM103 BF16 results from correctness-gated runs:

| suite | rows | speedup vs Quack (min / geomean / max) |
|---|---:|---:|
| DSv3 fused-add RMSNorm | 9 | 2.022x / 2.045x / 2.089x |
| DSv4 fused-add RMSNorm | 3 | 2.030x / 2.192x / 2.521x |

DSv3 per-shape results:

| M | N | Oink ms | Quack ms | speedup | Oink TB/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 6144 | 0.0360 | 0.0727 | 2.022x | 5.598 |
| 4096 | 7168 | 0.0396 | 0.0828 | 2.089x | 5.926 |
| 4096 | 8192 | 0.0479 | 0.0993 | 2.076x | 5.610 |
| 16384 | 6144 | 0.1206 | 0.2463 | 2.043x | 6.678 |
| 16384 | 7168 | 0.1393 | 0.2830 | 2.031x | 6.742 |
| 16384 | 8192 | 0.1574 | 0.3212 | 2.040x | 6.821 |
| 65536 | 6144 | 0.4575 | 0.9285 | 2.030x | 7.041 |
| 65536 | 7168 | 0.5329 | 1.0785 | 2.024x | 7.052 |
| 65536 | 8192 | 0.6077 | 1.2466 | 2.052x | 7.068 |

DSv4 per-shape results:

| M | N | Oink ms | Quack ms | speedup | Oink TB/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 7168 | 0.0415 | 0.1047 | 2.521x | 5.655 |
| 16384 | 7168 | 0.1388 | 0.2855 | 2.057x | 6.769 |
| 65536 | 7168 | 0.5314 | 1.0785 | 2.030x | 7.072 |

### RMSNorm backward

```bash
python benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --quack-suite --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_quack_suite.csv

python benchmarks/benchmark/benchmark_rmsnorm_bwd_sm100.py --dtype bf16 --weight-dtype fp32 --dsv3 --iters 100 --warmup-ms 25 \
  --csv /tmp/oink_rmsnorm_bwd_dsv3.csv
```

### Softmax (forward + backward)

```bash
python benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_quack_suite.json

python benchmarks/benchmark/benchmark_softmax_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_softmax_fwd_bwd_dsv3.json
```

### Cross-entropy (forward + backward)

```bash
python benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --quack-suite --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_quack_suite.json

python benchmarks/benchmark/benchmark_cross_entropy_sm100.py --dtype bf16 --mode fwd_bwd --dsv3 --iters 50 --warmup-ms 25 \
  --json /tmp/oink_cross_entropy_fwd_bwd_dsv3.json
```

### LayerNorm forward

```bash
python benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --quack-suite --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_quack_suite.json

python benchmarks/benchmark/benchmark_layernorm_sm100.py --dtype bf16 --dsv3 --iters 200 --warmup-ms 25 \
  --json /tmp/oink_layernorm_fwd_dsv3.json
```

## Notes

- These scripts intentionally avoid importing any external Oink checkout so the
  results reflect the in-tree KernelAgent-Oink kernels.
- `src/kernelagent_oink/blackwell/rmsnorm_with_stage2.py` is a compatibility
  facade. The stage-2 scheduling policy lives in `_rmsnorm_impl.py`; keep the
  facade for downstream imports.
- For RMSNorm, the stage-2 path is a fallback used when the pointer-based fast
  path cannot be used (for example when layouts/alignments are incompatible). You
  can force it for A/B testing via `KERNELAGENT_OINK_FORCE_RMSNORM_STAGE2=1`.
