# KernelAgent-Oink

KernelAgent-Oink is a small **CuTeDSL (CUTLASS DSL) kernel library** for
**NVIDIA Blackwell (SM10x / GB200 / GB300 / B200-class)**, bundled as a lightweight
Python package that can be used standalone or as a **vLLM general plugin**.

At the moment, the vLLM integration exposes the following `torch.library.custom_op`
entrypoints under the `oink::` namespace:

- `torch.ops.oink.rmsnorm(x, weight, eps) -> Tensor`
- `torch.ops.oink.fused_add_rms_norm(x, residual, weight, eps) -> None` (in-place)

The package also includes additional SM100 kernels used by the benchmark suite:
LayerNorm, Softmax (fwd+bwd), and CrossEntropy (fwd+bwd).

## Requirements

- GPU: **SM10x (Blackwell)** for the fast CuTeDSL paths. On other GPUs, Oink falls back to
  reference PyTorch implementations for correctness.
- Python dependencies:
  - `nvidia-cutlass-dsl` (CuTeDSL)
  - `cuda-python`
  - `torch` (provided by your environment / vLLM)

Recommended env vars:

```bash
export CUTE_DSL_ARCH=sm_100a
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

On **GB300 / SM103**, prefer:

```bash
export CUTE_DSL_ARCH=sm_103a
```

## Install (editable)

From the `KernelAgent` repo root:

```bash
pip install -e ./oink
```

For running the in-repo benchmark suite / plots:

```bash
pip install -e "./oink[bench]"
```

## Usage

### vLLM (general plugin)

1) Enable the plugin:

```bash
export VLLM_USE_OINK_RMSNORM=1
```

2) Ensure vLLM keeps `rms_norm` as a custom op when using `torch.compile` / CUDA graphs:

```python
from vllm import LLM

llm = LLM(
    model=...,
    tensor_parallel_size=...,
    enforce_eager=False,
    compilation_config={"custom_ops": ["none", "+rms_norm"]},
)
```

Without `+rms_norm`, Inductor may fuse RMSNorm into larger kernels and neither
vLLM’s CUDA RMSNorm nor Oink will run.

### Direct PyTorch usage (manual op registration)

For standalone use (outside vLLM), register the custom ops once:

```python
import kernelagent_oink
import torch

kernelagent_oink.register(force=True)

x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
y = torch.ops.oink.rmsnorm(x, w, 1e-6)
```

## Benchmarks

### GB200 / B200 (SM100) benchmark suite

The repo includes a Quack-style benchmark suite (tables + SVG plots) to compare
Oink against Quack and to reproduce the reported speedups. The pre-generated
plots below were measured on **GB200 / B200-class SM100** systems.

- How to run + methodology: `oink/benchmarks/README.md`
- Pre-generated plots: `oink/benchmarks/media/`

<div align="center">
  <img src="benchmarks/media/sm100_bf16_oink_vs_quack_with_layernorm.svg" alt="SM100 BF16: Oink vs Quack (Quack-suite)">
</div>

<div align="center">
  <img src="benchmarks/media/sm100_bf16_oink_vs_quack_dsv3_all.svg" alt="SM100 BF16: Oink vs Quack (DSv3-like shapes)">
</div>

### GB300 (SM103) Q/K-norm results

We also benchmarked the real Llama4x-style Q/K-norm workload on **GB300
(SM103)** using non-contiguous `q` / `k` views produced by `qkv.split()`. This
benchmark reports both the direct CuTeDSL/CUTLASS baseline and the optimized
Oink path for the production strided `[M, N]` views.

Example command (from the `KernelAgent` repo root):

```bash
source /home/leyuan/.local/miniconda3/bin/activate oink
export CUTE_DSL_ARCH=sm_103a
export PYTORCH_ALLOC_CONF=expandable_segments:True
python oink/qk_norm/benchmark_qk_norm_kernel.py
```

Representative steady-state medians from one GB300 run are shown below
(absolute microseconds may vary slightly run to run, but the ranking and
trend were stable).

#### Q path (`N=8192`, `scale=3.87`)

| M | CUTLASS (us) | Oink (us) | Speedup |
|---:|---:|---:|---:|
| 1 | 1.4 | 1.2 | 1.12x |
| 32 | 1.9 | 1.4 | 1.39x |
| 128 | 3.3 | 1.6 | 2.00x |
| 512 | 7.5 | 2.7 | 2.74x |
| 1024 | 12.6 | 4.0 | 3.12x |
| 4096 | 47.3 | 16.4 | 2.88x |
| 8192 | 93.7 | 38.0 | 2.47x |
| 16384 | 186.1 | 76.0 | 2.45x |
| 32768 | 371.5 | 152.7 | 2.43x |

#### K path (`N=1024`, `scale=1.0`)

| M | CUTLASS (us) | Oink (us) | Speedup |
|---:|---:|---:|---:|
| 1 | 1.3 | 1.2 | 1.06x |
| 32 | 1.6 | 1.3 | 1.21x |
| 128 | 1.6 | 1.3 | 1.21x |
| 512 | 2.3 | 1.4 | 1.58x |
| 1024 | 3.3 | 1.6 | 2.03x |
| 4096 | 7.6 | 2.5 | 3.03x |
| 8192 | 12.8 | 3.8 | 3.33x |
| 16384 | 23.1 | 6.5 | 3.56x |
| 32768 | 47.1 | 16.4 | 2.87x |

Takeaways from the GB300 Q/K-norm sweep:

- For the user-relevant multi-row workloads, Oink beats the CuTeDSL/CUTLASS
  baseline by comfortably more than 20%.
- The only cases below 20% are the tiny single-row latency-floor microcases:
  Q `M=1` is ~12% faster and K `M=1` is ~6% faster.
- Correctness spot-check from the same harness:
  - Q max diff vs eager: `0.03125`
  - K max diff vs eager: `0.007812`

## Links

| What | Link |
|---|---|
| Quack (expert baseline) | https://github.com/Dao-AILab/quack |
| KernelAgent (agentic framework) | https://github.com/meta-pytorch/KernelAgent |
| vLLM PR (Oink RMSNorm integration) | https://github.com/vllm-project/vllm/pull/31828 |
