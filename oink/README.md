# KernelAgent Oink (vLLM plugin)

This subproject provides an **out-of-tree vLLM plugin** that registers
`torch.library.custom_op` entrypoints under the `oink::` namespace:

- `torch.ops.oink.rmsnorm`
- `torch.ops.oink.fused_add_rms_norm`

The implementation is backed by a CuTeDSL (CUTLASS) RMSNorm kernel tuned for
**NVIDIA Blackwell (SM100)**.

## Install (editable)

From the `KernelAgent` repo root:

```bash
pip install -e ./oink
```

This plugin requires the CuTeDSL stack:

```bash
pip install nvidia-cutlass-dsl cuda-python
```

## Use with vLLM

1. Enable the vLLM integration:

```bash
export VLLM_USE_OINK_RMSNORM=1
```

2. Ensure vLLM keeps `rms_norm` as a custom op when using `torch.compile` /
CUDA graphs. In Python:

```python
from vllm import LLM

llm = LLM(
    model=...,
    tensor_parallel_size=...,
    enforce_eager=False,
    compilation_config={"custom_ops": ["none", "+rms_norm"]},
)
```

Without `+rms_norm`, Inductor may fuse RMSNorm into larger Triton kernels and
neither vLLM's CUDA RMSNorm nor Oink will run.

## Notes

- This plugin is designed to be **safe to import even when disabled**; it only
  registers ops when `VLLM_USE_OINK_RMSNORM` is truthy (`"1"` / `"true"`).
- The ops preserve **padded-row layouts** for 2D tensors (shape `[M, N]`,
  `stride(1) == 1`, and potentially `stride(0) > N`), which is required for
  `torch.compile` stride verification on some models (e.g., MLA padded inputs).
