from typing import Optional


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=3, num_warps=8
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B
    A: [M, K] (bf16), B: [K, N] (bf16), C: [M, N] (bf16)
    Accumulation is in fp32 for numerical stability.


    Tiling:
      - Each program computes a [BLOCK_M, BLOCK_N] tile of C
      - Reduction over K in steps of BLOCK_K
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    # Make offsets compiler-friendly for coalesced accesses
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(0, k_tiles):
        offs_k = k_tile * BLOCK_K + offs_k_init

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32; tl.dot uses tensor cores when possible
        acc = tl.dot(a, b, acc)

    # Write back result with proper masking
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast to BF16 on store
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def kernel_function(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B using a Triton kernel.


    - Computes BF16 GEMM with FP32 accumulation in the kernel.
    - Wrapper performs only argument checks, allocation, and kernel launch.
    - No additional ops (bias/activation) were specified by the test; thus, no fusion opportunities exist.
      If future requirements include bias/add/activation, they can be fused into this kernel's epilogue.


    Args:
        A: [M, K], torch.bfloat16 on CUDA
        B: [K, N], torch.bfloat16 on CUDA
        C/out: optional preallocated output buffer [M, N], torch.bfloat16 on CUDA


    Returns:
        C: [M, N], torch.bfloat16
    """
    # Normalize output argument names
    if out is None and C is not None:
        out = C

    # Basic checks
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)):
        raise TypeError("A and B must be torch.Tensor")
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"A and B must be 2D. Got A.dim()={A.dim()}, B.dim()={B.dim()}"
        )
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes: A: {tuple(A.shape)}, B: {tuple(B.shape)}"
        )
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors")
    if A.device != B.device:
        raise ValueError("A and B must be on the same CUDA device")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise TypeError(
            f"A and B must be torch.bfloat16. Got A.dtype={A.dtype}, B.dtype={B.dtype}"
        )

    M, K = A.shape
    KB, N = B.shape
    assert K == KB

    # Prepare output
    if out is None:
        out = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.shape != (M, N):
            raise ValueError(
                f"out has wrong shape: expected {(M, N)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be torch.bfloat16. Got out.dtype={out.dtype}")
        if out.device != A.device:
            raise ValueError("out must be on the same device as A/B")

    # Strides (supporting general layout; inputs are contiguous in test)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = out.stride()

    # Launch grid: 2D over M and N tiles
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_kernel[grid](
        A,
        B,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )

    return out
