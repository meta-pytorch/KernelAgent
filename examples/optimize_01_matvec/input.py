# kernel.py
# Matrix-vector multiplication using Triton: C = A @ B
# Implements the exact problem from the test:
#   - M = 2048
#   - K = 1,048,576
#   - A: (M, K), BF16
#   - B: (K, 1), BF16
#   - C: (M, 1), BF16
#
# Notes on fusion:
# - The entire operation (matrix-vector product) is executed in a single Triton kernel.
# - There is nothing else to fuse (no bias/activation in the test), so no extra kernel stages are required.
# - All math is performed inside the Triton kernel; the Python wrapper only validates/allocates/configures.
#
# Triton programming guidelines followed:
# - Use @triton.jit for kernels.
# - Use tl.constexpr for compile-time constants (BLOCK_M, BLOCK_K).
# - Proper indexing with tl.program_id, tl.arange, and tl.cdiv.
# - Use tl.load/tl.store with masks for OOB protection and coalesced access on contiguous inputs.
# - Accumulate in FP32 for numerical stability and convert to BF16 on store.

import triton
import triton.language as tl
import torch


@triton.jit
def _matvec_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program id along the M dimension (each program computes a block of rows)
    pid_m = tl.program_id(0)

    # Row indices this program handles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Help compiler with alignment information
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)

    # Initialize FP32 accumulator for BLOCK_M rows
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over K dimension in chunks of BLOCK_K
    # Use tl.range to ensure proper device-side looping
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # Also assist compiler with alignment info for K offsets
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

        # Compute pointers:
        # A tile is [BLOCK_M, BLOCK_K] region starting at (offs_m, offs_k)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B tile is a vector [BLOCK_K] at column 0 (since B is [K, 1])
        b_ptrs = b_ptr + (offs_k * stride_bk + 0 * stride_bn)

        # Load with masking to guard boundaries. Inputs are BF16; cast to FP32 for accumulation
        a = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0).to(
            tl.float32
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0).to(tl.float32)

        # Fused multiply-accumulate for rows in this tile:
        # sum over K tile dimension for each row
        acc += tl.sum(a * b[None, :], axis=1)

    # Convert accumulator to BF16 and store to C[:, 0]
    out = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + (offs_m * stride_cm + 0 * stride_cn)
    tl.store(c_ptrs, out, mask=m_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using a single Triton kernel.

    What is fused:
    - Entire matrix-vector multiplication is done in one pass inside the kernel.
    - No additional ops (e.g., bias/activation) are required by the test, so none are fused.

    Runtime constraints honored:
    - Wrapper only validates arguments, allocates output, and launches the Triton kernel.
    - All math is inside the Triton kernel; no torch.nn or torch.nn.functional usage.

    Args:
        A: [M, K] BF16 CUDA tensor
        B: [K, 1] BF16 CUDA tensor (also accepts shape [K], it will be viewed as [K, 1])

    Returns:
        C: [M, 1] BF16 CUDA tensor
    """
    # Validate device and dtype
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors.")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError("A and B must be torch.bfloat16 tensors.")

    if A.ndim != 2:
        raise ValueError("A must be 2D [M, K].")
    M, K = A.shape

    # Accept B as [K] or [K, 1]
    if B.ndim == 1:
        if B.shape[0] != K:
            raise ValueError(
                f"When B is 1D, expected shape [K]={K}, but got {tuple(B.shape)}"
            )
        Bv = B.view(K, 1)
    elif B.ndim == 2:
        if B.shape[0] != K or B.shape[1] != 1:
            raise ValueError(
                f"B must be [K, 1], got {tuple(B.shape)} (K must match A.shape[1])"
            )
        Bv = B
    else:
        raise ValueError("B must be 1D [K] or 2D [K, 1].")

    # Allocate output C [M, 1]
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

    # Extract strides (in elements, not bytes)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = Bv.stride()
    stride_cm, stride_cn = C.stride()

    # Kernel launch configuration
    # Choose modest tile sizes to balance register usage and loop count over K.
    # For the huge K in the test, BLOCK_K=256 works well without excessive register pressure.
    BLOCK_M = 128
    BLOCK_K = 256

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    _matvec_kernel[grid](
        A,
        Bv,
        C,
        M,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return C
