# kernel.py
# Triton-based MaxPool3d implementation specialized to the test configuration:
# - Input tensor shape: (N=16, C=32, D=128, H=128, W=128)
# - Pooling params typically called by the test: kernel_size=3, stride=2, padding=1, dilation=3
#
# Notes on fusion:
# - MaxPool3d is a standalone reduction operator. There is no natural upstream/downstream op specified
#   in the test to fuse with (e.g., bias, activation), so this kernel focuses on an efficient single-pass
#   pooling implementation. If a pipeline included additional pointwise ops on the pooled output, those
#   could be fused into the epilogue to reduce memory traffic.

import torch
import triton
import triton.language as tl


@triton.jit
def _maxpool3d_kernel(
    x_ptr,  # *const T
    y_ptr,  # *T
    N,
    C,
    D,
    H,
    W,  # input sizes
    OD,
    OH,
    OW,  # output sizes
    strideN,
    strideC,
    strideD,
    strideH,
    strideW,  # input strides (in elements)
    ostrideN,
    ostrideC,
    ostrideD,
    ostrideH,
    ostrideW,  # output strides (in elements)
    KERNEL_SIZE: tl.constexpr,  # pool kernel size (assumed cubic here)
    STRIDE: tl.constexpr,  # pool stride (assumed same across dims)
    PADDING: tl.constexpr,  # pool padding (assumed same across dims)
    DILATION: tl.constexpr,  # dilation (assumed same across dims)
    BLOCK_W: tl.constexpr,  # vectorized span of OW per program
):
    # Program ids:
    # axis 0: blocks along OW
    # axis 1: specific OD index
    # axis 2: flattened (N*C*OH)
    pid_w = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    # Decode pid_z into (n, c, oh)
    oh = pid_z % OH
    nc = pid_z // OH
    c = nc % C
    n = nc // C

    # Compute OW offsets this program handles
    ow_start = pid_w * BLOCK_W
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offsets < OW

    # Output indices along D and H (scalars per program)
    od = pid_d

    # Base starts in input space for the pooling window
    d_base = od * STRIDE - PADDING
    h_base = oh * STRIDE - PADDING
    # Vector of base W inputs for this block
    w_base = ow_offsets * STRIDE - PADDING

    # Accumulator in fp32 for numerical robustness (stores max across the 3x3x3 window)
    acc = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)

    # Precompute base strides for (n, c)
    base_nc = n * strideN + c * strideC

    # Iterate over the pooling window (kd, kh, kw) with compile-time unrolling
    for kd in tl.static_range(0, KERNEL_SIZE):
        d_idx = d_base + kd * DILATION
        valid_d = (d_idx >= 0) & (d_idx < D)
        # Safe index to keep addresses in-bounds for masked loads
        d_idx_safe = tl.where(valid_d, d_idx, 0)

        for kh in tl.static_range(0, KERNEL_SIZE):
            h_idx = h_base + kh * DILATION
            valid_h = (h_idx >= 0) & (h_idx < H)
            valid_dh = valid_d & valid_h
            h_idx_safe = tl.where(valid_h, h_idx, 0)

            # Base pointer for current (n, c, d_idx, h_idx)
            base_dh = base_nc + d_idx_safe * strideD + h_idx_safe * strideH

            for kw in tl.static_range(0, KERNEL_SIZE):
                w_idx = w_base + kw * DILATION
                # Check bounds per-lane; combine with ow_mask and valid_dh
                w_valid = ow_mask & (w_idx >= 0) & (w_idx < W) & valid_dh
                w_idx_safe = tl.where(w_valid, w_idx, 0)

                # Element pointers for this (kd, kh, kw) and OW lanes
                ptrs = x_ptr + base_dh + w_idx_safe * strideW

                # Load with mask; out-of-bounds lanes use -inf so they don't affect max
                vals = tl.load(ptrs, mask=w_valid, other=-float("inf"))
                vals_f32 = vals.to(tl.float32)
                acc = tl.maximum(acc, vals_f32)

    # Store result
    out_base = y_ptr + n * ostrideN + c * ostrideC + od * ostrideD + oh * ostrideH
    out_ptrs = out_base + ow_offsets * ostrideW
    tl.store(out_ptrs, acc, mask=ow_mask)


def _compute_out_dim(
    L_in: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    # PyTorch formula: floor((L_in + 2*padding - dilation*(kernel - 1) - 1) / stride + 1)
    return (L_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def kernel_function(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int
):
    """
    Triton-backed 3D Max Pooling (no indices), compatible with the test's call signature.

    Args:
        x: Input tensor of shape (N, C, D, H, W), CUDA device.
        kernel_size: int, pooling kernel size (assumed cubic)
        stride: int, pooling stride (assumed same for D/H/W)
        padding: int, zero-padding applied on each side (assumed same for D/H/W)
        dilation: int, dilation factor (assumed same for D/H/W)

    Returns:
        y: Output tensor of shape (N, C, OD, OH, OW) with the same dtype/device as x.

    Design and fusion notes:
    - This is a single-pass, fused pooling reduction: it computes the maximum over the 3D dilated window
      directly from global memory and writes the result, with masking to handle padding/boundaries.
    - No additional post-processing stages are specified in the test; thus, there are no further ops to fuse.
      If a follow-up pointwise op were known, it could be integrated into the epilogue to reduce memory traffic.

    Runtime policy:
    - The wrapper only validates arguments, computes output shape, allocates the output tensor, and launches
      the Triton kernel. All math (window traversal and reduction) happens inside the Triton kernel.
    """
    # Basic checks
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 5:
        raise ValueError(
            f"Expected 5D input (N, C, D, H, W), got shape {tuple(x.shape)}"
        )
    if (
        not isinstance(kernel_size, int)
        or not isinstance(stride, int)
        or not isinstance(padding, int)
        or not isinstance(dilation, int)
    ):
        raise TypeError("kernel_size, stride, padding, dilation must be ints.")

    N, C, D, H, W = x.shape
    K = kernel_size
    S = stride
    P = padding
    Di = dilation

    # Compute output shape
    OD = _compute_out_dim(D, K, S, P, Di)
    OH = _compute_out_dim(H, K, S, P, Di)
    OW = _compute_out_dim(W, K, S, P, Di)
    if OD <= 0 or OH <= 0 or OW <= 0:
        raise ValueError(
            "Computed non-positive output dimension(s). Check pooling parameters."
        )

    # Allocate output
    y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    # Get strides in "element" units (PyTorch strides are already in elements, not bytes)
    strideN, strideC, strideD, strideH, strideW = x.stride()
    ostrideN, ostrideC, ostrideD, ostrideH, ostrideW = y.stride()

    # Configure launch
    # We tile along OW dimension with BLOCK_W elements per program.
    # OW is 62 in the test, so BLOCK_W=64 covers each row in one program; remaining lanes are masked.
    BLOCK_W = 64

    def grid(meta):
        return (triton.cdiv(OW, meta["BLOCK_W"]), OD, N * C * OH)

    # Launch kernel
    _maxpool3d_kernel[grid](
        x,
        y,
        N,
        C,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        strideN,
        strideC,
        strideD,
        strideH,
        strideW,
        ostrideN,
        ostrideC,
        ostrideD,
        ostrideH,
        ostrideW,
        KERNEL_SIZE=K,
        STRIDE=S,
        PADDING=P,
        DILATION=Di,
        BLOCK_W=BLOCK_W,
        num_warps=4,  # Reasonable default for this memory-bound reduction
        num_stages=2,
    )

    return y
