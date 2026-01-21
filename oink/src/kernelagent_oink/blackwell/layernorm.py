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
LayerNorm kernel for SM100 (Blackwell) in CuteDSL.

This implementation:
- Mirrors Quack's LayerNorm tiling / cluster policy / cp.async pipeline
  but uses only local helpers so that it does not depend on the external
  `quack` package at runtime.
- Supports fp16 / bf16 / fp32 inputs with fp32 accumulation.
- Optionally writes out per-row `rstd` and `mean` buffers for reuse in
  backward or fused kernels.

Backward is implemented with dedicated CuteDSL kernels for input and
parameter gradients (dx, dweight, dbias), avoiding PyTorch autograd
while matching `torch.nn.functional.layer_norm`'s gradients numerically.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import operator
from typing import Optional, Tuple, Type

import torch
from torch import Tensor

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python

# CuTeDSL caches generated MLIR into a tempdir under a global default
# (`/tmp/$USER/cutlass_python_cache`). The cache bytecode format can differ across
# `nvidia-cutlass-dsl` versions, and cross-version cache sharing causes noisy
# warnings (and disables cache reuse).
if "CUTE_DSL_CACHE_DIR" not in os.environ:
    try:
        _dsl_ver = importlib.metadata.version("nvidia-cutlass-dsl")
    except Exception:
        _dsl_ver = "unknown"
    _dsl_ver = re.sub(r"[^0-9A-Za-z]+", "_", _dsl_ver)
    _user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    _tmp = os.environ.get("TMPDIR") or "/tmp"
    os.environ["CUTE_DSL_CACHE_DIR"] = os.path.join(
        _tmp, _user, f"cutlass_python_cache_{_dsl_ver}"
    )

try:
    import cutlass  # type: ignore  # noqa: F401
except Exception as e:
    raise ImportError(
        "kernelagent_oink.blackwell.layernorm requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute import runtime as rt
from cutlass.cute.runtime import from_dlpack

# Simple compile cache for the forward kernel
_COMPILE_CACHE: dict[Tuple[int, type[cutlass.Numeric], bool, bool, bool], object] = {}
_PTR_COMPILE_CACHE: dict[Tuple[object, ...], object] = {}

# Backward compile caches: one for dx, one for parameter gradients.
_BWD_DX_COMPILE_CACHE: dict[Tuple[int, Type[cutlass.Numeric]], object] = {}
_BWD_PARAM_COMPILE_CACHE: dict[Tuple[int, Type[cutlass.Numeric], bool], object] = {}

# Local helpers cloned from Quack via lite_quack so that this kernel does
# not depend on `quack` at runtime.
from kernelagent_oink.blackwell.lite_quack import (
    _KERNEL_ACCEPTS_LAYOUT_ARGS,
    TORCH2CUTE_DTYPE,
    ReductionBase as _ReductionBase,
    convert_from_dlpack as convert_from_dlpack_cute,
    domain_offset_i64,
    get_sm_count,
    predicate_k,
    row_reduce,
    warp_reduce,
)


def _convert_row_major(t: Tensor) -> cute.Tensor:
    """
    Convert a 2D row-major torch.Tensor to a CuTeDSL tensor with a compact,
    dynamic layout on the leading dimension.
    """
    return from_dlpack(t.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=(0, 1),
    )


class LayerNormSM100(_ReductionBase):
    """
    SM100 LayerNorm forward kernel.

    This mirrors `quack.layernorm.LayerNorm`'s schedule:
    - Stage=2 pipeline: first pass computes mean, second pass computes
      variance / rstd and normalization.
    - Threads-per-row and cluster_n policy follow Quack's LayerNorm
      heuristics to keep tensor-core friendly tiles across N.
    - Optional `reload_from` hint enables reloading X from SMEM for large-N
      shapes to shorten register lifetimes.

    Differences vs Quack:
    - Bias is optional and supported directly in the kernel.
    - Dtype mapping and reduction helpers come from `lite_quack`.
    """

    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=2)  # 2 stages for mean and var
        # Default reload policy mirrors Quack: use SMEM reload only for
        # very large hidden sizes. We keep this conservative for LayerNorm
        # and tune primarily via threads-per-block / cluster_n.
        self.reload_from: Optional[str] = None if N <= 16384 else "smem"
        self.delay_w_load: bool = False

    def _calculate_threads_per_row(self) -> int:
        # Match Quack's LayerNorm threads-per-row buckets.
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
            )
        )

    def _set_cluster_n(self) -> None:
        # Cluster_n policy mirrors quack.layernorm.LayerNorm._set_cluster_n.
        N = self.N
        if const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mB: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype

        # Tiling and cluster policy (mirrors Quack LayerNorm).
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0]) if _KERNEL_ACCEPTS_LAYOUT_ARGS else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE

        # Expand weight / bias to match tiler_mn[0] rows per CTA.
        mW = cute.make_tensor(
            mW.iterator,
            cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
        )
        if const_expr(mB is not None):
            mB = cute.make_tensor(
                mB.iterator,
                cute.prepend(mB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
            )
        if const_expr(mRstd is not None):
            mRstd = cute.make_tensor(
                mRstd.iterator,
                cute.append(mRstd.layout, cute.make_layout((self.N,), stride=(0,))),
            )
        if const_expr(mMean is not None):
            mMean = cute.make_tensor(
                mMean.iterator,
                cute.append(mMean.layout, cute.make_layout((self.N,), stride=(0,))),
            )

        kernel = (
            self.kernel(
                mX,
                mW,
                mB,
                mO,
                mRstd,
                mMean,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(self.reload_from),
                const_expr(self.delay_w_load),
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mW,
                mB,
                mO,
                mRstd,
                mMean,
                eps,
            )
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[
                1,
                self.cluster_n,
                1,
            ]
            if const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer,
        ptr_b: Optional[cute.Pointer],
        ptr_out: cute.Pointer,
        ptr_rstd: Optional[cute.Pointer],
        ptr_mean: Optional[cute.Pointer],
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions.

        This reconstructs cute.Tensor views from raw device pointers + explicit
        layouts inside the JIT graph, reusing the tuned LayerNormSM100 schedule.
        """
        # The kernel uses 128-bit vectorized copies for X. Mirror Quack's
        # `divisibility=128 // dtype.width` contract so the compiler can
        # prove alignment for cp.async.
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        # Match `mark_compact_shape_dynamic(mode=0, ...)`: M is dynamic, N is static.
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        layout_n = cute.make_layout((self.N,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)
        mW = cute.make_tensor(ptr_w, layout_n)
        mB = cute.make_tensor(ptr_b, layout_n) if const_expr(ptr_b is not None) else None
        mRstd = (
            cute.make_tensor(ptr_rstd, layout_m)
            if const_expr(ptr_rstd is not None)
            else None
        )
        mMean = (
            cute.make_tensor(ptr_mean, layout_m)
            if const_expr(ptr_mean is not None)
            else None
        )

        self.__call__(mX, mW, mB, mO, mRstd, mMean, stream, eps)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mB: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr,
        delay_w_load: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Slice for CTAs: use domain_offset_i64 to handle >2^31 elements.
        mX, mO = [
            domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)
        ]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
        gB = (
            cute.local_tile(mB, tiler_mn, (0, cluster_y))
            if const_expr(mB is not None)
            else None
        )
        gRstd = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if const_expr(mRstd is not None)
            else None
        )
        gMean = (
            cute.local_tile(mMean, tiler_mn, (bidx, cluster_y))
            if const_expr(mMean is not None)
            else None
        )

        # Copy atoms for X / W / B / O.
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_load_WB = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=128,
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X_async,
            tv_layout,
            tiler_mn,
        ).get_slice(tidx)
        thr_copy_WB = cute.make_tiled_copy(
            copy_atom_load_WB,
            tv_layout,
            tiler_mn,
        ).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(
            copy_atom_store_O,
            tv_layout,
            tiler_mn,
        ).get_slice(tidx)

        tWgW = thr_copy_WB.partition_S(gW)
        tBgB = (
            thr_copy_WB.partition_S(gB)
            if const_expr(gB is not None)
            else None
        )
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXrRstd = (
            thr_copy_O.partition_D(gRstd)
            if const_expr(mRstd is not None)
            else None
        )
        tXrMean = (
            thr_copy_O.partition_D(gMean)
            if const_expr(mMean is not None)
            else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # Fragments for gmem->rmem.
        tWrW = cute.make_fragment_like(tWgW)
        tBrB = (
            cute.make_fragment_like(tBgB)
            if const_expr(mB is not None)
            else None
        )
        tXrW = thr_copy_X.retile(tWrW)
        tXrB = (
            thr_copy_X.retile(tBrB)
            if const_expr(mB is not None)
            else None
        )
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=False)

        tXpX = predicate_k(
            thr_copy_X.partition_S(cX),
            limit=shape[1],
        )
        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()

        tWpW = predicate_k(
            thr_copy_WB.partition_S(cX),
            limit=shape[1],
        )
        if const_expr(not delay_w_load):
            cute.copy(copy_atom_load_WB, tWgW, tWrW, pred=tWpW)
            if const_expr(mB is not None):
                cute.copy(copy_atom_load_WB, tBgB, tBrB, pred=tWpW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        threads_per_row = tv_layout.shape[0][0]
        sum_x = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1)
                else None
            ),
        )
        mean = sum_x / shape[1]

        if const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
        elif const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(Float32)

        sum_sq_x_sub_mean = row_reduce(
            (x - mean) * (x - mean),
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)

        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (
                    self.cluster_n == 1
                    or cute.arch.block_idx_in_cluster() == 0
                )
            ):
                tXrRstd[0] = rstd

        if const_expr(mMean is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (
                    self.cluster_n == 1
                    or cute.arch.block_idx_in_cluster() == 0
                )
            ):
                tXrMean[0] = mean

        if const_expr(delay_w_load):
            cute.copy(copy_atom_load_WB, tWgW, tWrW, pred=tWpW)
            if const_expr(mB is not None):
                cute.copy(copy_atom_load_WB, tBgB, tBrB, pred=tWpW)

        if const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
        elif const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(Float32)

        x_hat = (x - mean) * rstd
        w = tXrW.load().to(Float32)
        y = x_hat * w
        if const_expr(mB is not None):
            b = tXrB.load().to(Float32)
            y = y + b

        tXrO.store(y.to(tXrO.element_type))
        tOpO = predicate_k(
            thr_copy_O.partition_S(cX),
            limit=shape[1],
        )
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mB: Optional[cute.Tensor],
            mO: cute.Tensor,
            mRstd: Optional[cute.Tensor],
            mMean: Optional[cute.Tensor],
            eps: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
            reload_from: cutlass.Constexpr,
            delay_w_load: cutlass.Constexpr,
        ):
            self._kernel_impl(
                mX,
                mW,
                mB,
                mO,
                mRstd,
                mMean,
                eps,
                tv_layout,
                tiler_mn,
                reload_from,
                delay_w_load,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mB: Optional[cute.Tensor],
            mO: cute.Tensor,
            mRstd: Optional[cute.Tensor],
            mMean: Optional[cute.Tensor],
            eps: Float32,
        ):
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(
                mX,
                mW,
                mB,
                mO,
                mRstd,
                mMean,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(self.reload_from),
                const_expr(self.delay_w_load),
            )


# -----------------------------------------------------------------------------
# Public Python API
# -----------------------------------------------------------------------------


def layernorm(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    """
    LayerNorm forward pass using the SM100 CuteDSL kernel.

    Args:
        x: Input tensor of shape (M, N).
        weight: Scale parameter of shape (N,), typically fp32.
        bias: Optional bias parameter of shape (N,).
        eps: Small value for numerical stability.
        return_rstd: Whether to return per-row reciprocal std (shape (M,)).
        return_mean: Whether to return per-row mean (shape (M,)).
    """
    assert x.is_cuda and weight.is_cuda, "x and weight must be CUDA tensors"
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    assert weight.dim() == 1, "weight must be 1D"
    assert x.shape[1] == weight.shape[0], "Last dim of x must match weight.size(0)"
    if bias is not None:
        assert bias.is_cuda, "bias must be on CUDA"
        assert bias.dim() == 1 and bias.shape[0] == weight.shape[0], (
            "bias must be 1D and match weight"
        )

    M, N = x.shape
    dtype = TORCH2CUTE_DTYPE[x.dtype]

    rstd = torch.empty(M, device=x.device, dtype=torch.float32) if return_rstd else None
    mean = torch.empty(M, device=x.device, dtype=torch.float32) if return_mean else None

    # Fast path: bypass DLPack conversions when the inputs are in the common
    # contiguous row-major layout and weights/bias are fp32 (Quack-style).
    if _can_use_ptr_path(x, weight, bias):
        out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
        _layernorm_forward_ptr_into(
            x=x,
            weight=weight,
            bias=bias,
            out=out,
            rstd=rstd,
            mean=mean,
            eps=eps,
        )
        if return_mean and return_rstd:
            return out, rstd, mean
        if return_rstd and not return_mean:
            return out, rstd
        if return_mean and not return_rstd:
            return out, mean
        return out

    out = torch.empty_like(x)
    mX = _convert_row_major(x)
    mO = _convert_row_major(out)

    # Weight/bias live in feature dimension (N).
    mW = convert_from_dlpack_cute(
        weight.detach(),
        leading_dim=0,
        alignment=16,
        divisibility=128 // cutlass.Float32.width,
    )
    mB = (
        convert_from_dlpack_cute(
            bias.detach(),
            leading_dim=0,
            alignment=16,
            divisibility=128 // cutlass.Float32.width,
        )
        if bias is not None
        else None
    )

    mRstd = (
        from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if rstd is not None
        else None
    )
    mMean = (
        from_dlpack(mean.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if mean is not None
        else None
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (N, dtype, mB is not None, mRstd is not None, mMean is not None)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = LayerNormSM100(dtype, N)
        compiled = cute.compile(
            op,
            mX,
            mW,
            mB,
            mO,
            mRstd,
            mMean,
            stream,
            Float32(eps),
        )
        _COMPILE_CACHE[key] = compiled

    compiled(
        mX,
        mW,
        mB,
        mO,
        mRstd,
        mMean,
        stream,
        Float32(eps),
    )

    if return_mean and return_rstd:
        return out, rstd, mean
    if return_rstd and not return_mean:
        return out, rstd
    if return_mean and not return_rstd:
        return out, mean
    return out


def _can_use_ptr_path(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> bool:
    """Return True if we can safely use the pointer-based fast path.

    This is intentionally conservative: we target the common inference-like
    layout (2D row-major with stride(1)==1) and Quack-style fp32 weights.
    """
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.stride(1) != 1:
        return False
    if not weight.is_cuda or weight.dim() != 1:
        return False
    if weight.dtype != torch.float32:
        return False
    if not weight.is_contiguous():
        return False
    if bias is not None:
        if not bias.is_cuda or bias.dim() != 1:
            return False
        if bias.dtype != torch.float32:
            return False
        if not bias.is_contiguous():
            return False
    # Require 16B alignment for 128-bit vector copies (matches Quack's assumed_align=16).
    if (x.data_ptr() % 16) != 0:
        return False
    if (weight.data_ptr() % 16) != 0:
        return False
    if bias is not None and (bias.data_ptr() % 16) != 0:
        return False
    # The kernel uses 128-bit vectorized loads; require the leading dimension
    # to preserve 16B alignment for every row start.
    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    divby = 128 // dtype_x.width
    if (x.stride(0) % divby) != 0:
        return False
    return True


def _layernorm_forward_ptr_into(
    *,
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    out: Tensor,
    rstd: Optional[Tensor],
    mean: Optional[Tensor],
    eps: float,
) -> None:
    """Launch the pointer-based LayerNorm kernel into preallocated outputs."""
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    assert weight.is_cuda and weight.dim() == 1 and weight.shape[0] == N
    if bias is not None:
        assert bias.is_cuda and bias.dim() == 1 and bias.shape[0] == N
    assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype
    assert out.stride() == x.stride(), "Pointer path expects out to match x strides"
    if rstd is not None:
        assert rstd.is_cuda and rstd.shape == (M,) and rstd.dtype == torch.float32
    if mean is not None:
        assert mean.is_cuda and mean.shape == (M,) and mean.dtype == torch.float32

    device_index = x.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)

    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    key = (
        "ptr",
        int(N),
        dtype_x,
        bias is not None,
        rstd is not None,
        mean is not None,
        int(device_index),
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = LayerNormSM100(dtype_x, int(N))
        ptr_x = rt.make_ptr(
            dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_w = rt.make_ptr(
            cutlass.Float32,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=16,
        )
        ptr_b = (
            rt.make_ptr(
                cutlass.Float32,
                bias.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if bias is not None
            else None
        )
        ptr_rstd = (
            rt.make_ptr(
                cutlass.Float32,
                rstd.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=4,
            )
            if rstd is not None
            else None
        )
        ptr_mean = (
            rt.make_ptr(
                cutlass.Float32,
                mean.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=4,
            )
            if mean is not None
            else None
        )
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_b,
            ptr_out,
            ptr_rstd,
            ptr_mean,
            Int32(int(M)),
            ld,
            stream,
            Float32(float(eps)),
        )
        _PTR_COMPILE_CACHE[key] = compiled

    ptr_x = rt.make_ptr(dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16)
    ptr_out = rt.make_ptr(
        dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_w = rt.make_ptr(
        cutlass.Float32,
        weight.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=16,
    )
    ptr_b = (
        rt.make_ptr(
            cutlass.Float32,
            bias.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=16,
        )
        if bias is not None
        else None
    )
    ptr_rstd = (
        rt.make_ptr(
            cutlass.Float32,
            rstd.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        if rstd is not None
        else None
    )
    ptr_mean = (
        rt.make_ptr(
            cutlass.Float32,
            mean.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        if mean is not None
        else None
    )
    ld = Int32(int(x.stride(0)))
    compiled(
        ptr_x,
        ptr_w,
        ptr_b,
        ptr_out,
        ptr_rstd,
        ptr_mean,
        Int32(int(M)),
        ld,
        stream,
        Float32(float(eps)),
    )


def layernorm_ref(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Reference LayerNorm implemented via torch.nn.functional.layer_norm.
    """
    x_f32 = x.float()
    w = weight.float()
    b = bias.float() if bias is not None else None
    y = torch.nn.functional.layer_norm(x_f32, (x.shape[-1],), w, b, eps)
    return y.to(x.dtype)


def _as_2d(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
    if x.dim() == 2:
        return x, x.shape
    original_shape = x.shape
    M = int(torch.prod(torch.tensor(original_shape[:-1])).item())
    N = original_shape[-1]
    return x.reshape(M, N), original_shape


def _restore_shape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return x.reshape(shape)


@cute.kernel
def _layernorm_backward_dx_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mdO: cute.Tensor,
    mRstd: cute.Tensor,
    mMean: cute.Tensor,
    mdX: cute.Tensor,
):
    """
    Simple CTA-per-row LayerNorm backward kernel for dx only.

    Each block processes one row of shape (N,), using block_threads threads.
    It performs two passes over the row:
      1) Compute mean_wdy and mean_xhat_wdy in fp32.
      2) Compute dx using the standard LayerNorm backward formula:
         dx = rstd * (wdy - mean_wdy - x_hat * mean_xhat_wdy),
         where wdy = dy * gamma and x_hat = (x - mean) * rstd.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_threads = const_expr(256)
    shape = mX.shape
    M = shape[0]
    N = shape[1]

    row = bidx
    if row < M:
        # Shared buffers for warp-level reductions across the block.
        smem = cutlass.utils.SmemAllocator()
        num_warps = const_expr(block_threads // cute.arch.WARP_SIZE)
        warp_sums_layout = cute.make_layout((num_warps,), stride=(1,))
        warp_sums_wdy = smem.allocate_tensor(Float32, warp_sums_layout, byte_alignment=4)
        warp_sums_xhatwdy = smem.allocate_tensor(Float32, warp_sums_layout, byte_alignment=4)

        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        rstd_val = mRstd[row].to(Float32)
        mean_val = mMean[row].to(Float32)

        # Pass 1: compute local partial sums of wdy and x_hat*wdy.
        local_wdy = Float32(0.0)
        local_xhatwdy = Float32(0.0)
        for col in cutlass.range(tidx, N, block_threads):
            x_val = mX[row, col].to(Float32)
            dy_val = mdO[row, col].to(Float32)
            gamma = mW[col].to(Float32)
            x_mu = x_val - mean_val
            x_hat = x_mu * rstd_val
            wdy = dy_val * gamma
            local_wdy += wdy
            local_xhatwdy += x_hat * wdy

        # Warp-level reduction, then block-level reduction via shared memory.
        red_op = operator.add  # type: ignore[assignment]
        local_wdy = warp_reduce(local_wdy, red_op)
        local_xhatwdy = warp_reduce(local_xhatwdy, red_op)

        if lane == 0:
            warp_sums_wdy[warp_idx] = local_wdy
            warp_sums_xhatwdy[warp_idx] = local_xhatwdy

        cute.arch.barrier()

        total_wdy = Float32(0.0)
        total_xhatwdy = Float32(0.0)
        if warp_idx == 0 and lane == 0:
            for wi in cutlass.range_constexpr(num_warps):
                total_wdy += warp_sums_wdy[wi]
                total_xhatwdy += warp_sums_xhatwdy[wi]
            # Store totals back into first slots for broadcast.
            warp_sums_wdy[0] = total_wdy
            warp_sums_xhatwdy[0] = total_xhatwdy

        cute.arch.barrier()

        total_wdy = warp_sums_wdy[0]
        total_xhatwdy = warp_sums_xhatwdy[0]
        inv_N = Float32(1.0 / float(N))
        mean_wdy = total_wdy * inv_N
        mean_xhatwdy = total_xhatwdy * inv_N

        # Pass 2: compute dx and write back.
        for col in cutlass.range(tidx, N, block_threads):
            x_val = mX[row, col].to(Float32)
            dy_val = mdO[row, col].to(Float32)
            gamma = mW[col].to(Float32)
            x_mu = x_val - mean_val
            x_hat = x_mu * rstd_val
            wdy = dy_val * gamma
            dx_val = (wdy - mean_wdy - x_hat * mean_xhatwdy) * rstd_val
            mdX[row, col] = dx_val.to(mdX.element_type)


@cute.jit
def _layernorm_backward_dx(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mdO: cute.Tensor,
    mRstd: cute.Tensor,
    mMean: cute.Tensor,
    mdX: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """
    JIT wrapper that launches the dx-only LayerNorm backward kernel.
    One CTA processes one row of length N with 256 threads.
    """
    M = mX.shape[0]
    _layernorm_backward_dx_kernel(
        mX,
        mW,
        mdO,
        mRstd,
        mMean,
        mdX,
    ).launch(
        grid=[M, 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )


@cute.kernel
def _layernorm_backward_param_kernel(
    mX: cute.Tensor,
    mdO: cute.Tensor,
    mRstd: cute.Tensor,
    mMean: cute.Tensor,
    mdW_partial: Optional[cute.Tensor],
    mdB_partial: Optional[cute.Tensor],
    num_blocks: Int32,
) -> None:
    """
    Parameter-gradient kernel for LayerNorm.

    Each CTA accumulates partial dweight/dbias over a stripe of rows:
      - Grid dim X: num_blocks (sm_count-style persistent CTAs).
      - Threads in a CTA partition the N dimension.
      - For each assigned column, a thread streams over rows
        row = blockIdx.x, blockIdx.x + num_blocks, ...

    This mirrors the persistent-CTA pattern used by RMSNorm backward,
    but uses a simpler per-thread accumulation since columns are
    independent.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_threads = const_expr(256)
    M = mX.shape[0]
    N = mX.shape[1]

    if bidx < num_blocks:
        for col in cutlass.range(tidx, N, block_threads):
            dw_local = Float32(0.0)
            db_local = Float32(0.0)
            for row in cutlass.range(bidx, M, num_blocks):
                x_val = mX[row, col].to(Float32)
                dy_val = mdO[row, col].to(Float32)
                rstd_val = mRstd[row].to(Float32)
                mean_val = mMean[row].to(Float32)
                x_mu = x_val - mean_val
                x_hat = x_mu * rstd_val
                dw_local += dy_val * x_hat
                db_local += dy_val

            if const_expr(mdW_partial is not None):
                mdW_partial[bidx, col] = dw_local
            if const_expr(mdB_partial is not None):
                mdB_partial[bidx, col] = db_local


@cute.jit
def _layernorm_backward_param(
    mX: cute.Tensor,
    mdO: cute.Tensor,
    mRstd: cute.Tensor,
    mMean: cute.Tensor,
    mdW_partial: Optional[cute.Tensor],
    mdB_partial: Optional[cute.Tensor],
    num_blocks: Int32,
    stream: cuda.CUstream,
) -> None:
    """
    JIT wrapper that launches the parameter-gradient kernel.
    """
    _layernorm_backward_param_kernel(
        mX,
        mdO,
        mRstd,
        mMean,
        mdW_partial,
        mdB_partial,
        num_blocks,
    ).launch(
        grid=[num_blocks, 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )


def _layernorm_backward_dx_sm100(
    dout_2d: Tensor,
    x_2d: Tensor,
    weight: Tensor,
    rstd_1d: Tensor,
    mean_1d: Tensor,
    dx_2d: Tensor,
) -> None:
    """
    Host-side helper to run the dx-only LayerNorm backward kernel.
    """
    M, N = x_2d.shape
    assert dout_2d.shape == (M, N)
    assert rstd_1d.numel() == M
    assert mean_1d.numel() == M

    dtype = TORCH2CUTE_DTYPE[x_2d.dtype]

    mX = _convert_row_major(x_2d)
    mdO = _convert_row_major(dout_2d)
    mdX = _convert_row_major(dx_2d)

    mW = convert_from_dlpack_cute(
        weight.detach(),
        leading_dim=0,
        alignment=16,
        divisibility=128 // cutlass.Float32.width,
    )
    mRstd = from_dlpack(rstd_1d.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
    mMean = from_dlpack(mean_1d.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (N, dtype)
    compiled = _BWD_DX_COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            _layernorm_backward_dx,
            mX,
            mW,
            mdO,
            mRstd,
            mMean,
            mdX,
            stream,
        )
        _BWD_DX_COMPILE_CACHE[key] = compiled

    compiled(
        mX,
        mW,
        mdO,
        mRstd,
        mMean,
        mdX,
        stream,
    )


def _layernorm_backward_params_sm100(
    dout_2d: Tensor,
    x_2d: Tensor,
    rstd_1d: Tensor,
    mean_1d: Tensor,
    dw_partial: Optional[Tensor],
    db_partial: Optional[Tensor],
    sm_count: int,
) -> None:
    """
    Host-side helper to run the parameter-gradient kernel that populates
    dw_partial / db_partial of shape (sm_count, N).
    """
    M, N = x_2d.shape
    assert dout_2d.shape == (M, N)
    assert rstd_1d.numel() == M
    assert mean_1d.numel() == M
    if dw_partial is None and db_partial is None:
        return

    dtype = TORCH2CUTE_DTYPE[x_2d.dtype]

    mX = _convert_row_major(x_2d)
    mdO = _convert_row_major(dout_2d)
    mRstd = from_dlpack(rstd_1d.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
    mMean = from_dlpack(mean_1d.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)

    mdW_partial = (
        from_dlpack(dw_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if dw_partial is not None
        else None
    )
    mdB_partial = (
        from_dlpack(db_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if db_partial is not None
        else None
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    has_bias = db_partial is not None
    key = (N, dtype, has_bias)
    compiled = _BWD_PARAM_COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            _layernorm_backward_param,
            mX,
            mdO,
            mRstd,
            mMean,
            mdW_partial,
            mdB_partial,
            Int32(sm_count),
            stream,
        )
        _BWD_PARAM_COMPILE_CACHE[key] = compiled

    compiled(
        mX,
        mdO,
        mRstd,
        mMean,
        mdW_partial,
        mdB_partial,
        Int32(sm_count),
        stream,
    )


def layernorm_backward(
    dout: Tensor,
    x: Tensor,
    weight: Tensor,
    rstd: Tensor,
    mean: Tensor,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    LayerNorm backward implemented in CuteDSL / CUTLASS.

    Computes gradients w.r.t. input, weight, and optional bias using
    two kernels:
      - A dx kernel (CTA-per-row) that streams over N.
      - A parameter-gradient kernel that accumulates dw/db over a
        persistent grid of CTAs across the M dimension.
    """
    assert x.shape == dout.shape, "x and dout must have the same shape"
    assert x.is_cuda and dout.is_cuda, "x and dout must be CUDA tensors"
    assert weight.dim() == 1, "weight must be 1D"
    if bias is not None:
        assert bias.dim() == 1, "bias must be 1D"

    x_2d, orig_shape = _as_2d(x)
    dout_2d, _ = _as_2d(dout)
    M, N = x_2d.shape

    # Flatten to 2D for the kernels.
    mean_flat = mean.view(M)
    rstd_flat = rstd.view(M)

    dx_2d = torch.empty_like(x_2d)
    _layernorm_backward_dx_sm100(
        dout_2d,
        x_2d,
        weight,
        rstd_flat,
        mean_flat,
        dx_2d,
    )

    device = x.device
    sm_count = get_sm_count(N, device, M=M, dtype=x.dtype)

    dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)
    db_partial = (
        torch.empty(sm_count, N, device=device, dtype=torch.float32)
        if bias is not None
        else None
    )

    _layernorm_backward_params_sm100(
        dout_2d,
        x_2d,
        rstd_flat,
        mean_flat,
        dw_partial,
        db_partial,
        sm_count,
    )

    dweight = dw_partial.sum(dim=0).to(weight.dtype)
    dbias = db_partial.sum(dim=0).to(bias.dtype) if bias is not None else None

    dx = _restore_shape(dx_2d, orig_shape)
    return dx, dweight, dbias


if __name__ == "__main__":
    # Allow direct execution for a quick functional check.
    if not torch.cuda.is_available():
        print("CUDA not available; LayerNormSM100 test skipped.")
        raise SystemExit(0)

    device = "cuda"
    M, N = 2048, 4096
    dtype = torch.bfloat16
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)

    y_ref = layernorm_ref(x, w, b)
    y, rstd, mean = layernorm(x, w, b, return_rstd=True, return_mean=True)
    torch.testing.assert_close(
        y,
        y_ref,
        atol=5e-2 if dtype != torch.float32 else 1e-5,
        rtol=5e-2 if dtype != torch.float32 else 1e-5,
    )

    print("LayerNormSM100 forward correctness check passed.")
