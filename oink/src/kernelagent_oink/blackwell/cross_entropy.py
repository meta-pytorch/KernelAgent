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
Cross-entropy forward + backward kernels for SM100 (Blackwell) in CuteDSL.

This module implements numerically stable cross-entropy over the last
dimension of 2D logits tensors `(M, N)` together with its backward pass,
targeting SM100 with Quack-style tiling, cp.async pipelines, and (for the
forward pass) optional cluster-wide online softmax reductions, but without
depending on the external `quack` package at runtime.

Public APIs:

- ``cross_entropy_forward(logits, target, ignore_index=-100, reduction="none")``
  returns ``(loss, lse)`` where ``loss`` follows the requested reduction and
  ``lse`` is always per-example log-sum-exp (shape ``(M,)``).
- ``cross_entropy_backward(dloss, logits, target, lse, ignore_index=-100)``
  returns per-logit gradients ``dlogits`` matching PyTorch /
  ``quack.cross_entropy_bwd`` semantics for ``reduction="none"``.
- ``cross_entropy(logits, target, ignore_index=-100, reduction="mean"|"sum"|"none")``
  is a convenience wrapper that mirrors ``torch.nn.functional.cross_entropy``
  reductions using the SM100 CuteDSL kernels for the forward pass.

The kernels are self-contained and use only local helpers in
`kernelagent_oink.blackwell.lite_quack` plus CuTeDSL/CUTLASS.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import re
from typing import Literal, Optional, Type

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
        "kernelagent_oink.blackwell.cross_entropy requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import runtime as rt
from cutlass.cute.runtime import from_dlpack

from kernelagent_oink.blackwell.lite_quack import (
    _KERNEL_ACCEPTS_LAYOUT_ARGS,
    TORCH2CUTE_DTYPE,
    ReductionBase,
    domain_offset_i64,
    fill_oob,
    online_softmax_reduce,
    predicate_k,
)

_FWD_COMPILE_CACHE: dict[tuple[type[cutlass.Numeric], int], cute.Kernel] = {}
_BWD_COMPILE_CACHE: dict[tuple[type[cutlass.Numeric], int], cute.Kernel] = {}
_PTR_FWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_PTR_BWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _convert_logits_2d(x: Tensor) -> cute.Tensor:
    """Convert a 2D logits tensor (M, N) into a CuTe tensor.

    We assume 16-byte alignment and mark the layout compact and row-major
    in the last dimension, matching the conventions used in the SM100
    softmax and RMSNorm kernels.
    """
    assert x.dim() == 2, "Input logits must be 2D (M, N)"
    return (
        from_dlpack(x.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )


def _convert_1d(t: Tensor, assumed_align: int) -> cute.Tensor:
    """Convert a 1D tensor with a fully dynamic layout."""
    assert t.dim() == 1, "Expected a 1D tensor"
    return from_dlpack(t.detach(), assumed_align=assumed_align).mark_layout_dynamic()


class CrossEntropyFwdSM100(ReductionBase):
    """SM100-tuned cross-entropy forward kernel.

    This mirrors the structure of ``quack.cross_entropy.CrossEntropy`` but
    is simplified to always use the single-pass online softmax reduction and
    never computes gradients inside the forward kernel.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # Use one stage with an Int64 reduction buffer packing (max, sum_exp)
        # pairs via lite_quack.online_softmax_reduce.
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Int64)

    def _calculate_threads_per_row(self) -> int:
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
        # Match Quack's cluster_n growth policy while keeping it explicit so
        # we can tune SM100-specific shapes later if needed.
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
        else:  # fp32
            cluster_n = (
                1
                if N <= 16 * 1024
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
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        ignore_index: Int32,
        stream: cuda.CUstream,
    ) -> None:
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        # If N is not divisible by the full 128-bit vector width, step down
        # to the largest compatible vector size as in Quack.
        num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
        num_threads = (
            cute.size(tv_layout, mode=[0]) if _KERNEL_ACCEPTS_LAYOUT_ARGS else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(
                mX,
                mTarget,
                mLoss,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mTarget,
                mLoss,
                mLSE,
                ignore_index,
            )
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_logits: cute.Pointer,
        ptr_target: cute.Pointer,
        ptr_loss: cute.Pointer,
        ptr_lse: cute.Pointer,
        M: Int32,
        ld: Int32,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions."""
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        layout_m = cute.make_layout((M,), stride=(1,))
        mX = cute.make_tensor(ptr_logits, layout_mn)
        mTarget = cute.make_tensor(ptr_target, layout_m)
        mLoss = cute.make_tensor(ptr_loss, layout_m)
        mLSE = cute.make_tensor(ptr_lse, layout_m)
        self.__call__(mX, mTarget, mLoss, mLSE, ignore_index, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        ignore_index: Int32,  # Index to ignore in loss computation
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape: cute.Shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Slice per-CTA region; use 64-bit indexing for large tensors.
        mX_off = domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX_off, tiler_mn, (0, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # Copy setup: gmem -> smem via cp.async, 128-bit or narrower as needed.
        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            gX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # Fill out-of-bounds values with -inf so they are ignored in max/sum.
        if const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        should_ignore = Boolean(target == ignore_index)

        # Load the target logit if this row is not ignored. Use Int64 indexing
        # to safely handle very large tensors.
        target_logit = Float32.zero
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            mX_row = domain_offset_i64((row, 0), mX)
            target_logit = Float32(mX_row[0, target])

        threads_per_row = tv_layout.shape[0][0]
        max_x, denom, _ = online_softmax_reduce(
            x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            phase=None,
            return_exp_x=False,
        )

        # Write loss and lse to gmem. Only one CTA in the cluster writes to
        # avoid duplicate stores.
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(denom, fastmath=True)
            loss_val = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,  # (M, N)
            mTarget: cute.Tensor,  # (M,)
            mLoss: cute.Tensor,  # (M,)
            mLSE: Optional[cute.Tensor],  # (M,)
            ignore_index: Int32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(
                mX,
                mTarget,
                mLoss,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,  # (M, N)
            mTarget: cute.Tensor,  # (M,)
            mLoss: cute.Tensor,  # (M,)
            mLSE: Optional[cute.Tensor],  # (M,)
            ignore_index: Int32,
        ) -> None:
            num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
            tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
            self._kernel_impl(
                mX,
                mTarget,
                mLoss,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )


class CrossEntropyBackwardSM100:
    """SM100-tuned cross-entropy backward kernel.

    This is a direct port of ``quack.cross_entropy.CrossEntropyBackward`` to
    the local lite_quack helpers, using cp.async tiling over the (M, N)
    logits and broadcasting ``dloss`` / ``lse`` across the row dimension.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N

    def _get_num_threads(self) -> int:
        # Keep in sync with _get_tv_layout() (we tile N in 16k blocks).
        N = min(self.N, 16384)
        return 128 if N <= 16384 else 256

    def _calculate_threads_per_row(self) -> int:
        N = min(self.N, 16384)  # We split by blocks of 16k in N.
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
            )
        )

    def _get_tv_layout(self, num_copy_bits: int = 128) -> tuple[cute.Shape, cute.Layout]:
        vecsize = num_copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        N = min(self.N, 16384)
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        ignore_index: Int32,  # Index to ignore in gradient computation
        stream: cuda.CUstream,
    ) -> None:
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
        num_threads = (
            cute.size(tv_layout, mode=[0]) if _KERNEL_ACCEPTS_LAYOUT_ARGS else self._get_num_threads()
        )
        # Broadcast (M,) tensors along the N dimension with stride 0.
        mDLoss, mTarget, mLSE = [
            cute.make_tensor(
                X.iterator,
                cute.append(X.layout, cute.make_layout((self.N,), stride=(0,))),
            )
            for X in (mDLoss, mTarget, mLSE)
        ]
        smem_size = cute.size_in_bytes(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
        )
        kernel = (
            self.kernel(
                mX,
                mTarget,
                mDLoss,
                mdX,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mTarget,
                mDLoss,
                mdX,
                mLSE,
                ignore_index,
            )
        )
        kernel.launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                cute.ceil_div(mX.shape[1], tiler_mn[1]),
                1,
            ],
            block=[num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_logits: cute.Pointer,
        ptr_target: cute.Pointer,
        ptr_dloss: cute.Pointer,
        ptr_dx: cute.Pointer,
        ptr_lse: cute.Pointer,
        M: Int32,
        ld: Int32,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions."""
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        layout_m = cute.make_layout((M,), stride=(1,))
        mX = cute.make_tensor(ptr_logits, layout_mn)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        mTarget = cute.make_tensor(ptr_target, layout_m)
        mDLoss = cute.make_tensor(ptr_dloss, layout_m)
        mLSE = cute.make_tensor(ptr_lse, layout_m)
        self.__call__(mX, mTarget, mDLoss, mdX, mLSE, ignore_index, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mDLoss: cute.Tensor,  # (M,)
        mdX: cute.Tensor,  # (M, N)
        mLSE: cute.Tensor,  # (M,)
        ignore_index: Int32,  # Index to ignore in gradient computation
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        shape = mX.shape

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        idX = cute.make_identity_tensor(shape)
        mX_off, mdX_off = [
            domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mdX)
        ]
        gX, gdX = [cute.local_tile(mT, tiler_mn, (0, bidy)) for mT in (mX_off, mdX_off)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))

        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            gX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        copy_atom_store_dX = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gdX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_dX = cute.make_tiled_copy(copy_atom_store_dX, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy_X.partition_S(cX)
        tXgdX = thr_copy_dX.partition_D(gdX)

        tXrX, tXrdX = [cute.make_fragment_like(thr) for thr in (tXgX, tXgdX)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        row = tXcX[0][0]
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target = Int32.zero
        dloss = Float32.zero
        lse = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            should_ignore = Boolean(target == ignore_index)
            dloss = Float32(mDLoss[row]) if not should_ignore else Float32.zero
            lse = Float32(mLSE[row])

        log2_e = math.log2(math.e)
        probs = cute.math.exp2(x * log2_e - (lse * log2_e), fastmath=True)
        prob_shifted = probs - 1.0
        mask = cute.make_fragment_like(tXrX, cutlass.Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss

        tXrdX.store(grad.to(tXrdX.element_type))
        tXpdX = (
            predicate_k(thr_copy_dX.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_store_dX, tXrdX, tXgdX, pred=tXpdX)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,  # (M, N)
            mTarget: cute.Tensor,  # (M,)
            mDLoss: cute.Tensor,  # (M,)
            mdX: cute.Tensor,  # (M, N)
            mLSE: cute.Tensor,  # (M,)
            ignore_index: Int32,  # Index to ignore in gradient computation
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(
                mX,
                mTarget,
                mDLoss,
                mdX,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,  # (M, N)
            mTarget: cute.Tensor,  # (M,)
            mDLoss: cute.Tensor,  # (M,)
            mdX: cute.Tensor,  # (M, N)
            mLSE: cute.Tensor,  # (M,)
            ignore_index: Int32,  # Index to ignore in gradient computation
        ) -> None:
            num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
            tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
            self._kernel_impl(
                mX,
                mTarget,
                mDLoss,
                mdX,
                mLSE,
                ignore_index,
                tv_layout,
                tiler_mn,
            )


def cross_entropy_forward(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> tuple[Tensor, Tensor]:
    """SM100 CuteDSL cross-entropy forward pass.

    Args:
        logits: Tensor of shape ``(M, N)`` on CUDA.
        target: Tensor of shape ``(M,)`` with integer class indices.
        ignore_index: Target value to ignore when computing the loss.
        reduction: One of ``"none"``, ``"mean"``, or ``"sum"`` following
            ``torch.nn.functional.cross_entropy`` semantics.

    Returns:
        A tuple ``(loss, lse)`` where:
        - ``loss`` has shape ``(M,)`` if ``reduction="none"`` or is a scalar
          otherwise.
        - ``lse`` is the per-example log-sum-exp with shape ``(M,)``.
    """
    assert logits.dim() == 2, "logits must be 2D (M, N)"
    assert target.dim() == 1, "target must be 1D (M,)"
    assert logits.shape[0] == target.shape[0], "Batch dimensions must match"
    assert logits.is_cuda and target.is_cuda, "logits and target must be on CUDA device"
    assert logits.dtype in TORCH2CUTE_DTYPE, "Unsupported logits dtype"
    assert target.dtype in (torch.int32, torch.int64), "target must be int32 or int64"

    M, N = logits.shape
    device = logits.device
    dtype_cute = TORCH2CUTE_DTYPE[logits.dtype]

    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32)

    if _can_use_ptr_path_logits(logits) and _can_use_ptr_path_target(target):
        _cross_entropy_forward_ptr_into(
            logits=logits,
            target=target,
            loss=loss,
            lse=lse,
            ignore_index=int(ignore_index),
        )
        if reduction == "none":
            return loss, lse
        with torch.no_grad():
            mask = target != ignore_index
            if reduction == "sum":
                reduced = loss.sum()
            elif reduction == "mean":
                valid = mask.sum()
                if valid > 0:
                    reduced = loss[mask].sum() / valid.to(loss.dtype)
                else:
                    reduced = loss.sum() * 0.0
            else:
                raise ValueError(
                    f"Invalid reduction mode: {reduction}. Expected 'none', 'mean', or 'sum'."
                )
        return reduced, lse

    mX = _convert_logits_2d(logits)
    mTarget = _convert_1d(target.to(torch.int64), assumed_align=8)
    mLoss = _convert_1d(loss, assumed_align=4)
    mLSE = _convert_1d(lse, assumed_align=4)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype_cute, N)
    kernel = _FWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = CrossEntropyFwdSM100(dtype_cute, N)
        kernel = cute.compile(
            op,
            mX,
            mTarget,
            mLoss,
            mLSE,
            Int32(ignore_index),
            current_stream,
        )
        _FWD_COMPILE_CACHE[compile_key] = kernel

    kernel(mX, mTarget, mLoss, mLSE, Int32(ignore_index), current_stream)

    if reduction == "none":
        return loss, lse

    with torch.no_grad():
        mask = target != ignore_index
        if reduction == "sum":
            reduced = loss.sum()
        elif reduction == "mean":
            valid = mask.sum()
            if valid > 0:
                reduced = loss[mask].sum() / valid.to(loss.dtype)
            else:
                reduced = loss.sum() * 0.0
        else:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Expected 'none', 'mean', or 'sum'."
            )
    return reduced, lse


def _cross_entropy_backward_sm100(
    logits: Tensor,
    target: Tensor,
    dloss: Tensor,
    lse: Tensor,
    dx: Tensor,
    ignore_index: int = -100,
) -> None:
    """Internal SM100 cross-entropy backward dispatch using CuteDSL."""
    assert logits.dim() == 2, "logits must be 2D (M, N)"
    assert target.dim() == 1, "target must be 1D (M,)"
    assert dloss.dim() == 1, "dloss must be 1D (M,)"
    assert lse.dim() == 1, "lse must be 1D (M,)"
    assert logits.shape[0] == target.shape[0] == dloss.shape[0] == lse.shape[0], (
        "Batch dimensions must match"
    )
    assert logits.is_cuda and target.is_cuda and dloss.is_cuda and lse.is_cuda, (
        "All tensors must be on CUDA device"
    )
    assert logits.dtype in TORCH2CUTE_DTYPE, "Unsupported logits dtype"
    assert target.dtype in (torch.int32, torch.int64), "target must be int32 or int64"

    M, N = logits.shape
    dtype_cute = TORCH2CUTE_DTYPE[logits.dtype]

    if (
        _can_use_ptr_path_logits(logits)
        and _can_use_ptr_path_logits(dx)
        and _can_use_ptr_path_target(target)
        and _can_use_ptr_path_f32_1d(dloss)
        and _can_use_ptr_path_f32_1d(lse)
        and logits.stride() == dx.stride()
    ):
        _cross_entropy_backward_ptr_into(
            logits=logits,
            target=target,
            dloss=dloss,
            lse=lse,
            dx=dx,
            ignore_index=int(ignore_index),
        )
        return

    mX = _convert_logits_2d(logits)
    mdX = _convert_logits_2d(dx)
    mTarget = _convert_1d(target.to(torch.int64), assumed_align=8)
    mDLoss = _convert_1d(dloss.to(torch.float32), assumed_align=4)
    mLSE = _convert_1d(lse.to(torch.float32), assumed_align=4)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype_cute, N)
    kernel = _BWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = CrossEntropyBackwardSM100(dtype_cute, N)
        kernel = cute.compile(
            op,
            mX,
            mTarget,
            mDLoss,
            mdX,
            mLSE,
            Int32(ignore_index),
            current_stream,
        )
        _BWD_COMPILE_CACHE[compile_key] = kernel

    kernel(mX, mTarget, mDLoss, mdX, mLSE, Int32(ignore_index), current_stream)


def _can_use_ptr_path_logits(x: Tensor) -> bool:
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.dtype not in TORCH2CUTE_DTYPE:
        return False
    if x.stride(1) != 1:
        return False
    if (x.data_ptr() % 16) != 0:
        return False
    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    divby = 128 // dtype_x.width
    if (x.stride(0) % divby) != 0:
        return False
    return True


def _can_use_ptr_path_target(t: Tensor) -> bool:
    if not t.is_cuda or t.dim() != 1:
        return False
    if t.dtype is not torch.int64:
        return False
    if not t.is_contiguous():
        return False
    if t.stride(0) != 1:
        return False
    if (t.data_ptr() % 8) != 0:
        return False
    return True


def _can_use_ptr_path_f32_1d(t: Tensor) -> bool:
    if not t.is_cuda or t.dim() != 1:
        return False
    if t.dtype is not torch.float32:
        return False
    if not t.is_contiguous():
        return False
    if t.stride(0) != 1:
        return False
    if (t.data_ptr() % 4) != 0:
        return False
    return True


def _cross_entropy_forward_ptr_into(
    *,
    logits: Tensor,
    target: Tensor,
    loss: Tensor,
    lse: Tensor,
    ignore_index: int,
) -> None:
    assert logits.is_cuda and logits.dim() == 2
    assert target.is_cuda and target.dim() == 1 and target.shape[0] == logits.shape[0]
    assert target.dtype is torch.int64
    assert loss.is_cuda and loss.shape == (logits.shape[0],) and loss.dtype is torch.float32
    assert lse.is_cuda and lse.shape == (logits.shape[0],) and lse.dtype is torch.float32

    M, N = logits.shape
    device_index = logits.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    dtype_x = TORCH2CUTE_DTYPE[logits.dtype]
    key = ("ptr_fwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_FWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = CrossEntropyFwdSM100(dtype_x, int(N))
        ptr_logits = rt.make_ptr(
            dtype_x, logits.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_target = rt.make_ptr(
            cutlass.Int64,
            target.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=8,
        )
        ptr_loss = rt.make_ptr(
            cutlass.Float32,
            loss.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_lse = rt.make_ptr(
            cutlass.Float32,
            lse.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_logits,
            ptr_target,
            ptr_loss,
            ptr_lse,
            Int32(int(M)),
            Int32(int(logits.stride(0))),
            Int32(int(ignore_index)),
            stream,
        )
        _PTR_FWD_COMPILE_CACHE[key] = compiled

    ptr_logits = rt.make_ptr(
        dtype_x, logits.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_target = rt.make_ptr(
        cutlass.Int64,
        target.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=8,
    )
    ptr_loss = rt.make_ptr(
        cutlass.Float32,
        loss.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_lse = rt.make_ptr(
        cutlass.Float32,
        lse.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    compiled(
        ptr_logits,
        ptr_target,
        ptr_loss,
        ptr_lse,
        Int32(int(M)),
        Int32(int(logits.stride(0))),
        Int32(int(ignore_index)),
        stream,
    )


def _cross_entropy_backward_ptr_into(
    *,
    logits: Tensor,
    target: Tensor,
    dloss: Tensor,
    lse: Tensor,
    dx: Tensor,
    ignore_index: int,
) -> None:
    assert logits.is_cuda and logits.dim() == 2
    assert target.is_cuda and target.dim() == 1 and target.shape[0] == logits.shape[0]
    assert target.dtype is torch.int64
    assert dloss.is_cuda and dloss.shape == (logits.shape[0],) and dloss.dtype is torch.float32
    assert lse.is_cuda and lse.shape == (logits.shape[0],) and lse.dtype is torch.float32
    assert dx.is_cuda and dx.shape == logits.shape and dx.dtype == logits.dtype
    assert dx.stride() == logits.stride(), "Pointer path expects dx to match logits strides"

    M, N = logits.shape
    device_index = logits.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    dtype_x = TORCH2CUTE_DTYPE[logits.dtype]
    key = ("ptr_bwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_BWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = CrossEntropyBackwardSM100(dtype_x, int(N))
        ptr_logits = rt.make_ptr(
            dtype_x, logits.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_target = rt.make_ptr(
            cutlass.Int64,
            target.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=8,
        )
        ptr_dloss = rt.make_ptr(
            cutlass.Float32,
            dloss.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_dx = rt.make_ptr(
            dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_lse = rt.make_ptr(
            cutlass.Float32,
            lse.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_logits,
            ptr_target,
            ptr_dloss,
            ptr_dx,
            ptr_lse,
            Int32(int(M)),
            Int32(int(logits.stride(0))),
            Int32(int(ignore_index)),
            stream,
        )
        _PTR_BWD_COMPILE_CACHE[key] = compiled

    ptr_logits = rt.make_ptr(
        dtype_x, logits.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_target = rt.make_ptr(
        cutlass.Int64,
        target.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=8,
    )
    ptr_dloss = rt.make_ptr(
        cutlass.Float32,
        dloss.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_dx = rt.make_ptr(dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16)
    ptr_lse = rt.make_ptr(
        cutlass.Float32,
        lse.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    compiled(
        ptr_logits,
        ptr_target,
        ptr_dloss,
        ptr_dx,
        ptr_lse,
        Int32(int(M)),
        Int32(int(logits.stride(0))),
        Int32(int(ignore_index)),
        stream,
    )


def cross_entropy_backward(
    dloss: Tensor,
    logits: Tensor,
    target: Tensor,
    lse: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """SM100 CuteDSL cross-entropy backward pass.

    Args:
        dloss: Upstream gradient of shape ``(M,)`` corresponding to
            ``reduction="none"``.
        logits: Input logits tensor of shape ``(M, N)``.
        target: Integer class indices of shape ``(M,)``.
        lse: Per-example log-sum-exp tensor of shape ``(M,)`` as returned
            by :func:`cross_entropy_forward`.
        ignore_index: Target value to ignore in gradient computation.

    Returns:
        ``dlogits`` of shape ``(M, N)`` with the same dtype as ``logits``.
    """
    assert logits.dim() == 2, "logits must be 2D (M, N)"
    assert dloss.dim() == 1, "dloss must be 1D (M,)"
    assert logits.size(0) == dloss.size(0), "Batch dimensions must match"
    assert logits.is_cuda and dloss.is_cuda, "logits and dloss must be on CUDA device"

    dx = torch.empty_like(logits)
    _cross_entropy_backward_sm100(
        logits,
        target,
        dloss,
        lse,
        dx,
        ignore_index=ignore_index,
    )
    return dx


def cross_entropy(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Convenience wrapper mirroring ``torch.nn.functional.cross_entropy`` reductions.

    This uses :func:`cross_entropy_forward` under the hood but returns only
    the reduced loss tensor.
    """
    loss, _lse = cross_entropy_forward(
        logits,
        target,
        ignore_index=ignore_index,
        reduction="none",
    )
    if reduction == "none":
        return loss
    mask = target != ignore_index
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        valid = mask.sum()
        if valid > 0:
            return loss[mask].sum() / valid.to(loss.dtype)
        return loss.sum() * 0.0
    raise ValueError(
        f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'."
    )


def verify_cross_entropy_parity(
    M: int,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    ignore_index: int = -100,
) -> None:
    """Compare SM100 CuteDSL cross-entropy against PyTorch for a single shape."""
    device = torch.device("cuda")
    torch.manual_seed(0)

    logits = 0.1 * torch.randn(M, N, device=device, dtype=dtype)
    logits.requires_grad_(True)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    # Optionally sprinkle some ignore_index entries for robustness.
    if ignore_index != -100:
        mask = torch.rand(M, device=device) < 0.1
        target[mask] = ignore_index

    loss, lse = cross_entropy_forward(logits, target, ignore_index=ignore_index, reduction="none")

    logits_ref = logits.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.float(),
        target_ref,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Forward parity
    if dtype in (torch.float16, torch.bfloat16):
        atol = 5e-2
        rtol = 5e-2
    else:
        atol = 1e-4
        rtol = 1e-4
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)

    # Backward parity
    dloss = torch.randn_like(loss_ref)
    (dx_ref,) = torch.autograd.grad(loss_ref, logits_ref, grad_outputs=dloss)
    dx = cross_entropy_backward(dloss, logits, target, lse, ignore_index=ignore_index)
    torch.testing.assert_close(dx, dx_ref.to(logits.dtype), atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Minimal functional check when executed directly. For performance
    # comparisons and detailed tuning, use the dedicated benchmark harness.
    if not torch.cuda.is_available():
        print("CUDA not available; cross-entropy parity check skipped.")
        raise SystemExit(0)

    M, N = 1024, 8192
    dtype = torch.bfloat16
    verify_cross_entropy_parity(M, N, dtype=dtype, ignore_index=-100)
    print("SM100 cross-entropy CuteDSL parity check passed.")
