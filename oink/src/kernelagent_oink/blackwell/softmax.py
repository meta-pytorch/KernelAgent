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
Softmax forward + backward kernels for SM100 (Blackwell) in CuteDSL.

This module implements numerically stable softmax over the last dimension of
2D tensors (M, N) and its backward pass, targeting SM100 with Quack-style
tiling, cp.async pipelines, and cluster reductions, but without depending on
the `quack` package at runtime.

The kernels are self-contained and use only local helpers in
`kernelagent_oink.blackwell.lite_quack` plus CuTeDSL/CUTLASS.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from typing import Type

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
        "kernelagent_oink.blackwell.softmax requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
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
    row_reduce,
)

_FWD_COMPILE_CACHE: dict[tuple[Type[cutlass.Numeric], int], object] = {}
_BWD_COMPILE_CACHE: dict[tuple[Type[cutlass.Numeric], int], object] = {}
_PTR_FWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_PTR_BWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


class SoftmaxFwdSM100(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # One-stage online reduction: pack (max, sum_exp) into Int64 reduction buffer.
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Int64)

    def _calculate_threads_per_row(self) -> int:
        # Match Quack's bucketed policy for Softmax.
        N = self.N
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3072:
            return 32
        if N <= 6144:
            return 64
        if N <= 16384:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
        # Quack-style growth of cluster_n with N and dtype.
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
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream) -> None:
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        # Use the generic ReductionBase tiling with 128-bit vectorization.
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(mX, mO, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mO)
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
        ptr_x: cute.Pointer,
        ptr_out: cute.Pointer,
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions.

        Reconstructs cute.Tensor views from raw pointers + explicit layouts
        inside the JIT graph, matching the existing SM100 schedule.
        """
        # Mirror Quack/LayerNorm contracts: assume 16B alignment and an LD that
        # preserves 128-bit vectorized copies for every row start.
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)
        self.__call__(mX, mO, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Slice per-CTA region; use 64-bit indexing for large tensors.
        mX, mO = [domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        # Copy atoms for gmem <-> smem and smem <-> gmem.
        # Use 128-bit cp.async for global->shared and 128-bit vectorized stores.
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gO.element_type,
            num_bits_per_copy=128,
        )

        thr_copy_load = cute.make_tiled_copy(
            copy_atom_load, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(
            copy_atom_store, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgX = thr_copy_load.partition_S(gX)
        tXsX = thr_copy_load.partition_D(sX)
        tXgO = thr_copy_store.partition_D(gO)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        # Register fragments.
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        # Predicate and cp.async pipeline for potential tail tiles.
        is_even_N = const_expr(self.N == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        threads_per_row = tv_layout.shape[0][0]

        # Online softmax reduction: compute max and sum_exp in a single pass, with
        # optional cluster-wide aggregation via an Int64 reduction buffer.
        max_x, denom, exp_x = online_softmax_reduce(
            x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            phase=None,
            return_exp_x=True,
        )

        y = exp_x * cute.arch.rcp_approx(denom)
        tXrO.store(y.to(tXrO.element_type))

        tOpO = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tOpO)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(mX, mO, tv_layout, tiler_mn)
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
        ) -> None:
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(mX, mO, tv_layout, tiler_mn)


class SoftmaxBwdSM100(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # One stage for dot(dy, y) per row.
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)

    def _calculate_threads_per_row(self) -> int:
        # Match Quack backward softmax buckets.
        N = self.N
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3072:
            return 32
        if N <= 6144:
            return 64
        if N <= 8192:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
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

    def _get_num_threads(self) -> int:
        # Slightly more aggressive threading for large N than the base class.
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps: int) -> int:
        # Store both y and dy tiles plus reduction buffers and mbarriers.
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        assert mdY.element_type == self.dtype
        assert mY.element_type == self.dtype
        assert mdX.element_type == self.dtype
        # Use the generic ReductionBase tiling with 128-bit vectorization.
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        kernel = (
            self.kernel(mdY, mY, mdX, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mdY, mY, mdX)
        )
        kernel.launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_dy: cute.Pointer,
        ptr_y: cute.Pointer,
        ptr_dx: cute.Pointer,
        M: Int32,
        ld: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Pointer-based entrypoint that bypasses DLPack conversions."""
        ld_assumed = cute.assume(ld, divby=128 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        mdY = cute.make_tensor(ptr_dy, layout_mn)
        mY = cute.make_tensor(ptr_y, layout_mn)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        self.__call__(mdY, mY, mdX, stream)

    @cute.jit
    def _kernel_impl(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)

        mdY, mY, mdX = [
            domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mdY, mY, mdX)
        ]
        gdY, gY, gdX = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mdY, mY, mdX)
        ]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sY = smem.allocate_tensor(
            mY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mdY.element_type,
            num_bits_per_copy=128,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gdX.element_type,
            num_bits_per_copy=128,
        )

        thr_copy_load = cute.make_tiled_copy(
            copy_atom_load, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(
            copy_atom_store, tv_layout, tiler_mn
        ).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tYgY = thr_copy_load.partition_S(gY)
        tYsY = thr_copy_load.partition_D(sY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tdYrdY, tYrY, tdXrdX = [
            cute.make_fragment_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)
        ]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(self.N == tiler_mn[1] * self.cluster_n)
        tdYpdY = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tYgY, tYsY, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(Float32)
        y = tYrY.load().to(Float32)

        threads_per_row = tv_layout.shape[0][0]
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))

        tdXpdX = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mdY: cute.Tensor,
            mY: cute.Tensor,
            mdX: cute.Tensor,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ) -> None:
            self._kernel_impl(mdY, mY, mdX, tv_layout, tiler_mn)
    else:

        @cute.kernel
        def kernel(
            self,
            mdY: cute.Tensor,
            mY: cute.Tensor,
            mdX: cute.Tensor,
        ) -> None:
            tiler_mn, tv_layout = self._get_tv_layout()
            self._kernel_impl(mdY, mY, mdX, tv_layout, tiler_mn)


def _convert_2d_tensor(x: Tensor) -> cute.Tensor:
    # Match Quack's Softmax conversion exactly: assume 16B alignment and mark
    # the shape compact with row-major stride order (0, 1), with mode=0 (batch).
    # We intentionally do not call mark_layout_dynamic here to avoid the
    # leading_dim stride==1 constraint used in RMSNorm.
    return from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )


def _can_use_ptr_path_2d(x: Tensor) -> bool:
    """Conservative guard for the pointer-based fast path."""
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.dtype not in TORCH2CUTE_DTYPE:
        return False
    # Require row-major last-dim contiguous.
    if x.stride(1) != 1:
        return False
    # Require 16B alignment (matches from_dlpack(..., assumed_align=16)).
    if (x.data_ptr() % 16) != 0:
        return False
    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    divby = 128 // dtype_x.width
    # Softmax uses ReductionBase default num_copy_bits=128, so N must be divisible.
    if (x.shape[1] % divby) != 0:
        return False
    # Ensure each row start remains aligned for 128-bit vectorized copies.
    if (x.stride(0) % divby) != 0:
        return False
    return True


def _softmax_forward_ptr_into(*, x: Tensor, out: Tensor) -> None:
    """Launch the pointer-based Softmax forward kernel into preallocated `out`."""
    assert x.is_cuda and x.dim() == 2
    assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype
    assert out.stride() == x.stride(), "Pointer path expects out to match x strides"

    M, N = x.shape
    device_index = x.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    key = ("ptr_fwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_FWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = SoftmaxFwdSM100(dtype_x, int(N))
        ptr_x = rt.make_ptr(
            dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_out,
            Int32(int(M)),
            ld,
            stream,
        )
        _PTR_FWD_COMPILE_CACHE[key] = compiled

    ptr_x = rt.make_ptr(
        dtype_x, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_out = rt.make_ptr(
        dtype_x, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    compiled(ptr_x, ptr_out, Int32(int(M)), Int32(int(x.stride(0))), stream)


def _softmax_backward_ptr_into(*, dy: Tensor, y: Tensor, dx: Tensor) -> None:
    """Launch the pointer-based Softmax backward kernel into preallocated `dx`."""
    assert dy.is_cuda and dy.dim() == 2
    assert y.is_cuda and y.shape == dy.shape and y.dtype == dy.dtype
    assert dx.is_cuda and dx.shape == dy.shape and dx.dtype == dy.dtype
    assert dy.stride() == y.stride() == dx.stride(), (
        "Pointer path expects matching strides"
    )

    M, N = dy.shape
    device_index = dy.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    dtype_x = TORCH2CUTE_DTYPE[dy.dtype]
    key = ("ptr_bwd", int(N), dtype_x, int(device_index))
    compiled = _PTR_BWD_COMPILE_CACHE.get(key)
    if compiled is None:
        op = SoftmaxBwdSM100(dtype_x, int(N))
        ptr_dy = rt.make_ptr(
            dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_y = rt.make_ptr(
            dtype_x, y.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_dx = rt.make_ptr(
            dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ld = Int32(int(dy.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_dy,
            ptr_y,
            ptr_dx,
            Int32(int(M)),
            ld,
            stream,
        )
        _PTR_BWD_COMPILE_CACHE[key] = compiled

    ptr_dy = rt.make_ptr(
        dtype_x, dy.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_y = rt.make_ptr(
        dtype_x, y.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_dx = rt.make_ptr(
        dtype_x, dx.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    compiled(ptr_dy, ptr_y, ptr_dx, Int32(int(M)), Int32(int(dy.stride(0))), stream)


def softmax_forward(x: Tensor) -> Tensor:
    """SM100 CuteDSL softmax forward pass: y = softmax(x, dim=-1)."""
    assert x.dim() == 2, "Input must be 2D (M, N)"
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype in TORCH2CUTE_DTYPE, "Unsupported dtype"

    N = x.size(1)
    dtype = TORCH2CUTE_DTYPE[x.dtype]
    if _can_use_ptr_path_2d(x):
        out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
        _softmax_forward_ptr_into(x=x, out=out)
        return out

    out = torch.empty_like(x)

    x_tensor = _convert_2d_tensor(x)
    out_tensor = _convert_2d_tensor(out)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    kernel = _FWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = SoftmaxFwdSM100(dtype, N)
        kernel = cute.compile(op, x_tensor, out_tensor, current_stream)
        _FWD_COMPILE_CACHE[compile_key] = kernel
    kernel(x_tensor, out_tensor, current_stream)
    return out


def softmax_backward(dy: Tensor, y: Tensor) -> Tensor:
    """SM100 CuteDSL softmax backward pass."""
    assert dy.dim() == 2 and y.dim() == 2, "dy and y must be 2D (M, N)"
    assert dy.shape == y.shape, "dy and y must have the same shape"
    assert dy.is_cuda and y.is_cuda, "dy and y must be on CUDA device"
    assert dy.dtype in TORCH2CUTE_DTYPE, "Unsupported dtype"
    assert y.dtype == dy.dtype, "dy and y must have the same dtype"

    N = dy.size(1)
    dtype = TORCH2CUTE_DTYPE[dy.dtype]
    if (
        _can_use_ptr_path_2d(dy)
        and _can_use_ptr_path_2d(y)
        and dy.stride() == y.stride()
    ):
        dx = torch.empty_strided(
            dy.shape, dy.stride(), device=dy.device, dtype=dy.dtype
        )
        _softmax_backward_ptr_into(dy=dy, y=y, dx=dx)
        return dx

    dx = torch.empty_like(dy)

    dy_tensor = _convert_2d_tensor(dy)
    y_tensor = _convert_2d_tensor(y)
    dx_tensor = _convert_2d_tensor(dx)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    kernel = _BWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = SoftmaxBwdSM100(dtype, N)
        kernel = cute.compile(op, dy_tensor, y_tensor, dx_tensor, current_stream)
        _BWD_COMPILE_CACHE[compile_key] = kernel
    kernel(dy_tensor, y_tensor, dx_tensor, current_stream)
    return dx


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = softmax_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor]:
        (y,) = ctx.saved_tensors
        dx = softmax_backward(dy, y)
        return dx


def softmax(x: Tensor) -> Tensor:
    """Autograd-friendly softmax using the SM100 CuteDSL kernel."""
    return SoftmaxFunction.apply(x)


def _torch_softmax_reference(x: Tensor) -> Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


def verify_softmax_parity(
    M: int,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> None:
    """Compare SM100 CuteDSL softmax against PyTorch for a single shape."""
    device = torch.device("cuda")
    x = torch.randn(M, N, device=device, dtype=dtype)
    x.requires_grad_(True)

    # Forward parity
    y_ref = _torch_softmax_reference(x)
    y = softmax(x)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    # Backward parity
    dy = torch.randn_like(y)
    (dx_ref,) = torch.autograd.grad(y_ref, x, dy, retain_graph=False)
    dx = softmax_backward(dy, y)
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
