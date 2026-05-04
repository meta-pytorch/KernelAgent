# ruff: noqa: E402  # CuTeDSL cache setup must run before importing cutlass.
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
LayerNorm kernels for Blackwell SM10x in CuteDSL.

This implementation:
- Mirrors Quack's LayerNorm tiling / cluster policy / cp.async pipeline
  but uses only local helpers so that it does not depend on the external
  `quack` package at runtime.
- Supports fp16 / bf16 / fp32 inputs with fp32 accumulation.
- Optionally writes out per-row `rstd` and `mean` buffers for reuse in
  backward or fused kernels.

Backward is self-contained in this repo. Validated GB300/SM103 DSv3/DSv4
hidden sizes use the fused pointer fast path; unsupported shapes/layouts use
local fallback kernels for dx and parameter gradients while matching
`torch.nn.functional.layer_norm`'s gradients numerically.
"""

from __future__ import annotations

import math
import operator
from functools import partial
from typing import Optional, Tuple, Type

import torch
from torch import Tensor

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python

from kernelagent_oink.blackwell._cutedsl_cache import ensure_versioned_cutedsl_cache_dir

ensure_versioned_cutedsl_cache_dir()

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

from kernelagent_oink.blackwell.lite_quack import (
    _KERNEL_ACCEPTS_LAYOUT_ARGS,
    TORCH2CUTE_DTYPE,
    RMSNormBackward as _LiteRMSNormBackward,
    ReductionBase as _ReductionBase,
    atomic_add_tensor_f32,
    convert_from_dlpack as convert_from_dlpack_cute,
    coord_offset_i64,
    copy as _quack_copy,
    fill_oob,
    get_copy_atom,
    get_sm_count,
    predicate_k,
    row_reduce,
    row_reduce_add,
    warp_reduce,
)
from kernelagent_oink.blackwell.fast_launch import (
    StableF32Arg,
    StableI32Arg,
    disable_fast_launch,
    fast_launch_enabled,
    set_runtime_ptr,
    tls_cache as _tls_fast_launch_cache,
)

# Simple compile cache for the forward kernel
_COMPILE_CACHE: dict[Tuple[object, ...], object] = {}
_PTR_COMPILE_CACHE: dict[Tuple[object, ...], object] = {}

# Backward compile caches: one for dx, one for parameter gradients.
_BWD_DX_COMPILE_CACHE: dict[
    Tuple[int, Type[cutlass.Numeric], Type[cutlass.Numeric], Type[cutlass.Numeric]],
    object,
] = {}
_BWD_PARAM_COMPILE_CACHE: dict[
    Tuple[int, Type[cutlass.Numeric], Type[cutlass.Numeric], bool], object
] = {}
_BWD_PTR_COMPILE_CACHE: dict[Tuple[object, ...], object] = {}
_BWD_WORKSPACE_CACHE: dict[
    Tuple[int, int, int, int, bool], Tuple[Tensor, Optional[Tensor]]
] = {}
_BWD_COMBINED_BIAS_WORKSPACE_CACHE: dict[
    Tuple[int, int, int, int], Tuple[Tensor, Tensor]
] = {}
_BWD_COMBINED_PAIR_CACHE: dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}
_BWD_ATOMIC_WORKSPACE_CACHE: dict[Tuple[int, int, int], Tensor] = {}
_BWD_REDUCTION_STREAM_CACHE: dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def _reduce_partial_sum_fp32(partial: Tensor, *, device_index: int) -> Tensor:
    """Reduce a (sm_count, N) fp32 partial buffer into an (N,) fp32 result."""
    assert partial.dtype is torch.float32
    assert partial.dim() == 2
    # On GB300, the generic reduction kernel used by `sum(dim=0)` is faster for
    # these LayerNorm partial buffers than routing the reduction through GEMM.
    _ = device_index  # kept for call-site compatibility / future tuning.
    return partial.sum(dim=0)


def _get_layernorm_bwd_workspace(
    *,
    device_index: int,
    stream_handle: int,
    sm_count: int,
    N: int,
    has_bias: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    key = (
        int(device_index),
        int(stream_handle),
        int(sm_count),
        int(N),
        bool(has_bias),
    )
    cached = _BWD_WORKSPACE_CACHE.get(key)
    if cached is not None:
        if has_bias:
            dw_partial, db_partial = cached
            assert db_partial is not None
            pair_key = (int(dw_partial.data_ptr()), int(db_partial.data_ptr()))
            if pair_key not in _BWD_COMBINED_PAIR_CACHE:
                combined_key = (
                    int(device_index),
                    int(stream_handle),
                    int(sm_count),
                    int(N),
                )
                combined_cached = _BWD_COMBINED_BIAS_WORKSPACE_CACHE.get(combined_key)
                if combined_cached is not None:
                    _BWD_COMBINED_PAIR_CACHE[pair_key] = combined_cached
        return cached

    device = torch.device("cuda", device_index)
    if has_bias:
        combined_partial = torch.empty(
            (2, sm_count, N), device=device, dtype=torch.float32
        )
        dw_partial = combined_partial[0]
        db_partial = combined_partial[1]
        reduced_pair = torch.empty((2, N), device=device, dtype=torch.float32)
        _BWD_COMBINED_BIAS_WORKSPACE_CACHE[
            (int(device_index), int(stream_handle), int(sm_count), int(N))
        ] = (combined_partial, reduced_pair)
        _BWD_COMBINED_PAIR_CACHE[
            (int(dw_partial.data_ptr()), int(db_partial.data_ptr()))
        ] = (combined_partial, reduced_pair)
    else:
        dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)
        db_partial = None
    cached = (dw_partial, db_partial)
    _BWD_WORKSPACE_CACHE[key] = cached
    return cached


def _get_layernorm_bwd_reduction_streams(
    device: torch.device,
) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    cached = _BWD_REDUCTION_STREAM_CACHE.get(int(device_index))
    if cached is not None:
        return cached
    streams = (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    )
    _BWD_REDUCTION_STREAM_CACHE[int(device_index)] = streams
    return streams


def _get_layernorm_bwd_atomic_dw_workspace(
    *,
    device_index: int,
    stream_handle: int,
    N: int,
) -> Tensor:
    key = (int(device_index), int(stream_handle), int(N))
    cached = _BWD_ATOMIC_WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached

    dw_acc = torch.empty(
        N, device=torch.device("cuda", device_index), dtype=torch.float32
    )
    _BWD_ATOMIC_WORKSPACE_CACHE[key] = dw_acc
    return dw_acc


def _finalize_layernorm_bwd_partials(
    *,
    dw_partial: Tensor,
    db_partial: Optional[Tensor],
    weight: Tensor,
    bias: Optional[Tensor],
    device: torch.device,
) -> Tuple[Tensor, Optional[Tensor]]:
    if db_partial is not None:
        pair_cached = _BWD_COMBINED_PAIR_CACHE.get(
            (int(dw_partial.data_ptr()), int(db_partial.data_ptr()))
        )
        if pair_cached is not None:
            combined_partial, reduced_pair = pair_cached
            torch.sum(combined_partial, dim=1, out=reduced_pair)
            dweight_fp32 = reduced_pair[0]
            dbias_fp32 = reduced_pair[1]
            dweight = (
                dweight_fp32.clone()
                if weight.dtype == torch.float32
                else dweight_fp32.to(weight.dtype)
            )
            assert bias is not None
            dbias = (
                dbias_fp32.clone()
                if bias.dtype == torch.float32
                else dbias_fp32.to(bias.dtype)
            )
            return dweight, dbias

    if db_partial is None:
        # No-bias is the hot DSv3 path.  Avoid auxiliary stream handshakes here;
        # the reduction is small and launch latency dominates for M=4096.
        dweight_fp32 = _reduce_partial_sum_fp32(
            dw_partial, device_index=weight.get_device()
        )
        dweight = (
            dweight_fp32
            if weight.dtype == torch.float32
            else dweight_fp32.to(weight.dtype)
        )
        return dweight, None

    dw_stream, db_stream = _get_layernorm_bwd_reduction_streams(device)
    current_stream = torch.cuda.current_stream(device=device)

    dw_stream.wait_stream(current_stream)
    db_stream.wait_stream(current_stream)

    with torch.cuda.stream(dw_stream):
        dweight_fp32 = _reduce_partial_sum_fp32(
            dw_partial, device_index=weight.get_device()
        )
        dweight = (
            dweight_fp32
            if weight.dtype == torch.float32
            else dweight_fp32.to(weight.dtype)
        )

    assert bias is not None
    with torch.cuda.stream(db_stream):
        dbias_fp32 = _reduce_partial_sum_fp32(
            db_partial, device_index=bias.get_device()
        )
        dbias = dbias_fp32 if bias.dtype == torch.float32 else dbias_fp32.to(bias.dtype)

    current_stream.wait_stream(dw_stream)
    current_stream.wait_stream(db_stream)

    return dweight, dbias


def _finalize_layernorm_bwd_atomic_dw(*, dw_acc: Tensor, weight: Tensor) -> Tensor:
    return dw_acc.clone() if weight.dtype == torch.float32 else dw_acc.to(weight.dtype)


class _PtrLayernormFastLaunch:
    def __init__(
        self,
        *,
        compiled: object,
        executor: object,
        capi_func: object,
        ptr_x: object,
        ptr_w: object,
        ptr_b: Optional[object],
        ptr_out: object,
        ptr_rstd: Optional[object],
        ptr_mean: Optional[object],
        arg_m: StableI32Arg,
        arg_ld: StableI32Arg,
        arg_eps: StableF32Arg,
        stream: cuda.CUstream,
        assumed_align_xo: int,
        packed_args: object,
        keepalive: tuple[object, ...],
    ):
        self._compiled = compiled
        self._executor = executor
        self._capi_func = capi_func
        self._ptr_x = ptr_x
        self._ptr_w = ptr_w
        self._ptr_b = ptr_b
        self._ptr_out = ptr_out
        self._ptr_rstd = ptr_rstd
        self._ptr_mean = ptr_mean
        self._arg_m = arg_m
        self._arg_ld = arg_ld
        self._arg_eps = arg_eps
        self._stream = stream
        self._assumed_align_xo = int(assumed_align_xo)
        self._packed_args = packed_args
        self._keepalive = keepalive

        self._use_fast_launch = True
        self._cuda_result = getattr(executor, "cuda_result", None)

        self._last_x_ptr = -1
        self._last_w_ptr = -1
        self._last_b_ptr = -1
        self._last_out_ptr = -1
        self._last_rstd_ptr = -1
        self._last_mean_ptr = -1
        self._last_m = -1
        self._last_ld = -1
        self._last_eps = float("nan")

    def launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        out: Tensor,
        rstd: Optional[Tensor],
        mean: Optional[Tensor],
        M: int,
        ld: int,
        eps: float,
    ) -> None:
        if not fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(
                x=x,
                weight=weight,
                bias=bias,
                out=out,
                rstd=rstd,
                mean=mean,
                M=M,
                ld=ld,
                eps=eps,
            )
            return

        x_ptr = x.data_ptr()
        if x_ptr != self._last_x_ptr:
            try:
                set_runtime_ptr(self._ptr_x, x_ptr)
                self._last_x_ptr = x_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x,
                    weight=weight,
                    bias=bias,
                    out=out,
                    rstd=rstd,
                    mean=mean,
                    M=M,
                    ld=ld,
                    eps=eps,
                )
                return

        w_ptr = weight.data_ptr()
        if w_ptr != self._last_w_ptr:
            try:
                set_runtime_ptr(self._ptr_w, w_ptr)
                self._last_w_ptr = w_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x,
                    weight=weight,
                    bias=bias,
                    out=out,
                    rstd=rstd,
                    mean=mean,
                    M=M,
                    ld=ld,
                    eps=eps,
                )
                return

        if self._ptr_b is not None and bias is not None:
            b_ptr = bias.data_ptr()
            if b_ptr != self._last_b_ptr:
                try:
                    set_runtime_ptr(self._ptr_b, b_ptr)
                    self._last_b_ptr = b_ptr
                except AttributeError:
                    self._disable_fast_launch()
                    self._fallback_launch(
                        x=x,
                        weight=weight,
                        bias=bias,
                        out=out,
                        rstd=rstd,
                        mean=mean,
                        M=M,
                        ld=ld,
                        eps=eps,
                    )
                    return

        out_ptr = out.data_ptr()
        if out_ptr != self._last_out_ptr:
            try:
                set_runtime_ptr(self._ptr_out, out_ptr)
                self._last_out_ptr = out_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x,
                    weight=weight,
                    bias=bias,
                    out=out,
                    rstd=rstd,
                    mean=mean,
                    M=M,
                    ld=ld,
                    eps=eps,
                )
                return

        if self._ptr_rstd is not None and rstd is not None:
            rstd_ptr = rstd.data_ptr()
            if rstd_ptr != self._last_rstd_ptr:
                try:
                    set_runtime_ptr(self._ptr_rstd, rstd_ptr)
                    self._last_rstd_ptr = rstd_ptr
                except AttributeError:
                    self._disable_fast_launch()
                    self._fallback_launch(
                        x=x,
                        weight=weight,
                        bias=bias,
                        out=out,
                        rstd=rstd,
                        mean=mean,
                        M=M,
                        ld=ld,
                        eps=eps,
                    )
                    return

        if self._ptr_mean is not None and mean is not None:
            mean_ptr = mean.data_ptr()
            if mean_ptr != self._last_mean_ptr:
                try:
                    set_runtime_ptr(self._ptr_mean, mean_ptr)
                    self._last_mean_ptr = mean_ptr
                except AttributeError:
                    self._disable_fast_launch()
                    self._fallback_launch(
                        x=x,
                        weight=weight,
                        bias=bias,
                        out=out,
                        rstd=rstd,
                        mean=mean,
                        M=M,
                        ld=ld,
                        eps=eps,
                    )
                    return

        if M != self._last_m:
            self._arg_m.set(M)
            self._last_m = M
        if ld != self._last_ld:
            self._arg_ld.set(ld)
            self._last_ld = ld
        if eps != self._last_eps:
            self._arg_eps.set(eps)
            self._last_eps = eps

        if self._cuda_result is not None:
            self._cuda_result.value = 0
        ret = self._capi_func(self._packed_args)  # type: ignore[misc]
        if ret != 0:
            raise RuntimeError(f"CuTeDSL capi_func returned non-zero: {ret}")
        if self._cuda_result is not None:
            err = int(self._cuda_result.value)
            if err != 0:
                raise RuntimeError(f"CuTeDSL kernel launch failed (cuda_result={err})")

    def _disable_fast_launch(self) -> None:
        self._use_fast_launch = False
        disable_fast_launch()

    def _fallback_launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        out: Tensor,
        rstd: Optional[Tensor],
        mean: Optional[Tensor],
        M: int,
        ld: int,
        eps: float,
    ) -> None:
        dtype_x = TORCH2CUTE_DTYPE[x.dtype]
        dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
        dtype_b = TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
        stream_handle = int(torch.cuda.current_stream().cuda_stream)
        stream = cuda.CUstream(stream_handle)
        ptr_x = rt.make_ptr(
            dtype_x,
            x.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_xo,
        )
        ptr_out = rt.make_ptr(
            dtype_x,
            out.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_xo,
        )
        ptr_w = rt.make_ptr(
            dtype_w,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=16,
        )
        ptr_b = (
            rt.make_ptr(
                dtype_b,
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
        self._compiled(
            ptr_x,
            ptr_w,
            ptr_b,
            ptr_out,
            ptr_rstd,
            ptr_mean,
            Int32(int(M)),
            Int32(int(ld)),
            stream,
            Float32(float(eps)),
        )


def _get_fast_ptr_layernorm_launcher(
    *,
    compiled: object,
    N: int,
    dtype_x: type[cutlass.Numeric],
    dtype_w: type[cutlass.Numeric],
    dtype_b: Optional[type[cutlass.Numeric]],
    has_bias: bool,
    has_rstd: bool,
    has_mean: bool,
    device_index: int,
    stream_handle: int,
    assumed_align_xo: int,
    eps: float,
) -> Optional[_PtrLayernormFastLaunch]:
    if not fast_launch_enabled():
        return None
    key = (
        "ptr_fast",
        id(compiled),
        int(N),
        dtype_x,
        dtype_w,
        dtype_b,
        bool(has_bias),
        bool(has_rstd),
        bool(has_mean),
        int(device_index),
        int(stream_handle),
        int(assumed_align_xo),
    )
    cache = _tls_fast_launch_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    ptr_x = rt.make_ptr(
        dtype_x, 0, mem_space=rt.AddressSpace.gmem, assumed_align=int(assumed_align_xo)
    )
    ptr_out = rt.make_ptr(
        dtype_x, 0, mem_space=rt.AddressSpace.gmem, assumed_align=int(assumed_align_xo)
    )
    ptr_w = rt.make_ptr(dtype_w, 0, mem_space=rt.AddressSpace.gmem, assumed_align=16)
    ptr_b = (
        rt.make_ptr(dtype_b, 0, mem_space=rt.AddressSpace.gmem, assumed_align=16)
        if has_bias
        else None
    )
    ptr_rstd = (
        rt.make_ptr(cutlass.Float32, 0, mem_space=rt.AddressSpace.gmem, assumed_align=4)
        if has_rstd
        else None
    )
    ptr_mean = (
        rt.make_ptr(cutlass.Float32, 0, mem_space=rt.AddressSpace.gmem, assumed_align=4)
        if has_mean
        else None
    )

    arg_m = StableI32Arg(0)
    arg_ld = StableI32Arg(N)
    arg_eps = StableF32Arg(eps)
    stream = cuda.CUstream(int(stream_handle))
    executor = compiled.to(device_index)  # type: ignore[attr-defined]

    try:
        exe_args, adapted_args = executor.generate_execution_args(
            ptr_x,
            ptr_w,
            ptr_b,
            ptr_out,
            ptr_rstd,
            ptr_mean,
            arg_m,
            arg_ld,
            stream,
            arg_eps,
        )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        disable_fast_launch()
        return None

    keepalive: tuple[object, ...] = (
        executor,
        ptr_x,
        ptr_w,
        ptr_b,
        ptr_out,
        ptr_rstd,
        ptr_mean,
        arg_m,
        arg_ld,
        arg_eps,
        stream,
        *adapted_args,
    )
    launcher = _PtrLayernormFastLaunch(
        compiled=compiled,
        executor=executor,
        capi_func=capi_func,
        ptr_x=ptr_x,
        ptr_w=ptr_w,
        ptr_b=ptr_b,
        ptr_out=ptr_out,
        ptr_rstd=ptr_rstd,
        ptr_mean=ptr_mean,
        arg_m=arg_m,
        arg_ld=arg_ld,
        arg_eps=arg_eps,
        stream=stream,
        assumed_align_xo=int(assumed_align_xo),
        packed_args=packed_args,
        keepalive=keepalive,
    )
    cache[key] = launcher
    return launcher


class _LayerNormBackwardSM103(_LiteRMSNormBackward):
    """Self-contained Oink LayerNorm backward for SM103.

    Contract:
      - math: ``x_hat = (x - mean) * rstd``;
        ``dx = (wdy - mean(wdy) - x_hat * mean(x_hat * wdy)) * rstd``;
        ``dweight = sum(dout * x_hat)`` and optional ``dbias = sum(dout)``.
      - precision: row reductions and partial parameter gradients use fp32;
        ``dx`` is stored in the input dtype and final parameter gradients are
        cast to the weight/bias dtype by the host finalizer.
      - variants: partial-buffer accumulation is the default; atomic dW/dB are
        compile-time flags enabled only by host-side shape policy.
      - dispatch: the GB300 pointer path is fenced by dtype/layout/alignment
        checks below; unsupported layouts fall back to local non-pointer kernels.

    The implementation intentionally depends only on repo-local helpers and
    never dispatches to external kernels.
    """

    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        super().__init__(dtype, N)
        self.atomic_dw = False

    def _get_num_threads(self) -> int:
        nt = getattr(self, "_nt_override", None)
        if nt is not None:
            return int(nt)
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self) -> int:
        tpr = getattr(self, "_tpr_override", None)
        if tpr is not None:
            return int(tpr)
        N = self.N
        for limit, threads in [
            (64, 8),
            (128, 16),
            (256, 32),
            (512, 64),
            (4096, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self) -> None:
        cn = getattr(self, "_cluster_n_override", None)
        if cn is not None:
            self.cluster_n = int(cn)
            return

        N = self.N
        if N <= 8192:
            cluster_n = 1
        elif self.dtype.width == 16:
            if N <= 16 * 1024:
                cluster_n = 2
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=128 // t.element_type.width),
                t.stride[1],
            )

        mX, mdO, mdX = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            for t in (mX, mdO, mdX)
        ]
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mW.element_type.width if mW is not None else 0,
                mdO.element_type.width,
                mdX.element_type.width,
            )
        )
        num_copy_bits = const_expr(128 // largest_dtype_width * mX.element_type.width)
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=int(num_copy_bits))
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout,
                cute.make_layout((tiler_mn[0],), stride=(0,)),
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        kernel = (
            self.kernel(mX, mW, mdO, mMean, mRstd, mdX, mdW, mdB, tv_layout, tiler_mn)
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mW, mdO, mMean, mRstd, mdX, mdW, mdB)
        )
        kernel.launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, do_dtype=mdO.element_type
            ),
            stream=stream,
        )

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        M = shape[0]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem,
            tv_layout,
            is_persistent=True,
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        num_copy_elems_X = (
            tv_layout.shape[1]
            if cutlass.const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if cutlass.const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        copy_atom_load_X = get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=False
        )
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems_X))
        thr_copy_X = cute.make_tiled_copy_tv(
            copy_atom_load_X, thr_layout, val_layout
        ).get_slice(tidx)
        copy_fn = partial(_quack_copy, num_copy_elems=num_copy_elems_X)

        gX, gdO, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y))
            for mT in (mX, mdO, mdX, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdO, tXgdX)
        ]

        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
            if not is_even_N
            else None
        )

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if not is_even_N:
                tXrW.fill(0.0)
            copy_fn(tXgW, tXrW, pred=tXpX)

        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            tXgX_cur = coord_offset_i64(bidx_start, tXgX, dim=3)[None, None, None, 0]
            tXgdO_cur = coord_offset_i64(bidx_start, tXgdO, dim=3)[None, None, None, 0]
            copy_fn(
                tXgX_cur,
                tXsX[None, None, None, 0],
                pred=tXpX,
                is_async=True,
            )
            copy_fn(
                tXgdO_cur,
                tXsdO[None, None, None, 0],
                pred=tXpX,
                is_async=True,
            )
        elif tiler_mn[0] > 1:
            fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
            fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                tXgX_cur = coord_offset_i64(bidx + gdim, tXgX, dim=3)[
                    None, None, None, 0
                ]
                tXgdO_cur = coord_offset_i64(bidx + gdim, tXgdO, dim=3)[
                    None, None, None, 0
                ]
                copy_fn(
                    tXgX_cur,
                    tXsX[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
                copy_fn(
                    tXgdO_cur,
                    tXsdO[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
            elif tiler_mn[0] > 1:
                fill_oob(
                    tXsX[None, None, None, stage ^ 1],
                    None,
                    fill_value=mX.element_type.zero,
                )
                fill_oob(
                    tXsdO[None, None, None, stage ^ 1],
                    None,
                    fill_value=mdO.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            rstd_val = cutlass.Float.zero
            mean_val = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd_val = mRstd[row]
                mean_val = mMean[row]
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = (x - mean_val) * rstd_val
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                row_reduce_add(
                    x_hat * wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )
            mean_wdy = (
                row_reduce_add(
                    wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage ^ 1],
                    None,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage,
                        peer_cta_rank_in_cluster=lane_idx,
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - mean_wdy - x_hat * mean_xhat_wdy) * rstd_val
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                tXgdX_cur = coord_offset_i64(bidx, tXgdX, dim=3)[None, None, None, 0]
                copy_fn(tXrdX, tXgdX_cur, pred=tXpX)
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row0 = tXcX[None, None, None, 0][0][0]
                if row0 > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row0 == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0],
                            tXsdW.layout,
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    if const_expr(self.atomic_dw):
                        atomic_add_tensor_f32(tXrdW, tXgdW, pred=tXpX)
                    else:
                        copy_fn(tXrdW, tXgdW, pred=tXpX)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row0 = tXcX[None, None, None, 0][0][0]
                if row0 > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row0 == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0],
                            tXsdB.layout,
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy_fn(tXrdB, tXgdB, pred=tXpX)
        else:
            if const_expr(mdW is not None):
                if const_expr(self.atomic_dw):
                    atomic_add_tensor_f32(tXrdW, tXgdW, pred=tXpX)
                else:
                    copy_fn(tXrdW, tXgdW, pred=tXpX)
            if const_expr(mdB is not None):
                copy_fn(tXrdB, tXgdB, pred=tXpX)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mdO: cute.Tensor,
            mMean: cute.Tensor,
            mRstd: cute.Tensor,
            mdX: cute.Tensor,
            mdW: cute.Tensor | None,
            mdB: cute.Tensor | None,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            self._kernel_impl(
                mX,
                mW,
                mdO,
                mMean,
                mRstd,
                mdX,
                mdW,
                mdB,
                tv_layout,
                tiler_mn,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mdO: cute.Tensor,
            mMean: cute.Tensor,
            mRstd: cute.Tensor,
            mdX: cute.Tensor,
            mdW: cute.Tensor | None,
            mdB: cute.Tensor | None,
        ):
            largest_dtype_width = const_expr(
                max(
                    mX.element_type.width,
                    mW.element_type.width if mW is not None else 0,
                    mdO.element_type.width,
                    mdX.element_type.width,
                )
            )
            tiler_mn, tv_layout = self._get_tv_layout(
                num_copy_bits=128 // largest_dtype_width * mX.element_type.width
            )
            self._kernel_impl(
                mX,
                mW,
                mdO,
                mMean,
                mRstd,
                mdX,
                mdW,
                mdB,
                tv_layout,
                tiler_mn,
            )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer,
        ptr_dout: cute.Pointer,
        ptr_rstd: cute.Pointer,
        ptr_mean: cute.Pointer,
        ptr_dx: cute.Pointer,
        ptr_dw_acc: cute.Pointer,
        ptr_db_acc: Optional[cute.Pointer],
        M: Int32,
        ld: Int32,
        sm_count: Int32,
        stream: cuda.CUstream,
    ) -> None:
        ld_assumed = cute.assume(ld, divby=256 // self.dtype.width)
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        layout_n = cute.make_layout((self.N,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))
        layout_dw = cute.make_layout(
            (sm_count, self.N), stride=((0 if self.atomic_dw else self.N), 1)
        )
        layout_db = cute.make_layout((sm_count, self.N), stride=(self.N, 1))

        mX = cute.make_tensor(ptr_x, layout_mn)
        mW = cute.make_tensor(ptr_w, layout_n)
        mdO = cute.make_tensor(ptr_dout, layout_mn)
        mRstd = cute.make_tensor(ptr_rstd, layout_m)
        mMean = cute.make_tensor(ptr_mean, layout_m)
        mdX = cute.make_tensor(ptr_dx, layout_mn)
        mdW = cute.make_tensor(ptr_dw_acc, layout_dw)
        mdB = (
            cute.make_tensor(ptr_db_acc, layout_db)
            if const_expr(ptr_db_acc is not None)
            else None
        )

        self.__call__(
            mX,
            mW,
            mdO,
            mMean,
            mRstd,
            mdX,
            mdW,
            mdB,
            sm_count,
            stream,
        )


class _PtrLayernormBwdFastLaunch:
    def __init__(
        self,
        *,
        compiled: object,
        executor: object,
        capi_func: object,
        ptr_x: object,
        ptr_w: object,
        ptr_dout: object,
        ptr_rstd: object,
        ptr_mean: object,
        ptr_dx: object,
        ptr_dw_partial: object,
        ptr_db_partial: Optional[object],
        arg_m: StableI32Arg,
        arg_ld: StableI32Arg,
        arg_sm_count: StableI32Arg,
        stream: cuda.CUstream,
        assumed_align_x: int,
        assumed_align_w: int,
        assumed_align_dw: int,
        weight_dtype: type[cutlass.Numeric],
        packed_args: object,
        keepalive: tuple[object, ...],
    ):
        self._compiled = compiled
        self._executor = executor
        self._capi_func = capi_func
        self._ptr_x = ptr_x
        self._ptr_w = ptr_w
        self._ptr_dout = ptr_dout
        self._ptr_rstd = ptr_rstd
        self._ptr_mean = ptr_mean
        self._ptr_dx = ptr_dx
        self._ptr_dw_partial = ptr_dw_partial
        self._ptr_db_partial = ptr_db_partial
        self._arg_m = arg_m
        self._arg_ld = arg_ld
        self._arg_sm_count = arg_sm_count
        self._stream = stream
        self._assumed_align_x = int(assumed_align_x)
        self._assumed_align_w = int(assumed_align_w)
        self._assumed_align_dw = int(assumed_align_dw)
        self._weight_dtype = weight_dtype
        self._packed_args = packed_args
        self._keepalive = keepalive

        self._use_fast_launch = True
        self._cuda_result = getattr(executor, "cuda_result", None)

        self._last_x_ptr = -1
        self._last_w_ptr = -1
        self._last_dout_ptr = -1
        self._last_rstd_ptr = -1
        self._last_mean_ptr = -1
        self._last_dx_ptr = -1
        self._last_dw_ptr = -1
        self._last_db_ptr = -1
        self._last_m = -1
        self._last_ld = -1
        self._last_sm_count = -1

    def launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        dout: Tensor,
        rstd: Tensor,
        mean: Tensor,
        dx: Tensor,
        dw_partial: Tensor,
        db_partial: Optional[Tensor],
        M: int,
        ld: int,
        sm_count: int,
    ) -> None:
        if not fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return

        def _update_ptr(last_name: str, ptr_obj: object, value: int) -> bool:
            last_attr = (
                last_name if last_name.startswith("_last_") else f"_last_{last_name}"
            )
            if value == getattr(self, last_attr):
                return True
            try:
                set_runtime_ptr(ptr_obj, value)
                setattr(self, last_attr, value)
                return True
            except AttributeError:
                self._disable_fast_launch()
                return False

        if not _update_ptr("x_ptr", self._ptr_x, x.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("w_ptr", self._ptr_w, weight.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("dout_ptr", self._ptr_dout, dout.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("rstd_ptr", self._ptr_rstd, rstd.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("mean_ptr", self._ptr_mean, mean.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("dx_ptr", self._ptr_dx, dx.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return
        if not _update_ptr("dw_ptr", self._ptr_dw_partial, dw_partial.data_ptr()):
            self._fallback_launch(
                x=x,
                weight=weight,
                dout=dout,
                rstd=rstd,
                mean=mean,
                dx=dx,
                dw_partial=dw_partial,
                db_partial=db_partial,
                M=M,
                ld=ld,
                sm_count=sm_count,
            )
            return

        if self._ptr_db_partial is not None and db_partial is not None:
            if not _update_ptr("db_ptr", self._ptr_db_partial, db_partial.data_ptr()):
                self._fallback_launch(
                    x=x,
                    weight=weight,
                    dout=dout,
                    rstd=rstd,
                    mean=mean,
                    dx=dx,
                    dw_partial=dw_partial,
                    db_partial=db_partial,
                    M=M,
                    ld=ld,
                    sm_count=sm_count,
                )
                return

        if M != self._last_m:
            self._arg_m.set(M)
            self._last_m = M
        if ld != self._last_ld:
            self._arg_ld.set(ld)
            self._last_ld = ld
        if sm_count != self._last_sm_count:
            self._arg_sm_count.set(sm_count)
            self._last_sm_count = sm_count

        if self._cuda_result is not None:
            self._cuda_result.value = 0
        ret = self._capi_func(self._packed_args)  # type: ignore[misc]
        if ret != 0:
            raise RuntimeError(f"CuTeDSL capi_func returned non-zero: {ret}")
        if self._cuda_result is not None:
            err = int(self._cuda_result.value)
            if err != 0:
                raise RuntimeError(f"CuTeDSL kernel launch failed (cuda_result={err})")

    def _disable_fast_launch(self) -> None:
        self._use_fast_launch = False
        disable_fast_launch()

    def _fallback_launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        dout: Tensor,
        rstd: Tensor,
        mean: Tensor,
        dx: Tensor,
        dw_partial: Tensor,
        db_partial: Optional[Tensor],
        M: int,
        ld: int,
        sm_count: int,
    ) -> None:
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x = rt.make_ptr(
            dtype,
            x.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_x,
        )
        ptr_w = rt.make_ptr(
            self._weight_dtype,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_w,
        )
        ptr_dout = rt.make_ptr(
            dtype,
            dout.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_x,
        )
        ptr_rstd = rt.make_ptr(
            TORCH2CUTE_DTYPE[rstd.dtype],
            rstd.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_mean = rt.make_ptr(
            TORCH2CUTE_DTYPE[mean.dtype],
            mean.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_dx = rt.make_ptr(
            dtype,
            dx.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_x,
        )
        ptr_dw_partial = rt.make_ptr(
            TORCH2CUTE_DTYPE[dw_partial.dtype],
            dw_partial.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align_dw,
        )
        ptr_db_partial = (
            rt.make_ptr(
                TORCH2CUTE_DTYPE[db_partial.dtype],
                db_partial.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=self._assumed_align_dw,
            )
            if db_partial is not None
            else None
        )
        self._compiled(
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_mean,
            ptr_dx,
            ptr_dw_partial,
            ptr_db_partial,
            Int32(M),
            Int32(ld),
            Int32(sm_count),
            self._stream,
        )


def _get_fast_ptr_layernorm_bwd_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    weight_dtype: type[cutlass.Numeric],
    N: int,
    device_index: int,
    stream_handle: int,
    has_db_partial: bool,
    assumed_align_x: int,
    assumed_align_w: int,
    assumed_align_dw: int,
) -> Optional[_PtrLayernormBwdFastLaunch]:
    if not fast_launch_enabled():
        return None

    key = (
        "layernorm_bwd_ptr_fast",
        id(compiled),
        int(N),
        dtype,
        weight_dtype,
        bool(has_db_partial),
        int(device_index),
        int(stream_handle),
        int(assumed_align_x),
        int(assumed_align_w),
        int(assumed_align_dw),
    )
    cache = _tls_fast_launch_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    ptr_x = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=int(assumed_align_x)
    )
    ptr_w = rt.make_ptr(
        weight_dtype,
        0,
        mem_space=rt.AddressSpace.gmem,
        assumed_align=int(assumed_align_w),
    )
    ptr_dout = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=int(assumed_align_x)
    )
    ptr_rstd = rt.make_ptr(
        cutlass.Float32,
        0,
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_mean = rt.make_ptr(
        cutlass.Float32,
        0,
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_dx = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=int(assumed_align_x)
    )
    ptr_dw_partial = rt.make_ptr(
        cutlass.Float32,
        0,
        mem_space=rt.AddressSpace.gmem,
        assumed_align=int(assumed_align_dw),
    )
    ptr_db_partial = (
        rt.make_ptr(
            cutlass.Float32,
            0,
            mem_space=rt.AddressSpace.gmem,
            assumed_align=int(assumed_align_dw),
        )
        if has_db_partial
        else None
    )

    arg_m = StableI32Arg(0)
    arg_ld = StableI32Arg(N)
    arg_sm_count = StableI32Arg(0)
    stream = cuda.CUstream(int(stream_handle))
    executor = compiled.to(device_index)  # type: ignore[attr-defined]

    try:
        exe_args, adapted_args = executor.generate_execution_args(
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_mean,
            ptr_dx,
            ptr_dw_partial,
            ptr_db_partial,
            arg_m,
            arg_ld,
            arg_sm_count,
            stream,
        )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        disable_fast_launch()
        return None

    keepalive: tuple[object, ...] = (
        executor,
        ptr_x,
        ptr_w,
        ptr_dout,
        ptr_rstd,
        ptr_mean,
        ptr_dx,
        ptr_dw_partial,
        ptr_db_partial,
        arg_m,
        arg_ld,
        arg_sm_count,
        stream,
        *adapted_args,
    )
    launcher = _PtrLayernormBwdFastLaunch(
        compiled=compiled,
        executor=executor,
        capi_func=capi_func,
        ptr_x=ptr_x,
        ptr_w=ptr_w,
        ptr_dout=ptr_dout,
        ptr_rstd=ptr_rstd,
        ptr_mean=ptr_mean,
        ptr_dx=ptr_dx,
        ptr_dw_partial=ptr_dw_partial,
        ptr_db_partial=ptr_db_partial,
        arg_m=arg_m,
        arg_ld=arg_ld,
        arg_sm_count=arg_sm_count,
        stream=stream,
        assumed_align_x=int(assumed_align_x),
        assumed_align_w=int(assumed_align_w),
        assumed_align_dw=int(assumed_align_dw),
        weight_dtype=weight_dtype,
        packed_args=packed_args,
        keepalive=keepalive,
    )
    cache[key] = launcher
    return launcher


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

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        N: int,
        *,
        copy_bits_x: Optional[int] = None,
        direct_gmem: bool = False,
    ):
        super().__init__(dtype, N, stage=2)  # 2 stages for mean and var
        # Default reload policy mirrors Quack: use SMEM reload only for
        # very large hidden sizes. We keep this conservative for LayerNorm
        # and tune primarily via threads-per-block / cluster_n.
        self.reload_from: Optional[str] = None if N <= 16384 else "smem"
        # SM100 tuning: for DSv3 hidden sizes where we fuse mean+var stats,
        # delay loading fp32 weights/bias until after the reductions to lower
        # register pressure.
        self.delay_w_load: bool = bool(N in (4096, 6144, 7168, 8192))
        self.copy_bits_x: Optional[int] = (
            int(copy_bits_x) if copy_bits_x is not None else None
        )
        self.direct_gmem: bool = bool(direct_gmem)

    def _get_num_threads(self) -> int:
        nt = getattr(self, "_nt_override", None)
        if nt is not None:
            return int(nt)
        return super()._get_num_threads()

    def _calculate_threads_per_row(self) -> int:
        tpr = getattr(self, "_tpr_override", None)
        if tpr is not None:
            return int(tpr)
        # Match Quack's LayerNorm threads-per-row buckets.
        N = self.N
        if N in (4096, 6144):
            return 128
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (
                    32
                    if N <= 3072
                    else (64 if N <= 6144 else (128 if N <= 16384 else 256))
                )
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
        largest_dtype_width = const_expr(
            max(
                t.element_type.width
                for t in (mX, mW, mB, mO, mRstd, mMean)
                if t is not None
            )
        )
        # Match Quack's unified RMSNorm/LayerNorm kernel: pick vecsize based on
        # the widest dtype participating in the op (e.g. fp32 weights => fp16
        # X uses 64b vectorization).
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        default_copy_bits_x = vecsize * self.dtype.width
        num_copy_bits_x = (
            int(self.copy_bits_x)
            if self.copy_bits_x is not None
            else default_copy_bits_x
        )
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits_x)
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
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
        # Mirror Quack-style divisibility contracts so the compiler can prove
        # alignment for vectorized loads/stores (and cp.async when enabled).
        divby = (
            int(self.copy_bits_x) // self.dtype.width
            if const_expr(self.copy_bits_x is not None)
            else (128 // self.dtype.width)
        )
        ld_assumed = cute.assume(ld, divby=divby)
        # Match `mark_compact_shape_dynamic(mode=0, ...)`: M is dynamic, N is static.
        layout_mn = cute.make_layout((M, self.N), stride=(ld_assumed, 1))
        layout_n = cute.make_layout((self.N,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)
        mW = cute.make_tensor(ptr_w, layout_n)
        mB = (
            cute.make_tensor(ptr_b, layout_n) if const_expr(ptr_b is not None) else None
        )
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
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Quack-style CTA tiling: let CuTe compute the CTA offsets directly.
        # (Avoids the extra 64-bit address arithmetic in `domain_offset_i64` on
        # the common inference/benchmark sizes.)
        gX, gO = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO)]
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

        # Copy atoms for X / W / B / O (mirror Quack's vector-size contract).
        num_copy_elems_x = (
            tv_layout.shape[1]
            if const_expr(cute.rank(tv_layout.shape[1]) == 1)
            else tv_layout.shape[1][0]
        )
        threads_per_row = (
            tv_layout.shape[0]
            if const_expr(cute.rank(tv_layout.shape[0]) == 1)
            else tv_layout.shape[0][0]
        )
        num_copy_bits_x = mX.element_type.width * num_copy_elems_x
        num_copy_bits_x_async = const_expr(min(128, num_copy_bits_x))
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=num_copy_bits_x,
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=num_copy_bits_x_async,
        )
        num_copy_bits_wb = const_expr(
            min(128, mW.element_type.width * num_copy_elems_x)
        )
        copy_atom_load_WB = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=num_copy_bits_wb,
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=num_copy_bits_x,
        )

        # Quack-style partitioning: use `make_tiled_copy_tv` (2D thread/value
        # layout) and let partitioning over the CTA tile handle the N loop.
        thr_layout = cute.make_ordered_layout(
            (tiler_mn[0], threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, num_copy_elems_x))
        thr_copy = cute.make_tiled_copy_tv(
            copy_atom_load_X, thr_layout, val_layout
        ).get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXgO = thr_copy.partition_D(gO)
        tXgW = thr_copy.partition_S(gW)
        tXgB = thr_copy.partition_S(gB) if const_expr(gB is not None) else None
        tXrRstd = thr_copy.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]

        # Fragments for gmem->rmem.
        tXrW = cute.make_fragment_like(tXgW)
        tXrB = cute.make_fragment_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=False)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            None if is_even_N else predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        row = tXcX[0][0]
        if const_expr(not self.direct_gmem):
            if row < shape[0]:
                cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

        if const_expr(not delay_w_load):
            cute.copy(copy_atom_load_WB, tXgW, tXrW, pred=tXpX)
            if const_expr(mB is not None):
                cute.copy(copy_atom_load_WB, tXgB, tXrB, pred=tXpX)

        if const_expr(not self.direct_gmem):
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        else:
            if row < shape[0]:
                cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        x = tXrX.load().to(Float32)
        if const_expr(self.cluster_n == 1 and self.N in (4096, 6144, 7168, 8192)):
            # SM100 tuning for DSv3 hidden sizes:
            # Compute (sum_x, sum_x2) together so we can derive mean + variance
            # without a second reduction pass (and without re-materializing
            # x-mean for the variance reduction).
            sum_x = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
            sum_x2 = (x * x).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
            )
            sum_x = warp_reduce(
                sum_x,
                operator.add,
                width=min(threads_per_row, cute.arch.WARP_SIZE),
            )
            sum_x2 = warp_reduce(
                sum_x2,
                operator.add,
                width=min(threads_per_row, cute.arch.WARP_SIZE),
            )
            warps_per_row, cluster_n = reduction_buffer.shape[1]
            if const_expr(warps_per_row > 1 or cluster_n > 1):
                lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
                row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx, 0] = sum_x
                    reduction_buffer[row_idx, col_idx, 1] = sum_x2
                cute.arch.barrier()
                block_sum_x = 0.0
                block_sum_x2 = 0.0
                if lane_idx < warps_per_row:
                    block_sum_x = reduction_buffer[row_idx, lane_idx, 0]
                    block_sum_x2 = reduction_buffer[row_idx, lane_idx, 1]
                sum_x = warp_reduce(block_sum_x, operator.add)
                sum_x2 = warp_reduce(block_sum_x2, operator.add)
            mean = sum_x / shape[1]
            var = sum_x2 / shape[1] - mean * mean
            var = cute.arch.fmax(var, 0.0)
            rstd = cute.math.rsqrt(var + eps, fastmath=True)
        else:
            sum_x = row_reduce(
                x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                hook_fn=(
                    cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None
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
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd

        if const_expr(mMean is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrMean[0] = mean

        if const_expr(delay_w_load):
            cute.copy(copy_atom_load_WB, tXgW, tXrW, pred=tXpX)
            if const_expr(mB is not None):
                cute.copy(copy_atom_load_WB, tXgB, tXrB, pred=tXpX)

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
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tXpX)

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
            largest_dtype_width = const_expr(
                max(
                    mX.element_type.width,
                    mW.element_type.width,
                    mB.element_type.width if const_expr(mB is not None) else 0,
                    mO.element_type.width,
                    mRstd.element_type.width if const_expr(mRstd is not None) else 0,
                    mMean.element_type.width if const_expr(mMean is not None) else 0,
                )
            )
            vecsize = math.gcd(self.N, 128 // largest_dtype_width)
            default_copy_bits_x = vecsize * mX.element_type.width
            num_copy_bits_x = (
                int(self.copy_bits_x)
                if const_expr(self.copy_bits_x is not None)
                else default_copy_bits_x
            )
            tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits_x)
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
    dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
    dtype_b = TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    key = (N, dtype, dtype_w, dtype_b, mRstd is not None, mMean is not None)
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

    This path supports both Quack-style fp32 weights/bias and same-dtype
    weights/bias for bf16/fp16 activations, as long as the layout stays in the
    common row-major form.
    """
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.stride(1) != 1:
        return False
    if not weight.is_cuda or weight.dim() != 1:
        return False
    if weight.dtype != x.dtype:
        if weight.dtype != torch.float32:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
    if not weight.is_contiguous():
        return False
    if bias is not None:
        if not bias.is_cuda or bias.dim() != 1:
            return False
        if bias.dtype != weight.dtype:
            return False
        if not bias.is_contiguous():
            return False
    # Require 16B alignment for vectorized loads/stores.
    if (x.data_ptr() % 16) != 0:
        return False
    if (weight.data_ptr() % 16) != 0:
        return False
    if bias is not None and (bias.data_ptr() % 16) != 0:
        return False
    # The kernel uses vectorized loads; require the leading dimension to
    # preserve 16B alignment for every row start.
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
    dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
    dtype_b = TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    # Keep the pointer path aligned with Quack's LayerNorm schedule:
    # - <=128b vectorization (cp.async-compatible)
    # - shared-memory staging for X (gmem->smem->rmem) to amortize global latency
    direct_gmem = False
    copy_bits_x: Optional[int] = None
    assumed_align_xo = 16

    # DSv3 hidden sizes are often latency-bound on small M. For these N buckets,
    # a direct-GMEM schedule (skip gmem->smem cp.async) can reduce overhead.
    #
    # Keep the Quack-like staged path for large M where cp.async overlap tends to win.
    if dtype_x.width == 16:
        # DSv3 default hidden size (7168) is a common inference hot shape and
        # benefits from the lower-overhead direct-GMEM path on this SM100.
        if N == 7168 and M <= 65536:
            direct_gmem = True
        elif N == 8192 and M <= 16384:
            direct_gmem = True

    # DSv3 smallest point (M=4096, N=7168) is latency-sensitive. Increasing
    # per-row parallelism improves the reduction path and consistently beats
    # Quack on this machine.
    tpr_override: Optional[int] = None
    nt_override: Optional[int] = None
    if dtype_x.width == 16 and N == 7168 and M <= 4096:
        tpr_override = 224
        nt_override = 224

    # NOTE: We previously experimented with a direct-GMEM + 256b vectorized
    # schedule for N=4096, but it was consistently slower on this GB200.
    # Keep the pointer path on the Quack-like staged (cp.async) schedule.
    key = (
        "ptr",
        int(N),
        dtype_x,
        dtype_w,
        dtype_b,
        bias is not None,
        rstd is not None,
        mean is not None,
        bool(direct_gmem),
        int(copy_bits_x) if copy_bits_x is not None else None,
        tpr_override,
        nt_override,
        int(assumed_align_xo),
        int(device_index),
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = LayerNormSM100(
            dtype_x,
            int(N),
            copy_bits_x=copy_bits_x,
            direct_gmem=direct_gmem,
        )
        if tpr_override is not None:
            op._tpr_override = tpr_override  # type: ignore[attr-defined]
        if nt_override is not None:
            op._nt_override = nt_override  # type: ignore[attr-defined]
        ptr_x = rt.make_ptr(
            dtype_x,
            x.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_xo,
        )
        ptr_out = rt.make_ptr(
            dtype_x,
            out.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_xo,
        )
        ptr_w = rt.make_ptr(
            dtype_w,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=16,
        )
        ptr_b = (
            rt.make_ptr(
                dtype_b,
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

    launcher = _get_fast_ptr_layernorm_launcher(
        compiled=compiled,
        N=int(N),
        dtype_x=dtype_x,
        dtype_w=dtype_w,
        dtype_b=dtype_b,
        has_bias=bias is not None,
        has_rstd=rstd is not None,
        has_mean=mean is not None,
        device_index=int(device_index),
        stream_handle=stream_handle,
        assumed_align_xo=int(assumed_align_xo),
        eps=float(eps),
    )
    ld_val = int(x.stride(0))
    if launcher is not None:
        launcher.launch(
            x=x,
            weight=weight,
            bias=bias,
            out=out,
            rstd=rstd,
            mean=mean,
            M=int(M),
            ld=ld_val,
            eps=float(eps),
        )
        return

    ptr_x = rt.make_ptr(
        dtype_x,
        x.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_xo,
    )
    ptr_out = rt.make_ptr(
        dtype_x,
        out.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_xo,
    )
    ptr_w = rt.make_ptr(
        dtype_w,
        weight.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=16,
    )
    ptr_b = (
        rt.make_ptr(
            dtype_b,
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
    ld = Int32(ld_val)
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
        warp_sums_wdy = smem.allocate_tensor(
            Float32, warp_sums_layout, byte_alignment=4
        )
        warp_sums_xhatwdy = smem.allocate_tensor(
            Float32, warp_sums_layout, byte_alignment=4
        )

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
    mRstd = from_dlpack(rstd_1d.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    mMean = from_dlpack(mean_1d.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
    dtype_dout = TORCH2CUTE_DTYPE[dout_2d.dtype]
    key = (N, dtype, dtype_w, dtype_dout)
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
    mRstd = from_dlpack(rstd_1d.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    mMean = from_dlpack(mean_1d.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

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
    dtype_dout = TORCH2CUTE_DTYPE[dout_2d.dtype]
    key = (N, dtype, dtype_dout, has_bias)
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


def _can_use_ptr_path_bwd(
    x: Tensor,
    weight: Tensor,
    dout: Tensor,
    rstd: Tensor,
    mean: Tensor,
) -> bool:
    """Return whether tensors satisfy the optimized pointer-path contract.

    The pointer kernel assumes row-major 2D activations/gradients, contiguous
    fp32 per-row statistics, 16B activation alignment, and enough hidden-dim
    divisibility for vectorized/cp.async copies.  Keep this predicate strict:
    callers that fail it are still handled by the local fallback kernels.
    """
    if not x.is_cuda or not dout.is_cuda or x.dim() != 2 or dout.shape != x.shape:
        return False
    if dout.dtype != x.dtype:
        return False
    if x.stride(1) != 1 or dout.stride(1) != 1:
        return False
    if dout.stride(0) != x.stride(0):
        return False
    if not weight.is_cuda or weight.dim() != 1 or weight.shape[0] != x.shape[1]:
        return False
    if weight.dtype != x.dtype:
        if weight.dtype != torch.float32:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
    if not weight.is_contiguous():
        return False
    if (x.data_ptr() % 16) != 0 or (dout.data_ptr() % 16) != 0:
        return False
    assumed_align_w = 32 if weight.dtype == torch.float32 else 16
    if (weight.data_ptr() % assumed_align_w) != 0:
        return False
    if (rstd.data_ptr() % 4) != 0 or (mean.data_ptr() % 4) != 0:
        return False
    if (not rstd.is_cuda) or (not mean.is_cuda):
        return False
    if rstd.dtype != torch.float32 or mean.dtype != torch.float32:
        return False
    if (not rstd.is_contiguous()) or (not mean.is_contiguous()):
        return False
    if rstd.numel() != x.shape[0] or mean.numel() != x.shape[0]:
        return False
    dtype_x = TORCH2CUTE_DTYPE[x.dtype]
    divby = 256 // dtype_x.width
    if (x.stride(0) % divby) != 0:
        return False
    # The CuTe tiled-copy path is vectorized along N; requiring a multiple of 8
    # for fp16/bf16 covers the DSv3/DSv4 hidden sizes while avoiding a slower
    # predicated tail specialization in the hot pointer path.
    if (x.shape[1] % 8) != 0:
        return False
    return True


def _get_layernorm_bwd_sm_count(
    N: int,
    device: torch.device,
    *,
    M: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    sm_count_multiple = (
        16
        if N <= 256
        else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    sm_count = (
        sm_count * sm_count_multiple
        if N <= 8192
        else sm_count // 2
        if N <= 16384
        else sm_count * 2
    )
    # The self-contained SM103 LayerNorm backward kernel is register/smem limited
    # to about two resident CTAs per SM for N=8192.  Large-M shapes need both
    # slots populated, while small-M shapes are more launch-tail sensitive.
    if (
        props.major == 10
        and props.minor == 3
        and props.multi_processor_count == 152
        and dtype in (torch.float16, torch.bfloat16)
        and N == 8192
    ):
        sm_count = (
            props.multi_processor_count
            if M is not None and M <= 4096
            else props.multi_processor_count * 2
        )
    return int(sm_count)


def _get_layernorm_bwd_tuning(
    N: int,
    dtype_x: type[cutlass.Numeric],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    tpr_override: Optional[int] = None
    nt_override: Optional[int] = None
    cluster_n_override: Optional[int] = None

    if N == 4096 and dtype_x.width == 16:
        # On GB300, widening the persistent grid is already enough for 4k hidden
        # sizes; a 128-thread CTA / 128 threads-per-row schedule keeps the fused
        # kernel exact for the larger DSv3 LayerNorm backward cases.
        tpr_override = 128
        nt_override = 128
    elif N == 8192 and dtype_x.width == 16:
        # One row per CTA, with a full 256-thread row reduction to reduce per-thread
        # register work on the wide 8k DSv3 rows.
        tpr_override = 256
        nt_override = 256
        cluster_n_override = 1

    return tpr_override, nt_override, cluster_n_override


def _should_use_layernorm_bwd_ptr(x: Tensor, weight: Tensor) -> bool:
    """Return True only for shapes where the pointer bwd path is a stable win."""
    N = int(x.shape[-1])
    M = int(x.numel() // N)
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False

    props = torch.cuda.get_device_properties(x.device)
    is_gb300 = (
        props.major == 10 and props.minor == 3 and props.multi_processor_count == 152
    )

    # DSv3/DSv4 LayerNorm backward target on GB300. Route the wide bf16 rows
    # through Oink's self-contained fused pointer path only when the stricter
    # layout/alignment checks pass; all other shapes use local fallback kernels.
    if is_gb300 and N in (6144, 7168, 8192):
        return weight.dtype == x.dtype
    if N == 4096 and M >= 8192 and is_gb300:
        return weight.dtype in (x.dtype, torch.float32)
    return False


def _should_use_layernorm_bwd_atomic_dw_ptr(x: Tensor, weight: Tensor) -> bool:
    """Use atomic dW only where it wins over partial-buffer reduction on GB300."""
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        return False
    props = torch.cuda.get_device_properties(x.device)
    is_gb300 = (
        props.major == 10 and props.minor == 3 and props.multi_processor_count == 152
    )
    if not is_gb300:
        return False
    N = int(x.shape[-1])
    M = int(x.numel() // N)
    if N == 8192:
        return M >= 16384
    if N == 6144:
        return M >= 65536
    return False


def _layernorm_backward_atomic_ptr(
    *,
    dout_2d: Tensor,
    x_2d: Tensor,
    weight: Tensor,
    rstd_1d: Tensor,
    mean_1d: Tensor,
    dx_2d: Tensor,
    dw_acc: Tensor,
    db_acc: Optional[Tensor],
    sm_count: int,
) -> None:
    """Run the pointer kernel with direct atomic dW accumulation.

    dB remains a regular (sm_count, N) partial buffer when requested; only dW
    uses the direct atomic accumulation variant.
    """
    assert _LayerNormBackwardSM103 is not None
    assert _can_use_ptr_path_bwd(x_2d, weight, dout_2d, rstd_1d, mean_1d)

    M, N = x_2d.shape
    device_index = x_2d.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)

    dtype_x = TORCH2CUTE_DTYPE[x_2d.dtype]
    dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
    assumed_align_x = 16
    assumed_align_w = 32 if weight.dtype == torch.float32 else 16
    assumed_align_dw = 32

    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)
    ld_val = int(x_2d.stride(0))

    tpr_override, nt_override, cluster_n_override = _get_layernorm_bwd_tuning(
        N, dtype_x
    )

    use_atomic_dw = dw_acc.dim() == 1
    assert db_acc is None or db_acc.dim() == 2

    key = (
        "layernorm_bwd_atomic_ptr",
        int(N),
        dtype_x,
        dtype_w,
        bool(db_acc is not None),
        bool(use_atomic_dw),
        int(assumed_align_x),
        int(assumed_align_w),
        int(assumed_align_dw),
        tpr_override,
        nt_override,
        cluster_n_override,
        int(device_index),
    )
    compiled = _BWD_PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = _LayerNormBackwardSM103(dtype_x, N)
        op.atomic_dw = bool(use_atomic_dw)
        if tpr_override is not None:
            op._tpr_override = tpr_override  # type: ignore[attr-defined]
        if nt_override is not None:
            op._nt_override = nt_override  # type: ignore[attr-defined]
        if cluster_n_override is not None:
            op._cluster_n_override = cluster_n_override  # type: ignore[attr-defined]

        ptr_x = rt.make_ptr(
            dtype_x,
            x_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_w = rt.make_ptr(
            dtype_w,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_w,
        )
        ptr_dout = rt.make_ptr(
            dtype_x,
            dout_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_rstd = rt.make_ptr(
            cutlass.Float32,
            rstd_1d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_mean = rt.make_ptr(
            cutlass.Float32,
            mean_1d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_dx = rt.make_ptr(
            dtype_x,
            dx_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_dw = rt.make_ptr(
            cutlass.Float32,
            dw_acc.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_dw,
        )
        ptr_db = (
            rt.make_ptr(
                cutlass.Float32,
                db_acc.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=assumed_align_dw,
            )
            if db_acc is not None
            else None
        )
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_mean,
            ptr_dx,
            ptr_dw,
            ptr_db,
            Int32(M),
            Int32(ld_val),
            Int32(int(sm_count)),
            stream,
        )
        _BWD_PTR_COMPILE_CACHE[key] = compiled

    ptr_x = rt.make_ptr(
        dtype_x,
        x_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_w = rt.make_ptr(
        dtype_w,
        weight.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_w,
    )
    ptr_dout = rt.make_ptr(
        dtype_x,
        dout_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_rstd = rt.make_ptr(
        cutlass.Float32,
        rstd_1d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_mean = rt.make_ptr(
        cutlass.Float32,
        mean_1d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_dx = rt.make_ptr(
        dtype_x,
        dx_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_dw = rt.make_ptr(
        cutlass.Float32,
        dw_acc.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_dw,
    )
    ptr_db = (
        rt.make_ptr(
            cutlass.Float32,
            db_acc.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_dw,
        )
        if db_acc is not None
        else None
    )
    compiled(
        ptr_x,
        ptr_w,
        ptr_dout,
        ptr_rstd,
        ptr_mean,
        ptr_dx,
        ptr_dw,
        ptr_db,
        Int32(M),
        Int32(ld_val),
        Int32(int(sm_count)),
        stream,
    )


def _layernorm_backward_ptr(
    *,
    dout_2d: Tensor,
    x_2d: Tensor,
    weight: Tensor,
    rstd_1d: Tensor,
    mean_1d: Tensor,
    dx_2d: Tensor,
    dw_partial: Tensor,
    db_partial: Optional[Tensor],
    sm_count: int,
) -> None:
    assert _LayerNormBackwardSM103 is not None
    assert _can_use_ptr_path_bwd(x_2d, weight, dout_2d, rstd_1d, mean_1d)

    M, N = x_2d.shape
    device_index = x_2d.get_device()
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)

    dtype_x = TORCH2CUTE_DTYPE[x_2d.dtype]
    dtype_w = TORCH2CUTE_DTYPE[weight.dtype]
    assumed_align_x = 16
    assumed_align_w = 32 if weight.dtype == torch.float32 else 16
    assumed_align_dw = 32

    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    stream = cuda.CUstream(stream_handle)
    ld_val = int(x_2d.stride(0))

    tpr_override, nt_override, cluster_n_override = _get_layernorm_bwd_tuning(
        N, dtype_x
    )

    key = (
        "layernorm_bwd_ptr",
        int(N),
        dtype_x,
        dtype_w,
        bool(db_partial is not None),
        int(assumed_align_x),
        int(assumed_align_w),
        int(assumed_align_dw),
        tpr_override,
        nt_override,
        cluster_n_override,
        int(device_index),
    )
    compiled = _BWD_PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = _LayerNormBackwardSM103(dtype_x, N)
        if tpr_override is not None:
            op._tpr_override = tpr_override  # type: ignore[attr-defined]
        if nt_override is not None:
            op._nt_override = nt_override  # type: ignore[attr-defined]
        if cluster_n_override is not None:
            op._cluster_n_override = cluster_n_override  # type: ignore[attr-defined]

        ptr_x = rt.make_ptr(
            dtype_x,
            x_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_w = rt.make_ptr(
            dtype_w,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_w,
        )
        ptr_dout = rt.make_ptr(
            dtype_x,
            dout_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_rstd = rt.make_ptr(
            cutlass.Float32,
            rstd_1d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_mean = rt.make_ptr(
            cutlass.Float32,
            mean_1d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=4,
        )
        ptr_dx = rt.make_ptr(
            dtype_x,
            dx_2d.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_x,
        )
        ptr_dw = rt.make_ptr(
            cutlass.Float32,
            dw_partial.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_dw,
        )
        ptr_db = (
            rt.make_ptr(
                cutlass.Float32,
                db_partial.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=assumed_align_dw,
            )
            if db_partial is not None
            else None
        )
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_dout,
            ptr_rstd,
            ptr_mean,
            ptr_dx,
            ptr_dw,
            ptr_db,
            Int32(M),
            Int32(ld_val),
            Int32(int(sm_count)),
            stream,
        )
        _BWD_PTR_COMPILE_CACHE[key] = compiled

    launcher = _get_fast_ptr_layernorm_bwd_launcher(
        compiled=compiled,
        dtype=dtype_x,
        weight_dtype=dtype_w,
        N=N,
        device_index=device_index,
        stream_handle=stream_handle,
        has_db_partial=db_partial is not None,
        assumed_align_x=assumed_align_x,
        assumed_align_w=assumed_align_w,
        assumed_align_dw=assumed_align_dw,
    )
    if launcher is not None:
        launcher.launch(
            x=x_2d,
            weight=weight,
            dout=dout_2d,
            rstd=rstd_1d,
            mean=mean_1d,
            dx=dx_2d,
            dw_partial=dw_partial,
            db_partial=db_partial,
            M=M,
            ld=ld_val,
            sm_count=int(sm_count),
        )
        return

    ptr_x = rt.make_ptr(
        dtype_x,
        x_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_w = rt.make_ptr(
        dtype_w,
        weight.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_w,
    )
    ptr_dout = rt.make_ptr(
        dtype_x,
        dout_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_rstd = rt.make_ptr(
        cutlass.Float32,
        rstd_1d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_mean = rt.make_ptr(
        cutlass.Float32,
        mean_1d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=4,
    )
    ptr_dx = rt.make_ptr(
        dtype_x,
        dx_2d.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_x,
    )
    ptr_dw = rt.make_ptr(
        cutlass.Float32,
        dw_partial.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align_dw,
    )
    ptr_db = (
        rt.make_ptr(
            cutlass.Float32,
            db_partial.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align_dw,
        )
        if db_partial is not None
        else None
    )
    compiled(
        ptr_x,
        ptr_w,
        ptr_dout,
        ptr_rstd,
        ptr_mean,
        ptr_dx,
        ptr_dw,
        ptr_db,
        Int32(M),
        Int32(ld_val),
        Int32(int(sm_count)),
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
    LayerNorm backward implemented with CuTeDSL / CUTLASS-backed kernels.

    Tuned GB300 shapes use Oink's self-contained pointer fast-launch fused
    LayerNorm backward schedule. Other shapes fall back to the local
    self-contained dx and parameter-gradient kernels.
    """
    assert x.shape == dout.shape, "x and dout must have the same shape"
    assert x.is_cuda and dout.is_cuda, "x and dout must be CUDA tensors"
    assert weight.dim() == 1, "weight must be 1D"
    assert weight.shape[0] == x.shape[-1], "weight shape must match hidden dim"
    if bias is not None:
        assert bias.dim() == 1, "bias must be 1D"
        assert bias.shape == weight.shape, "bias must match weight shape"

    use_ptr_path = (
        _LayerNormBackwardSM103 is not None and _should_use_layernorm_bwd_ptr(x, weight)
    )

    x_2d, orig_shape = _as_2d(x)
    dout_2d, _ = _as_2d(dout)
    M, N = x_2d.shape
    mean_flat = mean.view(M)
    rstd_flat = rstd.view(M)

    if use_ptr_path and _can_use_ptr_path_bwd(
        x_2d, weight, dout_2d, rstd_flat, mean_flat
    ):
        device = x.device
        sm_count = _get_layernorm_bwd_sm_count(N, device, M=M, dtype=x.dtype)
        stream_handle = int(torch.cuda.current_stream(device=device).cuda_stream)
        dx_2d = torch.empty_like(x_2d)
        use_atomic_dw = _should_use_layernorm_bwd_atomic_dw_ptr(x, weight)

        if use_atomic_dw:
            dw_atomic = _get_layernorm_bwd_atomic_dw_workspace(
                device_index=x.get_device(),
                stream_handle=stream_handle,
                N=N,
            )
            dw_atomic.zero_()
            db_partial = (
                _get_layernorm_bwd_workspace(
                    device_index=x.get_device(),
                    stream_handle=stream_handle,
                    sm_count=int(sm_count),
                    N=N,
                    has_bias=True,
                )[1]
                if bias is not None
                else None
            )
            _layernorm_backward_atomic_ptr(
                dout_2d=dout_2d,
                x_2d=x_2d,
                weight=weight,
                rstd_1d=rstd_flat,
                mean_1d=mean_flat,
                dx_2d=dx_2d,
                dw_acc=dw_atomic,
                db_acc=db_partial,
                sm_count=int(sm_count),
            )
            dweight = _finalize_layernorm_bwd_atomic_dw(dw_acc=dw_atomic, weight=weight)
            if bias is not None:
                assert db_partial is not None
                dbias = _reduce_partial_sum_fp32(
                    db_partial, device_index=bias.get_device()
                )
                if bias.dtype != torch.float32:
                    dbias = dbias.to(bias.dtype)
            else:
                dbias = None
        else:
            dw_partial, db_partial = _get_layernorm_bwd_workspace(
                device_index=x.get_device(),
                stream_handle=stream_handle,
                sm_count=int(sm_count),
                N=N,
                has_bias=bias is not None,
            )
            _layernorm_backward_ptr(
                dout_2d=dout_2d,
                x_2d=x_2d,
                weight=weight,
                rstd_1d=rstd_flat,
                mean_1d=mean_flat,
                dx_2d=dx_2d,
                dw_partial=dw_partial,
                db_partial=db_partial,
                sm_count=int(sm_count),
            )
            # Keep the post-kernel reduction exact. Experimental custom reducers
            # can change dW rounding, so use the trusted fp32 partial-sum finalizer.
            dweight, dbias = _finalize_layernorm_bwd_partials(
                dw_partial=dw_partial,
                db_partial=db_partial,
                weight=weight,
                bias=bias,
                device=device,
            )
        dx = _restore_shape(dx_2d, orig_shape)
        return dx, dweight, dbias

    # Flatten to 2D for the local fallback kernels.
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
