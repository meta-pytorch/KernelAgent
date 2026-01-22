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
RMSNorm kernel for SM100 (Blackwell) in CuteDSL.

This implementation targets Blackwell with:
- A stride-preserving pointer path for padded-row layouts (e.g. MLA stride0> N).
- A one-pass fused-add RMSNorm schedule for bf16/fp16 (DSv3 N=7168) that keeps
  `x + residual` in registers (avoids re-reading gmem) and uses FP32 accumulation.
- Optional experimental schedule knobs (env vars) to explore copy widths and
  stage-2 cp.async variants.

Note: This file expects the local CuTeDSL (cutlass) and SM100 helper modules
to be available in the Python environment (e.g., `nvidia-cutlass-dsl` and
`cuda-python`). It is shipped as part of the KernelAgent Oink vLLM plugin.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import re
import subprocess
import sys
import threading
from typing import Optional, Tuple

_HERE = os.path.dirname(__file__)

# CuTeDSL caches generated MLIR into a tempdir under a global default
# (`/tmp/$USER/cutlass_python_cache`). The cache bytecode format can differ across
# `nvidia-cutlass-dsl` versions (e.g. 4.3.2 vs 4.3.4), and cross-version cache
# sharing causes noisy "invalid section ID" warnings (and disables cache reuse).
#
# If the user has not pinned `CUTE_DSL_CACHE_DIR`, isolate by version so multiple
# CuTeDSL envs can coexist on the same machine without stepping on each other.
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
        "kernelagent_oink.blackwell.rmsnorm requires CuTeDSL's Python package "
        "(`cutlass`, typically provided by `nvidia-cutlass-dsl`)."
    ) from e

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

import cuda.bindings.driver as cuda  # provided by NVIDIA cuda-python  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Float32, Int32, const_expr  # noqa: E402
from cutlass.cute import runtime as rt  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

# Simple compile cache declared early so direct execution works
_PTR_COMPILE_CACHE = {}

# Thread-local cache for the fast-launch path. We keep per-thread packed args and
# pointer/scalar storage so concurrent callers don't race on in-place updates.
_PTR_FAST_LAUNCH_TLS = threading.local()


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off", ""}


# Fast-launch uses a few private-ish CuTeDSL internals (packed args plumbing and
# runtime pointer descriptors). Keep it enabled by default for our pinned CuTeDSL
# environment, but allow disabling it via env var and auto-disable it if those
# internals are not present in a future upgrade.
_ENABLE_FAST_LAUNCH = _env_flag("OINK_CUTEDSL_FAST_LAUNCH", default=True)
_FAST_LAUNCH_SUPPORTED = True

# Fused-add RMSNorm schedule knobs (read once at import time; set env vars before
# importing this module if you want to override).
_DIRECT_GMEM_POLICY = (
    os.environ.get("OINK_RMSNORM_DIRECT_GMEM", "auto").strip().lower() or "auto"
)
_COPY_BITS_POLICY = (
    os.environ.get("OINK_RMSNORM_COPY_BITS", "auto").strip().lower() or "auto"
)
_ENABLE_CLUSTER_ILP = _env_flag("OINK_RMSNORM_ENABLE_CLUSTER_ILP", default=False)
_ENABLE_CLUSTER_ILP_UNSAFE = _env_flag(
    "OINK_RMSNORM_ENABLE_CLUSTER_ILP_UNSAFE", default=False
)
_ENABLE_TPR256 = _env_flag("OINK_RMSNORM_ENABLE_TPR256", default=False)
_ENABLE_STAGE2 = _env_flag("OINK_RMSNORM_ENABLE_STAGE2", default=False)

# Forward dispatch control:
# - Default behavior: use the pointer-based path when safe, otherwise fall back
#   to the stage-2 module (then the torch reference).
# - If you want to force stage-2 even when the pointer path is available (for
#   experimentation / A-B testing), set this env var **before** importing this
#   module.
_FORCE_RMSNORM_STAGE2_FWD = _env_flag(
    "KERNELAGENT_OINK_FORCE_RMSNORM_STAGE2", default=False
)

# CuTeDSL stability probe for the experimental cluster_n>1 + direct-GMEM schedule.
#
# Some CuTeDSL builds segfault during JIT compilation when combining:
# - cluster launches (cluster_n>1) and
# - direct-GMEM loads/stores (no staging SMEM tiles).
#
# We keep the schedule gated behind `OINK_RMSNORM_ENABLE_CLUSTER_ILP=1` +
# `OINK_RMSNORM_ENABLE_CLUSTER_ILP_UNSAFE=1`, and additionally run a one-time
# out-of-process compile probe so we can safely fall back to the staged SMEM
# path instead of crashing the parent process.
#
# This is (currently) sensitive to the vector width: we have observed
# reproducible segfaults for the 256b universal-copy path, while the 128b path
# can succeed. Cache the maximum supported copy width (0 = unsupported).
_CLUSTER_DIRECT_GMEM_MAX_COPY_BITS: Optional[int] = None
_CLUSTER_DIRECT_GMEM_PROBE_LOCK = threading.Lock()
_CLUSTER_DIRECT_GMEM_PROBE_WARNED = False


def _probe_cluster_direct_gmem_max_copy_bits() -> int:
    global _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS
    global _CLUSTER_DIRECT_GMEM_PROBE_WARNED

    override = os.environ.get("OINK_RMSNORM_CLUSTER_DIRECT_GMEM_MAX_COPY_BITS")
    if override is not None and override.strip() != "":
        try:
            value = int(override)
        except ValueError:
            value = 0
        value = 256 if value >= 256 else 128 if value >= 128 else 0
        _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS = value
        return value

    if _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS is not None:
        return _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS

    with _CLUSTER_DIRECT_GMEM_PROBE_LOCK:
        if _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS is not None:
            return _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS

        script_template = r"""
import os

os.environ["OINK_CUTEDSL_FAST_LAUNCH"] = "0"

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Float32, Int32
from cutlass.cute import runtime as rt

from kernelagent_oink.blackwell import rmsnorm

N = 7168
dtype = cutlass.BFloat16

copy_bits = int(os.environ["OINK_PROBE_COPY_BITS"])
assumed_align = int(os.environ["OINK_PROBE_ASSUMED_ALIGN"])

op = rmsnorm.RMSNormSM100(
    N,
    dtype,
    stage=1,
    copy_bits=copy_bits,
    use_async=False,
    direct_gmem=True,
)
op._cluster_n_override = 2  # 2 CTAs per row

ptr_x = rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align)
ptr_res = rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align)
ptr_w = rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align)

_ = cute.compile(
    op.launch_from_ptrs_fused_add_inplace,
    ptr_x,
    ptr_w,
    ptr_res,
    Int32(4096),
    Int32(N),
    Int32(N),
    cuda.CUstream(0),
    Float32(1e-6),
)
print(f"ok {copy_bits}")
"""

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"

        def run_probe(copy_bits: int, assumed_align: int):
            probe_env = env.copy()
            probe_env["OINK_PROBE_COPY_BITS"] = str(copy_bits)
            probe_env["OINK_PROBE_ASSUMED_ALIGN"] = str(assumed_align)
            return subprocess.run(
                [sys.executable, "-c", script_template],
                env=probe_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120.0,
            )

        proc_256 = None
        proc_128 = None
        try:
            proc_256 = run_probe(256, 32)
            if proc_256.returncode == 0:
                max_bits = 256
            else:
                proc_128 = run_probe(128, 16)
                max_bits = 128 if proc_128.returncode == 0 else 0
        except Exception:
            max_bits = 0

        if not _CLUSTER_DIRECT_GMEM_PROBE_WARNED and max_bits != 256:
            _CLUSTER_DIRECT_GMEM_PROBE_WARNED = True
            if max_bits == 128:
                print(
                    "Oink: cluster_n>1 + direct_gmem 256b compile probe failed; "
                    "using 128b copies for the cluster ILP schedule.",
                    file=sys.stderr,
                )
                if proc_256 is not None and proc_256.stderr:
                    tail = "\n".join(proc_256.stderr.splitlines()[-12:])
                    print(f"Oink: probe stderr tail:\n{tail}", file=sys.stderr)
            else:
                rc = (
                    proc_128.returncode
                    if proc_128 is not None
                    else (proc_256.returncode if proc_256 is not None else "unknown")
                )
                print(
                    "Oink: cluster_n>1 + direct_gmem compile probe failed; "
                    f"falling back to staged SMEM path (returncode={rc}).",
                    file=sys.stderr,
                )
                failing_proc = proc_128 if proc_128 is not None else proc_256
                if failing_proc is not None and failing_proc.stderr:
                    tail = "\n".join(failing_proc.stderr.splitlines()[-12:])
                    print(f"Oink: probe stderr tail:\n{tail}", file=sys.stderr)

        _CLUSTER_DIRECT_GMEM_MAX_COPY_BITS = max_bits
        return max_bits


def _parse_version_tuple(version: str) -> Tuple[int, int, int]:
    parts = version.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        match = re.match(r"^(\d+)", part)
        nums.append(int(match.group(1)) if match is not None else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _cutlass_dsl_version() -> Optional[Tuple[int, int, int]]:
    try:
        return _parse_version_tuple(importlib.metadata.version("nvidia-cutlass-dsl"))
    except Exception:
        return None


_CUTLASS_DSL_VERSION = _cutlass_dsl_version()
# CuTeDSL 4.3.4 tightened some kernel argument expectations (notably around
# passing Layout/Shape/Constexpr objects into @cute.kernel functions). Keep the
# older signature for 4.3.2, but switch to a 4.3.4-compatible signature when we
# detect 4.3.4+ (or when version detection is unavailable).
_KERNEL_ACCEPTS_LAYOUT_ARGS = (
    _CUTLASS_DSL_VERSION is not None and _CUTLASS_DSL_VERSION < (4, 3, 4)
)

if _ENABLE_CLUSTER_ILP and not _ENABLE_CLUSTER_ILP_UNSAFE:
    # We have observed reproducible segfaults in some CuTeDSL builds when using
    # cluster launches for this schedule. Require an explicit UNSAFE opt-in to
    # avoid accidental crashes.
    _ENABLE_CLUSTER_ILP = False
    print(
        "Oink: OINK_RMSNORM_ENABLE_CLUSTER_ILP requested but disabled by default due to "
        "known instability; set OINK_RMSNORM_ENABLE_CLUSTER_ILP_UNSAFE=1 to force-enable.",
        file=sys.stderr,
    )


def _fast_launch_enabled() -> bool:
    return _ENABLE_FAST_LAUNCH and _FAST_LAUNCH_SUPPORTED


def _direct_gmem_from_policy(*, default: bool) -> bool:
    """Resolve the direct-GMEM schedule flag from the (import-time) policy string."""
    if _DIRECT_GMEM_POLICY in {"0", "false", "no", "off"}:
        return False
    if _DIRECT_GMEM_POLICY in {"1", "true", "yes", "on"}:
        return True
    return default


def _copy_bits_from_policy(*, default: int, can_use_256: bool) -> int:
    """Resolve copy width (in bits) from the (import-time) policy string."""
    if _COPY_BITS_POLICY in {"128"}:
        return 128
    if _COPY_BITS_POLICY in {"256"} and can_use_256:
        return 256
    return default


class _StableI32Arg:
    """A stable Int32 runtime arg (avoids per-call Int32().__c_pointers__ allocations)."""

    def __init__(self, value: int):
        self._c_value = ctypes.c_int32(int(value))
        self._c_pointer = ctypes.cast(ctypes.pointer(self._c_value), ctypes.c_void_p)

    def set(self, value: int) -> None:
        self._c_value.value = int(value)

    def __c_pointers__(self):
        return [self._c_pointer]


class _StableF32Arg:
    """A stable Float32 runtime arg (avoids per-call Float32().__c_pointers__ allocations)."""

    def __init__(self, value: float):
        self._c_value = ctypes.c_float(float(value))
        self._c_pointer = ctypes.cast(ctypes.pointer(self._c_value), ctypes.c_void_p)

    def set(self, value: float) -> None:
        self._c_value.value = float(value)

    def __c_pointers__(self):
        return [self._c_pointer]


def _tls_fast_launch_cache() -> dict[tuple[object, ...], object]:
    cache = getattr(_PTR_FAST_LAUNCH_TLS, "cache", None)
    if cache is None:
        cache = {}
        _PTR_FAST_LAUNCH_TLS.cache = cache
    return cache


def _set_runtime_ptr(ptr: object, device_ptr: int) -> None:
    # Runtime pointer objects cache a `ctypes.c_void_p` descriptor and pass
    # its address to the compiled function. Updating `_desc.value` updates
    # the device pointer without changing the address of the descriptor.
    #
    # This relies on internal CuTeDSL runtime pointer fields (`_desc`, `_pointer`,
    # etc.). If these internals change in a future CuTeDSL upgrade, callers
    # should catch AttributeError and fall back to the regular launch path.
    device_ptr = int(device_ptr)
    ptr._pointer = device_ptr  # type: ignore[attr-defined]
    if getattr(ptr, "_c_pointer", None) is None:
        ptr.__c_pointers__()  # type: ignore[attr-defined]
    ptr._desc.value = device_ptr  # type: ignore[attr-defined]


class _PtrRmsnormFastLaunch:
    def __init__(
        self,
        *,
        compiled: object,
        executor: object,
        capi_func: object,
        ptr_x: object,
        ptr_w: Optional[object],
        ptr_out: object,
        arg_m: _StableI32Arg,
        arg_n: _StableI32Arg,
        arg_ld: _StableI32Arg,
        arg_eps: _StableF32Arg,
        stream: cuda.CUstream,
        packed_args: object,
        keepalive: tuple[object, ...],
    ):
        self._compiled = compiled
        self._executor = executor
        self._capi_func = capi_func
        self._ptr_x = ptr_x
        self._ptr_w = ptr_w
        self._ptr_out = ptr_out
        self._arg_m = arg_m
        self._arg_n = arg_n
        self._arg_ld = arg_ld
        self._arg_eps = arg_eps
        self._stream = stream
        self._packed_args = packed_args
        self._keepalive = keepalive

        self._use_fast_launch = True

        self._cuda_result = getattr(executor, "cuda_result", None)

        self._last_x_ptr = -1
        self._last_w_ptr = -1
        self._last_out_ptr = -1
        self._last_m = -1
        self._last_ld = -1
        self._last_eps = float("nan")

    def launch(
        self,
        *,
        x: Tensor,
        weight: Optional[Tensor],
        out: Tensor,
        M: int,
        N: int,
        ld: int,
        eps: float,
    ) -> None:
        if not _fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(x=x, weight=weight, out=out, M=M, N=N, ld=ld, eps=eps)
            return

        x_ptr = x.data_ptr()
        if x_ptr != self._last_x_ptr:
            try:
                _set_runtime_ptr(self._ptr_x, x_ptr)
                self._last_x_ptr = x_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x, weight=weight, out=out, M=M, N=N, ld=ld, eps=eps
                )
                return

        if self._ptr_w is not None:
            w_ptr = weight.data_ptr()  # type: ignore[union-attr]
            if w_ptr != self._last_w_ptr:
                try:
                    _set_runtime_ptr(self._ptr_w, w_ptr)
                    self._last_w_ptr = w_ptr
                except AttributeError:
                    self._disable_fast_launch()
                    self._fallback_launch(
                        x=x, weight=weight, out=out, M=M, N=N, ld=ld, eps=eps
                    )
                    return

        out_ptr = out.data_ptr()
        if out_ptr != self._last_out_ptr:
            try:
                _set_runtime_ptr(self._ptr_out, out_ptr)
                self._last_out_ptr = out_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x, weight=weight, out=out, M=M, N=N, ld=ld, eps=eps
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

        # Clear the error slot before launch (mirrors JitExecutor behavior).
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
        global _FAST_LAUNCH_SUPPORTED
        self._use_fast_launch = False
        _FAST_LAUNCH_SUPPORTED = False

    def _fallback_launch(
        self,
        *,
        x: Tensor,
        weight: Optional[Tensor],
        out: Tensor,
        M: int,
        N: int,
        ld: int,
        eps: float,
    ) -> None:
        # If the packed-args or runtime pointer mutation path stops working
        # (e.g. due to a CuTeDSL upgrade), fall back to the regular call path.
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x = rt.make_ptr(
            dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_w = (
            rt.make_ptr(
                dtype,
                weight.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if weight is not None
            else None
        )
        self._compiled(
            ptr_x,
            ptr_w,
            None,  # ptr_b
            None,  # ptr_res
            ptr_out,
            None,  # ptr_res_out
            None,  # ptr_rstd
            Int32(M),
            Int32(N),
            Int32(ld),
            self._stream,
            Float32(eps),
        )


class _PtrFusedAddRmsnormFastLaunch:
    def __init__(
        self,
        *,
        compiled: object,
        executor: object,
        capi_func: object,
        ptr_x: object,
        ptr_w: object,
        ptr_res: object,
        arg_m: _StableI32Arg,
        arg_n: _StableI32Arg,
        arg_ld_x: _StableI32Arg,
        arg_eps: _StableF32Arg,
        stream: cuda.CUstream,
        assumed_align: int,
        packed_args: object,
        keepalive: tuple[object, ...],
    ):
        self._compiled = compiled
        self._executor = executor
        self._capi_func = capi_func
        self._ptr_x = ptr_x
        self._ptr_w = ptr_w
        self._ptr_res = ptr_res
        self._arg_m = arg_m
        self._arg_n = arg_n
        self._arg_ld_x = arg_ld_x
        self._arg_eps = arg_eps
        self._stream = stream
        self._assumed_align = int(assumed_align)
        self._packed_args = packed_args
        self._keepalive = keepalive

        self._use_fast_launch = True

        self._cuda_result = getattr(executor, "cuda_result", None)

        self._last_x_ptr = -1
        self._last_w_ptr = -1
        self._last_res_ptr = -1
        self._last_m = -1
        self._last_ld_x = -1
        self._last_eps = float("nan")

    def launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        residual: Tensor,
        M: int,
        N: int,
        ld_x: int,
        eps: float,
    ) -> None:
        if not _fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(
                x=x, weight=weight, residual=residual, M=M, N=N, ld_x=ld_x, eps=eps
            )
            return

        x_ptr = x.data_ptr()
        if x_ptr != self._last_x_ptr:
            try:
                _set_runtime_ptr(self._ptr_x, x_ptr)
                self._last_x_ptr = x_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x, weight=weight, residual=residual, M=M, N=N, ld_x=ld_x, eps=eps
                )
                return

        w_ptr = weight.data_ptr()
        if w_ptr != self._last_w_ptr:
            try:
                _set_runtime_ptr(self._ptr_w, w_ptr)
                self._last_w_ptr = w_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x, weight=weight, residual=residual, M=M, N=N, ld_x=ld_x, eps=eps
                )
                return

        res_ptr = residual.data_ptr()
        if res_ptr != self._last_res_ptr:
            try:
                _set_runtime_ptr(self._ptr_res, res_ptr)
                self._last_res_ptr = res_ptr
            except AttributeError:
                self._disable_fast_launch()
                self._fallback_launch(
                    x=x, weight=weight, residual=residual, M=M, N=N, ld_x=ld_x, eps=eps
                )
                return

        if M != self._last_m:
            self._arg_m.set(M)
            self._last_m = M
        if ld_x != self._last_ld_x:
            self._arg_ld_x.set(ld_x)
            self._last_ld_x = ld_x
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
        global _FAST_LAUNCH_SUPPORTED
        self._use_fast_launch = False
        _FAST_LAUNCH_SUPPORTED = False

    def _fallback_launch(
        self,
        *,
        x: Tensor,
        weight: Tensor,
        residual: Tensor,
        M: int,
        N: int,
        ld_x: int,
        eps: float,
    ) -> None:
        dtype = TORCH2CUTE_DTYPE[x.dtype]
        ptr_x = rt.make_ptr(
            dtype,
            x.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align,
        )
        ptr_res = rt.make_ptr(
            dtype,
            residual.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align,
        )
        ptr_w = rt.make_ptr(
            dtype,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=self._assumed_align,
        )
        self._compiled(
            ptr_x,
            ptr_w,
            ptr_res,
            Int32(M),
            Int32(N),
            Int32(ld_x),
            self._stream,
            Float32(eps),
        )


def _get_fast_ptr_rmsnorm_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    N: int,
    device_index: int,
    stream_handle: int,
    has_weight: bool,
    eps: float,
) -> Optional[_PtrRmsnormFastLaunch]:
    if not _fast_launch_enabled():
        return None
    # Keyed by the compiled object identity so schedule changes (e.g. copy width,
    # async/staged variants, etc.) never alias in the fast-launch cache.
    key = (
        "ptr_fast",
        id(compiled),
        N,
        dtype,
        device_index,
        int(stream_handle),
        has_weight,
    )
    cache = _tls_fast_launch_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    # Create stable runtime args and pointer descriptors once.
    ptr_x = rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=16)
    ptr_out = rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=16)
    ptr_w = (
        rt.make_ptr(dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=16)
        if has_weight
        else None
    )

    arg_m = _StableI32Arg(0)
    arg_n = _StableI32Arg(N)
    arg_ld = _StableI32Arg(N)
    arg_eps = _StableF32Arg(eps)

    stream = cuda.CUstream(int(stream_handle))

    # Create an executor (loads the CUDA library once).
    executor = compiled.to(device_index)  # type: ignore[attr-defined]

    # Use generate_execution_args once to build the packed args array, and keep
    # any adapted args alive for the lifetime of the cache entry.
    try:
        exe_args, adapted_args = executor.generate_execution_args(
            ptr_x,
            ptr_w,
            None,  # ptr_b
            None,  # ptr_res
            ptr_out,
            None,  # ptr_res_out
            None,  # ptr_rstd
            arg_m,
            arg_n,
            arg_ld,
            stream,
            arg_eps,
        )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        global _FAST_LAUNCH_SUPPORTED
        _FAST_LAUNCH_SUPPORTED = False
        return None

    keepalive: tuple[object, ...] = (
        executor,
        ptr_x,
        ptr_w,
        ptr_out,
        arg_m,
        arg_n,
        arg_ld,
        arg_eps,
        stream,
        *adapted_args,
    )

    launcher = _PtrRmsnormFastLaunch(
        compiled=compiled,
        executor=executor,
        capi_func=capi_func,
        ptr_x=ptr_x,
        ptr_w=ptr_w,
        ptr_out=ptr_out,
        arg_m=arg_m,
        arg_n=arg_n,
        arg_ld=arg_ld,
        arg_eps=arg_eps,
        stream=stream,
        packed_args=packed_args,
        keepalive=keepalive,
    )
    cache[key] = launcher
    return launcher


def _get_fast_ptr_fused_add_rmsnorm_launcher(
    *,
    compiled: object,
    dtype: type[cutlass.Numeric],
    N: int,
    device_index: int,
    stream_handle: int,
    copy_bits: int,
    use_async: bool,
    tpr: int,
    direct_gmem: bool,
    assumed_align: int,
    eps: float,
) -> Optional[_PtrFusedAddRmsnormFastLaunch]:
    if not _fast_launch_enabled():
        return None
    key = (
        "ptr_fused_add_fast",
        id(compiled),
        N,
        dtype,
        device_index,
        int(stream_handle),
        int(copy_bits),
        bool(use_async),
        int(tpr),
        bool(direct_gmem),
        int(assumed_align),
    )
    cache = _tls_fast_launch_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    ptr_x = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )
    ptr_res = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )
    ptr_w = rt.make_ptr(
        dtype, 0, mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )

    arg_m = _StableI32Arg(0)
    arg_n = _StableI32Arg(N)
    arg_ld_x = _StableI32Arg(N)
    arg_eps = _StableF32Arg(eps)

    stream = cuda.CUstream(int(stream_handle))

    executor = compiled.to(device_index)  # type: ignore[attr-defined]

    try:
        exe_args, adapted_args = executor.generate_execution_args(
            ptr_x,
            ptr_w,
            ptr_res,
            arg_m,
            arg_n,
            arg_ld_x,
            stream,
            arg_eps,
        )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        global _FAST_LAUNCH_SUPPORTED
        _FAST_LAUNCH_SUPPORTED = False
        return None

    keepalive: tuple[object, ...] = (
        executor,
        ptr_x,
        ptr_w,
        ptr_res,
        arg_m,
        arg_n,
        arg_ld_x,
        arg_eps,
        stream,
        *adapted_args,
    )

    launcher = _PtrFusedAddRmsnormFastLaunch(
        compiled=compiled,
        executor=executor,
        capi_func=capi_func,
        ptr_x=ptr_x,
        ptr_w=ptr_w,
        ptr_res=ptr_res,
        arg_m=arg_m,
        arg_n=arg_n,
        arg_ld_x=arg_ld_x,
        arg_eps=arg_eps,
        stream=stream,
        assumed_align=assumed_align,
        packed_args=packed_args,
        keepalive=keepalive,
    )
    cache[key] = launcher
    return launcher


# Local helpers for reduction, dtype mapping, and coordinate/predicate utilities.
#
# NOTE: Avoid `from . import ...` imports here: CuTeDSL's AST preprocessor may
# mishandle that form (module=None in the AST). Use fully-qualified imports.
from kernelagent_oink.blackwell import lite_quack as qutils  # noqa: E402
from kernelagent_oink.blackwell.lite_quack import (  # noqa: E402
    TORCH2CUTE_DTYPE,
    RMSNormBackward as BaseRMSNormBackward,
    convert_from_dlpack as convert_from_dlpack_cute,
    get_sm_count,
    row_reduce,
)


# -------------------------
# Copy helpers (allow up to 256b)
# -------------------------


@cute.jit
def get_copy_atom_bw(
    dtype: type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False
) -> cute.CopyAtom:
    # cp.async (SIMT) supports up to 128b per op; use 256b for sync when possible
    max_bits = const_expr(128 if is_async else 256)
    num_copy_bits = const_expr(min(max_bits, num_copy_elems * dtype.width))
    from cutlass.cute.nvgpu import cpasync

    # Prefer GLOBAL cache policy for bulk streaming reads at large M
    copy_op = (
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
        if is_async
        else cute.nvgpu.CopyUniversalOp()
    )
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@cute.jit
def copy_tiled(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> None:
    atom = get_copy_atom_bw(src.element_type, num_copy_elems, is_async)
    cute.copy(atom, src, dst, pred=pred)


# -------------------------
# RMSNorm Kernel (SM100)
# -------------------------


class RMSNormSM100:
    def __init__(
        self,
        N: int,
        dtype: type[cutlass.Numeric],
        stage: Optional[int] = None,
        *,
        copy_bits: int = 128,
        use_async: bool = True,
        direct_gmem: bool = False,
    ):
        self.N = N
        self.dtype = dtype
        # Match Quack default for RMSNorm: stage = 1 unless explicitly overridden
        self.stage = 1 if stage is None else stage
        self.reduction_dtype = cutlass.Float32
        self.copy_bits = int(copy_bits)
        self.use_async = bool(use_async)
        self.direct_gmem = bool(direct_gmem)

    def _threads_per_row(self) -> int:
        try:
            return self._tpr_override  # type: ignore[attr-defined]
        except Exception:
            pass
        # Tune mid-size buckets for large-M rows.
        N = self.N
        # DSv3 MLA (padded/strided) hot shape. Prefer a threads-per-row that
        # makes the tile width exactly match N with 128b vectors (bf16/fp16),
        # avoiding the ~33% padded work from rounding 1536 -> 2048.
        if N == 1536 and self.dtype.width == 16:
            return 96
        # DSv3 default hidden size (7168). Choose a threads-per-row that matches
        # the selected vector width to avoid padded work:
        # - 128b copies (vec=8 for bf16/fp16): 7168/8 = 896 = 7 * 128  -> tpr=128
        # - 256b copies (vec=16 for bf16/fp16): 7168/16 = 448 = 2 * 224 -> tpr=224
        #
        # The fused direct-GMEM path often uses 256b copies on 32B-aligned
        # tensors, while the non-fused path defaults to 128b copies.
        if N == 7168 and self.dtype.width == 16:
            return 224 if self.copy_bits >= 256 else 128
        # For small-N, use at least one full warp per row. The kernel
        # implementation assumes one row per CTA; returning <32 here can
        # produce multi-row tiles (cols_per_block > 1) which is not supported.
        if N <= 1024:
            return 32
        elif N <= 4096:
            return 128
        elif N <= 8192:
            # Allow an override (used by 2-rows/CTA path for N≈6k/8k)
            try:
                return self._tpr_override  # type: ignore[attr-defined]
            except Exception:
                return 128
        elif N <= 16384:
            return 256
        else:
            return 256

    def _cluster_n(self) -> int:
        try:
            return self._cluster_n_override  # type: ignore[attr-defined]
        except Exception:
            pass
        N = self.N
        # Default policy
        if N <= 8192:
            return 1
        if const_expr(self.dtype.width == 16):
            if N <= 16 * 1024:
                return 2
            elif N <= 32 * 1024:
                return 2
            elif N <= 64 * 1024:
                return 4
            elif N <= 128 * 1024:
                return 8
            else:
                return 16
        else:
            if N <= 32 * 1024:
                return 1
            elif N <= 64 * 1024:
                return 2
            elif N <= 128 * 1024:
                return 4
            elif N <= 256 * 1024:
                return 8
            else:
                return 16

    def _num_threads(self) -> int:
        # Favor 128 threads up to N=16k to reduce per-row partitioning overhead.
        # This keeps cols_per_block=1 at N=8192 (bf16), which benchmarks faster for large-M.
        try:
            return self._nt_override  # type: ignore[attr-defined]
        except Exception:
            if self.N == 1536 and self.dtype.width == 16:
                return 96
            if self.N == 7168 and self.dtype.width == 16:
                return 224 if self.copy_bits >= 256 else 128
            if self.N <= 1024:
                return 32
            return 128 if self.N <= 16384 else 256

    def _tv_layout(self, num_copy_bits: int = 256) -> Tuple[cute.Shape, cute.Layout]:
        vecsize = num_copy_bits // self.dtype.width
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        tpr = self._threads_per_row()
        cluster_n = self._cluster_n()
        # Allow tails: compute number of vector columns with ceil
        num_cols_vec = cute.ceil_div(self.N, vecsize)
        num_blocks_N = cute.ceil_div(num_cols_vec, tpr * cluster_n)
        cols_per_block = num_threads // tpr
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * tpr)
        tv_layout = cute.make_layout(
            ((tpr, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * tpr),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_bytes(self, tiler_mn, num_warps) -> int:
        # smem for X tile (+ residual if present) + reduction buffers + mbar(s)
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage
            * num_warps
            * self._cluster_n()
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        # Make last dim static (N)
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=256 // t.element_type.width),
                t.stride[1],
            )

        mX, mRes, mO, mResO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype

        copy_bits = int(self.copy_bits)
        tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        threads_per_row = (
            tv_layout.shape[0][0]
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._threads_per_row()
        )
        warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
        cluster_n = self._cluster_n()

        if const_expr(mW is not None):
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

        # No SMEM reload mode switch; overlap is controlled in the K-loop path

        # Compute smem usage considering staged buffers.
        #
        # In direct-gmem mode, we skip the gmem->smem tiles entirely and only
        # keep the reduction buffers in shared memory.
        stage_bufs = 2 if self.stage > 1 else 1
        tile_bytes_x = (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * stage_bufs
            if const_expr(not self.direct_gmem)
            else 0
        )
        tile_bytes_res = (
            cute.size_in_bytes(mRes.element_type, cute.make_layout(tiler_mn))
            * stage_bufs
            if const_expr(mRes is not None and not self.direct_gmem)
            else 0
        )
        red_bytes = (
            self.stage * num_warps * cluster_n * (self.reduction_dtype.width // 8)
        )
        # mbarriers are only allocated/used for cluster_n>1. Some CuTeDSL builds
        # require mbarrier state to be 16B-aligned in shared memory; account for
        # the alignment padding when computing dynamic smem bytes.
        smem_bytes = tile_bytes_x + tile_bytes_res + red_bytes
        if cluster_n > 1:
            # Align up to 16B before placing the mbarrier array.
            smem_bytes = ((smem_bytes + 15) // 16) * 16
            smem_bytes += self.stage * (cutlass.Int64.width // 8)

        kernel = (
            self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(cluster_n),
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
            )
        )
        kernel.launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, cluster_n, 1] if cluster_n > 1 else None),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def launch_from_ptrs(
        self,
        ptr_x: cute.Pointer,
        ptr_w: Optional[cute.Pointer],
        ptr_b: Optional[cute.Pointer],
        ptr_res: Optional[cute.Pointer],
        ptr_out: cute.Pointer,
        ptr_res_out: Optional[cute.Pointer],
        ptr_rstd: Optional[cute.Pointer],
        M: Int32,
        N_dyn: Int32,
        ld: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        """Pointer-based entrypoint to reuse the existing RMSNorm schedule.

        This reconstructs cute.Tensor views from raw pointers plus sizes,
        avoiding any DLPack conversions at the Python boundary.
        """
        # Use a dynamic N for the leading-dimension stride so that the
        # subsequent cute.assume(...) in __call__ sees a dynamic expression
        # rather than a plain Python int.
        # The compile-time N for the kernel (self.N) is still used to
        # specialize the schedule.
        # Assume row-major [M, N] with an arbitrary leading-dimension stride
        # (common for padded-row / packed-attention layouts).
        layout_mn = cute.make_layout((M, N_dyn), stride=(ld, 1))
        layout_n = cute.make_layout((N_dyn,), stride=(1,))
        layout_m = cute.make_layout((M,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_mn)
        mO = cute.make_tensor(ptr_out, layout_mn)

        mRes = (
            cute.make_tensor(ptr_res, layout_mn)
            if const_expr(ptr_res is not None)
            else None
        )
        mResO = (
            cute.make_tensor(ptr_res_out, layout_mn)
            if const_expr(ptr_res_out is not None)
            else None
        )
        mW = (
            cute.make_tensor(ptr_w, layout_n) if const_expr(ptr_w is not None) else None
        )
        mB = (
            cute.make_tensor(ptr_b, layout_n) if const_expr(ptr_b is not None) else None
        )
        mRstd = (
            cute.make_tensor(ptr_rstd, layout_m)
            if const_expr(ptr_rstd is not None)
            else None
        )

        # Reuse the main JIT entry to launch the scheduled kernel.
        self.__call__(mX, mW, mB, mRes, mO, mResO, mRstd, stream, eps)

    @cute.jit
    def launch_from_ptrs_fused_add_inplace(
        self,
        ptr_x: cute.Pointer,
        ptr_w: cute.Pointer,
        ptr_res: cute.Pointer,
        M: Int32,
        N_dyn: Int32,
        ld_x: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        """Pointer-based entrypoint for vLLM-style fused_add_rms_norm semantics.

        This specialized entrypoint supports:
        - `x` / output with an arbitrary leading-dimension stride (`ld_x`), and
        - `residual` / residual-out as a contiguous [M, N] tensor (ld_res = N).

        Both `x` and `residual` are updated in-place:
          residual <- x + residual
          x <- RMSNorm(residual) * weight
        """
        layout_x = cute.make_layout((M, N_dyn), stride=(ld_x, 1))
        layout_res = cute.make_layout((M, N_dyn), stride=(N_dyn, 1))
        layout_n = cute.make_layout((N_dyn,), stride=(1,))

        mX = cute.make_tensor(ptr_x, layout_x)
        mO = cute.make_tensor(ptr_x, layout_x)
        mRes = cute.make_tensor(ptr_res, layout_res)
        mResO = cute.make_tensor(ptr_res, layout_res)
        mW = cute.make_tensor(ptr_w, layout_n)

        self.__call__(
            mX,
            mW,
            None,  # bias
            mRes,
            mO,
            mResO,
            None,  # rstd
            stream,
            eps,
        )

    @cute.jit
    def _kernel_impl(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        cluster_n: cutlass.Constexpr[int],
        num_warps: cutlass.Constexpr[int],
        warps_per_row: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cluster_n > 1):
            cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        else:
            cta_rank_in_cluster = const_expr(0)
        n_off = cta_rank_in_cluster * tiler_mn[1]

        smem = cutlass.utils.SmemAllocator()
        # Allocate one or two SMEM buffers depending on stage depth
        sX0 = (
            smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(not self.direct_gmem)
            else None
        )
        sX1 = (
            smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(self.stage > 1 and not self.direct_gmem)
            else None
        )
        sRes0 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None and not self.direct_gmem)
            else None
        )
        sRes1 = (
            smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=32,
            )
            if const_expr(mRes is not None and self.stage > 1 and not self.direct_gmem)
            else None
        )

        # Reduction buffers + mbar for cluster reduce (reused by row_reduce helper)
        red_layout = cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype, red_layout, byte_alignment=4
        )
        if const_expr(cluster_n > 1):
            # Some CuTeDSL builds appear sensitive to the shared-memory alignment of
            # mbarrier state. `SmemAllocator.allocate_array` does not currently
            # expose an alignment parameter, so allocate an Int64 tensor with an
            # explicit alignment and pass its iterator as the pointer.
            mbar_tensor = smem.allocate_tensor(
                cutlass.Int64,
                cute.make_layout((self.stage,), stride=(1,)),
                byte_alignment=16,
            )
            mbar_ptr = mbar_tensor.iterator
        else:
            mbar_ptr = None

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        limit_k = shape[1] - n_off

        # Tiled copy setup
        num_copy_elems_X = tv_layout.shape[1][0]
        use_async = const_expr(
            self.use_async and self.N >= 1024 and not self.direct_gmem
        )
        copy_atom = get_copy_atom_bw(
            mX.element_type, num_copy_elems_X, is_async=use_async
        )
        thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)

        # Tail predicate for the N dimension (when tile width > N). Reuse this
        # for W/B loads so we never read past the end of those 1D tensors.
        is_even_N_wb = const_expr(shape[1] == tiler_mn[1] * cluster_n)
        if const_expr(not is_even_N_wb):
            cX0 = cute.local_tile(idX, tiler_mn, (0, 0))
            tXp_wb = qutils.predicate_k(thr_copy.partition_S(cX0), limit=limit_k)
        else:
            tXp_wb = None

        # Weight/bias loads:
        #
        # - Direct-GMEM schedule: load weight/bias up front to hide latency.
        # - Staged SMEM schedule: loading after the reduction reduces register
        #   pressure during the long-scoreboard reduction phase (better for large-M),
        #   but it measurably hurts small-M latency for the non-fused (no residual,
        #   no bias) case. For that specific case, prefetch weight up front as well.
        tXrW = None
        tXrB = None
        prefetch_w_early = bool(
            mW is not None and (self.direct_gmem or (mRes is None and mB is None))
        )
        if const_expr(prefetch_w_early):
            gW = cute.local_tile(
                qutils.domain_offset_i64((0, n_off), mW), tiler_mn, (0, 0)
            )
            tXgW = thr_copy.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N_wb):
                tXrW.fill(0)
            cute.copy(
                get_copy_atom_bw(mW.element_type, num_copy_elems_X, is_async=False),
                tXgW,
                tXrW,
                pred=tXp_wb,
            )
        if const_expr(self.direct_gmem and mB is not None):
            gB = cute.local_tile(
                qutils.domain_offset_i64((0, n_off), mB), tiler_mn, (0, 0)
            )
            tXgB = thr_copy.partition_S(gB)
            tXrB = cute.make_fragment_like(tXgB)
            if const_expr(not is_even_N_wb):
                tXrB.fill(0)
            cute.copy(
                get_copy_atom_bw(mB.element_type, num_copy_elems_X, is_async=False),
                tXgB,
                tXrB,
                pred=tXp_wb,
            )

        # Non-persistent per-CTA execution (one tile in M)
        self._init_cluster(tidx, mbar_ptr)

        mX_i, mRes_i, mO_i, mResO_i = [
            qutils.domain_offset_i64((bidx * tiler_mn[0], 0), t)
            if t is not None
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        mX_i, mRes_i, mO_i, mResO_i = [
            qutils.domain_offset_i64((0, n_off), t) if t is not None else None
            for t in (mX_i, mRes_i, mO_i, mResO_i)
        ]
        gX_i = cute.local_tile(mX_i, tiler_mn, (0, 0))
        gO_i = cute.local_tile(mO_i, tiler_mn, (0, 0))
        gRes_i = (
            cute.local_tile(mRes_i, tiler_mn, (0, 0))
            if const_expr(mRes is not None)
            else None
        )
        gResO_i = (
            cute.local_tile(mResO_i, tiler_mn, (0, 0))
            if const_expr(mResO is not None)
            else None
        )
        gRstd_i = (
            cute.local_tile(mRstd, tiler_mn, (bidx, 0))
            if const_expr(mRstd is not None)
            else None
        )
        cX_i = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Common identity/row index partitions reused by both default and K-loop paths
        tXcX_i = thr_copy.partition_S(cX_i)[(0, None), None, None]
        row_i = tXcX_i[0][0]
        tXgRstd_i = (
            thr_copy.partition_D(gRstd_i) if const_expr(mRstd is not None) else None
        )

        # Stage-2 intra-row K-loop cp.async ping-pong (two tiles). This reduces
        # per-thread fragment size and can improve memory-latency hiding for
        # N=7168 at large M. It is enabled by setting `stage=2` when constructing
        # the RMSNormSM100 op (see `_fused_add_rmsnorm_forward_ptr_inplace`).
        if const_expr(
            self.stage > 1
            and not self.direct_gmem
            and use_async
            and cluster_n == 1
            and shape[1] == 7168
        ):
            vecsize = tv_layout.shape[1][0]
            tpr = threads_per_row
            target_tile_n = const_expr(4096)
            tile_factor = const_expr(target_tile_n // (vecsize * tpr))
            if const_expr(tile_factor > 0):
                tile_n = vecsize * tpr * tile_factor
                num_tiles = cute.ceil_div(shape[1], tile_n)

                tiler_mn_tile = (tiler_mn[0], tile_n)
                sX0_tile = cute.local_tile(sX0, tiler_mn_tile, (0, 0))
                sX1_tile = cute.local_tile(sX1, tiler_mn_tile, (0, 0))
                sRes0_tile = (
                    cute.local_tile(sRes0, tiler_mn_tile, (0, 0))
                    if const_expr(mRes is not None)
                    else None
                )
                sRes1_tile = (
                    cute.local_tile(sRes1, tiler_mn_tile, (0, 0))
                    if const_expr(mRes is not None)
                    else None
                )

                tv_layout_tile = cute.make_layout(
                    ((tpr, tiler_mn[0]), (vecsize, tile_factor)),
                    stride=(
                        (vecsize * tiler_mn[0], 1),
                        (tiler_mn[0], tiler_mn[0] * vecsize * tpr),
                    ),
                )
                thr_copy_tile = cute.make_tiled_copy(
                    copy_atom, tv_layout_tile, tiler_mn_tile
                ).get_slice(tidx)

                # Accumulate per-thread partial sums across tiles; reduce once.
                sum_sq_thread = cute.Float32(0.0)

                # Preload tile 0 into sX0/sRes0.
                k_off0 = const_expr(0) * tile_n
                gX_0 = cute.local_tile(
                    qutils.domain_offset_i64((0, k_off0), mX_i), tiler_mn_tile, (0, 0)
                )
                tXgX_0 = thr_copy_tile.partition_S(gX_0)
                tXsX_0 = thr_copy_tile.partition_D(sX0_tile)
                cX_0 = cute.local_tile(
                    cute.domain_offset((0, k_off0), cX_i), tiler_mn_tile, (0, 0)
                )
                tXc_0 = thr_copy_tile.partition_S(cX_0)
                tXp_0 = qutils.predicate_k(tXc_0, limit=limit_k)

                tXp_ping = tXp_0
                tXp_pong = tXp_0

                if row_i < shape[0]:
                    copy_tiled(
                        tXgX_0,
                        tXsX_0,
                        num_copy_elems=vecsize,
                        is_async=True,
                        pred=tXp_0,
                    )
                    if const_expr(mRes is not None):
                        gRes_0 = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off0), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_0 = thr_copy_tile.partition_S(gRes_0)
                        tXsRes_0 = thr_copy_tile.partition_D(sRes0_tile)
                        copy_tiled(
                            tXgRes_0,
                            tXsRes_0,
                            num_copy_elems=vecsize,
                            is_async=True,
                            pred=tXp_0,
                        )
                cute.arch.cp_async_commit_group()

                for t in cutlass.range_constexpr(num_tiles):
                    next_t = t + 1
                    if next_t < num_tiles:
                        k_off_n = next_t * tile_n
                        gX_n = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off_n), mX_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgX_n = thr_copy_tile.partition_S(gX_n)
                        cX_n = cute.local_tile(
                            cute.domain_offset((0, k_off_n), cX_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXc_n = thr_copy_tile.partition_S(cX_n)
                        tXp_n = qutils.predicate_k(tXc_n, limit=limit_k)

                        if const_expr((t % 2) == 0):
                            tXsX_n = thr_copy_tile.partition_D(sX1_tile)
                            tXsRes_n = (
                                thr_copy_tile.partition_D(sRes1_tile)
                                if const_expr(mRes is not None)
                                else None
                            )
                            tXp_pong = tXp_n
                        else:
                            tXsX_n = thr_copy_tile.partition_D(sX0_tile)
                            tXsRes_n = (
                                thr_copy_tile.partition_D(sRes0_tile)
                                if const_expr(mRes is not None)
                                else None
                            )
                            tXp_ping = tXp_n

                        if row_i < shape[0]:
                            copy_tiled(
                                tXgX_n,
                                tXsX_n,
                                num_copy_elems=vecsize,
                                is_async=True,
                                pred=tXp_n,
                            )
                            if const_expr(mRes is not None):
                                gRes_n = cute.local_tile(
                                    qutils.domain_offset_i64((0, k_off_n), mRes_i),
                                    tiler_mn_tile,
                                    (0, 0),
                                )
                                tXgRes_n = thr_copy_tile.partition_S(gRes_n)
                                copy_tiled(
                                    tXgRes_n,
                                    tXsRes_n,
                                    num_copy_elems=vecsize,
                                    is_async=True,
                                    pred=tXp_n,
                                )
                        cute.arch.cp_async_commit_group()

                    cute.arch.cp_async_wait_group(1 if next_t < num_tiles else 0)

                    # Current tile buffer (ping/pong).
                    if const_expr((t % 2) == 0):
                        tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes0_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        pred_cur = tXp_ping
                    else:
                        tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes1_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                        pred_cur = tXp_pong

                    k_off = t * tile_n
                    gX_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mX_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgX_t = thr_copy_tile.partition_S(gX_t)
                    tXrX_t = cute.make_fragment_like(tXgX_t)
                    cute.autovec_copy(tXsX_cur, tXrX_t)
                    x_t = tXrX_t.load().to(cute.Float32)
                    if const_expr(mRes is not None):
                        gRes_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                        tXrRes_t = cute.make_fragment_like(tXgRes_t)
                        cute.autovec_copy(tXsRes_cur, tXrRes_t)
                        x_t += tXrRes_t.load().to(cute.Float32)

                    if const_expr(mResO is not None):
                        gResO_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mResO_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgResO_t = thr_copy_tile.partition_D(gResO_t)
                        tXrResO_t = cute.make_fragment_like(tXgResO_t)
                        tXrResO_t.store(x_t.to(tXrResO_t.element_type))
                        if row_i < shape[0]:
                            copy_tiled(
                                tXrResO_t,
                                tXgResO_t,
                                num_copy_elems=vecsize,
                                is_async=False,
                                pred=pred_cur,
                            )

                    sum_sq_thread = sum_sq_thread + (x_t * x_t).reduce(
                        cute.ReductionOp.ADD,
                        init_val=0.0,
                        reduction_profile=0,
                    )

                sum_sq = row_reduce(
                    sum_sq_thread,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                )
                rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

                if const_expr(mRstd is not None):
                    if tXcX_i[0][1] == 0 and row_i < shape[0]:
                        tXgRstd_i[0] = rstd

                for t in cutlass.range_constexpr(num_tiles):
                    k_off = t * tile_n
                    cX_t = cute.local_tile(
                        cute.domain_offset((0, k_off), cX_i), tiler_mn_tile, (0, 0)
                    )
                    tXc_t = thr_copy_tile.partition_S(cX_t)
                    tXp_t = qutils.predicate_k(tXc_t, limit=limit_k)

                    if const_expr((t % 2) == 0):
                        tXsX_cur = thr_copy_tile.partition_D(sX0_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes0_tile)
                            if const_expr(mRes is not None)
                            else None
                        )
                    else:
                        tXsX_cur = thr_copy_tile.partition_D(sX1_tile)
                        tXsRes_cur = (
                            thr_copy_tile.partition_D(sRes1_tile)
                            if const_expr(mRes is not None)
                            else None
                        )

                    gX_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mX_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgX_t = thr_copy_tile.partition_S(gX_t)
                    tXrX_t = cute.make_fragment_like(tXgX_t)
                    cute.autovec_copy(tXsX_cur, tXrX_t)
                    x_t = tXrX_t.load().to(cute.Float32)
                    if const_expr(mRes is not None):
                        gRes_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mRes_i),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tXgRes_t = thr_copy_tile.partition_S(gRes_t)
                        tXrRes_t = cute.make_fragment_like(tXgRes_t)
                        cute.autovec_copy(tXsRes_cur, tXrRes_t)
                        x_t += tXrRes_t.load().to(cute.Float32)

                    y_t = x_t * rstd
                    if const_expr(mW is not None):
                        gW_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mW),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tWgW_t = thr_copy_tile.partition_S(gW_t)
                        tWrW_t = cute.make_fragment_like(tWgW_t)
                        copy_tiled(
                            tWgW_t,
                            tWrW_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )
                        y_t = y_t * tWrW_t.load().to(cute.Float32)
                    if const_expr(mB is not None):
                        gB_t = cute.local_tile(
                            qutils.domain_offset_i64((0, k_off), mB),
                            tiler_mn_tile,
                            (0, 0),
                        )
                        tWgB_t = thr_copy_tile.partition_S(gB_t)
                        tWrB_t = cute.make_fragment_like(tWgB_t)
                        copy_tiled(
                            tWgB_t,
                            tWrB_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )
                        y_t = y_t + tWrB_t.load().to(cute.Float32)

                    gO_t = cute.local_tile(
                        qutils.domain_offset_i64((0, k_off), mO_i),
                        tiler_mn_tile,
                        (0, 0),
                    )
                    tXgO_t = thr_copy_tile.partition_D(gO_t)
                    tXrO_t = cute.make_fragment_like(tXgO_t)
                    tXrO_t.store(y_t.to(tXrO_t.element_type))
                    if row_i < shape[0]:
                        copy_tiled(
                            tXrO_t,
                            tXgO_t,
                            num_copy_elems=vecsize,
                            is_async=False,
                            pred=tXp_t,
                        )

                return

        # Single-stage path: one-row-per-CTA
        tXgX_i = thr_copy.partition_S(gX_i)
        tXgRes_i = (
            thr_copy.partition_S(gRes_i) if const_expr(mRes is not None) else None
        )
        tXgO_i = thr_copy.partition_D(gO_i)
        tXgResO_i = (
            thr_copy.partition_D(gResO_i) if const_expr(mResO is not None) else None
        )
        # tXgRstd_i / tXcX_i / row_i prepared above
        is_even_N_i = const_expr(shape[1] == tiler_mn[1] * cluster_n)
        tXpX_i = (
            qutils.predicate_k(thr_copy.partition_S(cX_i), limit=limit_k)
            if not is_even_N_i
            else None
        )

        tXrX = cute.make_fragment_like(tXgX_i)
        tXrRes = (
            cute.make_fragment_like(tXgRes_i) if const_expr(mRes is not None) else None
        )
        if const_expr(self.direct_gmem):
            if const_expr(not is_even_N_i):
                tXrX.fill(0)
                if const_expr(tXrRes is not None):
                    tXrRes.fill(0)
            if row_i < shape[0]:
                cute.copy(copy_atom, tXgX_i, tXrX, pred=tXpX_i)
                if const_expr(tXrRes is not None):
                    cute.copy(copy_atom, tXgRes_i, tXrRes, pred=tXpX_i)
        else:
            # If N is not a multiple of the tile width, the predicated gmem->smem
            # copies leave out-of-bounds lanes uninitialized. Clear the SMEM tile
            # so masked lanes read as 0 for reduction/output.
            if const_expr(not is_even_N_i):
                thr_copy.partition_D(sX0).fill(0)
                if const_expr(mRes is not None):
                    thr_copy.partition_D(sRes0).fill(0)

            if row_i < shape[0]:
                cute.copy(copy_atom, tXgX_i, thr_copy.partition_D(sX0), pred=tXpX_i)
                if const_expr(mRes is not None):
                    cute.copy(
                        copy_atom, tXgRes_i, thr_copy.partition_D(sRes0), pred=tXpX_i
                    )
            if const_expr(use_async):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(thr_copy.partition_D(sX0), tXrX)
            if const_expr(tXrRes is not None):
                cute.autovec_copy(thr_copy.partition_D(sRes0), tXrRes)
        x_red = tXrX.load().to(cute.Float32)
        if const_expr(tXrRes is not None):
            x_red += tXrRes.load().to(cute.Float32)

        if const_expr(mResO is not None):
            tXrResO = cute.make_fragment_like(tXgResO_i)
            tXrResO.store(x_red.to(tXrResO.element_type))
            if row_i < shape[0]:
                cute.copy(
                    get_copy_atom_bw(
                        tXrResO.element_type, num_copy_elems_X, is_async=False
                    ),
                    tXrResO,
                    tXgResO_i,
                    pred=tXpX_i,
                )

        sum_sq = row_reduce(
            x_red * x_red,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

        if const_expr(mRstd is not None):
            if (
                tXcX_i[0][1] == 0
                and row_i < shape[0]
                and (cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXgRstd_i[0] = rstd

        if const_expr(not self.direct_gmem and (mRes is not None or mB is not None)):
            # Load weight/bias after the reduction so they don't inflate register
            # pressure during the long-scoreboard reduction phase (helping occupancy
            # when registers are the limiting factor).
            if const_expr(mW is not None):
                gW = cute.local_tile(
                    qutils.domain_offset_i64((0, n_off), mW), tiler_mn, (0, 0)
                )
                tXgW = thr_copy.partition_S(gW)
                tXrW = cute.make_fragment_like(tXgW)
                if const_expr(not is_even_N_wb):
                    tXrW.fill(0)
                cute.copy(
                    get_copy_atom_bw(mW.element_type, num_copy_elems_X, is_async=False),
                    tXgW,
                    tXrW,
                    pred=tXp_wb,
                )
            if const_expr(mB is not None):
                gB = cute.local_tile(
                    qutils.domain_offset_i64((0, n_off), mB), tiler_mn, (0, 0)
                )
                tXgB = thr_copy.partition_S(gB)
                tXrB = cute.make_fragment_like(tXgB)
                if const_expr(not is_even_N_wb):
                    tXrB.fill(0)
                cute.copy(
                    get_copy_atom_bw(mB.element_type, num_copy_elems_X, is_async=False),
                    tXgB,
                    tXrB,
                    pred=tXp_wb,
                )

        # Reuse `x_red` (x + residual, in fp32) for the output path so we don't
        # keep both `tXrX` and `tXrRes` fragments live across the reduction.
        y = x_red * rstd
        if const_expr(mW is not None):
            y = y * tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y = y + tXrB.load().to(cute.Float32)

        tXrO = cute.make_fragment_like(tXgO_i)
        tXrO.store(y.to(tXrO.element_type))
        if row_i < shape[0]:
            cute.copy(
                get_copy_atom_bw(tXrO.element_type, num_copy_elems_X, is_async=False),
                tXrO,
                tXgO_i,
                pred=tXpX_i,
            )

    if _KERNEL_ACCEPTS_LAYOUT_ARGS:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: Optional[cute.Tensor],
            mB: Optional[cute.Tensor],
            mRes: Optional[cute.Tensor],
            mO: cute.Tensor,
            mResO: Optional[cute.Tensor],
            mRstd: Optional[cute.Tensor],
            eps: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
            cluster_n: cutlass.Constexpr[int],
            num_warps: cutlass.Constexpr[int],
            warps_per_row: cutlass.Constexpr[int],
            threads_per_row: cutlass.Constexpr[int],
        ):
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                cluster_n,
                num_warps,
                warps_per_row,
                threads_per_row,
            )
    else:

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: Optional[cute.Tensor],
            mB: Optional[cute.Tensor],
            mRes: Optional[cute.Tensor],
            mO: cute.Tensor,
            mResO: Optional[cute.Tensor],
            mRstd: Optional[cute.Tensor],
            eps: Float32,
        ):
            copy_bits = int(self.copy_bits)
            tiler_mn, tv_layout = self._tv_layout(num_copy_bits=copy_bits)
            num_threads = self._num_threads()
            num_warps = num_threads // cute.arch.WARP_SIZE
            threads_per_row = self._threads_per_row()
            warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
            cluster_n = self._cluster_n()
            self._kernel_impl(
                mX,
                mW,
                mB,
                mRes,
                mO,
                mResO,
                mRstd,
                eps,
                tv_layout,
                tiler_mn,
                const_expr(cluster_n),
                const_expr(num_warps),
                const_expr(warps_per_row),
                const_expr(threads_per_row),
            )

    @cute.jit
    def _init_cluster(self, tidx: cutlass.Int32, mbar_ptr: Optional[cute.Pointer]):
        if const_expr(mbar_ptr is not None):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()


def _can_use_ptr_path(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
) -> bool:
    """Fast path precondition for the pointer-based CuTeDSL entry.

    We require a row-major 2D layout where the last dimension is
    contiguous (stride(1) == 1). The leading dimension (stride(0))
    may be larger than N (padded-row / packed-attention layouts),
    and is passed to the kernel as `ld`.
    """
    if x.stride(1) != 1:
        return False
    # All participating tensors are interpreted as the same element type
    # (derived from x.dtype) in the pointer-based path. If dtypes differ,
    # we'd read the wrong bit patterns and silently produce incorrect output.
    if residual is not None and residual.dtype != x.dtype:
        return False
    if weight is not None and weight.dtype != x.dtype:
        return False
    if bias is not None and bias.dtype != x.dtype:
        return False
    # The kernel assumes `ld` satisfies a divisibility constraint used by
    # cute.assume(..., divby=...) for vectorization.
    elem_bits = TORCH2CUTE_DTYPE[x.dtype].width
    divby = 256 // elem_bits
    if (x.stride(0) % divby) != 0:
        return False
    # The kernel uses 128-bit vectorized copies (16B). Require at least 16B
    # alignment on all participating tensors to avoid misaligned global loads.
    if (x.data_ptr() % 16) != 0:
        return False
    if residual is not None and residual.stride(1) != 1:
        return False
    if residual is not None and residual.stride(0) != x.stride(0):
        return False
    if residual is not None and (residual.data_ptr() % 16) != 0:
        return False
    if weight is not None and not weight.is_contiguous():
        return False
    if bias is not None and not bias.is_contiguous():
        return False
    if weight is not None and (weight.data_ptr() % 16) != 0:
        return False
    if bias is not None and (bias.data_ptr() % 16) != 0:
        return False
    return True


def _can_use_ptr_path_fused_add_inplace(
    x: Tensor,
    weight: Tensor,
    residual: Tensor,
) -> bool:
    """Fast-path precondition for fused_add_rmsnorm_forward_inplace.

    We allow the common vLLM layout where:
    - `x` is strided/padded row-major (stride(1) == 1, stride(0) >= N)
    - `residual` is contiguous row-major (stride(0) == N)
    """
    if x.stride(1) != 1:
        return False
    if residual.dtype != x.dtype:
        return False
    if weight.dtype != x.dtype:
        return False
    if residual.stride(1) != 1:
        return False
    if not residual.is_contiguous():
        return False
    if not weight.is_contiguous():
        return False

    dtype = TORCH2CUTE_DTYPE[x.dtype]
    divby = 256 // dtype.width
    if (x.stride(0) % divby) != 0:
        return False
    if (residual.stride(0) % divby) != 0:
        return False

    if (x.data_ptr() % 16) != 0:
        return False
    if (residual.data_ptr() % 16) != 0:
        return False
    if (weight.data_ptr() % 16) != 0:
        return False
    return True


def _rmsnorm_forward_ptr(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
    eps: float,
    store_rstd: bool,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Pointer-based RMSNorm forward that bypasses DLPack entirely.

    This path reconstructs cute.Tensor views from raw device pointers
    and explicit layouts inside the JIT graph, avoiding any runtime
    DLPack conversions while reusing the tuned RMSNormSM100 schedule.
    """
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    M, N = x.shape

    # Preserve the input's 2D stride so downstream users that rely on
    # padded-row layouts (stride0 > N) continue to see the expected layout.
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    residual_out: Optional[Tensor] = None
    rstd: Optional[Tensor] = None

    if residual is not None:
        residual_out = torch.empty_strided(
            residual.shape,
            residual.stride(),
            device=residual.device,
            dtype=residual.dtype,
        )
    if store_rstd:
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    _rmsnorm_forward_ptr_into(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        out=out,
        residual_out=residual_out,
        rstd=rstd,
        eps=eps,
    )
    return out, rstd, residual_out


def _rmsnorm_forward_ptr_into(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
    out: Tensor,
    residual_out: Optional[Tensor],
    rstd: Optional[Tensor],
    eps: float,
) -> None:
    """Internal helper that launches the pointer-based kernel into preallocated outputs.

    This enables integration into frameworks like vLLM that manage their
    own buffers and prefer in-place or out-parameter semantics.
    """
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    M, N = x.shape
    device_index = x.get_device()
    dtype = TORCH2CUTE_DTYPE[x.dtype]

    if bias is None and residual is None and residual_out is None and rstd is None:
        # Fast-launch path: cache packed args and update pointers/scalars in-place to
        # avoid Python-side argument marshalling overhead that dominates small-batch cases.
        #
        # If fast-launch is disabled (or CuTeDSL internals changed), we fall back
        # to calling the compiled function directly.
        if torch.cuda.current_device() != device_index:
            torch.cuda.set_device(device_index)
        stream_handle = int(torch.cuda.current_stream().cuda_stream)
        has_weight = weight is not None

        stage = 1
        compiled_key = (
            "ptr",
            N,
            dtype,
            False,  # residual
            has_weight,
            False,  # bias
            False,  # residual_out
            False,  # rstd
            stage,
            device_index,
        )
        compiled = _PTR_COMPILE_CACHE.get(compiled_key)
        if compiled is None:
            op = RMSNormSM100(N, dtype, stage=stage)
            ld_val = int(x.stride(0))
            ptr_x = rt.make_ptr(
                dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
            )
            ptr_out = rt.make_ptr(
                dtype, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
            )
            ptr_w = (
                rt.make_ptr(
                    dtype,
                    weight.data_ptr(),
                    mem_space=rt.AddressSpace.gmem,
                    assumed_align=16,
                )
                if has_weight
                else None
            )
            stream = cuda.CUstream(stream_handle)
            ld = Int32(ld_val)
            compiled = cute.compile(
                op.launch_from_ptrs,
                ptr_x,
                ptr_w,
                None,  # ptr_b
                None,  # ptr_res
                ptr_out,
                None,  # ptr_res_out
                None,  # ptr_rstd
                Int32(M),
                Int32(N),
                ld,
                stream,
                Float32(eps),
            )
            _PTR_COMPILE_CACHE[compiled_key] = compiled

        launcher = _get_fast_ptr_rmsnorm_launcher(
            compiled=compiled,
            dtype=dtype,
            N=N,
            device_index=device_index,
            stream_handle=stream_handle,
            has_weight=has_weight,
            eps=eps,
        )
        ld_val = int(x.stride(0))
        if launcher is not None:
            launcher.launch(x=x, weight=weight, out=out, M=M, N=N, ld=ld_val, eps=eps)
            return

        ptr_x = rt.make_ptr(
            dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_w = (
            rt.make_ptr(
                dtype,
                weight.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if has_weight
            else None
        )
        stream = cuda.CUstream(stream_handle)
        ld = Int32(ld_val)
        compiled(
            ptr_x,
            ptr_w,
            None,  # ptr_b
            None,  # ptr_res
            ptr_out,
            None,  # ptr_res_out
            None,  # ptr_rstd
            Int32(M),
            Int32(N),
            ld,
            stream,
            Float32(eps),
        )
        return

    # Fallback: general path (supports bias/residual/rstd, but is slower to launch).
    stage = 1
    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    key = (
        "ptr",
        N,
        dtype,
        residual is not None,
        weight is not None,
        bias is not None,
        residual_out is not None,
        rstd is not None,
        stage,
        device_index,
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = RMSNormSM100(N, dtype, stage=stage)
        ptr_x = rt.make_ptr(
            dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_out = rt.make_ptr(
            dtype, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        ptr_res = (
            rt.make_ptr(
                dtype,
                residual.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if residual is not None
            else None
        )
        ptr_res_out = (
            rt.make_ptr(
                dtype,
                residual_out.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if residual_out is not None
            else None
        )
        ptr_w = (
            rt.make_ptr(
                dtype,
                weight.data_ptr(),
                mem_space=rt.AddressSpace.gmem,
                assumed_align=16,
            )
            if weight is not None
            else None
        )
        ptr_b = (
            rt.make_ptr(
                dtype, bias.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
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
        stream = cuda.CUstream(stream_handle)
        ld = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs,
            ptr_x,
            ptr_w,
            ptr_b,
            ptr_res,
            ptr_out,
            ptr_res_out,
            ptr_rstd,
            Int32(M),
            Int32(N),
            ld,
            stream,
            Float32(eps),
        )
        _PTR_COMPILE_CACHE[key] = compiled
    ptr_x = rt.make_ptr(
        dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_out = rt.make_ptr(
        dtype, out.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
    )
    ptr_res = (
        rt.make_ptr(
            dtype, residual.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        if residual is not None
        else None
    )
    ptr_res_out = (
        rt.make_ptr(
            dtype,
            residual_out.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=16,
        )
        if residual_out is not None
        else None
    )
    ptr_w = (
        rt.make_ptr(
            dtype, weight.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
        )
        if weight is not None
        else None
    )
    ptr_b = (
        rt.make_ptr(
            dtype, bias.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=16
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
    stream = cuda.CUstream(stream_handle)
    ld = Int32(int(x.stride(0)))
    compiled(
        ptr_x,
        ptr_w,
        ptr_b,
        ptr_res,
        ptr_out,
        ptr_res_out,
        ptr_rstd,
        Int32(M),
        Int32(N),
        ld,
        stream,
        Float32(eps),
    )


def _fused_add_rmsnorm_forward_ptr_inplace(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> None:
    """Pointer-based fused_add_rmsnorm that updates `x` and `residual` in-place."""
    assert x.is_cuda
    assert x.dim() == 2
    assert residual.is_cuda
    assert residual.dim() == 2
    assert x.shape == residual.shape

    M, N = x.shape
    device_index = x.get_device()
    dtype = TORCH2CUTE_DTYPE[x.dtype]
    stage = 1

    if torch.cuda.current_device() != device_index:
        torch.cuda.set_device(device_index)
    stream_handle = int(torch.cuda.current_stream().cuda_stream)

    # Latency-optimized schedule for small-M cases: avoid the gmem->smem
    # staging path (large dynamic smem + extra barriers) and load directly
    # from gmem into registers.
    copy_bits = 128
    # Use a direct-GMEM schedule (no staging SMEM tiles) for DSv3 hidden size
    # (7168, bf16/fp16). This improves both:
    # - small-M latency (fewer barriers + less dynamic shared memory), and
    # - large-M bandwidth (lower overhead, better vectorization when 32B-aligned).
    #
    # This is a policy decision: it is tuned for DSv3's N=7168. If you want to
    # benchmark other models/shapes, you can override it with:
    #   - OINK_RMSNORM_DIRECT_GMEM=0  (force staging/cp.async path)
    #   - OINK_RMSNORM_DIRECT_GMEM=1  (force direct-gmem path)
    direct_gmem = _direct_gmem_from_policy(
        default=bool(dtype.width == 16 and N == 7168)
    )
    use_async = not direct_gmem
    tpr_override: Optional[int] = None
    nt_override: Optional[int] = None
    cluster_n_override: Optional[int] = None
    direct_gmem_max_copy_bits: Optional[int] = None

    # Experimental stage-2 cp.async path (2-tile ping-pong) for N=7168. This is
    # primarily about improving memory-latency hiding / reducing long-scoreboard
    # stalls for large-M workloads.
    if _ENABLE_STAGE2 and dtype.width == 16 and N == 7168 and M >= 4096:
        stage = 2
        direct_gmem = False
        use_async = True

    # Experimental ILP variant (clusters): split each row across 2 CTAs.
    #
    # NOTE: This is currently opt-in because some CuTeDSL builds exhibit
    # instability with cluster launches for this specific schedule. To reduce
    # the chance of accidental crashes, we require an additional explicit
    # opt-in via `OINK_RMSNORM_ENABLE_CLUSTER_ILP_UNSAFE=1`.
    if _ENABLE_CLUSTER_ILP and not _ENABLE_STAGE2:
        if dtype.width == 16 and N == 7168 and M >= 4096:
            cluster_n_override = 2
            if direct_gmem:
                # Cluster launches + direct-GMEM has exhibited reproducible compiler
                # instability (segfaults) in some CuTeDSL builds, especially for the
                # 256b vector path. Probe it out-of-process once so we can safely
                # select a working copy width (or fall back to the staged SMEM path)
                # instead of crashing the parent process.
                max_bits = _probe_cluster_direct_gmem_max_copy_bits()
                if max_bits == 0:
                    direct_gmem = False
                    use_async = True
                else:
                    direct_gmem_max_copy_bits = max_bits

    # Experimental per-row partitioning: use 256 threads/row for N=7168 to
    # increase concurrency/ILP (accepts a small tail-predicate region).
    if _ENABLE_TPR256 and cluster_n_override is None and not _ENABLE_STAGE2:
        if dtype.width == 16 and N == 7168 and M >= 4096:
            tpr_override = 256
            nt_override = 256

    can_use_256 = bool(
        direct_gmem
        and (direct_gmem_max_copy_bits is None or direct_gmem_max_copy_bits >= 256)
        and dtype.width == 16
        and (x.data_ptr() % 32) == 0
        and (residual.data_ptr() % 32) == 0
        and (weight.data_ptr() % 32) == 0
    )
    assumed_align = 32 if can_use_256 else 16
    if can_use_256:
        copy_bits = 256

    copy_bits = _copy_bits_from_policy(default=copy_bits, can_use_256=can_use_256)
    if copy_bits == 128:
        assumed_align = 16
    elif copy_bits == 256 and can_use_256:
        assumed_align = 32
    else:
        copy_bits = 128
        assumed_align = 16

    key = (
        "ptr_fused_add_inplace",
        N,
        dtype,
        stage,
        device_index,
        copy_bits,
        use_async,
        tpr_override,
        nt_override,
        direct_gmem,
        cluster_n_override,
    )
    compiled = _PTR_COMPILE_CACHE.get(key)
    if compiled is None:
        op = RMSNormSM100(
            N,
            dtype,
            stage=stage,
            copy_bits=copy_bits,
            use_async=use_async,
            direct_gmem=direct_gmem,
        )
        if tpr_override is not None:
            op._tpr_override = tpr_override  # type: ignore[attr-defined]
        if nt_override is not None:
            op._nt_override = nt_override  # type: ignore[attr-defined]
        if cluster_n_override is not None:
            op._cluster_n_override = cluster_n_override  # type: ignore[attr-defined]
        ptr_x = rt.make_ptr(
            dtype,
            x.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        ptr_res = rt.make_ptr(
            dtype,
            residual.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        ptr_w = rt.make_ptr(
            dtype,
            weight.data_ptr(),
            mem_space=rt.AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        stream = cuda.CUstream(stream_handle)
        ld_x = Int32(int(x.stride(0)))
        compiled = cute.compile(
            op.launch_from_ptrs_fused_add_inplace,
            ptr_x,
            ptr_w,
            ptr_res,
            Int32(M),
            Int32(N),
            ld_x,
            stream,
            Float32(eps),
        )
        _PTR_COMPILE_CACHE[key] = compiled
    launcher = _get_fast_ptr_fused_add_rmsnorm_launcher(
        compiled=compiled,
        dtype=dtype,
        N=N,
        device_index=device_index,
        stream_handle=stream_handle,
        copy_bits=copy_bits,
        use_async=use_async,
        tpr=tpr_override or 0,
        direct_gmem=direct_gmem,
        assumed_align=assumed_align,
        eps=eps,
    )
    if launcher is not None:
        launcher.launch(
            x=x,
            weight=weight,
            residual=residual,
            M=M,
            N=N,
            ld_x=int(x.stride(0)),
            eps=eps,
        )
        return

    # Fast-launch is disabled/unavailable (or CuTeDSL internals changed). Fall back
    # to calling the compiled function directly.
    ptr_x = rt.make_ptr(
        dtype, x.data_ptr(), mem_space=rt.AddressSpace.gmem, assumed_align=assumed_align
    )
    ptr_res = rt.make_ptr(
        dtype,
        residual.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align,
    )
    ptr_w = rt.make_ptr(
        dtype,
        weight.data_ptr(),
        mem_space=rt.AddressSpace.gmem,
        assumed_align=assumed_align,
    )
    stream = cuda.CUstream(stream_handle)
    ld_x = Int32(int(x.stride(0)))
    compiled(ptr_x, ptr_w, ptr_res, Int32(M), Int32(N), ld_x, stream, Float32(eps))


# -------------------------
# Public API (forward + verify)
# -------------------------


def rmsnorm_forward(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    assert x.is_cuda
    assert x.dim() == 2, "Use (M, N) tensor; flatten batch/seq beforehand."
    M, N = x.shape

    # Fast path: use the pointer-based entry whenever we can represent the
    # inputs as a row-major [M, N] view with stride(1) == 1 and dtype contracts
    # are satisfied (vLLM uses this in inference).
    #
    # When the pointer path can't be used (e.g. float32 weights for Quack-style
    # APIs, or non-standard layouts), fall back to the CuTeDSL stage-2 module
    # before using the slow torch reference implementation.
    force_stage2 = _FORCE_RMSNORM_STAGE2_FWD

    if not force_stage2 and _can_use_ptr_path(x, weight, bias, residual):
        return _rmsnorm_forward_ptr(x, weight, bias, residual, eps, store_rstd)

    # CuTeDSL fallback for cases that aren't safe for the pointer path.
    # Import lazily to keep vLLM plugin startup and common inference fast paths
    # lightweight.
    try:
        import importlib

        rms2 = importlib.import_module(
            ".rmsnorm_with_stage2",
            package=__package__ or "kernelagent_oink.blackwell",
        )
    except Exception:
        rms2 = None  # type: ignore[assignment]
    if rms2 is not None:
        y, rstd, residual_out = rms2.rmsnorm_forward_with_stage2(
            x,
            weight=weight,
            bias=bias,
            residual=residual,
            eps=eps,
            store_rstd=store_rstd,
        )
        # Preserve stride contracts for torch.compile consistency, even
        # when using the optional stage-2 implementation.
        if y.stride() != x.stride():
            y_strided = torch.empty_strided(
                x.shape, x.stride(), device=x.device, dtype=x.dtype
            )
            y_strided.copy_(y)
            y = y_strided
        if residual is not None and residual_out is not None:
            if residual_out.stride() != residual.stride():
                residual_out_strided = torch.empty_strided(
                    residual.shape,
                    residual.stride(),
                    device=residual.device,
                    dtype=residual.dtype,
                )
                residual_out_strided.copy_(residual_out)
                residual_out = residual_out_strided
        return y, rstd, residual_out

    # Safe fallback (correctness-first). This is expected to be rare in vLLM.
    y = rmsnorm_ref(x, weight, bias, residual, eps)
    # Preserve the input stride contract even on the fallback path so
    # torch.compile sees a consistent output layout across all branches.
    if y.stride() != x.stride():
        y_strided = torch.empty_strided(
            x.shape, x.stride(), device=x.device, dtype=x.dtype
        )
        y_strided.copy_(y)
        y = y_strided
    rstd = None
    if store_rstd:
        xf = x.float()
        if residual is not None:
            xf = xf + residual.float()
        rstd = torch.rsqrt(xf.square().mean(dim=-1) + eps).to(torch.float32)
    residual_out = None
    if residual is not None:
        residual_out = (x.float() + residual.float()).to(x.dtype)
        if residual_out.stride() != residual.stride():
            residual_out_strided = torch.empty_strided(
                residual.shape,
                residual.stride(),
                device=residual.device,
                dtype=residual.dtype,
            )
            residual_out_strided.copy_(residual_out)
            residual_out = residual_out_strided
    return y, rstd, residual_out


def rmsnorm_ref(
    x: Tensor,
    w: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if w is not None:
        y = y * w.float()
    if b is not None:
        y = y + b.float()
    return y.to(x.dtype)


def fused_add_rmsnorm_forward(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Fused residual-add + RMSNorm for SM100 in CuteDSL.

    This is a convenience wrapper around ``rmsnorm_forward`` that matches the
    semantics of vLLM's ``fused_add_rms_norm``:

        z = x + residual
        y = RMSNorm(z, weight, eps)

    It returns ``(y, z)`` where ``z`` has the same dtype/shape as the inputs.
    """
    assert x.is_cuda and residual.is_cuda
    assert x.shape == residual.shape
    assert x.dtype == residual.dtype

    orig_shape = x.shape
    N = orig_shape[-1]

    x_2d = x.view(-1, N)
    res_2d = residual.view(-1, N)

    y_2d, _rstd, z_2d = rmsnorm_forward(
        x_2d,
        weight=weight,
        bias=None,
        residual=res_2d,
        eps=eps,
        store_rstd=False,
    )

    y = y_2d.view(orig_shape)
    z = z_2d.view(orig_shape)
    return y, z


def fused_add_rmsnorm_forward_inplace(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """In-place fused residual-add + RMSNorm matching vLLM semantics.

    This variant writes:

        z = x + residual     (stored into ``residual``)
        y = RMSNorm(z, w)    (stored into ``x``)

    i.e., it uses ``x`` as the normalized output buffer and ``residual`` as
    the residual-out buffer, mirroring vLLM's fused_add_rms_norm kernel.
    """
    fused_add_rmsnorm_inplace_(x, residual, weight, eps=eps)
    return x, residual


def fused_add_rmsnorm_inplace_(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> None:
    """In-place fused residual-add + RMSNorm matching vLLM semantics.

    This is the lowest-overhead Python entrypoint (returns `None`) intended
    for performance-critical call sites like `torch.ops.oink.fused_add_rms_norm`.
    """
    assert x.is_cuda and residual.is_cuda
    assert x.shape == residual.shape
    assert x.dtype == residual.dtype

    N = x.shape[-1]
    x_2d = x if x.dim() == 2 else x.view(-1, N)
    res_2d = residual if residual.dim() == 2 else residual.view(-1, N)

    # Fast path: vLLM-compatible layout where x may be strided/padded but
    # residual is contiguous. This updates both tensors in-place without
    # additional allocations.
    if _can_use_ptr_path_fused_add_inplace(x_2d, weight, res_2d):
        _fused_add_rmsnorm_forward_ptr_inplace(x_2d, res_2d, weight, eps)
        return None

    # Fallback: allocate via the regular fused path, then copy results into
    # the user-provided buffers so that semantics remain identical.
    y, z = fused_add_rmsnorm_forward(x, residual, weight, eps)
    x.copy_(y)
    residual.copy_(z)
    return None


# -------------------------
# Backward kernel (SM100)
# -------------------------


class RMSNormBackwardSM100(BaseRMSNormBackward):
    """SM100-tuned RMSNorm backward.

    This is a thin wrapper around the generic `lite_quack.RMSNormBackward`
    base implementation, with SM100-friendly tiling heuristics that mirror
    the forward policy used by Oink.
    """

    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N)

    def _get_num_threads(self) -> int:
        # Keep 128 threads only up to N=4k; use 256 for larger rows to ensure
        # threads_per_row <= num_threads across buckets.
        try:
            return self._nt_override  # type: ignore[attr-defined]
        except Exception:
            return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self) -> int:
        # Mirror RMSNormSM100 forward's tiling.
        N = self.N
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 1024:
            return 32
        if N <= 4096:
            return 128
        if N <= 8192:
            try:
                return self._tpr_override  # type: ignore[attr-defined]
            except Exception:
                return 128
        if N <= 16384:
            return 256
        return 256

    def _set_cluster_n(self) -> None:
        # Reuse the SM100 forward cluster growth policy so large-N shapes can
        # fan out across multiple CTAs in the same row.
        try:
            self.cluster_n = self._cluster_n_override  # type: ignore[attr-defined]
            return
        except Exception:
            pass

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
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        # Match forward's 32B alignment on the leading dimension to unlock
        # wider vectorization when legal.
        semistatic_shape = (*mX.shape[:-1], self.N)

        def new_stride(t):
            return (
                cute.assume(t.stride[0], divby=256 // t.element_type.width),
                t.stride[1],
            )

        mX, mdO, mdResO, mdX, mdRes = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            if const_expr(t is not None)
            else None
            for t in (mX, mdO, mdResO, mdX, mdRes)
        ]

        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mdO.element_type.width,
                mdX.element_type.width,
                mdResO.element_type.width if mdResO is not None else 0,
                mdRes.element_type.width if mdRes is not None else 0,
            )
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )
        num_threads = (
            cute.size(tv_layout, mode=[0])
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self._get_num_threads()
        )
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        kernel = (
            self.kernel(
                mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes, tv_layout, tiler_mn
            )
            if _KERNEL_ACCEPTS_LAYOUT_ARGS
            else self.kernel(mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes)
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


_BWD_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _rmsnorm_bwd_sm100(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Optional[Tensor],
    db_partial: Optional[Tensor] = None,
    dresidual_out: Optional[Tensor] = None,
    dresidual: Optional[Tensor] = None,
    sm_count: Optional[int] = None,
) -> None:
    """SM100-specific RMSNorm backward dispatch.

    Mirrors Quack's `quack.rmsnorm._rmsnorm_bwd`, but instantiates
    `RMSNormBackwardSM100` (SM100-tuned heuristics).
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if weight is not None:
        assert weight.dim() == 1
        assert x.shape[-1] == weight.shape[0]
        assert weight.is_cuda
        assert weight.dtype in (torch.float32, torch.bfloat16, torch.float16)
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.is_cuda
        assert dresidual_out.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.is_cuda
        assert dresidual.dtype in (torch.float16, torch.bfloat16, torch.float32)

    M, N = x.size(0), x.size(1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = (
            dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
        )

    # Match Quack's conversion strategy for activations/gradients: keep the
    # (M, N) layout dynamic without enforcing additional compact-shape
    # constraints. This reduces per-call Python overhead for small-M shapes.
    def _convert_mx(t: Tensor) -> cute.Tensor:
        return from_dlpack(
            t.detach(),
            assumed_align=16,
        ).mark_layout_dynamic(leading_dim=1)

    x_tensor, dout_tensor, dres_out_tensor, dx_tensor, dres_tensor = [
        _convert_mx(t) if t is not None else None
        for t in (x, dout, dresidual_out, dx, dresidual)
    ]

    if weight is not None:
        weight_dtype = TORCH2CUTE_DTYPE[weight.dtype]
        weight_tensor = convert_from_dlpack_cute(
            weight.detach(),
            leading_dim=0,
            divisibility=128 // weight_dtype.width,
        )
    else:
        weight_tensor = None

    dw_partial_tensor = (
        from_dlpack(dw_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if dw_partial is not None
        else None
    )
    db_partial_tensor = (
        from_dlpack(db_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if db_partial is not None
        else None
    )
    rstd_tensor = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        M,
        N,
        x_tensor.element_type,
        weight_tensor.element_type if weight is not None else None,
        db_partial.dtype if db_partial is not None else None,
        dresidual.dtype if dresidual is not None else None,
        dresidual_out.dtype if dresidual_out is not None else None,
    )
    kernel = _BWD_COMPILE_CACHE.get(compile_key)
    if kernel is None:
        op = RMSNormBackwardSM100(x_tensor.element_type, N)

        # Shape-specific tuning overrides for DSv3-style N=8192 rows.
        if isinstance(op, RMSNormBackwardSM100) and N == 8192:
            if M >= 65536:
                op._tpr_override = 256  # type: ignore[attr-defined]
                op._nt_override = 256  # type: ignore[attr-defined]
            elif M >= 16384:
                op._tpr_override = 256  # type: ignore[attr-defined]

        kernel = cute.compile(
            op,
            x_tensor,
            weight_tensor,
            dout_tensor,
            dres_out_tensor,
            rstd_tensor,
            dx_tensor,
            dw_partial_tensor,
            dres_tensor,
            db_partial_tensor,
            Int32(sm_count if sm_count is not None else 0),
            current_stream,
        )
        _BWD_COMPILE_CACHE[compile_key] = kernel

    kernel(
        x_tensor,
        weight_tensor,
        dout_tensor,
        dres_out_tensor,
        rstd_tensor,
        dx_tensor,
        dw_partial_tensor,
        dres_tensor,
        db_partial_tensor,
        Int32(sm_count if sm_count is not None else 0),
        current_stream,
    )


def rmsnorm_backward(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,
    has_bias: bool = False,
    has_residual: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Public SM100 RMSNorm backward entry point.

    Signature mirrors `quack.rmsnorm.rmsnorm_bwd` for easy comparisons.
    """
    device = x.device
    M, N = x.size(0), x.size(1)
    dx = torch.empty_like(x)
    if dresidual_out is not None and dresidual_out.dtype != dx.dtype:
        dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
    else:
        dresidual = None

    # Shared SM100 tuning policy (used by both RMSNorm and LayerNorm).
    sm_count = get_sm_count(N, device, M=M, dtype=x.dtype)

    # Quack-suite smallest case (M=8192, N=4096) is extremely sensitive to
    # Python/allocator overhead because the kernel itself is very fast.
    #
    # The default `lite_quack.get_sm_count` adds a small-M occupancy boost for
    # N=4096, which increases `dw_partial` size and can amplify allocator
    # pressure in benchmark/verify loops. Clamp to Quack's baseline policy
    # (`sm_count = num_sms * 2` for N=4096) for this regime.
    if N == 4096 and M <= 8192 and x.dtype in (torch.float16, torch.bfloat16):
        try:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
            sm_count = min(int(sm_count), int(num_sms) * 2)
        except Exception:
            pass

    if weight is not None:
        dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)
    else:
        dw_partial = None
    db_partial = (
        torch.empty(sm_count, N, device=device, dtype=torch.float32)
        if has_bias
        else None
    )

    _rmsnorm_bwd_sm100(
        x,
        weight,
        dout,
        rstd,
        dx,
        dw_partial,
        db_partial,
        dresidual_out,
        dresidual,
        sm_count,
    )

    dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
    db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
    if has_residual and dresidual is None:
        dresidual = dx
    return dx, dw, db, dresidual


# Quack-style alias for benchmarks
rmsnorm_bwd = rmsnorm_backward


if __name__ == "__main__":
    # Minimal ad-hoc test (functionality only). For performance comparisons, use the benchmark harness.
    if not torch.cuda.is_available():
        print("CUDA not available; functional test skipped.")
        sys.exit(0)
    M, N = 1024, 8192
    dtype = torch.bfloat16
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    y_ref = rmsnorm_ref(x, w)
    y, _, _ = rmsnorm_forward(x, w)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
    print("RMSNormSM100 correctness check passed.")

# (compile cache moved to top)
