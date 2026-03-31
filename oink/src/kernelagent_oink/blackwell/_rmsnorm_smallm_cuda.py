from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from pathlib import Path

import torch
from torch import Tensor

_CUDA_HOME = "/usr/local/cuda-13.0"
_NVCC = f"{_CUDA_HOME}/bin/nvcc"
_CACHE_DIR = Path(os.environ.get("CUTE_DSL_CACHE_DIR", "/tmp")) / "oink_smallm_cuda"
_BUILD_LOCK = threading.Lock()
_LAUNCHER = None
_BUILD_FAILED = False

_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>

template <int NumThreads>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = NumThreads / 2; offset >= 1; offset /= 2) {
    x += __shfl_down_sync(0xffffffff, x, offset, NumThreads);
  }
  return __shfl_sync(0xffffffff, x, 0, NumThreads);
}

constexpr int kHidden = 4096;
constexpr int kBf16PerFloat4 = sizeof(float4) / sizeof(__nv_bfloat16);
constexpr int kFloat4sPerThread = 4;
constexpr int kRowThreads = 128;
constexpr int kElemsPerThread = kFloat4sPerThread * kBf16PerFloat4;
constexpr int kWarpsPerRow = kRowThreads / 32;
constexpr int kFloat4sPerRow = kHidden / kBf16PerFloat4;
constexpr float kInvHidden = 1.0f / float(kHidden);

__global__ __launch_bounds__(kRowThreads) void rmsnorm_noweight_4096_bf16_kernel(
    int M,
    float eps,
    const float4* __restrict__ input,
    float4* __restrict__ output) {
  const int row = blockIdx.x;
  if (row >= M) {
    return;
  }

  __shared__ float partial[kWarpsPerRow];
  __shared__ float denom;

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int row_vec_base = row * kFloat4sPerRow;

  float vals[kElemsPerThread];

#pragma unroll
  for (int i = 0; i < kFloat4sPerThread; ++i) {
    const int vec_idx = row_vec_base + i * kRowThreads + threadIdx.x;
    float4 v = input[vec_idx];
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < kBf16PerFloat4; ++j) {
      vals[i * kBf16PerFloat4 + j] = __bfloat162float(p[j]);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    sum += vals[i] * vals[i];
  }

  const float warp_sum = warp_reduce_sum<32>(sum);
  if (lane == 0) {
    partial[warp_id] = warp_sum;
  }
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    total = lane < kWarpsPerRow ? partial[lane] : 0.0f;
    total = warp_reduce_sum<32>(total);
    if (lane == 0) {
      denom = rsqrtf(eps + total * kInvHidden);
    }
  }
  __syncthreads();

  const float scale = denom;

#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    vals[i] *= scale;
  }

#pragma unroll
  for (int i = 0; i < kFloat4sPerThread; ++i) {
    float4 outv;
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&outv);
#pragma unroll
    for (int j = 0; j < kBf16PerFloat4; ++j) {
      p[j] = __float2bfloat16(vals[i * kBf16PerFloat4 + j]);
    }
    const int vec_idx = row_vec_base + i * kRowThreads + threadIdx.x;
    output[vec_idx] = outv;
  }
}

extern "C" void launch_rmsnorm_noweight_4096_bf16(
    int M,
    float eps,
    const void* input,
    void* output,
    void* stream) {
  rmsnorm_noweight_4096_bf16_kernel<<<M, kRowThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      M,
      eps,
      reinterpret_cast<const float4*>(input),
      reinterpret_cast<float4*>(output));
}
"""


def _smallm_bounds() -> tuple[int, int]:
    min_m = int(os.environ.get("OINK_RMSNORM_SMALLM_MIN_M", "4096"))
    max_m = int(os.environ.get("OINK_RMSNORM_SMALLM_MAX_M", "4096"))
    return min_m, max_m


def _disabled() -> bool:
    val = os.environ.get("OINK_RMSNORM_DISABLE_SMALLM_CUDA", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _artifact_paths() -> tuple[Path, Path]:
    digest = hashlib.sha1(_CUDA_SRC.encode("utf-8")).hexdigest()[:12]
    build_dir = _CACHE_DIR / digest
    cu_path = build_dir / "rmsnorm_smallm.cu"
    so_path = build_dir / "librmsnorm_smallm.so"
    return cu_path, so_path


def _build_launcher():
    global _LAUNCHER, _BUILD_FAILED
    if _disabled() or _BUILD_FAILED:
        return None
    if _LAUNCHER is not None:
        return _LAUNCHER
    with _BUILD_LOCK:
        if _LAUNCHER is not None:
            return _LAUNCHER
        if _BUILD_FAILED:
            return None
        cu_path, so_path = _artifact_paths()
        try:
            cu_path.parent.mkdir(parents=True, exist_ok=True)
            if not so_path.exists():
                cu_path.write_text(_CUDA_SRC)
                env = os.environ.copy()
                env["PATH"] = f"{_CUDA_HOME}/bin:{env.get('PATH', '')}"
                cmd = [
                    _NVCC,
                    "-arch=sm_100",
                    "-O3",
                    "--shared",
                    "-Xcompiler=-fPIC",
                    str(cu_path),
                    "-o",
                    str(so_path),
                ]
                subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            lib = ctypes.CDLL(str(so_path))
            fn = lib.launch_rmsnorm_noweight_4096_bf16
            fn.argtypes = [
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = None
            _LAUNCHER = fn
        except Exception:
            _BUILD_FAILED = True
            return None
    return _LAUNCHER


def can_use_smallm_noweight_cuda(x: Tensor, out: Tensor) -> bool:
    if x.dtype is not torch.bfloat16 or out.dtype is not torch.bfloat16:
        return False
    if not x.is_cuda or not out.is_cuda:
        return False
    if x.dim() != 2 or out.dim() != 2 or x.shape != out.shape:
        return False
    if int(x.shape[1]) != 4096:
        return False
    if x.stride() != (4096, 1) or out.stride() != (4096, 1):
        return False
    min_m, max_m = _smallm_bounds()
    return min_m <= int(x.shape[0]) <= max_m


def try_rmsnorm_smallm_noweight_cuda(x: Tensor, out: Tensor, eps: float) -> bool:
    if not can_use_smallm_noweight_cuda(x, out):
        return False
    launcher = _build_launcher()
    if launcher is None:
        return False
    stream = torch.cuda.current_stream(x.device).cuda_stream
    launcher(
        int(x.shape[0]),
        float(eps),
        ctypes.c_void_p(int(x.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_void_p(int(stream)),
    )
    return True
