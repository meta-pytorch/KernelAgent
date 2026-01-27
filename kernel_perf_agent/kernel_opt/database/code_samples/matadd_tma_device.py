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

# ======== matadd with on-device Tensor Memory Accelerator (TMA) integration ==========
from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # device TMA
    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    x = x_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
    y = y_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
    output = x + y
    output_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], output)


def add(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    add_kernel[grid](
        x,
        y,
        output,
        M,
        N,
        x.stride(0),
        x.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output
