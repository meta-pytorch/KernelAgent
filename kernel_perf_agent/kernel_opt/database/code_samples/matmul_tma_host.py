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

# ======== matmul with on-host Tensor Memory Accelerator (TMA) integration ==========
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    _num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])  # TMA load of A
        b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])  # TMA load of B
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # TMA descriptors for loading A, B and storing C
    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    matmul_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c
