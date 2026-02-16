# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ON_DEVICE_TMA = """
============================= On-Device Tensor Memory Accelerator (TMA) ===================================
## What is TMA?
The Tensor Memory Accelerator (TMA) is a hardware feature introduced in NVIDIA Hopper GPUs
for performing asynchronous memory copies between a GPU's global memory (GMEM) and the
shared memory (SMEM) of its thread blocks (i.e., CTAs). TMA offloads some of the data
transfer work from software, thereby improving performance by overlapping memory transfers
with computation, freeing up warps, and reducing register pressure.

## On-Device TMA:
TMA data load/store operations require a TMA tensor descriptor object. This descriptor
specifies the tensor's address, strides, shapes, and other attributes necessary for the
copy operation. TMA descriptors can be initialized on the device. On-device descriptors
allocate a large chunk of memory in GMEM, and each program have to find its own offset
and initialize descriptors there.

## How to integrate on-device TMA into a Triton program?
To enable on-device TMA in a Triton program, we need to add support from both the host and kernel programs.
In the host program, a global memory allocation is needed by adding the following function:
```
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)
```
In addition, we need to import the method `from typing import Optional`.
In the kernel program, instead of loading and storing a tensor block with a range of pointers,
we declare a TMA descriptor for each tensor and then use the descriptor to load and store the tensor in blocks.
An example of a TMA descriptor declaration is
```
x_desc = tl.make_tensor_descriptor(
    x_ptr,                                      # the pointer to the tensor
    shape=[M, N],                               # the shape of the tensor
    strides=[stride_m, stride_n],               # the stride of the tensor
    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],   # the block size of each TMA load/store
)
```
An example of the TMA load is
```
x = x_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N]) # the start offset of the TMA load
```
"""
