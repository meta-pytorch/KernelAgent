<!--
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Tutorial: How to integrate NV TMA into kernels
## Background
TMA is a hardware unit introduced by the NV Hopper GPU. It takes over some of the data transfer work from softwares and thus improves the performance by freeing up warps or reducing register pressures etc. In practice, Triton kernel authors can update the kernel code by simply replacing `tl.load` and `tl.store` with TMA API calls to get this performance boost.

## TMA APIs
TMA API is going through changes (from experimental to official) on upstream Triton. While we’re working out a plan to migrate, we’ll support the “old” experimental API that’s currently being used in our fbsource codebase. This tutorial will be based on the experimental API.

TMA data load/store needs a TMA tensor descriptor object. The descriptor will describe the tensor address, strides, shapes etc. of the tensor to be copied (treat it as the CUDA `TensorMap` object). The descriptor itself needs to be stored somewhere. Depending on where we initialize the descriptor, we have two types of descriptors: on-host and on-device. The former allocates the memory on host memory, initializes descriptors there and then copies them by value to GMEM. The latter will allocate a big chunk of memory on GMEM, and then have each program to find their own offset and initialize descriptors there.

To leverage TMA, we need to decide between on-host and on-device descriptors. That decision could be yet another topic. Here we quickly highlight a few key differences:
- Only on-device descriptors can handle dynamic shapes where not all programs are handling the same box size, which is typical in kernels like Jagged Flash Attention or HSTU. The reason is that on-host descriptors are initialized before kernel launch while on-device ones are initialized in the kernel where the box size is known.
- Torch Inductor, especially AOTI, currently only supports on-device descriptors
- On-device descriptors are initialized by every kernel program in the grid while on-host ones are initialized by host code so likely on-device descriptors take more compute resources
- Current on-device descriptors implementation (experimental API) might take more global memory because the number of programs is not necessarily known when allocating memory chunk for descriptors (e.g. depending on auto tuned BLOCK_SIZE_M), so we need to be conservative and allocate more memory

Note: neither of these two types of TMA is necessarily faster than the other. It depends on actual use cases.

Now for the sake of this tutorial we’ll start with on-device descriptors. And also we’ll use the example of copying 2d tensors as it’s the most common.

With those premises, here’re the APIs to call:

- Allocate memory chunk to store descriptors on host:
```
TMA_DESC_SIZE = 128 # size in bytes used by a single descriptor, tunable
NUM_DESC_PER_PROGRAM = ... # how many different tensors to load/store by each program. e.g. 3 for GEMM `C=AB`, 4 for HSTU Q,K,V,O tensors
NUM_OF_PROGRAMS = ... # same as specified in kernel `grid`. If grid size is related to auto tune config, use a reasonable upper bound by hard coding "minimal block M size" etc. for now.
workspace = torch.empty(
           TMA_DESC_SIZE * NUM_DESC_PER_PROGRAM * NUM_OF_PROGRAMS,
           dtype=torch.uint8,
           device="cuda",)
# then pass `workspace` to kernel
```
- Initialize descriptor object:
```
desc_ptr = workspace + TMA_DESC_SIZE * <program id offset> + TMA_DESC_SIZE * <in program offset> # in program offset in range [0,NUM_DESC_PER_PROGRAM)


tl.extra.cuda.experimental_device_tensormap_create2d(
desc_ptr=desc_ptr,
global_address=<tensor_ptr>, # tensor to load into or store from
load_size=[BOX_SIZE_0, BOX_SIZE_1], # size of the 2D box to copy
global_size=[GLOBAL_SIZE_0, GLOBAL_SIZE_1], # this defines a "global box" in GMEM. TMA load/store won't go over this boundary if load_size is not divisble by global_size. e.g. Assuming GLOBAL_SIZE_0 == 1.5 * BLOCK_SIZE_0 and GLOBAL_SIZE_1 == BLOCK_SIZE_1, then: for TMA load, the second box will return a tensor of size (BLOCK_SIZE_0, BLOCK_SIZE_1) but the second half of the tensor is all 0; for TMA store, the second box will only have its first half written to GMEM.
element_ty=<element_ty> # usually tensor_ptr.dtype.element_ty
)
```
- Acquire fence on a TensorMap/descriptor object:
```
tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(<desc_ptr>)
```
- Load data from GMEM to SMEM:
```
x = tl._experimental_descriptor_load(
                   <in_desc_ptr>, #initialized, and acquired fence above
                   [OFFSET_0, OFFSET_1], # offset in "global box" for the 2D loading box to start from
                   [BOX_SIZE_0, BOX_SIZE_1], # keep the same as descriptor's `load_size`
                   <dtype>,)
```
- Store data from SMEM to GMEM:
```
tl._experimental_descriptor_store(
                   <out_desc_ptr>, #initialized, and acquired fence above
                   <output_tensor>, #the tensor to be stored on GMEM
                   [OFFSET_0, OFFSET_1], # offset in "global box" for the 2D loading box to start from
)
```

## Example
### Store
Let’s assume we have the following non TMA store code now:

```
start_m = pid * BLOCK_M
offs_m = start_m + tl.arange(0, BLOCK_M)
offs_v_d = tl.arange(0, BLOCK_D_V)
off_o = Out + seq_start * stride_om + off_h * stride_oh # TMA will use Out as global address, and include seq_start * stride_om + off_h * stride_oh as part of offsets
out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])

# Essentially, it tries to store the tensor `acc` into this box:
#  Out[
# 	 (seq_start + pid * BLOCK_M : seq_start + (pid+1) * BLOCK_M),
#	 (off_h * stride_oh : off_h * stride_oh + BLOCK_D_V)
#   ]
#  In other words, it's a box of size (BLOCK_M, BLOCK_D_V) starting at [seq_start + pid * BLOCK_M, off_h * stride_oh]. This will be the bases for our TMA desc init and load/store op.
#  And the rows with dim0 larger than (seq_start + seq_len) will be masked. Note that (seq_start + seq_len) == seq_end, which we'll use in TMA store below
```
The equivalent TMA store code would be:
```
# pyre-ignore [20]
tl.extra.cuda.experimental_device_tensormap_create2d(
     desc_ptr=device_desc_o,
     global_address=Out, # Out is of shape (L, H, DimV)
     load_size=[BLOCK_M, BLOCK_D_V], #box size as explained in comments above
     global_size=[seq_end.to(tl.int32), H * DimV], # this eliminates the need for `mask`, TMA automatically take care of boundaries.
     element_ty=Out.dtype.element_ty,
)
# pyre-ignore [20]
tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_o)
tl._experimental_descriptor_store(
      device_desc_o,
      acc, # acc needs to be casted to the right dtype
      [ #offset as explained in comments above (where the box starts at)
        (seq_start + pid * BLOCK_M).to(tl.int32),
        (off_h * stride_oh).to(tl.int32),
       ],
  )
```
### Load
Assume we have this non TMA load code:
```
Q_block_ptr = tl.make_block_ptr(
                 base=Q + off_h * stride_qh + seq_start * stride_qm,
                 shape=(seq_len, BLOCK_D_Q),
                 strides=(stride_qm, 1),
                 offsets=(start_m, 0),
                 block_shape=(BLOCK_M, BLOCK_D_Q),
                 order=(1, 0),
               )
q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")


# Essentially this tries to load this box into q:
#   Q[
#	(seq_start + start_m : seq_start + start_m + BLOCK_M),
#	(off_h * stride_qh : off_h * stride_qh + BLOCK_D_Q)
#    ]
#  In other words, it's a box of size (BLOCK_M, BLOCK_D_Q) starting at [seq_start + start_m, off_h * stride_qh]. This will be the bases for our TMA desc init and load/store op.
# And the rows with dim0 larger than seq_len will be filled with zero, with shape of q always being (BLOCK_M, BLOCK_D_Q).
```
The equivalent TMA load code will be:
```
# pyre-ignore [20]
tl.extra.cuda.experimental_device_tensormap_create2d(
     desc_ptr=device_desc_q,
     global_address=Q, # shape (L, H, DimQ)
     load_size=[BLOCK_M,BLOCK_D_Q], #box size as explained in comments above
     global_size=[seq_end.to(tl.int32), H * DimQ], # seq_end == seq_start + seq_len
     element_ty=Q.dtype.element_ty,
  )
# pyre-ignore [20]
               tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(device_desc_q)


q = tl._experimental_descriptor_load(
          device_desc_q,
          [ #offset as explained in comments above (where the box starts at)
            (seq_start + start_m).to(tl.int32),
            (off_h * stride_qh).to(tl.int32),
          ],
          [BLOCK_M,BLOCK_D_Q],
          Q.dtype.element_ty,
      )
```
