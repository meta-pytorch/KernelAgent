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

PERSISTENCE = """
================================ Persistent Programming Style =======================================
## What it is:
The persistent programming style in GPU is a kernel design pattern where a fixed number of
blocks is launched, typically equal to the number of streaming multiprocessors (SMs),
instead of launching blocks proportional to the problem size. This pattern is particularly effective
for large-scale computations where the problem size exceeds the GPU's parallel capacity.

## Traditional Approach:
In an unoptimized Triton GPU kernel, the number of blocks launched is dependent on the input size,
typically calculated as `triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]`
in the grid argument.
Each block processes exactly one tile of work, and the number of blocks can be much larger
than the available hardware resources.

## Persistent Approach:
In a persistent style implementation, a fixed number of blocks is launched, which can be the number
of streaming multiprocessors (SMs) on the GPU by calling `torch.cuda.get_device_properties("cuda").multi_processor_count`.
In the kernel code, each block iterates over the program ID with a stride equal to the total number of blocks,
ensuring that the computation is completed by a fixed number of blocks.
These blocks "persist" and loop until all work is completed.

## Advantages:
* Better resource utilization: Matches hardware capabilities exactly
* Reduced launch overhead: Fewer kernel launches for large problems
* Improved occupancy: Keeps all SMs busy throughout execution
* Better cache locality: Blocks can reuse data across multiple iterations
* Load balancing: Work is distributed more evenly across SMs
"""
