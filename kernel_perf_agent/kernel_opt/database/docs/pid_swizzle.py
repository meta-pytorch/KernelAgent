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

PID_SWIZZLE = """
===================================== PID Swizzling ===========================================
## What it is:
PID swizzling is a GPU optimization technique used in Triton programming that remaps
program identifiers (`pid_m` and `pid_n`) to create better memory access patterns,
specifically for L2 cache locality. This technique is commonly used in high-performance GPU kernels,
particularly for GEMM (General Matrix Multiply) operations in frameworks like Triton.

## Traditional Approach:
The program launch order matters as it affects the L2 cache hit rate.
In an unoptimized GPU kernel, each program instance computes a [BLOCK_SIZE_M, BLOCK_SIZE_N]
block of the output tensor, and the program identifiers are arranged in a simple row-major ordering
by `pid_m = pid // num_pid_n` and `pid_n = pid % num_pid_n`.
This creates poor cache locality because adjacent programs access memory locations that are far apart.

## PID Swizzling Approach:
PID swizzling forms "super-grouping" of programs with a fixed row size `GROUP_SIZE_M`.
The number of programs in a group is `GROUP_SIZE_M * num_pid_n`.
The `group_id` is calculated by dividing the program id by the number of programs in a group.
If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the row size of the last group is smaller
and can be calculated by subtracting `GROUP_SIZE_M * group_id` from `num_pid_m`.
The programs within a group are arranged in a column-major order.
"""
