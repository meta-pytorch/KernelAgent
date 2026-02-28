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
BackendBench Integration for KernelAgent.

This package provides evaluation utilities for testing KernelAgent-generated
Triton kernels using the BackendBench infrastructure.

"""

from .eval import evaluate_kernels, generate_kernels

__all__ = [
    "generate_kernels",
    "evaluate_kernels",
]
