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

"""Platform abstraction layer for the optimization manager.

**Manager-level interfaces** (coarse — replace an entire subsystem):
    KernelVerifier, KernelBenchmarker, WorkerRunner

NVIDIA / CUDA implementations (default when nothing else is supplied):
    NvidiaVerifier, NvidiaBenchmarker, NvidiaWorkerRunner
"""

from triton_kernel_agent.platform.interfaces import (
    KernelBenchmarker,
    KernelVerifier,
    WorkerRunner,
)
from triton_kernel_agent.platform.nvidia import (
    NvidiaBenchmarker,
    NvidiaVerifier,
    NvidiaWorkerRunner,
)

__all__ = [
    # Manager-level interfaces
    "KernelVerifier",
    "KernelBenchmarker",
    "WorkerRunner",
    # NVIDIA implementations
    "NvidiaVerifier",
    "NvidiaBenchmarker",
    "NvidiaWorkerRunner",
]
