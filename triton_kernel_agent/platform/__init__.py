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

Interfaces (ABCs):
    KernelVerifier, KernelBenchmarker, WorkerRunner

No-op implementations (for testing / new-backend bootstrapping):
    NoOpVerifier, NoOpBenchmarker, NoOpWorkerRunner

NVIDIA / CUDA implementations (default when nothing else is supplied):
    NvidiaVerifier, NvidiaBenchmarker, NvidiaWorkerRunner
"""

from triton_kernel_agent.platform.interfaces import (
    KernelBenchmarker,
    KernelVerifier,
    WorkerRunner,
)
from triton_kernel_agent.platform.noop import (
    NoOpBenchmarker,
    NoOpVerifier,
    NoOpWorkerRunner,
)

__all__ = [
    # Interfaces
    "KernelVerifier",
    "KernelBenchmarker",
    "WorkerRunner",
    # No-op implementations
    "NoOpVerifier",
    "NoOpBenchmarker",
    "NoOpWorkerRunner",
]
