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

**Worker-level interfaces** (fine — swap individual components inside
the optimisation worker):
    AcceleratorSpecsProvider, KernelProfilerBase, RooflineAnalyzerBase,
    BottleneckAnalyzerBase, RAGPrescriberBase

No-op implementations (for testing / new-backend bootstrapping):
    NoOpVerifier, NoOpBenchmarker, NoOpWorkerRunner,
    NoOpAcceleratorSpecsProvider, NoOpKernelProfiler,
    NoOpRooflineAnalyzer, NoOpBottleneckAnalyzer, NoOpRAGPrescriber

NVIDIA / CUDA implementations (default when nothing else is supplied):
    NvidiaVerifier, NvidiaBenchmarker, NvidiaWorkerRunner,
    NvidiaAcceleratorSpecsProvider, NvidiaKernelProfiler,
    NvidiaRooflineAnalyzer, NvidiaBottleneckAnalyzer, NvidiaRAGPrescriber

Registry (for config-driven component selection):
    registry  — the global :class:`PlatformRegistry` singleton
"""

from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RAGPrescriberBase,
    RooflineAnalyzerBase,
    WorkerRunner,
)
from triton_kernel_agent.platform.registry import PlatformRegistry, registry
from triton_kernel_agent.platform.nvidia import (
    NvidiaAcceleratorSpecsProvider,
    NvidiaBenchmarker,
    NvidiaBottleneckAnalyzer,
    NvidiaKernelProfiler,
    NvidiaRAGPrescriber,
    NvidiaRooflineAnalyzer,
    NvidiaVerifier,
    NvidiaWorkerRunner,
)
from triton_kernel_agent.platform.noop import (
    NoOpAcceleratorSpecsProvider,
    NoOpBenchmarker,
    NoOpBottleneckAnalyzer,
    NoOpKernelProfiler,
    NoOpRAGPrescriber,
    NoOpRooflineAnalyzer,
    NoOpVerifier,
    NoOpWorkerRunner,
)

__all__ = [
    # Manager-level interfaces
    "KernelVerifier",
    "KernelBenchmarker",
    "WorkerRunner",
    # Worker-level interfaces
    "AcceleratorSpecsProvider",
    "KernelProfilerBase",
    "RooflineAnalyzerBase",
    "BottleneckAnalyzerBase",
    "RAGPrescriberBase",
    # No-op implementations (manager)
    "NoOpVerifier",
    "NoOpBenchmarker",
    "NoOpWorkerRunner",
    # NVIDIA implementations (manager)
    "NvidiaVerifier",
    "NvidiaBenchmarker",
    "NvidiaWorkerRunner",
    # NVIDIA implementations (worker)
    "NvidiaAcceleratorSpecsProvider",
    "NvidiaKernelProfiler",
    "NvidiaRooflineAnalyzer",
    "NvidiaBottleneckAnalyzer",
    "NvidiaRAGPrescriber",
    # No-op implementations (worker)
    "NoOpAcceleratorSpecsProvider",
    "NoOpKernelProfiler",
    "NoOpRooflineAnalyzer",
    "NoOpBottleneckAnalyzer",
    "NoOpRAGPrescriber",
    # Registry
    "PlatformRegistry",
    "registry",
]
