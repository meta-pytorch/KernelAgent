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

from pathlib import Path

from kernel_perf_agent.kernel_opt.database.docs import (
    on_device_tma,
    on_host_tma,
    persistence,
    pid_swizzle,
)


class OptNode:
    def __init__(self, level: int, dsl: str, opt_desc: str) -> None:
        """Initialize the optimization node with the given level, description, and DSL.
        :param level: int, Level in the tree
        :param dsl: str, DSL used in the node
        :param opt_desc: str, Description of the optimization
        :param opt_parents: List[str], Parent nodes description
        :param opt_children: List[OptNode], Children nodes
        """

        self.level = level  # int, Level in the tree
        self.dsl = dsl
        self.opt_desc = opt_desc  # str, Root node description
        self.opt_parents = []  # List[str], Parent nodes description
        self.opt_children = []  # List[OptNode], Children nodes

    def add_children(self, child_nodes):
        """Adds a child node to the current node."""
        self.opt_children.extend(child_nodes)

    def remove_children(self, child_nodes):
        """Removes a child node from the current node."""
        for child in child_nodes:
            if child in self.opt_children:
                self.opt_children.remove(child)

    def add_parents(self, parent_nodes):
        """Adds a child node to the current node."""
        self.opt_parents.extend(parent_nodes)

    def remove_parents(self, parent_nodes):
        """Removes a child node from the current node."""
        for parent in parent_nodes:
            if parent in self.opt_parents:
                self.opt_parents.remove(parent)

    def __repr__(self):
        """String representation of the node for easy printing."""
        return f"OptNode at level {self.level}: ({self.opt_desc})"


class OptHierarchy:
    def __init__(self) -> None:
        """Initialize the optimization hierarchy with the root node."""
        self.root = OptNode(level=0, dsl="text", opt_desc="root")

    def get_root(self):
        return self.root

    def hard_initialize(self, common_path) -> None:
        """Hard initialize the hierarchy with pre-programmed database."""

        # Level 1 nodes - Latency, Memory, Utilization bottlenecks
        optnode_latency = OptNode(
            level=1,
            dsl="text",
            opt_desc="""To optimize compute-bound kernels, we employ techniques to reduce kernel execution latency, including:
            - Persistent programming style to minimize kernel launch overhead
            - Software pipelining to improve instruction-level parallelism and reduce execution time
        """,
        )
        optnode_memory = OptNode(
            level=1,
            dsl="text",
            opt_desc="""To optimize memory-bound kernels, we employ techniques to improve performance, including:
            - PID swizzling to enhance L2 cache locality
            - Leveraging new architecture features, such as Tensor Memory Accelerator (TMA) to overlap memory transfers
            with compute operations
        """,
        )
        optnode_utilization = OptNode(
            level=1,
            dsl="text",
            opt_desc="""To optimize kernels that are not fully utilizing hardware resources, we employ techniques
        to increase resource utilization and occupancy rates, including:
            - Leveraging Tensor Memory Accelerator (TMA) to overlap memory transfers with compute operations
            - Enabling warp specializations to improve instruction-level parallelism and reduce register pressure
            - Autotuning to identify and apply optimal kernel configurations that maximize resource usage
        """,
        )
        level_1_opts = [optnode_latency, optnode_memory, optnode_utilization]
        self.root.add_children(level_1_opts)
        optnode_latency.add_parents([self.root])
        optnode_memory.add_parents([self.root])
        optnode_utilization.add_parents([self.root])

        # Level 2 nodes - TMA, PID swizzling, persistent programming style
        optnode_host_TMA = OptNode(
            level=2, dsl="text", opt_desc=on_host_tma.ON_HOST_TMA
        )
        optnode_device_TMA = OptNode(
            level=2, dsl="text", opt_desc=on_device_tma.ON_DEVICE_TMA
        )
        optnode_PID_swizzling = OptNode(
            level=2, dsl="text", opt_desc=pid_swizzle.PID_SWIZZLE
        )
        optnode_persistence = OptNode(
            level=2, dsl="text", opt_desc=persistence.PERSISTENCE
        )

        optnode_latency.add_children([optnode_persistence])
        optnode_memory.add_children(
            [
                optnode_host_TMA,
                optnode_device_TMA,
                optnode_PID_swizzling,
                optnode_persistence,
            ]
        )
        optnode_utilization.add_children([optnode_persistence])

        optnode_host_TMA.add_parents([optnode_memory])
        optnode_device_TMA.add_parents([optnode_memory])
        optnode_PID_swizzling.add_parents([optnode_memory])
        optnode_persistence.add_parents(
            [optnode_latency, optnode_memory, optnode_utilization]
        )

        # Level 3 nodes - code example of each kernel
        # common_path="../kernel_opt/database/code_samples/"
        optnode_matmul = OptNode(
            level=3, dsl="triton", opt_desc=Path(common_path / "matmul.py").read_text()
        )
        optnode_matmul_pid_swizzling = OptNode(
            level=3,
            dsl="triton",
            opt_desc=Path(common_path / "matmul_sw.py").read_text(),
        )
        optnode_matmul_tma_host = OptNode(
            level=3,
            dsl="triton",
            opt_desc=Path(common_path / "matmul_tma_host.py").read_text(),
        )
        optnode_matadd = OptNode(
            level=3, dsl="triton", opt_desc=Path(common_path / "matadd.py").read_text()
        )
        optnode_matadd_persistence = OptNode(
            level=3,
            dsl="triton",
            opt_desc=Path(common_path / "matadd_perst.py").read_text(),
        )
        optnode_matadd_tma_host = OptNode(
            level=3,
            dsl="triton",
            opt_desc=Path(common_path / "matadd_tma_host.py").read_text(),
        )
        optnode_matadd_tma_device = OptNode(
            level=3,
            dsl="triton",
            opt_desc=Path(common_path / "matadd_tma_device.py").read_text(),
        )

        optnode_host_TMA.add_children(
            [
                optnode_matmul,
                optnode_matmul_tma_host,
                optnode_matadd,
                optnode_matadd_tma_host,
            ]
        )
        optnode_device_TMA.add_children([optnode_matadd, optnode_matadd_tma_device])
        optnode_PID_swizzling.add_children(
            [optnode_matmul, optnode_matmul_pid_swizzling]
        )
        optnode_persistence.add_children([optnode_matadd, optnode_matadd_persistence])

        optnode_matmul.add_parents([optnode_host_TMA, optnode_PID_swizzling])
        optnode_matmul_pid_swizzling.add_parents([optnode_PID_swizzling])
        optnode_matmul_tma_host.add_parents([optnode_host_TMA])
        optnode_matadd.add_parents(
            [optnode_host_TMA, optnode_device_TMA, optnode_persistence]
        )
        optnode_matadd_persistence.add_parents([optnode_persistence])
        optnode_matadd_tma_host.add_parents([optnode_host_TMA])
        optnode_matadd_tma_device.add_parents([optnode_device_TMA])
