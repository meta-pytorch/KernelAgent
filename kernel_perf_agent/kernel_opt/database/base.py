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

from __future__ import annotations

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

        Args:
            level: Level in the tree (0=root, 1=bottleneck type, 2=technique, 3=code example)
            dsl: DSL used in the node (e.g., "text", "triton")
            opt_desc: Description of the optimization or code example
        """
        self.level = level
        self.dsl = dsl
        self.opt_desc = opt_desc
        self.opt_parents: list[OptNode] = []
        self.opt_children: list[OptNode] = []

    def add_children(self, child_nodes: list[OptNode]) -> None:
        """Adds child nodes to this node."""
        self.opt_children.extend(child_nodes)

    def add_parents(self, parent_nodes: list[OptNode]) -> None:
        """Adds parent nodes to this node."""
        self.opt_parents.extend(parent_nodes)

    def __repr__(self) -> str:
        """String representation of the node for easy printing."""
        return f"OptNode at level {self.level}: ({self.opt_desc})"


def add_relation(parent: OptNode, children: list[OptNode]) -> None:
    """Add parent-child relationship symmetrically.

    Args:
        parent: The parent node
        children: List of child nodes to add
    """
    parent.add_children(children)
    for child in children:
        child.add_parents([parent])


class OptHierarchy:
    def __init__(self) -> None:
        """Initialize the optimization hierarchy with the root node."""
        self.root = OptNode(level=0, dsl="text", opt_desc="root")

    def get_root(self) -> OptNode:
        return self.root

    def hard_initialize(self, common_path: Path) -> None:
        """Hard initialize the hierarchy with pre-programmed database.

        Args:
            common_path: Path to the code_samples directory
        """
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
        add_relation(self.root, [optnode_latency, optnode_memory, optnode_utilization])

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

        add_relation(optnode_latency, [optnode_persistence])
        add_relation(
            optnode_memory,
            [
                optnode_host_TMA,
                optnode_device_TMA,
                optnode_PID_swizzling,
                optnode_persistence,
            ],
        )
        add_relation(optnode_utilization, [optnode_persistence])

        # Level 3 nodes - code examples
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

        add_relation(
            optnode_host_TMA,
            [
                optnode_matmul,
                optnode_matmul_tma_host,
                optnode_matadd,
                optnode_matadd_tma_host,
            ],
        )
        add_relation(optnode_device_TMA, [optnode_matadd, optnode_matadd_tma_device])
        add_relation(
            optnode_PID_swizzling, [optnode_matmul, optnode_matmul_pid_swizzling]
        )
        add_relation(optnode_persistence, [optnode_matadd, optnode_matadd_persistence])
