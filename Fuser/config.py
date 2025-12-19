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
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json
import time
import uuid
from triton_kernel_agent.platform_config import PlatformConfig, get_platform


@dataclass
class OrchestratorConfig:
    problem_path: Path
    model: str
    workers: int = 4
    max_iters: int = 10
    llm_timeout_s: int = 120
    run_timeout_s: int = 180
    stream_mode: str = "all"  # all|winner|none
    store_responses: bool = False
    isolated: bool = False
    deny_network: bool = False
    enable_reasoning_extras: bool = True
    target_platform: str = "cuda"

    def to_json(self) -> str:
        d = asdict(self)
        d["problem_path"] = str(self.problem_path)
        return json.dumps(d, indent=2)

    @property
    def platform_config(self) -> "PlatformConfig":
        """Resolve to PlatformConfig at runtime."""

        return get_platform(self.target_platform)


@dataclass
class OrchestratorState:
    run_id: str
    run_dir: Path
    started_ts: float


@dataclass
class WorkerConfig:
    run_id: str
    worker_id: str
    variant_index: int
    model: str
    max_iters: int
    llm_timeout_s: int
    run_timeout_s: int
    store_responses: bool
    isolated: bool
    deny_network: bool
    enable_reasoning_extras: bool
    stream_dir: Path
    workspace_dir: Path
    shared_digests_dir: Path
    target_platform: str = "cuda"

    @property
    def platform_config(self) -> "PlatformConfig":
        """Resolve to PlatformConfig at runtime."""

        return get_platform(self.target_platform)


@dataclass
class ResultSummary:
    run_id: str
    winner_worker_id: Optional[str]
    artifact_path: Optional[str]
    reason: str


def new_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short}"
