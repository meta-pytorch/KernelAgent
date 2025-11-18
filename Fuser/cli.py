#!/usr/bin/env python3
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
Orchestrates parallel LLM workers to generate and verify
fused code.

Runs multiple workers concurrently against a KernelBench problem file, each
worker attempting to generate a valid solution. The first worker to produce a
passing candidate wins.

CLI (Hydra-based):
  python -m Fuser.cli problem=/abs/path/to/problem.py

  # Override config values:
  python -m Fuser.cli problem=/abs/path/to/problem.py \
      model=gpt-5 \
      workers=4 \
      max_iters=10 \
      stream=winner

  # Or use a custom config:
  python -m Fuser.cli --config-name custom_fuser \
      problem=/abs/path/to/problem.py

Config file: configs/pipeline/orchestrator.yaml

Requirements:
- OPENAI_API_KEY (.env in CWD or environment)

Outputs:
- Run directory path printed to stdout
- Artifacts in .fuse/<run_id>/
"""

from __future__ import annotations
import json
import sys
import os
import multiprocessing as mp
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig

from .config import new_run_id, OrchestratorConfig
from .constants import ExitCode
from .paths import ensure_abs_regular_file, make_run_dirs, PathSafetyError
from .logging_utils import setup_file_logger
from .orchestrator import Orchestrator

FUSE_BASE_DIR = Path.cwd() / ".fuse"


def _load_dotenv_if_present() -> None:
    """Load KEY=VALUE from .env in CWD without logging secrets.
    Existing env vars are not overridden."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


@hydra_main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent.parent / "configs/pipeline"),
    config_name="orchestrator",
)
def main(cfg: DictConfig) -> int:
    _load_dotenv_if_present()

    try:
        problem_path = ensure_abs_regular_file(cfg.problem)
    except PathSafetyError as e:
        print(str(e), file=sys.stderr)
        return int(ExitCode.INVALID_ARGS)

    orch_cfg = OrchestratorConfig(
        problem_path=problem_path,
        model=cfg.model,
        workers=cfg.workers,
        max_iters=cfg.max_iters,
        llm_timeout_s=cfg.llm_timeout_s,
        run_timeout_s=cfg.run_timeout_s,
        stream_mode=cfg.stream,
        store_responses=cfg.store_responses,
        isolated=cfg.isolated,
        deny_network=cfg.deny_network,
        enable_reasoning_extras=cfg.enable_reasoning_extras,
    )

    run_id = new_run_id()
    FUSE_BASE_DIR.mkdir(exist_ok=True)
    try:
        d = make_run_dirs(FUSE_BASE_DIR, run_id)
    except FileExistsError:
        print(
            "Run directory already exists unexpectedly; retry.",
            file=sys.stderr,
        )
        return int(ExitCode.GENERIC_FAILURE)

    orch_dir = d["orchestrator"]
    run_dir = d["run_dir"]

    # Write orchestrator metadata
    (orch_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": json.loads(orch_cfg.to_json()),
            },
            indent=2,
        )
    )

    logger = setup_file_logger(orch_dir / "orchestrator.log")
    logger.info("created run %s at %s", run_id, run_dir)

    # Start the run and print path immediately for discoverability
    print(str(run_dir))

    # Spawn orchestrator and execute first-wins
    mp.set_start_method("spawn", force=True)
    orch = Orchestrator(
        orch_cfg,
        run_dir=run_dir,
        workers_dir=d["workers"],
        orchestrator_dir=orch_dir,
    )
    summary = orch.run()

    # Map summary to exit codes
    if summary.artifact_path is None and summary.winner_worker_id is not None:
        return int(ExitCode.PACKAGING_FAILURE)
    if summary.winner_worker_id is None:
        if "canceled" in summary.reason:
            return int(ExitCode.CANCELED_BY_SIGNAL)
        return int(ExitCode.NO_PASSING_SOLUTION)
    return int(ExitCode.SUCCESS)


if __name__ == "__main__":
    sys.exit(main())
