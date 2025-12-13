from __future__ import annotations

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
import argparse
import json
import sys
import os
import multiprocessing as mp
from pathlib import Path

from .constants import ExitCode
from .config import OrchestratorConfig, new_run_id
from .paths import ensure_abs_regular_file, make_run_dirs, PathSafetyError
from .logging_utils import setup_file_logger
from .orchestrator import Orchestrator
from triton_kernel_agent.platform_config import get_platform_choices

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


def cmd_run(argv: list[str]) -> int:
    _load_dotenv_if_present()
    p = argparse.ArgumentParser(
        prog="fuse run", description="Fuse Orchestrator — first-wins runner"
    )
    p.add_argument(
        "--problem", required=True, help="Absolute path to the Python problem file"
    )
    p.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model name (Responses API, default: gpt-5)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--llm-timeout-s", type=int, default=120)
    p.add_argument("--run-timeout-s", type=int, default=180)
    p.add_argument("--stream", choices=["all", "winner", "none"], default="all")
    p.add_argument("--store-responses", action="store_true", default=False)
    p.add_argument("--isolated", action="store_true", default=False)
    p.add_argument("--deny-network", action="store_true", default=False)
    p.add_argument("--enable-reasoning-extras", action="store_true", default=True)
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform",
    )
    args = p.parse_args(argv)

    try:
        problem_path = ensure_abs_regular_file(args.problem)
    except PathSafetyError as e:
        print(str(e), file=sys.stderr)
        return int(ExitCode.INVALID_ARGS)

    cfg = OrchestratorConfig(
        problem_path=problem_path,
        model=args.model,
        workers=args.workers,
        max_iters=args.max_iters,
        llm_timeout_s=args.llm_timeout_s,
        run_timeout_s=args.run_timeout_s,
        stream_mode=args.stream,
        store_responses=args.store_responses,
        isolated=args.isolated,
        deny_network=args.deny_network,
        enable_reasoning_extras=args.enable_reasoning_extras,
        target_platform=args.target_platform,
    )

    run_id = new_run_id()
    FUSE_BASE_DIR.mkdir(exist_ok=True)
    try:
        d = make_run_dirs(FUSE_BASE_DIR, run_id)
    except FileExistsError:
        print("Run directory already exists unexpectedly; retry.", file=sys.stderr)
        return int(ExitCode.GENERIC_FAILURE)

    orch_dir = d["orchestrator"]
    run_dir = d["run_dir"]

    # Write orchestrator metadata
    (orch_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": json.loads(cfg.to_json()),
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
        cfg, run_dir=run_dir, workers_dir=d["workers"], orchestrator_dir=orch_dir
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: fuse run --problem /abs/path.py [--model <name>] [flags]",
            file=sys.stderr,
        )
        return int(ExitCode.INVALID_ARGS)
    cmd = argv[0]
    if cmd == "run":
        return cmd_run(argv[1:])
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        return int(ExitCode.INVALID_ARGS)


if __name__ == "__main__":
    sys.exit(main())
