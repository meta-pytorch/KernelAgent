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
One-shot pipeline runner: extract → dispatch → compose.

Usage:
  python -m Fuser.pipeline \
    --problem /abs/path/to/kernelbench_problem.py \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    [--dispatch-jobs 1] \
    --compose-model o4-mini \
    --workers 4 --max-iters 5 \
    --llm-timeout-s 1200 --run-timeout-s 1200 \
    --out-root ./.fuse \
    [--verify] [--compose-max-iters 5]

Writes all artifacts into the run directory created by the extractor. The final
composed kernel and composition summary live under <run_dir>/compose_out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .subgraph_extractor import extract_subgraphs_to_json
from .dispatch_kernel_agent import run as dispatch_run
from .compose_end_to_end import compose
from triton_kernel_agent.platform_config import get_platform_choices


def run_pipeline(
    problem_path: Path,
    extract_model: str,
    dispatch_model: Optional[str],
    compose_model: str,
    dispatch_jobs: int | str,
    workers: int,
    max_iters: int,
    llm_timeout_s: int,
    run_timeout_s: int,
    out_root: Optional[Path] = None,
    verify: bool = True,
    compose_max_iters: int = 5,
    target_platform: str = "cuda",
) -> dict:
    # Select default KernelAgent model if not provided: prefer GPT-5 for Level 2/3
    if dispatch_model is None:
        pp = str(problem_path)
        is_l2 = (
            ("/KernelBench/KernelBench/level2/" in pp)
            or ("/KernelBench/level2/" in pp)
            or ("level2/" in pp)
        )
        is_l3 = (
            ("/KernelBench/KernelBench/level3/" in pp)
            or ("/KernelBench/level3/" in pp)
            or ("level3/" in pp)
        )
        if is_l2 or is_l3:
            dispatch_model = "gpt-5"
        else:
            dispatch_model = "o4-mini"

    # Step 1: extract
    run_dir, subgraphs_path = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=extract_model,
        workers=workers,
        max_iters=max_iters,
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
        target_platform=target_platform,
    )

    # Step 2: dispatch to KernelAgent
    out_dir = Path(run_dir) / "kernels_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve dispatch concurrency (support "auto")
    jobs_val: int
    if isinstance(dispatch_jobs, str) and dispatch_jobs.strip().lower() == "auto":
        try:
            with Path(subgraphs_path).open("r", encoding="utf-8") as f:
                _items = json.load(f)
            jobs_val = max(1, int(len(_items))) if isinstance(_items, list) else 1
        except Exception:
            jobs_val = 1
    else:
        try:
            jobs_val = max(1, int(dispatch_jobs))
        except Exception:
            jobs_val = 1

    summary_path = dispatch_run(
        subgraphs_path=Path(subgraphs_path),
        out_dir=out_dir,
        agent_model=dispatch_model,
        jobs=jobs_val,
        target_platform=target_platform,
    )

    # Step 3: compose end-to-end
    compose_out = Path(run_dir) / "compose_out"
    compose_out.mkdir(parents=True, exist_ok=True)
    comp_res = compose(
        problem_path=problem_path,
        subgraphs_path=Path(subgraphs_path),
        kernels_summary_path=summary_path,
        out_dir=compose_out,
        model_name=compose_model,
        verify=verify,
        max_iters=compose_max_iters,
        target_platform=target_platform,
    )
    return {
        "run_dir": str(run_dir),
        "subgraphs": str(subgraphs_path),
        "kernels_summary": str(summary_path),
        "composition": comp_res,
    }


def main(argv: Optional[list[str]] = None) -> int:
    # Load .env if present for OPENAI_API_KEY, proxies, etc.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: extract → dispatch → compose"
    )
    p.add_argument("--problem", required=True, help="Absolute path to the problem file")
    p.add_argument("--extract-model", default="gpt-5")
    p.add_argument(
        "--dispatch-model",
        default=None,
        help="KernelAgent model (default: gpt-5 for level2 problems, else o4-mini)",
    )
    p.add_argument(
        "--dispatch-jobs",
        type=str,
        default="2",
        help="Max concurrent KernelAgent subgraph tasks (default: 2); use 'auto' to match subgraph count",
    )
    p.add_argument("--compose-model", default="o4-mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=5, help="Extractor iter budget")
    p.add_argument("--llm-timeout-s", type=int, default=1200)
    p.add_argument("--run-timeout-s", type=int, default=1200)
    p.add_argument("--out-root", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--compose-max-iters", type=int, default=5)
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform",
    )
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}")
        return 2

    try:
        res = run_pipeline(
            problem_path=problem_path,
            extract_model=args.extract_model,
            dispatch_model=args.dispatch_model,
            compose_model=args.compose_model,
            dispatch_jobs=args.dispatch_jobs,
            workers=args.workers,
            max_iters=args.max_iters,
            llm_timeout_s=args.llm_timeout_s,
            run_timeout_s=args.run_timeout_s,
            out_root=Path(args.out_root) if args.out_root else None,
            verify=args.verify,
            compose_max_iters=args.compose_max_iters,
            target_platform=args.target_platform,
        )
        print(json.dumps(res, indent=2))
        return 0
    except SystemExit as e:
        try:
            return int(e.code) if e.code is not None else 1
        except Exception:
            try:
                import sys as _sys

                print(str(e), file=_sys.stderr)
            except Exception:
                pass
            return 1
    except Exception as e:
        print(f"pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
