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

CLI (Hydra-based):
  python -m Fuser.pipeline problem=/abs/path/to/kernelbench_problem.py

  # Override config values:
  python -m Fuser.pipeline problem=/abs/path/to/kernelbench_problem.py \
      extract_model=gpt-5 \
      dispatch_model=o4-mini \
      compose_model=o4-mini \
      workers=4 \
      max_iters=5 \
      llm_timeout_s=1200 \
      run_timeout_s=1200 \
      fuser.verify=true \
      fuser.compose_max_iters=5 \
      dispatch_jobs=2

  # Or use a custom config:
  python -m Fuser.pipeline --config-name custom_pipeline \
      problem=/abs/path/to/kernelbench_problem.py

Config file: configs/pipeline/pipeline.yaml

Writes all artifacts into the run directory created by the extractor. The final
composed kernel and composition summary live under <run_dir>/compose_out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hydra import main as hydra_main
from omegaconf import DictConfig

from .subgraph_extractor import extract_subgraphs_to_json
from .dispatch_kernel_agent import run as dispatch_run
from .compose_end_to_end import compose


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
    )
    return {
        "run_dir": str(run_dir),
        "subgraphs": str(subgraphs_path),
        "kernels_summary": str(summary_path),
        "composition": comp_res,
    }


@hydra_main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent.parent / "configs/pipeline"),
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    # Load .env if present for OPENAI_API_KEY, proxies, etc.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    problem_path = Path(cfg.problem).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}")
        return 2

    try:
        res = run_pipeline(
            problem_path=problem_path,
            extract_model=cfg.extract_model,
            dispatch_model=cfg.dispatch_model,
            compose_model=cfg.compose_model,
            dispatch_jobs=cfg.dispatch_jobs,
            workers=cfg.workers,
            max_iters=cfg.max_iters,
            llm_timeout_s=cfg.llm_timeout_s,
            run_timeout_s=cfg.run_timeout_s,
            out_root=Path(cfg.out_root) if cfg.out_root else None,
            verify=cfg.verify,
            compose_max_iters=cfg.compose_max_iters,
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
