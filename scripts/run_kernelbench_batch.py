#!/usr/bin/env python3
"""
Batch runner for KernelBench problems.

Given a root directory (e.g., KernelBench/level2), this script:
- Discovers problem files that look like KernelBench tasks (contain a Model class and get_inputs)
- Feeds each problem's source as the problem_description to TritonKernelAgent
- Saves session outputs and an aggregate summary JSON
- Optionally runs the generated test for each problem

Usage:
  python scripts/run_kernelbench_batch.py \
    --root /path/to/KernelBench/KernelBench/level2 \
    --model gpt-5 \
    --seeds 4 \
    --rounds 10 \
    --run-tests

Env:
  Reads API keys and defaults from .env if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure repo root is on sys.path when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernel_agent import TritonKernelAgent


def looks_like_kernelbench_problem(text: str) -> bool:
    return (
        "class Model" in text
        and "def get_inputs" in text
    )


def find_problem_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*.py"):
        # Skip common non-problem files
        name = p.name
        if name in {"__init__.py", "setup.py"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if looks_like_kernelbench_problem(text):
            files.append(p)
    return sorted(files)


def sanitize_relpath(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    # Replace path separators with underscores for folder naming
    return "_".join(rel.with_suffix("").parts)


def run_batch(
    root: Path,
    model: str | None,
    seeds: int,
    rounds: int,
    run_tests: bool,
    limit: int | None,
    files: list[Path] | None = None,
) -> int:
    load_dotenv()

    batch_ts = int(time.time())
    # Use a dedicated batch log dir; Agent will create per-session subfolders
    batch_log_dir = Path.cwd() / f"triton_kernel_logs/batch_{batch_ts}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    problems = files if files is not None else find_problem_files(root)
    if limit:
        problems = problems[:limit]

    print(f"Discovered {len(problems)} problem(s) under {root}")

    summary = {
        "root": str(root),
        "batch_log_dir": str(batch_log_dir),
        "model": model or os.getenv("OPENAI_MODEL"),
        "seeds": seeds,
        "rounds": rounds,
        "count": len(problems),
        "started": time.time(),
        "results": [],
    }

    failures = 0

    for idx, problem_path in enumerate(problems, 1):
        print("=" * 80)
        print(f"[{idx}/{len(problems)}] {problem_path}")
        try:
            problem_text = problem_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"! Failed to read file: {e}")
            failures += 1
            summary["results"].append(
                {
                    "path": str(problem_path),
                    "success": False,
                    "error": f"read_error: {e}",
                }
            )
            continue

        # Create a fresh agent per problem to avoid cross-run state leakage
        agent = TritonKernelAgent(
            num_workers=seeds,
            max_rounds=rounds,
            log_dir=str(batch_log_dir),
            model_name=model,
        )
        try:
            # Run the agent
            result = agent.generate_kernel(problem_text, test_code=None)
        finally:
            # Ensure resources are cleaned per problem
            try:
                agent.cleanup()
            except Exception:
                pass

        # Record
        rec = {
            "path": str(problem_path),
            "success": bool(result.get("success")),
            "session_dir": result.get("session_dir"),
            "worker_id": result.get("worker_id"),
            "rounds": result.get("rounds"),
        }

        if not result.get("success"):
            failures += 1
            rec["error"] = result.get("message", "unknown_error")
            print("✗ Failed to generate kernel")
            print(f"  Message: {rec['error']}")
            print(f"  Session directory: {rec['session_dir']}")
            summary["results"].append(rec)
            continue

        print("✓ Success")
        print(f"  Worker {result['worker_id']} in {result['rounds']} rounds")
        print(f"  Session: {result['session_dir']}")

        # Optionally run the generated test
        if run_tests:
            try:
                session_dir = Path(result["session_dir"])  # type: ignore
                test_code = (session_dir / "test.py").read_text(encoding="utf-8")
                # Provide kernel.py for the test to import
                (Path.cwd() / "kernel.py").write_text(result["kernel_code"])  # type: ignore
                (Path.cwd() / "final_test.py").write_text(test_code)
                rc = os.system(f"{sys.executable} final_test.py")
                if rc != 0:
                    print(f"! Test failed (rc={rc})")
                    rec["test_rc"] = rc
                else:
                    print("Test passed")
            finally:
                # Cleanup temp files
                for f in ["kernel.py", "final_test.py"]:
                    p = Path.cwd() / f
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

        summary["results"].append(rec)

    summary["finished"] = time.time()
    summary_path = batch_log_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("=" * 80)
    print(f"Batch finished. Summary: {summary_path}")
    print(f"Failures: {failures}/{len(problems)}")
    return 0 if failures == 0 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run KernelBench problems in batch")
    ap.add_argument("--root", required=True, help="Path to KernelBench level directory")
    ap.add_argument("--model", default=None, help="Model name (overrides OPENAI_MODEL)")
    ap.add_argument("--seeds", type=int, default=4, help="Number of kernel seeds (workers)")
    ap.add_argument("--rounds", type=int, default=10, help="Max refinement rounds")
    ap.add_argument("--run-tests", action="store_true", help="Run generated test for each")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    ap.add_argument("--summary", default=None, help="Path to previous batch_summary.json to filter reruns")
    ap.add_argument("--rerun-failed", action="store_true", help="If set with --summary, rerun only failed entries")
    ap.add_argument("--skip-succeeded-from", default=None, help="Path to summary to skip succeeded items")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        return 2
    selected_files: list[Path] | None = None
    if args.summary and args.rerun_failed:
        summary_path = Path(args.summary).resolve()
        if not summary_path.exists():
            print(f"Summary not found: {summary_path}")
            return 2
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            failed_paths = [Path(item.get("path", "")) for item in data.get("results", []) if not item.get("success")]
            # Keep only existing files
            selected_files = [p for p in failed_paths if p and p.exists()]
            print(f"Rerunning {len(selected_files)} failed problem(s) from summary")
        except Exception as e:
            print(f"Failed to parse summary: {e}")
            return 2
    elif args.skip_succeeded_from:
        # Build a file list excluding succeeded problems from a previous summary
        summary_path = Path(args.skip_succeeded_from).resolve()
        if not summary_path.exists():
            print(f"Summary not found: {summary_path}")
            return 2
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            succeeded = {item.get("path") for item in data.get("results", []) if item.get("success")}
            all_files = find_problem_files(root)
            selected_files = [p for p in all_files if str(p) not in succeeded]
            print(f"Skipping {len(succeeded)} succeeded problems from summary; running {len(selected_files)} remaining")
        except Exception as e:
            print(f"Failed to parse summary: {e}")
            return 2
    return run_batch(
        root=root,
        model=args.model,
        seeds=args.seeds,
        rounds=args.rounds,
        run_tests=args.run_tests,
        limit=args.limit,
        files=selected_files,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
