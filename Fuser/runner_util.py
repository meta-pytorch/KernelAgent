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

"""Utility functions for the compose worker."""

import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path


def _kill_process_group(pid: int) -> None:
    """Kill the process group of pid. Similar behavior as os.killpg."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    # Give processes a moment to terminate gracefully
    time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_candidate_process(
    exec_filename: str,
    run_dir: Path,
    argv: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    """Target function for multiprocessing.Process that executes the candidate."""
    # Create a new process group to contain potential subprocesses for cancel/timeout.
    # This is similar to subprocess.Popen(start_new_session=True) behavior.
    os.setpgrp()

    with stdout_path.open("wb") as f_out, stderr_path.open("wb") as f_err:
        os.chdir(str(run_dir))
        # Replace the child's environment (similar to subprocess.Popen(...env=env)).
        # This sets up the environment for the candidate execution.
        os.environ.clear()
        os.environ.update(env)
        # Redirect stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(f_out.fileno(), 1)
        os.dup2(f_err.fileno(), 2)

        # Explicitly execute sitecustomization (network blocking) if configured
        sitecustomize_path = run_dir / "sitecustomize.py"
        if sitecustomize_path.exists():
            with open(sitecustomize_path, "r") as f:
                sc_code = compile(f.read(), str(sitecustomize_path), "exec")
                exec(sc_code, {})

        # Find the script in argv and take everything from there
        if argv and exec_filename in argv:
            script_idx = argv.index(exec_filename)
            sys.argv = argv[script_idx:]
        else:
            sys.argv = [exec_filename]

        # Set up sys.path to allow imports from the script's directory
        sys.path.insert(0, str(run_dir))
        with open(exec_filename, "r") as f:
            code = compile(f.read(), exec_filename, "exec")
            exec(
                code,
                {
                    "__name__": "__main__",
                    "__file__": str(run_dir / exec_filename),
                    "__package__": None,
                },
            )


def _run_candidate_multiprocess(
    exec_filename,
    run_dir,
    argv,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    t_started: float,
    timeout_s: float,
    cancel_event,
):
    """Run candidate with multiprocessing.Process."""
    p = multiprocessing.Process(
        target=_run_candidate_process,
        args=(exec_filename, run_dir, argv, env, stdout_path, stderr_path),
    )
    p.start()

    rc: int
    try:
        while True:
            p.join(timeout=0.1)
            if not p.is_alive():
                rc = p.exitcode if p.exitcode is not None else 0
                break

            if cancel_event is not None and cancel_event.is_set():
                _kill_process_group(p.pid)
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
                rc = p.exitcode if p.exitcode is not None else -9
                break

            # Check wall-clock timeout
            if time.time() - t_started > timeout_s:
                _kill_process_group(p.pid)
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
                rc = p.exitcode if p.exitcode is not None else -9
                break
        # End while
    finally:
        t_finished = time.time()
        try:
            (run_dir / "EXEC_FINISHED").write_text(str(t_finished), encoding="utf-8")
        except Exception:
            pass

    return rc, t_finished
