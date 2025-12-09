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
import sys
import time
from pathlib import Path

def _run_candidate_process(
    exec_filename: str,
    run_dir: Path,
    argv: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    """Target function for multiprocessing.Process that executes the candidate."""
    with stdout_path.open("wb") as f_out, stderr_path.open("wb") as f_err:
        os.chdir(str(run_dir))
        # Replace environment
        os.environ.clear()
        os.environ.update(env)
        # Redirect stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(f_out.fileno(), sys.stdout.fileno())
        os.dup2(f_err.fileno(), sys.stderr.fileno())

        # Find the script in argv and take everything from there
        if argv and exec_filename in argv:
            script_idx = argv.index(exec_filename)
            sys.argv = argv[script_idx:]
        else:
            sys.argv = [exec_filename]
        # Execute the target script
        with open(exec_filename, "r") as f:
            code = compile(f.read(), exec_filename, "exec")
            exec(code, {"__name__": "__main__"})


def _run_candidate_multiprocess(
    exec_filename, 
    run_dir, 
    argv, 
    env: dict[str, str],
    stdout_path: Path, 
    stderr_path: Path, 
    t_started: float, 
    timeout_s: float, 
    cancel_event):
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
                # Terminate process
                p.terminate()
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
                rc = p.exitcode if p.exitcode is not None else -9
                break

            # Check wall-clock timeout
            if time.time() - t_started > timeout_s:
                p.terminate()
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass
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
