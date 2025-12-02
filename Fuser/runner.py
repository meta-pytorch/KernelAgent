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
import os
import re
import shutil
import sys
import time
import stat
import random
import threading
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

STDOUT_MAX_TAIL = 20000  # bytes
STDERR_MAX_TAIL = 20000  # bytes
MAX_SCAN_BYTES = 512 * 1024  # bytes for bounded full-file scan

_SENTINEL = "ALL_TESTS_PASSED"
_PASS_REGEX = re.compile(r"\bPASS\b")


@dataclass(frozen=True)
class RunResult:
    rc: int
    passed: bool
    validator_used: str  # run_tests|sentinel|unknown
    reason: str
    t_started: float
    t_finished: float
    stdout_path: Path
    stderr_path: Path


def _tail_bytes(p: Path, max_bytes: int) -> bytes:
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            take = min(size, max_bytes)
            f.seek(size - take)
            return f.read()
    except FileNotFoundError:
        return b""


def _read_all_text_bounded(p: Path, max_bytes: int) -> tuple[str, bool]:
    try:
        with p.open("rb") as f:
            data = f.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace"), truncated
    except FileNotFoundError:
        return "", False


def _allowlist_env() -> dict[str, str]:
    allow: dict[str, str] = {}
    for k, v in os.environ.items():
        if k == "PATH":
            allow[k] = v
        elif k == "PYTHONPATH":
            # sanitize: keep only absolute, existing dirs
            parts = [p for p in v.split(os.pathsep) if p]
            keep: list[str] = []
            for p in parts:
                try:
                    pp = os.path.abspath(p)
                    if os.path.isabs(pp) and os.path.isdir(pp):
                        keep.append(pp)
                except Exception:
                    continue
            if keep:
                allow["PYTHONPATH"] = os.pathsep.join(keep)
        elif k.startswith("LANG") or k.startswith("LC_"):
            allow[k] = v
    # Determinism and small resource caps
    allow["PYTHONHASHSEED"] = "0"
    allow.setdefault("OMP_NUM_THREADS", "1")
    allow.setdefault("MKL_NUM_THREADS", "1")
    allow.setdefault("OPENBLAS_NUM_THREADS", "1")
    return allow


def _write_sitecustomize_block_network(dst_dir: Path) -> None:
    code = (
        "import socket\n"
        "def _block(*a, **k):\n    raise RuntimeError('network disabled')\n"
        "class _Blocked(socket.socket):\n    def connect(self, *a, **k):\n        _block()\n    def connect_ex(self, *a, **k):\n        _block()\n"
        "socket.socket = _Blocked\n"
        "socket.create_connection = _block\n"
    )
    (dst_dir / "sitecustomize.py").write_text(code, encoding="utf-8")


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
        # Execute the target script
        with open(exec_filename, "r") as f:
            code = compile(f.read(), exec_filename, "exec")
            exec(code, {"__name__": "__main__"})


def run_candidate(
    artifacts_code_path: Path,
    run_root: Path,
    timeout_s: int,
    isolated: bool,
    deny_network: bool,
    cancel_event: Optional["threading.Event"] = None,
) -> RunResult:
    """
    Execute a candidate program in a fresh run directory under run_root.
    - Copies artifacts_code_path to run_dir/candidate_main.py
    - Runs [sys.executable, '-u', 'candidate_main.py'] with optional -I (isolated)
    - If deny_network, injects sitecustomize.py to block sockets and do NOT use -I
    - Captures stdout/stderr to files; kills on timeout or cancel_event
    - Classifies pass/fail according to design precedence
    """
    run_dir = (
        run_root
        / f"attempt_{int(time.time() * 1000)}_{os.getpid()}_{random.randint(0, 9999):04d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # Prepare working files. We intentionally avoid the name "code.py" here because
    # Python's stdlib exposes a module with that name, and PyTorch's import stack
    # (via pdb -> code) would accidentally load the candidate file instead of the
    # stdlib module, leading to partially initialised torch packages.
    exec_filename = "candidate_main.py"
    code_dst = run_dir / exec_filename
    st = artifacts_code_path.lstat()
    if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise ValueError("artifacts_code_path must be a regular file (no symlink)")
    shutil.copy2(artifacts_code_path, code_dst)

    if deny_network:
        _write_sitecustomize_block_network(run_dir)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    argv = [sys.executable, "-u"]
    if isolated and not deny_network:
        argv.append("-I")
    argv.append(exec_filename)

    env = _allowlist_env()

    t_started = time.time()
    (run_dir / "EXEC_STARTED").write_text(str(t_started), encoding="utf-8")

    # Create process using multiprocessing
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

    # Read bounded scan for classification
    out_text, scan_truncated = _read_all_text_bounded(stdout_path, MAX_SCAN_BYTES)

    # Classification
    passed = False
    validator = "unknown"
    reason = ""
    if rc == 0:
        # Prefer explicit run_tests PASS if present in stdout
        if _PASS_REGEX.search(out_text):
            passed = True
            validator = "run_tests"
            reason = "run_tests printed PASS and exited 0"
        elif _SENTINEL in out_text:
            passed = True
            validator = "sentinel"
            reason = "sentinel ALL_TESTS_PASSED found and exited 0"
        else:
            passed = False
            validator = "unknown"
            if scan_truncated:
                reason = (
                    "rc==0 but neither PASS nor sentinel found (scan_truncated=true)"
                )
            else:
                reason = "rc==0 but neither PASS nor sentinel found"
    else:
        passed = False
        reason = f"nonzero exit code: {rc}"

    return RunResult(
        rc=rc,
        passed=passed,
        validator_used=validator,
        reason=reason,
        t_started=t_started,
        t_finished=t_finished,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
