from __future__ import annotations
from pathlib import Path
import stat


class PathSafetyError(Exception):
    pass


def ensure_abs_regular_file(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        raise PathSafetyError(f"problem path must be absolute: {path}")
    try:
        st = path.lstat()
    except FileNotFoundError:
        raise PathSafetyError(f"problem path does not exist: {path}")
    if stat.S_ISLNK(st.st_mode):
        raise PathSafetyError(f"problem path must not be a symlink: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise PathSafetyError(f"problem path must be a regular file: {path}")
    return path


def make_run_dirs(base: Path, run_id: str) -> dict[str, Path]:
    run_dir = base / run_id
    orch = run_dir / "orchestrator"
    workers = run_dir / "workers"
    shared = run_dir / "shared" / "digests"
    for d in (run_dir, orch, workers, shared):
        d.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "orchestrator": orch,
        "workers": workers,
        "digests": shared,
    }
