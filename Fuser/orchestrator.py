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

import json
import multiprocessing as mp
import signal
import tarfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from typing import Any, Dict

from dotenv import load_dotenv

from .config import OrchestratorConfig, ResultSummary, WorkerConfig
from .logging_utils import redact, setup_file_logger

# Note: Worker is imported inside the child process entry to avoid heavy imports here.


# Spawn-safe worker entrypoint (top-level function; pass plain payload)
def _worker_process_main(
    cfg_payload: Dict[str, Any],
    problem_path: str,
    winner_queue: Any,
    cancel_event: Any,
    console_q: Any,
) -> None:
    from pathlib import Path as _P

    from .config import WorkerConfig as _WC
    from .worker import Worker as _Worker

    # Load environment variables in worker process
    load_dotenv()

    # Rehydrate dataclass (Path fields from str)
    wcfg = _WC(
        run_id=cfg_payload["run_id"],
        worker_id=cfg_payload["worker_id"],
        variant_index=cfg_payload["variant_index"],
        model=cfg_payload["model"],
        max_iters=cfg_payload["max_iters"],
        llm_timeout_s=cfg_payload["llm_timeout_s"],
        run_timeout_s=cfg_payload["run_timeout_s"],
        store_responses=cfg_payload["store_responses"],
        isolated=cfg_payload["isolated"],
        deny_network=cfg_payload["deny_network"],
        enable_reasoning_extras=cfg_payload["enable_reasoning_extras"],
        stream_dir=_P("."),  # unused by Worker
        workspace_dir=_P(cfg_payload["workspace_dir"]),
        shared_digests_dir=_P(cfg_payload["shared_digests_dir"]),
        target_platform=cfg_payload["target_platform"],
    )

    def _on_delta(s: str) -> None:
        try:
            console_q.put_nowait(s)
        except Exception:
            pass

    _Worker(
        cfg=wcfg,
        problem_path=_P(problem_path),
        winner_queue=winner_queue,
        cancel_event=cancel_event,
        on_delta=_on_delta,
    ).run()


class Orchestrator:
    def __init__(
        self,
        cfg: OrchestratorConfig,
        run_dir: Path,
        workers_dir: Path,
        orchestrator_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.workers_dir = workers_dir
        self.orchestrator_dir = orchestrator_dir
        self.logger = setup_file_logger(orchestrator_dir / "orchestrator.log")
        self.cancel_event = mp.Event()
        self.winner_queue: mp.Queue[Any] = mp.Queue(maxsize=1)
        self.console_threads: list[threading.Thread] = []
        self._stop_console = threading.Event()
        self._stream_mode = cfg.stream_mode
        self.target_platform = cfg.target_platform

    def _make_worker_cfg(self, idx: int) -> WorkerConfig:
        worker_id = f"worker_{idx + 1:02d}"
        wdir = self.workers_dir / worker_id
        wdir.mkdir(parents=True, exist_ok=True)
        digests_dir = self.run_dir / "shared" / "digests"
        return WorkerConfig(
            run_id=str(self.run_dir.name),
            worker_id=worker_id,
            variant_index=idx % 4,
            model=self.cfg.model,
            max_iters=self.cfg.max_iters,
            llm_timeout_s=self.cfg.llm_timeout_s,
            run_timeout_s=self.cfg.run_timeout_s,
            store_responses=self.cfg.store_responses,
            isolated=self.cfg.isolated,
            deny_network=self.cfg.deny_network,
            enable_reasoning_extras=self.cfg.enable_reasoning_extras,
            stream_dir=self.orchestrator_dir / "stream.log",
            workspace_dir=wdir,
            shared_digests_dir=digests_dir,
            target_platform=self.target_platform,
        )

    def _start_console_mux(self, queues: dict[str, mp.Queue[str]]) -> None:
        if self._stream_mode == "none":
            return

        def loop() -> None:
            last_flush: dict[str, float] = {k: 0.0 for k in queues}
            stream_path = self.orchestrator_dir / "stream.log"
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            f = stream_path.open("a", encoding="utf-8")
            try:
                while not self._stop_console.is_set():
                    now = time.time()
                    for wid, q in queues.items():
                        # throttle to ~20 Hz per worker
                        if now - last_flush[wid] < 0.05:
                            continue
                        chunks: list[str] = []
                        while True:
                            try:
                                delta = q.get_nowait()
                                chunks.append(delta)
                            except Empty:
                                break
                        if chunks:
                            last_flush[wid] = now
                            text = redact("".join(chunks))
                            try:
                                f.write(f"[{wid}] " + text)
                                f.flush()
                            except Exception:
                                pass
                            if self._stream_mode == "all":
                                try:
                                    print(f"[{wid}] ", end="", flush=False)
                                    print(text, end="", flush=True)
                                except Exception:
                                    pass
                            elif self._stream_mode == "winner":
                                # Print only for winner once known
                                if getattr(self, "_winner_id", None) == wid:
                                    try:
                                        print(f"[{wid}] ", end="", flush=False)
                                        print(text, end="", flush=True)
                                    except Exception:
                                        pass
                    time.sleep(0.02)
            finally:
                try:
                    f.close()
                except Exception:
                    pass

        t = threading.Thread(target=loop, name="console-mux", daemon=True)
        t.start()
        self.console_threads.append(t)

    def _stop_console_mux(self) -> None:
        self._stop_console.set()
        for t in self.console_threads:
            t.join(timeout=1.0)

    def run(self) -> ResultSummary:
        # Signal handling: set cancel on SIGINT/SIGTERM when running on main thread
        is_main_thread = threading.current_thread() is threading.main_thread()
        if is_main_thread:

            def _sig_handler(signum: int, frame: Any) -> None:  # pragma: no cover
                self.logger.info("received signal %s; canceling", signum)
                self.cancel_event.set()

            old_int = signal.signal(signal.SIGINT, _sig_handler)
            old_term = signal.signal(signal.SIGTERM, _sig_handler)
        else:
            old_int = old_term = None

        try:
            # Per-worker output queues for console mux
            delta_queues: dict[str, mp.Queue[str]] = {}
            procs: list[mp.Process] = []

            for i in range(self.cfg.workers):
                wcfg = self._make_worker_cfg(i)
                dq: mp.Queue[str] = mp.Queue(maxsize=256)
                delta_queues[wcfg.worker_id] = dq

                def on_delta_factory(wid: str, q: mp.Queue[str]):
                    def on_delta(s: str) -> None:
                        try:
                            q.put_nowait(s)
                        except Exception:
                            pass

                    return on_delta

                cfg_payload = {
                    "run_id": wcfg.run_id,
                    "worker_id": wcfg.worker_id,
                    "variant_index": wcfg.variant_index,
                    "model": wcfg.model,
                    "max_iters": wcfg.max_iters,
                    "llm_timeout_s": wcfg.llm_timeout_s,
                    "run_timeout_s": wcfg.run_timeout_s,
                    "store_responses": wcfg.store_responses,
                    "isolated": wcfg.isolated,
                    "deny_network": wcfg.deny_network,
                    "enable_reasoning_extras": wcfg.enable_reasoning_extras,
                    "workspace_dir": str(wcfg.workspace_dir),
                    "shared_digests_dir": str(wcfg.shared_digests_dir),
                    "target_platform": wcfg.target_platform,
                }
                p = mp.Process(
                    target=_worker_process_main,
                    name=wcfg.worker_id,
                    args=(
                        cfg_payload,
                        str(self.cfg.problem_path),
                        self.winner_queue,
                        self.cancel_event,
                        dq,
                    ),
                )
                p.start()
                procs.append(p)

            self._start_console_mux(delta_queues)

            winner: dict[str, Any] | None = None
            canceled_by_signal = False
            while True:
                if self.cancel_event.is_set():
                    canceled_by_signal = True
                    break
                try:
                    winner = self.winner_queue.get(timeout=0.1)
                    break
                except Empty:
                    # If all workers exited and no winner, treat as no-pass
                    if all(not p.is_alive() for p in procs):
                        break
            # Winner found or canceled
            if winner is not None:
                # expose winner id for console mux in 'winner' mode
                try:
                    self._winner_id = winner["worker_id"]
                except Exception:
                    self._winner_id = None
                self.cancel_event.set()
                self.logger.info("winner: %s", winner)
            # Stop console mux soon after cancel
            self._stop_console_mux()

            # Terminate other workers
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                try:
                    p.join(timeout=5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(timeout=1.0)

            reason = (
                "canceled"
                if winner is None and canceled_by_signal
                else (
                    f"pass via {winner.get('validator')} at iter {winner.get('iter')}"
                    if winner is not None
                    else "no_passing_solution"
                )
            )
            summary = ResultSummary(
                run_id=str(self.run_dir.name),
                winner_worker_id=winner.get("worker_id") if winner else None,
                artifact_path=None,
                reason=reason,
            )

            if winner is not None:
                try:
                    art_dir = (
                        Path(winner["artifacts_dir"]) / f"iteration_{winner['iter']}"
                    )
                    result_tar = self.run_dir / "result.tar.gz"
                    with tarfile.open(result_tar, "w:gz") as tf:
                        # Include code.py and runs dir for the winning iteration
                        tf.add(str(art_dir / "code.py"), arcname="code.py")
                        runs_dir = Path(winner.get("runs_dir", ""))
                        if runs_dir.is_dir():
                            tf.add(str(runs_dir), arcname="runs")
                    summary.artifact_path = str(result_tar)
                    # Winner files
                    win_dir = self.orchestrator_dir / "winner"
                    win_dir.mkdir(parents=True, exist_ok=True)
                    (win_dir / "worker_id.txt").write_text(
                        winner["worker_id"], encoding="utf-8"
                    )
                    (win_dir / "iter.txt").write_text(
                        str(winner["iter"]), encoding="utf-8"
                    )
                except Exception as e:
                    self.logger.error("packaging failed: %s", e)
                    # Exit code responsibility at CLI level; we record reason here
                    summary.reason = f"packaging_failure: {e}"

            (self.orchestrator_dir / "summary.json").write_text(
                json.dumps(asdict(summary), indent=2), encoding="utf-8"
            )
            return summary
        finally:
            if is_main_thread:
                assert old_int is not None and old_term is not None
                signal.signal(signal.SIGINT, old_int)
                signal.signal(signal.SIGTERM, old_term)
