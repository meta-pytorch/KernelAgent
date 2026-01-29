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
import queue
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .config import WorkerConfig
from .event_adapter import EventAdapter
from .prompting import render_prompt, SYSTEM_PROMPT
from .code_extractor import extract_single_python_file, sha256_of_code
from .runner import run_candidate
from .logging_utils import setup_file_logger
from .dedup import register_digest

from utils.providers import get_model_provider


@dataclass
class WorkerState:
    worker_id: str
    iter_index: int
    last_response_id: str | None
    last_error: str | None
    passed: bool


def _ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "input": base / "input",
        "prompts": base / "prompts",
        "responses": base / "responses",
        "artifacts": base / "artifacts",
        "runs": base / "runs",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (dirs["artifacts"] / "latest").mkdir(parents=True, exist_ok=True)
    return dirs


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _tail_text(p: Path, max_bytes: int = 20000) -> str:
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            take = min(size, max_bytes)
            f.seek(size - take)
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


class Worker:
    def __init__(
        self,
        cfg: WorkerConfig,
        problem_path: Path,
        winner_queue: Any,
        cancel_event: Any,
        on_delta: Callable[[str, None]] | None = None,
        remote_url: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.problem_path = problem_path
        self.winner_queue = winner_queue
        self.cancel_event = cancel_event
        self.on_delta = on_delta
        self.remote_url = remote_url
        self.logger = setup_file_logger(
            cfg.workspace_dir / "logs" / "worker.log", name=f"worker-{cfg.worker_id}"
        )
        self.dirs = _ensure_dirs(cfg.workspace_dir)

    def run(self) -> None:
        state = WorkerState(
            worker_id=self.cfg.worker_id,
            iter_index=0,
            last_response_id=None,
            last_error=None,
            passed=False,
        )
        _write_json(self.cfg.workspace_dir / "state.json", asdict(state))

        for k in range(1, self.cfg.max_iters + 1):
            if self.cancel_event.is_set():
                self.logger.info("cancel seen; exiting")
                return

            state.iter_index = k
            _write_json(self.cfg.workspace_dir / "state.json", asdict(state))

            # Render prompt
            rp = render_prompt(
                problem_path=self.problem_path,
                variant_index=self.cfg.variant_index,
                attempt_index=k,
                error_context=state.last_error,
                enable_reasoning_extras=self.cfg.enable_reasoning_extras,
                model_name=self.cfg.model,
            )
            prompt_path = self.dirs["prompts"] / f"iteration_{k}.txt"
            prompt_path.write_text(rp.user, encoding="utf-8")

            """
            Temporary MUX to support Relay while we migrate to OpenAI Responses
            API.

            Uses EventAdapter for OpenAI otherwise Provider inferface
            """
            provider = get_model_provider(self.cfg.model)
            if provider.name != "openai":
                # Call LLM directly using provider
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": rp.user},
                ]
                try:
                    response = provider.get_response(
                        self.cfg.model, messages, max_tokens=16000, **rp.extras
                    )
                    result = {
                        "output_text": response.content or "",
                        "response_id": response.response_id or None,
                        "error": None,
                    }
                except Exception as e:
                    error = f"stream_error: {e.__class__.__name__}: {e}"
                    result = {
                        "output_text": "",
                        "response_id": None,
                        "error": error,
                    }
            else:
                # Stream via EventAdapter
                jsonl_path = self.dirs["responses"] / f"iteration_{k}.stream.jsonl"
                adapter = EventAdapter(
                    model=self.cfg.model,
                    store_responses=self.cfg.store_responses,
                    timeout_s=self.cfg.llm_timeout_s,
                    jsonl_path=jsonl_path,
                    stop_event=self.cancel_event,
                    on_delta=self.on_delta,
                )
                result = adapter.stream(
                    system_prompt=SYSTEM_PROMPT, user_prompt=rp.user, extras=rp.extras
                )

            state.last_response_id = result.get("response_id")
            _write_json(
                self.dirs["responses"] / f"iteration_{k}.final.json",
                result,
            )

            if self.cancel_event.is_set():
                self.logger.info("cancel after streaming; exiting")
                return

            # Extract code
            try:
                extracted = extract_single_python_file(result.get("output_text", ""))
            except Exception as e:
                state.last_error = f"EXTRACT_FAIL: {e}"
                self.logger.warning("iteration %d: extract failed: %s", k, e)
                continue

            iter_art_dir = self.dirs["artifacts"] / f"iteration_{k}"
            latest_dir = self.dirs["artifacts"] / "latest"
            iter_art_dir.mkdir(parents=True, exist_ok=True)
            (iter_art_dir / "code.py").write_text(extracted.code, encoding="utf-8")
            (latest_dir / "code.py").write_text(extracted.code, encoding="utf-8")

            # Dedup registration
            sha = sha256_of_code(extracted.code)
            status, owner = register_digest(
                self.cfg.shared_digests_dir, sha, self.cfg.worker_id, k
            )
            if status == "duplicate_cross_worker":
                self.logger.info("duplicate across workers (owner=%s); exiting", owner)
                return
            if status == "duplicate_same_worker":
                self.logger.info("duplicate in same worker; continuing")
                continue

            # Execute
            run_root = self.dirs["runs"] / f"iteration_{k}"
            run_root.mkdir(parents=True, exist_ok=True)
            rr = run_candidate(
                artifacts_code_path=latest_dir / "code.py",
                run_root=run_root,
                timeout_s=self.cfg.run_timeout_s,
                isolated=self.cfg.isolated,
                deny_network=self.cfg.deny_network,
                cancel_event=self.cancel_event,
                remote_url=self.remote_url,
            )

            if rr.passed:
                self.logger.info("PASS at iter %d via %s", k, rr.validator_used)
                try:
                    self.winner_queue.put(
                        {
                            "worker_id": self.cfg.worker_id,
                            "iter": k,
                            "validator": rr.validator_used,
                            "runs_dir": str(run_root),
                            "artifacts_dir": str(self.dirs["artifacts"]),
                        },
                        timeout=0.1,
                    )
                except queue.Full:
                    pass
                state.passed = True
                _write_json(self.cfg.workspace_dir / "state.json", asdict(state))
                return

            # Build ERROR_CONTEXT and continue
            out_tail = _tail_text(rr.stdout_path)
            err_tail = _tail_text(rr.stderr_path)
            state.last_error = f"RUN_FAIL: {rr.reason}\nSTDOUT_TAIL:\n{out_tail}\nSTDERR_TAIL:\n{err_tail}"
            _write_json(self.cfg.workspace_dir / "state.json", asdict(state))

        # Done all iterations
        self.logger.info("exhausted max_iters without PASS")
