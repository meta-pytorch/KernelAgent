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
import time
from pathlib import Path
from typing import Tuple, Optional


def register_digest(
    shared_digests_dir: Path, sha256: str, worker_id: str, iter_index: int
) -> Tuple[str, Optional[str]]:
    """
    Atomically register a digest in shared_digests_dir.
    Returns (status, owner_worker_id or None), where status is one of:
      - "unique": first time seen
      - "duplicate_same_worker": seen earlier by same worker (same or earlier iter)
      - "duplicate_cross_worker": seen by a different worker
    """
    shared_digests_dir.mkdir(parents=True, exist_ok=True)
    entry_path = shared_digests_dir / f"{sha256}.json"
    now = time.time()
    payload = {
        "sha256": sha256,
        "owner_worker_id": worker_id,
        "iter": iter_index,
        "ts": now,
    }
    try:
        with entry_path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2))
        return "unique", None
    except FileExistsError:
        try:
            existing = json.loads(entry_path.read_text(encoding="utf-8"))
            owner = existing.get("owner_worker_id")
        except Exception:
            owner = None
        if owner == worker_id:
            return "duplicate_same_worker", owner
        return "duplicate_cross_worker", owner
