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

"""Data records for tracking optimization attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Outcome(Enum):
    """Result of an optimization attempt."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    FAILED = "failed"


@dataclass
class AttemptRecord:
    """A single optimization attempt."""

    id: str
    kernel_code: str
    time_ms: float
    outcome: Outcome
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_id: str | None = None

    def __repr__(self) -> str:
        return f"AttemptRecord(id={self.id}, time_ms={self.time_ms:.4f}, outcome={self.outcome.value})"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "kernel_code": self.kernel_code,
            "time_ms": self.time_ms,
            "outcome": self.outcome.value,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
        }

    @staticmethod
    def from_dict(data: dict) -> AttemptRecord:
        """Deserialize from dictionary."""
        return AttemptRecord(
            id=data["id"],
            kernel_code=data["kernel_code"],
            time_ms=data["time_ms"],
            outcome=Outcome(data["outcome"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_id=data.get("parent_id"),
        )
