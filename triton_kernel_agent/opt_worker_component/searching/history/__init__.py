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

"""History module for tracking optimization programs.

Provides persistent storage for kernel programs discovered during optimization,
enabling resume, cross-pollination, and trajectory analysis.
"""

from .records import AttemptRecord, Outcome
from .store import ProgramDatabase
from .models import ProgramEntry, ProgramMetrics
from .json_db import JSONProgramDatabase

__all__ = [
    "AttemptRecord",
    "Outcome",
    "ProgramDatabase",
    "JSONProgramDatabase",
    "ProgramEntry",
    "ProgramMetrics",
]
