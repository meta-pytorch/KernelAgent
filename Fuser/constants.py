"""Common constants and exit codes for Fuser CLIs."""

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

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    GENERIC_FAILURE = 1
    INVALID_ARGS = 2
    LLM_FAILURE = 3
    NO_PASSING_SOLUTION = 4
    CANCELED_BY_SIGNAL = 5
    PACKAGING_FAILURE = 6
