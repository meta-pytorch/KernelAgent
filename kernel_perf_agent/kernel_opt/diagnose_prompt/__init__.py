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

"""
Diagnose Prompt Module for Hardware Bottleneck Analysis.

This module provides prompt building utilities for the Judge LLM that
analyzes NCU profiling metrics to identify performance bottlenecks.
"""

from .gpu_specs import get_gpu_specs
from .judger_prompts import (
    build_judge_optimization_prompt,
    extract_judge_response,
    validate_judge_response,
)

__all__ = [
    "get_gpu_specs",
    "build_judge_optimization_prompt",
    "extract_judge_response",
    "validate_judge_response",
]
