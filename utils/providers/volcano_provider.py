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

"""Volcano Cloud (火山引擎) provider implementation.

Uses the Ark runtime API which is OpenAI-compatible.
Set ARK_API_KEY to your Volcano Engine API key.
"""

from .openai_base import OpenAICompatibleProvider

VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"


class VolcanoProvider(OpenAICompatibleProvider):
    """Volcano Cloud (火山引擎 / ByteDance Ark) API provider.

    OpenAI-compatible endpoint for Doubao and other Volcano models.
    Requires the ARK_API_KEY environment variable.
    """

    def __init__(self):
        super().__init__(api_key_env="ARK_API_KEY", base_url=VOLCANO_BASE_URL)

    @property
    def name(self) -> str:
        return "volcano"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for Volcano Cloud models."""
        if "256k" in model_name:
            return 16384
        elif "128k" in model_name:
            return 16384
        elif "32k" in model_name:
            return 8192
        elif "4k" in model_name:
            return 4096
        else:
            return 8192
