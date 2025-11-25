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

"""DeepSeek provider implementation."""

from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (OpenAI-compatible)."""

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com"
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for DeepSeek models."""
        # DeepSeek API 的 max_tokens 限制是 8192
        # 注意：这是输出token限制，不是上下文长度限制
        return 8192

    def supports_multiple_completions(self) -> bool:
        """DeepSeek API does not support n > 1."""
        return False