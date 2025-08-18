"""
DeepSeek provider implementation.
"""

from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (OpenAI-compatible)."""

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """DeepSeek models typically support higher token limits."""
        return 32000
