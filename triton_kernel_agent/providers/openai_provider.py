"""
OpenAI provider implementation.
"""

from .openai_base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        super().__init__(api_key_env="OPENAI_API_KEY")
    
    @property
    def name(self) -> str:
        return "openai"
    
    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for OpenAI models."""
        if model_name.startswith(("gpt-5", "gpt-4", "o3", "o1")):
            return 32000
        elif model_name.startswith("gpt-3.5"):
            return 16000
        else:
            return 8192