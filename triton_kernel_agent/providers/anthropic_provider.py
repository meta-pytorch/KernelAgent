"""
Anthropic provider implementation.
"""

from typing import List, Dict
from .base import BaseProvider, LLMResponse

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""
    
    def _initialize_client(self) -> None:
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = Anthropic(api_key=api_key)
    
    def get_response(self, model_name: str, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")
        
        user_content = messages[-1]["content"] if messages else ""
        response = self.client.messages.create(
            model=model_name,
            max_tokens=min(kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": user_content}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model_name,
            provider=self.name
        )
    
    def get_multiple_responses(self, model_name: str, messages: List[Dict[str, str]], n: int = 1, **kwargs) -> List[LLMResponse]:
        return [
            self.get_response(model_name, messages, temperature=kwargs.get("temperature", 0.7) + i * 0.1)
            for i in range(n)
        ]
    
    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None
    
    @property
    def name(self) -> str:
        return "anthropic"