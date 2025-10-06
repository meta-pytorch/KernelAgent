"""
Base provider for OpenAI-compatible APIs.
"""

from typing import List, Dict, Any, Optional
import logging
from .base import BaseProvider, LLMResponse
from ..utils import configure_proxy_environment

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: Optional[str] = None):
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables)
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def get_response(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
        )

    def get_multiple_responses(
        self, model_name: str, messages: List[Dict[str, str]], n: int = 1, **kwargs
    ) -> List[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logging.getLogger(__name__).info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=choice.message.content,
                model=model_name,
                provider=self.name,
                usage=response.usage.dict()
                if hasattr(response, "usage") and response.usage
                else None,
            )
            for choice in response.choices
        ]

    def _build_api_params(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Auto-enable high reasoning for GPT-5
        if model_name.startswith("gpt-5"):
            params["reasoning_effort"] = "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"

        return params

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
