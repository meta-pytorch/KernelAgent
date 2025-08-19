"""
Base provider interface for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self):
        self.client = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider's client."""
        pass

    @abstractmethod
    def get_response(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """
        Get response from the LLM provider.

        Args:
            model_name: Name of the model to use
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with standardized format
        """
        pass

    @abstractmethod
    def get_multiple_responses(
        self, model_name: str, messages: List[Dict[str, str]], n: int = 1, **kwargs
    ) -> List[LLMResponse]:
        """
        Get multiple responses from the LLM provider.

        Args:
            model_name: Name of the model to use
            messages: List of message dicts with 'role' and 'content'
            n: Number of responses to generate
            **kwargs: Additional parameters

        Returns:
            List of LLMResponse objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key set, client initialized)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    def supports_multiple_completions(self) -> bool:
        """Whether this provider supports native multiple completions (n > 1)."""
        return False

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get the maximum tokens limit for a model."""
        return 8192  # Default limit

    def _get_api_key(self, env_var: str) -> Optional[str]:
        """Helper to get API key from environment."""
        api_key = os.getenv(env_var)
        if api_key and api_key != "your-api-key-here":
            return api_key
        return None
