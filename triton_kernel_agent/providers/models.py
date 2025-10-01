"""
Model registry and configuration for KernelAgent.
"""

from dataclasses import dataclass
from typing import Dict, Type

from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .relay_provider import RelayProvider


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider_class: Type[BaseProvider]
    description: str = ""


# Registry of all available models
AVAILABLE_MODELS = [
    ModelConfig(
        name="o4-mini",
        provider_class=OpenAIProvider,
        description="OpenAI o4-mini - fast reasoning model",
    ),
    # OpenAI GPT-5 Model (Only GPT-5)
    ModelConfig(
        name="gpt-5",
        provider_class=OpenAIProvider,
        description="GPT-5 flagship model (Released Aug 2025)",
    ),
    # Anthropic Claude 4 Models (Latest)
    ModelConfig(
        name="claude-opus-4-1-20250805",
        provider_class=AnthropicProvider,
        description="Claude 4.1 Opus - most capable (Released Aug 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_class=AnthropicProvider,
        description="Claude 4 Sonnet - high performance (Released May 2025)",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_class=RelayProvider,
        description="[Relay] Claude 4 Sonnet",
    ),
]

# Create lookup dictionaries
MODEL_NAME_TO_CONFIG: Dict[str, ModelConfig] = {
    model.name: model for model in AVAILABLE_MODELS
}


# Provider instances cache
_provider_instances: Dict[Type[BaseProvider], BaseProvider] = {}


def get_model_provider(model_name: str) -> BaseProvider:
    """
    Get the appropriate provider instance for a given model.

    Args:
        model_name: Name of the model

    Returns:
        Provider instance

    Raises:
        ValueError: If model is not found or provider is not available
    """
    if model_name not in MODEL_NAME_TO_CONFIG:
        available = list(MODEL_NAME_TO_CONFIG.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    model_config = MODEL_NAME_TO_CONFIG[model_name]
    provider_class = model_config.provider_class

    # Use cached instance if available
    if provider_class not in _provider_instances:
        _provider_instances[provider_class] = provider_class()

    provider = _provider_instances[provider_class]

    if not provider.is_available():
        raise ValueError(
            f"Provider '{provider.name}' for model '{model_name}' is not available. "
            f"Check API keys and dependencies."
        )

    return provider


def is_model_available(model_name: str) -> bool:
    """Check if a model is available and its provider is ready."""
    try:
        provider = get_model_provider(model_name)
        return provider.is_available()
    except (ValueError, Exception):
        return False
