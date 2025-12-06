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

"""Model registry and configuration for KernelAgent."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .relay_provider import RelayProvider


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider_classes: List[Type[BaseProvider]]
    description: str = ""


# Registry of all available models
AVAILABLE_MODELS = [
    ModelConfig(
        name="o4-mini",
        provider_classes=[OpenAIProvider],
        description="OpenAI o4-mini - fast reasoning model",
    ),
    # OpenAI GPT-5 Model (Only GPT-5)
    ModelConfig(
        name="gpt-5",
        provider_classes=[RelayProvider, OpenAIProvider],
        description="GPT-5 flagship model (Released Aug 2025)",
    ),
    # Anthropic Claude 4 Models (Latest)
    ModelConfig(
        name="claude-opus-4-1-20250805",
        provider_classes=[AnthropicProvider],
        description="Claude 4.1 Opus - most capable (Released Aug 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_classes=[AnthropicProvider],
        description="Claude 4 Sonnet - high performance (Released May 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-5-20250929",
        provider_classes=[AnthropicProvider],
        description="Claude 4.5 Sonnet - latest balanced model (Sep 2025)",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_classes=[RelayProvider],
        description="[Relay] Claude 4 Sonnet",
    ),
]

# Create lookup dictionaries
MODEL_NAME_TO_CONFIG: Dict[str, ModelConfig] = {
    model.name: model for model in AVAILABLE_MODELS
}


# Provider instances cache
_provider_instances: Dict[Type[BaseProvider], BaseProvider] = {}


def _get_or_create_provider(
    provider_class: Type[BaseProvider],
) -> BaseProvider:
    """Get a cached provider instance or create a new one."""
    if provider_class not in _provider_instances:
        _provider_instances[provider_class] = provider_class()
    return _provider_instances[provider_class]


def get_model_provider(
    model_name: str, preferred_provider: Optional[Type[BaseProvider]] = None
) -> BaseProvider:
    """
    Get the first available provider instance for a given model. If a preferred
    provider is specified, only it will be tried

    Args:
        model_name: Name of the model
        preferred_provider: Optional preffered provider class

    Returns:
        Provider instance
        (first available from the list of providers, or the preferred one)

    Raises:
        ValueError: If model is not found or no provider is available
    """
    if model_name not in MODEL_NAME_TO_CONFIG:
        available = list(MODEL_NAME_TO_CONFIG.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    model_config = MODEL_NAME_TO_CONFIG[model_name]

    # Determine which providers to try
    if preferred_provider is not None:
        if preferred_provider not in model_config.provider_classes:
            allowed = [p.__name__ for p in model_config.provider_classes]
            raise ValueError(
                f"Preferred provider '{preferred_provider.__name__}' "
                f"is not configured for model '{model_name}'. "
                f"Allowed providers: {allowed}"
            )
        providers_to_try = [preferred_provider]
    else:
        providers_to_try = model_config.provider_classes

    # Try each provider and return the first available one
    for provider_class in providers_to_try:
        provider = _get_or_create_provider(provider_class)
        if provider.is_available():
            return provider

    # No provider was available
    tried_names = [p.name() for p in providers_to_try]
    raise ValueError(
        f"No available provider for model '{model_name}'. "
        f"Tried providers: {tried_names}. "
        f"Check API keys and dependencies."
    )


def is_model_available(
    model_name: str, preferred_provider: Optional[Type[BaseProvider]] = None
) -> bool:
    """Check if a model is available and its provider is ready.
    If a preferred provider is specified, only it will be checked
    """
    try:
        provider = get_model_provider(model_name, preferred_provider)
        return provider.is_available()
    except (ValueError, Exception):
        return False
