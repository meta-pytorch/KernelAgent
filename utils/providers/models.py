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

from typing import Dict, List, Optional, Type

from .base import BaseProvider
from .model_config import ModelConfig

# Cached model lookup dictionary (lazily initialized)
_model_name_to_config: Optional[Dict[str, ModelConfig]] = None

# Provider instances cache
_provider_instances: Dict[Type[BaseProvider], BaseProvider] = {}


def _get_or_create_provider(
    provider_class: Type[BaseProvider],
) -> BaseProvider:
    """Get a cached provider instance or create a new one."""
    if provider_class not in _provider_instances:
        _provider_instances[provider_class] = provider_class()
    return _provider_instances[provider_class]


def get_available_models() -> List[ModelConfig]:
    from .available_models import AVAILABLE_MODELS

    return AVAILABLE_MODELS


def _get_model_name_to_config() -> Dict[str, ModelConfig]:
    """Get the model name to config lookup dictionary (lazily initialized)."""
    global _model_name_to_config
    if _model_name_to_config is None:
        _model_name_to_config = {model.name: model for model in get_available_models()}
    return _model_name_to_config


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
    model_name_to_config = _get_model_name_to_config()
    if model_name not in model_name_to_config:
        available = list(model_name_to_config.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    model_config = model_name_to_config[model_name]

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
    tried_names = [_get_or_create_provider(p).name for p in providers_to_try]
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
