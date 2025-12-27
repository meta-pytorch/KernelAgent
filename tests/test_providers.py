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

"""Tests for LLM providers."""

import pytest
from utils.providers import (
    DeepSeekProvider,
    OpenAIProvider,
    AnthropicProvider,
    get_available_models,
    get_model_provider,
)


def test_deepseek_provider_import():
    """Test that DeepSeekProvider can be imported."""
    assert DeepSeekProvider is not None


def test_deepseek_provider_instantiation():
    """Test that DeepSeekProvider can be instantiated."""
    provider = DeepSeekProvider()
    assert provider is not None
    assert provider.name == "deepseek"


def test_deepseek_provider_properties():
    """Test DeepSeekProvider properties."""
    provider = DeepSeekProvider()
    
    # Test provider name
    assert provider.name == "deepseek"
    
    # Test max tokens limits
    assert provider.get_max_tokens_limit("deepseek-chat") == 8192
    assert provider.get_max_tokens_limit("deepseek-reasoner") == 8192
    
    # Test supports multiple completions (inherited from OpenAICompatibleProvider)
    assert provider.supports_multiple_completions() is True


def test_deepseek_models_in_registry():
    """Test that DeepSeek models are registered."""
    models = get_available_models()
    model_names = [model.name for model in models]
    
    assert "deepseek-chat" in model_names
    assert "deepseek-reasoner" in model_names
    
    # Verify model configs
    deepseek_models = [m for m in models if "deepseek" in m.name]
    assert len(deepseek_models) == 2
    
    for model in deepseek_models:
        assert DeepSeekProvider in model.provider_classes
        assert model.description != ""


def test_get_model_provider_deepseek():
    """Test getting provider for DeepSeek models."""
    # This should not raise an error even if API key is not available
    # It should just return a provider that is not available
    try:
        provider = get_model_provider("deepseek-chat")
        assert provider is not None
        assert provider.name == "deepseek"
    except ValueError as e:
        # This is expected if no API key is set
        assert "No available provider" in str(e)


def test_all_providers_can_be_instantiated():
    """Test that all provider classes can be instantiated."""
    providers = [
        OpenAIProvider,
        AnthropicProvider,
        DeepSeekProvider,
    ]
    
    for provider_class in providers:
        provider = provider_class()
        assert provider is not None
        assert provider.name is not None
        assert isinstance(provider.name, str)
        assert len(provider.name) > 0
