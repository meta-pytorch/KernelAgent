"""
Model registry and configuration for KernelAgent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .deepseek_provider import DeepSeekProvider


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider_class: Type[BaseProvider]
    description: str = ""


# Registry of all available models
AVAILABLE_MODELS = [
    # OpenAI Models
    ModelConfig(
        name="gpt-4o",
        provider_class=OpenAIProvider,
        description="Latest GPT-4 optimized model"
    ),
    ModelConfig(
        name="gpt-4",
        provider_class=OpenAIProvider,
        description="Standard GPT-4 model"
    ),
    ModelConfig(
        name="gpt-3.5-turbo",
        provider_class=OpenAIProvider,
        description="Fast and cost-effective model"
    ),
    ModelConfig(
        name="o3-2025-04-16",
        provider_class=OpenAIProvider,
        description="Reasoning-focused model (requires verified org)"
    ),
    ModelConfig(
        name="o1-preview",
        provider_class=OpenAIProvider,
        description="Advanced reasoning model"
    ),
    
    # Anthropic/Claude Models
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_class=AnthropicProvider,
        description="Latest Claude Sonnet 4 model"
    ),
    ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider_class=AnthropicProvider,
        description="Claude 3.5 Sonnet model"
    ),
    ModelConfig(
        name="claude-3-5-haiku-20241022",
        provider_class=AnthropicProvider,
        description="Fast Claude 3.5 Haiku model"
    ),
    ModelConfig(
        name="claude-3-opus-20240229",
        provider_class=AnthropicProvider,
        description="Most capable Claude 3 model"
    ),
    
    # DeepSeek Models
    ModelConfig(
        name="deepseek-chat",
        provider_class=DeepSeekProvider,
        description="DeepSeek chat model"
    ),
    ModelConfig(
        name="deepseek-reasoner",
        provider_class=DeepSeekProvider,
        description="DeepSeek reasoning model"
    ),
    ModelConfig(
        name="deepseek-coder",
        provider_class=DeepSeekProvider,
        description="DeepSeek coding model"
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
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
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