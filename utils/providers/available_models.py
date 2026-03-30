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

"""External available models for KernelAgent."""

from utils.providers.model_config import ModelConfig
from utils.providers.openai_provider import OpenAIProvider
from utils.providers.anthropic_provider import AnthropicProvider
from utils.providers.relay_provider import RelayProvider
from utils.providers.volcano_provider import VolcanoProvider


# Registry of all available models (external/OSS version)
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
    ModelConfig(
        name="gpt-5.2",
        provider_classes=[OpenAIProvider],
        description="GPT-5.2 flagship model (Released Dec 2025)",
    ),
    # Anthropic Claude 4 Models (Latest)
    ModelConfig(
        name="claude-opus-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Opus - most intelligent (Released Feb 2026)",
    ),
    ModelConfig(
        name="claude-sonnet-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Sonnet - fast and powerful (Released Feb 2026)",
    ),
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
        description="Claude 4.5 Sonnet - latest balanced model (Released Sep 2025)",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_classes=[RelayProvider],
        description="[Relay] Claude 4 Sonnet",
    ),
    ModelConfig(
        name="claude-opus-4.5",
        provider_classes=[RelayProvider],
        description="Claude 4.5 Opus (Released Nov 2025)",
    ),
    ModelConfig(
        name="gpt-5-2",
        provider_classes=[RelayProvider],
        description="GPT-5.2 flagship model (Dec 2025) - Note the name is different from the OpenAI model",
    ),
    # Volcano Cloud (火山引擎) Doubao Models
    ModelConfig(
        name="doubao-seed-2.0-code",
        provider_classes=[VolcanoProvider],
        description="Doubao Seed 2.0 Code - code-specialized model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-1-5-pro-32k",
        provider_classes=[VolcanoProvider],
        description="Doubao 1.5 Pro 32K - high-performance model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-1-5-pro-256k",
        provider_classes=[VolcanoProvider],
        description="Doubao 1.5 Pro 256K - long-context model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-1-5-thinking-pro-32k",
        provider_classes=[VolcanoProvider],
        description="Doubao 1.5 Thinking Pro 32K - reasoning model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-1-5-thinking-pro-m-32k",
        provider_classes=[VolcanoProvider],
        description="Doubao 1.5 Thinking Pro M 32K - mid-size reasoning model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-pro-32k",
        provider_classes=[VolcanoProvider],
        description="Doubao Pro 32K - production model from Volcano Cloud",
    ),
    ModelConfig(
        name="doubao-pro-128k",
        provider_classes=[VolcanoProvider],
        description="Doubao Pro 128K - long-context production model from Volcano Cloud",
    ),
]
