#!/usr/bin/env python3
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

"""Utility script to list all available models and their providers."""

import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from utils.providers import AVAILABLE_MODELS, is_model_available

# Load environment variables
load_dotenv()


def main():
    """List all available models."""
    print("=" * 80)
    print("KernelAgent - Available Models")
    print("=" * 80)

    providers: dict[type, list] = defaultdict(list)
    for model in AVAILABLE_MODELS:
        for provider in model.provider_classes:
            providers[provider].append(model)

    # Display models by provider
    for provider, models in providers.items():
        print(f"\n🔹 {provider().name.upper()} Provider:")
        print("-" * 50)

        for model in models:
            available = is_model_available(model.name, provider)
            status = "✅ Available" if available else "❌ Not Available"

            print(f"  {model.name:<35} {status}")
            if model.description:
                print(f"    └─ {model.description}")

    print("\n" + "=" * 80)
    print("Usage:")
    print("Set OPENAI_MODEL in .env file to any available model name.")
    print("Example: OPENAI_MODEL=gpt-5")
    print("=" * 80)


if __name__ == "__main__":
    main()
