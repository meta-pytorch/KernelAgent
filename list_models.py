#!/usr/bin/env python3
"""
Utility script to list all available models and their providers.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from triton_kernel_agent.providers import AVAILABLE_MODELS, is_model_available

# Load environment variables
load_dotenv()


def main():
    """List all available models."""
    print("=" * 80)
    print("KernelAgent - Available Models")
    print("=" * 80)

    models = AVAILABLE_MODELS
    providers = {}

    # Group models by provider
    for model in models:
        provider_name = model.provider_class().name
        if provider_name not in providers:
            providers[provider_name] = []
        providers[provider_name].append(model)

    # Display models by provider
    for provider_name, provider_models in providers.items():
        print(f"\n🔹 {provider_name.upper()} Provider:")
        print("-" * 50)

        for model in provider_models:
            # Check if model is available (API key set)
            available = is_model_available(model.name)
            status = "✅ Available" if available else "❌ Not Available (check API key)"

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
