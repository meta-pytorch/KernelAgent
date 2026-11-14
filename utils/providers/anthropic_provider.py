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

"""Anthropic provider implementation."""

import os

from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


def _anthropic_client_kwargs(api_key: str) -> dict:
    kwargs: dict = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    headers: dict[str, str] = {}
    for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        name, value = name.strip(), value.strip()
        if name and value:
            headers[name] = value
    if headers:
        kwargs["default_headers"] = headers
    return kwargs


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self):
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            # Initialize client (proxy configured via environment variables).
            # ANTHROPIC_BASE_URL and ANTHROPIC_CUSTOM_HEADERS carry the AMD
            # gateway route. Header values stay out of logs.
            self.client = Anthropic(**_anthropic_client_kwargs(api_key))

    def _create_kwargs(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs,
    ) -> dict:
        # The installed SDK dropped the temperature argument. The API still
        # accepts it on the JSON body.
        params = {
            "model": model_name,
            "max_tokens": min(
                kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
            ),
            "messages": messages,
            "extra_body": {"temperature": temperature},
        }
        return params

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")

        user_content = messages[-1]["content"] if messages else ""
        # Callers such as seed generation already put temperature in kwargs.
        # _create_kwargs takes that value as its own argument.
        forwarded = {k: v for k, v in kwargs.items() if k != "temperature"}
        response = self.client.messages.create(
            **self._create_kwargs(
                model_name,
                [{"role": "user", "content": user_content}],
                kwargs.get("temperature", 0.7),
                **forwarded,
            )
        )

        return LLMResponse(
            content=response.content[0].text, model=model_name, provider=self.name
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        base_temperature = kwargs.get("temperature", 0.7)
        return [
            self.get_response(
                model_name,
                messages,
                temperature=base_temperature + i * 0.1,
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )
            for i in range(n)
        ]

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None

    @property
    def name(self) -> str:
        return "anthropic"
