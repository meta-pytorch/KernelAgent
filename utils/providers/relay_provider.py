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

"""Relay provider implementation."""

from typing import Dict, List

import requests
import logging
import os

from .base import BaseProvider, LLMResponse


class RelayProvider(BaseProvider):
    """
    Plugboard server provider.

    This backend requires the plugboard server to be running:
      buck run @//mode/inplace run_plugboard_server -- --model gcp-claude-4-sonnet --pipeline usecase-dev-ai-user

    The RelayProvider class communicates with a local plugboard server (default: http://127.0.0.1:11434)
    to relay LLM requests and responses.
    """

    def __init__(self):
        self.server_url = os.environ.get("LLM_RELAY_URL", "http://127.0.0.1:11434")
        self.is_available_flag = False
        super().__init__()

    def _initialize_client(self) -> None:
        # Test connection to the server
        try:
            requests.get(f"{self.server_url}/", timeout=5)
            self.is_available_flag = True
        except Exception:
            self.is_available_flag = False

    def get_response(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """
        Supported kwargs:
        - max_tokens: int (default 8192)
        - text: str
        - high_reasoning_effort: bool (default False)
        - reasoning: dict

        TODO: Reasoning is handled twice (reasoning_effort and reasoning)
        this is due to multiple call sites conventions (orchestrator, KA)
        and OpenAI moving to Responses (vs Completion) should be cleaned up
        """

        max_tokens = kwargs.get("max_tokens", 8192)

        # Prepare request data for the plugboard server
        # TODO: Add support for passing in a custom temperature and top_p
        request_data = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }

        # Add reasoning config if high_reasoning_effort is set
        if kwargs.get("high_reasoning_effort", None):
            request_data["reasoning"] = {"effort": "high"}

        # Add pass-through kwargs
        nargs = ["text", "reasoning"]
        for arg in nargs:
            if arg in kwargs:
                request_data[arg] = kwargs[arg]

        logging.debug("\n=== DEBUG: PROMPT SENT TO LLM RELAY ===")
        logging.debug(request_data)
        logging.debug("=== END PROMPT ===\n")

        response = requests.post(
            self.server_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=int(os.environ.get("LLM_RELAY_TIMEOUT_S", 120)),
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Server returned status {response.status_code}: {response.text}"
            )

        response_data = response.json()
        logging.debug("\n=== DEBUG: RAW LLM RELAY RESPONSE ===")
        logging.debug(response_data)
        logging.debug("=== END RESPONSE ===\n")

        content = response_data.get("output", "")
        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.name,
            # Note: Plugboard doesn't have a response_id, so request_id is used
            response_id=response_data.get("plugboard_request_id", None),
        )

    def get_multiple_responses(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        n: int = 1,
        **kwargs,
    ) -> List[LLMResponse]:
        return [
            self.get_response(
                model_name,
                messages,
                temperature=kwargs.get("temperature", 0.7) + i * 0.1,
            )
            for i in range(n)
        ]

    def is_available(self) -> bool:
        return self.is_available_flag

    @property
    def name(self) -> str:
        return "relay"
