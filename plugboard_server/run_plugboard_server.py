# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""
Creates servers which will allow you to do inference via Plugboard
for KernelAgent (https://github.com/meta-pytorch/KernelAgent).

This is a fork of the original work from
https://www.internalfb.com/code/fbsource/[c8633a18b706664acbd22bee74218a775d092422]/fbcode/scripts/rahulkindi/internal_llm_relay/
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import os
import uuid
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, List, Optional, Sequence

import flask
import servicerouter.py3
import tiktoken
from facebook.ai_productivity.plugboard.plugboard.clients import (
    AiProductivity_Plugboard,
)
from facebook.ai_productivity.plugboard.plugboard.types import (
    Message as plugboard_Message,
    ModelParams,
    ReasoningConfig,
    ReasoningEffort,
    RunPipelineRequest,
    TextVerbosity,
)
from flask import Flask, jsonify, request
from kernelagent.plugboard_server.types import RequestData, ResponseData, Usage
from servicerouter.py3 import ClientParams

app = Flask(__name__)


LLM_RELAY_CLIENT_ID = "kernel_agent_llm_relay"
PLUGBOARD_PIPELINE = "usecase-dev-ai-user"
PLUGBOARD_TIER = "metamate_platform.plugboard"


@dataclass
class PlugboardConfig:
    """Configuration for plugboard requests."""

    model: str = "gpt-5"
    pipeline: str = PLUGBOARD_PIPELINE
    use_streaming: bool = False
    verbose: bool = False


def get_plugboard_message_list(
    lst: List[Dict[str, str]],
) -> tuple[str, Sequence[plugboard_Message]]:
    if len(lst) == 0:
        raise ValueError("No messages specified")
    last_message = lst[-1]
    if "content" not in last_message:
        raise ValueError("No content specified in last message")
    prompt = last_message["content"]
    messages = []
    for item in lst[:-1]:
        if "role" not in item or "content" not in item:
            continue
        messages.append(plugboard_Message(role=item["role"], content=item["content"]))
    return (prompt, messages)


@functools.cache
def get_tiktokenizer() -> Optional[tiktoken.Encoding]:
    try:
        return tiktoken.encoding_for_model("gpt-4-turbo")
    except OSError:  # Network error
        return None


def count_tokens(message: str) -> int:
    tokenizer = get_tiktokenizer()
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(message))


def message_list_to_str(messages: List[Dict[str, str]]) -> str:
    return "\n".join([f"{item['role']}: {item['content']}" for item in messages])


@app.route("/", methods=["POST"])
def handle_plugboard() -> flask.Response:
    print(request)
    data = request.get_json()
    args = parse_args()

    response_datas = run_plugboard_sync(
        messages=data.get("messages", []),
        model=data.get("model") or args.model,
        temperature=data.get("temperature", 0.7),
        n=data.get("n", 1),
        top_p=data.get("top_p", 1),
        max_tokens=data.get("max_tokens", 8192),
        pipeline=args.pipeline,
        use_streaming=args.use_streaming,
        verbose=args.verbose,
        reasoning=data.get("reasoning"),
        text=data.get("text"),
    )

    return jsonify([dataclasses.asdict(r) for r in response_datas])


async def run_plugboard_request(
    data: RequestData, config: Optional[PlugboardConfig] = None
) -> ResponseData:
    """
    Core plugboard request handler.

    This function can be called directly (for internal provider) or via Flask endpoint.

    Args:
        data: Request data containing messages, model params, etc.
        config: Optional configuration. If None, uses defaults from CLI args.

    Returns:
        ResponseData containing the model output and usage info.
    """
    if config is None:
        args = parse_args()
        config = PlugboardConfig(
            model=args.model,
            pipeline=args.pipeline,
            use_streaming=args.use_streaming,
            verbose=args.verbose,
        )

    params = ClientParams()
    params.setClientId(LLM_RELAY_CLIENT_ID)

    model = data.model or config.model
    model_params_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": data.max_tokens,
    }

    # Only add reasoning for models that support them
    if data.reasoning and model in (
        "o4-mini",
        "gpt-5",
        "gpt-5-2",
        "gpt-5-4",
        "gpt-5-5",
    ):
        effort = data.reasoning["effort"]
        enum_effort = ReasoningEffort(
            1
            if effort == "low"
            else (2 if effort == "medium" else (3 if effort == "high" else 4))
        )
        model_params_kwargs["reasoning_config"] = ReasoningConfig(effort=enum_effort)

    # Force high effort for avocado
    if model in (
        "guacamole-metamate-main-5-1",
        "guacamole-metamate-54-demo",
        "guacamole-metamate-58-reason",
        "guacamole-metamate-59-reason",
    ):
        model_params_kwargs["reasoning_config"] = ReasoningConfig(
            effort=ReasoningEffort(3)
        )

    # OpenAI o-series models don't support temperature and top_p parameters
    if model not in ("o4-mini", "gpt-5", "gpt-5-2", "gpt-5-4", "gpt-5-5"):
        model_params_kwargs["temperature"] = data.temperature
        model_params_kwargs["top_p"] = data.top_p

        # Claude 4.5/6 doesn't support simultaneous temperature and top_p
        # Take a preference on temperature over top_p if temp is non-zero
        if model == "claude-opus-4.5" or model == "claude-opus-4.6":
            if data.temperature == 0.0:
                del model_params_kwargs["temperature"]
            else:
                del model_params_kwargs["top_p"]

    if text_options := data.text:
        if verbosity := text_options.get("verbosity"):
            model_params_kwargs["text_verbosity"] = TextVerbosity(
                1 if verbosity == "low" else (2 if verbosity == "medium" else 3)
            )

    model_params = ModelParams(**model_params_kwargs)
    print("ModelParams", model_params)

    client = servicerouter.py3.get_sr_client(
        AiProductivity_Plugboard, tier=PLUGBOARD_TIER, params=params
    )

    (prompt, history) = get_plugboard_message_list(data.messages)
    history = list(history)
    if len(prompt) > 0:
        history.append(plugboard_Message(role="user", content=prompt))

    pipeline_request = RunPipelineRequest(
        history=history,
        pipeline=config.pipeline,
        request_correlator=str(uuid.uuid4()),
        model_params=model_params,
        user_id_insecure=None,
        service_overrides=None,
    )

    if config.use_streaming:
        pipeline_response = await run_pipeline_streaming_request(
            client, pipeline_request, config.verbose
        )
    else:
        async with client:
            pipeline_response = await client.run_pipeline(pipeline_request)

    if (
        pipeline_response
        and pipeline_response.response
        and pipeline_response.response.content
    ):
        content = pipeline_response.response.content
        finish_reason = pipeline_response.response.finish_reason
        if config.verbose:
            print("💁 PROMPT:")
            pprint(prompt)
            print("🕒 HISTORY:")
            pprint(history)
            print("🤖 RESPONSE:")
            pprint(content)

        response_data = ResponseData(
            output=content,
            usage=Usage(
                prompt_tokens=count_tokens(message_list_to_str(data.messages)),
                completion_tokens=count_tokens(content),
            ),
            # pyrefly: ignore [missing-attribute]
            finish_reason=finish_reason.name,
            plugboard_request_id=pipeline_response.request_correlator,
        )
        return response_data
    else:
        raise ValueError("No response from Plugboard")


def run_plugboard_sync(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    n: int = 1,
    top_p: float = 1,
    max_tokens: int = 8192,
    pipeline: str = PLUGBOARD_PIPELINE,
    use_streaming: bool = False,
    verbose: bool = False,
    reasoning: Optional[Dict[str, str]] = None,
    text: Optional[Dict[str, Any]] = None,
) -> List[ResponseData]:
    """
    Synchronous interface for plugboard requests.

    This is the main entry point for direct internal usage without HTTP.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model name to use (e.g., 'gpt-5', 'gcp-claude-4-sonnet').
        temperature: Sampling temperature (0.0-1.0).
        n: Number of responses to generate.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens in the response.
        pipeline: Plugboard pipeline to use.
        use_streaming: Whether to use streaming API.
        verbose: Whether to print debug information.
        reasoning: Optional reasoning config for supported models.
        text: Optional text config for supported models.

    Returns:
        List of ResponseData containing the model output and usage info.
    """
    data = RequestData(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
        text=text,
    )

    config = PlugboardConfig(
        model=model or "gpt-5",
        pipeline=pipeline,
        use_streaming=use_streaming,
        verbose=verbose,
    )

    async def requests() -> list[ResponseData]:
        return await asyncio.gather(
            *(run_plugboard_request(data, config) for _ in range(n))
        )

    return asyncio.run(requests())


async def run_pipeline_streaming_request(client, pipeline_request, verbose: bool):
    """
    Use run_pipeline_streaming to avoid timeout issues with long-running requests.
    Collects all streamed chunks and returns a RunPipelineResponse matching
    the non-streaming response format.

    This will eventually be refactored with a non-processed streaming API
    """
    from facebook.ai_productivity.plugboard.plugboard.types import (
        Response,
        RunPipelineResponse,
        STREAM_ASSISTANT,
    )

    content_chunks = []
    finish_reason = None

    async with client:
        (init_response, stream) = await client.run_pipeline_streaming(pipeline_request)
        async for chunk in stream:
            # Only collect content from ASSISTANT stream (the actual text response)
            if chunk.stream_id == STREAM_ASSISTANT:
                if chunk.message and chunk.message.content:
                    content_chunks.append(chunk.message.content)
                if chunk.message and chunk.message.finish_reason:
                    finish_reason = chunk.message.finish_reason

    full_content = "".join(content_chunks)

    if verbose:
        print("🔄 STREAMING: Collected", len(content_chunks), "chunks")

    return RunPipelineResponse(
        request_correlator=init_response.request_correlator,
        role=init_response.role,
        response=Response(
            content=full_content,
            finish_reason=finish_reason,
        ),
    )


@functools.cache
def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--port",
        type=int,
        default=11434,
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
    )
    argparser.add_argument(
        "--pipeline",
        type=str,
        default=PLUGBOARD_PIPELINE,
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    argparser.add_argument(
        "--use-streaming",
        action="store_true",
        default=False,
        help="Use run_pipeline_streaming instead of run_pipeline to avoid timeout issues",
    )
    return argparser.parse_args()


def main() -> None:
    # Make sure the Tiktoken cache directory is set
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(cur_dir, "resources", "tiktoken")

    args = parse_args()
    app.run(port=args.port, threaded=True)


if __name__ == "__main__":
    main()
