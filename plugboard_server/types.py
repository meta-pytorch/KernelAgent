# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RequestData:
    messages: List[Dict[str, str]]
    response_prefix: str | None = None  # assumes llama4 tokenization
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 256
    max_completion_tokens: Optional[int] = None
    """This is a synonym for `max_tokens`."""
    frequency_penalty: Optional[float] = 1.0
    model: Optional[str] = None
    n: int = 1
    stop: Optional[str] = None
    reasoning: Optional[Dict[str, str]] = None
    text: Optional[Dict[str, Any]] = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class ResponseData:
    output: str
    usage: Usage
    finish_reason: Optional[str]
    plugboard_request_id: str | None = None
