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

"""FSM engine and base state class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

# Sentinel value returned by terminal states
TERMINAL = "__TERMINAL__"


class State(ABC):
    """Base class for FSM states.

    Each state implements execute() which performs its work and returns
    the name of the next state to transition to. Terminal states return
    the TERMINAL sentinel.
    """

    @property
    def name(self) -> str:
        """State name used for registration and transitions."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self, ctx: Any) -> str:
        """Execute this state's logic.

        Args:
            ctx: The context object (OptimizationContext or WorkerContext)

        Returns:
            Name of the next state, or TERMINAL to stop
        """
        ...


class FSMEngine:
    """Simple FSM engine that runs states in sequence.

    Usage:
        engine = FSMEngine(logger=logger)
        engine.add_state(MyState())
        engine.run("MyState", ctx)
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.states: dict[str, State] = {}
        self.logger = logger or logging.getLogger(__name__)

    def add_state(self, state: State) -> FSMEngine:
        """Register a state. Returns self for chaining."""
        self.states[state.name] = state
        return self

    def run(self, initial_state: str, ctx: Any) -> None:
        """Run the FSM from initial_state until a terminal state.

        Args:
            initial_state: Name of the first state to execute
            ctx: Context object passed to every state
        """
        current = initial_state
        while current != TERMINAL:
            if current not in self.states:
                raise ValueError(
                    f"Unknown state '{current}'. "
                    f"Registered: {list(self.states.keys())}"
                )
            state = self.states[current]
            self.logger.debug(f"FSM: entering state '{current}'")
            next_state = state.execute(ctx)
            self.logger.debug(f"FSM: '{current}' -> '{next_state}'")
            current = next_state
