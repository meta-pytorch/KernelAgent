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

"""Platform component registry.

Maps ``(component_key, implementation_name)`` pairs to factory
callables so that a YAML config like::

    platform:
      verifier: noop
      benchmarker: nvidia
      profiler: noop

can drive component selection at runtime.

Usage::

    from triton_kernel_agent.platform.registry import registry

    # Resolve a single component
    verifier = registry.create("verifier", "noop")

    # Resolve a full config dict (unrecognised kwargs are filtered
    # per-factory so callers can pass a shared kwargs bag)
    components = registry.create_from_config(
        {"verifier": "noop", "benchmarker": "nvidia"},
        log_dir=some_path, logger=some_logger, benchmark_lock=lock,
    )

    # Register a new backend
    registry.register("verifier", "rocm", RocmVerifier)
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


class PlatformRegistry:
    """Central registry that maps component keys to named implementations.

    Each *component key* (e.g. ``"verifier"``) has one or more named
    implementations (e.g. ``"nvidia"``, ``"noop"``).  Each
    implementation is stored as a callable **factory** — typically a
    class, but any ``(**kwargs) -> instance`` callable works.

    When :meth:`create` is called the factory receives only the subset
    of ``**kwargs`` that its signature actually accepts, so callers can
    pass a shared bag of context (``log_dir``, ``logger``, …) without
    worrying about ``TypeError`` from unrelated keys.
    """

    def __init__(self) -> None:
        # {component_key: {impl_name: factory}}
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        component: str,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        """Register *factory* under ``(component, name)``.

        Args:
            component: Component key, e.g. ``"verifier"``.
            name: Implementation name, e.g. ``"nvidia"`` or ``"noop"``.
            factory: A callable (class or function) that returns an
                instance of the component.  It will be called with
                filtered ``**kwargs`` (see :meth:`create`).
        """
        self._factories.setdefault(component, {})[name] = factory

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def list_components(self) -> list[str]:
        """Return all registered component keys."""
        return sorted(self._factories)

    def list_implementations(self, component: str) -> list[str]:
        """Return all registered names for *component*."""
        return sorted(self._factories.get(component, {}))

    def has(self, component: str, name: str) -> bool:
        """Check whether ``(component, name)`` is registered."""
        return name in self._factories.get(component, {})

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, component: str, name: str, **kwargs: Any) -> Any:
        """Instantiate ``(component, name)`` with filtered *kwargs*.

        Only the kwargs whose names match the factory's ``__init__``
        (or function) signature are forwarded; the rest are silently
        dropped.  This makes it safe for callers to pass a superset of
        context.

        Raises:
            KeyError: If *component* or *name* is not registered.
        """
        impls = self._factories.get(component)
        if impls is None:
            raise KeyError(
                f"Unknown component {component!r}. "
                f"Registered: {self.list_components()}"
            )
        factory = impls.get(name)
        if factory is None:
            raise KeyError(
                f"Unknown implementation {name!r} for {component!r}. "
                f"Registered: {self.list_implementations(component)}"
            )
        filtered = _filter_kwargs(factory, kwargs)
        return factory(**filtered)

    def create_from_config(
        self,
        config: dict[str, str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create multiple components from a ``{component: name}`` dict.

        Every component listed in *config* is instantiated via
        :meth:`create`, sharing the same *kwargs* bag (each factory
        only receives the kwargs it accepts).

        Returns:
            ``{component_key: instance}`` for every entry in *config*.
        """
        return {
            component: self.create(component, name, **kwargs)
            for component, name in config.items()
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _filter_kwargs(
    factory: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return the subset of *kwargs* accepted by *factory*'s signature.

    If the factory accepts ``**kwargs`` (VAR_KEYWORD), all kwargs are
    passed through unfiltered.
    """
    try:
        # For classes, inspect __init__
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return kwargs

    accepted: set[str] = set()
    has_var_keyword = False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            break
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(param.name)

    if has_var_keyword:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in accepted}


# ==================================================================
# Global singleton & built-in registrations
# ==================================================================

registry = PlatformRegistry()


def _register_builtins() -> None:
    """Register the built-in noop and nvidia implementations."""

    # --- noop (no hardware required) ---------------------------------
    from triton_kernel_agent.platform.noop import (
        NoOpAcceleratorSpecsProvider,
        NoOpBenchmarker,
        NoOpBottleneckAnalyzer,
        NoOpKernelProfiler,
        NoOpRAGPrescriber,
        NoOpRooflineAnalyzer,
        NoOpVerifier,
        NoOpWorkerRunner,
    )

    _noop = {
        "verifier": NoOpVerifier,
        "benchmarker": NoOpBenchmarker,
        "worker_runner": NoOpWorkerRunner,
        "specs_provider": NoOpAcceleratorSpecsProvider,
        "profiler": NoOpKernelProfiler,
        "roofline_analyzer": NoOpRooflineAnalyzer,
        "bottleneck_analyzer": NoOpBottleneckAnalyzer,
        "rag_prescriber": NoOpRAGPrescriber,
    }
    for component, factory in _noop.items():
        registry.register(component, "noop", factory)

    # --- nvidia (CUDA / NCU) ------------------------------------------
    from triton_kernel_agent.platform.nvidia import (
        NvidiaAcceleratorSpecsProvider,
        NvidiaBenchmarker,
        NvidiaBottleneckAnalyzer,
        NvidiaKernelProfiler,
        NvidiaRAGPrescriber,
        NvidiaRooflineAnalyzer,
        NvidiaVerifier,
        NvidiaWorkerRunner,
    )

    _nvidia = {
        # Manager-level
        "verifier": NvidiaVerifier,
        "benchmarker": NvidiaBenchmarker,
        "worker_runner": NvidiaWorkerRunner,
        # Worker-level
        "specs_provider": NvidiaAcceleratorSpecsProvider,
        "profiler": NvidiaKernelProfiler,
        "roofline_analyzer": NvidiaRooflineAnalyzer,
        "bottleneck_analyzer": NvidiaBottleneckAnalyzer,
        "rag_prescriber": NvidiaRAGPrescriber,
    }
    for component, factory in _nvidia.items():
        registry.register(component, "nvidia", factory)


_register_builtins()
