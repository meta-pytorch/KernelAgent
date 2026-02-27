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

import functools
import inspect
from omegaconf import OmegaConf


def _merge_args(func, args, kwargs):
    """Merge explicit args, YAML config, and defaults for the given function."""
    config_path = kwargs.pop("config", None)
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)

    config_data = {}
    if config_path is not None:
        config_data = OmegaConf.to_container(
            OmegaConf.load(config_path), resolve=True
        )

    for param in sig.parameters.values():
        if param.name not in bound_args.arguments and param.name in config_data:
            bound_args.arguments[param.name] = config_data[param.name]

    bound_args.apply_defaults()

    missing = [
        name
        for name, param in sig.parameters.items()
        if param.default is param.empty and name not in bound_args.arguments
    ]
    if missing:
        raise TypeError(f"Missing required arguments: {missing}")

    return bound_args.arguments


def config_injectable(target):
    """Decorator that allows function or class arguments to be supplied via a YAML config file.

    Can decorate functions, methods, or classes. When decorating a class, the
    ``config`` keyword is intercepted during ``__init__``.

    Argument priority: explicit > yaml > default.
    A ``TypeError`` is raised for any required arguments still missing after
    all three sources are consulted.
    """
    if isinstance(target, type):
        original_init = target.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            merged = _merge_args(original_init, (self, *args), kwargs)
            original_init(**merged)

        target.__init__ = new_init
        return target
    else:

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            return target(**_merge_args(target, args, kwargs))

        return wrapper
