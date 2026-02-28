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
    """Merge explicit args, YAML config, and defaults for the given function.

    For functions with **kwargs, YAML keys that don't match any named parameter
    are routed into **kwargs. Explicit **kwargs take precedence over YAML values.
    """
    config_path = kwargs.pop("config", None)
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)

    config_data = {}
    if config_path is not None:
        config_data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    named_params = set()
    var_keyword_param = None
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_param = param
        else:
            named_params.add(param.name)
        if param.name not in bound_args.arguments and param.name in config_data:
            bound_args.arguments[param.name] = config_data[param.name]

    # Route remaining config keys into **kwargs if the function accepts them
    if var_keyword_param is not None:
        kwargs_dict = bound_args.arguments.get(var_keyword_param.name, {})
        for key, value in config_data.items():
            if key not in named_params and key not in kwargs_dict:
                kwargs_dict[key] = value
        bound_args.arguments[var_keyword_param.name] = kwargs_dict

    bound_args.apply_defaults()

    missing = [
        name
        for name, param in sig.parameters.items()
        if param.default is param.empty and name not in bound_args.arguments
    ]
    if missing:
        raise TypeError(f"Missing required arguments: {missing}")

    # Flatten VAR_KEYWORD so func(**result) unpacks correctly
    result = {}
    for name, value in bound_args.arguments.items():
        param = sig.parameters.get(name)
        if param and param.kind == inspect.Parameter.VAR_KEYWORD:
            result.update(value)
        else:
            result[name] = value
    return result


def config_injectable(target):
    """Decorator that allows function or class arguments to be supplied via a YAML config file.

    Can decorate functions, methods, or classes. When decorating a class, the
    ``config`` keyword is intercepted during ``__init__``.

    Argument priority: explicit > yaml > default.
    A ``TypeError`` is raised for any required arguments still missing after
    all three sources are consulted.

    For functions/classes with ``**kwargs``, YAML keys that don't match any named
    parameter are routed into ``**kwargs``. Explicit ``**kwargs`` values take
    precedence over YAML values.
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
