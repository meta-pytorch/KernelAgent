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

import os
import tempfile

import pytest

from utils.config_injectable import config_injectable


@config_injectable
def sample_func(a, b, c=10):
    return a + b + c


def test_config_injectable_all_args_provided():
    """All arguments passed directly; no config needed."""
    assert sample_func(1, 2, c=3) == 6


def test_config_injectable_from_yaml():
    """All arguments filled from a YAML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("a: 5\nb: 10\nc: 20\n")
        f.flush()
        try:
            assert sample_func(config=f.name) == 35
        finally:
            os.unlink(f.name)


def test_config_injectable_partial_override():
    """Positional args take precedence; config fills the rest."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("b: 100\nc: 200\n")
        f.flush()
        try:
            assert sample_func(1, config=f.name) == 301
        finally:
            os.unlink(f.name)


def test_config_injectable_missing_required():
    """Raises TypeError when required arguments are missing."""
    with pytest.raises(TypeError, match="Missing required arguments"):
        sample_func(1)


class SampleClass:
    @config_injectable
    def add(self, a, b, c=10):
        return a + b + c


def test_config_injectable_class_method():
    """Decorator works on instance methods; self is bound normally."""
    obj = SampleClass()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("b: 100\nc: 200\n")
        f.flush()
        try:
            assert obj.add(1, config=f.name) == 301
        finally:
            os.unlink(f.name)


@config_injectable
class SampleClassDecorated:
    def __init__(self, a, b, c=10):
        self.a = a
        self.b = b
        self.c = c


def test_config_injectable_class_init_no_config():
    """Decorated class works normally without a config file."""
    obj = SampleClassDecorated(1, 2, c=3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


def test_config_injectable_class_init():
    """Decorator on a class injects config into __init__."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("a: 5\nb: 10\nc: 20\n")
        f.flush()
        try:
            obj = SampleClassDecorated(config=f.name)
            assert obj.a == 5
            assert obj.b == 10
            assert obj.c == 20
        finally:
            os.unlink(f.name)


def test_config_injectable_class_init_partial():
    """Explicit args override config values during class init."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("a: 999\nb: 100\nc: 200\n")
        f.flush()
        try:
            obj = SampleClassDecorated(1, config=f.name)
            assert obj.a == 1  # explicit overrides yaml's 999
            assert obj.b == 100
            assert obj.c == 200
        finally:
            os.unlink(f.name)


def test_config_injectable_missing_required_from_yaml():
    """Raises TypeError when YAML config only provides some required arguments."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("a: 5\n")
        f.flush()
        try:
            with pytest.raises(TypeError, match="Missing required arguments"):
                sample_func(config=f.name)
        finally:
            os.unlink(f.name)
