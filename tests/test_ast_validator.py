"""Tests for AST-based static code validation."""

import ast
import re

import pytest
from triton_kernel_agent.worker_util import validate_kernel_ast


def naive_string_validator(code_str: str) -> bool:
    """A regex-based validator for baseline comparison."""
    if re.search(r'\bimport\s+(os|sys|importlib)\b', code_str):
        return False
    if re.search(r'\beval\s*\(', code_str):
        return False
    if re.search(r'\bexec\s*\(', code_str):
        return False
    return True


def test_ast_validator_allows_safe_code() -> None:
    """Tests standard Python code."""
    safe_code = '''
def kernel_function(x):
    return x + 1
'''
    assert naive_string_validator(safe_code) is True
    assert validate_kernel_ast(safe_code) is True


def test_ast_validator_allows_safe_triton_code() -> None:
    """Tests complex legitimate Triton kernels."""
    triton_code = '''
import triton
import triton.language as tl

@triton.jit
def kernel_function(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2.0
    tl.store(y_ptr + offsets, output, mask=mask)
'''
    assert naive_string_validator(triton_code) is True
    assert validate_kernel_ast(triton_code) is True


def test_ast_validator_catches_import_os() -> None:
    """Tests direct module imports."""
    code = "import os\nos.system('ls')"
    assert naive_string_validator(code) is False
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_dynamic_import_bypass() -> None:
    """Tests string concatenated __import__ calls."""
    code = "__import__('o' + 's').system('ls')"
    assert naive_string_validator(code) is True
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_getattr_bypass() -> None:
    """Tests dynamic method retrieval via getattr."""
    code = "getattr(__import__('os'), 'system')('ls')"
    assert naive_string_validator(code) is True
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_eval_bypass() -> None:
    """Tests aliased eval calls."""
    code = "f = eval; f('1 + 1')"
    assert naive_string_validator(code) is True
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_importlib() -> None:
    """Tests dynamic module imports via importlib."""
    code = "import importlib\nos = importlib.import_module('os')\nos.system('ls')"
    assert naive_string_validator(code) is False
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_sys_modules() -> None:
    """Tests accessing sys.modules."""
    code = "import sys\nsys.modules['os'].system('ls')"
    assert naive_string_validator(code) is False
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_builtins_dict() -> None:
    """Tests accessing builtins.__dict__."""
    code = "import builtins\nbuiltins.__dict__['eval']('1+1')"
    assert naive_string_validator(code) is True
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_nested_eval() -> None:
    """Tests nested eval/exec calls."""
    code = "exec(eval('\"import os; os.system(\\'ls\\')\"'))"
    assert naive_string_validator(code) is False
    assert validate_kernel_ast(code) is False


def test_ast_validator_catches_string_constructed_getattr() -> None:
    """Tests getattr calls constructed with strings."""
    code = "f = getattr; f(sys, 'modules')"
    assert naive_string_validator(code) is True
    assert validate_kernel_ast(code) is False


def test_curve_ball_1_syntax_error() -> None:
    """Tests handling of invalid syntax."""
    code = "def kernel_function( :"
    assert validate_kernel_ast(code) is False


def test_curve_ball_2_dictionary_lookup() -> None:
    """Tests dictionary lookup on builtins."""
    code = "__builtins__['__import__']('os')"
    assert validate_kernel_ast(code) is False


def test_curve_ball_2_function_aliasing() -> None:
    """Tests function aliasing."""
    code = "sneaky_get = getattr; sneaky_get(tl, 'system')"
    assert validate_kernel_ast(code) is False
