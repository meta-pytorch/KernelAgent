from __future__ import annotations
import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Optional

_CODE_BLOCK_RE = re.compile(
    r"^```[ \t]*(\w+)?[ \t]*\n([\s\S]*?)^```[ \t]*$", re.MULTILINE | re.IGNORECASE
)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def canonicalize_code(s: str) -> str:
    """Canonicalize code for stable hashing and dedup.
    - Normalize newlines to LF
    - Strip trailing spaces on each line
    - Trim outer leading/trailing blank lines
    """
    s = _normalize_newlines(s)
    lines = [ln.rstrip(" \t") for ln in s.split("\n")]
    # Trim outer blank lines
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


@dataclass(frozen=True)
class ExtractedCode:
    code: str
    lang_tag: str


def extract_single_python_file(output_text: str) -> ExtractedCode:
    """Extract the last complete fenced Python code block from output_text.
    Raises ValueError if none found or if ast.parse fails.
    """
    text = _normalize_newlines(output_text)
    matches = list(_CODE_BLOCK_RE.finditer(text))
    # Walk from the end to find the last python-tagged block
    chosen = None
    for m in reversed(matches):
        lang = (m.group(1) or "").strip().lower()
        if lang == "python":
            chosen = m
            break
    if chosen is None:
        raise ValueError("no python fenced code block found in output")
    body = chosen.group(2)
    code = canonicalize_code(body)
    # Ensure it parses as Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"extracted code has syntax error: {e}")
    return ExtractedCode(code=code, lang_tag="python")


def sha256_of_code(code: str) -> str:
    c = canonicalize_code(code)
    return hashlib.sha256(c.encode("utf-8")).hexdigest()
