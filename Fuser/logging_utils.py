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
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_REDACT_KEYS = {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"}


def redact(s: str) -> str:
    # Minimal redaction placeholder; expand as needed
    for k in _REDACT_KEYS:
        if k in s:
            s = s.replace(k, "<REDACTED_KEY>")
    return s


def setup_file_logger(log_file: Path, name: str = "fuse") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called multiple times
    if not any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "_fuse_tag", None) == str(log_file)
        for h in logger.handlers
    ):
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        fh._fuse_tag = str(log_file)
        logger.addHandler(fh)
    return logger
