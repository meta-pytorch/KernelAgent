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
