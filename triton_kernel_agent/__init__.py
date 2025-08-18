"""
Triton Kernel Generation Agent

An AI-powered system for generating and optimizing OpenAI Triton kernels for GPUs.
"""

__version__ = "0.1.0"

from .agent import TritonKernelAgent
from .worker import VerificationWorker
from .manager import WorkerManager
from .prompt_manager import PromptManager
from .utils import (
    get_meta_proxy_config,
    configure_proxy_environment,
    restore_proxy_environment,
)

__all__ = [
    "TritonKernelAgent",
    "VerificationWorker",
    "WorkerManager",
    "PromptManager",
    "get_meta_proxy_config",
    "configure_proxy_environment",
    "restore_proxy_environment",
]
