"""
Utility functions for KernelAgent.
"""

import os
import subprocess
import logging
from typing import Dict, Optional


def get_meta_proxy_config() -> Optional[Dict[str, str]]:
    """
    Get Meta's proxy configuration if available.

    This function checks if the Meta environment proxy tools are available
    and returns the proxy configuration for HTTP/HTTPS requests.

    Returns:
        Dictionary with proxy settings (http_proxy, https_proxy) or None if not available
    """
    try:
        # Check if with-proxy command exists (Meta environment)
        result = subprocess.run(
            ["which", "with-proxy"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        # Get proxy environment variables from with-proxy
        result = subprocess.run(
            ["with-proxy", "env"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        # Parse proxy settings
        proxy_config = {}
        for line in result.stdout.split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                if key.lower() in ["http_proxy", "https_proxy"]:
                    proxy_config[key.lower()] = value

        return proxy_config if proxy_config else None

    except Exception:
        return None


def configure_proxy_environment() -> Optional[Dict[str, Optional[str]]]:
    """
    Configure proxy environment variables for Meta environment.
    This is the centralized proxy configuration logic used by all providers.

    Returns:
        Dictionary of original environment variable values for restoration,
        or None if no proxy configuration is needed.
    """
    proxy_config = get_meta_proxy_config()
    if not proxy_config:
        return None

    # Configure client with proxy via environment variables
    logging.getLogger().info(
        f"Using Meta proxy: {proxy_config.get('https_proxy', proxy_config.get('http_proxy'))}"
    )

    # Store original proxy settings
    original_proxy_env = {}
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ]:
        original_proxy_env[key] = os.environ.get(key)

    # Set proxy environment variables
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ]:
        proxy_url = proxy_config.get("https_proxy") or proxy_config.get("http_proxy")
        if proxy_url:
            os.environ[key] = proxy_url

    return original_proxy_env
