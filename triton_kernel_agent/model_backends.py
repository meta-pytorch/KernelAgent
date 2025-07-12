"""
Model backends for different LLM providers.
Provides a unified interface for OpenAI API, llama-cpp-python, and other model providers.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model backend.

        Args:
            model_name: Name/identifier of the model
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            Generated completion text
        """
        pass

    @abstractmethod
    def generate_multiple_completions(self, prompt: str, n: int, **kwargs) -> List[str]:
        """
        Generate multiple completions for the given prompt.

        Args:
            prompt: Input prompt text
            n: Number of completions to generate
            **kwargs: Generation parameters

        Returns:
            List of generated completion texts
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and properly configured."""
        pass


class OpenAIBackend(ModelBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        high_reasoning_effort: bool = True,
        proxy_config: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI backend.

        Args:
            model_name: OpenAI model name (e.g., 'o3-2025-04-16')
            api_key: OpenAI API key
            high_reasoning_effort: Whether to use high reasoning effort
            proxy_config: Proxy configuration for Meta environments
            **kwargs: Additional OpenAI client parameters
        """
        super().__init__(model_name, **kwargs)

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.high_reasoning_effort = high_reasoning_effort
        self.proxy_config = proxy_config

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError("OpenAI API key not provided")

        # Setup proxy if provided
        if proxy_config:
            self.logger.info(
                f"Using proxy: {proxy_config.get('https_proxy', proxy_config.get('http_proxy'))}"
            )
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                proxy_url = proxy_config.get("https_proxy") or proxy_config.get(
                    "http_proxy"
                )
                if proxy_url:
                    os.environ[key] = proxy_url

        try:
            self.client = OpenAI(api_key=self.api_key, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate a single completion using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
            
        try:
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "max_completion_tokens": kwargs.get("max_tokens", 32000),
            }

            # Add reasoning effort for supported models
            if self.high_reasoning_effort and "o3" in self.model_name.lower():
                api_params["reasoning_effort"] = "high"

            # Add other parameters
            if "temperature" in kwargs:
                api_params["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                api_params["top_p"] = kwargs["top_p"]

            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def generate_multiple_completions(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple completions using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
            
        try:
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "n": n,
                "max_completion_tokens": kwargs.get("max_tokens", 32000),
            }

            # Add reasoning effort for supported models
            if self.high_reasoning_effort and "o3" in self.model_name.lower():
                api_params["reasoning_effort"] = "high"

            # Add other parameters
            if "temperature" in kwargs:
                api_params["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                api_params["top_p"] = kwargs["top_p"]

            response = self.client.chat.completions.create(**api_params)
            return [choice.message.content for choice in response.choices]

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI backend is available."""
        return bool(OPENAI_AVAILABLE and self.api_key and self.api_key != 'your-api-key-here')


class LlamaCppBackend(ModelBackend):
    """llama-cpp-python backend for local model inference."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize llama-cpp-python backend.

        Args:
            model_name: Model identifier (for logging/reference)
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Enable verbose logging
            **kwargs: Additional llama-cpp parameters
        """
        super().__init__(model_name, **kwargs)

        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not available. Install with: pip install llama-cpp-python"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        # Initialize the model
        try:
            self.logger.info(f"Loading model from {model_path}")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                **kwargs,
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt for the specific model.
        This can be customized for different model formats (ChatML, Alpaca, etc.)
        """
        # For DeepSeek R1, use a simple format
        # You can customize this based on the model's expected format
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate a single completion using llama-cpp-python."""
        if not hasattr(self, 'llm') or self.llm is None:
            raise RuntimeError("llama-cpp model not initialized")
            
        try:
            formatted_prompt = self._format_prompt(prompt)

            # Set default parameters
            generation_params = {
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stop": kwargs.get("stop", ["<|user|>", "<|end|>"]),
                "echo": False,
            }

            response = self.llm(formatted_prompt, **generation_params)
            return response["choices"][0]["text"].strip()

        except Exception as e:
            self.logger.error(f"llama-cpp generation error: {e}")
            raise

    def generate_multiple_completions(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple completions using llama-cpp-python."""
        completions = []

        # Generate completions one by one (llama-cpp doesn't support batch generation like OpenAI)
        for i in range(n):
            try:
                # Add some randomness for diversity
                temp_kwargs = kwargs.copy()
                temp_kwargs["temperature"] = kwargs.get("temperature", 0.7) + (i * 0.1)

                completion = self.generate_completion(prompt, **temp_kwargs)
                completions.append(completion)

            except Exception as e:
                self.logger.error(f"Error generating completion {i+1}: {e}")
                # Continue with other completions
                continue

        return completions

    def is_available(self) -> bool:
        """Check if llama-cpp backend is available."""
        return LLAMA_CPP_AVAILABLE and self.model_path.exists()


class DeepSeekR1Backend(LlamaCppBackend):
    """Specialized backend for DeepSeek R1 models."""

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize DeepSeek R1 backend.

        Args:
            model_path: Path to DeepSeek R1 GGUF model file
            **kwargs: Additional parameters
        """
        super().__init__(
            model_name="deepseek-r1-0528",
            model_path=model_path,
            n_ctx=kwargs.get("n_ctx", 32768),  # DeepSeek R1 supports long context
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            **kwargs,
        )

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt specifically for DeepSeek R1."""
        # DeepSeek R1 uses a specific chat format
        return f"<｜begin▁of▁sentence｜>User: {prompt}\n\nAssistant: "


def create_model_backend(backend_type: str, **config) -> ModelBackend:
    """
    Factory function to create model backends.

    Args:
        backend_type: Type of backend ('openai', 'llama-cpp', 'deepseek-r1')
        **config: Backend-specific configuration

    Returns:
        Initialized model backend

    Raises:
        ValueError: If backend_type is not supported
        ImportError: If required dependencies are not available
    """
    backend_type = backend_type.lower()

    if backend_type == "openai":
        return OpenAIBackend(**config)
    elif backend_type == "llama-cpp":
        return LlamaCppBackend(**config)
    elif backend_type == "deepseek-r1":
        return DeepSeekR1Backend(**config)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_available_backends() -> Dict[str, bool]:
    """
    Get information about available backends.

    Returns:
        Dictionary mapping backend names to availability status
    """
    return {
        "openai": OPENAI_AVAILABLE,
        "llama-cpp": LLAMA_CPP_AVAILABLE,
        "deepseek-r1": LLAMA_CPP_AVAILABLE,  # DeepSeek R1 uses llama-cpp
    }


# Configuration helpers
def load_backend_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load backend configuration from file.

    Args:
        config_path: Path to configuration file (JSON format)

    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Default configuration
    return {
        "backend_type": os.getenv("MODEL_BACKEND", "openai"),
        "openai": {
            "model_name": os.getenv("OPENAI_MODEL", "o3-2025-04-16"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "high_reasoning_effort": True,
        },
        "llama-cpp": {
            "model_name": "llama-cpp-model",
            "model_path": os.getenv("LLAMA_CPP_MODEL_PATH", ""),
            "n_ctx": int(os.getenv("LLAMA_CPP_N_CTX", "32768")),
            "n_gpu_layers": int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "-1")),
        },
        "deepseek-r1": {
            "model_path": os.getenv("DEEPSEEK_R1_MODEL_PATH", ""),
            "n_ctx": int(os.getenv("DEEPSEEK_R1_N_CTX", "32768")),
            "n_gpu_layers": int(os.getenv("DEEPSEEK_R1_N_GPU_LAYERS", "-1")),
        },
    }
