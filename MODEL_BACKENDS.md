# Model Backends for KernelAgent

KernelAgent now supports multiple model backends, including OpenAI API, llama-cpp-python, and specialized support for DeepSeek R1.

## Supported Backends

### 1. OpenAI API (Default)
- Supports GPT-4, o3, and other OpenAI models
- Meta proxy configuration support
- High reasoning effort mode support

### 2. llama-cpp-python
- Supports local GGUF format models
- GPU acceleration support
- Configurable context window size

### 3. DeepSeek R1 (Specialized)
- Based on llama-cpp-python
- Optimized for DeepSeek R1 0528 models
- Specialized prompt formatting

## Installation Dependencies

### Basic Dependencies
```bash
pip install openai  # OpenAI API support
```

### llama-cpp-python Support
```bash
# CPU version
pip install llama-cpp-python

# GPU version (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# GPU version (Metal - macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## Configuration Methods

### Method 1: Environment Variable Configuration

```bash
# Select backend type
export MODEL_BACKEND=deepseek-r1

# DeepSeek R1 configuration
export DEEPSEEK_R1_MODEL_PATH=/path/to/deepseek-r1-0528.gguf
export DEEPSEEK_R1_N_CTX=32768
export DEEPSEEK_R1_N_GPU_LAYERS=-1

# OpenAI configuration (backup)
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=o3-2025-04-16
```

### Method 2: Configuration File

Create configuration file `config.json`:

```json
{
  "backend_type": "deepseek-r1",
  "deepseek-r1": {
    "model_path": "/path/to/deepseek-r1-0528.gguf",
    "n_ctx": 32768,
    "n_gpu_layers": -1,
    "verbose": false
  },
  "openai": {
    "model_name": "o3-2025-04-16",
    "api_key": "your-openai-api-key",
    "high_reasoning_effort": true
  }
}
```

### Method 3: Direct Configuration in Code

```python
from triton_kernel_agent import TritonKernelAgent

# DeepSeek R1 configuration
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "/path/to/deepseek-r1-0528.gguf",
        "n_ctx": 32768,
        "n_gpu_layers": -1
    }
}

agent = TritonKernelAgent(model_backend_config=config)
```

## Usage Examples

### Basic Usage

```python
from triton_kernel_agent import TritonKernelAgent

# Using configuration file
agent = TritonKernelAgent(backend_config_path="config.json")

# Or pass configuration directly
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "/path/to/deepseek-r1-0528.gguf"
    }
}
agent = TritonKernelAgent(model_backend_config=config)

# Generate kernel
result = agent.generate_kernel(
    problem_description="Implement an efficient matrix multiplication kernel"
)

if result["success"]:
    print("Generated kernel code:")
    print(result["kernel_code"])
else:
    print("Kernel generation failed:", result["message"])
```

### Usage in UI

Modify `triton_ui.py` to support model backend configuration:

```python
import streamlit as st
from triton_kernel_agent import TritonKernelAgent

# Add model configuration to sidebar
st.sidebar.header("Model Configuration")
backend_type = st.sidebar.selectbox(
    "Select Model Backend",
    ["openai", "deepseek-r1", "llama-cpp"]
)

if backend_type == "deepseek-r1":
    model_path = st.sidebar.text_input(
        "DeepSeek R1 Model Path",
        value="/path/to/deepseek-r1-0528.gguf"
    )
    config = {
        "backend_type": "deepseek-r1",
        "deepseek-r1": {
            "model_path": model_path,
            "n_ctx": 32768,
            "n_gpu_layers": -1
        }
    }
elif backend_type == "openai":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model_name = st.sidebar.text_input("Model Name", value="o3-2025-04-16")
    config = {
        "backend_type": "openai",
        "openai": {
            "api_key": api_key,
            "model_name": model_name,
            "high_reasoning_effort": True
        }
    }

# Create agent using configuration
agent = TritonKernelAgent(model_backend_config=config)
```

## DeepSeek R1 Model Acquisition

### 1. Download Models

Download DeepSeek R1 GGUF format models from Hugging Face:

```bash
# Using huggingface-hub
pip install huggingface-hub
huggingface-cli download deepseek-ai/DeepSeek-R1 --include "*.gguf" --local-dir ./models/
```

### 2. Model File Structure

```
models/
├── deepseek-r1-0528-q4_k_m.gguf    # 4-bit quantized version (recommended)
├── deepseek-r1-0528-q8_0.gguf      # 8-bit quantized version
└── deepseek-r1-0528-f16.gguf       # 16-bit version (highest quality)
```

### 3. Choosing Appropriate Quantization Version

- **q4_k_m**: 4-bit quantization, balanced performance and quality, recommended for daily use
- **q8_0**: 8-bit quantization, higher quality, requires more memory
- **f16**: 16-bit, highest quality, requires substantial memory and computational resources

## Performance Optimization

### GPU Configuration

```python
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "/path/to/model.gguf",
        "n_ctx": 32768,          # Context window size
        "n_gpu_layers": -1,      # -1 means all layers on GPU
        "n_threads": 8,          # CPU thread count
        "n_batch": 512,          # Batch size
        "verbose": False
    }
}
```

### Memory Optimization

```python
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "/path/to/model.gguf",
        "n_ctx": 16384,          # Reduce context window
        "n_gpu_layers": 20,      # Only put some layers on GPU
        "low_vram": True,        # Low VRAM mode
        "mmap": True,            # Use memory mapping
        "mlock": False           # Don't lock memory
    }
}
```

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   FileNotFoundError: Model file not found: /path/to/model.gguf
   ```
   Solution: Check if the model file path is correct

2. **Out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce `n_gpu_layers` or use smaller quantized versions

3. **llama-cpp-python not installed**
   ```
   ImportError: llama-cpp-python not available
   ```
   Solution: Install llama-cpp-python

### Debug Mode

Enable verbose logging:

```python
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "/path/to/model.gguf",
        "verbose": True  # Enable verbose logging
    }
}
```

### Backend Availability Check

```python
from triton_kernel_agent.model_backends import get_available_backends

# Check available backends
backends = get_available_backends()
print("Available backends:", backends)

# Check if specific backend is available
if backends['deepseek-r1']:
    print("DeepSeek R1 backend is available")
else:
    print("DeepSeek R1 backend not available, please install llama-cpp-python")
```

## Extensibility

### Adding New Model Backends

1. Inherit from `ModelBackend` base class
2. Implement necessary methods
3. Register new backend in factory function

```python
from triton_kernel_agent.model_backends import ModelBackend

class CustomBackend(ModelBackend):
    def generate_completion(self, prompt: str, **kwargs) -> str:
        # Implement your model calling logic
        pass

    def generate_multiple_completions(self, prompt: str, n: int, **kwargs) -> List[str]:
        # Implement multiple completion generation
        pass

    def is_available(self) -> bool:
        # Check if backend is available
        return True
```

### Custom Prompt Formatting

For special model format requirements, you can override the `_format_prompt` method:

```python
class CustomDeepSeekBackend(DeepSeekR1Backend):
    def _format_prompt(self, prompt: str) -> str:
        # Custom prompt formatting
        return f"<custom_format>{prompt}</custom_format>"
```

This extensible design makes adding new model backends simple while maintaining backward compatibility.
