# DeepSeek R1 + llama-cpp-python Quick Start Guide

This guide will help you quickly use llama-cpp-python to load DeepSeek R1 models for generating Triton kernels.

## 1. Install Dependencies

### Basic Dependencies
```bash
pip install llama-cpp-python huggingface-hub
```

### GPU Accelerated Version (Recommended)

**CUDA (NVIDIA GPU):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Metal (Apple Silicon Mac):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## 2. Download DeepSeek R1 Models

Choose the model version that fits your hardware:

### Small Model (1.5B - Recommended for Getting Started)
```bash
# ~1GB, suitable for testing
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
  deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf --local-dir ./models/
```

### Medium Model (7B - Balanced Performance)
```bash
# ~4.5GB, better performance
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  deepseek-r1-distill-qwen-7b-q4_k_m.gguf --local-dir ./models/
```

### Large Model (14B - Best Performance)
```bash
# ~8GB, best performance
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF \
  deepseek-r1-distill-qwen-14b-q4_k_m.gguf --local-dir ./models/
```

## 3. Set Environment Variables

```bash
# Set model path (adjust according to your downloaded model)
export DEEPSEEK_R1_MODEL_PATH="./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"

# Optional: Set other parameters
export DEEPSEEK_R1_N_CTX=8192        # Context window size
export DEEPSEEK_R1_N_GPU_LAYERS=-1   # GPU layers (-1 means all)
```

## 4. Quick Test

### Method 1: Direct llama-cpp-python Usage

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # Use all GPU layers
    verbose=False
)

# Test inference
prompt = "Implement matrix multiplication in Python:"
formatted_prompt = f"<｜begin▁of▁sentence｜>User: {prompt}\n\nAssistant: "

response = llm(
    formatted_prompt,
    max_tokens=512,
    temperature=0.7,
    stop=["<｜end▁of▁sentence｜>", "User:"]
)

print(response['choices'][0]['text'])
```

### Method 2: Using KernelAgent Backend

```python
from triton_kernel_agent import TritonKernelAgent

# Configure DeepSeek R1 backend
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": -1
    }
}

# Create agent
agent = TritonKernelAgent(model_backend_config=config)

# Generate kernel
result = agent.generate_kernel(
    problem_description="Implement a vector addition kernel"
)

if result["success"]:
    print("Generated kernel code:")
    print(result["kernel_code"])
else:
    print("Generation failed:", result["message"])
```

## 5. Run Test Script

We provide a complete test script:

```bash
# Run test script
python examples/test_deepseek_r1_loading.py
```

This script will:
1. Check if dependencies are correctly installed
2. Test direct llama-cpp-python usage
3. Test KernelAgent backend integration
4. Optional: Test complete kernel generation workflow

## 6. Configuration File Approach

Create configuration file `deepseek_config.json`:

```json
{
  "backend_type": "deepseek-r1",
  "deepseek-r1": {
    "model_path": "./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "verbose": false
  }
}
```

Then use the configuration file:

```python
from triton_kernel_agent import TritonKernelAgent

agent = TritonKernelAgent(backend_config_path="deepseek_config.json")
result = agent.generate_kernel("Implement matrix multiplication kernel")
```

## 7. Performance Optimization Tips

### Memory Optimization
```python
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "your-model.gguf",
        "n_ctx": 4096,          # Reduce context window
        "n_gpu_layers": 20,     # Only put some layers on GPU
        "n_threads": 4,         # Limit CPU threads
        "n_batch": 256          # Reduce batch size
    }
}
```

### GPU Optimization
```python
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": "your-model.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": -1,     # All layers on GPU
        "n_threads": 8,         # Increase CPU threads
        "n_batch": 512          # Increase batch size
    }
}
```

## 8. Common Issues

### Q: Model loading is very slow?
A: This is normal, first-time loading takes time. You can:
- Use smaller models (1.5B)
- Reduce `n_gpu_layers` parameter
- Store model files on SSD

### Q: Out of memory?
A:
- Use smaller quantized versions (q4_k_m instead of f16)
- Reduce `n_ctx` context window size
- Reduce `n_gpu_layers` parameter

### Q: Poor generation quality?
A:
- Use larger models (7B or 14B)
- Adjust `temperature` parameter (0.7-0.9)
- Optimize prompt formatting

### Q: How to check if GPU is being used?
A: Set `verbose=True` to see detailed logs, or use system monitoring tools:
```bash
# NVIDIA GPU
nvidia-smi

# Apple Silicon
sudo powermetrics --samplers gpu_power -n 1
```

## 9. Complete Example

```python
#!/usr/bin/env python3
import os
from triton_kernel_agent import TritonKernelAgent

# Set model path
model_path = "./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Model file does not exist: {model_path}")
    print("Please download the model file first")
    exit(1)

# Configure DeepSeek R1 backend
config = {
    "backend_type": "deepseek-r1",
    "deepseek-r1": {
        "model_path": model_path,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "verbose": True  # Show detailed logs
    }
}

# Create agent
print("Initializing TritonKernelAgent...")
agent = TritonKernelAgent(
    model_backend_config=config,
    num_workers=2,
    max_rounds=5
)

# Generate kernels
problems = [
    "Implement a vector addition kernel",
    "Implement a matrix transpose kernel",
    "Implement a ReLU activation function kernel"
]

for problem in problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print('='*60)

    result = agent.generate_kernel(problem_description=problem)

    if result["success"]:
        print("✓ Generation successful!")
        print(f"Worker process: {result['worker_id']}")
        print(f"Optimization rounds: {result['rounds']}")
        print(f"Session directory: {result['session_dir']}")
        print("\nKernel code preview:")
        print("-" * 40)
        code_preview = result["kernel_code"][:500]
        print(code_preview + "..." if len(result["kernel_code"]) > 500 else code_preview)
    else:
        print("✗ Generation failed")
        print(f"Error: {result.get('message', 'Unknown error')}")

print("\nAll tests completed!")
```

## 10. Next Steps

- Check `MODEL_BACKENDS.md` for more configuration options
- Run `examples/deepseek_r1_example.py` for more examples
- Check generated session directories for detailed generation process
- Adjust configuration parameters according to your hardware for optimal performance

Now you can use DeepSeek R1 + llama-cpp-python to generate Triton kernels!
