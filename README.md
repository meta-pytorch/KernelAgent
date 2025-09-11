# KernelAgent 🚀

**Autonomous Triton Kernel Generation Agent**

KernelAgent (codename: Falcon) automatically generates optimized OpenAI Triton kernels from natural language descriptions using multiple parallel workers with iterative refinement. Successfully handles KernelBench tasks and real-world GPU operations.


## ✨ Key Features

- **🤖 Autonomous Generation**: Parallel workers with LLM-driven iterative refinement
- **🧠 Multi-turn Reasoning**: Workers learn from failed attempts using conversation history
- **📝 Triton Guidelines**: Built-in best practices for high-quality kernel generation
- **⚡ Parallel Architecture**: Multiple workers maximize success rate and speed
- **🌐 Web Interface**: Gradio UI for easy interaction
- **📊 Session Management**: Complete logging and artifact preservation

## 🏗️ How It Works

```
Natural Language → Test Generation → Seed Generation → Parallel Workers → Success!
     Description        (LLM)          (LLM n=4)        (4 workers)      (First to pass)
```

1. **Generate Test**: LLM creates test code from problem description
2. **Create Seeds**: Generate multiple initial kernel implementations
3. **Parallel Refinement**: Workers independently test and improve kernels
4. **Early Success**: First passing kernel stops all workers

## 🚀 Quick Start

### Prerequisites

```bash
# CUDA-enabled GPU
# Python 3.8+
# OpenAI API key
# Triton Nightly (for latest features)
```

### Installation

1. **Clone the repository**:
```bash
git clone git@github.com:pytorch-labs/KernelAgent.git
cd KernelAgent
```

2. **Create a virtual environment and install**:

```bash
pip install -e .         # Basic installation
pip install -e ".[dev]"  # With development dependencies
```

**Note**: Triton is not automatically installed. Install separately based on your system:

```bash
# For CUDA systems
pip install triton

# For development/latest features
pip install git+https://github.com/triton-lang/triton.git
```

3. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Basic Usage

#### Command Line Interface

```python
from triton_kernel_agent import TritonKernelAgent

# Initialize agent
agent = TritonKernelAgent()

# Generate kernel from description
result = agent.generate_kernel(
    problem_description="Implement a fused matrix multiplication with ReLU activation",
    test_code=None  # Auto-generate test
)

if result["success"]:
    print(f"Success! Generated kernel:\n{result['kernel_code']}")
else:
    print(f"Failed: {result['message']}")
```

#### Web Interface

```bash
python triton_ui.py
```

### Meta Dev GPU (VPN + Proxy)

If you are running on a Meta dev GPU host (detected by the presence of `/var/facebook/x509_identities/server.pem`), the UI launches with SSL bound to the devserver hostname. To ensure outbound access (e.g., OpenAI and Gradio internals) works behind the forward proxy, set proxy variables and bypass local/devserver hosts via NO_PROXY.

```bash
# In the 'kernelagent' conda environment (or prefix with `conda run -n kernelagent env ...`)

# 1) Forward proxy (lowercase + uppercase for compatibility)
export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# 2) Do not proxy local/devserver hostnames
export NO_PROXY="127.0.0.1,localhost,::1,$(hostname -f)"
export no_proxy="$NO_PROXY"

# 3) Start the UI (Meta devserver mode enables SSL automatically)
GRADIO_LAUNCH_BROWSER=false \
python triton_ui.py --port 8085

# Access from your laptop (VPN required):
#   https://$(hostname -f):8085/
```

Notes:
- The HTTPS endpoint requires Meta VPN and the devserver’s internal certificate; most corp browsers trust it.
- If your browser can’t reach the port, it’s likely a network policy rule; try a commonly open port (e.g., 8080/8000) or use SSH port forwarding.
- Avoid tunneling local checks (httpx/gradio) through the proxy by including the devserver hostname in NO_PROXY; otherwise you may see `ProxyError: 502 Failed to resolve host`.

### Example: KernelBench Level 1 Problem

```python
# Actual KernelBench Level 1 ReLU task (19_ReLU.py)
problem_description = """
import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    Simple model that performs a ReLU activation.
    \"\"\"
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        \"\"\"
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
"""

agent = TritonKernelAgent()
result = agent.generate_kernel(problem_description)
# Result: Optimized Triton ReLU kernel, fully autonomous generation!
```

## ⚙️ Configuration

Configure via environment variables in `.env`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=o3-2025-04-16

# Worker Configuration
NUM_KERNEL_SEEDS=4              # Number of parallel workers
MAX_REFINEMENT_ROUNDS=10        # Max iterations per worker

# Logging
LOG_LEVEL=INFO
```

### Advanced Configuration

```python
agent = TritonKernelAgent(
    num_workers=6,                    # More workers for complex problems
    max_rounds=15,                    # More refinement rounds
    model_name="o4-mini-2025-04-16",  # Different model
    high_reasoning_effort=True        # Enhanced reasoning (default)
)
```

## 📁 Project Structure

```
KernelAgent/
├── triton_kernel_agent/          # Core agent implementation
│   ├── agent.py                   # Main TritonKernelAgent class
│   ├── manager.py                 # WorkerManager for parallel execution
│   ├── worker.py                  # Individual VerificationWorker
│   ├── prompt_manager.py          # Jinja2 template management
│   └── triton_guidelines.py       # Triton programming guidelines
├── templates/                     # Jinja2 prompt templates
│   ├── kernel_generation.j2       # Initial kernel generation
│   ├── kernel_refinement.j2       # Error-based refinement
│   ├── test_generation.j2         # Test code generation
│   └── triton_guidelines.j2       # Triton best practices
├── tests/                         # Test suite
│   ├── __init__.py
│   └── test_basic.py              # Basic functionality tests
├── .github/workflows/             # CI/CD configuration
│   └── ci.yml                     # GitHub Actions workflow
├── triton_ui.py                   # Gradio web interface
├── e2e_test.py                    # End-to-end testing
├── pyproject.toml                 # Project configuration and dependencies
└── README.md                      # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=triton_kernel_agent

# Run end-to-end test
uv run python e2e_test.py
```

## 📊 Performance

- **Success Rate**: High success rate on fundamental GPU operations
- **Speed**: 3-4x faster than sequential approaches
- **Quality**: Production-ready, numerically-correct Triton kernels
- **Automation**: Fully autonomous with zero manual intervention

## 🎯 Use Cases

- **Research**: Rapid prototyping of GPU algorithms
- **Optimization**: Converting PyTorch ops to high-performance Triton
- **Education**: Learning Triton programming
- **Production**: Creating optimized kernels for inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Create a Pull Request

## 📋 Templates

Customize Jinja2 templates in `templates/`:
- `kernel_generation.j2` - Initial kernel generation
- `kernel_refinement.j2` - Error-based improvements
- `test_generation.j2` - Test code creation
- `triton_guidelines.j2` - Triton best practices

## 🔍 Troubleshooting

**Common Issues:**
- **OpenAI API Errors**: Check your API key and model access
- **CUDA Not Available**: Ensure CUDA-enabled GPU is accessible
- **Worker Timeouts**: Increase `MAX_REFINEMENT_ROUNDS`
- **Memory Issues**: Reduce `NUM_KERNEL_SEEDS`

**Debug Mode:**
```bash
export LOG_LEVEL=DEBUG
python your_script.py
```

## 📈 Roadmap

### 🔄 **Next Steps**
- Performance-tuning loop with profiling-guided search
- Advanced operations (attention, jagged tensors, fusion planning)
- Higher-level kernel challenges

### 🔮 **Future**
- [ ] Automated performance optimization
- [ ] Additional GPU backends (ROCm, MTIA, etc.)
- [ ] More LLM providers (Anthropic, local models)
- [ ] ML framework integration
- [ ] Production deployment tools
