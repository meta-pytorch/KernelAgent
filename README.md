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

2. **Install dependencies**:

If you are using MacBook, please build and install Triton from source since triton cpu package is not available for macos.

```bash
# Install Triton from source
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt
pip install -e .
```

After Triton is installed, please go back to the KernelAgent directory and continue to install other dependencies.
| Note: for Devserver, `triton-nightly` will be installed automatically when you run the following command.

```bash
cd KernelAgent
# Install other dependencies
pip install -r requirements.txt
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

### Performance Profiling Configuration

Enable NCU profiler-guided optimization by setting environment variables:

```bash
# Enable performance profiling with NCU
ENABLE_PERFORMANCE_PROFILING=true

# Performance thresholds (percentages)
PERFORMANCE_THRESHOLD_SM=80.0          # SM throughput threshold
PERFORMANCE_THRESHOLD_DRAM=70.0        # DRAM throughput threshold

# Maximum rounds for performance optimization
MAX_PERFORMANCE_ROUNDS=3

# NCU metrics to collect
NCU_METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
```

Copy `performance_config.env` to `.env` or set these variables directly for profiler-guided optimization.

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
│   ├── kernel_performance_refinement.j2  # NCU-guided optimization
│   ├── test_generation.j2         # Test code generation
│   └── triton_guidelines.j2       # Triton best practices
├── triton_ui.py                   # Gradio web interface
├── e2e_test.py                    # End-to-end testing
└── README.md                      # This file
```

## 🧪 Testing

Run the end-to-end test:

```bash
python e2e_test.py
```

This will generate a cumsum kernel and verify it works correctly.

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
- `kernel_performance_refinement.j2` - NCU profiler-guided optimization
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
- ✅ Performance-tuning loop with NCU profiling-guided optimization
- Advanced operations (attention, jagged tensors, fusion planning)
- Higher-level kernel challenges

### 🔮 **Future**
- ✅ Automated performance optimization with NCU profiler integration
- [ ] Additional GPU backends (ROCm, MTIA, etc.)
- [ ] More LLM providers (Anthropic, local models)
- [ ] ML framework integration
- [ ] Production deployment tools
