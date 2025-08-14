import pytest
from unittest.mock import patch, MagicMock
import os
import sys

def test_imports():
    """Test that main modules can be imported."""
    import triton_kernel_agent
    from triton_kernel_agent import TritonKernelAgent
    from triton_kernel_agent.agent import TritonKernelAgent
    from triton_kernel_agent.manager import WorkerManager
    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.prompt_manager import PromptManager
    assert True

def test_triton_guidelines():
    """Test that triton guidelines can be loaded."""
    from triton_kernel_agent.triton_guidelines import TRITON_GUIDELINES
    assert isinstance(TRITON_GUIDELINES, str)
    assert len(TRITON_GUIDELINES) > 0
    assert "Triton" in TRITON_GUIDELINES

def test_prompt_manager_initialization():
    """Test PromptManager initialization."""
    from triton_kernel_agent.prompt_manager import PromptManager
    pm = PromptManager()
    assert pm.templates_dir.exists()
    # Check that templates can be loaded
    assert hasattr(pm, 'env')
    assert pm.env is not None

def test_worker_initialization():
    """Test VerificationWorker initialization without OpenAI."""
    from triton_kernel_agent.worker import VerificationWorker
    from pathlib import Path
    import tempfile
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "worker1"
        logdir = Path(tmpdir) / "logs"
        workdir.mkdir(exist_ok=True)
        logdir.mkdir(exist_ok=True)
        
        worker = VerificationWorker(
            worker_id=1,
            workdir=workdir,
            log_dir=logdir,
            max_rounds=5,
            openai_api_key=None  # No API key for test
        )
        assert worker.worker_id == 1
        assert worker.max_rounds == 5
        assert worker.workdir == workdir

def test_templates_exist():
    """Test that all required templates exist."""
    from pathlib import Path
    template_dir = Path(__file__).parent.parent / "templates"
    required_templates = [
        "kernel_generation.j2",
        "kernel_refinement.j2", 
        "test_generation.j2",
        "triton_guidelines.j2"
    ]
    for template in required_templates:
        template_path = template_dir / template
        assert template_path.exists(), f"Template {template} not found"

def test_environment_detection():
    """Test environment detection for CUDA availability."""
    import torch
    cuda_available = torch.cuda.is_available()
    if sys.platform == "darwin":
        assert not cuda_available, "CUDA should not be available on macOS"
    print(f"CUDA available: {cuda_available}")
    assert True