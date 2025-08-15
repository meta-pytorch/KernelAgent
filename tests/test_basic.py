def test_imports():
    """Test that main modules can be imported."""
    import triton_kernel_agent
    from triton_kernel_agent.agent import TritonKernelAgent
    from triton_kernel_agent.manager import WorkerManager
    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.prompt_manager import PromptManager

    assert triton_kernel_agent is not None
    assert TritonKernelAgent is not None
    assert WorkerManager is not None
    assert VerificationWorker is not None
    assert PromptManager is not None