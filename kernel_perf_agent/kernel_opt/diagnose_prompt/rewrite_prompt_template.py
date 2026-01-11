"""Rewrite Prompt template."""

REWRITE_PROMPT_TEMPLATE = """
You are a professional performance engineer who is an expert in rewriting {dsl} kernels to improve their performance.

Your task is to rewrite the following {dsl} kernel to integrate the specific optimization.
The kernel name is {kernel_name}.
The function of this kernel is {func_prompt}.
The kernel source code is:
{input_kernel}

The required optimization to integrate is:
{opt_prompt}

Here are the necessary context about the specific optimization:
{context}

IMPORTANT:
1. Rewrite the given kernel at {dsl} level.
2. Generate the complete implementation that contains both the host code and the kernel code.
3. Please use markdown formatting (like ```python) in your output to wrap the code that you generate.
"""
