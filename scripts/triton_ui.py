#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradio UI for Triton Kernel Agent."""

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv


from triton_kernel_agent import TritonKernelAgent
from utils.providers.models import get_available_models
from utils.providers.openai_provider import OpenAIProvider
from utils.providers.anthropic_provider import AnthropicProvider


KERNELBENCH_BASE_PATH = (
    Path(__file__).resolve().parent / "external" / "KernelBench" / "KernelBench"
)
KERNELBENCH_LEVEL_LABELS = {
    "level1": "Level 1",
    "level2": "Level 2",
}


def load_kernelbench_problem_map(
    levels: Tuple[str, ...] = ("level1", "level2"),
) -> Dict[str, Path]:
    problem_map: Dict[str, Path] = {}
    for level in levels:
        level_dir = KERNELBENCH_BASE_PATH / level
        if not level_dir.is_dir():
            continue

        level_label = KERNELBENCH_LEVEL_LABELS.get(level, level.title())
        for path in sorted(level_dir.glob("*.py")):
            problem_name = path.stem
            label = f"{level_label} · {problem_name}"
            if label in problem_map:
                label = f"{label} ({path.name})"
            problem_map[label] = path

    return problem_map


class TritonKernelUI:
    """Gradio UI wrapper for TritonKernelAgent"""

    def __init__(self):
        """Initialize the UI"""
        load_dotenv()
        self.agent = None
        self.last_result = None
        # Build a mapping from model name -> provider class for quick lookup
        self._model_to_provider = {
            cfg.name: cfg.provider_class for cfg in get_available_models()
        }

    def _provider_env_var_for_model(self, model_name: str) -> str:
        """Return the correct API key env var name for the selected model."""
        provider_cls = self._model_to_provider.get(model_name)
        if provider_cls is OpenAIProvider:
            return "OPENAI_API_KEY"
        if provider_cls is AnthropicProvider:
            return "ANTHROPIC_API_KEY"
        # Relay provider or unknown default to no key required
        return ""

    def generate_kernel(
        self,
        problem_description: str,
        test_code: Optional[str] = None,
        model_name: str = "o3-2025-04-16",
        high_reasoning_effort: bool = True,
        user_api_key: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Generate a Triton kernel based on the problem description

        Args:
            problem_description: Description of the kernel to generate
            test_code: Optional custom test code
            model_name: OpenAI model to use
            high_reasoning_effort: Whether to use high reasoning effort
            user_api_key: Optional OpenAI API key (not saved, used only for this session)

        Returns:
            - status: Success/failure message
            - kernel_code: Generated kernel code
            - test_code: Generated or provided test code
            - logs: Generation logs and metrics
            - session_info: Session details
            - download_links: Links to generated files
        """
        if not problem_description.strip():
            status = "❌ Please provide a problem description."
            return status, "", "", "", "", ""

        # Determine provider-specific API key env var based on selected model
        key_env_var = self._provider_env_var_for_model(model_name)
        api_key = user_api_key.strip() if user_api_key else None
        env_api_key = os.getenv(key_env_var) if key_env_var else ""

        # For providers that require a key, check availability
        if key_env_var and not (api_key or env_api_key):
            provider_label = (
                "OpenAI" if key_env_var == "OPENAI_API_KEY" else "Anthropic"
            )
            status = f"❌ Please provide a {provider_label} API key or set {key_env_var} in your environment/.env."
            return status, "", "", "", "", ""

        try:
            # Create agent with selected model and reasoning effort
            start_time = time.time()

            # Temporarily set provider-specific API key if provided by user (session-only)
            original_env_key = None
            if api_key and key_env_var:
                original_env_key = os.environ.get(key_env_var)
                os.environ[key_env_var] = api_key

            agent = TritonKernelAgent(
                model_name=model_name, high_reasoning_effort=high_reasoning_effort
            )

            # If provider failed to initialize, return a clear error immediately
            if not getattr(agent, "provider", None):
                provider_label = (
                    ("OpenAI" if key_env_var == "OPENAI_API_KEY" else "Anthropic")
                    if key_env_var
                    else "Relay"
                )
                details = getattr(agent, "_provider_error", "Provider unavailable")
                status = (
                    f"❌ Provider initialization failed for {provider_label}: {details}"
                )
                return status, "", "", "", "", ""

            # Use provided test code or let agent generate it
            test_input = test_code.strip() if test_code and test_code.strip() else None

            # Generate kernel
            result = agent.generate_kernel(
                problem_description=problem_description, test_code=test_input
            )

            generation_time = time.time() - start_time

            # Store result for potential download
            self.last_result = result

            # Format response
            if result["success"]:
                status = f"✅ **SUCCESS!** Generated kernel in {generation_time:.2f}s"

                kernel_code = result["kernel_code"]

                # Read the generated test code
                test_file_path = os.path.join(result["session_dir"], "test.py")
                try:
                    with open(test_file_path, "r") as f:
                        generated_test = f.read()
                except (FileNotFoundError, IOError):
                    generated_test = "Test code not available"

                logs = self._format_logs(result, generation_time)
                session_info = self._format_session_info(result)
                download_links = self._create_download_info(result)

                return (
                    status,
                    kernel_code,
                    generated_test,
                    logs,
                    session_info,
                    download_links,
                )

            else:
                status = (
                    f"❌ **FAILED** after {generation_time:.2f}s: {result['message']}"
                )
                logs = self._format_error_logs(result, generation_time)
                session_info = self._format_session_info(result)

                return status, "", "", logs, session_info, ""

        except Exception as e:
            error_msg = (
                f"❌ **ERROR**: {str(e)}\n\n**Traceback:**\n{traceback.format_exc()}"
            )
            return error_msg, "", "", "", "", ""
        finally:
            # Clean up: restore original API key environment variable
            if api_key and key_env_var:
                if original_env_key is not None:
                    os.environ[key_env_var] = original_env_key
                else:
                    # Remove the key if it wasn't set originally
                    if key_env_var in os.environ:
                        del os.environ[key_env_var]

    def _format_logs(self, result: Dict[str, Any], generation_time: float) -> str:
        """Format generation logs for display"""
        logs = f"""## Generation Summary

**⏱️ Time:** {generation_time:.2f} seconds
**🏆 Winner:** Worker {result["worker_id"]}
**🔄 Rounds:** {result["rounds"]} refinement rounds
**📁 Session:** `{os.path.basename(result["session_dir"])}`

## Performance Metrics

- **Successful Worker:** {result["worker_id"]}
- **Refinement Iterations:** {result["rounds"]}
- **Total Generation Time:** {generation_time:.2f}s
- **Average Time per Round:** {generation_time / max(result["rounds"], 1):.2f}s

## Next Steps

1. Review the generated kernel code above
2. Test the kernel with your specific use case
3. Download files for further development
4. Modify parameters and re-generate if needed
"""
        return logs

    def _format_error_logs(self, result: Dict[str, Any], generation_time: float) -> str:
        """Format error logs for display"""
        logs = f"""## Generation Failed

**⏱️ Time:** {generation_time:.2f} seconds  
**❌ Error:** {result["message"]}
**📁 Session:** `{os.path.basename(result["session_dir"])}`

## Troubleshooting

1. **Check Problem Description:** Ensure it's clear and specific
2. **Review Test Code:** If provided, make sure it's valid Python
3. **Check Environment:** Verify OpenAI API key and model access
4. **Try Simpler Problem:** Start with basic operations

## Debug Information

- Session Directory: `{result["session_dir"]}`
- Check agent logs in the session directory for detailed error information
"""
        return logs

    def _format_session_info(self, result: Dict[str, Any]) -> str:
        """Format session information"""
        session_path = result["session_dir"]
        session_name = os.path.basename(session_path)

        info = f"""## Session Information

**📁 Session Directory:** `{session_name}`
**🕐 Timestamp:** {session_name.split("_")[-1] if "_" in session_name else "Unknown"}
**📂 Full Path:** `{session_path}`

## Generated Files

The following files were created in the session directory:

- `problem.txt` - Original problem description
- `test.py` - Generated or provided test code
- `seed_*.py` - Initial kernel variations
- `final_kernel.py` - Final successful kernel (if any)
- `result.json` - Complete generation results
- `workers/` - Individual worker logs and attempts

## Accessing Files

You can find all generated files in:
```
{session_path}
```
"""
        return info

    def _create_download_info(self, result: Dict[str, Any]) -> str:
        """Create download information"""
        if not result["success"]:
            return ""

        session_dir = result["session_dir"]

        download_info = f"""## 📥 Download Generated Files

### Main Files
- **Kernel Code:** `{session_dir}/final_kernel.py`
- **Test Code:** `{session_dir}/test.py`
- **Results:** `{session_dir}/result.json`

### All Files Location
```bash
# Copy all generated files
cp -r {session_dir} ./my_kernel_session/
```

### Quick Start
```python
# Use the generated kernel
from my_kernel_session.final_kernel import *

# Run the test
cd my_kernel_session && python test.py
```
"""
        return download_info


def _create_app() -> gr.Blocks:
    """Create and return the Gradio interface (without launching)"""
    ui = TritonKernelUI()
    # Load KernelBench problems
    kernelbench_problem_map = load_kernelbench_problem_map()

    # Add external problems (manual entries)
    extra_problem_map: Dict[str, Path] = {}
    try:
        external_cf = Path(
            "/home/leyuan/workplace/kernel_fuser/external/control_flow.py"
        )
        if external_cf.exists():
            extra_problem_map["External · control_flow"] = external_cf
        else:
            # Fallback to repo-relative external/control_flow.py
            repo_rel_cf = (
                Path(__file__).resolve().parent.parent / "external" / "control_flow.py"
            )
            if repo_rel_cf.exists():
                extra_problem_map["External · control_flow"] = repo_rel_cf
    except Exception:
        # Ignore issues probing external paths
        pass

    # Combine: external first, then KernelBench
    combined_problem_map: Dict[str, Path] = {
        **extra_problem_map,
        **kernelbench_problem_map,
    }
    problem_choices = list(combined_problem_map.keys())
    default_problem_choice = problem_choices[0] if problem_choices else None
    problem_cache: Dict[str, str] = {}

    if default_problem_choice:
        try:
            problem_cache[default_problem_choice] = combined_problem_map[
                default_problem_choice
            ].read_text(encoding="utf-8")
        except OSError:
            problem_cache[default_problem_choice] = ""

    # Create Gradio interface
    with gr.Blocks(
        title="Triton Kernel Agent",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .code-block { font-family: 'Monaco', 'Consolas', monospace; }
        .success { color: #22c55e; }
        .error { color: #ef4444; }
        """,
    ) as app:
        gr.Markdown(
            """
        # 🚀 Triton Kernel Agent
        
        **AI-Powered GPU Kernel Generation**
        
        Generate optimized OpenAI Triton kernels from high-level descriptions.
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📝 Configuration")

                # API Key input (optional, not saved)
                api_key_input = gr.Textbox(
                    label="🔑 LLM API Key (Optional)",
                    placeholder="Paste your API key here, or set it in .env",
                    type="password",
                    interactive=True,
                    info=(
                        "⚠️ Not saved — session-only. Will use the correct env var based on the selected model."
                    ),
                )

                CLAUDE_SONNET_4_5_MODEL_NAME = "claude-sonnet-4-5-20250929"
                CLAUDE_SONNET_4_5_LABEL = "Claude Sonnet 4.5"
                choices = []
                for config in get_available_models():
                    label = config.description
                    if config.name == CLAUDE_SONNET_4_5_MODEL_NAME:
                        label = CLAUDE_SONNET_4_5_LABEL  # Shorten the Anthropic label for clarity
                    choices.append((label, config.name))
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    label="🤖 Model Selection",
                    value=choices[0][1],
                    interactive=True,
                )

                # High reasoning effort checkbox
                high_reasoning_effort_checkbox = gr.Checkbox(
                    label="🧠 High Reasoning Effort",
                    value=True,
                    info="Use high reasoning effort for better quality (o4-mini and o3 series only)",
                    interactive=True,
                )

                gr.Markdown("## 🧩 Problem Library")

                problem_dropdown = gr.Dropdown(
                    choices=problem_choices,
                    label="Problem selection",
                    value=default_problem_choice,
                    interactive=bool(problem_choices),
                    info=(
                        "Select a KernelBench problem or an External problem to auto-fill the descriptor below."
                        if problem_choices
                        else (
                            f"No KernelBench problems found at {KERNELBENCH_BASE_PATH} and no external problems detected"
                        )
                    ),
                    allow_custom_value=False,
                )

                # Problem description input
                problem_input = gr.Textbox(
                    label="Override problem descriptor (optional)",
                    placeholder="Select a KernelBench problem above or paste your own problem descriptor...",
                    lines=10,
                    max_lines=20,
                    value=problem_cache.get(default_problem_choice, ""),
                    info="Editing this field overrides the selected descriptor.",
                )

                # Optional test code
                gr.Markdown("### 🧪 Test Code (Optional)")
                gr.Markdown("*Leave empty to auto-generate test code*")
                test_input = gr.Textbox(
                    label="Custom test code",
                    placeholder="# Optional: Provide custom test code\n# If empty, test will be auto-generated",
                    lines=8,
                    max_lines=15,
                )

                # Generate button
                generate_btn = gr.Button(
                    "🚀 Generate Kernel", variant="primary", size="lg"
                )

            with gr.Column(scale=3):
                gr.Markdown("## 📊 Results")

                # Status display
                status_output = gr.Markdown(
                    label="Status", value="*Ready to generate kernels...*"
                )

                # Generated kernel code
                with gr.Tab("🔧 Generated Kernel"):
                    kernel_output = gr.Code(
                        label="Kernel Code",
                        language="python",
                        interactive=False,
                        lines=20,
                    )

                # Generated test code
                with gr.Tab("🧪 Test Code"):
                    test_output = gr.Code(
                        label="Test Code",
                        language="python",
                        interactive=False,
                        lines=15,
                    )

                # Logs and metrics
                with gr.Tab("📈 Generation Logs"):
                    logs_output = gr.Markdown(
                        label="Logs", value="*No generation logs yet...*"
                    )

                # Session information
                with gr.Tab("📁 Session Info"):
                    session_output = gr.Markdown(
                        label="Session Details", value="*No session information yet...*"
                    )

                # Download information
                with gr.Tab("📥 Downloads"):
                    download_output = gr.Markdown(
                        label="Download Information",
                        value="*Generate a kernel to see download options...*",
                    )

        # Event handlers

        def update_problem_descriptor(selection: Optional[str]):
            if not selection:
                return gr.update()

            path = combined_problem_map.get(selection)
            if not path or not path.exists():
                return gr.update(
                    value=f"# Unable to load descriptor for {selection}\n\nMissing file: {path}"
                )

            if selection not in problem_cache:
                try:
                    problem_cache[selection] = path.read_text(encoding="utf-8")
                except OSError as exc:
                    return gr.update(value=f"# Error loading {selection}\n\n{exc}")

            return gr.update(value=problem_cache[selection])

        def generate_with_status(
            problem_desc, test_code, model_name, high_reasoning_effort, user_api_key
        ):
            """Wrapper for generate_kernel with status updates"""
            try:
                return ui.generate_kernel(
                    problem_desc,
                    test_code,
                    model_name,
                    high_reasoning_effort,
                    user_api_key,
                )
            except Exception as e:
                error_msg = f"❌ **UI ERROR**: {str(e)}\n\n**Traceback:**\n{traceback.format_exc()}"
                return error_msg, "", "", "", "", ""

        # Wire up events
        # Update API key input hint based on selected model/provider
        def update_api_key_hint(selected_model_name: Optional[str]):
            if not selected_model_name:
                return gr.update()
            key_env_var = ui._provider_env_var_for_model(selected_model_name)
            if key_env_var == "OPENAI_API_KEY":
                return gr.update(
                    label="🔑 OpenAI API Key (Optional)",
                    placeholder="sk-... (or set OPENAI_API_KEY)",
                    info=(
                        "Used only for this session. You can also set OPENAI_API_KEY in your environment or .env."
                    ),
                )
            elif key_env_var == "ANTHROPIC_API_KEY":
                return gr.update(
                    label="🔑 Anthropic API Key (Optional)",
                    placeholder="sk-ant-... (or set ANTHROPIC_API_KEY)",
                    info=(
                        "Used only for this session. You can also set ANTHROPIC_API_KEY in your environment or .env."
                    ),
                )
            else:
                return gr.update(
                    label="🔑 API Key (Not required for Relay)",
                    placeholder="Relay provider uses local server; no key required.",
                    info=(
                        "Relay models use a local relay server. Ensure it’s running; no API key needed."
                    ),
                )

        model_dropdown.change(
            fn=update_api_key_hint,
            inputs=model_dropdown,
            outputs=api_key_input,
        )

        if problem_choices:
            problem_dropdown.change(
                fn=update_problem_descriptor,
                inputs=problem_dropdown,
                outputs=problem_input,
            )

            def handle_problem_select(evt: gr.SelectData):
                return update_problem_descriptor(evt.value)

            problem_dropdown.select(
                fn=handle_problem_select,
                inputs=None,
                outputs=problem_input,
            )

        generate_btn.click(
            fn=generate_with_status,
            inputs=[
                problem_input,
                test_input,
                model_dropdown,
                high_reasoning_effort_checkbox,
                api_key_input,
            ],
            outputs=[
                status_output,
                kernel_output,
                test_output,
                logs_output,
                session_output,
                download_output,
            ],
            show_progress=True,
        )

        # Footer
        gr.Markdown(
            """
        ---
        
        **💡 Tips:**
        - Be specific about input/output shapes and data types
        - Include PyTorch equivalent code for reference  
        - Check the logs for detailed generation information
        
        **🔧 Configuration:** 
        - Provide your OpenAI or Anthropic API key above (not saved; session-only)
        - Or set the appropriate env var in `.env` (OPENAI_API_KEY or ANTHROPIC_API_KEY)
        - The key is only used for this session and automatically cleared
        """
        )

    return app


def main():
    """Create and launch the Gradio interface"""
    parser = argparse.ArgumentParser(description="Triton Kernel Agent UI")
    parser.add_argument("--port", type=int, default=8085, help="Port to run the UI on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()

    app = _create_app()

    # Check if running on Meta devserver (has Meta SSL certs)
    meta_keyfile = "/var/facebook/x509_identities/server.pem"
    is_meta_devserver = os.path.exists(meta_keyfile)

    print("🚀 Starting Triton Kernel Agent UI...")
    print("📝 Provide your OpenAI API key in the UI or configure in .env file")

    if is_meta_devserver:
        # Meta devserver configuration
        server_name = os.uname()[1]  # Get devserver hostname
        print(f"🌐 Opening on Meta devserver: https://{server_name}:{args.port}/")
        print("💡 Make sure you're connected to Meta VPN to access the demo")

        app.launch(
            share=False,
            show_error=True,
            server_name=server_name,
            server_port=args.port,
            ssl_keyfile=meta_keyfile,
            ssl_certfile=meta_keyfile,
            ssl_verify=False,
            inbrowser=False,  # Don't auto-open browser on remote server
        )
    else:
        # Local development configuration
        print(f"🌐 Opening locally: http://{args.host}:{args.port}/")
        print(
            f"🚨 IMPORTANT: If Chrome shows blank page, try Safari: open -a Safari http://{args.host}:{args.port}/ 🚨"
        )

        app.launch(
            share=False,
            show_error=True,
            server_name=args.host,
            server_port=args.port,
            inbrowser=True,  # Auto-open browser for local development
        )


if __name__ == "__main__":
    main()
