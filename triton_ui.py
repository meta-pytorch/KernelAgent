#!/usr/bin/env python3
"""
Gradio UI for Triton Kernel Agent
Interactive web interface for generating and testing Triton kernels
"""

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv


from triton_kernel_agent import TritonKernelAgent


class TritonKernelUI:
    """Gradio UI wrapper for TritonKernelAgent"""

    def __init__(self):
        """Initialize the UI"""
        load_dotenv()
        self.agent = None
        self.last_result = None

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

        # Check API key availability
        api_key = user_api_key.strip() if user_api_key else None
        env_api_key = os.getenv("OPENAI_API_KEY")

        if not api_key and not env_api_key:
            status = (
                "❌ Please provide an OpenAI API key or set OPENAI_API_KEY environment variable."
            )
            return status, "", "", "", "", ""

        try:
            # Create agent with selected model and reasoning effort
            start_time = time.time()

            # Temporarily set API key if provided by user (without saving to environment)
            original_env_key = None
            if api_key:
                original_env_key = os.environ.get("OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = api_key

            agent = TritonKernelAgent(
                model_name=model_name, high_reasoning_effort=high_reasoning_effort
            )

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
                status = f"❌ **FAILED** after {generation_time:.2f}s: {result['message']}"
                logs = self._format_error_logs(result, generation_time)
                session_info = self._format_session_info(result)

                return status, "", "", logs, session_info, ""

        except Exception as e:
            error_msg = f"❌ **ERROR**: {str(e)}\n\n**Traceback:**\n{traceback.format_exc()}"
            return error_msg, "", "", "", "", ""
        finally:
            # Clean up: restore original API key environment variable
            if api_key:
                if original_env_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_env_key
                else:
                    # Remove the key if it wasn't set originally
                    if "OPENAI_API_KEY" in os.environ:
                        del os.environ["OPENAI_API_KEY"]

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


def main():
    """Create and launch the Gradio interface"""

    # Create UI instance
    ui = TritonKernelUI()

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
                    label="🔑 OpenAI API Key (Optional)",
                    placeholder="sk-... (leave empty to use environment variable)",
                    type="password",
                    interactive=True,
                    info="⚠️ Not saved - only used for this session",
                )

                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=["o3-2025-04-16", "o4-mini-2025-04-16"],
                    label="🤖 Model Selection",
                    value="o3-2025-04-16",
                    interactive=True,
                )

                # High reasoning effort checkbox
                high_reasoning_effort_checkbox = gr.Checkbox(
                    label="🧠 High Reasoning Effort",
                    value=True,
                    info="Use high reasoning effort for better quality (o4-mini and o3 series only)",
                    interactive=True,
                )

                gr.Markdown("## 📝 Problem Description")

                # Problem description input
                problem_input = gr.Textbox(
                    label="Describe your kernel problem",
                    placeholder="Enter a clear description of the kernel you want to generate...",
                    lines=10,
                    max_lines=20,
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
                generate_btn = gr.Button("🚀 Generate Kernel", variant="primary", size="lg")

            with gr.Column(scale=3):
                gr.Markdown("## 📊 Results")

                # Status display
                status_output = gr.Markdown(label="Status", value="*Ready to generate kernels...*")

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
                    logs_output = gr.Markdown(label="Logs", value="*No generation logs yet...*")

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
        - Provide your OpenAI API key above (not saved, session-only)
        - Or set OPENAI_API_KEY environment variable in `.env` file
        - API key is only used for this session and automatically cleared
        """
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Kernel Agent UI")
    parser.add_argument("--port", type=int, default=8085, help="Port to run the UI on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()

    app = main()

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
            show_api=False,
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
            show_api=False,
            inbrowser=True,  # Auto-open browser for local development
        )
