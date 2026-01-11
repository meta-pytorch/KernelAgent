"""Prompt management."""

import ast
import inspect
from pathlib import Path
from typing import Callable, Tuple

from kernel_perf_agent.kernel_opt.database.base import OptHierarchy, OptNode
from kernel_perf_agent.kernel_opt.prompts.rewrite_prompt_template import (
    REWRITE_PROMPT_TEMPLATE,
)
from kernel_perf_agent.kernel_opt.utils.parser_util import (
    get_module_path,
    remove_decorators_from_file,
)


class PromptManager:
    """Manages prompt construction."""

    def __init__(
        self,
        func_source_code: str,
        func_prompt: str,
        opt_prompt: str,
        model: str,
        dsl: str,
        kernel_name: str,
        database: OptHierarchy,
        opt_node: OptNode,
        module_path: Path,
        debug: bool,
    ):
        """Initialize prompt manager.
        :param func: Function to optimize
        :param func_prompt: Function prompt
        :param opt_prompt: Optimization prompt
        :param model: LLM model to use
        :param dsl: Target DSL (e.g., "triton")
        :param kernel_name: Name of the kernel (defaults to function name)
        :param database: Knowledge database of kernel optimizations
        :param opt_node: The most relevant optimization node in database
        :param module_path: Path to the module containing the function
        :param debug: Whether to print debug information
        """

        self.func_source_code = func_source_code
        self.func_prompt = func_prompt
        self.opt_prompt = opt_prompt
        self.model = model
        self.dsl = dsl
        self.kernel_name = kernel_name
        self.database = database
        self.opt_node = opt_node
        self.module_path = module_path
        self.debug = debug

    def build_rewrite_prompt(self) -> Tuple[str, str]:
        """Build rewrite prompt."""

        # Get context by traversing opt_node to all leaf nodes
        context = ""
        leaf = False
        cur_level = [self.opt_node]
        while cur_level:
            child_level = []
            for node in cur_level:
                # Leaf nodes are code examples
                if not leaf and not node.opt_children:
                    leaf = True
                    context += """
Here are code examples before and after the optimization:
"""
                context += node.opt_desc
                for child in node.opt_children:
                    if child not in child_level:
                        child_level.append(child)
                cur_level = child_level

        debug_str = ""
        #         if self.debug:
        #             debug_str += f"""
        # ****** Context ****** :
        # {context}
        # """
        # if str(self.module_path) != "":
        #     debug_context_path = self.module_path / "debug_output" / "context.log"
        #     with open(str(debug_context_path), "w") as file:
        #         file.write(debug_str)
        #         # file.write("****** Context ****** : \n")
        #         # file.write(context)

        # Rewriting the kernels at the same DSL level as the input.
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            dsl=self.dsl,
            kernel_name=self.kernel_name,
            func_prompt=self.func_prompt,
            input_kernel=self.func_source_code,
            opt_prompt=self.opt_prompt,
            context=context,
        )

        if self.debug:
            debug_str += f"""
****** Prompt ****** :
{prompt}
"""
            # if str(self.module_path) != "":
            #     debug_prompt_path = self.module_path / "debug_output" / "prompt.log"
            #     with open(str(debug_prompt_path), "w") as file:
            #         file.write("****** Prompt ****** : \n")
            #         file.write(prompt)

        return prompt, debug_str
