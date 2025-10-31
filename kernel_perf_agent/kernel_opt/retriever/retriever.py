import os
import warnings
from pathlib import Path

import numpy as np
from kernel_perf_agent.kernel_opt.database.base import OptHierarchy
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings


class Retriever:
    """Retriever for kernel generation."""

    def __init__(
        self,
        func_prompt: str,
        opt_prompt: str,
        model: str,
        dsl: str,
        kernel_name: str,
        database: OptHierarchy,
        module_path: Path,
        debug: bool,
    ):
        """
        Initialize the retriever.
        :param func_prompt: Description of the kernel to generate
        :param opt_prompt: Description of the optimization to perform
        :param model: LLM model to use
        :param dsl: Target DSL (e.g., "triton")
        :param kernel_name: Name of the kernel (defaults to function name)
        :param database: Knowledge database of kernel optimizations
        :param module_path: Path to the module containing the function
        :param debug: Whether to print debug information
        """

        self.func_prompt = func_prompt
        self.opt_prompt = opt_prompt
        self.model = model
        self.dsl = dsl
        self.kernel_name = kernel_name
        self.database = database
        self.module_path = module_path
        self.debug = debug

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def retrieve(self):
        """Retrieve the relevant context in the database from the key (opt_prompt)."""
        embeddings = OpenAIEmbeddings()
        key = self.opt_prompt
        key_embedding = embeddings.embed_query(key)

        # Compute the similarity score for all nodes in the database tree
        root = self.database.get_root()
        cur_level = [root_child for root_child in root.opt_children]
        opt_similarity = dict()
        while cur_level:
            child_level = []
            for node in cur_level:
                opt_embedding = embeddings.embed_query(node.opt_desc)
                opt_similarity[node] = self._cosine_similarity(
                    key_embedding, opt_embedding
                )
                for child in node.opt_children:
                    if child not in child_level:
                        child_level.append(child)
            cur_level = child_level

        # Get the node with the highest similarity
        opt_similarity_sorted = sorted(
            opt_similarity.items(), key=lambda item: item[1], reverse=True
        )
        opt_most_similar = opt_similarity_sorted[0][0]

        # Print the nodes and their similarity scores
        debug_str = ""
        if self.debug:
            for key, value in opt_similarity_sorted:
                debug_str += f"""
---------------------------------
{key.opt_desc.splitlines(keepends=True)[:2]}
---------------------------------
{"The similarity score: " + str(value)}
"""

        return opt_most_similar, debug_str
