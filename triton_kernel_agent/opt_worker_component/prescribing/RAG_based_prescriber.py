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

"""RAG-based prescriber for retrieving optimization patterns from the database.

This module provides embedding-based retrieval of optimization patterns from
a hierarchical knowledge base. The database is organized in 3 levels:
- L1: Bottleneck types (Compute, Memory, Under-utilization)
- L2: Optimization techniques (TMA, PID swizzling, Persistence)
- L3: Code examples (matmul.py, matmul_sw.py, etc.)

Usage:
    prescriber = RAGPrescriber(database_path=Path("..."))

    # Retrieve best matching pattern for an optimization hint
    opt_node, similarity_scores = prescriber.retrieve("use TMA for memory optimization")

    # Build context from the retrieved node (traverses to leaf code examples)
    context = prescriber.build_context(opt_node)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from kernel_perf_agent.kernel_opt.database.base import OptHierarchy, OptNode


class RAGPrescriber:
    """
    Embedding-based retriever for optimization patterns.

    Uses OpenAI embeddings to find the most similar optimization node
    based on the optimization prompt, then traverses to leaf nodes
    to collect code examples.
    """

    def __init__(
        self,
        database_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the RAG prescriber.

        Args:
            database_path: Path to code_samples directory. Defaults to
                           kernel_perf_agent/kernel_opt/database/code_samples
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.opt_hierarchy: OptHierarchy | None = None
        self._openai_client = None  # Lazy-loaded
        self._node_embeddings: dict[
            OptNode, list[float]
        ] = {}  # Precomputed L1/L2 embeddings

        # Default path: navigate to project root (where pyproject.toml is)
        if database_path is None:
            project_root = Path(__file__).resolve()
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            database_path = (
                project_root
                / "kernel_perf_agent"
                / "kernel_opt"
                / "database"
                / "code_samples"
            )

        self._initialize_database(database_path)

    def _initialize_database(self, db_path: Path) -> None:
        """Initialize the optimization hierarchy from disk."""
        if not db_path.exists():
            self.logger.warning(f"RAG database path not found: {db_path}")
            return

        self.opt_hierarchy = OptHierarchy()
        self.opt_hierarchy.hard_initialize(db_path)
        self._precompute_embeddings()
        self.logger.info(f"Initialized RAG database from {db_path}")

    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for L1/L2 nodes."""
        if not self.opt_hierarchy:
            return

        root = self.opt_hierarchy.get_root()
        cur_level = list(root.opt_children)

        while cur_level:
            child_level = []
            for node in cur_level:
                # Only embed L1/L2 nodes (nodes with children), skip L3 code examples
                if node.opt_children:
                    self._node_embeddings[node] = self._embed_query(node.opt_desc)
                    for child in node.opt_children:
                        if child not in child_level:
                            child_level.append(child)
            cur_level = child_level

        self.logger.info(
            f"Precomputed embeddings for {len(self._node_embeddings)} L1/L2 nodes"
        )

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI()
        return self._openai_client

    def _embed_query(self, text: str) -> list[float]:
        """Get embedding for a text query."""
        client = self._get_openai_client()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large",  # text-embedding-3-small for lower cost
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def retrieve(self, opt_prompt: str) -> tuple[OptNode | None, dict[OptNode, float]]:
        """
        Retrieve the most relevant optimization node using embedding similarity.

        Computes similarity against precomputed L1/L2 node embeddings and returns
        the node with highest similarity. Use build_context() to traverse down
        to L3 code examples.

        Args:
            opt_prompt: Description of the desired optimization
                        (e.g., "use TMA for memory optimization")

        Returns:
            Tuple of (best_matching_node, similarity_scores_dict)
        """
        if not self.opt_hierarchy:
            self.logger.warning("Database not initialized")
            return None, {}

        key_embedding = self._embed_query(opt_prompt)

        # Compute similarity against precomputed L1/L2 node embeddings
        opt_similarity: dict[OptNode, float] = {}
        for node, node_embedding in self._node_embeddings.items():
            opt_similarity[node] = self._cosine_similarity(
                key_embedding, node_embedding
            )

        if not opt_similarity:
            return None, {}

        # Get node with highest similarity
        opt_similarity_sorted = sorted(
            opt_similarity.items(), key=lambda item: item[1], reverse=True
        )
        best_node = opt_similarity_sorted[0][0]

        self.logger.info(
            f"Retrieved optimization pattern (similarity={opt_similarity_sorted[0][1]:.3f}): "
            f"{best_node.opt_desc[:80]}..."
        )

        return best_node, opt_similarity

    def build_context(
        self,
        opt_node: OptNode,
        max_chars: int = 8192,
        max_code_examples: int = 2,
    ) -> str:
        """
        Build context from opt_node with technique description and code examples.

        Args:
            opt_node: Starting node (typically from retrieve())
            max_chars: Maximum character budget for context
            max_code_examples: Maximum number of leaf code examples to include

        Returns:
            Context string with technique description and limited code examples
        """
        parts: list[str] = []
        code_examples: list[str] = []

        # BFS to collect technique descriptions (non-leaf) and code examples (leaf)
        cur_level = [opt_node]
        while cur_level:
            child_level = []
            for node in cur_level:
                if node.opt_children:
                    # Non-leaf: technique description
                    parts.append(node.opt_desc.strip())
                else:
                    # Leaf: code example
                    code_examples.append(node.opt_desc.strip())

                for child in node.opt_children:
                    if child not in child_level:
                        child_level.append(child)
            cur_level = child_level

        # Build context with separators
        context = ""

        # Add technique descriptions
        if parts:
            context += "## Optimization Technique\n\n"
            context += "\n\n---\n\n".join(parts)

        # Add limited code examples
        if code_examples:
            selected_examples = code_examples[:max_code_examples]
            if parts:
                context += "\n\n---\n\n"  # Separator between sections
            context += "## Code Examples\n\n"
            context += "\n\n---\n\n".join(selected_examples)

            if len(code_examples) > max_code_examples:
                context += f"\n\n(Showing {max_code_examples} of {len(code_examples)} examples)"

        # Truncate if over budget
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n... (truncated)"
            self.logger.warning(
                f"Context truncated from {len(context)} to {max_chars} chars"
            )

        return context
