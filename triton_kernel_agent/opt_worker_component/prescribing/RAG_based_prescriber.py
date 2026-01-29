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
- L1: Bottleneck types (Latency, Memory, Utilization)
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
        self._embeddings = None  # Lazy-loaded

        # Default path
        if database_path is None:
            database_path = (
                Path(__file__).parent.parent.parent.parent
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
        self.logger.info(f"Initialized RAG database from {db_path}")

    def _get_embeddings(self):
        """Lazy-load OpenAI embeddings."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings()
        return self._embeddings

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

        Traverses the entire database tree, computes embedding similarity for
        each node's description, and returns the node with highest similarity.

        Args:
            opt_prompt: Description of the desired optimization
                        (e.g., "use TMA for memory optimization")

        Returns:
            Tuple of (best_matching_node, similarity_scores_dict)
        """
        if not self.opt_hierarchy:
            self.logger.warning("Database not initialized")
            return None, {}

        embeddings = self._get_embeddings()
        key_embedding = embeddings.embed_query(opt_prompt)

        # Traverse tree and compute similarity for all nodes
        root = self.opt_hierarchy.get_root()
        cur_level = list(root.opt_children)
        opt_similarity: dict[OptNode, float] = {}

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

    def build_context(self, opt_node: OptNode) -> str:
        """
        Build context by traversing from opt_node to all leaf nodes.

        Collects optimization descriptions and code examples from the
        node and all its descendants.

        Args:
            opt_node: Starting node (typically from retrieve())

        Returns:
            Context string with technique descriptions and code examples
        """
        context = ""
        leaf_reached = False
        cur_level = [opt_node]

        while cur_level:
            child_level = []
            for node in cur_level:
                # Mark when we reach leaf nodes (code examples)
                if not leaf_reached and not node.opt_children:
                    leaf_reached = True
                    context += (
                        "\nHere are code examples before and after the optimization:\n"
                    )

                context += node.opt_desc

                for child in node.opt_children:
                    if child not in child_level:
                        child_level.append(child)

            cur_level = child_level

        return context
