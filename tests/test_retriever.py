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

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


def test_doc_preprocess():
    """Preprocess the documents."""
    # Langchain
    # =========================================
    # DirectoryLoader - load only .py and .md files, exclude __init__.py
    dir_loader = DirectoryLoader(
        "../kernel_perf_agent/kernel_opt/database/",
        glob="**/*",
        exclude=["**/__init__.py", "**/__pycache__/**"],
        recursive=True,
    )
    dir_docs = dir_loader.load()
    print(f"Loaded {len(dir_docs)} documents")

    # =========================================
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    query = "add Tensor Memory Accelerator (TMA) support to the kernel"
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    for doc in dir_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = cosine_similarity([query_embedding], doc_embedding)[0]
        print(doc.metadata, similarity)


if __name__ == "__main__":
    print("Testing doc_preprocess function...")
    print("=" * 60)

    print("Running test_doc_preprocess()...")
    test_doc_preprocess()
    print("\n" + "=" * 60)
    print("Test completed!")
