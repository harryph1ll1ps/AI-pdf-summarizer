"""
Utility functions for generating embeddings using a local Ollama model.

This module provides:
- A custom EmbeddingError exception
- A function to embed a single piece of text
- A function to embed a list of texts (chunks)
"""

from typing import List
import ollama
from backend.config import MAX_CHUNK_CHARS

EMBEDDING_MODEL = "nomic-embed-text"

class EmbeddingError(Exception):
    "Custom exception for embedding-related errors"
    pass

def embed_text(text: str) -> List[float]:
    """
    Generate an embedding vector for a single text string (a chunk).

    Args:
        text (str): Input text to embed.

    Returns:
        List[float]: Embedding vector.

    Raised:
        EmbeddingError: If something goes wrong during embedding
    """

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    
    # basic guard against empty text
    normalised = " ".join(text.split())
    if not normalised:
        raise EmbeddingError("Cannot embed empty text")
    
    if len(normalised) > MAX_CHUNK_CHARS:
        raise EmbeddingError(f"Chunk too large to embed ({len(normalised)} chars)")

    try:
        resp = ollama.embed(
            model = EMBEDDING_MODEL,
            input=normalised
        )
    except Exception as e:
        raise EmbeddingError(f"Failed to get embedding from Ollama: {e}")
    
    # resp is returned as a dictionary with keys such as model, prompt, embeddings
    # { "embeddings": [ [0.1, 0.2, ... ] ] }
    embeddings_list = resp.get("embeddings")
    if embeddings_list is None or len(embeddings_list) == 0:
        raise EmbeddingError("Ollama did not return embeddings")
    
    # return a vector where each number in the vector represents the texts' location in a dimension: 
    # e.g. [0.2, 0.8] means x = 0.2, y = 0.8 <- however there are 768 dimensions (e.g. technicality, similarity, etc.) 
    return embeddings_list[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embedding vectors for a list of text strings.

    This is a simple loop-based implementation. It's fine for v0
    and easier to debug. If needed later, we can add batching / concurrency.

    Args:
        texts (List[str]): List of text chunks to embed

    Returns :
        List[List[float]]: List of embedding vectors aligned with input order

    Raises:
        EmbeddingError: If any embedding call fails
    """

    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")
    
    embeddings: List[List[float]] = []

    for i, text in enumerate(texts):
        try:
            emb = embed_text(text)
            embeddings.append(emb)
        except EmbeddingError as e:
            print(f"[WARNING] Skipping chunk {i}: {e}")
            continue
    
    if not embeddings:
        raise EmbeddingError("No chunks could be embedded successfully")
    
    return embeddings


if __name__ == "__main__":
    print("=== Manual tests for embeddings ===\n")

    sample = "This is a small test sentence for embeddings."
    print("Single text test:")
    print("Input:", sample)
    emb = embed_text(sample)
    print("Embedding length:", len(emb))
    print("First 5 values:", emb[:5])
    print("-" * 70)

    texts = [
        "First chunk of a document.",
        "Second chunk with different content.",
        "Third chunk about something else."
    ]
    print("Batch embedding test:")
    embs = embed_texts(texts)
    print("Number of embeddings:", len(embs))
    for i, e in enumerate(embs):
        print(f"  Chunk {i}: length={len(e)}, first 3 values={e[:3]}")
    print("-" * 70)