"""
Utility functions for generating embeddings using a local Ollama model.

This module provides:
- A custom EmbeddingError exception
- A function to embed a single piece of text
- A function to embed a list of texts (chunks)
"""

from typing import List
import ollama

EMBEDDING_MODEL = "nomic-embed-text"

class EmbeddingError(Exception):
    "Custom exception for embedding-related errors"
    pass

def embed_text(text: str) -> List[float]:
    """
    Generate an embedding vector for a single text string.

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

    try:
        resp = ollama.embeddings(
            model = EMBEDDING_MODEL,
            prompt=normalised
        )
    except Exception as e:
        raise EmbeddingError(f"Failed to get embedding from Ollama: {e}")
    
    embedding = resp.get("embedding")
    if embedding is None:
        raise EmbeddingError("No 'embedding' field returned by Ollama")
    
    return embedding


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
            # can skip or fail fast; for v0, fail fast:
            raise EmbeddingError(f"Failed to embed text at index {i}")
        
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