"""
Vector store utilities using ChromaDB for a PDF RAG system.

Responsibilities:
- Initialize a persistent Chroma client
- Provide access to a single collection for PDF chunks
- Add document chunks + embeddings to the collection
- Query most similar chunks for a given session and query embedding
"""

from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings


# ===================#
# Configuration      #
# ===================#
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"

class VectorStoreError(Exception):
    """
    Custom exception for vector store related problems.
    """
    pass


# ===================#
# Client / Collection #
# ===================#

def _get_chroma_client() -> chromadb.Client:
    """
    Initialise and return a chroma client with persistent storage
    """
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    client = chromadb.Client(
        Settings(
            chroma_db_impl = "duckdb+parquet",
            persist_directory = PERSIST_DIRECTORY,
        )
    )
    return client

def _get_collection():
    """
    Get or create the main collection used for storing PDF chunks

    Returns:
        A Chroma collection object
    """

    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


# ===================#
# Public API          #
# ===================#

def add_document(
    session_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> None:
    """
    Add a document's chunks and embeddings to the vector store.

    Each chunk is stored with:
    - id: f"{session_id}_{chunk_index}:
    - document: chunk text
    - embedding: embedding vector
    - metadata: {"session_id": session_id, "chunk_index": index}

    Args:
        session_id (str): Unique ID representing this PDF/document
        chunks (List[str]): List of chunked text segments
        embeddings (List[List[float]]): List of embedding vectors corresponding 1:1 with chunks

    Raises:
        VectorStoreError: If validation fails or Chroma operations fail
    """

    if not session_id or not isinstance(session_id, str):
        raise VectorStoreError("session_id must be a non-empty string")
    
    if not isinstance(chunks, list) or not isinstance(embeddings, list):
        raise VectorStoreError("chunks or embeddings must be lists")
    
    if len(chunks) == 0:
        raise VectorStoreError("No chunks provided")
    
    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            f"Number of chunks ({len(chunks)}) does not match "
            f"number of embeddings ({len(embeddings)})"
        )
    
    collection = _get_collection()

    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"session_id": session_id, "chunk_index": i} for i in range(len(chunks))
    ]

    try:
        collection.add(
            ids = ids,
            documents = chunks,
            embeddings = embeddings,
            metadatas = metadatas,
        )
    except Exception as e:
        raise VectorStoreError(f"Failed to add document to vector store: {e}")
    
    def query_document(
        session_id: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the vector store for the most similar chunks within a given session.

        Args:
            session_id (str): ID of the document/session to search within.
            query_embedding (List[float]): Embedding vector for the query.
            n_results (int): Number of top results to retreive.

        Returns:
            Dict[str, any]: A dictionary with keys such as "documents", "metadata", etc.

        Raises:
            VectorStore Error: If the query fails 
        """
        if not session_id or not isinstance(session_id, str):
            raise VectorStoreError("session_id must be a non-empty string")
        
        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            raise VectorStoreError("query_embedding must be a non-empty list of floats")
        
        if n_results <= 0:
            raise VectorStoreError("n_results must be > 0")
        
        collection = _get_collection()

        try:
            results = collection.query(
                query_embeddings = [query_embedding],
                n_results = n_results,
                where={"session_id": session_id},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to query vector store: {e}")
        
        return results
    



if __name__ == "__main__":
    """
    Manual test to visualize vector store behavior without needing Ollama.

    We:
    - Create some fake "chunks"
    - Create some fake embeddings (small dimension)
    - Add them to the store under a session_id
    - Query with a fake embedding
    - Print out results
    """

    print("=== Manual test for vector_store.py ===\n")

    # Fake data for demonstration
    session_id = "test-session-123"

    chunks = [
        "The cat sat on the mat.",
        "The stock market experienced significant volatility today.",
        "Machine learning models can generate text embeddings.",
    ]

    # Fake 3D embeddings just for demo; in real code, use real embeddings
    fake_embeddings = [
        [0.1, 0.2, 0.3],   # for chunk 0
        [0.9, 0.8, 0.7],   # for chunk 1
        [0.4, 0.5, 0.6],   # for chunk 2
    ]

    print("Adding fake document to vector store...")
    add_document(session_id=session_id, chunks=chunks, embeddings=fake_embeddings)
    print("Document added.\n")

    # Fake query embedding (pretend it's close to chunk 2)
    query_emb = [0.35, 0.45, 0.55]

    print("Querying vector store...")
    result = query_document(session_id=session_id, query_embedding=query_emb, n_results=2)

    print("\nQuery results (raw):")
    print(result)

    # Nicely print the top documents and metadata
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    print("\nTop matches:")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        print(f"  Rank {i+1}:")
        print(f"    Distance: {dist}")
        print(f"    Chunk index: {meta.get('chunk_index')}")
        print(f"    Text: {doc}")
        print()

    print("=== End of manual test ===")
