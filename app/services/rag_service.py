"""
RAG Service — Retrieval-Augmented Generation using ELA_2025-2028.docx

Handles:
1. Loading and chunking the ELA document
2. Building an embedding index using sentence-transformers
3. Caching the embeddings to disk using pickle
4. Retrieving the most relevant chunks via scikit-learn cosine similarity
"""

import os
import pickle
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded globals — initialized on first use
_rag_ready: bool = False
_chunks: List[str] = []
_embeddings = None
_embedding_model = None


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for better retrieval coverage."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _load_docx(path: str) -> str:
    """Extract all paragraph text from a .docx file."""
    try:
        from docx import Document
    except ImportError:
        logger.error("[RAG] python-docx is not installed.")
        raise
        
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _build_or_load_index(doc_path: str, store_dir: str) -> None:
    """
    Build the vector embeddings from the docx if not already built,
    otherwise load the existing numpy arrays from the cache file.
    """
    global _chunks, _embeddings, _embedding_model
    from sentence_transformers import SentenceTransformer

    os.makedirs(store_dir, exist_ok=True)
    cache_path = os.path.join(store_dir, "rag_cache.pkl")

    # Load embedding model once
    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # If cache exists, load it
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                _chunks = data["chunks"]
                _embeddings = data["embeddings"]
            logger.info("[RAG] Loaded existing index from disk cache.")
            return
        except Exception as e:
            logger.warning(f"[RAG] Failed to load cache: {e}. Re-indexing...")

    # Otherwise, parse docx and build index
    logger.info(f"[RAG] Indexing {doc_path} ...")
    raw_text = _load_docx(doc_path)
    _chunks = _chunk_text(raw_text, chunk_size=500, overlap=100)
    
    if not _chunks:
        raise ValueError(f"Document {doc_path} is empty or unreadable.")

    # Encode all chunks to vectors
    _embeddings = _embedding_model.encode(_chunks, convert_to_numpy=True)

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump({
            "chunks": _chunks,
            "embeddings": _embeddings
        }, f)

    logger.info(f"[RAG] Ready. {len(_chunks)} chunks indexed.")


def initialize_rag(doc_path: str, store_dir: str) -> None:
    """
    Initialize the RAG index at application startup.
    Call this once from main.py lifespan so the index is ready before requests come in.
    """
    global _rag_ready

    if not os.path.exists(doc_path):
        logger.warning(f"[RAG] Document not found at '{doc_path}'. RAG disabled.")
        return

    try:
        _build_or_load_index(doc_path, store_dir)
        _rag_ready = True
        logger.info("[RAG] Initialization complete.")
    except Exception as e:
        logger.error(f"[RAG] Failed to initialize: {e}")
        _rag_ready = False


def retrieve_context(query: str, top_k: int = 3) -> Optional[str]:
    """
    Retrieve the most relevant document chunks for a user query.

    Args:
        query: The user's question or message.
        top_k: How many chunks to return.

    Returns:
        A single string with the top-K chunks joined, or None if RAG is not ready.
    """
    global _chunks, _embeddings, _embedding_model
    
    if not _rag_ready or _embeddings is None or _embedding_model is None:
        return None

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 1. Embed the query
        query_emb = _embedding_model.encode([query], convert_to_numpy=True)
        
        # 2. Compute similarity against all chunks
        similarities = cosine_similarity(query_emb, _embeddings)[0]
        
        # 3. Get top-K indices (sorted descending)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 4. Fetch the actual text chunks
        docs = [_chunks[i] for i in top_indices]
        
        if not docs:
            return None
            
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return None


def is_ready() -> bool:
    """Return True if the RAG index is loaded and ready to use."""
    return _rag_ready
