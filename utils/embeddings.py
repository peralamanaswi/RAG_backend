"""Embedding and ChromaDB vector store utilities."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the Sentence Transformers embedding model."""
    logger.info("Initializing embedding model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def sanitize_collection_name(name: str) -> str:
    """Create a Chroma-safe collection name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    cleaned = cleaned[:60] or "legal_documents"
    if len(cleaned) < 3:
        cleaned = f"{cleaned}_docs"
    return cleaned


def create_vector_store(
    documents: List[Document],
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "legal_documents",
) -> Chroma:
    """Create a fresh persistent Chroma vector store from documents."""
    if not documents:
        raise ValueError("No document chunks were provided for embedding.")

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    collection_name = sanitize_collection_name(collection_name)

    try:
        logger.info("Connecting to ChromaDB persistent store at: %s", persist_path.resolve())
        client = chromadb.PersistentClient(path=str(persist_path))
        logger.info("ChromaDB connection successful.")

        existing_collection = None
        try:
            existing_collection = client.get_collection(collection_name)
        except Exception:
            logger.info("No existing Chroma collection found: %s", collection_name)

        if existing_collection is not None:
            metadata = existing_collection.metadata or {}
            stored_model = metadata.get("embedding_model")
            stored_count = existing_collection.count()
            if stored_model == EMBEDDING_MODEL_NAME and stored_count > 0:
                logger.info(
                    "Loading existing Chroma collection '%s'. embedding_model=%s, stored_chunks=%s.",
                    collection_name,
                    stored_model,
                    stored_count,
                )
                return Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=get_embedding_model(),
                    persist_directory=str(persist_path),
                )

            logger.warning(
                "Rebuilding Chroma collection '%s' because embedding model changed or collection is empty. "
                "stored_model=%s, current_model=%s, stored_chunks=%s.",
                collection_name,
                stored_model,
                EMBEDDING_MODEL_NAME,
                stored_count,
            )
            client.delete_collection(collection_name)

        logger.info("Creating Chroma collection '%s' with %s document chunks.", collection_name, len(documents))
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=get_embedding_model(),
            client=client,
            collection_name=collection_name,
            collection_metadata={"embedding_model": EMBEDDING_MODEL_NAME},
            persist_directory=str(persist_path),
        )
        stored_count = vector_store._collection.count()
        logger.info(
            "Created Chroma collection '%s'. Requested chunks=%s, stored chunks=%s.",
            collection_name,
            len(documents),
            stored_count,
        )
        return vector_store
    except Exception as exc:
        logger.exception("Failed to create Chroma vector store.")
        raise RuntimeError(f"Could not create vector database: {exc}") from exc


def load_vector_store(
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "legal_documents",
) -> Chroma:
    """Load an existing Chroma vector store."""
    persist_path = Path(persist_directory)
    collection_name = sanitize_collection_name(collection_name)
    logger.info("Loading Chroma collection '%s' from: %s", collection_name, persist_path.resolve())
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_path),
    )


def get_collection_count(vector_store: Chroma) -> int:
    """Return the number of stored embeddings in a Chroma collection."""
    try:
        count = vector_store._collection.count()
        logger.info("Chroma collection contains %s embedded chunks.", count)
        return count
    except Exception as exc:
        logger.exception("Could not count Chroma collection items.")
        raise RuntimeError(f"Could not count Chroma collection items: {exc}") from exc


def run_similarity_search(vector_store: Chroma, query: str, top_k: int = 3) -> List[Document]:
    """Run and log a sample Chroma similarity search."""
    try:
        logger.info("Running Chroma similarity search. query=%r, top_k=%s", query, top_k)
        results = vector_store.similarity_search(query, k=top_k)
        logger.info("Chroma similarity search returned %s chunks.", len(results))
        for index, doc in enumerate(results, start=1):
            logger.info(
                "Retrieved chunk %s | source=%s | page=%s | chars=%s | preview=%s",
                index,
                doc.metadata.get("source", "Unknown"),
                doc.metadata.get("page", "Unknown"),
                len(doc.page_content),
                doc.page_content[:300].replace("\n", " "),
            )
        return results
    except Exception as exc:
        logger.exception("Chroma similarity search failed.")
        raise RuntimeError(f"Chroma similarity search failed: {exc}") from exc
