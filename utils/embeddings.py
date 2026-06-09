"""Embedding and ChromaDB vector store utilities."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List

if os.getenv("VERCEL"):
    SERVERLESS_TMP = Path("/tmp")
    SERVERLESS_CACHE = SERVERLESS_TMP / ".cache"
    os.environ.setdefault("HOME", str(SERVERLESS_TMP))
    os.environ.setdefault("XDG_CACHE_HOME", str(SERVERLESS_CACHE))
    os.environ.setdefault("HF_HOME", str(SERVERLESS_CACHE / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(SERVERLESS_CACHE / "huggingface" / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SERVERLESS_CACHE / "sentence-transformers"))
else:
    SERVERLESS_CACHE = None

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

if SERVERLESS_CACHE is not None:
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = SERVERLESS_CACHE / "chroma" / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "onnx").strip().lower()
EMBEDDING_MODEL_NAME = (
    "chroma-onnx-all-MiniLM-L6-v2"
    if EMBEDDING_BACKEND == "onnx"
    else os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
)
CHROMA_ADD_BATCH_SIZE = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "32"))


class ChromaOnnxEmbeddings(Embeddings):
    """LangChain adapter for Chroma's lightweight ONNX MiniLM embedding function."""

    def __init__(self) -> None:
        self._embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._embedding_function(texts)
        return [embedding.astype("float32").tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def is_chroma_cloud_enabled() -> bool:
    """Return True when Chroma Cloud credentials are configured."""
    return bool(os.getenv("CHROMA_API_KEY"))


def get_chroma_client(persist_path: Path) -> Any:
    """Create a Chroma Cloud client on Render or a local persistent client for development."""
    if is_chroma_cloud_enabled():
        logger.info(
            "Connecting to Chroma Cloud. tenant_configured=%s database_configured=%s",
            bool(os.getenv("CHROMA_TENANT")),
            bool(os.getenv("CHROMA_DATABASE")),
        )
        cloud_kwargs = {"api_key": os.getenv("CHROMA_API_KEY")}
        if os.getenv("CHROMA_TENANT"):
            cloud_kwargs["tenant"] = os.getenv("CHROMA_TENANT")
        if os.getenv("CHROMA_DATABASE"):
            cloud_kwargs["database"] = os.getenv("CHROMA_DATABASE")
        return chromadb.CloudClient(**cloud_kwargs)

    persist_path.mkdir(parents=True, exist_ok=True)
    logger.info("Connecting to local ChromaDB persistent store at: %s", persist_path.resolve())
    return chromadb.PersistentClient(path=str(persist_path))


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return the configured embedding model."""
    logger.info("Initializing embedding model. backend=%s model=%s", EMBEDDING_BACKEND, EMBEDDING_MODEL_NAME)
    if EMBEDDING_BACKEND == "onnx":
        return ChromaOnnxEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": CHROMA_ADD_BATCH_SIZE},
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
    collection_name = sanitize_collection_name(collection_name)

    try:
        client = get_chroma_client(persist_path)
        logger.info("ChromaDB connection successful. mode=%s", "cloud" if is_chroma_cloud_enabled() else "local")

        existing_collection = None
        try:
            existing_collection = client.get_collection(collection_name)
        except Exception:
            logger.info("No existing Chroma collection found: %s", collection_name)

        if existing_collection is not None:
            metadata = existing_collection.metadata or {}
            stored_model = metadata.get("embedding_model")
            stored_count = existing_collection.count()
            if stored_model == EMBEDDING_MODEL_NAME and stored_count >= len(documents):
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
                )

            logger.warning(
                "Rebuilding Chroma collection '%s' because embedding model changed or collection is incomplete. "
                "stored_model=%s, current_model=%s, stored_chunks=%s.",
                collection_name,
                stored_model,
                EMBEDDING_MODEL_NAME,
                stored_count,
            )
            client.delete_collection(collection_name)

        logger.info(
            "Creating Chroma collection '%s' with %s document chunks. batch_size=%s.",
            collection_name,
            len(documents),
            CHROMA_ADD_BATCH_SIZE,
        )
        chroma_kwargs = {
            "collection_name": collection_name,
            "embedding_function": get_embedding_model(),
            "client": client,
            "collection_metadata": {"embedding_model": EMBEDDING_MODEL_NAME},
        }
        if not is_chroma_cloud_enabled():
            chroma_kwargs["persist_directory"] = str(persist_path)

        vector_store = Chroma(**chroma_kwargs)
        for start in range(0, len(documents), CHROMA_ADD_BATCH_SIZE):
            batch = documents[start : start + CHROMA_ADD_BATCH_SIZE]
            vector_store.add_documents(batch)
            logger.info(
                "Added Chroma batch %s-%s of %s.",
                start + 1,
                min(start + len(batch), len(documents)),
                len(documents),
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
    client = get_chroma_client(persist_path)
    logger.info("Loading Chroma collection '%s'. mode=%s", collection_name, "cloud" if is_chroma_cloud_enabled() else "local")
    chroma_kwargs = {
        "client": client,
        "collection_name": collection_name,
        "embedding_function": get_embedding_model(),
    }
    if not is_chroma_cloud_enabled():
        chroma_kwargs["persist_directory"] = str(persist_path)
    return Chroma(**chroma_kwargs)


def get_existing_collection_count(
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "legal_documents",
) -> int | None:
    """Return the collection count when it already exists, otherwise None."""
    try:
        persist_path = Path(persist_directory)
        client = get_chroma_client(persist_path)
        collection = client.get_collection(sanitize_collection_name(collection_name))
        return collection.count()
    except Exception:
        return None


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
