"""Hybrid dense + BM25 retrieval utilities."""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BM25_DIR = Path("bm25_index")
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def _collection_name(vector_store) -> str:
    """Get the active Chroma collection name."""
    try:
        return vector_store._collection.name
    except Exception:
        return "legal_docs_bge"


def _index_path(collection_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection_name)
    return BM25_DIR / f"{safe_name}.pkl"


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 keyword search."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_bm25_index(collection_name: str, docs: List[Document]) -> None:
    """Build and persist a BM25 index for the uploaded document chunks."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    path = _index_path(collection_name)
    tokenized_docs = [_tokenize(doc.page_content) for doc in docs]
    payload = {
        "collection_name": collection_name,
        "docs": docs,
        "tokenized_docs": tokenized_docs,
        "bm25": BM25Okapi(tokenized_docs) if tokenized_docs else None,
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)
    logger.info("Saved BM25 index for collection '%s' with %s chunks at %s.", collection_name, len(docs), path)


def load_bm25_index(collection_name: str) -> Dict:
    """Load a persisted BM25 index."""
    path = _index_path(collection_name)
    with path.open("rb") as file:
        payload = pickle.load(file)
    logger.info("Loaded BM25 index for collection '%s' from %s.", collection_name, path)
    return payload


def build_or_load_bm25_index(collection_name: str, docs: List[Document]) -> Dict:
    """Load BM25 when available; otherwise build it during PDF upload."""
    path = _index_path(collection_name)
    if path.exists():
        return load_bm25_index(collection_name)

    save_bm25_index(collection_name, docs)
    return load_bm25_index(collection_name)


def _dense_search(vector_store, query: str, top_k: int) -> List[Document]:
    return vector_store.similarity_search(query, k=top_k)


def _bm25_search(collection_name: str, query: str, top_k: int) -> List[Document]:
    payload = load_bm25_index(collection_name)
    bm25 = payload.get("bm25")
    docs: List[Document] = payload.get("docs", [])
    if bm25 is None or not docs:
        return []

    scores = bm25.get_scores(_tokenize(query))
    scored_docs = sorted(zip(scores, docs), key=lambda item: float(item[0]), reverse=True)
    results = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["bm25_score"] = float(score)
        results.append(doc)
    return results


def hybrid_retrieve(query: str, vector_store, top_k: int = 10) -> List[Document]:
    """Combine Chroma dense search and BM25 sparse search with simple weighted merging."""
    collection_name = _collection_name(vector_store)
    combined: Dict[str, Dict] = {}

    try:
        dense_docs = _dense_search(vector_store, query, top_k)
        logger.info("Dense retrieval returned %s chunks.", len(dense_docs))
        for rank, doc in enumerate(dense_docs, start=1):
            key = _hash_content(doc.page_content)
            combined.setdefault(key, {"doc": doc, "score": 0.0})
            combined[key]["score"] += DENSE_WEIGHT * (1.0 / rank)
            doc.metadata["dense_rank"] = rank
    except Exception as exc:
        logger.exception("ChromaDB dense retrieval failed.")
        dense_docs = []

    try:
        bm25_docs = _bm25_search(collection_name, query, top_k)
        logger.info("BM25 retrieval returned %s chunks.", len(bm25_docs))
        for rank, doc in enumerate(bm25_docs, start=1):
            key = _hash_content(doc.page_content)
            combined.setdefault(key, {"doc": doc, "score": 0.0})
            combined[key]["score"] += BM25_WEIGHT * (1.0 / rank)
            doc.metadata["bm25_rank"] = rank
    except Exception as exc:
        logger.exception("BM25 retrieval failed. Continuing with vector retrieval only.")

    if not combined:
        return dense_docs[:top_k]

    merged = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    final_docs = []
    for item in merged[:top_k]:
        doc = item["doc"]
        doc.metadata["hybrid_score"] = float(item["score"])
        final_docs.append(doc)

    logger.info("Hybrid retrieval returned %s merged chunks.", len(final_docs))
    return final_docs
