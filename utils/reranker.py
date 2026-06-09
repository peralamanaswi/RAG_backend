"""Cross-encoder reranking for retrieved legal chunks."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder reranker once."""
    if not ENABLE_RERANKER:
        raise RuntimeError("Reranker is disabled. Set ENABLE_RERANKER=true to enable it.")
    from sentence_transformers import CrossEncoder

    logger.info("Initializing reranker model: %s", RERANKER_MODEL_NAME)
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_documents(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    """Score retrieved documents with a CrossEncoder and return the best chunks."""
    if not docs:
        return []
    if not ENABLE_RERANKER:
        logger.info("Reranker disabled; returning top %s retrieved chunks without CrossEncoder.", top_k)
        return docs[:top_k]

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda item: float(item[0]), reverse=True)
    reranked = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        reranked.append(doc)
    logger.info("Reranked %s chunks and selected top %s.", len(docs), len(reranked))
    return reranked
