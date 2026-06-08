"""Deterministic retrieval validator for Corrective RAG (CRAG)."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def evaluate_retrieval(query: str, retrieved_chunks: List[Document], reranker: Any = None) -> Dict[str, Any]:
    """
    Validate retrieval quality and decide whether CRAG correction is needed.

    Confidence formula:
    (0.5 * avg_similarity) + (0.3 * avg_rerank) + (0.2 * retrieval_overlap)
    """
    try:
        if not retrieved_chunks:
            return {
                "confidence": 0.0,
                "quality": "poor",
                "retry_needed": True,
                "failure_reason": "empty_results",
                "filtered_chunks": [],
                "avg_similarity": 0.0,
                "avg_rerank": 0.5,
                "retrieval_overlap": 0.0,
                "keyword_overlap": 0.0,
                "duplicate_count": 0,
            }

        deduped_chunks, duplicate_count = deduplicate_chunks(retrieved_chunks)
        avg_similarity = _average([_chunk_similarity(chunk) for chunk in deduped_chunks])
        keyword_overlap = calculate_keyword_overlap(query, deduped_chunks)
        agreement = calculate_retriever_agreement(deduped_chunks)
        retrieval_overlap = max(keyword_overlap, agreement)
        avg_rerank = calculate_reranker_score(query, deduped_chunks, reranker)
        confidence = round(
            (0.5 * _clamp(avg_similarity))
            + (0.3 * _clamp(avg_rerank))
            + (0.2 * _clamp(retrieval_overlap)),
            3,
        )
        quality = confidence_quality(confidence)
        failure_reason = detect_failure_reason(
            chunks=deduped_chunks,
            avg_similarity=avg_similarity,
            avg_rerank=avg_rerank,
            keyword_overlap=keyword_overlap,
            duplicate_count=duplicate_count,
            confidence=confidence,
        )

        return {
            "confidence": confidence,
            "quality": quality,
            "retry_needed": quality == "poor",
            "failure_reason": failure_reason,
            "filtered_chunks": filter_noisy_chunks(deduped_chunks, query),
            "avg_similarity": round(avg_similarity, 3),
            "avg_rerank": round(avg_rerank, 3),
            "retrieval_overlap": round(retrieval_overlap, 3),
            "keyword_overlap": round(keyword_overlap, 3),
            "retriever_agreement": round(agreement, 3),
            "duplicate_count": duplicate_count,
        }
    except Exception:
        logger.exception("CRAG validator failed. Falling back to medium confidence.")
        return {
            "confidence": 0.5,
            "quality": "medium",
            "retry_needed": False,
            "failure_reason": "validator_failure",
            "filtered_chunks": retrieved_chunks,
            "avg_similarity": 0.5,
            "avg_rerank": 0.5,
            "retrieval_overlap": 0.5,
            "keyword_overlap": 0.5,
            "duplicate_count": 0,
        }


def deduplicate_chunks(chunks: Iterable[Document]) -> Tuple[List[Document], int]:
    """Remove duplicate chunks by content hash."""
    seen = set()
    deduped: List[Document] = []
    duplicate_count = 0
    for chunk in chunks:
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        deduped.append(chunk)
    return deduped, duplicate_count


def filter_noisy_chunks(chunks: List[Document], query: str) -> List[Document]:
    """Keep useful chunks while preserving enough context for legal answers."""
    if len(chunks) <= 3:
        return chunks

    query_terms = set(_tokens(query))
    scored: List[Tuple[float, Document]] = []
    for chunk in chunks:
        chunk_terms = set(_tokens(chunk.page_content))
        overlap = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
        similarity = _chunk_similarity(chunk)
        rerank = _metadata_rerank(chunk)
        score = (0.45 * similarity) + (0.35 * rerank) + (0.20 * overlap)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def calculate_keyword_overlap(query: str, chunks: List[Document]) -> float:
    """Calculate query-term overlap against retrieved chunks."""
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0

    context_terms = set()
    for chunk in chunks:
        context_terms.update(_tokens(chunk.page_content))

    return _clamp(len(query_terms.intersection(context_terms)) / len(query_terms))


def calculate_retriever_agreement(chunks: List[Document]) -> float:
    """Measure agreement between dense and BM25 retrieval metadata."""
    if not chunks:
        return 0.0

    agreed = 0
    for chunk in chunks:
        if chunk.metadata.get("dense_rank") is not None and chunk.metadata.get("bm25_rank") is not None:
            agreed += 1
    return _clamp(agreed / len(chunks))


def calculate_reranker_score(query: str, chunks: List[Document], reranker: Any = None) -> float:
    """Use existing rerank metadata or an optional reranker object; fallback to 0.5."""
    metadata_scores = [_metadata_rerank(chunk) for chunk in chunks if chunk.metadata.get("rerank_score") is not None]
    if metadata_scores:
        return _average(metadata_scores)

    if reranker is None:
        return 0.5

    try:
        pairs = [(query, chunk.page_content) for chunk in chunks]
        scores = reranker.predict(pairs)
        return _average([_normalize_rerank(float(score)) for score in scores])
    except Exception:
        logger.exception("Validator reranker scoring failed. Falling back to 0.5.")
        return 0.5


def confidence_quality(confidence: float) -> str:
    """Map confidence to CRAG quality labels with legal-document-tolerant thresholds."""
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.25:
        return "medium"
    return "poor"


def detect_failure_reason(
    chunks: List[Document],
    avg_similarity: float,
    avg_rerank: float,
    keyword_overlap: float,
    duplicate_count: int,
    confidence: float,
) -> str | None:
    """Identify the main deterministic reason for weak retrieval."""
    if not chunks:
        return "empty_results"
    if duplicate_count > 0:
        return "duplicate_chunks"
    if len(chunks) == 1 and confidence < 0.70:
        return "single_weak_chunk"
    if avg_similarity < 0.50:
        return "low_similarity"
    if avg_rerank < 0.40:
        return "poor_reranker_score"
    if keyword_overlap < 0.30:
        return "low_keyword_overlap"
    if _same_page_only(chunks) and len(chunks) >= 3:
        return "same_page_only"
    if confidence < 0.55:
        return "low_confidence"
    return None


def _same_page_only(chunks: List[Document]) -> bool:
    pages = {str(chunk.metadata.get("page", "Unknown")) for chunk in chunks}
    return len(pages) == 1


def _chunk_similarity(chunk: Document) -> float:
    if chunk.metadata.get("vector_similarity") is not None:
        return _clamp(float(chunk.metadata.get("vector_similarity", 0.0)))
    if chunk.metadata.get("hybrid_score") is not None:
        return _clamp(float(chunk.metadata.get("hybrid_score", 0.0)))
    dense_rank = chunk.metadata.get("dense_rank")
    if dense_rank:
        return _clamp(1.0 / float(dense_rank))
    return 0.5


def _metadata_rerank(chunk: Document) -> float:
    return _normalize_rerank(float(chunk.metadata.get("rerank_score", 0.0)))


def _normalize_rerank(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    if math.isnan(score):
        return 0.0
    return _clamp((score + 10.0) / 20.0)


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp(value: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))
