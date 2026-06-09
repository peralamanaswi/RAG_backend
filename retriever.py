"""Corrective RAG retrieval built on the existing hybrid search and reranker."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from analytics import elapsed_ms, now_ms
from memory import detect_conversational_references, get_previous_chunks, get_recent_memory_context
from utils.hybrid_search import hybrid_retrieve
from utils.query_rewriter import rewrite_query
from validator import evaluate_retrieval

logger = logging.getLogger(__name__)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"


def retrieve_crag(
    query: str,
    vector_store: Any,
    chat_history: List[Dict[str, Any]] | None = None,
    base_k: int = 5,
) -> Dict[str, Any]:
    """
    Run real CRAG retrieval:
    retrieve, validate, correct weak retrieval once, validate again.
    """
    started = now_ms()
    correction_started = 0.0
    reranking_ms = 0
    retry_count = 0
    query_rewritten = False
    retry_triggered = False
    memory_used = False
    original_query = query
    query_type = classify_crag_query(query)
    base_k, final_k = _retrieval_depth_for_query_type(query_type, base_k)
    search_query = _expand_short_or_memory_query(query, chat_history or [])
    if search_query != query:
        memory_used = True

    logger.info(
        "CRAG query-aware routing. query_type=%s base_k=%s final_k=%s query=%r",
        query_type,
        base_k,
        final_k,
        query,
    )

    docs, first_rerank_ms = _retrieve_and_rerank(search_query, vector_store, candidate_k=base_k, final_k=final_k)
    reranking_ms += first_rerank_ms
    chunks_before = len(docs)
    evaluation = evaluate_retrieval(search_query, docs)
    logger.info("CRAG initial evaluation: %s", _loggable_evaluation(evaluation, len(docs)))

    if _needs_correction(evaluation, docs):
        retry_triggered = True
        retry_count = 1
        correction_started = now_ms()
        corrected_query = search_query
        failure_reason = evaluation.get("failure_reason")

        if failure_reason in {"low_similarity", "low_confidence", "empty_results", "single_weak_chunk"}:
            corrected_query = rewrite_query(search_query)
            query_rewritten = corrected_query != search_query
        elif failure_reason == "low_keyword_overlap":
            corrected_query = _expand_with_memory(search_query, chat_history or [])
            memory_used = corrected_query != search_query

        retry_k = _retry_k_for_failure(failure_reason, len(docs), query_type)
        corrected_docs, retry_rerank_ms = _retrieve_and_rerank(
            corrected_query,
            vector_store,
            candidate_k=retry_k,
            final_k=min(retry_k, 12 if query_type in {"summary", "legal_scenario"} else 10),
            force_diversity=failure_reason in {"duplicate_chunks", "same_page_only"},
        )
        reranking_ms += retry_rerank_ms
        corrected_evaluation = evaluate_retrieval(corrected_query, corrected_docs)
        logger.info("CRAG corrected evaluation: %s", _loggable_evaluation(corrected_evaluation, len(corrected_docs)))

        docs = corrected_evaluation.get("filtered_chunks", corrected_docs)
        evaluation = corrected_evaluation
        search_query = corrected_query

    filtered_docs = evaluation.get("filtered_chunks", docs) or docs
    correction_ms = elapsed_ms(correction_started) if correction_started else 0

    return {
        "documents": filtered_docs,
        "confidence": float(evaluation.get("confidence", 0.5)),
        "quality": evaluation.get("quality", "medium"),
        "failure_reason": evaluation.get("failure_reason"),
        "retry_needed": bool(evaluation.get("retry_needed")),
        "retry_triggered": retry_triggered,
        "retry_count": retry_count,
        "query_rewritten": query_rewritten,
        "memory_used": memory_used,
        "query_type": query_type,
        "chunks_before": chunks_before,
        "chunks_after": len(filtered_docs),
        "reranking": ENABLE_RERANKER,
        "retrieval_ms": elapsed_ms(started),
        "reranking_ms": reranking_ms,
        "correction_ms": correction_ms,
        "optimized_query": search_query,
        "original_query": original_query,
        "evaluation": evaluation,
    }


def _retrieve_and_rerank(
    query: str,
    vector_store: Any,
    candidate_k: int,
    final_k: int,
    force_diversity: bool = False,
) -> Tuple[List[Document], int]:
    """Retrieve with existing hybrid search and rerank with existing CrossEncoder."""
    try:
        candidates = hybrid_retrieve(query, vector_store, top_k=candidate_k)
    except Exception:
        logger.exception("Hybrid retrieval failed. Falling back to vector retrieval.")
        try:
            candidates = vector_store.similarity_search(query, k=candidate_k)
            for rank, doc in enumerate(candidates, start=1):
                doc.metadata["dense_rank"] = rank
                doc.metadata["vector_similarity"] = 1.0 / rank
        except Exception:
            logger.exception("Vector retrieval fallback failed.")
            return [], 0

    if force_diversity:
        candidates = _diversify_by_page(candidates)

    started = now_ms()
    try:
        from utils.reranker import rerank_documents

        reranked = rerank_documents(query, candidates, top_k=final_k)
        return reranked, elapsed_ms(started)
    except Exception:
        logger.exception("CRAG reranker failed. Continuing with retrieved candidates.")
        return candidates[:final_k], elapsed_ms(started)


def _needs_correction(evaluation: Dict[str, Any], docs: List[Document]) -> bool:
    """Apply the CRAG decision matrix."""
    confidence = float(evaluation.get("confidence", 0.5))
    failure_reason = evaluation.get("failure_reason")
    if not docs:
        return True
    if len(docs) == 1 and confidence < 0.70:
        return True
    if failure_reason in {"duplicate_chunks", "same_page_only"}:
        return True
    return confidence < 0.35


def classify_crag_query(query: str) -> str:
    """Classify retrieval breadth for CRAG without adaptive answer blocking."""
    text = (query or "").lower()
    tokens = set(re.findall(r"[a-zA-Z0-9]+", text))

    summary_terms = {"summary", "summarize", "overview", "brief", "gist", "key", "important", "main"}
    clause_terms = {"clause", "section", "provision", "article", "paragraph", "term", "terms"}
    scenario_terms = {
        "if",
        "when",
        "whether",
        "can",
        "could",
        "liable",
        "liability",
        "eligible",
        "eligibility",
        "terminate",
        "termination",
        "penalty",
        "breach",
        "violation",
        "allowed",
        "required",
        "obligation",
    }

    if tokens.intersection(summary_terms):
        return "summary"
    if tokens.intersection(clause_terms):
        return "clause"
    if tokens.intersection(scenario_terms) or len(tokens) > 14:
        return "legal_scenario"
    return "normal"


def _retrieval_depth_for_query_type(query_type: str, default_k: int) -> Tuple[int, int]:
    """Choose CRAG retrieval breadth based on legal query shape."""
    if query_type == "summary":
        return max(default_k, 12), 10
    if query_type == "legal_scenario":
        return max(default_k, 12), 10
    if query_type == "clause":
        return max(default_k, 7), 7
    return default_k, default_k


def _retry_k_for_failure(failure_reason: str | None, chunk_count: int, query_type: str = "normal") -> int:
    """Choose one deterministic correction depth."""
    if query_type in {"summary", "legal_scenario"}:
        return 15
    if failure_reason == "empty_results":
        return 15
    if failure_reason == "poor_reranker_score":
        return max(10, chunk_count + 5)
    if failure_reason in {"duplicate_chunks", "same_page_only"}:
        return 15
    if failure_reason == "low_similarity":
        return 10
    if failure_reason == "low_keyword_overlap":
        return 10
    return 8


def _expand_short_or_memory_query(query: str, chat_history: List[Dict[str, Any]]) -> str:
    """Add controlled memory context for short or referential queries."""
    terms = re.findall(r"[a-zA-Z0-9]+", query)
    if len(terms) >= 3 and not detect_conversational_references(query):
        return query
    return _expand_with_memory(query, chat_history)


def _expand_with_memory(query: str, chat_history: List[Dict[str, Any]]) -> str:
    """Use recent conversation and previous chunks without inventing new facts."""
    try:
        memory_context = get_recent_memory_context(chat_history, window=5 if detect_conversational_references(query) else 3)
        previous_chunks = get_previous_chunks(chat_history, window=3)
        chunk_text = " ".join(chunk.get("preview", "")[:300] for chunk in previous_chunks[:2])
        additions = " ".join(part for part in [memory_context, chunk_text] if part).strip()
        if not additions:
            return query
        return f"{query}\n\nRecent conversation and retrieved context for reference:\n{additions}"
    except Exception:
        logger.exception("Memory-aware query expansion failed. Using original query.")
        return query


def _diversify_by_page(chunks: List[Document]) -> List[Document]:
    """Prefer chunks from different pages before filling the rest."""
    selected: List[Document] = []
    seen_pages = set()
    remaining: List[Document] = []
    for chunk in chunks:
        page = str(chunk.metadata.get("page", "Unknown"))
        if page not in seen_pages:
            selected.append(chunk)
            seen_pages.add(page)
        else:
            remaining.append(chunk)
    return selected + remaining


def _loggable_evaluation(evaluation: Dict[str, Any], chunk_count: int) -> Dict[str, Any]:
    return {
        "confidence": evaluation.get("confidence"),
        "quality": evaluation.get("quality"),
        "retry_needed": evaluation.get("retry_needed"),
        "failure_reason": evaluation.get("failure_reason"),
        "chunks": chunk_count,
        "avg_similarity": evaluation.get("avg_similarity"),
        "avg_rerank": evaluation.get("avg_rerank"),
        "retrieval_overlap": evaluation.get("retrieval_overlap"),
    }
