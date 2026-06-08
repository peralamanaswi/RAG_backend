"""Retrieval analytics and latency tracking for Corrective RAG."""

from __future__ import annotations

import time
from typing import Any, Dict


def now_ms() -> float:
    """Return current high-resolution time in milliseconds."""
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> int:
    """Return elapsed milliseconds from a start timestamp."""
    return int(now_ms() - start_ms)


def build_analytics(
    query_type: str,
    strategy: str,
    confidence: float,
    chunks: int,
    reranking: bool,
    prompt_type: str,
    routing_ms: int = 0,
    retrieval_ms: int = 0,
    reranking_ms: int = 0,
    total_ms: int = 0,
    confidence_action: str = "proceed_normally",
) -> Dict[str, Any]:
    """Return UI-friendly retrieval analytics."""
    return {
        "query_type": query_type,
        "strategy": strategy,
        "confidence": round(float(confidence), 3),
        "chunks": int(chunks),
        "reranking": bool(reranking),
        "latency_ms": int(total_ms),
        "routing_ms": int(routing_ms),
        "retrieval_ms": int(retrieval_ms),
        "reranking_ms": int(reranking_ms),
        "prompt_type": prompt_type,
        "confidence_action": confidence_action,
    }


def build_crag_analytics(
    confidence: float,
    quality: str,
    retry_triggered: bool,
    query_rewritten: bool,
    chunks_before: int,
    chunks_after: int,
    reranking: bool,
    latency_ms: int,
    retrieval_ms: int = 0,
    reranking_ms: int = 0,
    correction_ms: int = 0,
    retry_count: int = 0,
    failure_reason: str | None = None,
    memory_used: bool = False,
    query_type: str = "normal",
    blocking_reason: str | None = None,
) -> Dict[str, Any]:
    """Return UI-friendly CRAG analytics."""
    return {
        "confidence": round(float(confidence), 3),
        "quality": quality,
        "retry_triggered": bool(retry_triggered),
        "query_rewritten": bool(query_rewritten),
        "chunks_before": int(chunks_before),
        "chunks_after": int(chunks_after),
        "reranking": bool(reranking),
        "latency_ms": int(latency_ms),
        "retrieval_ms": int(retrieval_ms),
        "reranking_ms": int(reranking_ms),
        "correction_ms": int(correction_ms),
        "retry_count": int(retry_count),
        "failure_reason": failure_reason,
        "memory_used": bool(memory_used),
        "query_type": query_type,
        "blocking_reason": blocking_reason,
    }
