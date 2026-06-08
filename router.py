"""CRAG confidence helpers.

Adaptive query-type routing was removed during the CRAG migration. Retrieval
quality is now decided by validator.py after documents are retrieved.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def calculate_confidence(
    vector_similarity: float = 0.0,
    reranker_score: float = 0.0,
    retrieval_overlap: float = 0.0,
) -> float:
    """
    Deterministic CRAG confidence score.

    Formula:
    (0.5 * vector_similarity) + (0.3 * reranker_score) + (0.2 * retrieval_overlap)
    """
    try:
        score = (
            0.5 * _clamp(vector_similarity)
            + 0.3 * _clamp(reranker_score)
            + 0.2 * _clamp(retrieval_overlap)
        )
        return round(score, 3)
    except Exception:
        logger.exception("Confidence scoring failed. Falling back to 0.5.")
        return 0.5


def confidence_label(confidence: float) -> str:
    """Convert numeric confidence into CRAG quality labels."""
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "poor"


def _clamp(value: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))
