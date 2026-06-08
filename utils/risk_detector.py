"""Simple legal risk detection from uploaded document text."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

RISK_KEYWORDS: Dict[str, List[str]] = {
    "Penalties": ["penalty", "penalties", "fine", "late fee", "default fee", "liquidated damages"],
    "Liability": ["liability", "liable", "indemnify", "indemnification", "damages", "losses"],
    "Hidden Obligations": ["shall", "must", "required to", "responsible for", "obligated to"],
    "Termination Risks": ["terminate", "termination", "breach", "notice period", "without cause", "for cause"],
    "Payment Risks": ["payment", "interest", "invoice", "non-payment", "overdue"],
}


def _sentences(text: str) -> List[str]:
    """Split text into readable sentence-like snippets."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def detect_risky_clauses(document_text: str, max_items_per_category: int = 5) -> Dict[str, List[str]]:
    """Detect potentially risky clauses by keyword matching."""
    if not document_text.strip():
        return {"No text found": ["Information not found in uploaded document."]}

    try:
        sentences = _sentences(document_text)
        risks: Dict[str, List[str]] = {}

        for category, keywords in RISK_KEYWORDS.items():
            matches: List[str] = []
            for sentence in sentences:
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in keywords):
                    matches.append(sentence[:500])
                if len(matches) >= max_items_per_category:
                    break
            risks[category] = matches or ["Information not found in uploaded document."]

        return risks
    except Exception as exc:
        logger.exception("Risk detection failed.")
        raise RuntimeError(f"Could not detect risky clauses: {exc}") from exc


def format_risk_report(risks: Dict[str, List[str]]) -> str:
    """Format risk results for download or display."""
    lines: List[str] = []
    for category, findings in risks.items():
        lines.append(f"{category}:")
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")
    return "\n".join(lines).strip()
