"""Document summary and clause extraction helpers."""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.prompts import PromptTemplate

from utils.rag_pipeline import deprecated_model_error_message, is_deprecated_model_error, get_groq_llm

logger = logging.getLogger(__name__)

MAX_ANALYSIS_CHARS = 14000


SUMMARY_TEMPLATE = """
You are an AI Legal Document Assistant.

Use ONLY the document text below. Do not use outside legal knowledge.
If a requested detail is not present, write "Information not found in uploaded document."
Keep the language beginner-friendly.

Document Text:
{document_text}

Return the analysis in this exact structure:

Document Summary:
- 

Important Clauses:
- 

Obligations:
- 

Deadlines:
- 
"""


def _trim_text(text: str, max_chars: int = MAX_ANALYSIS_CHARS) -> str:
    """Keep analysis prompts compact for student-level usage."""
    clean_text = " ".join(text.split())
    return clean_text[:max_chars]


def generate_document_analysis(document_text: str) -> str:
    """Generate summary, clauses, obligations, and deadlines with Groq."""
    if not document_text.strip():
        return "Information not found in uploaded document."

    try:
        prompt = PromptTemplate(
            template=SUMMARY_TEMPLATE,
            input_variables=["document_text"],
        )
        chain = prompt | get_groq_llm()
        response = chain.invoke({"document_text": _trim_text(document_text)})
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        if is_deprecated_model_error(exc):
            logger.exception("Groq model error while generating document analysis: %s", exc)
            raise RuntimeError(deprecated_model_error_message()) from exc

        if "api" in str(exc).lower() or "connection" in str(exc).lower() or "groq" in str(exc).lower():
            logger.exception("Groq API failure while generating document analysis.")
            raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

        logger.exception("Document analysis failed.")
        raise RuntimeError(f"Could not generate document analysis: {exc}") from exc


def split_analysis_sections(analysis_text: str) -> Dict[str, str]:
    """Split the LLM analysis into UI-friendly sections."""
    section_names = [
        "Document Summary:",
        "Important Clauses:",
        "Obligations:",
        "Deadlines:",
    ]
    sections: Dict[str, str] = {name.rstrip(":"): "" for name in section_names}

    current = None
    for line in analysis_text.splitlines():
        stripped = line.strip()
        if stripped in section_names:
            current = stripped.rstrip(":")
            continue
        if current and stripped:
            sections[current] += stripped + "\n"

    if not any(value.strip() for value in sections.values()):
        sections["Document Summary"] = analysis_text

    return {key: value.strip() for key, value in sections.items()}
