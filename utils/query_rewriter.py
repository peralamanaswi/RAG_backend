"""Query rewriting for legal retrieval."""

from __future__ import annotations

import logging

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

REWRITER_PROMPT = """
Convert the user query into a legal retrieval-friendly query.

Rules:
1. Preserve meaning
2. Improve retrieval relevance
3. Add legal synonyms when useful
4. No hallucination
5. No assumptions
6. Keep concise (<50 words)

User Query:
{query}

Optimized Query:
"""


def rewrite_query(query: str) -> str:
    """Rewrite a user query for better legal retrieval; fall back to original query."""
    clean_query = query.strip()
    if not clean_query:
        return clean_query

    try:
        from utils.rag_pipeline import get_groq_llm

        prompt = PromptTemplate(template=REWRITER_PROMPT, input_variables=["query"])
        chain = prompt | get_groq_llm(temperature=0.0)
        response = chain.invoke({"query": clean_query})
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        rewritten = " ".join(rewritten.split())
        if not rewritten:
            return clean_query
        words = rewritten.split()
        if len(words) > 50:
            rewritten = " ".join(words[:50])
        logger.info("Query rewritten. original=%r rewritten=%r", clean_query, rewritten)
        return rewritten
    except Exception as exc:
        logger.exception("Query rewrite failed. Falling back to original query.")
        return clean_query
