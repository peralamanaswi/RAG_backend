"""Adaptive prompt templates for factual, explanatory, comparison, and follow-up RAG."""

from __future__ import annotations

from typing import Dict

from langchain_core.documents import Document

COMMON_RULES = """
STRICT INSTRUCTIONS:
1. Answer ONLY using retrieved document context.
2. Be legally precise; avoid oversimplified statements.
3. Do NOT make absolute claims unless explicitly stated.
4. Include all important legal conditions, limitations, exceptions, and time periods.
5. If a clause has conditions, such as territory, similar role, duration, notice, payment, or eligibility, explicitly mention them.
6. Combine multiple clauses when needed.
7. Use legal wording but explain clearly.
8. Never say "Yes" or "No" alone; explain why with evidence from the retrieved context.
9. Prefer precision over brevity.
10. If information is partial, say "likely" instead of making an absolute claim.
11. Do not hallucinate, guess, or use outside knowledge.
12. If the uploaded document does not contain the answer, say: Information not found in uploaded document.
"""

FACTUAL_TEMPLATE = """
You are an expert legal document analyst.
Provide a legally precise factual answer using retrieved context only.
{rules}

Retrieved Context:
{context}

Question:
{question}

Output Format:

Answer:
[Legally precise answer based strictly on retrieved clauses]

Source Pages:
[page numbers]

Simple Explanation:
[Easy English explanation]
"""

EXPLANATION_TEMPLATE = """
You are an expert legal document analyst.
Explain the legal effect clearly using retrieved context only.
{rules}

Retrieved Context:
{context}

Question:
{question}

Output Format:

Answer:
[Legally precise answer based strictly on retrieved clauses]

Source Pages:
[page numbers]

Simple Explanation:
[Easy English explanation]
"""

COMPARISON_TEMPLATE = """
You are an expert legal document analyst.
Compare the legal concepts, obligations, rights, or clauses clearly using retrieved context only.
{rules}

Retrieved Context:
{context}

Question:
{question}

Output Format:

Answer:
[Legally precise answer based strictly on retrieved clauses]

Source Pages:
[page numbers]

Simple Explanation:
[Easy English explanation]
"""

FOLLOW_UP_TEMPLATE = """
You are an expert legal document analyst answering a follow-up question.
Use the recent conversation context to resolve references like it, that, those, or previous.
Then answer only from the retrieved document context.
{rules}

Recent Conversation:
{memory_context}

Retrieved Context:
{context}

Question:
{question}

Output Format:

Answer:
[Legally precise answer based strictly on retrieved clauses]

Source Pages:
[page numbers]

Simple Explanation:
[Easy English explanation]
"""

PROMPT_TEMPLATES: Dict[str, str] = {
    "factual": FACTUAL_TEMPLATE,
    "explanation": EXPLANATION_TEMPLATE,
    "comparison": COMPARISON_TEMPLATE,
    "follow-up": FOLLOW_UP_TEMPLATE,
}


def get_prompt_template(prompt_type: str) -> str:
    """Return the prompt template for the current retrieval path."""
    return PROMPT_TEMPLATES.get(prompt_type, EXPLANATION_TEMPLATE)


def format_context(documents: list[Document], max_context_tokens: int = 2500) -> str:
    """Format retrieved documents and trim to a simple token budget."""
    chunks = []
    for doc in documents:
        chunks.append(
            "Source File: {source}\nSource Page: {page}\nContent:\n{content}".format(
                source=doc.metadata.get("source", "Uploaded document"),
                page=doc.metadata.get("page", "Unknown"),
                content=doc.page_content.strip(),
            )
        )

    text = "\n\n".join(chunks)
    words = text.split()
    if len(words) > max_context_tokens:
        text = " ".join(words[:max_context_tokens])
    return text


def build_adaptive_prompt(
    question: str,
    documents: list[Document],
    prompt_type: str,
    max_context_tokens: int,
    memory_context: str = "",
) -> str:
    """Create the final prompt string sent to the LLM."""
    template = get_prompt_template(prompt_type)
    return template.format(
        rules=COMMON_RULES,
        context=format_context(documents, max_context_tokens=max_context_tokens),
        question=question,
        memory_context=memory_context or "No previous conversation context.",
    )
