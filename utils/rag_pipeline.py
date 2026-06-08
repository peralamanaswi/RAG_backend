"""Naive RAG pipeline using LangChain RetrievalQA and Groq."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import BadRequestError
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from pydantic import ConfigDict

try:
    from langchain.chains import RetrievalQA
except ModuleNotFoundError:
    from langchain_classic.chains import RetrievalQA

from utils.hybrid_search import hybrid_retrieve
from utils.query_rewriter import rewrite_query
from utils.reranker import rerank_documents

logger = logging.getLogger(__name__)
load_dotenv()

FAST_MODEL = "llama-3.1-8b-instant"
LEGAL_ANALYSIS_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODEL = LEGAL_ANALYSIS_MODEL
DEPRECATED_MODEL_REPLACEMENTS = {
    "llama3-8b-8192": FAST_MODEL,
    "llama3-70b-8192": LEGAL_ANALYSIS_MODEL,
}

LEGAL_QA_TEMPLATE = """
You are an expert legal document analyst.

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


def log_langsmith_status() -> Dict[str, Any]:
    """Log whether LangSmith tracing environment variables are configured."""
    tracing_value = os.getenv("LANGCHAIN_TRACING_V2", "")
    tracing_enabled = tracing_value.lower() == "true"
    api_key_present = bool(os.getenv("LANGCHAIN_API_KEY"))
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    status = {
        "tracing_enabled": tracing_enabled,
        "api_key_present": api_key_present,
        "project": project,
    }

    if tracing_enabled and api_key_present:
        logger.info("LangSmith tracing is active. project=%s", project)
    elif tracing_enabled and not api_key_present:
        logger.warning("LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing.")
    else:
        logger.info(
            "LangSmith tracing is inactive. LANGCHAIN_TRACING_V2=%r, LANGCHAIN_API_KEY present=%s, project=%s",
            tracing_value or "<unset>",
            api_key_present,
            project,
        )

    return status


def format_documents_for_prompt(documents: List[Document]) -> str:
    """Format retrieved chunks exactly like the context shown to the LLM."""
    formatted_chunks: List[str] = []
    for doc in documents:
        formatted_chunks.append(
            "Source File: {source}\nSource Page: {page}\nContent:\n{content}".format(
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", "Unknown"),
                content=doc.page_content,
            )
        )
    return "\n\n".join(formatted_chunks)


def normalize_groq_model(model_name: str | None = None) -> str:
    """Return a supported Groq model, replacing deprecated model ids."""
    selected_model = model_name or DEFAULT_MODEL
    replacement = DEPRECATED_MODEL_REPLACEMENTS.get(selected_model)
    if replacement:
        logger.warning(
            "Groq model '%s' is deprecated. Using supported model '%s' instead.",
            selected_model,
            replacement,
        )
        return replacement
    return selected_model


def is_deprecated_model_error(exc: Exception) -> bool:
    """Check whether a Groq/LangChain exception is caused by a deprecated model."""
    message = str(exc).lower()
    return (
        "decommissioned" in message
        or "deprecated" in message
        or any(model in message for model in DEPRECATED_MODEL_REPLACEMENTS)
    )


def deprecated_model_error_message() -> str:
    """Return a clear action-oriented deprecated model message."""
    return (
        "Groq rejected a deprecated model. Use "
        f"'{FAST_MODEL}' for fast responses or '{LEGAL_ANALYSIS_MODEL}' "
        "for stronger legal document analysis."
    )


class AdvancedLegalRetriever(BaseRetriever):
    """Advanced retriever: query rewriting, hybrid search, and reranking."""

    vector_store: Any
    candidate_k: int = 10
    final_k: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        optimized_query = rewrite_query(query)
        try:
            candidates = hybrid_retrieve(optimized_query, self.vector_store, top_k=self.candidate_k)
        except Exception as exc:
            logger.exception("Hybrid retrieval failed. Falling back to dense retrieval.")
            candidates = self.vector_store.similarity_search(optimized_query, k=self.final_k)

        if not candidates:
            logger.warning("No chunks retrieved for query=%r optimized_query=%r", query, optimized_query)
            return []

        try:
            final_docs = rerank_documents(optimized_query, candidates, top_k=self.final_k)
        except Exception as exc:
            logger.exception("Reranker failed. Falling back to dense top %s retrieval.", self.final_k)
            final_docs = self.vector_store.similarity_search(optimized_query, k=self.final_k)
        logger.info(
            "Advanced retrieval complete. original_query=%r optimized_query=%r candidates=%s final=%s",
            query,
            optimized_query,
            len(candidates),
            len(final_docs),
        )
        return final_docs


def get_groq_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatGroq:
    """Create the Groq chat model."""
    log_langsmith_status()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    supported_model = normalize_groq_model(model_name)
    logger.info("Initializing Groq ChatGroq client. model=%s, temperature=%s", supported_model, temperature)
    return ChatGroq(
        api_key=api_key,
        model=supported_model,
        temperature=temperature,
    )


def build_retrieval_qa(vector_store, top_k: int = 3, candidate_k: int = 10) -> RetrievalQA:
    """Build a RetrievalQA chain over the Chroma vector store."""
    prompt = PromptTemplate(
        template=LEGAL_QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    document_prompt = PromptTemplate(
        template="Source File: {source}\nSource Page: {page}\nContent:\n{page_content}",
        input_variables=["source", "page", "page_content"],
    )

    retriever = AdvancedLegalRetriever(vector_store=vector_store, candidate_k=candidate_k, final_k=top_k)

    return RetrievalQA.from_chain_type(
        llm=get_groq_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "document_prompt": document_prompt},
    )


def ask_question(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """Ask a question and return answer plus source metadata."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        retrieved_documents: List[Document] = []
        retriever = getattr(qa_chain, "retriever", None)
        if retriever is not None:
            retrieved_documents = retriever.invoke(question)
            logger.info("Retriever returned %s chunks before LLM call.", len(retrieved_documents))
            if not retrieved_documents:
                return {
                    "answer": "Information not found in uploaded document.",
                    "pages": "Unknown",
                    "sources": "Uploaded document",
                    "source_documents": [],
                }
            for index, doc in enumerate(retrieved_documents, start=1):
                logger.info(
                    "Top retrieved chunk %s | source=%s | page=%s | chunk_id=%s | preview=%s",
                    index,
                    doc.metadata.get("source", "Unknown"),
                    doc.metadata.get("page", "Unknown"),
                    doc.metadata.get("chunk_id", "Unknown"),
                    doc.page_content[:500].replace("\n", " "),
                )
            logger.info(
                "Final retrieved context sent to Groq/Llama prompt:\n%s",
                format_documents_for_prompt(retrieved_documents),
            )

        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "Information not found in uploaded document.")
        source_documents: List[Document] = result.get("source_documents", [])
        pages = sorted({str(doc.metadata.get("page", "Unknown")) for doc in source_documents})
        sources = sorted({str(doc.metadata.get("source", "Uploaded document")) for doc in source_documents})

        return {
            "answer": answer,
            "pages": ", ".join(pages) if pages else "Unknown",
            "sources": ", ".join(sources) if sources else "Uploaded document",
            "source_documents": source_documents,
        }
    except Exception as exc:
        if isinstance(exc, BadRequestError) or is_deprecated_model_error(exc):
            logger.exception("Groq model error while answering a question: %s", exc)
            raise RuntimeError(deprecated_model_error_message()) from exc

        if "api" in str(exc).lower() or "connection" in str(exc).lower() or "groq" in str(exc).lower():
            logger.exception("Groq API failure while answering question.")
            raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

        logger.exception("RAG question answering failed.")
        raise RuntimeError(f"Could not answer the question: {exc}") from exc


def ask_crag_question(vector_store, question: str, chat_history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Answer a question with Corrective RAG retrieval validation and one correction retry."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    total_started = None
    try:
        from analytics import build_crag_analytics, elapsed_ms, now_ms
        from memory import detect_conversational_references, get_memory_window, get_previous_chunks, get_recent_memory_context
        from prompt import build_adaptive_prompt
        from retriever import retrieve_crag
        from router import confidence_label

        total_started = now_ms()
        try:
            has_documents = bool(getattr(vector_store, "_collection", None)) and vector_store._collection.count() > 0
        except Exception:
            has_documents = True
        if not has_documents:
            return {
                "answer": "No documents uploaded.",
                "pages": "Unknown",
                "sources": "Uploaded document",
                "source_documents": [],
                "analytics": {},
                "confidence_label": "poor",
                "retrieved_chunks": [],
            }

        retrieval_result = retrieve_crag(question, vector_store, chat_history=chat_history or [])
        confidence = float(retrieval_result.get("confidence", 0.5))
        quality = str(retrieval_result.get("quality", "medium"))
        docs: List[Document] = retrieval_result.get("documents", [])
        chunks_before = int(retrieval_result.get("chunks_before", 0))
        chunks_after = int(retrieval_result.get("chunks_after", len(docs)))
        query_type = str(retrieval_result.get("query_type", "normal"))
        logger.info(
            "CRAG retrieval result. query_type=%s confidence=%s quality=%s chunks_before=%s "
            "chunks_after=%s retry_triggered=%s failure_reason=%s optimized_query=%r",
            query_type,
            confidence,
            quality,
            chunks_before,
            chunks_after,
            retrieval_result.get("retry_triggered"),
            retrieval_result.get("failure_reason"),
            retrieval_result.get("optimized_query"),
        )

        analytics = build_crag_analytics(
            confidence=confidence,
            quality=quality,
            retry_triggered=bool(retrieval_result.get("retry_triggered")),
            query_rewritten=bool(retrieval_result.get("query_rewritten")),
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            reranking=bool(retrieval_result.get("reranking", True)),
            latency_ms=elapsed_ms(total_started),
            retrieval_ms=int(retrieval_result.get("retrieval_ms", 0)),
            reranking_ms=int(retrieval_result.get("reranking_ms", 0)),
            correction_ms=int(retrieval_result.get("correction_ms", 0)),
            retry_count=int(retrieval_result.get("retry_count", 0)),
            failure_reason=retrieval_result.get("failure_reason"),
            memory_used=bool(retrieval_result.get("memory_used")),
            query_type=query_type,
            blocking_reason=None,
        )

        if chunks_after == 0:
            analytics["blocking_reason"] = "chunks_after_zero"
            logger.warning(
                "CRAG blocking answer. reason=chunks_after_zero query_type=%s confidence=%s "
                "quality=%s chunks_before=%s chunks_after=%s failure_reason=%s",
                query_type,
                confidence,
                quality,
                chunks_before,
                chunks_after,
                retrieval_result.get("failure_reason"),
            )
            return {
                "answer": "Information not found in uploaded document.",
                "pages": "Unknown",
                "sources": "Uploaded document",
                "source_documents": docs,
                "analytics": analytics,
                "confidence_label": confidence_label(confidence),
                "retrieved_chunks": _serialize_chunks(docs),
            }
        logger.info(
            "CRAG answer allowed. query_type=%s confidence=%s quality=%s chunks_before=%s "
            "chunks_after=%s blocking_reason=None",
            query_type,
            confidence,
            quality,
            chunks_before,
            chunks_after,
        )

        memory_window = get_memory_window(question, chat_history or [], confidence) if detect_conversational_references(question) else 0
        memory_context = get_recent_memory_context(chat_history or [], window=memory_window) if memory_window else ""

        previous_chunks = get_previous_chunks(chat_history or [], window=memory_window) if memory_window else []
        if previous_chunks:
            memory_context = (
                f"{memory_context}\n\nPreviously Retrieved Chunks:\n"
                + "\n\n".join(chunk.get("preview", "") for chunk in previous_chunks[:3])
            ).strip()

        prompt_type = "follow-up" if memory_window else "explanation"
        prompt_text = build_adaptive_prompt(
            question=question,
            documents=docs,
            prompt_type=prompt_type,
            max_context_tokens=4500 if query_type in {"summary", "legal_scenario"} else (4000 if quality == "high" else 3000),
            memory_context=memory_context,
        )
        logger.info(
            "Final CRAG context passed to LLM. confidence=%s quality=%s retrieved_chunks=%s context_preview=%s",
            confidence,
            quality,
            len(docs),
            format_documents_for_prompt(docs)[:4000],
        )

        try:
            response = get_groq_llm().invoke(prompt_text)
            answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
        except Exception as exc:
            if isinstance(exc, BadRequestError) or is_deprecated_model_error(exc):
                logger.exception("Groq model error while answering CRAG question.")
                raise RuntimeError(deprecated_model_error_message()) from exc
            logger.exception("Groq API failure while answering CRAG question.")
            raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

        pages = sorted({str(doc.metadata.get("page", "Unknown")) for doc in docs})
        sources = sorted({str(doc.metadata.get("source", "Uploaded document")) for doc in docs})
        analytics["latency_ms"] = elapsed_ms(total_started)
        logger.info("CRAG analytics: %s", analytics)

        return {
            "answer": answer or "Information not found in uploaded document.",
            "pages": ", ".join(pages) if pages else "Unknown",
            "sources": ", ".join(sources) if sources else "Uploaded document",
            "source_documents": docs,
            "analytics": analytics,
            "confidence_label": confidence_label(confidence),
            "retrieved_chunks": _serialize_chunks(docs),
        }
    except RuntimeError:
        raise
    except Exception:
        logger.exception("CRAG failed. Returning safe fallback.")
        try:
            from analytics import build_crag_analytics, elapsed_ms

            total_ms = elapsed_ms(total_started) if total_started else 0
            analytics = build_crag_analytics(
                confidence=0.5,
                quality="poor",
                retry_triggered=True,
                query_rewritten=False,
                chunks_before=0,
                chunks_after=0,
                reranking=False,
                latency_ms=total_ms,
                retry_count=1,
                failure_reason="crag_failure",
            )
        except Exception:
            analytics = {}
        return {
            "answer": "Information not found in uploaded document.",
            "pages": "Unknown",
            "sources": "Uploaded document",
            "source_documents": [],
            "analytics": analytics,
            "confidence_label": "low",
            "retrieved_chunks": [],
        }


def ask_adaptive_question(vector_store, question: str, chat_history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Backward-compatible alias for the migrated CRAG pipeline."""
    return ask_crag_question(vector_store, question, chat_history=chat_history)


def _serialize_chunks(documents: List[Document]) -> List[Dict[str, Any]]:
    """Store lightweight previous chunks in chat history for follow-up questions."""
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        chunks.append(
            {
                "source": doc.metadata.get("source", "Uploaded document"),
                "page": doc.metadata.get("page", "Unknown"),
                "preview": doc.page_content[:900],
            }
        )
    return chunks
