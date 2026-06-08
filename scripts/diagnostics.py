"""End-to-end diagnostics for the Legal Document Explainer project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.embeddings import create_vector_store, get_collection_count, run_similarity_search
from utils.hybrid_search import build_or_load_bm25_index, hybrid_retrieve
from utils.pdf_loader import load_pdf_documents, split_documents
from utils.rag_pipeline import ask_question, build_retrieval_qa, log_langsmith_status
from utils.summarizer import generate_document_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_legal_pdf(pdf_path: Path) -> None:
    """Create a small text-based legal PDF for diagnostics."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    text = (
        "Service Agreement.\n\n"
        "Payment Terms: The client must pay all invoices within 15 days of receipt.\n\n"
        "Termination Clause: Either party may terminate this agreement by giving 30 days written notice.\n\n"
        "Obligations: The tenant must maintain the premises and report damage promptly.\n\n"
        "Penalty: Late payments may result in a late fee of 2 percent per month."
    )
    page.insert_textbox(fitz.Rect(72, 72, 520, 260), text, fontsize=11)
    document.save(pdf_path)
    document.close()
    logger.info("Created sample legal PDF: %s", pdf_path)


def main() -> None:
    """Run the diagnostics workflow."""
    load_dotenv(PROJECT_ROOT / ".env")
    logger.info("Starting Legal Document Explainer diagnostics.")

    langsmith_status = log_langsmith_status()
    logger.info("LangSmith status: %s", langsmith_status)

    sample_pdf = PROJECT_ROOT / "uploads" / "_diagnostics_sample.pdf"
    create_sample_legal_pdf(sample_pdf)

    pages, document_text = load_pdf_documents([sample_pdf])
    logger.info("PDF processing complete. pages=%s, chars=%s", len(pages), len(document_text))

    chunks = split_documents(pages)
    logger.info("Chunking complete. chunks=%s", len(chunks))

    vector_store = create_vector_store(
        documents=chunks,
        persist_directory=PROJECT_ROOT / "chroma_db",
        collection_name="legal_docs_bge_diagnostics",
    )
    stored_count = get_collection_count(vector_store)
    logger.info("Chroma storage check complete. stored_chunks=%s", stored_count)
    build_or_load_bm25_index("legal_docs_bge_diagnostics", chunks)

    retrieved_chunks = run_similarity_search(
        vector_store,
        query="What is the termination notice period?",
        top_k=3,
    )
    if not retrieved_chunks:
        raise RuntimeError("Similarity search returned no chunks.")

    hybrid_chunks = hybrid_retrieve(
        query="What is the termination notice period?",
        vector_store=vector_store,
        top_k=10,
    )
    if not hybrid_chunks:
        raise RuntimeError("Hybrid retrieval returned no chunks.")
    logger.info("Hybrid retrieval diagnostics returned %s chunks.", len(hybrid_chunks))

    analysis = generate_document_analysis(document_text)
    logger.info("Document analysis generated. preview=%s", analysis[:500].replace("\n", " "))

    qa_chain = build_retrieval_qa(vector_store, top_k=3)
    result = ask_question(qa_chain, "What is the termination notice period?")
    logger.info("Final answer:\n%s", result["answer"])
    logger.info("Answer source pages: %s", result["pages"])

    sample_pdf.unlink(missing_ok=True)
    logger.info("Diagnostics completed successfully.")
    print("DIAGNOSTICS_OK")


if __name__ == "__main__":
    main()
