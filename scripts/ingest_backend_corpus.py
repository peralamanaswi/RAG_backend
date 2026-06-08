"""Ingest backend domain PDFs into ChromaDB and BM25 without using Streamlit upload."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.embeddings import create_vector_store, get_collection_count
from utils.hybrid_search import build_or_load_bm25_index
from utils.pdf_loader import load_pdf_documents, split_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE_DOCS_DIR = PROJECT_ROOT / "source_documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
MIN_DOCUMENTS = 50


def get_pdf_paths() -> list[Path]:
    """Return backend source PDFs sorted by filename."""
    return sorted(SOURCE_DOCS_DIR.rglob("*.pdf"), key=lambda path: str(path).lower())


def pdf_path_signature(pdf_paths: list[Path]) -> str:
    """Create a signature for the backend corpus."""
    digest = hashlib.sha256()
    for pdf_path in pdf_paths:
        stat = pdf_path.stat()
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]


def main() -> None:
    """Build the backend corpus indexes."""
    load_dotenv(PROJECT_ROOT / ".env")
    pdf_paths = get_pdf_paths()

    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {SOURCE_DOCS_DIR}. Add your domain PDFs first.")

    if len(pdf_paths) < MIN_DOCUMENTS:
        logger.warning("Only %s PDFs found. Module 6 expects at least %s source documents.", len(pdf_paths), MIN_DOCUMENTS)

    logger.info("Ingesting %s backend PDF(s).", len(pdf_paths))
    page_documents, _ = load_pdf_documents(pdf_paths)
    if not page_documents:
        raise RuntimeError("No readable text found in backend PDFs.")

    chunks = split_documents(page_documents)
    signature = pdf_path_signature(pdf_paths)
    collection_name = f"legal_docs_bge_{signature}"

    vector_store = create_vector_store(chunks, CHROMA_DIR, collection_name)
    stored_count = get_collection_count(vector_store)
    build_or_load_bm25_index(collection_name, chunks)

    print(f"INGESTION_OK collection={collection_name} pdfs={len(pdf_paths)} chunks={len(chunks)} stored={stored_count}")


if __name__ == "__main__":
    main()
