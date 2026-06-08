"""FastAPI backend for the Legal Document Explainer RAG system."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DOCS_DIR = BASE_DIR / "source_documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
MIN_DOCUMENTS = 50
DEFAULT_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="Legal Document Explainer Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
collection_name = ""
loaded_files: List[str] = []


class AskRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, Any]] = []


def get_pdf_paths() -> List[Path]:
    """Return all backend corpus PDFs recursively."""
    return sorted(SOURCE_DOCS_DIR.rglob("*.pdf"), key=lambda path: str(path).lower())


def corpus_signature(pdf_paths: List[Path]) -> str:
    """Create a stable collection signature for the current corpus."""
    digest = hashlib.sha256()
    for pdf_path in pdf_paths:
        stat = pdf_path.stat()
        digest.update(str(pdf_path.relative_to(SOURCE_DOCS_DIR)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]


def ensure_vector_store():
    """Load or build the backend corpus vector store."""
    global vector_store, collection_name, loaded_files
    if vector_store is not None:
        return vector_store

    from utils.embeddings import create_vector_store, get_collection_count
    from utils.hybrid_search import build_or_load_bm25_index
    from utils.pdf_loader import load_pdf_documents, split_documents

    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        raise RuntimeError("No backend PDFs found in backend/source_documents.")

    signature = corpus_signature(pdf_paths)
    collection_name = f"legal_docs_bge_{signature}"
    loaded_files = [str(path.relative_to(SOURCE_DOCS_DIR)) for path in pdf_paths]

    logger.info("Loading backend corpus. pdfs=%s collection=%s", len(pdf_paths), collection_name)
    page_documents, _ = load_pdf_documents(pdf_paths)
    if not page_documents:
        raise RuntimeError("No readable text found in backend PDFs.")

    chunks = split_documents(page_documents)
    vector_store = create_vector_store(chunks, CHROMA_DIR, collection_name)
    build_or_load_bm25_index(collection_name, chunks)
    logger.info("Backend corpus ready. chunks=%s stored=%s", len(chunks), get_collection_count(vector_store))
    return vector_store


def count_stored_chunks() -> int | None:
    """Count Chroma chunks only after the vector store is already loaded."""
    if vector_store is None:
        return None
    from utils.embeddings import get_collection_count

    return get_collection_count(vector_store)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.get("/corpus/status")
def corpus_status() -> Dict[str, Any]:
    """Return backend corpus status."""
    pdf_paths = get_pdf_paths()
    return {
        "pdf_count": len(pdf_paths),
        "minimum_required": MIN_DOCUMENTS,
        "collection_name": collection_name,
        "stored_chunks": count_stored_chunks(),
        "folders": sorted({path.parent.name for path in pdf_paths}),
        "loaded_files": loaded_files or [str(path.relative_to(SOURCE_DOCS_DIR)) for path in pdf_paths],
    }


@app.post("/corpus/ingest")
def ingest() -> Dict[str, Any]:
    """Build or load the backend corpus indexes."""
    try:
        store = ensure_vector_store()
        from utils.embeddings import get_collection_count

        return {
            "status": "ready",
            "collection_name": collection_name,
            "pdf_count": len(get_pdf_paths()),
            "stored_chunks": get_collection_count(store),
        }
    except Exception as exc:
        logger.exception("Backend ingestion failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    """Answer a legal question from the backend corpus."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        store = ensure_vector_store()
        from utils.rag_pipeline import ask_crag_question

        return ask_crag_question(store, request.question, chat_history=request.chat_history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Question answering failed.")
        raise HTTPException(status_code=500, detail="Question answering failed.") from exc
