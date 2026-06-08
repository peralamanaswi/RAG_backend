"""PDF loading and chunking utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - handled at runtime for optional OCR support.
    pytesseract = None

logger = logging.getLogger(__name__)
OCR_DPI = 200


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_uploaded_files(uploaded_files: Iterable, upload_dir: str | Path = "uploads") -> List[Path]:
    """Save Streamlit uploaded PDF files locally and return their paths."""
    directory = ensure_directory(upload_dir)
    saved_paths: List[Path] = []

    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        file_path = directory / safe_name
        try:
            file_path.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(file_path)
            logger.info("Saved uploaded file: %s", file_path)
        except Exception as exc:
            logger.exception("Failed to save uploaded file %s", safe_name)
            raise RuntimeError(f"Could not save {safe_name}: {exc}") from exc

    return saved_paths


def _configure_tesseract() -> None:
    """Use an optional Tesseract path from the environment."""
    if pytesseract is None:
        return

    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _ocr_page(page: fitz.Page) -> str:
    """Extract text from a scanned page using OCR."""
    if pytesseract is None:
        raise RuntimeError(
            "This PDF appears to be scanned. Install pytesseract and the Tesseract OCR app, "
            "or upload a text-based PDF."
        )

    _configure_tesseract()
    try:
        zoom = OCR_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "This PDF appears to be scanned, but Tesseract OCR is not installed or not on PATH. "
            "Install Tesseract OCR, then restart Streamlit. On Windows, you can set "
            "TESSERACT_CMD in .env to the full tesseract.exe path."
        ) from exc
    except Exception as exc:
        logger.exception("OCR failed for one PDF page.")
        raise RuntimeError(f"OCR failed while reading a scanned PDF page: {exc}") from exc


def extract_pdf_pages(pdf_path: str | Path) -> List[Document]:
    """Extract page-wise text from one PDF while preserving page numbers."""
    path = Path(pdf_path)
    documents: List[Document] = []

    try:
        with fitz.open(path) as pdf:
            for index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                extraction_method = "pymupdf"

                if not text:
                    logger.info("No selectable text on %s page %s; trying OCR.", path.name, index)
                    text = _ocr_page(page)
                    extraction_method = "ocr"

                if not text:
                    logger.warning("No text found on %s page %s after extraction.", path.name, index)
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": index,
                            "file_path": str(path),
                            "extraction_method": extraction_method,
                        },
                    )
                )
    except Exception as exc:
        logger.exception("Failed to extract PDF text from %s", path)
        raise RuntimeError(f"Could not read PDF {path.name}: {exc}") from exc

    return documents


def load_pdf_documents(pdf_paths: Iterable[str | Path]) -> Tuple[List[Document], str]:
    """Load multiple PDFs and return LangChain documents plus combined text."""
    all_pages: List[Document] = []

    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        logger.info("Extracted %s readable pages from PDF: %s", len(pages), Path(pdf_path).name)
        all_pages.extend(pages)

    combined_text = "\n\n".join(doc.page_content for doc in all_pages)
    logger.info("Loaded %s total readable PDF pages. Combined chars=%s.", len(all_pages), len(combined_text))
    return all_pages, combined_text


def split_documents(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split page documents into overlapping semantic chunks."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "Section", "Clause", ". ", "\n"],
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    logger.info(
        "Split %s page documents into %s chunks. chunk_size=%s, overlap=%s.",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
