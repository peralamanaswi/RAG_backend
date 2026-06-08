# Backend Source Documents

Place domain-specific PDF files for Module 6 in this folder or its subfolders.

The Streamlit app automatically loads PDFs from this folder when no uploaded files are provided.

Requirements:

- Use `.pdf` files only.
- Keep filenames clear and domain-specific.
- Add at least 50 source documents for the academic requirement.
- Current organization:
  - `california_legal_docs/`
  - `india_legal_docs/`
- Restart Streamlit after adding or replacing files.

The app builds:

- ChromaDB embeddings in `chroma_db/`
- BM25 keyword indexes in `bm25_index/`
