# Legal RAG Backend

FastAPI backend for the Legal Document Explainer. It stores the fixed domain corpus in `source_documents/`, builds ChromaDB and BM25 indexes, and answers questions with Groq + LangChain RAG.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

To use Chroma Cloud instead of local `chroma_db/`, also add:

```env
CHROMA_API_KEY=your_chroma_cloud_api_key_here
CHROMA_TENANT=your_chroma_tenant_id_here
CHROMA_DATABASE=your_chroma_database_name_here
```

Run locally:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/corpus/status`
- `POST http://127.0.0.1:8000/corpus/ingest`
- `POST http://127.0.0.1:8000/ask`

## Render Deployment

1. Create a new Render Web Service from this backend repo.
2. Use Python environment.
3. Render can read `render.yaml`, or configure manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - `GROQ_API_KEY`
   - `CHROMA_API_KEY`
   - `CHROMA_TENANT`
   - `CHROMA_DATABASE`
5. Deploy.
6. Visit `/health` to verify the service is running.

If Chroma Cloud variables are present, embeddings are stored in Chroma Cloud. Without them, the backend falls back to local `chroma_db/`. Generated folders like `chroma_db/` and `bm25_index/` are ignored because they can be rebuilt from the committed source PDFs.
