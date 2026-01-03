PDF Q&A - Local Retrieval-Augmented Generation (RAG)
===================================================

A local-first RAG app: upload a PDF, get an automatic summary, then ask questions answered strictly from the document.
Runs entirely locally using Ollama (LLM + embeddings) and ChromaDB (vector search).

Features
--------
- PDF upload with validation and size limits
- Page-by-page text extraction
- Automatic document summarisation
- Overlapping word-based chunking
- Local embeddings + semantic retrieval (ChromaDB)
- Grounded Q&A with returned source chunks
- Minimal frontend UI

How it works (RAG pipeline)
---------------------------
1) Ingest: validate PDF, extract text in memory
2) Chunk: normalise + split into overlapping chunks (with character limits)
3) Embed: generate embeddings for chunks (Ollama)
4) Store: persist chunks + embeddings in ChromaDB (session-scoped)
5) Ask: embed question -> retrieve top-k chunks -> LLM answers from retrieved context only

Tech Stack
----------
Backend: FastAPI, Ollama, ChromaDB, PyPDF, Jinja2
Frontend: HTML / CSS / JavaScript

Run locally
-----------
Prereqs:
- Python 3.10+
- Ollama installed and running: https://ollama.com

Pull models:
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text

Setup:
  git clone https://github.com/yourusername/pdf-rag.git
  cd pdf-rag
  python -m venv venv
  source venv/bin/activate        (Windows: venv\Scripts\activate)
  pip install -r requirements.txt

Start:
  uvicorn backend.main:app --reload

Open:
  http://localhost:8000

Project structure
-----------------
backend/
  main.py            (FastAPI routes)
  config.py          (constants)
  text_extraction.py (PDF parsing)
  text_chunking.py   (chunking)
  embeddings.py      (Ollama embeddings)
  summariser.py      (summarisation)
  vector_store.py    (ChromaDB)

frontend/
  templates/index.html
  static/app.js
  static/styles.css

Limitations
-----------
- Image-only PDFs not supported
- Single-user / local usage (no auth)
- Local persistence only (ChromaDB)
- No streaming responses

Future improvements
-------------------
- Highlight sources in the UI
- Stream model responses
- Support additional file types (DOCX/TXT)
- Multi-document search and saved sessions