# PDF Summarizer & Q&A

A small web app where you can upload a PDF, get a high-level summary, and ask questions about it.  
Answers are grounded in the PDF using local models (Ollama) and a vector database (Chroma).

## Features

- Upload a PDF
- Generate a short summary of the document
- Ask questions in natural language
- Uses local LLM + embeddings (no paid APIs)

## Tech Stack

- Backend: Python, FastAPI
- LLM runtime: Ollama (e.g. `llama3`)
- Embeddings: `nomic-embed-text` via Ollama
- Vector DB: Chroma (local)
- Frontend: simple HTML forms served by FastAPI

## Getting Started

1. Install Python 3.11+.
2. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>