from fastapi import FastAPI, HTTPException, UploadFile, File
import uuid
from pydantic import BaseModel
import ollama
from backend.text_chunking import chunk
from backend.embeddings import embed_text, embed_texts
from backend.vector_store import add_document, query_document, VectorStoreError
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNK_CHARS
from backend.text_extraction import PDFExtractionError, extract_text_from_pdf_bytes
from backend.summariser import summarise_doc

# ============================================================
# ==================== CLASSES & FUNCTIONS ====================
# ============================================================


class PDFIngestResponse(BaseModel):
    session_id: str
    pages: int
    chars: int
    preview: str
    summary: str

class AskRequest(BaseModel):
    session_id: str
    question: str

class SourceChunk(BaseModel):
    chunk_index: int | None = None
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

# ============================================================
# ============================ API ============================
# ============================================================

app = FastAPI()

@app.post("/ingest", response_model=PDFIngestResponse)
async def ingest_pdf(pdf_file: UploadFile = File(...)):
    # validate content tupe
    if pdf_file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")

    # read file into memory (bytes)
    try:
        pdf_bytes = await pdf_file.read() #await allows other tasks to occur whilst the file is being read
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")
    
    # enforce a size limit of 20 MB to avoid large RAM usage
    max_size_bytes = 20*1024*1024 # 20MB
    if len(pdf_bytes) > max_size_bytes:
        raise HTTPException(status_code=400, detail="PDF is too large (20MB limit)")
    
    # extract text + metadata
    try:
        result = extract_text_from_pdf_bytes(pdf_bytes)
    except PDFExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during PDF ingestion {e}")

    full_text = result["full_text"]

    # generate a session ID for this PDF
    session_id = str(uuid.uuid4())

    try:
        # chunk the text
        chunks = chunk(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="No non-empty chunks could be created from the PDF text")
        
        # generate summary
        summary = summarise_doc(chunks)

        # embed the chunks
        embeddings = embed_texts(chunks)

        # store in vector DB
        add_document(session_id=session_id, chunks=chunks, embeddings=embeddings)


    except HTTPException:
        raise # re-raise FastAPI exceptions as is

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during chunk/embed/store stage: {e}")
    
    # build preview
    preview_len = 300
    preview = full_text[:preview_len]

    # return structured metadata
    return PDFIngestResponse(
        session_id=session_id,
        pages=result["no_pages"],
        chars=result["no_chars"],
        preview=preview,
        summary=summary
    )



@app.post("/ask", response_model=AskResponse)
async def ask_pdf(request: AskRequest):

    # embed the question
    try:
        query_embedding = embed_text(request.question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {e}")


    # retrieve the relevant chunks from vector store
    try:
        results = query_document(
            session_id=request.session_id,
            query_embedding=query_embedding,
            n_results=5
        )
    
    except VectorStoreError as e:
        raise HTTPException(status_code=400, detail = str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store query failed: {e}")
    

    # chroma returns list of lists (one per query); we only have one query <- hence index 0
    # each query returns a dict with 'embeddings', 'documents', 'metadatas' as keys, each of which has a [[]] value
    # Output: docs  = ["chunk A text", "chunk B text"] and metas = [{"chunk_index": 3}, {"chunk_index": 7}]
    docs = results.get("documents", [[]])[0] #dict.get(key, default_value)
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant chunks found for this session_id")
    
    # build context string
    context_parts = []
    sources: list[SourceChunk] = []

    for doc, meta in zip(docs, metas): # using zip gives you ("chunk A text", {"chunk_index": 3})
        idx = meta.get("chunk_index")
        context_parts.append(doc)
        sources.append(SourceChunk(chunk_index=idx, text=doc))

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
    You are a helpful assistant answering questions about a PDF document.

    Use ONLY the information in the context below to answer the user's question.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {request.question}

    Answer clearly and concisely:
    """.strip()

    # call local LLM via Ollama
    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 128, "temperature": 0.2}
        )

        answer_text = resp["message"]["content"]

    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Failed to generate answer with LLM: {e}")

    return AskResponse(
        answer=answer_text,
        sources=sources
    )