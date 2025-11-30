
from fastapi import FastAPI, HTTPException, UploadFile, File
import uuid
from io import BytesIO
from pypdf import PdfReader
from pydantic import BaseModel
from text_chunking import chunk


#************************************************************************************************#
#************************************************************************************************#

class PDFExtractionError(Exception):
    "custom exception for PDF extraction problems"
    pass

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> dict:
    """
    Extract text and metadata from a PDF loaded as bytes
    Returns:
        dict: {full_text, no_pages, no_chars}
    Raises PDFExtractionError for known problems
    """

    try:
        pdf_stream = BytesIO(pdf_bytes) # wrap bytes in a file like object
        reader = PdfReader(pdf_stream) # read the file into an object

    except Exception as e:
        raise PDFExtractionError(f"Failed to read PDF: {e}")
    

    if reader.is_encrypted:
        # not going to decrypt in V0
        raise PDFExtractionError(f"Failed to read PDF due to encryption")
    

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract text from page {i}: {e}")
        
        #handle image-only/no text
        if text is None:
            text = ""

        pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    no_pages = len(reader.pages)
    no_chars = len(full_text)

    if no_chars == 0:
        raise PDFExtractionError("No text could be extracted from the PDF (possibly image only)")

    return {
        "full_text": full_text,
        "no_pages": no_pages,
        "no_chars": no_chars
    }


#************************************************************************************************#
#************************************************************************************************#

class PDFIngestResponse(BaseModel):
    session_id: str
    pages: int
    chars: int
    preview: str


#************************************************************************************************#
#************************************************************************************************#

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

    # generate a session ID for this PDF
    session_id = str(uuid.uuid4())

    # build preview
    full_text = result["full_text"]
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    chunks = chunk(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    preview_len = 300
    preview = full_text[:preview_len]

    # return structured metadata
    return PDFIngestResponse(
        session_id=session_id,
        pages=result["no_pages"],
        chars=result["no_chars"],
        preview=preview
    )


#************************************************************************************************#
#************************************************************************************************#

# @app.get("/")
# def root():
#     return ""

# @app.get("/health")
# def health_check():
#     return {"status": "ok"}


# @app.post("/upload")
# async def upload(files: list[UploadFile] = File(...)):
#     sizes = []
#     for f in files:
#         sizes.append(len(await f.read()))
#     return {"sizes": sizes}












# from pydantic import BaseModel

# app = FastAPI()

# class Item(BaseModel):
#     text: str = None
#     is_done: bool = False

# items = []

# @app.get("/")
# def root():
#     return {"Hello": "World"}


# @app.get("/items", response_model=list[Item])
# def list_items(limit: int = 10):
#     return items[0:limit]

# @app.post("/items")
# def create_item(item: Item):
#     items.append(item)
#     return items

# @app.get("/items/{item_id}", response_model = Item)
# def get_item(item_id: int) -> Item:
#     if item_id < len(items):
#         item = items[item_id]

#     else:
#         raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

#     return item