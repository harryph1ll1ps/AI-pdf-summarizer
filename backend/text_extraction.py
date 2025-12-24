from io import BytesIO
from pypdf import PdfReader

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

