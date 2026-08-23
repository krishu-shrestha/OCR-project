import io
import fitz  # PyMuPDF
from PIL import Image

def load_document_pages(filename: str, file_bytes: bytes) -> list[Image.Image]:
    """
    Converts a document (PDF or image) to a list of PIL Image pages.
    Supports rendering PDF pages as high-quality PNGs for UI display.
    """
    if not filename or not file_bytes:
        return []
        
    suffix = filename.split(".")[-1].lower()
    if suffix == "pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            images = []
            for page in doc:
                # 150 DPI is standard, providing a sharp preview without being slow
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                images.append(Image.open(io.BytesIO(img_data)))
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to render PDF pages: {e}")
    else:
        try:
            return [Image.open(io.BytesIO(file_bytes))]
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {e}")

def get_document_stats(text: str) -> dict:
    """Computes word count and character count from the extracted text."""
    if not text:
        return {"words": 0, "characters": 0}
        
    words = len(text.split())
    characters = len(text)
    return {
        "words": words,
        "characters": characters
    }
