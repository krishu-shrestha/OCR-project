import pymupdf
from PIL import Image

from ocr_app.config import PDF_RASTER_DPI
from ocr_app.engines.base import OCREngine
from ocr_app.preprocessing import preprocess_image


def extract_text_from_pdf_bytes(file_bytes: bytes, engine: OCREngine) -> str:
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        page_texts = []
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                pixmap = page.get_pixmap(dpi=PDF_RASTER_DPI)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                processed = preprocess_image(image)
                text = engine.extract_text(processed).strip()
            page_texts.append(f"--- Page {page_number} ---\n{text}")
        return "\n\n".join(page_texts)
