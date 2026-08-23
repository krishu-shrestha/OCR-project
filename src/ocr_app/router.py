from pathlib import Path

from ocr_app.config import ALLOWED_EXTENSIONS
from ocr_app.engines.base import OCREngine
from ocr_app.extractors.image_extractor import extract_text_from_image_bytes
from ocr_app.extractors.pdf_extractor import extract_text_from_pdf_bytes


def process_uploaded_file(filename: str, file_bytes: bytes, engine: OCREngine) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")
    if extension == ".pdf":
        return extract_text_from_pdf_bytes(file_bytes, engine)
    return extract_text_from_image_bytes(file_bytes, engine)
