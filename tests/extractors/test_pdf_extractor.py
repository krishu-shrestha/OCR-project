import io

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from ocr_app.engines.base import OCREngine
from ocr_app.engines.tesseract_engine import TesseractEngine
from ocr_app.extractors.pdf_extractor import extract_text_from_pdf_bytes


class CountingDummyEngine(OCREngine):
    def __init__(self):
        self.call_count = 0

    def extract_text(self, image):
        self.call_count += 1
        return "should not be used"


def _build_text_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_scanned_pdf_bytes(text: str) -> bytes:
    image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((10, 25), text, fill="black", font=font)

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=100)
    page.insert_image(pymupdf.Rect(0, 0, 400, 100), stream=_image_to_png_bytes(image))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extracts_direct_text_without_invoking_ocr_engine():
    engine = CountingDummyEngine()
    pdf_bytes = _build_text_pdf_bytes("Direct text page")
    result = extract_text_from_pdf_bytes(pdf_bytes, engine)
    assert "Direct text page" in result
    assert engine.call_count == 0


def test_falls_back_to_ocr_for_scanned_page():
    engine = TesseractEngine()
    pdf_bytes = _build_scanned_pdf_bytes("SCANNED PAGE")
    result = extract_text_from_pdf_bytes(pdf_bytes, engine)
    assert "SCANNED" in result.upper()


def test_falls_back_to_ocr_for_whitespace_only_page():
    # A page whose extractable text is whitespace-only (e.g. stray spaces
    # left over from document authoring tools) must be treated as "no
    # text" and trigger the OCR fallback, not be accepted as direct text.
    engine = CountingDummyEngine()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "   ")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf_bytes(pdf_bytes, engine)
    assert engine.call_count == 1
    assert "should not be used" in result
