import io

from PIL import Image, ImageDraw, ImageFont

from ocr_app.engines.tesseract_engine import TesseractEngine
from ocr_app.extractors.image_extractor import extract_text_from_image_bytes


def _make_text_image_bytes(text: str) -> bytes:
    image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((10, 25), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_text_from_image_bytes_returns_printed_text():
    engine = TesseractEngine()
    image_bytes = _make_text_image_bytes("WORKFLOW")
    result = extract_text_from_image_bytes(image_bytes, engine)
    assert "WORKFLOW" in result.upper()
