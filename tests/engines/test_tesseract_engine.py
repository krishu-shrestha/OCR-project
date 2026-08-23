from PIL import Image, ImageDraw, ImageFont

from ocr_app.engines.tesseract_engine import TesseractEngine


def _make_text_image(text: str) -> Image.Image:
    image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((10, 25), text, fill="black", font=font)
    return image


def test_extract_text_reads_printed_text_from_image():
    engine = TesseractEngine()
    image = _make_text_image("HELLO WORLD")
    result = engine.extract_text(image)
    assert "HELLO" in result.upper()
