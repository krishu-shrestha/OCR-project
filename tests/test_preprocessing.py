from PIL import Image, ImageDraw, ImageFont

from ocr_app.engines.tesseract_engine import TesseractEngine
from ocr_app.preprocessing import preprocess_image


def test_preprocess_image_converts_to_grayscale():
    image = Image.new("RGB", (10, 10), color=(120, 200, 50))
    processed = preprocess_image(image)
    assert processed.mode == "L"


def test_preprocess_image_does_not_binarize():
    image = Image.new("RGB", (10, 10), color=(120, 120, 120))
    processed = preprocess_image(image)
    pixel_values = set(processed.tobytes())
    # A mid-gray fill should stay mid-gray, not collapse to pure black/white
    # the way the old point(pixel > 128) threshold used to force it to.
    assert not pixel_values.issubset({0, 255})


def _draw_text_image(text, size, text_color, background_color, mode="RGB"):
    image = Image.new(mode, size, color=background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((10, 25), text, fill=text_color, font=font)
    return image


def test_low_contrast_text_survives_preprocessing_and_ocr():
    # Regression test for the hard global threshold bug: gray text on a
    # light-gray background used to be destroyed by point(128), producing
    # empty OCR output. Grayscale-only preprocessing preserves it because
    # Tesseract does its own adaptive binarization internally.
    image = _draw_text_image(
        "LOWCONTRAST",
        (400, 100),
        text_color=(150, 150, 150),
        background_color=(205, 205, 205),
    )
    processed = preprocess_image(image)
    engine = TesseractEngine()
    result = engine.extract_text(processed)
    assert result.strip() != ""
    assert "LOWCONTRAST" in result.upper()


def test_transparent_background_text_survives_preprocessing_and_ocr():
    # Regression test: converting an RGBA image straight to grayscale does
    # not composite transparent pixels onto a background, so black text on
    # a transparent background (RGB channels default to black, as many
    # tools produce) can become indistinguishable from the (also black)
    # transparent regions. preprocess_image must composite onto white first.
    image = Image.new("RGBA", (400, 100), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((10, 25), "TRANSPARENT", fill=(0, 0, 0, 255), font=font)

    processed = preprocess_image(image)
    engine = TesseractEngine()
    result = engine.extract_text(processed)
    assert result.strip() != ""
    assert "TRANSPARENT" in result.upper()
