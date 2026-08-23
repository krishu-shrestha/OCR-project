from ocr_app.config import ALLOWED_EXTENSIONS, OCR_LANGUAGE, PDF_RASTER_DPI


def test_allowed_extensions_include_expected_types():
    assert ALLOWED_EXTENSIONS == {".jpg", ".jpeg", ".png", ".pdf"}


def test_default_ocr_language_is_english():
    assert OCR_LANGUAGE == "eng"


def test_pdf_raster_dpi_is_set():
    assert PDF_RASTER_DPI == 300
