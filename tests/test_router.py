import pytest

from ocr_app.engines.base import OCREngine
from ocr_app.router import process_uploaded_file


class StubEngine(OCREngine):
    def extract_text(self, image):
        return "stub text"


def test_routes_pdf_files_to_pdf_extractor(monkeypatch):
    called_with = {}

    def fake_pdf_extractor(file_bytes, engine):
        called_with["file_bytes"] = file_bytes
        called_with["engine"] = engine
        return "pdf result"

    monkeypatch.setattr("ocr_app.router.extract_text_from_pdf_bytes", fake_pdf_extractor)
    engine = StubEngine()
    result = process_uploaded_file("document.pdf", b"pdf-bytes", engine)
    assert result == "pdf result"
    assert called_with == {"file_bytes": b"pdf-bytes", "engine": engine}


def test_routes_image_files_to_image_extractor(monkeypatch):
    called_with = {}

    def fake_image_extractor(file_bytes, engine):
        called_with["file_bytes"] = file_bytes
        called_with["engine"] = engine
        return "image result"

    monkeypatch.setattr("ocr_app.router.extract_text_from_image_bytes", fake_image_extractor)
    engine = StubEngine()
    result = process_uploaded_file("photo.PNG", b"image-bytes", engine)
    assert result == "image result"
    assert called_with == {"file_bytes": b"image-bytes", "engine": engine}


def test_raises_value_error_for_unsupported_extension():
    engine = StubEngine()
    with pytest.raises(ValueError):
        process_uploaded_file("archive.zip", b"data", engine)
