import pytest
from PIL import Image

from ocr_app.engines.base import OCREngine


def test_ocr_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        OCREngine()


def test_subclass_implementing_extract_text_can_be_instantiated():
    class DummyEngine(OCREngine):
        def extract_text(self, image: Image.Image) -> str:
            return "dummy text"

    engine = DummyEngine()
    assert engine.extract_text(Image.new("RGB", (10, 10))) == "dummy text"
