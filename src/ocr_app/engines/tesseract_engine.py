import pytesseract
from PIL import Image

from ocr_app.config import OCR_LANGUAGE
from ocr_app.engines.base import OCREngine


class TesseractEngine(OCREngine):
    def extract_text(self, image: Image.Image) -> str:
        return pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
