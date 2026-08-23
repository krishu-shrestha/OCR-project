import io

from PIL import Image

from ocr_app.engines.base import OCREngine
from ocr_app.preprocessing import preprocess_image


def extract_text_from_image_bytes(file_bytes: bytes, engine: OCREngine) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    processed = preprocess_image(image)
    return engine.extract_text(processed)
