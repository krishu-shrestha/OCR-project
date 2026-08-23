from abc import ABC, abstractmethod

from PIL import Image


class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        raise NotImplementedError
