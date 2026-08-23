import io
from PIL import Image
from ocr_app.utils import get_document_stats, load_document_pages

def test_get_document_stats_empty():
    assert get_document_stats("") == {"words": 0, "characters": 0}
    assert get_document_stats(None) == {"words": 0, "characters": 0}

def test_get_document_stats_normal():
    text = "This is a document with seven words."
    stats = get_document_stats(text)
    assert stats["words"] == 7
    assert stats["characters"] == len(text)

def test_load_document_pages_image():
    # Construct a valid dummy PNG in memory to test rendering
    img = Image.new("RGB", (150, 150), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    
    pages = load_document_pages("test_image.png", img_bytes)
    assert len(pages) == 1
    assert isinstance(pages[0], Image.Image)
    assert pages[0].size == (150, 150)
