import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sample_files"


def _make_text_image(text: str) -> Image.Image:
    image = Image.new("RGB", (900, 150), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((15, 45), text, fill="black", font=font)
    return image


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_printed_text_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page_one = doc.new_page()
    page_one.insert_text((72, 72), "This is a text-based PDF page one.")
    page_two = doc.new_page()
    page_two.insert_text((72, 72), "This is a text-based PDF page two.")
    doc.save(path)
    doc.close()


def generate_photographed_note_png(path: Path) -> None:
    image = _make_text_image("PHOTOGRAPHED NOTE SAMPLE")
    image.save(path)


def generate_scanned_document_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=900, height=150)
    image_bytes = _image_to_png_bytes(_make_text_image("SCANNED DOCUMENT SAMPLE"))
    page.insert_image(pymupdf.Rect(0, 0, 900, 150), stream=image_bytes)
    doc.save(path)
    doc.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    generate_printed_text_pdf(OUTPUT_DIR / "printed_text.pdf")
    generate_photographed_note_png(OUTPUT_DIR / "photographed_note.png")
    generate_scanned_document_pdf(OUTPUT_DIR / "scanned_document.pdf")
    print(f"Sample files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
