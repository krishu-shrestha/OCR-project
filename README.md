# OCR Streamlit App

Upload a JPG/PNG image or a PDF (text-based or scanned) and extract its
readable text, with an option to download the result as a `.txt` file.

## Setup

1. Install the Tesseract OCR system binary (required for OCR to work):

   ```bash
   brew install tesseract
   tesseract --version
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```
