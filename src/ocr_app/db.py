import json
import os
from pathlib import Path
import datetime

DB_FILE = Path("documents_db.json")
UPLOADS_DIR = Path("uploads")

def init_db():
    """Initializes the database file and uploads folder if they don't exist."""
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not DB_FILE.exists():
        with open(DB_FILE, "w") as f:
            json.dump([], f)

def get_all_documents() -> list[dict]:
    """Retrieves all documents from the database."""
    init_db()
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def get_document_by_id(doc_id: str) -> dict | None:
    """Finds a single document by its unique ID."""
    docs = get_all_documents()
    for doc in docs:
        if doc["id"] == doc_id:
            return doc
    return None

def save_document(doc: dict) -> None:
    """Saves or updates a document in the JSON database."""
    init_db()
    docs = get_all_documents()
    
    # Ensure upload_date is set
    if "upload_date" not in doc:
        doc["upload_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Check if doc already exists, update it, otherwise append
    exists = False
    for i, d in enumerate(docs):
        if d["id"] == doc["id"]:
            docs[i] = doc
            exists = True
            break
            
    if not exists:
        docs.append(doc)
        
    with open(DB_FILE, "w") as f:
        json.dump(docs, f, indent=4)

def save_uploaded_file(doc_id: str, filename: str, file_bytes: bytes) -> str:
    """Saves uploaded file bytes physically to the uploads directory."""
    init_db()
    suffix = Path(filename).suffix
    file_path = UPLOADS_DIR / f"{doc_id}{suffix}"
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return str(file_path)

def get_uploaded_file_bytes(file_path: str) -> bytes | None:
    """Reads file bytes from the uploads folder."""
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return None

def delete_document(doc_id: str) -> None:
    """Deletes a document from the database and its uploaded file."""
    docs = get_all_documents()
    doc_to_delete = get_document_by_id(doc_id)
    
    # Filter out the document from the list
    docs = [d for d in docs if d["id"] != doc_id]
    with open(DB_FILE, "w") as f:
        json.dump(docs, f, indent=4)
        
    # Delete the uploaded file from disk
    if doc_to_delete and "file_path" in doc_to_delete:
        file_path = doc_to_delete["file_path"]
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

def clear_all_documents() -> None:
    """Clears all records and associated upload files."""
    docs = get_all_documents()
    for doc in docs:
        delete_document(doc["id"])
    if DB_FILE.exists():
        with open(DB_FILE, "w") as f:
            json.dump([], f)
