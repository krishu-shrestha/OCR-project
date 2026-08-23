import os
import pytest
from pathlib import Path
from ocr_app import db

def test_db_operations(tmp_path, monkeypatch):
    temp_db = tmp_path / "test_db.json"
    temp_uploads = tmp_path / "uploads"
    
    # Override database paths for isolated testing
    monkeypatch.setattr(db, "DB_FILE", temp_db)
    monkeypatch.setattr(db, "UPLOADS_DIR", temp_uploads)
    
    # 1. Test database initialization
    db.init_db()
    assert temp_db.exists()
    assert temp_uploads.exists()
    
    # 2. Test saving and retrieving a document
    doc = {
        "id": "doc123",
        "filename": "invoice_test.pdf",
        "extracted_text": "Company Name\nInvoice details",
        "doc_type": "Invoice",
        "confidence": 0.95
    }
    db.save_document(doc)
    
    all_docs = db.get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0]["id"] == "doc123"
    assert "upload_date" in all_docs[0] # Auto-populated
    
    retrieved = db.get_document_by_id("doc123")
    assert retrieved is not None
    assert retrieved["extracted_text"] == "Company Name\nInvoice details"
    
    # 3. Test saving and reading uploaded file bytes
    file_bytes = b"PDF-Mock-Content-Data"
    file_path = db.save_uploaded_file("doc123", "invoice_test.pdf", file_bytes)
    assert Path(file_path).exists()
    assert db.get_uploaded_file_bytes(file_path) == file_bytes
    
    # 4. Test updating an existing document
    doc["confidence"] = 0.98
    doc["file_path"] = file_path
    db.save_document(doc)
    
    updated = db.get_document_by_id("doc123")
    assert updated["confidence"] == 0.98
    
    # 5. Test deleting a document (includes file deletion)
    db.delete_document("doc123")
    assert db.get_document_by_id("doc123") is None
    assert not Path(file_path).exists()
    
    # 6. Test clear all documents
    db.save_document({
        "id": "temp1",
        "filename": "temp.png"
    })
    db.clear_all_documents()
    assert len(db.get_all_documents()) == 0
