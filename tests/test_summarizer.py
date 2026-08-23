from unittest.mock import MagicMock, patch
import pytest
from google.genai import errors
from ocr_app.summarizer import chunk_text, generate_summary

def test_chunk_text_short():
    text = "Short text content."
    assert chunk_text(text, max_chars=100) == [text]

def test_chunk_text_long_paragraphs():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    # Max chars 20 should split by paragraphs
    chunks = chunk_text(text, max_chars=20)
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph one."
    assert chunks[1] == "Paragraph two."
    assert chunks[2] == "Paragraph three."

def test_chunk_text_long_single_paragraph():
    text = "Word1 Word2 Word3 Word4 Word5"
    # Max chars 11 should split long single paragraph into chunks of words
    chunks = chunk_text(text, max_chars=11)
    assert len(chunks) >= 3
    assert "Word1" in chunks[0]

def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text(None) == []

def test_generate_summary_validation():
    with pytest.raises(ValueError, match="No text provided"):
        generate_summary("   ")
    with pytest.raises(ValueError, match="No text provided"):
        generate_summary("")
    with pytest.raises(ValueError, match="No text provided"):
        generate_summary(None)

def test_generate_summary_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("ocr_app.summarizer.get_api_key", lambda: None)
    with pytest.raises(ValueError, match="API Key is not set"):
        generate_summary("Valid text")

@patch('ocr_app.summarizer.genai.Client')
def test_generate_summary_single_chunk(mock_client_class):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"document_type": "Invoice", "confidence": 0.95, "summary": "This is a summary.", "key_information": {"Name": "John"}}'
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client
    
    result = generate_summary("Some text to summarize", api_key="test-key")
    
    assert result == "This is a summary."
    mock_client.models.generate_content.assert_called_once()
    args, kwargs = mock_client.models.generate_content.call_args
    assert kwargs['model'] == 'gemini-2.5-flash'
    assert "Some text to summarize" in kwargs['contents']

@patch('ocr_app.summarizer.genai.Client')
def test_generate_summary_multi_chunk(mock_client_class):
    mock_client = MagicMock()
    
    mock_response1 = MagicMock()
    mock_response1.text = '{"document_type": "Invoice", "confidence": 0.95, "summary": "Summary of part 1", "key_information": {"Name": "John"}}'
    mock_response2 = MagicMock()
    mock_response2.text = "Summary of part 2"
    mock_response_final = MagicMock()
    mock_response_final.text = "Final merged summary"
    
    mock_client.models.generate_content.side_effect = [mock_response1, mock_response2, mock_response_final]
    mock_client_class.return_value = mock_client
    
    with patch('ocr_app.summarizer.chunk_text', return_value=["part1", "part2"]):
        result = generate_summary("Very long text", api_key="test-key")
        
    assert result == "Final merged summary"
    assert mock_client.models.generate_content.call_count == 3

@patch('ocr_app.summarizer.genai.Client')
def test_generate_summary_api_error(mock_client_class):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.APIError(
        code=429,
        response_json={"error": {"message": "Quota exceeded"}}
    )
    mock_client_class.return_value = mock_client
    
    with pytest.raises(RuntimeError, match="Gemini API error occurred: Quota exceeded"):
        generate_summary("Some text", api_key="test-key")

@patch('ocr_app.summarizer.httpx.post')
def test_analyze_document_openai(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"document_type": "Invoice", "confidence": 0.99, "summary": "OpenAI summary", "key_information": [{"key": "Date", "value": "2026-08-19"}]}'
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    
    from ocr_app.summarizer import analyze_document
    result = analyze_document("Sample document text", api_key="sk-testkey123")
    
    assert result["document_type"] == "Invoice"
    assert result["confidence"] == 0.99
    assert result["summary"] == "OpenAI summary"
    assert result["key_information"]["Date"] == "2026-08-19"
    
    mock_post.assert_called_once()
