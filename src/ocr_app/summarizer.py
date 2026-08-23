import os
import json
import re
import httpx
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# Load environment variables from .env if present
load_dotenv()

class DocumentAnalysis(BaseModel):
    document_type: str = Field(
        description="The category of the document. Must be exactly one of: Invoice, Resume/CV, Certificate, Report, Form, Letter, Business Document, Other."
    )
    confidence: float = Field(
        description="The confidence score of this classification, between 0.0 and 1.0 (e.g. 0.94)."
    )
    summary: str = Field(
        description="A concise summary of the document. For long/complex documents, use structured bullet points under headings like Main Topic, Key Points, Important Info, and Conclusion."
    )
    key_information: Dict[str, str] = Field(
        description="Extracted key information fields like Name, Organization, Date, Amount, Email, Phone, Address, Invoice Number, Due Date, Certificate Number. DO NOT invent fields; only include fields that are actually detected."
    )

ANALYSIS_PROMPT = """
You are an expert document intelligence AI. Analyze the following OCR-extracted text from a document.
Provide a classification, a confidence score, a structured summary, and extract key information.

1. Classification: Categorize the document into exactly one of:
   - Invoice
   - Resume/CV
   - Certificate
   - Report
   - Form
   - Letter
   - Business Document
   - Other

2. Summary Guidelines:
   - Provide a concise but comprehensive summary.
   - For long/complex text, structure it with:
     * **Main Topic/Purpose**
     * **Key Points**
     * **Important Information**
     * **Conclusion** (if applicable)
   - Preserve dates, names, and facts. Do not invent details.

3. Key Information Guidelines:
   - Extract important entities. Focus on relevant fields such as:
     * Name
     * Organization (Company, University, etc.)
     * Date (Invoice Date, Event Date, etc.)
     * Amount (Total Amount, Fee, etc.)
     * Email
     * Phone
     * Address
     * Invoice Number
     * Due Date
     * Certificate Number
   - Do not include fields that are completely blank or irrelevant to the document. Only output fields you find evidence for in the text.
"""

CHUNK_PROMPT = """
You are an expert document summarizer. Summarize this section of a longer document.
Focus on preserving key names, dates, amounts, and central concepts.
Do not invent details. Keep it concise.
"""

MERGE_PROMPT = """
You are an expert document summarizer. Below are summaries of different parts of a document.
Please synthesize these partial summaries into one final, coherent, and structured summary.
- The final summary must be concise and represent the entire document.
- Follow this structure:
  * **Main Topic/Purpose**: [Description]
  * **Key Points**:
    - [Point 1]
    - [Point 2]
    - ...
  * **Important Information**: [Details]
  * **Conclusion**: [Conclusion, if applicable]
"""

def chunk_text(text: str, max_chars: int = 20000) -> list[str]:
    """Split text into chunks of at most max_chars, trying to split on paragraph/sentence boundaries."""
    if not text:
        return []
        
    if len(text) <= max_chars:
        return [text]
        
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        if len(para) > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                
            words = para.split(" ")
            word_chunk = []
            word_len = 0
            for word in words:
                if word_len + len(word) + 1 > max_chars:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_len = len(word)
                else:
                    word_chunk.append(word)
                    word_len += len(word) + 1
            if word_chunk:
                chunks.append(" ".join(word_chunk))
        else:
            if current_len + len(para) + 2 > max_chars:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para) + 2
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def get_api_key() -> str:
    """Helper to retrieve the API key (OpenAI or Gemini) from environment or Streamlit secrets."""
    for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]:
        api_key = os.getenv(key)
        if api_key:
            return api_key
            
    try:
        import streamlit as st
        for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]:
            if key in st.secrets:
                return st.secrets[key]
    except Exception:
        pass
        
    return None

def analyze_document(text: str, api_key: str = None, mock_mode: bool = False) -> dict:
    """
    Analyzes OCR-extracted text using the Gemini API or OpenAI API.
    Classifies the document, calculates confidence, summarises text,
    and extracts key information. Handles long text via chunking.
    """
    if not text or not text.strip():
        raise ValueError("No text provided. Please extract some text first.")
        
    if mock_mode or api_key == "mock":
        return analyze_document_mock(text)
        
    if not api_key:
        api_key = get_api_key()
        
    if not api_key:
        raise ValueError(
            "API Key is not set. Please set the GEMINI_API_KEY or OPENAI_API_KEY environment variable "
            "in your .env file to use the AI Document Analysis feature."
        )
        
    if api_key.strip().startswith("sk-"):
        return analyze_document_openai(text, api_key)
        
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini Client: {e}")
        
    chunks = chunk_text(text)
    
    try:
        if len(chunks) == 1:
            prompt = f"{ANALYSIS_PROMPT}\n\nDocument Text:\n{chunks[0]}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DocumentAnalysis,
                )
            )
            return json.loads(response.text)
        else:
            # Multi-chunk strategy
            # 1. Do full structured analysis on the first chunk (contains headers, dates, entities)
            prompt = f"{ANALYSIS_PROMPT}\n\nDocument Text (Part 1):\n{chunks[0]}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DocumentAnalysis,
                )
            )
            first_chunk_analysis = json.loads(response.text)
            
            # 2. Generate text summaries for the remaining chunks
            chunk_summaries = [first_chunk_analysis["summary"]]
            for i in range(1, len(chunks)):
                chunk_prompt = f"{CHUNK_PROMPT}\n\nDocument Text (Part {i+1} of {len(chunks)}):\n{chunks[i]}"
                chunk_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=chunk_prompt,
                )
                chunk_summaries.append(chunk_response.text)
                
            # 3. Synthesize the chunk summaries into a final structured summary
            combined = "\n\n=== PART SUMMARY ===\n\n".join(chunk_summaries)
            merge_prompt = f"{MERGE_PROMPT}\n\nChunk Summaries:\n{combined}"
            merge_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=merge_prompt,
            )
            
            first_chunk_analysis["summary"] = merge_response.text
            return first_chunk_analysis
            
    except errors.APIError as e:
        raise RuntimeError(f"Gemini API error occurred: {e.message}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during document analysis: {e}")

def analyze_document_mock(text: str) -> dict:
    """Generates heuristic analysis (mock mode) without calling the Gemini API."""
    text_lower = text.lower()
    
    # Classification heuristic
    doc_type = "Other"
    confidence = 0.85
    
    if "invoice" in text_lower or "bill" in text_lower or "invoice number" in text_lower:
        doc_type = "Invoice"
    elif "resume" in text_lower or "cv" in text_lower or "education" in text_lower or "experience" in text_lower:
        doc_type = "Resume/CV"
    elif "certificate" in text_lower or "certified" in text_lower or "award" in text_lower:
        doc_type = "Certificate"
    elif "report" in text_lower or "analysis" in text_lower:
        doc_type = "Report"
    elif "letter" in text_lower or "dear" in text_lower:
        doc_type = "Letter"
        
    # Extract some mock key info using basic regex
    key_info = {}
    
    # Try finding emails
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if emails:
        key_info["Email"] = emails[0]
        
    # Try finding phones
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    if phones:
        key_info["Phone"] = phones[0]
        
    # Try finding dates
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    if dates:
        key_info["Date"] = dates[0]
        
    # Mock summary based on first few sentences
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    summary_sentences = sentences[:3]
    summary_text = " ".join(summary_sentences)
    
    summary = f"**Main Topic/Purpose**: Mock analysis of document (running offline).\n\n**Key Points**:\n- Extracted text contains {len(sentences)} sentence(s).\n- Document classified as {doc_type} via local heuristics.\n\n**Important Information**:\n{summary_text if summary_text else 'No text summary available.'}"
    
    return {
        "document_type": doc_type,
        "confidence": confidence,
        "summary": summary,
        "key_information": key_info
    }

def generate_summary(text: str, api_key: str = None) -> str:
    """Helper for backward compatibility with existing tests."""
    analysis = analyze_document(text, api_key)
    return analysis.get("summary", "")

def parse_openai_response(content: str) -> dict:
    """Parses OpenAI structured JSON where key_information is returned as an array of key/value pairs."""
    data = json.loads(content)
    if isinstance(data.get("key_information"), list):
        data["key_information"] = {
            item["key"]: item["value"] 
            for item in data["key_information"] 
            if isinstance(item, dict) and "key" in item and "value" in item
        }
    elif not isinstance(data.get("key_information"), dict):
        data["key_information"] = {}
    return data

def analyze_document_openai(text: str, api_key: str) -> dict:
    """Analyzes OCR-extracted text using OpenAI's API (gpt-4o-mini) with structured output."""
    chunks = chunk_text(text)
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Define structured output format schema for OpenAI (strict compliance)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "DocumentAnalysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "The category of the document. Must be exactly one of: Invoice, Resume/CV, Certificate, Report, Form, Letter, Business Document, Other."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "The confidence score of this classification, between 0.0 and 1.0 (e.g. 0.94)."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of the document. For long/complex documents, use structured bullet points under headings like Main Topic, Key Points, Important Info, and Conclusion."
                    },
                    "key_information": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": { "type": "string" },
                                "value": { "type": "string" }
                            },
                            "required": ["key", "value"],
                            "additionalProperties": False
                        },
                        "description": "Extracted key information fields like Name, Organization, Date, Amount, Email, Phone, Address, Invoice Number, Due Date, Certificate Number."
                    }
                },
                "required": ["document_type", "confidence", "summary", "key_information"],
                "additionalProperties": False
            }
        }
    }
    
    try:
        if len(chunks) == 1:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": f"Document Text:\n{chunks[0]}"}
                ],
                "response_format": schema
            }
            
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_openai_response(content)
        else:
            # Multi-chunk strategy
            # 1. First chunk analysis
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": f"Document Text (Part 1):\n{chunks[0]}"}
                ],
                "response_format": schema
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            first_chunk_analysis = parse_openai_response(content)
            
            # 2. Get summaries of remaining chunks
            chunk_summaries = [first_chunk_analysis["summary"]]
            for i in range(1, len(chunks)):
                chunk_payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": CHUNK_PROMPT},
                        {"role": "user", "content": f"Document Text (Part {i+1} of {len(chunks)}):\n{chunks[i]}"}
                    ]
                }
                c_resp = httpx.post(url, headers=headers, json=chunk_payload, timeout=60.0)
                c_resp.raise_for_status()
                chunk_summaries.append(c_resp.json()["choices"][0]["message"]["content"])
                
            # 3. Merge
            combined = "\n\n=== PART SUMMARY ===\n\n".join(chunk_summaries)
            merge_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": MERGE_PROMPT},
                    {"role": "user", "content": f"Chunk Summaries:\n{combined}"}
                ]
            }
            m_resp = httpx.post(url, headers=headers, json=merge_payload, timeout=60.0)
            m_resp.raise_for_status()
            first_chunk_analysis["summary"] = m_resp.json()["choices"][0]["message"]["content"]
            
            return first_chunk_analysis
            
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during OpenAI document analysis: {e}")


