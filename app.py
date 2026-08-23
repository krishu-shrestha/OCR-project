import streamlit as st
import uuid
import time
import datetime
import os
from pathlib import Path

from ocr_app.config import ALLOWED_EXTENSIONS
from ocr_app.engines.tesseract_engine import TesseractEngine
from ocr_app.router import process_uploaded_file
from ocr_app.summarizer import analyze_document
from ocr_app.db import (
    get_all_documents,
    get_document_by_id,
    save_document,
    save_uploaded_file,
    get_uploaded_file_bytes,
    delete_document,
    clear_all_documents,
)
from ocr_app.utils import load_document_pages, get_document_stats

# Configure page layout to wide for comfortable side-by-side view
st.set_page_config(page_title="DocIntel SaaS", layout="wide", page_icon="📂")

@st.cache_data(show_spinner=False)
def _run_ocr(file_name: str, file_bytes: bytes) -> str:
    engine = TesseractEngine()
    return process_uploaded_file(file_name, file_bytes, engine)

def render_dashboard():
    st.markdown("### Welcome to **DocIntel**")
    st.markdown("##### Understand your documents faster with AI.")
    
    # Show statistics summary
    docs = get_all_documents()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(docs))
    with col2:
        invoices = sum(1 for d in docs if d.get("doc_type") == "Invoice")
        st.metric("Invoices", invoices)
    with col3:
        resumes = sum(1 for d in docs if d.get("doc_type") == "Resume/CV")
        st.metric("Resumes/CVs", resumes)
    with col4:
        others = len(docs) - invoices - resumes
        st.metric("Other Types", others)
        
    st.markdown("---")
    
    # Active workspace check
    if st.session_state.get("active_doc_id"):
        render_workspace(st.session_state.active_doc_id)
        return
        
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse to upload your document",
        type=[extension.lstrip(".") for extension in ALLOWED_EXTENSIONS],
        help="Supported formats: JPG, JPEG, PNG, PDF"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        
        # Display Process Button
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            status_container = st.empty()
            
            try:
                start_time = time.time()
                
                # Step 1: Uploading
                status_container.info("⏳ Uploading document...")
                doc_id = uuid.uuid4().hex
                file_path = save_uploaded_file(doc_id, filename, file_bytes)
                
                # Step 2: OCR Extraction
                status_container.info("🔍 Extracting text with Tesseract OCR...")
                extracted_text = _run_ocr(filename, file_bytes)
                
                # Step 3: AI analysis & classification
                status_container.info("🧠 Analyzing & classifying document...")
                
                # Resolve API key
                api_key = st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                    except Exception:
                        pass
                
                analysis = analyze_document(
                    extracted_text, 
                    api_key=api_key, 
                    mock_mode=st.session_state.get("mock_mode", False)
                )
                
                # Compute stats
                stats_info = get_document_stats(extracted_text)
                pages = load_document_pages(filename, file_bytes)
                page_count = len(pages)
                
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)
                
                # Step 4: Save metadata
                status_container.info("💾 Saving record...")
                doc_record = {
                    "id": doc_id,
                    "filename": filename,
                    "file_path": file_path,
                    "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "extracted_text": extracted_text,
                    "summary": analysis.get("summary", ""),
                    "key_info": analysis.get("key_information", {}),
                    "doc_type": analysis.get("document_type", "Other"),
                    "confidence": analysis.get("confidence", 0.0),
                    "stats": {
                        "words": stats_info["words"],
                        "characters": stats_info["characters"],
                        "pages": page_count,
                        "processing_time": processing_time
                    },
                    "favorite": False
                }
                save_document(doc_record)
                
                status_container.success("Analysis Complete ✓")
                time.sleep(0.5)
                st.session_state.active_doc_id = doc_id
                st.rerun()
                
            except Exception as e:
                status_container.empty()
                st.error(f"Failed to process document: {e}")

def render_workspace(doc_id):
    doc = get_document_by_id(doc_id)
    if not doc:
        st.error("Document not found.")
        st.session_state.active_doc_id = None
        st.rerun()
        
    # Top Action Row
    col_back, col_title, col_fav = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Close", use_container_width=True):
            st.session_state.active_doc_id = None
            st.rerun()
    with col_title:
        st.markdown(f"#### 📄 {doc['filename']}")
    with col_fav:
        is_fav = doc.get("favorite", False)
        fav_label = "⭐ Favorite" if is_fav else "☆ Favorite"
        if st.button(fav_label, use_container_width=True):
            doc["favorite"] = not is_fav
            save_document(doc)
            st.rerun()
            
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    # Left column: Document Preview
    with col_left:
        st.subheader("👁️ Document Preview")
        file_bytes = get_uploaded_file_bytes(doc.get("file_path"))
        if file_bytes:
            try:
                pages = load_document_pages(doc["filename"], file_bytes)
                page_count = len(pages)
                
                if "preview_page" not in st.session_state:
                    st.session_state.preview_page = 0
                if st.session_state.preview_page >= page_count:
                    st.session_state.preview_page = 0
                    
                # Display current page
                st.image(pages[st.session_state.preview_page], use_container_width=True)
                
                # Navigation controls
                if page_count > 1:
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                    with nav_col1:
                        if st.button("◀ Previous", disabled=(st.session_state.preview_page == 0), use_container_width=True):
                            st.session_state.preview_page -= 1
                            st.rerun()
                    with nav_col2:
                        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {st.session_state.preview_page + 1} of {page_count}</div>", unsafe_allow_html=True)
                    with nav_col3:
                        if st.button("Next ▶", disabled=(st.session_state.preview_page == page_count - 1), use_container_width=True):
                            st.session_state.preview_page += 1
                            st.rerun()
            except Exception as e:
                st.error(f"Could not render preview: {e}")
        else:
            st.error("Original document file not found on disk.")
            
        # File info
        st.markdown("##### **File Details**")
        st.markdown(f"- **Filename:** {doc['filename']}")
        st.markdown(f"- **File Type:** {Path(doc['filename']).suffix.upper()}")
        st.markdown(f"- **Upload Date:** {doc['upload_date']}")
        
    # Right column: Document Analysis & OCR
    with col_right:
        tab_ocr, tab_ai = st.tabs(["🔍 Extracted Text", "🧠 AI Insights"])
        
        with tab_ocr:
            st.markdown("##### **Tesseract OCR Output**")
            st.text_area(
                "OCR Text",
                doc["extracted_text"],
                height=300,
                label_visibility="collapsed",
                disabled=True
            )
            
            # Action row for OCR
            st.download_button(
                "📥 Download OCR Text",
                doc["extracted_text"],
                file_name=f"ocr_{doc['filename']}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Document Statistics
            st.markdown("##### **📊 Document Statistics**")
            stats = doc.get("stats", {})
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.markdown(f"- **Pages:** {stats.get('pages', 1)}")
                st.markdown(f"- **Words:** {stats.get('words', 0)}")
            with stat_col2:
                st.markdown(f"- **Characters:** {stats.get('characters', 0)}")
                st.markdown(f"- **Processing Time:** {stats.get('processing_time', 0.0)}s")
                
        with tab_ai:
            # 1. Classification
            st.markdown("##### **Document Classification**")
            with st.container(border=True):
                cls_col1, cls_col2 = st.columns(2)
                with cls_col1:
                    st.markdown(f"**Document Type:** `{doc['doc_type']}`")
                with cls_col2:
                    confidence = int(doc.get("confidence", 0.0) * 100)
                    st.markdown(f"**Confidence:** `{confidence}%`")
                    
            # 2. AI Summary
            st.markdown("##### **✨ AI Summary**")
            st.markdown(doc["summary"])
            
            # 3. Key Information (Dynamic Entity Extraction)
            st.markdown("##### **🔑 Key Information**")
            key_info = doc.get("key_info", {})
            if key_info:
                valid_keys = {k: v for k, v in key_info.items() if v and str(v).strip()}
                if valid_keys:
                    for k, v in valid_keys.items():
                        st.markdown(f"**{k}:** {v}")
                else:
                    st.info("No key entities detected in document text.")
            else:
                st.info("No key information extracted.")
                
            # Download summary button
            st.download_button(
                "📥 Download AI Summary",
                doc["summary"],
                file_name=f"summary_{doc['filename']}.txt",
                mime="text/plain",
                use_container_width=True
            )

def render_documents_list():
    if st.session_state.get("active_doc_id"):
        render_workspace(st.session_state.active_doc_id)
        return
        
    st.markdown("### 📄 **My Documents**")
    
    docs = get_all_documents()
    if not docs:
        st.info("No documents processed yet. Go to the Dashboard to upload and analyze your first document.")
        return
        
    # Search and Filter Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        search_query = st.text_input("🔍 Search documents", placeholder="Search by name, summary...")
    with ctrl_col2:
        categories = ["All Categories"] + sorted(list(set(d["doc_type"] for d in docs)))
        selected_cat = st.selectbox("Filter by Category", categories)
    with ctrl_col3:
        sort_option = st.selectbox("Sort by", ["Newest First", "Oldest First", "Filename A-Z", "Filename Z-A"])
        
    # Filter records
    filtered_docs = docs
    if search_query:
        q = search_query.lower()
        filtered_docs = [
            d for d in filtered_docs
            if q in d["filename"].lower() or q in d.get("summary", "").lower() or q in d.get("doc_type", "").lower()
        ]
    if selected_cat != "All Categories":
        filtered_docs = [d for d in filtered_docs if d.get("doc_type") == selected_cat]
        
    # Sort records
    if sort_option == "Newest First":
        filtered_docs.sort(key=lambda x: x["upload_date"], reverse=True)
    elif sort_option == "Oldest First":
        filtered_docs.sort(key=lambda x: x["upload_date"])
    elif sort_option == "Filename A-Z":
        filtered_docs.sort(key=lambda x: x["filename"].lower())
    elif sort_option == "Filename Z-A":
        filtered_docs.sort(key=lambda x: x["filename"].lower(), reverse=True)
        
    st.markdown(f"Showing **{len(filtered_docs)}** of **{len(docs)}** documents")
    
    for doc in filtered_docs:
        with st.container(border=True):
            card_col1, card_col2, card_col3 = st.columns([3, 1, 1])
            with card_col1:
                fav_prefix = "⭐ " if doc.get("favorite") else ""
                st.markdown(f"##### {fav_prefix}📄 **{doc['filename']}**")
                st.markdown(f"`{doc.get('doc_type', 'Other')}` • Uploaded: {doc['upload_date']}")
                snippet = doc.get("summary", "")[:160] + "..." if len(doc.get("summary", "")) > 160 else doc.get("summary", "")
                st.markdown(snippet)
            with card_col2:
                stats = doc.get("stats", {})
                st.markdown(f"**Pages:** {stats.get('pages', 1)}")
                st.markdown(f"**Words:** {stats.get('words', 0)}")
            with card_col3:
                if st.button("Open →", key=f"open_{doc['id']}", use_container_width=True):
                    st.session_state.active_doc_id = doc["id"]
                    st.rerun()
                if st.button("🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True):
                    delete_document(doc["id"])
                    st.rerun()

def render_categories():
    if st.session_state.get("active_doc_id"):
        render_workspace(st.session_state.active_doc_id)
        return
        
    st.markdown("### 🗂️ **Categories**")
    
    docs = get_all_documents()
    if not docs:
        st.info("No documents processed yet.")
        return
        
    cat_groups = {}
    for doc in docs:
        cat = doc.get("doc_type", "Other")
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append(doc)
        
    # Render categories in a grid
    cols = st.columns(3)
    sorted_cats = sorted(cat_groups.keys())
    
    for i, cat in enumerate(sorted_cats):
        col = cols[i % 3]
        count = len(cat_groups[cat])
        with col:
            with st.container(border=True):
                st.markdown(f"### {cat}")
                st.markdown(f"📂 **{count}** document(s)")
                
                show_docs = st.checkbox(f"Show documents ({cat})", key=f"show_cat_{cat}")
                if show_docs:
                    for doc in cat_groups[cat]:
                        st.markdown(f"- {doc['filename']}")
                        if st.button("Open", key=f"open_cat_{doc['id']}"):
                            st.session_state.active_doc_id = doc["id"]
                            st.rerun()

def render_settings():
    st.markdown("### ⚙️ **Settings**")
    
    # Mock Mode toggle
    st.subheader("🛠️ Developer Mode")
    mock_mode_input = st.toggle(
        "Enable Mock Mode (Test local document analysis without Gemini API key)",
        value=st.session_state.get("mock_mode", False),
        help="When active, the app uses offline keyword and regex heuristics instead of the LLM."
    )
    if mock_mode_input != st.session_state.get("mock_mode", False):
        st.session_state.mock_mode = mock_mode_input
        st.rerun()
        
    # Database operations
    st.subheader("🧹 System Maintenance")
    st.warning("Danger Zone: Clearing the database cannot be undone.")
    confirm_clear = st.checkbox("I confirm that I want to delete all processed documents and files.")
    if st.button("Clear All Documents", disabled=not confirm_clear, type="primary"):
        clear_all_documents()
        st.success("Database and upload files cleared successfully!")
        st.rerun()
        
    # System info
    st.subheader("🖥️ System Status")
    docs = get_all_documents()
    st.write(f"- **Stored documents:** {len(docs)}")
    
    import shutil
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        st.success(f"Tesseract found: `{tesseract_path}`")
    else:
        st.error("Tesseract binary NOT found in system PATH. OCR features may fail.")

def main():
    # Injecting custom CSS variables for premium look matching dark/light themes
    st.markdown("""
        <style>
        .stMetric {
            border-radius: 8px;
            padding: 10px;
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        .stButton>button {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📂 **DocIntel**")
        st.markdown("Document Intelligence SaaS")
        st.markdown("---")
        
        if "navigation" not in st.session_state:
            st.session_state.navigation = "🏠 Dashboard"
            
        nav_selection = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📄 My Documents", "🗂️ Categories", "⚙️ Settings"],
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        if nav_selection != st.session_state.navigation:
            st.session_state.navigation = nav_selection
            st.session_state.active_doc_id = None
            st.rerun()
            
        st.markdown("---")
        
        api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except Exception:
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    pass
        
        mock_mode = st.session_state.get("mock_mode", False)
        if mock_mode:
            st.info("🛠️ Mock Mode Active")
        elif api_key:
            provider = "OpenAI" if api_key.strip().startswith("sk-") else "Gemini"
            st.success(f"API Connected ({provider})")
        else:
            st.warning("API Key Required")
            st.info("Configure your API key in your .env file.")
            
    # Page Routing
    if st.session_state.navigation == "🏠 Dashboard":
        render_dashboard()
    elif st.session_state.navigation == "📄 My Documents":
        render_documents_list()
    elif st.session_state.navigation == "🗂️ Categories":
        render_categories()
    elif st.session_state.navigation == "⚙️ Settings":
        render_settings()

if __name__ == "__main__":
    main()


