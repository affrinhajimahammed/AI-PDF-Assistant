"""
app.py - AI PDF Question-Answering Assistant (Streamlit Web Interface)

A beginner-friendly, 100% local Question-Answering application for PDF documents.
Uses PyMuPDF for text extraction and Scikit-learn (TF-IDF + Cosine Similarity)
for extractive document retrieval.

No paid API keys, zero hallucinations, fully verifiable page citations.
"""

import os
from typing import Optional
import streamlit as st
from qa_engine import (
    extract_text_from_pdf,
    split_into_passages,
    find_most_relevant_passages,
    clean_whitespace,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI PDF Question-Answering Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, clean, beginner-friendly UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .answer-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #15803D;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .score-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .system-notice {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state():
    """Initializes Streamlit session state keys if not already set."""
    if "pdf_bytes" not in st.session_state:
        st.session_state["pdf_bytes"] = None
    if "pdf_name" not in st.session_state:
        st.session_state["pdf_name"] = None
    if "pages_data" not in st.session_state:
        st.session_state["pages_data"] = []
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}
    if "passages" not in st.session_state:
        st.session_state["passages"] = []
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "qa_results" not in st.session_state:
        st.session_state["qa_results"] = None


init_session_state()


def reset_app():
    """Clears all session variables to start fresh."""
    st.session_state["pdf_bytes"] = None
    st.session_state["pdf_name"] = None
    st.session_state["pages_data"] = []
    st.session_state["metadata"] = {}
    st.session_state["passages"] = []
    st.session_state["last_query"] = ""
    st.session_state["qa_results"] = None
    st.rerun()


# ---------------------------------------------------------
# Sidebar: Controls, Document Info & Educational Guide
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Assistant Controls")

    # Document upload section
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload any native text PDF from your computer."
    )

    # Built-in sample PDF button for instant testing
    sample_pdf_path = "sample_ai_overview.pdf"
    if os.path.exists(sample_pdf_path):
        if st.button("🧪 Load Sample AI Overview PDF", use_container_width=True, help="Loads a ready-made 4-page AI & Machine Learning PDF"):
            try:
                with open(sample_pdf_path, "rb") as f:
                    sample_bytes = f.read()
                st.session_state["pdf_bytes"] = sample_bytes
                st.session_state["pdf_name"] = "sample_ai_overview.pdf"
                pages, meta = extract_text_from_pdf(sample_bytes)
                st.session_state["pages_data"] = pages
                st.session_state["metadata"] = meta
                st.session_state["passages"] = split_into_passages(pages)
                st.session_state["qa_results"] = None
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample PDF: {e}")

    # Retrieval settings
    st.markdown("---")
    st.subheader("🔍 Search Parameters")
    top_k = st.slider(
        "Top Passages to Retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="How many matching passages to display for each question."
    )
    min_similarity = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.30,
        value=0.08,
        step=0.01,
        help="Passages with similarity below this threshold are rejected as unhelpful."
    )

    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset Document & Questions", use_container_width=True):
        reset_app()

    # Educational notes in sidebar
    st.markdown("---")
    st.markdown("""
    ### 💡 How This Assistant Works
    1. **PyMuPDF**: Parses each page and extracts native text.
    2. **Passage Chunker**: Divides text into 120-word chunks with 25-word overlap so context is never cut off.
    3. **TF-IDF**: Converts chunks into numerical vectors based on term importance.
    4. **Cosine Similarity**: Measures mathematical similarity between question and chunks.
    5. **Extractive Sourcing**: Displays direct facts with exact page numbers.
    """)


# ---------------------------------------------------------
# Handle File Upload Processing
# ---------------------------------------------------------
if uploaded_file is not None:
    # Check if a new file was uploaded
    new_bytes = uploaded_file.getvalue()
    if st.session_state["pdf_bytes"] != new_bytes or st.session_state["pdf_name"] != uploaded_file.name:
        try:
            with st.spinner("Processing PDF and creating passage index..."):
                pages, meta = extract_text_from_pdf(new_bytes)
                st.session_state["pdf_bytes"] = new_bytes
                st.session_state["pdf_name"] = uploaded_file.name
                st.session_state["pages_data"] = pages
                st.session_state["metadata"] = meta
                st.session_state["passages"] = split_into_passages(pages)
                st.session_state["qa_results"] = None
        except Exception as err:
            st.error(f"⚠️ Failed to process PDF: {str(err)}")
            st.session_state["pdf_bytes"] = None
            st.session_state["pdf_name"] = None


# ---------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📄 AI PDF Question-Answering Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An Extractive Information Retrieval Assistant powered by <b>PyMuPDF</b>, <b>TF-IDF</b>, and <b>Cosine Similarity</b>.'
    '</div>',
    unsafe_allow_html=True
)

# Notice highlighting lack of external API / zero hallucination
st.markdown("""
<div class="system-notice">
    🛡️ <b>Transparent & Verifiable AI:</b> This assistant uses local mathematical document retrieval (TF-IDF & Cosine Similarity). 
    It does not use paid external APIs or generative LLMs, meaning it <b>never invents facts</b> and always cites the exact source page.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Document State: No Document Loaded
# ---------------------------------------------------------
if not st.session_state["pdf_bytes"]:
    st.info("👋 Welcome! To begin, upload a PDF file from the left sidebar or click **'Load Sample AI Overview PDF'** to test immediately.")

    # Educational onboarding cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📥 1. Upload Document")
        st.caption("Upload course notes, research papers, syllabus documents, or technical guides in standard PDF format.")
    with col2:
        st.markdown("### 🔎 2. Ask Any Question")
        st.caption("Type questions in natural language. TF-IDF will identify the most relevant terminology across all pages.")
    with col3:
        st.markdown("### 📑 3. Inspect Sources")
        st.caption("Read the extracted answers alongside page numbers and similarity confidence scores.")

    st.stop()


# ---------------------------------------------------------
# Document State: Document Loaded
# ---------------------------------------------------------
metadata = st.session_state["metadata"]
pages_data = st.session_state["pages_data"]
passages = st.session_state["passages"]
doc_name = st.session_state["pdf_name"]

# Edge Case: Scanned PDF or No Extractable Text
if not metadata.get("has_text", False):
    st.error(
        f"⚠️ **No extractable text found in `{doc_name}`.**\n\n"
        "This usually means the PDF is either scanned (an image of text) or contains empty pages.\n\n"
        "**What you can do:**\n"
        "- Upload a native text PDF (generated from Word, LaTeX, Google Docs, or printed to PDF).\n"
        "- If your PDF is scanned, run an OCR (Optical Character Recognition) tool to convert it to searchable text."
    )
    if st.button("Try Another PDF"):
        reset_app()
    st.stop()

# Display Document Metrics Bar
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Active Document", doc_name[:20] + "..." if len(doc_name) > 20 else doc_name)
with m_col2:
    st.metric("Total Pages", metadata.get("total_pages", 0))
with m_col3:
    st.metric("Total Words", f"{metadata.get('total_words', 0):,}")
with m_col4:
    st.metric("Indexed Passages", len(passages))

st.markdown("---")


# ---------------------------------------------------------
# Question Input Section
# ---------------------------------------------------------
st.subheader("❓ Ask a Question about the Document")

# Quick suggestion chips for the sample document
if doc_name == "sample_ai_overview.pdf":
    st.caption("💡 Try asking one of these sample questions:")
    suggested_cols = st.columns(3)
    sample_queries = [
        "What is supervised learning and what are examples?",
        "What is the role of ReLU and Backpropagation?",
        "How do TF-IDF and Cosine Similarity work?"
    ]
    for idx, query_example in enumerate(sample_queries):
        if suggested_cols[idx].button(f"👉 {query_example}", key=f"sample_btn_{idx}"):
            st.session_state["user_input_query"] = query_example

# Text Input box
default_val = st.session_state.get("user_input_query", "")
query_text = st.text_input(
    "Type your question here:",
    value=default_val,
    placeholder="e.g., What activation functions are used in neural networks?",
    help="Type any concept, term, or question found in the uploaded PDF."
)

c_search, c_clear = st.columns([1, 5])
search_clicked = c_search.button("🔍 Find Answer", type="primary", use_container_width=True)

if search_clicked or (query_text and query_text != st.session_state.get("last_query", "")):
    if not query_text.strip():
        st.warning("Please enter a question to search.")
    else:
        st.session_state["last_query"] = query_text
        with st.spinner("Searching document passages using TF-IDF and Cosine Similarity..."):
            results = find_most_relevant_passages(
                passages=passages,
                query=query_text,
                top_k=top_k,
                min_similarity=min_similarity
            )
            st.session_state["qa_results"] = results


# ---------------------------------------------------------
# Display QA Results
# ---------------------------------------------------------
results_data = st.session_state.get("qa_results")

if results_data is not None:
    status = results_data.get("status")
    matches = results_data.get("results", [])
    best_score = results_data.get("best_score", 0.0)

    st.markdown("### 📋 Answer & Evidence")

    if status == "SUCCESS" and matches:
        top_match = matches[0]

        # Top Direct Answer Card
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-title">🎯 Extractive Answer Snippet</div>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #1F2937;">
                "{top_match['snippet']}"
            </p>
            <div style="margin-top: 10px;">
                <span class="source-badge">📄 Found on Page {top_match['page_number']}</span>
                <span class="score-badge">📊 Relevance Score: {top_match['score_percent']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar showing similarity confidence
        st.caption(f"Search Confidence: **{top_match['score_percent']}%**")
        st.progress(min(1.0, float(top_match["score"])))

        # Retrieved Passages Breakdown
        st.markdown(f"#### 📚 Top {len(matches)} Relevant Passages (Source Context)")
        for rank, match in enumerate(matches, 1):
            with st.expander(
                f"Passage #{rank} (Page {match['page_number']}) — Similarity: {match['score_percent']}%",
                expanded=(rank == 1)
            ):
                st.markdown(f"**Extracted Text from Page {match['page_number']}:**")
                st.write(match["text"])
                st.caption(f"Chunk ID: {match['chunk_id']} | TF-IDF Cosine Similarity: {match['score']}")

    elif status == "NO_MATCH":
        st.warning(
            f"⚠️ **No sufficient match found in this document.**\n\n"
            f"{results_data.get('message')}\n\n"
            "**Tips to find an answer:**\n"
            "- Use specific terms that appear in the document.\n"
            "- Check the 'Document Content Inspector' below to see the topics in this PDF.\n"
            "- Try lowering the 'Confidence Threshold' slider in the left sidebar if you want looser matches."
        )


# ---------------------------------------------------------
# Document Content Inspector (Educational & Verifiable)
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📖 Document Content Inspector (Page-by-Page)", expanded=False):
    st.write(f"Inspect all extracted text from **{doc_name}** to verify document contents:")
    for page in pages_data:
        st.markdown(f"**--- Page {page['page_number']} ({page['word_count']} words, {page['char_count']} characters) ---**")
        st.text_area(
            label=f"Page {page['page_number']} Content",
            value=page["text"] if page["text"] else "[No text on this page]",
            height=120,
            key=f"inspector_page_{page['page_number']}",
            disabled=True
        )
