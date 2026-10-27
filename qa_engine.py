"""
qa_engine.py - Core Question-Answering Engine for PDF Documents

This module provides pure Python functions to:
1. Extract text and track page numbers from PDF files using PyMuPDF.
2. Split extracted text into overlapping, context-preserving passages.
3. Compute TF-IDF (Term Frequency-Inverse Document Frequency) vectors.
4. Measure Cosine Similarity to find passages that best answer user questions.
5. Extract key answer sentences with zero external API calls or hallucination.

Designed for beginners and AI/ML freshers with clear documentation.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import pymupdf  # Official PyMuPDF library for fast PDF parsing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF byte stream.

    Parameters:
        pdf_bytes (bytes): The raw bytes of the uploaded PDF file.

    Returns:
        tuple: (pages_data, doc_metadata)
            pages_data: List of dicts, each with 'page_number' (1-indexed) and 'text'.
            doc_metadata: Summary dict with total_pages, total_chars, has_text, etc.

    Raises:
        ValueError: If PDF is corrupt, encrypted, or cannot be parsed.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("The provided file is empty (0 bytes). Please upload a valid PDF.")

    try:
        # Open the PDF document from in-memory bytes
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not read the file as a valid PDF: {str(exc)}")

    # Check if document is password protected / encrypted
    if doc.is_encrypted:
        doc.close()
        raise ValueError("This PDF document is encrypted or password-protected. Please upload an unprotected PDF.")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("The PDF document contains 0 pages.")

    pages_data = []
    total_characters = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        # Extract plain text from page
        raw_text = page.get_text("text") or ""
        # Normalize excessive whitespace and blank lines
        cleaned_text = clean_whitespace(raw_text)

        char_count = len(cleaned_text)
        total_characters += char_count

        pages_data.append({
            "page_number": page_idx + 1,  # 1-indexed for user readability
            "text": cleaned_text,
            "char_count": char_count,
            "word_count": len(cleaned_text.split()) if cleaned_text else 0
        })

    doc.close()

    # Determine if text is extractable (e.g. Scanned image vs native digital text)
    has_text = total_characters > 10 and any(p["word_count"] > 3 for p in pages_data)

    metadata = {
        "total_pages": total_pages,
        "total_characters": total_characters,
        "total_words": sum(p["word_count"] for p in pages_data),
        "has_text": has_text
    }

    return pages_data, metadata


def clean_whitespace(text: str) -> str:
    """
    Normalizes multiple spaces, tabs, and newlines into clean readable text.
    Preserves paragraph breaks where appropriate.
    """
    if not text:
        return ""
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3 or more consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple inline whitespace characters with a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def split_into_passages(
    pages_data: List[Dict[str, Any]],
    chunk_size_words: int = 120,
    overlap_words: int = 25
) -> List[Dict[str, Any]]:
    """
    Splits page text into smaller, overlapping passages (chunks) for precise retrieval.

    Why chunking is needed:
    A whole page or an entire book chapter is usually too broad for a specific question.
    By breaking pages into smaller passages (e.g. 80-150 words) with a small overlap,
    TF-IDF can pinpoint the exact paragraph that answers the user's query without
    cutting off sentences across boundaries.

    Parameters:
        pages_data: List of page dictionaries returned by `extract_text_from_pdf`.
        chunk_size_words: Target word length for each passage (default: 120 words).
        overlap_words: Number of overlapping words between consecutive passages (default: 25).

    Returns:
        List of dicts: Each chunk has 'chunk_id', 'page_number', 'text', 'word_count'.
    """
    passages: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page in pages_data:
        page_num = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        # Split into words while retaining natural ordering
        words = page_text.split()
        total_words_in_page = len(words)

        if total_words_in_page == 0:
            continue

        # If page text is shorter than or roughly equal to chunk size, keep it as one passage
        if total_words_in_page <= chunk_size_words:
            passages.append({
                "chunk_id": chunk_counter,
                "page_number": page_num,
                "text": " ".join(words),
                "word_count": total_words_in_page
            })
            chunk_counter += 1
            continue

        # Create sliding window chunks with overlap
        step = max(1, chunk_size_words - overlap_words)
        start_idx = 0

        while start_idx < total_words_in_page:
            end_idx = min(start_idx + chunk_size_words, total_words_in_page)
            chunk_words = words[start_idx:end_idx]

            # Avoid tiny trailing chunks if under 15 words
            if len(chunk_words) < 15 and passages:
                # Merge tiny trailing fragment with the previous chunk if on the same page
                if passages[-1]["page_number"] == page_num:
                    passages[-1]["text"] += " " + " ".join(chunk_words)
                    passages[-1]["word_count"] = len(passages[-1]["text"].split())
                    break

            passages.append({
                "chunk_id": chunk_counter,
                "page_number": page_num,
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words)
            })
            chunk_counter += 1

            if end_idx >= total_words_in_page:
                break

            start_idx += step

    return passages


def extract_best_answer_snippet(passage_text: str, query: str) -> str:
    """
    Finds the most relevant 1-2 sentences within a retrieved passage that
    directly correlate with the query keywords.

    This ensures the user gets a quick, clear direct answer before reading
    the full context passage.
    """
    # Split passage into rough sentences
    sentences = re.split(r"(?<=[.!?])\s+", passage_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return passage_text

    # Extract keywords from query (ignoring short common words)
    stopwords = {
        "what", "is", "a", "an", "the", "in", "on", "of", "for", "with", "to",
        "and", "how", "why", "does", "do", "can", "tell", "me", "about", "which",
        "are", "was", "were", "it", "its", "that", "this", "from", "by", "as"
    }
    query_tokens = [
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if len(w) > 2 and w.lower() not in stopwords
    ]

    if not query_tokens:
        # Fall back to first 2 sentences if query has no specific keywords
        return " ".join(sentences[:2])

    scored_sentences: List[Tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        lower_sent = sentence.lower()
        # Count keyword occurrences
        match_count = sum(1 for token in query_tokens if token in lower_sent)
        # Ratio of matched keywords to sentence length
        score = match_count / (len(sentence.split()) + 5)
        scored_sentences.append((score, idx, sentence))

    # Sort by keyword match score descending
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    if scored_sentences[0][0] > 0:
        # Get the top sentence index
        best_idx = scored_sentences[0][1]
        # Return the best sentence plus immediately following sentence if available for context
        selected = [sentences[best_idx]]
        if best_idx + 1 < len(sentences) and len(selected[0]) < 180:
            selected.append(sentences[best_idx + 1])
        return " ".join(selected)

    return " ".join(sentences[:2])


def find_most_relevant_passages(
    passages: List[Dict[str, Any]],
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.08
) -> Dict[str, Any]:
    """
    Finds the most relevant passages for a query using TF-IDF and Cosine Similarity.

    How TF-IDF and Cosine Similarity work here:
    1. TF (Term Frequency): How often query terms appear in each passage.
    2. IDF (Inverse Document Frequency): Downweights ubiquitous words (e.g. 'the', 'is')
       and upweights rare, informative terms (e.g. 'gradient', 'backpropagation').
    3. TfidfVectorizer converts all passages and the question into mathematical vectors.
    4. Cosine Similarity measures the cosine of the angle between the question vector
       and each passage vector, returning a score between 0.0 (no similarity) and 1.0 (exact match).

    Parameters:
        passages: List of passage dictionaries.
        query: User's search query / question string.
        top_k: Number of top matching passages to return (default: 3).
        min_similarity: Minimum similarity threshold below which passages are rejected (default: 0.08).

    Returns:
        dict:
            - 'results': List of matched passages with 'score', 'score_percent', 'snippet', 'page_number', 'text'.
            - 'status': 'SUCCESS', 'NO_MATCH', or 'EMPTY_QUERY'.
            - 'message': User-friendly description of the retrieval outcome.
            - 'best_score': Highest similarity score found (float).
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "results": [],
            "status": "EMPTY_QUERY",
            "message": "Please type a question to search the document.",
            "best_score": 0.0
        }

    if not passages:
        return {
            "results": [],
            "status": "NO_PASSAGES",
            "message": "No text passages available in this document.",
            "best_score": 0.0
        }

    # Extract all passage texts
    corpus = [p["text"] for p in passages]

    # Initialize TF-IDF Vectorizer with unigrams and bigrams
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),  # Matches single words ("deep") and word pairs ("deep learning")
        sublinear_tf=True    # Uses log(1 + tf) to prevent word repetition from dominating
    )

    try:
        # Fit on document passages and compute TF-IDF matrix
        passage_vectors = vectorizer.fit_transform(corpus)
        # Transform user question using the exact same vocabulary
        query_vector = vectorizer.transform([clean_query])
    except ValueError:
        # Happens if document or query contains only unrecognized stop words
        return {
            "results": [],
            "status": "NO_MATCH",
            "message": "Could not extract searchable terms from the query.",
            "best_score": 0.0
        }

    # Compute cosine similarity between query vector and all passage vectors
    # Results in a 1D array of floats between 0.0 and 1.0
    similarity_scores = cosine_similarity(query_vector, passage_vectors).flatten()

    # Get indices of passages sorted by highest similarity first
    ranked_indices = np.argsort(similarity_scores)[::-1]

    top_matches: List[Dict[str, Any]] = []
    best_score = float(similarity_scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0

    # Collect top_k passages that meet or exceed min_similarity
    for idx in ranked_indices:
        score = float(similarity_scores[idx])
        if score < min_similarity:
            break

        passage = passages[idx]
        snippet = extract_best_answer_snippet(passage["text"], clean_query)

        top_matches.append({
            "chunk_id": passage["chunk_id"],
            "page_number": passage["page_number"],
            "text": passage["text"],
            "score": round(score, 4),
            "score_percent": round(score * 100, 1),
            "snippet": snippet
        })

        if len(top_matches) >= top_k:
            break

    # Determine status and helpful message
    if not top_matches:
        return {
            "results": [],
            "status": "NO_MATCH",
            "message": (
                f"No relevant passages found with sufficient confidence (best similarity was {round(best_score * 100, 1)}%, "
                f"threshold is {round(min_similarity * 100, 1)}%). Try rephrasing your question with keywords mentioned in the PDF."
            ),
            "best_score": best_score
        }

    return {
        "results": top_matches,
        "status": "SUCCESS",
        "message": f"Found {len(top_matches)} relevant passage(s) with highest match score of {round(best_score * 100, 1)}%.",
        "best_score": best_score
    }
