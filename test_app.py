"""
test_app.py - Automated Unit & Integration Tests for AI PDF QA Assistant

Validates:
1. PDF text extraction & page numbering via PyMuPDF.
2. Passage splitting & overlap mechanics.
3. TF-IDF vectorization & Cosine Similarity ranking.
4. Page provenance & extractive snippet selection.
5. Edge cases: Empty PDF, Scanned/Empty text, Irrelevant queries, Out-of-vocabulary terms.
"""

import os
import unittest
import pymupdf
from qa_engine import (
    extract_text_from_pdf,
    split_into_passages,
    find_most_relevant_passages,
    extract_best_answer_snippet,
    clean_whitespace,
)
from create_sample_pdf import generate_sample_pdf


class TestAIPDFAssistant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates sample PDF for the test suite."""
        cls.sample_pdf_path = "sample_ai_overview.pdf"
        if not os.path.exists(cls.sample_pdf_path):
            generate_sample_pdf(cls.sample_pdf_path)

        with open(cls.sample_pdf_path, "rb") as f:
            cls.sample_pdf_bytes = f.read()

        cls.pages_data, cls.metadata = extract_text_from_pdf(cls.sample_pdf_bytes)
        cls.passages = split_into_passages(cls.pages_data)

    def test_01_pdf_extraction_page_count(self):
        """Tests that all 4 pages are accurately extracted with correct 1-indexed page numbers."""
        self.assertEqual(len(self.pages_data), 4)
        self.assertEqual(self.metadata["total_pages"], 4)
        self.assertTrue(self.metadata["has_text"])
        self.assertGreater(self.metadata["total_words"], 200)

        for idx, page in enumerate(self.pages_data, 1):
            self.assertEqual(page["page_number"], idx)
            self.assertGreater(len(page["text"]), 50)
            self.assertGreater(page["word_count"], 10)

    def test_02_whitespace_cleaning(self):
        """Tests text sanitization for excessive newlines and tabs."""
        dirty = "Hello \xa0 world!   \n\n\n\nThis is   a test.   \r\n\r\nAnother line."
        cleaned = clean_whitespace(dirty)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Hello world!", cleaned)

    def test_03_passage_chunking(self):
        """Tests passage chunking preservation of page numbers and non-empty text."""
        self.assertGreaterEqual(len(self.passages), 4)
        for chunk in self.passages:
            self.assertIn("chunk_id", chunk)
            self.assertIn("page_number", chunk)
            self.assertIn("text", chunk)
            self.assertGreater(chunk["word_count"], 0)
            self.assertIn(chunk["page_number"], [1, 2, 3, 4])

    def test_04_query_page_1_supervised_learning(self):
        """Tests retrieval for a question answered on Page 1."""
        query = "What is supervised learning and labeled datasets?"
        res = find_most_relevant_passages(self.passages, query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(len(res["results"]), 0)
        top_match = res["results"][0]

        # Must cite Page 1
        self.assertEqual(top_match["page_number"], 1)
        self.assertIn("supervised", top_match["text"].lower())
        self.assertGreater(top_match["score"], 0.20)

    def test_05_query_page_2_relu_backpropagation(self):
        """Tests retrieval for a question answered on Page 2."""
        query = "How does ReLU activation and Backpropagation work?"
        res = find_most_relevant_passages(self.passages, query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        top_match = res["results"][0]

        # Must cite Page 2
        self.assertEqual(top_match["page_number"], 2)
        self.assertIn("relu", top_match["text"].lower())
        self.assertIn("backpropagation", top_match["text"].lower())

    def test_06_query_page_3_tfidf_cosine_similarity(self):
        """Tests retrieval for a question answered on Page 3."""
        query = "Explain TF-IDF and Cosine Similarity in document search"
        res = find_most_relevant_passages(self.passages, query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        top_match = res["results"][0]

        # Must cite Page 3
        self.assertEqual(top_match["page_number"], 3)
        self.assertIn("cosine", top_match["text"].lower())

    def test_07_query_page_4_precision_recall(self):
        """Tests retrieval for a question answered on Page 4."""
        query = "What are precision, recall, and F1 score?"
        res = find_most_relevant_passages(self.passages, query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        top_match = res["results"][0]

        # Must cite Page 4
        self.assertEqual(top_match["page_number"], 4)
        self.assertIn("recall", top_match["text"].lower())

    def test_08_irrelevant_out_of_scope_query(self):
        """Tests that questions unrelated to the document return NO_MATCH or score < 0.08."""
        irrelevant_query = "What are the ingredients needed to bake a chocolate strawberry birthday cake?"
        res = find_most_relevant_passages(self.passages, irrelevant_query, top_k=3, min_similarity=0.08)

        # Should either return NO_MATCH or have zero/near-zero matches
        self.assertIn(res["status"], ["NO_MATCH", "EMPTY_QUERY"])
        self.assertEqual(len(res["results"]), 0)

    def test_09_empty_and_blank_queries(self):
        """Tests handling of empty and blank query strings."""
        res_empty = find_most_relevant_passages(self.passages, "", top_k=3)
        self.assertEqual(res_empty["status"], "EMPTY_QUERY")

        res_spaces = find_most_relevant_passages(self.passages, "     ", top_k=3)
        self.assertEqual(res_spaces["status"], "EMPTY_QUERY")

    def test_10_empty_pdf_bytes_handling(self):
        """Tests that 0-byte or corrupted inputs raise informative ValueErrors."""
        with self.assertRaises(ValueError):
            extract_text_from_pdf(b"")

        with self.assertRaises(ValueError):
            extract_text_from_pdf(b"This is completely invalid PDF header content")

    def test_11_scanned_or_textless_pdf_detection(self):
        """Tests that a PDF with blank pages correctly reports has_text=False."""
        blank_doc = pymupdf.open()
        blank_doc.new_page()
        blank_bytes = blank_doc.tobytes()
        blank_doc.close()

        pages, meta = extract_text_from_pdf(blank_bytes)
        self.assertFalse(meta["has_text"])
        self.assertEqual(meta["total_pages"], 1)
        self.assertEqual(meta["total_words"], 0)

    def test_12_snippet_extraction(self):
        """Tests that extract_best_answer_snippet prioritizes query keywords."""
        passage = (
            "Machine Learning is a subset of AI. "
            "In Supervised Learning, models train on labeled datasets containing inputs and known outputs. "
            "Linear regression is a popular example."
        )
        snippet = extract_best_answer_snippet(passage, "What is supervised learning and labeled data?")
        self.assertIn("Supervised Learning", snippet)
        self.assertIn("labeled", snippet.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
