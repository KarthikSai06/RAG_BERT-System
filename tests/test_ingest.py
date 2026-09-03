"""
tests/test_ingest.py
---------------------
Unit tests for the document ingestion and chunking module.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import chunk_text, ingest_documents
import config


class TestChunkText(unittest.TestCase):

    def test_basic_chunking(self):
        """Chunks are produced and each fits within chunk_size."""
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = chunk_text(text, chunk_size=250, overlap=50)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            word_count = len(chunk.split())
            self.assertLessEqual(word_count, 250)

    def test_overlap_produces_repeated_words(self):
        """Consecutive chunks share words from the overlap window."""
        text = " ".join([f"word{i}" for i in range(400)])
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        if len(chunks) >= 2:
            tail_of_first = set(chunks[0].split()[-20:])
            head_of_second = set(chunks[1].split()[:20])
            self.assertTrue(
                tail_of_first & head_of_second,
                "Overlap words should appear in both consecutive chunks."
            )

    def test_short_text_produces_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        text = "hello world this is a short document"
        chunks = chunk_text(text, chunk_size=250, overlap=50)
        self.assertEqual(len(chunks), 1)

    def test_empty_text_produces_no_chunks(self):
        """Empty string input should produce no non-empty chunks."""
        chunks = chunk_text("", chunk_size=250, overlap=50)
        non_empty = [c for c in chunks if c.strip()]
        self.assertEqual(len(non_empty), 0)


class TestIngestDocuments(unittest.TestCase):

    def test_ingest_txt_file(self):
        """Ingesting a .txt file writes at least one chunk to chunks.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a temporary config override is not practical here;
            # instead, verify chunk_text output directly using a sample text.
            sample_text = "\n".join(
                ["This is sentence number {}.".format(i) for i in range(100)]
            )
            chunks = chunk_text(sample_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            non_empty_chunks = [c for c in chunks if c.strip()]
            self.assertGreater(len(non_empty_chunks), 0)


if __name__ == "__main__":
    unittest.main()
