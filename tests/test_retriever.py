"""
tests/test_retriever.py
------------------------
Unit tests for the FAISS retriever module.

If the FAISS index has not been built yet (indexer.py not run),
the retriever falls back gracefully and returns an empty list.
These tests validate both the fallback path and, when the index
exists, the shape and type of returned results.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.retriever import Retriever


class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()
        cls.index_available = cls.retriever.index is not None

    def test_retriever_initialises_without_error(self):
        """Retriever must initialise without raising an exception."""
        self.assertIsNotNone(self.retriever)

    def test_retrieve_returns_list(self):
        """retrieve() must always return a list."""
        result = self.retriever.retrieve("What is the fee deadline?")
        self.assertIsInstance(result, list)

    def test_retrieve_respects_k(self):
        """retrieve() must return at most k results."""
        if not self.index_available:
            self.skipTest("FAISS index not built. Run indexer.py first.")
        for k in [1, 3, 5]:
            results = self.retriever.retrieve("What is the attendance requirement?", k=k)
            self.assertLessEqual(len(results), k)

    def test_retrieve_result_has_required_keys(self):
        """Each retrieved chunk must contain chunk_id, source_file, text, and score."""
        if not self.index_available:
            self.skipTest("FAISS index not built. Run indexer.py first.")
        results = self.retriever.retrieve("How do I apply for a scholarship?", k=1)
        if results:
            required_keys = {"chunk_id", "source_file", "text", "score"}
            self.assertTrue(required_keys.issubset(results[0].keys()))

    def test_score_in_valid_range(self):
        """Cosine similarity score after L2 normalisation must be in [-1, 1]."""
        if not self.index_available:
            self.skipTest("FAISS index not built. Run indexer.py first.")
        results = self.retriever.retrieve("What is the hostel curfew time?", k=3)
        for r in results:
            self.assertGreaterEqual(r["score"], -1.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_retrieve_does_not_mutate_cached_chunks(self):
        """Two successive retrieve() calls must return independent dicts."""
        if not self.index_available:
            self.skipTest("FAISS index not built. Run indexer.py first.")
        query = "What is the fee deadline?"
        r1 = self.retriever.retrieve(query, k=1)
        r2 = self.retriever.retrieve(query, k=1)
        if r1 and r2:
            # Mutating r1 must not affect r2
            r1[0]["score"] = -999.0
            self.assertNotEqual(r2[0].get("score"), -999.0)


if __name__ == "__main__":
    unittest.main()
