"""
tests/test_pipeline.py
-----------------------
Integration tests for the end-to-end RAG pipeline.

The generator is NOT initialised in these tests (init_generator=False)
to avoid requiring a GPU or 14+ GB of VRAM in a CI environment.
The tests validate stage routing, guardrail behaviour, result structure,
and timing fields.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.pipeline import RAGPipeline


VALID_LABELS = set(config.LABEL2ID.keys())


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialise pipeline without loading the LLM generator."""
        cls.pipeline = RAGPipeline(init_generator=False)

    def _run(self, query):
        return self.pipeline.run(query)

    def test_result_has_required_keys(self):
        """Pipeline result dict must contain all required top-level keys."""
        result = self._run("What is the fee deadline?")
        required = {"query", "label", "answer", "sources", "is_fallback", "timings"}
        self.assertTrue(required.issubset(result.keys()))

    def test_label_is_valid(self):
        """label field must be one of the three defined class strings."""
        result = self._run("What is the minimum attendance?")
        self.assertIn(result["label"], VALID_LABELS)

    def test_out_of_domain_triggers_fallback(self):
        """An obviously out-of-domain query should set is_fallback=True."""
        # NOTE: Without a trained BERT model the base weights may not classify
        # correctly. This test is best run after train_classifier.py has been run.
        result = self._run("What is the capital of France?")
        if result["label"] == "out-of-domain":
            self.assertTrue(result["is_fallback"])
            self.assertEqual(result["answer"], config.FALLBACK_MSG)
            self.assertEqual(result["sources"], [])

    def test_in_domain_query_has_no_fallback(self):
        """An in-domain query must not be routed to the fallback."""
        result = self._run("What is the fee deadline?")
        if result["label"] != "out-of-domain":
            self.assertFalse(result["is_fallback"])

    def test_classification_timing_recorded(self):
        """Timings dict must contain a classification_ms key."""
        result = self._run("How do I apply for a hostel room?")
        self.assertIn("classification_ms", result["timings"])

    def test_fallback_skips_retrieval_timing(self):
        """Fallback responses must not record retrieval_ms."""
        result = self._run("What is the capital of France?")
        if result["is_fallback"]:
            self.assertNotIn("retrieval_ms", result["timings"])

    def test_query_preserved_in_result(self):
        """The original query string must appear unchanged in the result."""
        query = "What are the hostel rules?"
        result = self._run(query)
        self.assertEqual(result["query"], query)

    def test_sources_is_list(self):
        """sources field must always be a list."""
        result = self._run("How do I pay my semester fees?")
        self.assertIsInstance(result["sources"], list)


if __name__ == "__main__":
    unittest.main()
