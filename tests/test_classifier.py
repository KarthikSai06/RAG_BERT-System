"""
tests/test_classifier.py
-------------------------
Unit tests for the BERT query classifier module.
Tests cover label set validity, predict() return type,
and predict_with_confidence() confidence bounds.

Note: These tests load the base BERT weights (not fine-tuned) because
the fine-tuned model is only available after running train_classifier.py.
Predictions with the base model will be random but the interface contract
is what is being tested here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.classifier import QueryClassifier


VALID_LABELS = set(config.LABEL2ID.keys())


class TestQueryClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load classifier once for all tests in this class."""
        cls.classifier = QueryClassifier()

    def test_predict_returns_valid_label(self):
        """predict() must return one of the three defined label strings."""
        query = "What is the fee deadline?"
        label = self.classifier.predict(query)
        self.assertIn(label, VALID_LABELS)

    def test_predict_with_confidence_returns_tuple(self):
        """predict_with_confidence() must return a (str, float) tuple."""
        query = "How do I apply for a hostel room?"
        result = self.classifier.predict_with_confidence(query)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        label, confidence = result
        self.assertIn(label, VALID_LABELS)
        self.assertIsInstance(confidence, float)

    def test_confidence_in_range(self):
        """Confidence score must be in [0.0, 1.0]."""
        query = "Who won the FIFA World Cup?"
        _, confidence = self.classifier.predict_with_confidence(query)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_multiple_queries(self):
        """predict() must handle multiple queries without error."""
        queries = [
            "What is the minimum attendance required?",
            "How do I register for examinations?",
            "What is the capital of France?",
        ]
        for query in queries:
            label = self.classifier.predict(query)
            self.assertIn(label, VALID_LABELS, f"Invalid label for query: {query}")

    def test_predict_empty_string(self):
        """predict() must handle an empty string without raising an exception."""
        try:
            label = self.classifier.predict("")
            self.assertIn(label, VALID_LABELS)
        except Exception as e:
            self.fail(f"predict() raised an exception on empty string: {e}")


if __name__ == "__main__":
    unittest.main()
