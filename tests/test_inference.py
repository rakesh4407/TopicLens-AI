"""
tests/test_inference.py
Unit tests for src/inference.py TopicInferencer class.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.topic_model import TopicModeler
from src.inference import TopicInferencer


TRAIN_DOCS = [
    "Artificial intelligence and deep neural networks are transforming computer vision diagnostics.",
    "Cloud computing architectures with microservices provide scalable infrastructure for web platforms.",
    "Quantum computing algorithms utilize qubit superposition to solve complex optimization problems.",
    "NASA space telescope detected ancient galactic formations near the cosmic cosmological boundary.",
    "Mars rover missions analyze regolith mineral samples to detect biosignatures and subterranean ice.",
    "Rocket propulsion systems utilizing cryogenic liquid methane improve specific impulse and reusability.",
    "Cardiovascular exercise and balanced dietary nutrition lower arterial blood pressure risks.",
    "Clinical trials evaluate mRNA vaccine efficacy and synthetic antigen antibody responses.",
    "Pharmacological therapies targeting cellular inflammation demonstrate promising outcomes.",
    "Portfolio diversification across global equity markets and fixed income securities mitigates risk.",
    "Central bank interest rate decisions and monetary policies directly influence bond yields.",
    "Fintech innovations in distributed ledger settlements and automated clearinghouses lower fees.",
]


class TestTopicInferencer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tm = TopicModeler(n_topics=4, max_iter=20, random_state=42, min_df=1)
        cls.tm.fit(TRAIN_DOCS)
        cls.inferencer = TopicInferencer(cls.tm)

    def test_inferencer_requires_fitted_model(self):
        unfitted = TopicModeler(n_topics=3)
        with self.assertRaises(ValueError):
            TopicInferencer(unfitted)

    def test_predict_single_doc_structure(self):
        query = "Deep learning neural network model training for artificial intelligence vision."
        res = self.inferencer.predict(query)

        self.assertIn("text", res)
        self.assertIn("distribution", res)
        self.assertIn("dominant_topic_id", res)
        self.assertIn("dominant_topic_label", res)
        self.assertIn("dominant_probability", res)
        self.assertIn("confidence_level", res)

        self.assertEqual(len(res["distribution"]), 4)
        self.assertGreater(res["dominant_probability"], 0.0)
        self.assertIn(res["confidence_level"], ["High", "Medium", "Low"])

    def test_predict_empty_text_raises(self):
        with self.assertRaises(ValueError):
            self.inferencer.predict("")
        with self.assertRaises(ValueError):
            self.inferencer.predict("   ")

    def test_batch_predict_dataframe(self):
        test_batch = [
            "NASA rover discovers water ice beneath Mars regolith surface.",
            "Central bank monetary policies and interest rates determine corporate bond returns.",
            "Clinical research into mRNA vaccines and immunology.",
        ]
        batch_df = self.inferencer.batch_predict(test_batch)

        self.assertIsInstance(batch_df, pd.DataFrame)
        self.assertEqual(len(batch_df), len(test_batch))
        self.assertIn("dominant_topic_id", batch_df.columns)
        self.assertIn("dominant_topic_label", batch_df.columns)
        self.assertIn("dominant_probability", batch_df.columns)
        self.assertIn("confidence_level", batch_df.columns)

    def test_get_topic_keywords(self):
        kw = self.inferencer.get_topic_keywords(topic_id=1, top_n=5)
        self.assertEqual(len(kw), 5)
        self.assertTrue(all(isinstance(w, str) for w in kw))

    def test_get_topic_keywords_invalid_id(self):
        with self.assertRaises(ValueError):
            self.inferencer.get_topic_keywords(topic_id=0)
        with self.assertRaises(ValueError):
            self.inferencer.get_topic_keywords(topic_id=99)


if __name__ == "__main__":
    unittest.main()
