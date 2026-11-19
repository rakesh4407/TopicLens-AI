"""
tests/test_feature_engineering.py
Unit tests for src/feature_engineering.py FeatureEngineer class.
"""

import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import FeatureEngineer


SAMPLE_TEXTS = [
    "Machine learning algorithms are transforming artificial intelligence and data science applications.",
    "Deep neural networks learn representations from large datasets using gradient descent optimization.",
    "Space exploration missions study planetary atmospheres and exoplanet biosignatures from orbital telescopes.",
    "Rocket propulsion systems use liquid oxygen and methane fuels for reusable launch vehicles.",
    "Financial markets respond to central bank interest rate decisions and monetary policy announcements.",
    "Portfolio diversification across asset classes reduces systematic investment risk and volatility.",
    "Healthcare systems deploy electronic medical records and patient data analytics platforms.",
    "Clinical trials measure vaccine efficacy using randomized controlled study methodologies.",
    "Natural language processing enables machines to understand and generate human textual communication.",
    "Topic modelling discovers hidden semantic themes within unstructured document collections using LDA.",
]


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.fe = FeatureEngineer(min_df=1, max_df=1.0, max_features=500)

    def test_build_dtm_returns_correct_shape(self):
        dtm, feature_names = self.fe.build_dtm(SAMPLE_TEXTS)
        self.assertEqual(dtm.shape[0], len(SAMPLE_TEXTS))
        self.assertGreater(dtm.shape[1], 0)
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(dtm.shape[1], len(feature_names))

    def test_build_dtm_sets_is_fitted(self):
        self.assertFalse(self.fe.is_fitted)
        self.fe.build_dtm(SAMPLE_TEXTS)
        self.assertTrue(self.fe.is_fitted)

    def test_dtm_stats_accuracy(self):
        self.fe.build_dtm(SAMPLE_TEXTS)
        stats = self.fe.get_dtm_stats()
        self.assertEqual(stats["n_documents"], len(SAMPLE_TEXTS))
        self.assertGreater(stats["vocabulary_size"], 0)
        self.assertGreaterEqual(stats["sparsity"], 0.0)
        self.assertLessEqual(stats["sparsity"], 1.0)
        self.assertAlmostEqual(stats["sparsity"] + stats["density"], 1.0, places=4)

    def test_inspect_vocabulary_returns_dataframe(self):
        self.fe.build_dtm(SAMPLE_TEXTS)
        vocab_df = self.fe.inspect_vocabulary(top_n=10)
        self.assertIn("Term", vocab_df.columns)
        self.assertIn("Total Frequency", vocab_df.columns)
        self.assertIn("Document Frequency", vocab_df.columns)
        self.assertLessEqual(len(vocab_df), 10)

    def test_min_df_filtering(self):
        fe_strict = FeatureEngineer(min_df=2, max_df=1.0, max_features=500)
        fe_loose = FeatureEngineer(min_df=1, max_df=1.0, max_features=500)
        _, names_strict = fe_strict.build_dtm(SAMPLE_TEXTS)
        _, names_loose = fe_loose.build_dtm(SAMPLE_TEXTS)
        # min_df=2 should produce equal or smaller vocabulary
        self.assertLessEqual(len(names_strict), len(names_loose))

    def test_ngram_range_bigrams(self):
        fe_bigram = FeatureEngineer(min_df=1, max_df=1.0, max_features=500, ngram_range=(1, 2))
        _, names_bigram = fe_bigram.build_dtm(SAMPLE_TEXTS)
        # Bigram vocab should be larger than unigram
        _, names_unigram = self.fe.build_dtm(SAMPLE_TEXTS)
        self.assertGreaterEqual(len(names_bigram), len(names_unigram))

    def test_build_tfidf(self):
        tfidf_matrix, tfidf_names = self.fe.build_tfidf(SAMPLE_TEXTS)
        self.assertEqual(tfidf_matrix.shape[0], len(SAMPLE_TEXTS))
        self.assertGreater(len(tfidf_names), 0)

    def test_transform_new_texts(self):
        self.fe.build_dtm(SAMPLE_TEXTS)
        new_texts = ["Quantum computing leverages qubit superposition."]
        result = self.fe.transform(new_texts)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], len(self.fe.feature_names))

    def test_dtm_stats_raises_before_fit(self):
        fe_new = FeatureEngineer()
        with self.assertRaises(ValueError):
            fe_new.get_dtm_stats()

    def test_inspect_vocabulary_raises_before_fit(self):
        fe_new = FeatureEngineer()
        with self.assertRaises(ValueError):
            fe_new.inspect_vocabulary()

    def test_transform_raises_before_fit(self):
        fe_new = FeatureEngineer()
        with self.assertRaises(ValueError):
            fe_new.transform(["some text"])


if __name__ == "__main__":
    unittest.main()
