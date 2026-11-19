"""
tests/test_topic_model.py
Unit tests for LDA Topic Model using unittest.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from src.topic_model import TopicModeler


def get_mock_corpus():
    space_docs = [
        "astronomy space telescope planetary orbit galaxy mission exploration nasa",
        "rocket launch spacecraft astronaut lunar mars orbit satellite cosmic propulsion",
        "planetary astrophysics gravitational solar radiation celestial spectrum nebula"
    ] * 4

    health_docs = [
        "clinical medical health patient disease therapy treatment physician doctor",
        "cardiology surgery pharmaceutical dosage symptom infection prescription clinic hospital",
        "biological cellular genetic antibody vaccine trial medication diagnosis therapy"
    ] * 4

    return space_docs + health_docs


class TestTopicModel(unittest.TestCase):

    def test_topic_modeler_fit_and_summaries(self):
        corpus = get_mock_corpus()
        tm = TopicModeler(
            n_topics=2,
            max_iter=15,
            min_df=1,
            max_df=1.0,
            random_state=42,
            top_words_count=5
        )
        tm.fit(corpus)

        self.assertTrue(tm.is_fitted)
        self.assertEqual(len(tm.topic_summaries), 2)
        self.assertIsNotNone(tm.dtm)
        self.assertEqual(tm.dtm.shape[0], len(corpus))
        self.assertGreater(len(tm.feature_names), 5)

        s0 = tm.topic_summaries[0]
        self.assertIn('top_words', s0)
        self.assertIn('top_weights', s0)
        self.assertEqual(len(s0['top_words']), 5)

    def test_topic_modeler_document_topic_df(self):
        corpus = get_mock_corpus()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        dummy_df = pd.DataFrame({'clean_text': corpus})
        doc_df = tm.get_document_topic_df(dummy_df, 'clean_text')

        self.assertIn('Dominant_Topic_ID', doc_df.columns)
        self.assertIn('Dominant_Topic_Label', doc_df.columns)
        self.assertIn('Dominant_Topic_Probability', doc_df.columns)
        self.assertIn('Topic_1_Prob', doc_df.columns)
        self.assertIn('Topic_2_Prob', doc_df.columns)

        prob_sums = doc_df['Topic_1_Prob'] + doc_df['Topic_2_Prob']
        np.testing.assert_allclose(prob_sums, 1.0, atol=0.01)

    def test_topic_modeler_save_and_load(self):
        corpus = get_mock_corpus()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_lda_model.joblib"
            saved_path = tm.save(model_path)
            self.assertTrue(Path(saved_path).exists())

            loaded_tm = TopicModeler.load(saved_path)
            self.assertTrue(loaded_tm.is_fitted)
            self.assertEqual(loaded_tm.n_topics, 2)
            self.assertEqual(len(loaded_tm.feature_names), len(tm.feature_names))

            pred = loaded_tm.transform_new_text("space telescope galaxy orbit")
            self.assertIn('dominant_topic_label', pred)
            self.assertIn('dominant_probability', pred)
            self.assertGreater(pred['dominant_probability'], 0.5)

    def test_topic_modeler_custom_priors(self):
        corpus = get_mock_corpus()
        tm = TopicModeler(
            n_topics=2,
            min_df=1,
            random_state=42,
            learning_decay=0.8,
            doc_topic_prior=0.5,
            topic_word_prior=0.1
        )
        tm.fit(corpus)
        self.assertTrue(tm.is_fitted)
        self.assertEqual(tm.learning_decay, 0.8)
        self.assertEqual(tm.doc_topic_prior, 0.5)
        self.assertEqual(tm.topic_word_prior, 0.1)

    def test_topic_modeler_persists_preprocessor_and_custom_labels(self):
        from src.preprocessing import TextPreprocessor
        prep = TextPreprocessor(lowercase=True, min_word_len=4)
        corpus = get_mock_corpus()
        tm = TopicModeler(
            n_topics=2,
            min_df=1,
            random_state=42,
            preprocessor=prep,
            custom_labels={1: "Deep Space Exploration", 2: "Clinical Medicine"}
        )
        tm.fit(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "full_bundle_model.joblib"
            tm.save(model_path)
            loaded = TopicModeler.load(model_path)

            self.assertIsNotNone(loaded.preprocessor)
            self.assertEqual(loaded.preprocessor.min_word_len, 4)
            self.assertEqual(loaded.custom_labels.get(1), "Deep Space Exploration")
            self.assertEqual(loaded.custom_labels.get(2), "Clinical Medicine")

            # Check that custom label is used in transform_new_text
            pred = loaded.transform_new_text("space telescope planetary orbit")
            self.assertIn("Deep Space Exploration", pred['dominant_topic_label'])

    def test_document_topic_df_validates_shape_mismatch(self):
        corpus = get_mock_corpus()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        # Pass a DataFrame with different length
        wrong_df = pd.DataFrame({'clean_text': ["only one document"]})
        with self.assertRaises(ValueError):
            tm.get_document_topic_df(wrong_df, 'clean_text')


if __name__ == '__main__':
    unittest.main()
