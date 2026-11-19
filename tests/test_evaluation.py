"""
tests/test_evaluation.py
Unit tests for model evaluation metrics using unittest.
"""

import unittest
import numpy as np
import pandas as pd
from src.topic_model import TopicModeler
from src.evaluation import (
    compute_perplexity_and_log_likelihood,
    compute_topic_diversity,
    compute_umass_coherence,
    evaluate_model_comprehensively
)


def get_sample_data():
    tech_docs = [
        "python algorithm database server neural network cloud computing",
        "software engineering microservice backend framework security protocol",
        "cybersecurity firewall encryption authentication vulnerability penetration"
    ] * 3

    space_docs = [
        "astronomy telescope cosmic galaxy solar planetary exploration orbit",
        "rocket propulsion lunar spacecraft satellite mars trajectory atmosphere",
        "gravitational cosmological observatory astrophysical nebula universe radiation"
    ] * 3

    return tech_docs + space_docs


class TestEvaluation(unittest.TestCase):

    def test_compute_perplexity_and_log_likelihood(self):
        corpus = get_sample_data()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        metrics = compute_perplexity_and_log_likelihood(tm.lda_model, tm.dtm)
        self.assertIn('perplexity', metrics)
        self.assertIn('log_likelihood', metrics)
        self.assertGreater(metrics['perplexity'], 0)
        self.assertLess(metrics['log_likelihood'], 0)

    def test_compute_topic_diversity(self):
        summaries = [
            {'top_words': ['apple', 'banana', 'orange']},
            {'top_words': ['grape', 'peach', 'cherry']}
        ]
        diversity = compute_topic_diversity(summaries, top_n=3)
        self.assertEqual(diversity, 1.0)

        redundant_summaries = [
            {'top_words': ['apple', 'banana', 'orange']},
            {'top_words': ['apple', 'banana', 'orange']}
        ]
        redundant_diversity = compute_topic_diversity(redundant_summaries, top_n=3)
        self.assertEqual(redundant_diversity, 0.5)

    def test_compute_umass_coherence(self):
        corpus = get_sample_data()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        topic_words = [s['top_words'] for s in tm.topic_summaries]
        mean_umass, topic_scores = compute_umass_coherence(
            topic_words=topic_words,
            dtm=tm.dtm,
            feature_names=tm.feature_names,
            top_n=5
        )
        self.assertIsInstance(mean_umass, float)
        self.assertEqual(len(topic_scores), 2)
        self.assertLessEqual(mean_umass, 0.0)

    def test_compute_umass_coherence_raw_sum(self):
        corpus = get_sample_data()
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        topic_words = [s['top_words'] for s in tm.topic_summaries]
        raw_sum, topic_raw = compute_umass_coherence(
            topic_words=topic_words,
            dtm=tm.dtm,
            feature_names=tm.feature_names,
            top_n=5,
            normalize=False
        )
        self.assertLessEqual(raw_sum, 0.0)
        # Raw sum across 10 pairs (5*4/2) should be larger in magnitude than mean pairwise
        norm_mean, _ = compute_umass_coherence(
            topic_words=topic_words,
            dtm=tm.dtm,
            feature_names=tm.feature_names,
            top_n=5,
            normalize=True
        )
        self.assertAlmostEqual(raw_sum / 10, norm_mean, places=2)

    def test_evaluate_model_comprehensively(self):
        corpus = get_sample_data()
        df = pd.DataFrame({'clean_text': corpus})
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        results = evaluate_model_comprehensively(
            topic_modeler=tm,
            cleaned_df=df,
            text_column='clean_text',
            top_n=5
        )

        self.assertIn('perplexity', results)
        self.assertIn('log_likelihood', results)
        self.assertIn('topic_diversity', results)
        self.assertIn('mean_umass_coherence', results)
        self.assertIn('educational_note', results)
        self.assertEqual(results['n_topics'], 2)


if __name__ == '__main__':
    unittest.main()
