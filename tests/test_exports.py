"""
tests/test_exports.py
Unit tests for data export and report generation using unittest.
"""

import unittest
import pandas as pd
from src.topic_model import TopicModeler
from src.utils import (
    df_to_csv_download,
    generate_analysis_report_markdown,
    generate_analysis_report_html
)


class TestExports(unittest.TestCase):

    def test_df_to_csv_download(self):
        df = pd.DataFrame({'topic': ['A', 'B'], 'weight': [0.8, 0.2]})
        csv_bytes = df_to_csv_download(df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertIn(b"topic,weight", csv_bytes)

    def test_generate_analysis_report(self):
        corpus = [
            "python data science algorithm artificial intelligence",
            "astronomy telescope cosmic galaxy solar planetary"
        ] * 4
        tm = TopicModeler(n_topics=2, min_df=1, random_state=42)
        tm.fit(corpus)

        dataset_summary = {
            'total_documents': 8,
            'valid_documents': 8,
            'avg_word_count': 6.0,
            'duplicate_documents': 0,
            'text_column': 'text'
        }
        preprocessing_summary = {
            'original_document_count': 8,
            'final_document_count': 8,
            'raw_vocab_size': 10,
            'clean_vocab_size': 10,
            'vocab_reduction_percent': 0.0,
            'avg_clean_doc_length': 6.0
        }
        eval_metrics = {
            'perplexity': 15.2,
            'log_likelihood': -250.0,
            'topic_diversity': 1.0,
            'mean_umass_coherence': -1.2
        }

        md_report = generate_analysis_report_markdown(
            dataset_summary=dataset_summary,
            preprocessing_summary=preprocessing_summary,
            topic_modeler=tm,
            evaluation_metrics=eval_metrics
        )

        self.assertIn("TopicLens AI", md_report)
        self.assertIn("Executive Topic Discovery", md_report)
        self.assertIn("Model Perplexity", md_report)

        html_report = generate_analysis_report_html(md_report)
        self.assertIn("<html", html_report)
        self.assertIn("TopicLens AI", html_report)


if __name__ == '__main__':
    unittest.main()
