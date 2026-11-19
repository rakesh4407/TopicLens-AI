"""
tests/test_preprocessing.py
Unit tests for text preprocessing pipeline using unittest.
"""

import unittest
import pandas as pd
from src.preprocessing import (
    TextPreprocessor,
    preprocess_corpus,
    get_before_after_samples
)


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_basic(self):
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_punctuation=True,
            remove_numbers=True,
            remove_stopwords=True,
            lemmatize=True,
            min_word_len=3
        )

        dirty_text = "<p>Visit https://example.com for 100 new AI features and updates!</p>"
        cleaned = preprocessor.clean_text(dirty_text)

        # HTML tags removed
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("</p>", cleaned)
        # URL removed
        self.assertNotIn("http", cleaned)
        self.assertNotIn("example", cleaned)
        # Digits removed
        self.assertNotIn("100", cleaned)
        # Punctuation removed
        self.assertNotIn("!", cleaned)
        # Tokens present
        self.assertTrue("feature" in cleaned or "features" in cleaned)
        self.assertTrue("update" in cleaned or "updates" in cleaned)

    def test_preprocess_corpus_pipeline(self):
        df = pd.DataFrame({
            'doc_id': [1, 2, 3, 4],
            'raw_text': [
                "Natural Language Processing (NLP) enables computers to understand human language.",
                "Deep neural networks and convolutional layers power modern computer vision.",
                "Natural Language Processing (NLP) enables computers to understand human language.",  # Duplicate
                "    "  # Empty
            ]
        })

        cleaned_df, stats = preprocess_corpus(
            df=df,
            text_column='raw_text',
            drop_empty=True,
            drop_duplicates=True
        )

        self.assertEqual(stats['final_document_count'], 2)
        self.assertGreaterEqual(stats['removed_empty_count'], 1)
        self.assertIn('clean_text', cleaned_df.columns)
        self.assertIn('clean_word_count', cleaned_df.columns)
        self.assertGreater(stats['vocab_reduction_percent'], 0)

    def test_get_before_after_samples(self):
        orig = pd.Series(["Hello <b>world</b>!", "Testing NLP 123"])
        clean = pd.Series(["world", "testing nlp"])
        samples = get_before_after_samples(orig, clean, n_samples=2)
        self.assertEqual(len(samples), 2)
        self.assertIn("<b>", samples[0]['original'])
        self.assertIn("world", samples[0]['cleaned'])


if __name__ == '__main__':
    unittest.main()
