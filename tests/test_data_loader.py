"""
tests/test_data_loader.py
Unit tests for data loading and exploratory summary utilities using unittest.
"""

import unittest
from io import StringIO
import pandas as pd
from src.data_loader import (
    load_csv_data,
    load_txt_files,
    load_sample_dataset,
    get_dataset_summary,
    detect_text_columns
)


class TestDataLoader(unittest.TestCase):

    def test_detect_text_columns(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Machine Learning', 'Deep Learning', 'Neural Network'],
            'content': [
                'Artificial intelligence and machine learning are revolutionizing NLP.',
                'Deep neural networks require backpropagation and optimization.',
                'Recurrent networks and transformers excel at sequential textual modeling.'
            ]
        })
        detected = detect_text_columns(df)
        self.assertIn('content', detected)

    def test_load_csv_data_string_io(self):
        csv_text = "id,text\n1,First document text content\n2,Second document text content\n"
        buffer = StringIO(csv_text)
        df, cols, selected = load_csv_data(buffer)
        self.assertEqual(len(df), 2)
        self.assertEqual(selected, 'text')

    def test_load_sample_dataset(self):
        df, col = load_sample_dataset('customer_feedback')
        self.assertGreater(len(df), 0)
        self.assertIn(col, df.columns)

    def test_load_sample_dataset_biomedical(self):
        df, col = load_sample_dataset('biomedical_research')
        self.assertEqual(len(df), 25)
        self.assertIn(col, df.columns)

    def test_load_empty_csv_raises(self):
        empty_buffer = StringIO("")
        with self.assertRaises(ValueError):
            load_csv_data(empty_buffer)

        zero_rows_buffer = StringIO("col_a,col_b\n")
        with self.assertRaises(ValueError):
            load_csv_data(zero_rows_buffer)

    def test_get_dataset_summary(self):
        df = pd.DataFrame({
            'text': [
                'The quick brown fox jumps over the lazy dog.',
                'Natural language processing with Python is fascinating.',
                'Machine learning models require clean training datasets.',
                None,
                '   '
            ]
        })
        summary = get_dataset_summary(df, 'text')
        self.assertEqual(summary['total_documents'], 5)
        self.assertEqual(summary['missing_values'], 1)
        self.assertEqual(summary['empty_documents'], 1)
        self.assertEqual(summary['valid_documents'], 3)
        self.assertGreater(summary['avg_word_count'], 5.0)


if __name__ == '__main__':
    unittest.main()
