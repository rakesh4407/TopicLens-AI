"""
tests/test_e2e_pipeline.py
End-to-End integration test covering the entire TopicLens AI lifecycle:
Ingestion -> Preprocessing -> LDA Modelling -> Visualization -> Evaluation -> Export & Reload.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.data_loader import load_sample_dataset, get_dataset_summary
from src.preprocessing import TextPreprocessor, preprocess_corpus, get_before_after_samples
from src.topic_model import TopicModeler
from src.evaluation import evaluate_model_comprehensively
from src.visualization import (
    plot_topic_distribution,
    plot_topic_pie,
    plot_topic_top_words,
    plot_all_topics_grid,
    plot_document_topic_heatmap,
    plot_document_pca_projection,
    generate_topic_wordcloud,
    generate_overall_wordcloud
)
from src.utils import (
    generate_analysis_report_markdown,
    generate_analysis_report_html,
    df_to_csv_download
)


class TestE2EPipeline(unittest.TestCase):

    def test_complete_topic_modelling_workflow(self):
        print("\n--- 1. Testing Dataset Ingestion ---")
        df, text_col = load_sample_dataset('customer_feedback')
        self.assertGreater(len(df), 10)
        summary = get_dataset_summary(df, text_col)
        self.assertEqual(summary['missing_values'], 0)
        print(f"Loaded {len(df)} documents. Avg word count: {summary['avg_word_count']}")

        print("\n--- 2. Testing Text Preprocessing ---")
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_punctuation=True,
            remove_numbers=True,
            remove_stopwords=True,
            lemmatize=True,
            min_word_len=3,
            custom_stopwords=['laptop', 'battery']
        )
        cleaned_df, prep_stats = preprocess_corpus(
            df=df,
            text_column=text_col,
            preprocessor=preprocessor,
            drop_empty=True,
            drop_duplicates=True
        )
        self.assertGreater(len(cleaned_df), 5)
        self.assertGreater(prep_stats['clean_vocab_size'], 10)
        print(f"Cleaned documents: {len(cleaned_df)}, Clean vocab size: {prep_stats['clean_vocab_size']}")

        print("\n--- 3. Testing LDA Topic Modelling ---")
        modeler = TopicModeler(
            n_topics=3,
            max_iter=15,
            learning_method='batch',
            min_df=1,
            max_df=0.95,
            random_state=42,
            top_words_count=8
        )
        modeler.fit(cleaned_df['clean_text'])
        self.assertTrue(modeler.is_fitted)
        self.assertEqual(len(modeler.topic_summaries), 3)

        doc_topic_df = modeler.get_document_topic_df(cleaned_df, 'clean_text')
        self.assertEqual(len(doc_topic_df), len(cleaned_df))
        self.assertIn('Dominant_Topic_ID', doc_topic_df.columns)
        self.assertIn('Dominant_Topic_Probability', doc_topic_df.columns)

        print("\n--- 4. Testing Model Evaluation ---")
        eval_res = evaluate_model_comprehensively(
            topic_modeler=modeler,
            cleaned_df=cleaned_df,
            text_column='clean_text',
            top_n=8
        )
        self.assertGreater(eval_res['perplexity'], 0)
        self.assertLess(eval_res['log_likelihood'], 0)
        self.assertGreaterEqual(eval_res['topic_diversity'], 0.5)
        self.assertLessEqual(eval_res['mean_umass_coherence'], 0.0)
        print(f"Perplexity: {eval_res['perplexity']}, Coherence: {eval_res['mean_umass_coherence']}, Diversity: {eval_res['topic_diversity']}")

        print("\n--- 5. Testing Visualizations ---")
        fig_bar = plot_topic_distribution(doc_topic_df)
        self.assertIsNotNone(fig_bar)

        fig_pie = plot_topic_pie(doc_topic_df)
        self.assertIsNotNone(fig_pie)

        fig_words = plot_topic_top_words(modeler.topic_summaries[0], top_n=8)
        self.assertIsNotNone(fig_words)

        fig_grid = plot_all_topics_grid(modeler.topic_summaries, top_n=5)
        self.assertIsNotNone(fig_grid)

        fig_heat = plot_document_topic_heatmap(doc_topic_df, n_topics=3, max_docs=10)
        self.assertIsNotNone(fig_heat)

        fig_pca = plot_document_pca_projection(modeler.doc_topic_dist, doc_topic_df)
        self.assertIsNotNone(fig_pca)

        fig_wc = generate_topic_wordcloud(modeler.topic_summaries[0]['word_weight_dict'])
        self.assertIsNotNone(fig_wc)

        fig_corpus_wc = generate_overall_wordcloud(cleaned_df['clean_text'].tolist())
        self.assertIsNotNone(fig_corpus_wc)
        print("All visualizations (Plotly & WordCloud) created successfully!")

        print("\n--- 6. Testing Live Inference on Unseen Text ---")
        unseen_text = "The courier delivered the package late with torn packaging."
        prediction = modeler.transform_new_text(unseen_text)
        self.assertIn('dominant_topic_id', prediction)
        self.assertIn('dominant_probability', prediction)
        self.assertGreater(prediction['dominant_probability'], 0.3)
        print(f"Live prediction on unseen text: Topic {prediction['dominant_topic_id']} with probability {prediction['dominant_probability']}")

        print("\n--- 7. Testing Export Reports & Artifacts ---")
        md_report = generate_analysis_report_markdown(summary, prep_stats, modeler, eval_res)
        self.assertIn("TopicLens AI", md_report)
        html_report = generate_analysis_report_html(md_report)
        self.assertIn("<html", html_report)

        csv_docs = df_to_csv_download(doc_topic_df)
        self.assertGreater(len(csv_docs), 100)

        csv_kw = df_to_csv_download(modeler.get_topic_keywords_df())
        self.assertGreater(len(csv_kw), 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "e2e_model.joblib"
            modeler.save(save_path)
            self.assertTrue(save_path.exists())

            reloaded_modeler = TopicModeler.load(save_path)
            self.assertTrue(reloaded_modeler.is_fitted)
            reloaded_pred = reloaded_modeler.transform_new_text(unseen_text)
            self.assertEqual(reloaded_pred['dominant_topic_id'], prediction['dominant_topic_id'])

        print("--- Complete End-to-End Test Passed with 100% Success! ---")


if __name__ == '__main__':
    unittest.main()
