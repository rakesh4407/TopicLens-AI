"""
TopicLens AI - Package Initialization
"""

from src.data_loader import load_csv_data, load_txt_files, load_sample_dataset, get_dataset_summary
from src.preprocessing import TextPreprocessor, preprocess_corpus, get_before_after_samples
from src.topic_model import TopicModeler
from src.evaluation import evaluate_model_comprehensively, compute_umass_coherence
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
from src.utils import generate_analysis_report_markdown, generate_analysis_report_html, df_to_csv_download

__all__ = [
    "load_csv_data",
    "load_txt_files",
    "load_sample_dataset",
    "get_dataset_summary",
    "TextPreprocessor",
    "preprocess_corpus",
    "get_before_after_samples",
    "TopicModeler",
    "evaluate_model_comprehensively",
    "compute_umass_coherence",
    "plot_topic_distribution",
    "plot_topic_pie",
    "plot_topic_top_words",
    "plot_all_topics_grid",
    "plot_document_topic_heatmap",
    "plot_document_pca_projection",
    "generate_topic_wordcloud",
    "generate_overall_wordcloud",
    "generate_analysis_report_markdown",
    "generate_analysis_report_html",
    "df_to_csv_download",
]
