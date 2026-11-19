"""
src/export_manager.py
Centralized export utilities for TopicLens AI.
Handles CSV, Joblib model, Markdown report, and HTML report exports.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional
import joblib
import pandas as pd


def export_document_topics_csv(doc_topic_df: pd.DataFrame) -> bytes:
    """
    Export document-topic assignment DataFrame to UTF-8 CSV bytes.
    """
    return doc_topic_df.to_csv(index=False).encode("utf-8")


def export_topic_keywords_csv(topic_modeler) -> bytes:
    """
    Export topic keyword table to UTF-8 CSV bytes.
    """
    df = topic_modeler.get_topic_keywords_df()
    return df.to_csv(index=False).encode("utf-8")


def export_cleaned_dataset_csv(cleaned_df: pd.DataFrame) -> bytes:
    """
    Export the cleaned/preprocessed dataset to UTF-8 CSV bytes.
    """
    return cleaned_df.to_csv(index=False).encode("utf-8")


def export_model_joblib(topic_modeler) -> bytes:
    """
    Serialize the trained TopicModeler to Joblib bytes (BytesIO).
    Returns raw bytes suitable for st.download_button.
    """
    if not topic_modeler.is_fitted:
        raise ValueError("Cannot export an unfitted model.")

    bundle = {
        "n_topics": topic_modeler.n_topics,
        "max_iter": topic_modeler.max_iter,
        "learning_method": topic_modeler.learning_method,
        "learning_decay": getattr(topic_modeler, "learning_decay", 0.7),
        "doc_topic_prior": getattr(topic_modeler, "doc_topic_prior", None),
        "topic_word_prior": getattr(topic_modeler, "topic_word_prior", None),
        "min_df": topic_modeler.min_df,
        "max_df": topic_modeler.max_df,
        "max_features": topic_modeler.max_features,
        "random_state": topic_modeler.random_state,
        "top_words_count": topic_modeler.top_words_count,
        "vectorizer": topic_modeler.vectorizer,
        "tfidf_vectorizer": getattr(topic_modeler, "tfidf_vectorizer", None),
        "lda_model": topic_modeler.lda_model,
        "feature_names": topic_modeler.feature_names,
        "topic_summaries": topic_modeler.topic_summaries,
        "training_time_sec": topic_modeler.training_time_sec,
        "timestamp": datetime.now().isoformat(),
    }
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    return buffer.read()


def export_report_markdown(
    dataset_summary: Dict[str, Any],
    preprocessing_summary: Dict[str, Any],
    topic_modeler,
    evaluation_metrics: Dict[str, Any],
) -> str:
    """
    Generate a detailed Markdown analysis report string.
    Delegates to utils.generate_analysis_report_markdown for content.
    """
    from src.utils import generate_analysis_report_markdown
    return generate_analysis_report_markdown(
        dataset_summary=dataset_summary,
        preprocessing_summary=preprocessing_summary,
        topic_modeler=topic_modeler,
        evaluation_metrics=evaluation_metrics,
    )


def export_report_html(markdown_str: str) -> str:
    """
    Convert Markdown report string to a standalone HTML document.
    """
    from src.utils import generate_analysis_report_html
    return generate_analysis_report_html(markdown_str)
