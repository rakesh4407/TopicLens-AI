"""
src/session_manager.py
Manages Streamlit session state persistence across all pages and views.
"""

from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd
from src.config import APP_NAME


DEFAULT_STATE = {
    'raw_df': None,
    'text_col': None,
    'dataset_name': "None",
    'dataset_summary': None,
    'cleaned_df': None,
    'preprocessing_stats': None,
    'preprocessor': None,
    'preprocessor_config': None,
    'topic_modeler': None,
    'doc_topic_df': None,
    'evaluation_results': None,
    'custom_labels': {},
    'current_page': "Home Dashboard"
}


def init_session_state() -> None:
    """
    Initialize all default session state keys if not already set.
    """
    for key, default_val in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def reset_pipeline() -> None:
    """
    Reset all pipeline data and trained models.
    """
    for key, default_val in DEFAULT_STATE.items():
        if key != 'current_page':
            st.session_state[key] = default_val


def has_dataset() -> bool:
    """Check if a dataset is currently loaded."""
    return st.session_state.get('raw_df') is not None and st.session_state.get('text_col') is not None


def has_cleaned_data() -> bool:
    """Check if preprocessing has been executed."""
    return st.session_state.get('cleaned_df') is not None


def has_trained_model() -> bool:
    """Check if an LDA model is trained and available."""
    tm = st.session_state.get('topic_modeler')
    return tm is not None and getattr(tm, 'is_fitted', False)


def update_custom_label(topic_id: int, new_label: str) -> None:
    """
    Update a human-edited label for a discovered topic in session state.
    """
    if 'custom_labels' not in st.session_state:
        st.session_state['custom_labels'] = {}
    st.session_state['custom_labels'][topic_id] = new_label

    # Also update in topic_modeler and doc_topic_df if available
    tm = st.session_state.get('topic_modeler')
    if tm and tm.topic_summaries:
        for summary in tm.topic_summaries:
            if summary['topic_number'] == topic_id or summary['topic_id'] == topic_id:
                summary['auto_label'] = f"Topic {topic_id}: {new_label}"

    doc_df = st.session_state.get('doc_topic_df')
    if doc_df is not None and 'Dominant_Topic_ID' in doc_df.columns:
        mask = (doc_df['Dominant_Topic_ID'] == topic_id)
        doc_df.loc[mask, 'Dominant_Topic_Label'] = f"Topic {topic_id}: {new_label}"
