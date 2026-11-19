"""
src/views/preprocessing_view.py
Text Preprocessing Configuration, Pipeline Execution, and Before/After Diagnostics View.
"""

import streamlit as st
import pandas as pd
from src.session_manager import has_dataset, has_cleaned_data
from src.preprocessing import (
    TextPreprocessor,
    preprocess_corpus,
    get_before_after_samples,
    DEFAULT_STOPWORDS
)


def render_preprocessing_view():
    st.title("🧹 Text Preprocessing Pipeline")
    st.write(
        "Transform noisy unstructured text into clean, normalized tokens. "
        "Customize pipeline cleaning stages and inspect vocabulary reduction."
    )

    if not has_dataset():
        st.warning("⚠️ No dataset loaded. Please navigate to the **Dataset Upload** page first.")
        return

    raw_df = st.session_state['raw_df']
    text_col = st.session_state['text_col']

    st.subheader("⚙️ Pipeline Configuration")

    with st.expander("🛠️ Cleaning Stages & Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            lowercase = st.checkbox("Convert to Lowercase", value=True, help="Standardizes casing across all terms.")
            remove_html = st.checkbox("Strip HTML / XML Tags", value=True, help="Removes markup tags like <p>, <br>.")
            remove_urls = st.checkbox("Remove URLs & Hyperlinks", value=True, help="Removes http://, https://, and www links.")
            remove_punct = st.checkbox("Remove Punctuation", value=True, help="Strips symbols, brackets, and delimiters.")
            remove_numbers = st.checkbox("Filter Digits & Numbers", value=True, help="Removes numeric tokens.")

        with col2:
            remove_stops = st.checkbox("Remove Stopwords", value=True, help="Removes non-informative high-frequency words.")
            lemmatize = st.checkbox("Apply Lemmatization", value=True, help="Reduces words to root lemmas using WordNet.")
            drop_empty = st.checkbox("Remove Empty Documents", value=True, help="Discards documents that become empty after cleaning.")
            drop_dups = st.checkbox("Deduplicate Documents", value=True, help="Removes identical cleaned documents to prevent bias.")
            min_len = st.slider("Minimum Token Length", min_value=2, max_value=6, value=3, help="Tokens shorter than this are discarded.")

        custom_stops_input = st.text_area(
            "Add Custom Domain Stopwords (comma-separated):",
            value="",
            placeholder="e.g., product, company, website, year, think, really"
        )

    custom_stopwords_list = [w.strip() for w in custom_stops_input.split(",") if w.strip()]

    # Execution button
    if st.button("🚀 Run Preprocessing Pipeline", type="primary", use_container_width=True):
        with st.spinner("Applying NLP text cleaning, tokenization, and lemmatization..."):
            preprocessor = TextPreprocessor(
                lowercase=lowercase,
                remove_html=remove_html,
                remove_urls=remove_urls,
                remove_punctuation=remove_punct,
                remove_numbers=remove_numbers,
                remove_stopwords=remove_stops,
                lemmatize=lemmatize,
                min_word_len=min_len,
                custom_stopwords=custom_stopwords_list
            )

            cleaned_df, stats = preprocess_corpus(
                df=raw_df,
                text_column=text_col,
                preprocessor=preprocessor,
                drop_empty=drop_empty,
                drop_duplicates=drop_dups
            )

            # Store in session state
            st.session_state['preprocessor'] = preprocessor
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['preprocessing_stats'] = stats
            st.session_state['preprocessor_config'] = preprocessor.to_dict()

            # Invalidate downstream model if data changed
            st.session_state['topic_modeler'] = None
            st.session_state['doc_topic_df'] = None
            st.session_state['evaluation_results'] = None

        st.success(f"Preprocessing completed! Cleaned {len(cleaned_df):,} documents.")

    st.markdown("---")

    # Display results if available
    if has_cleaned_data():
        cleaned_df = st.session_state['cleaned_df']
        stats = st.session_state['preprocessing_stats']

        st.subheader("📈 Preprocessing Quality & Reduction Diagnostics")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Documents Retained", f"{stats['final_document_count']:,}", delta=f"{stats['final_document_count'] - stats['original_document_count']}")
        with c2:
            st.metric("Vocabulary Reduction", f"{stats['vocab_reduction_percent']}%", delta=f"-{stats['raw_vocab_size'] - stats['clean_vocab_size']} terms", delta_color="inverse")
        with c3:
            st.metric("Clean Vocabulary Size", f"{stats['clean_vocab_size']:,}")
        with c4:
            st.metric("Avg Clean Word Count", f"{stats['avg_clean_doc_length']:.1f} words")

        # Before & After Inspection
        st.subheader("🔍 Before & After Document Comparison")
        st.write("Inspect how the cleaning rules transformed actual document text:")

        samples = get_before_after_samples(
            original_series=raw_df[text_col],
            cleaned_series=cleaned_df['clean_text'],
            n_samples=3
        )

        for i, s in enumerate(samples, start=1):
            with st.container():
                st.markdown(f"**Document #{i} (Index {s['doc_index']})**")
                col_orig, col_clean = st.columns(2)
                with col_orig:
                    st.caption("🔴 ORIGINAL RAW TEXT")
                    st.markdown(f"> *{s['original']}*")
                with col_clean:
                    st.caption("🟢 PREPROCESSED CLEAN TOKENS")
                    st.markdown(f"> `{s['cleaned']}`")
                st.markdown("<br>", unsafe_allow_html=True)

        # Preview cleaned dataset
        st.subheader("📋 Preprocessed Dataset Table")
        st.dataframe(
            cleaned_df[['clean_text', 'clean_word_count']].head(15),
            use_container_width=True
        )

        st.info("✅ Preprocessing is complete. Proceed to **Topic Modelling** to train the LDA model.")
