"""
src/views/home_view.py
Home Dashboard for TopicLens AI.
"""

import streamlit as st
import pandas as pd
from src.session_manager import has_dataset, has_cleaned_data, has_trained_model
from src.data_loader import load_sample_dataset, get_dataset_summary
from src.visualization import plot_topic_distribution, plot_topic_pie
from src.config import APP_NAME, APP_TAGLINE, SAMPLE_DATASETS


def render_home_view():
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); padding: 26px 30px; border-radius: 12px; color: white; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin-bottom: 6px; font-size: 2.2rem; font-weight: 700;">🔍 {APP_NAME}</h1>
        <p style="font-size: 1.15rem; opacity: 0.95; margin-bottom: 10px; font-weight: 400;">
            {APP_TAGLINE}
        </p>
        <span style="background: rgba(255,255,255,0.22); padding: 5px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500;">
            Project #9: Text Topic Modelling • BCA AI &amp; Data Science Capstone
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Key KPI metrics row
    tm = st.session_state.get('topic_modeler')
    raw_df = st.session_state.get('raw_df')
    clean_df = st.session_state.get('cleaned_df')
    doc_df = st.session_state.get('doc_topic_df')

    total_docs = len(raw_df) if raw_df is not None else 0
    k_topics = tm.n_topics if (tm and tm.is_fitted) else 0
    vocab_size = len(tm.feature_names) if (tm and tm.feature_names) else (
        st.session_state.get('preprocessing_stats', {}).get('clean_vocab_size', 0) if st.session_state.get('preprocessing_stats') else 0
    )
    avg_len = (
        st.session_state.get('preprocessing_stats', {}).get('avg_clean_doc_length', 0.0)
        if st.session_state.get('preprocessing_stats')
        else (st.session_state.get('dataset_summary', {}).get('avg_word_count', 0.0) if st.session_state.get('dataset_summary') else 0.0)
    )
    status_label = "✅ Model Trained" if (tm and tm.is_fitted) else ("🧹 Data Preprocessed" if has_cleaned_data() else ("📂 Dataset Loaded" if has_dataset() else "⏳ Awaiting Dataset"))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Documents", value=f"{total_docs:,}")
    with col2:
        st.metric(label="Discovered Topics (K)", value=k_topics if k_topics > 0 else "—")
    with col3:
        st.metric(label="Vocabulary Size", value=f"{vocab_size:,}" if vocab_size > 0 else "—")
    with col4:
        st.metric(label="Avg Doc Length", value=f"{avg_len:.1f} w" if avg_len > 0 else "—")
    with col5:
        st.metric(label="Pipeline State", value=status_label)

    st.markdown("---")

    # If model is already trained: show dynamic interactive dashboard
    if has_trained_model() and doc_df is not None:
        st.subheader("📊 Active Model Topic Distribution Overview")
        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            fig_bar = plot_topic_distribution(doc_df)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        with chart_col2:
            fig_pie = plot_topic_pie(doc_df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🏷️ Discovered Topics & Keywords Summary")
        summary_rows = []
        for s in tm.topic_summaries:
            summary_rows.append({
                "Topic ID": s['topic_number'],
                "Suggested Label": s['auto_label'],
                "Top Keywords": ", ".join(s['top_words'][:7]),
                "Top Keyword Weight": f"{s['top_weights'][0]:.3f}"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.info("💡 Navigate using the sidebar to explore in-depth **Topic Details**, inspect **Documents**, run **Visual Analytics**, or check **Evaluation Metrics**.")

    else:
        # Guided Quick Start Section
        st.subheader("🚀 Quick Start Workflow")
        
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown("""
            #### 1. Dataset Ingestion
            Upload your custom CSV, multiple TXT files, or choose from 3 curated sample datasets (20 Newsgroups, Feedback, Tech News).
            """)
        with col_w2:
            st.markdown("""
            #### 2. Text Preprocessing
            Filter noise, URLs, HTML tags, punctuation, and digits. Tokenize, remove stopwords, and apply lemmatization.
            """)
        with col_w3:
            st.markdown("""
            #### 3. Topic Discovery (LDA)
            Train a Scikit-Learn LDA model over Document-Term counts, extract latent themes, and inspect semantic metrics.
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚡ 1-Click Demonstration Dataset")
        st.write("Want to test the full pipeline right away? Select and load a pre-configured sample dataset:")

        sample_options = {ds["label"]: ds for ds in SAMPLE_DATASETS}
        selected_label = st.selectbox(
            "Select a curated dataset for instant testing:",
            options=list(sample_options.keys()),
            help="Choose any of the pre-configured datasets."
        )
        chosen_ds = sample_options[selected_label]

        if chosen_ds.get("is_synthetic"):
            st.caption("ℹ️ *Note: This dataset is synthetic, generated for algorithmic demonstration.*")

        if st.button("📥 Load Sample Dataset & Initialize Pipeline", type="primary"):
            with st.spinner("Loading sample dataset..."):
                df, text_col = load_sample_dataset(chosen_ds["id"])
                summary = get_dataset_summary(df, text_col)

                st.session_state['raw_df'] = df
                st.session_state['text_col'] = text_col
                st.session_state['dataset_name'] = chosen_ds["label"]
                st.session_state['dataset_summary'] = summary
                
                # Reset downstream states
                st.session_state['cleaned_df'] = None
                st.session_state['topic_modeler'] = None
                st.session_state['doc_topic_df'] = None
                st.session_state['evaluation_results'] = None

            st.success(f"Successfully loaded '{chosen_ds['label']}' with {len(df)} documents! Proceed to **Data Preprocessing** or **Topic Modelling**.")
            st.rerun()
