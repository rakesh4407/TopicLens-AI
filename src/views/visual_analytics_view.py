"""
src/views/visual_analytics_view.py
Visual Analytics View: Heatmaps, Topic Macro Proportions,
and Overall Corpus Word Clouds.
"""

import streamlit as st
from src.session_manager import has_trained_model
from src.visualization import (
    plot_document_topic_heatmap,
    plot_topic_distribution,
    plot_topic_pie,
    generate_overall_wordcloud,
)


def render_visual_analytics_view():
    st.title("📊 Visual Analytics & Topic Distributions")
    st.write(
        "Explore document-topic probability heatmaps, macro corpus distributions, "
        "and overall corpus word clouds."
    )

    if not has_trained_model():
        st.warning("⚠️ No trained LDA model found. Please train a model in **Topic Modelling** first.")
        return

    tm = st.session_state['topic_modeler']
    doc_df = st.session_state.get('doc_topic_df')
    cleaned_df = st.session_state.get('cleaned_df')

    tabs = st.tabs([
        "🔥 Document-Topic Heatmap",
        "📈 Topic Prevalence & Proportions",
        "☁️ Overall Corpus Word Cloud"
    ])

    # --- TAB 1: DOCUMENT-TOPIC HEATMAP ---
    with tabs[0]:
        st.subheader("🔥 Document-Topic Probability Heatmap")
        st.write("Inspect how topic probabilities are distributed across individual documents.")

        max_docs = st.slider(
            "Number of Documents to Display:",
            min_value=10,
            max_value=min(60, len(doc_df)),
            value=min(25, len(doc_df))
        )
        fig_heat = plot_document_topic_heatmap(doc_df, n_topics=tm.n_topics, max_docs=max_docs)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)

    # --- TAB 2: TOPIC PREVALENCE ---
    with tabs[1]:
        st.subheader("📈 Macro Topic Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = plot_topic_distribution(doc_df)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            fig_pie = plot_topic_pie(doc_df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 3: CORPUS WORD CLOUD ---
    with tabs[2]:
        st.subheader("☁️ Overall Corpus Term Frequency Cloud")
        st.write("Visual representation of highest frequency cleaned terms across all documents.")
        if cleaned_df is not None and 'clean_text' in cleaned_df.columns:
            fig_corpus_wc = generate_overall_wordcloud(cleaned_df['clean_text'].tolist())
            if fig_corpus_wc:
                st.pyplot(fig_corpus_wc, use_container_width=True)
            else:
                st.info("Word Cloud requires `wordcloud` library.")
