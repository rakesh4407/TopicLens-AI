"""
src/views/topic_exploration_view.py
Topic Exploration View: Term Weight Visualizations, Topic Word Clouds,
Human-in-the-Loop Label Customization, and Representative Documents.
"""

import streamlit as st
import pandas as pd
from src.session_manager import has_trained_model, update_custom_label
from src.visualization import (
    plot_topic_top_words,
    plot_all_topics_grid,
    generate_topic_wordcloud
)


def render_topic_exploration_view():
    st.title("🔍 Topic Exploration & Interpretation")
    st.write(
        "Inspect keyword distributions, visualize topic word clouds, validate semantic meanings, "
        "and assign human-verified domain labels to discovered topics."
    )

    if not has_trained_model():
        st.warning("⚠️ No trained LDA model found. Please train a model in **Topic Modelling** first.")
        return

    tm = st.session_state['topic_modeler']
    doc_df = st.session_state.get('doc_topic_df')
    summaries = tm.topic_summaries

    tabs = st.tabs(["🔎 Single Topic Deep Dive", "📊 All Topics Comparison Grid"])

    # --- TAB 1: SINGLE TOPIC DEEP DIVE ---
    with tabs[0]:
        topic_options = [s['topic_name'] for s in summaries]
        selected_topic_name = st.selectbox("Select a Topic to Explore:", options=topic_options, index=0)
        
        # Find selected summary
        selected_summary = next(s for s in summaries if s['topic_name'] == selected_topic_name)
        topic_id = selected_summary['topic_number']

        st.markdown(f"### Exploring {selected_topic_name}")

        # Human-in-the-loop Label Customization
        with st.expander("🏷️ Validate & Customize Topic Label (Human-in-the-Loop)", expanded=True):
            st.caption(
                "Automated labels are statistical heuristics. As the human expert, "
                "provide a verified semantic title for this topic based on its keywords and documents."
            )
            col_l1, col_l2 = st.columns([3, 1])
            with col_l1:
                current_custom = st.session_state.get('custom_labels', {}).get(
                    topic_id,
                    selected_summary['auto_label'].replace(f"Topic {topic_id}: ", "")
                )
                new_label_input = st.text_input(
                    f"Verified Label for {selected_topic_name}:",
                    value=current_custom
                )
            with col_l2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Save Label", key=f"btn_save_lbl_{topic_id}", type="primary"):
                    update_custom_label(topic_id, new_label_input.strip())
                    st.success("Label updated successfully across all views!")
                    st.rerun()

        # Visualizations row: Bar chart + Word Cloud
        col_vis1, col_vis2 = st.columns([1, 1])
        with col_vis1:
            st.subheader("📊 Top Keyword Weights")
            fig_bar = plot_topic_top_words(selected_summary, top_n=tm.top_words_count)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_vis2:
            st.subheader("☁️ Topic Word Cloud")
            fig_wc = generate_topic_wordcloud(selected_summary['word_weight_dict'])
            if fig_wc:
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.info("Word Cloud requires `wordcloud` library.")

        # Keywords Table
        st.subheader("📋 Top Terms & Statistical Probabilities")
        terms_df = pd.DataFrame({
            'Rank': range(1, len(selected_summary['top_words']) + 1),
            'Keyword': selected_summary['top_words'],
            'Importance Weight': selected_summary['top_weights'],
            'Word Probability': selected_summary['top_probabilities']
        })
        st.dataframe(terms_df, use_container_width=True, hide_index=True)

        # Representative Documents for this Topic
        if doc_df is not None:
            st.subheader(f"📄 Exemplar Documents for {selected_topic_name}")
            st.write(
                f"Documents with the highest posterior probability for **{selected_summary['auto_label']}**:"
            )
            prob_col = f"Topic_{topic_id}_Prob"
            if prob_col in doc_df.columns:
                top_exemplars = doc_df.sort_values(by=prob_col, ascending=False).head(5)
                
                text_field = 'clean_text' if 'clean_text' in doc_df.columns else doc_df.columns[0]
                for rank, (_, row) in enumerate(top_exemplars.iterrows(), start=1):
                    with st.container():
                        st.markdown(
                            f"**#{rank} | Match Confidence: `{row[prob_col]:.2%}`** "
                            f"(Dominant Topic: `{row.get('Dominant_Topic_Label', 'N/A')}`)"
                        )
                        st.markdown(f"> *{str(row[text_field])[:350]}...*")
                        st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)

    # --- TAB 2: ALL TOPICS COMPARISON GRID ---
    with tabs[1]:
        st.subheader("🌐 Corpus-Wide Topic Term Comparison")
        fig_grid = plot_all_topics_grid(summaries, top_n=8)
        if fig_grid:
            st.plotly_chart(fig_grid, use_container_width=True)
        else:
            st.info("Comparative grid visualization ready.")
