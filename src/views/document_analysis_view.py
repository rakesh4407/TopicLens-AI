"""
src/views/document_analysis_view.py
Document-Level Topic Assignment, Probability Distributions, and Live Inference View.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.session_manager import has_trained_model
from src.inference import TopicInferencer

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_document_analysis_view():
    st.title("📄 Document Analysis & Topic Inference")
    st.write(
        "Examine topic probability assignments across individual documents, "
        "inspect document-topic distributions, and test the model with live custom text."
    )

    if not has_trained_model():
        st.warning("⚠️ No trained LDA model found. Please train a model in **Topic Modelling** first.")
        return

    tm = st.session_state['topic_modeler']
    doc_df = st.session_state.get('doc_topic_df')

    if doc_df is None:
        st.warning("Document topic assignments are not available.")
        return

    tabs = st.tabs([
        "📋 Document-Topic Explorer",
        "🎯 Document Probability Inspector",
        "🧪 Live Inference on Custom Text"
    ])

    # --- TAB 1: DOCUMENT-TOPIC EXPLORER ---
    with tabs[0]:
        st.subheader("📑 Corpus Document-Topic Assignments")

        # Filters
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            all_topics = ["All Topics"] + list(doc_df['Dominant_Topic_Label'].unique())
            selected_filter_topic = st.selectbox("Filter by Dominant Topic:", options=all_topics)
        with f_col2:
            search_query = st.text_input("Search documents by keyword:", value="", placeholder="e.g., telescope, network, screen")

        filtered_df = doc_df.copy()
        if selected_filter_topic != "All Topics":
            filtered_df = filtered_df[filtered_df['Dominant_Topic_Label'] == selected_filter_topic]

        text_col = 'clean_text' if 'clean_text' in filtered_df.columns else filtered_df.columns[0]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df[text_col].str.contains(search_query.strip(), case=False, na=False)]

        st.caption(f"Showing **{len(filtered_df):,}** of **{len(doc_df):,}** documents.")

        # Display table
        display_cols = ['Dominant_Topic_ID', 'Dominant_Topic_Label', 'Dominant_Topic_Probability', text_col]
        # Include raw text column if available and different
        raw_col = st.session_state.get('text_col')
        if raw_col and raw_col in filtered_df.columns and raw_col != text_col:
            display_cols = ['Dominant_Topic_ID', 'Dominant_Topic_Label', 'Dominant_Topic_Probability', raw_col, text_col]

        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[available_cols], use_container_width=True, height=400)

    # --- TAB 2: SINGLE DOCUMENT INSPECTOR ---
    with tabs[1]:
        st.subheader("🔬 Single Document Probability Breakdown")

        doc_idx = st.number_input(
            "Select Document Index to Inspect:",
            min_value=0,
            max_value=len(doc_df) - 1,
            value=0,
            step=1
        )

        selected_row = doc_df.iloc[doc_idx]
        text_field = 'clean_text' if 'clean_text' in doc_df.columns else doc_df.columns[0]
        raw_text_field = st.session_state.get('text_col')

        st.markdown(f"**Document #{doc_idx + 1}**")
        if raw_text_field and raw_text_field in selected_row:
            st.markdown(f"**Original Text:** *{selected_row[raw_text_field]}*")
        st.markdown(f"**Cleaned Tokens:** `{selected_row[text_field]}`")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Dominant Topic", selected_row.get('Dominant_Topic_Label', 'N/A'))
        with c2:
            st.metric("Topic Probability", f"{selected_row.get('Dominant_Topic_Probability', 0.0):.2%}")

        # Probability distribution chart
        prob_cols = [f"Topic_{i + 1}_Prob" for i in range(tm.n_topics) if f"Topic_{i + 1}_Prob" in selected_row]
        topic_labels = [s['auto_label'] for s in tm.topic_summaries]
        topic_probs = [selected_row[col] for col in prob_cols]

        prob_df = pd.DataFrame({
            'Topic': topic_labels,
            'Probability': topic_probs
        })

        if HAS_PLOTLY:
            fig_doc = px.bar(
                prob_df,
                x='Probability',
                y='Topic',
                orientation='h',
                text='Probability',
                color='Topic',
                title=f"<b>Topic Probability Distribution for Document #{doc_idx + 1}</b>",
                template="plotly_white"
            )
            fig_doc.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_doc.update_layout(height=350, showlegend=False, xaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig_doc, use_container_width=True)

    # --- TAB 3: LIVE INFERENCE ON UNSEEN TEXT ---
    with tabs[2]:
        st.subheader("🧪 Live Topic Prediction on Unseen Text")
        st.write(
            "Enter any arbitrary sentence or paragraph to classify its thematic topic "
            "using the trained LDA vectorizer and model."
        )

        user_input_text = st.text_area(
            "Enter custom text to analyze:",
            value="Astronomers observed an unprecedented supernova explosion in a distant spiral galaxy using orbital space telescopes.",
            height=120
        )

        if st.button("🔮 Predict Topic", type="primary", key="btn_predict_custom"):
            if not user_input_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                try:
                    inferencer = TopicInferencer(tm)
                    prediction = inferencer.predict(user_input_text)

                    st.markdown("---")
                    st.subheader("Prediction Result")

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric("Predicted Dominant Topic", prediction['dominant_topic_label'])
                    with r2:
                        st.metric("Probability", f"{prediction['dominant_probability']:.2%}")
                    with r3:
                        st.metric("Confidence Level", prediction['confidence_level'])
                    with r4:
                        st.metric("Matched Terms", prediction.get('matched_terms', 0))

                    if prediction.get('is_oov'):
                        st.warning(
                            "⚠️ **Out-of-Vocabulary Notice:** None of the terms in the input text matched "
                            "the model vocabulary. The assigned probabilities reflect the Dirichlet prior distribution."
                        )
                    else:
                        st.caption(f"**Preprocessed Tokens Evaluated:** `{prediction.get('cleaned_text', '')}`")

                    # Distribution chart
                    pred_dist_df = pd.DataFrame([
                        {'Topic': topic_name, 'Probability': prob}
                        for topic_name, prob in prediction['distribution'].items()
                    ])

                    if HAS_PLOTLY:
                        fig_pred = px.bar(
                            pred_dist_df,
                            x='Probability',
                            y='Topic',
                            orientation='h',
                            text='Probability',
                            color='Topic',
                            title="<b>Predicted Topic Probabilities for Unseen Input</b>",
                            template="plotly_white"
                        )
                        fig_pred.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                        fig_pred.update_layout(height=350, showlegend=False, xaxis=dict(range=[0, 1.05]))
                        st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"Inference error: {e}")
