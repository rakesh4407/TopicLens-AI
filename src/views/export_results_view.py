"""
src/views/export_results_view.py
Export Results view for TopicLens AI.
Provides CSV, Joblib model, and report downloads.
"""

import streamlit as st
from src.session_manager import has_dataset, has_cleaned_data, has_trained_model
from src.export_manager import (
    export_document_topics_csv,
    export_topic_keywords_csv,
    export_cleaned_dataset_csv,
    export_model_joblib,
    export_report_markdown,
    export_report_html,
)


def render_export_results_view():
    st.title("Export Results")
    st.write(
        "Download your analysis outputs: document-topic assignments, topic keywords, "
        "the trained LDA model, cleaned dataset, and a full analysis report."
    )

    if not has_trained_model():
        st.warning("No trained model found. Please complete Dataset → Preprocessing → Topic Modelling first.")
        return

    tm = st.session_state["topic_modeler"]
    doc_topic_df = st.session_state.get("doc_topic_df")
    cleaned_df = st.session_state.get("cleaned_df")
    dataset_summary = st.session_state.get("dataset_summary", {})
    preprocessing_stats = st.session_state.get("preprocessing_stats", {})
    evaluation_results = st.session_state.get("evaluation_results", {})

    st.subheader("Available Exports")

    col1, col2 = st.columns(2)

    # 1. Document-Topic CSV
    with col1:
        st.markdown("#### Document-Topic Assignments")
        st.caption(f"One row per document. Includes topic probabilities and dominant topic label. ({doc_topic_df.shape[0] if doc_topic_df is not None else 0} rows)")
        if doc_topic_df is not None:
            csv_bytes = export_document_topics_csv(doc_topic_df)
            st.download_button(
                label="Download Document-Topic CSV",
                data=csv_bytes,
                file_name="topiclens_document_topic_assignments.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Train the LDA model to generate document assignments.")

    # 2. Topic Keywords CSV
    with col2:
        st.markdown("#### Topic Keywords Table")
        st.caption(f"All topics with ranked keywords and weights. ({tm.n_topics} topics)")
        kw_bytes = export_topic_keywords_csv(tm)
        st.download_button(
            label="Download Topic Keywords CSV",
            data=kw_bytes,
            file_name="topiclens_topic_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    # 3. Cleaned Dataset CSV
    with col3:
        st.markdown("#### Preprocessed Dataset")
        st.caption("Cleaned text corpus after all preprocessing steps.")
        if cleaned_df is not None:
            clean_bytes = export_cleaned_dataset_csv(cleaned_df)
            st.download_button(
                label="Download Cleaned Dataset CSV",
                data=clean_bytes,
                file_name="topiclens_cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Cleaned dataset not available.")

    # 4. Trained Model (Joblib)
    with col4:
        st.markdown("#### Trained LDA Model")
        st.caption(f"Trained model bundle (K={tm.n_topics} topics, {len(tm.feature_names):,} terms). Reload with `TopicModeler.load()`.")
        try:
            model_bytes = export_model_joblib(tm)
            st.download_button(
                label="Download Model (.joblib)",
                data=model_bytes,
                file_name="topiclens_lda_model.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Model export error: {e}")

    st.markdown("---")
    st.subheader("Analysis Report")

    # 5 & 6. Report downloads (Markdown + HTML)
    if dataset_summary and evaluation_results:
        md_report = export_report_markdown(
            dataset_summary=dataset_summary,
            preprocessing_summary=preprocessing_stats,
            topic_modeler=tm,
            evaluation_metrics=evaluation_results
        )
        html_report = export_report_html(md_report)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.download_button(
                label="Download Report (Markdown)",
                data=md_report.encode("utf-8"),
                file_name="topiclens_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with rc2:
            st.download_button(
                label="Download Report (HTML)",
                data=html_report.encode("utf-8"),
                file_name="topiclens_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )

        # Preview
        with st.expander("Preview Analysis Report", expanded=False):
            st.markdown(md_report)
    else:
        st.info("Complete the full pipeline to generate the analysis report.")

    st.markdown("---")
    st.info(
        "**Note:** The downloaded Joblib model can be reloaded for offline inference using: "
        "`TopicModeler.load('path/to/model.joblib')`"
    )
