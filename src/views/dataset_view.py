"""
src/views/dataset_view.py
Dataset ingestion, validation, and exploratory analysis view for TopicLens AI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.session_manager import has_dataset
from src.data_loader import load_csv_data, load_txt_files, load_sample_dataset, get_dataset_summary
from src.config import SAMPLE_DATASETS


def render_dataset_view():
    st.title("Dataset Upload & Management")
    st.write(
        "Load your text dataset from a CSV file, multiple TXT files, "
        "or choose one of the curated sample datasets for instant demonstration."
    )

    tab_upload, tab_txt, tab_sample = st.tabs(["CSV Upload", "TXT Files Upload", "Sample Datasets"])

    # ------------------------------------------------------------------ TAB 1: CSV Upload
    with tab_upload:
        st.subheader("Upload a CSV File")
        uploaded_csv = st.file_uploader(
            "Select a CSV file containing a text column",
            type=["csv"],
            key="csv_uploader"
        )
        if uploaded_csv:
            try:
                df, text_cols, suggested_col = load_csv_data(uploaded_csv)
                st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

                selected_col = st.selectbox(
                    "Select the text column to analyse:",
                    options=text_cols,
                    index=0,
                    help="TopicLens AI detected these columns as likely text columns."
                )

                st.dataframe(df[[selected_col]].head(10), use_container_width=True)

                if st.button("Confirm & Load CSV Dataset", type="primary"):
                    summary = get_dataset_summary(df, selected_col)
                    st.session_state["raw_df"] = df
                    st.session_state["text_col"] = selected_col
                    st.session_state["dataset_name"] = uploaded_csv.name
                    st.session_state["dataset_summary"] = summary
                    st.session_state["cleaned_df"] = None
                    st.session_state["topic_modeler"] = None
                    st.session_state["doc_topic_df"] = None
                    st.session_state["evaluation_results"] = None
                    st.success(f"Dataset confirmed: {len(df):,} documents from column `{selected_col}`.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ------------------------------------------------------------------ TAB 2: TXT Files
    with tab_txt:
        st.subheader("Upload Multiple TXT Files")
        st.info("Each TXT file is treated as one document. Upload at least 5 files for meaningful topic discovery.")
        uploaded_txts = st.file_uploader(
            "Select one or more TXT files",
            type=["txt"],
            accept_multiple_files=True,
            key="txt_uploader"
        )
        if uploaded_txts:
            try:
                df = load_txt_files(uploaded_txts)
                st.success(f"Loaded {len(df):,} documents from {len(uploaded_txts)} TXT files.")
                st.dataframe(df[["filename", "word_count", "text"]].head(10), use_container_width=True)

                if st.button("Confirm & Load TXT Documents", type="primary"):
                    summary = get_dataset_summary(df, "text")
                    st.session_state["raw_df"] = df
                    st.session_state["text_col"] = "text"
                    st.session_state["dataset_name"] = f"{len(uploaded_txts)} TXT files"
                    st.session_state["dataset_summary"] = summary
                    st.session_state["cleaned_df"] = None
                    st.session_state["topic_modeler"] = None
                    st.session_state["doc_topic_df"] = None
                    st.session_state["evaluation_results"] = None
                    st.success(f"Loaded {len(df):,} documents.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading TXT files: {e}")

    # ------------------------------------------------------------------ TAB 3: Sample Datasets
    with tab_sample:
        st.subheader("Curated Sample Datasets")
        st.write("Choose from pre-loaded datasets to explore the full TopicLens AI pipeline immediately.")

        sample_options = {ds["label"]: ds for ds in SAMPLE_DATASETS}
        selected_label = st.selectbox("Select a sample dataset:", list(sample_options.keys()))
        selected_ds = sample_options[selected_label]

        # Description
        st.markdown(f"**Description:** {selected_ds['description']}")

        # Synthetic warning
        if selected_ds["is_synthetic"]:
            st.warning(
                "**Synthetic Dataset Notice:** This dataset was algorithmically generated for demonstration "
                "and testing purposes. It does not represent real-world documents or user-generated content. "
                "Results are illustrative only."
            )

        if st.button("Load Selected Sample Dataset", type="primary"):
            with st.spinner(f"Loading {selected_label}..."):
                try:
                    df, text_col = load_sample_dataset(selected_ds["id"])
                    summary = get_dataset_summary(df, text_col)
                    st.session_state["raw_df"] = df
                    st.session_state["text_col"] = text_col
                    st.session_state["dataset_name"] = selected_label
                    st.session_state["dataset_summary"] = summary
                    st.session_state["cleaned_df"] = None
                    st.session_state["topic_modeler"] = None
                    st.session_state["doc_topic_df"] = None
                    st.session_state["evaluation_results"] = None
                    st.success(f"Loaded '{selected_label}' — {len(df):,} documents, text column: `{text_col}`.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {e}")

    # ------------------------------------------------------------------ Dataset Summary Section
    if has_dataset():
        raw_df = st.session_state["raw_df"]
        text_col = st.session_state["text_col"]
        summary = st.session_state.get("dataset_summary", {})
        dataset_name = st.session_state.get("dataset_name", "Dataset")

        st.markdown("---")
        st.subheader(f"Dataset Overview — {dataset_name}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Documents", f"{summary.get('total_documents', 0):,}")
        with c2:
            st.metric("Valid Documents", f"{summary.get('valid_documents', 0):,}")
        with c3:
            st.metric("Missing Values", summary.get("missing_values", 0))
        with c4:
            st.metric("Duplicates", summary.get("duplicate_documents", 0))

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Avg Words / Doc", f"{summary.get('avg_word_count', 0):.1f}")
        with c6:
            st.metric("Min Words", summary.get("min_word_count", 0))
        with c7:
            st.metric("Max Words", summary.get("max_word_count", 0))

        # Document length histogram
        st.subheader("Document Length Distribution")
        word_counts = raw_df[text_col].dropna().astype(str).apply(lambda x: len(x.split()))
        fig = px.histogram(
            x=word_counts,
            nbins=30,
            labels={"x": "Word Count per Document", "y": "Number of Documents"},
            title="Word Count Distribution Across Documents",
            color_discrete_sequence=["#2563EB"],
            template="plotly_white"
        )
        fig.update_layout(showlegend=False, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        # Data preview
        st.subheader("Data Preview")
        preview_cols = [text_col] + [c for c in raw_df.columns if c != text_col][:4]
        st.dataframe(raw_df[preview_cols].head(15), use_container_width=True)

        # Missing value report
        if summary.get("missing_values", 0) > 0 or summary.get("duplicate_documents", 0) > 0:
            st.subheader("Data Quality Report")
            if summary.get("missing_values", 0) > 0:
                st.warning(
                    f"Found **{summary['missing_values']} missing values** in column `{text_col}`. "
                    "These will be dropped during preprocessing."
                )
            if summary.get("duplicate_documents", 0) > 0:
                st.info(
                    f"Found **{summary['duplicate_documents']} duplicate documents**. "
                    "Duplicates can be removed in the Preprocessing step."
                )
        else:
            st.success("No missing values or duplicate documents detected.")
