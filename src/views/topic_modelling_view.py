"""
src/views/topic_modelling_view.py
Topic Modelling Training & Configuration View using Scikit-Learn LDA and CountVectorizer.
"""

import streamlit as st
import pandas as pd
from src.session_manager import has_dataset, has_cleaned_data, has_trained_model
from src.preprocessing import TextPreprocessor, preprocess_corpus
from src.topic_model import TopicModeler
from src.evaluation import evaluate_model_comprehensively


def render_topic_modelling_view():
    st.title("⚙️ Topic Modelling (LDA)")
    st.write(
        "Uncover latent themes and probability distributions from the document collection "
        "using Latent Dirichlet Allocation (LDA) and a Document-Term Matrix (CountVectorizer)."
    )

    if not has_dataset():
        st.warning("⚠️ No dataset loaded. Please go to **Dataset Upload** first.")
        return

    # If data hasn't been cleaned yet, provide quick option
    if not has_cleaned_data():
        st.info("ℹ️ Your dataset is loaded but has not been preprocessed yet.")
        if st.button("⚡ Run Default Preprocessing & Proceed", type="primary"):
            with st.spinner("Preprocessing documents..."):
                preprocessor = TextPreprocessor()
                cleaned_df, stats = preprocess_corpus(
                    df=st.session_state['raw_df'],
                    text_column=st.session_state['text_col'],
                    preprocessor=preprocessor
                )
                st.session_state['preprocessor'] = preprocessor
                st.session_state['preprocessor_config'] = preprocessor.to_dict()
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['preprocessing_stats'] = stats
            st.success("Default preprocessing applied! You can now configure and train LDA.")
            st.rerun()
        return

    cleaned_df = st.session_state['cleaned_df']
    n_docs = len(cleaned_df)

    # Architectural Educational Note
    with st.expander("📚 Engineering Note: Why CountVectorizer for Standard LDA?", expanded=False):
        st.markdown("""
        **Latent Dirichlet Allocation (Blei et al., 2003)** is a hierarchical Bayesian generative model:
        - It assumes documents are mixtures of topics, and topics are discrete multinomial distributions over discrete word counts.
        - Consequently, LDA expects **integer word frequency counts** as input, which are constructed via **`CountVectorizer`** (Document-Term Matrix).
        - **Why not TF-IDF by default?** TF-IDF produces continuous non-integer normalized weights. Supplying continuous TF-IDF scores to standard LDA violates the multinomial likelihood model. TF-IDF can be used for non-negative matrix factorization (NMF) or auxiliary analysis, but CountVectorizer is the theoretically correct input for standard LDA.
        """)

    st.subheader("🎛️ LDA Hyperparameter Configuration")

    col1, col2 = st.columns(2)
    with col1:
        n_topics = st.slider(
            "Number of Topics (K)",
            min_value=2,
            max_value=min(15, max(3, n_docs // 2)),
            value=4,
            help="The number of latent thematic clusters to discover."
        )
        top_words_count = st.slider(
            "Top Words per Topic",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of representative keywords to display for each topic."
        )
        learning_method = st.selectbox(
            "Learning Algorithm",
            options=['batch', 'online'],
            index=0,
            help="'batch' updates variational bounds over the entire corpus (more stable for small/medium datasets); 'online' processes mini-batches."
        )
        max_iter = st.slider(
            "Maximum Iterations",
            min_value=10,
            max_value=80,
            value=25,
            help="Number of iterations for variational Bayes optimization."
        )

    with col2:
        min_df = st.number_input(
            "Minimum Document Frequency (min_df)",
            min_value=1,
            max_value=max(1, n_docs // 4),
            value=min(2, max(1, n_docs // 10)),
            help="Ignore words that appear in fewer than this number of documents."
        )
        max_df = st.slider(
            "Maximum Document Frequency (max_df)",
            min_value=0.50,
            max_value=1.00,
            value=0.95,
            step=0.05,
            help="Ignore terms that appear in more than this fraction of documents (corpus-wide stops)."
        )
        max_features = st.selectbox(
            "Maximum Vocabulary Size",
            options=[500, 1000, 1500, 2500, 5000, None],
            index=2,
            help="Upper bound on vocabulary features to prevent sparse noise."
        )
        random_seed = st.number_input(
            "Random State Seed",
            min_value=0,
            max_value=99999,
            value=42,
            help="Ensures reproducible Dirichlet parameter estimations."
        )

    with st.expander("⚙️ Advanced Hyperparameters (Dirichlet Priors & Decay)", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            learning_decay = st.slider(
                "Learning Decay",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Learning rate decay parameter (controls learning rate across iterations)."
            )
        with adv_col2:
            custom_alpha = st.checkbox("Custom Document-Topic Prior (Alpha)", value=False)
            doc_topic_prior = st.number_input(
                "Alpha Prior",
                min_value=0.01,
                max_value=5.0,
                value=round(1.0 / n_topics, 3),
                step=0.05,
                disabled=not custom_alpha,
                help="Dirichlet distribution hyperparameter for document-topic mixtures (lower = sparser topic mixtures)."
            ) if custom_alpha else None
        with adv_col3:
            custom_beta = st.checkbox("Custom Topic-Word Prior (Beta / Eta)", value=False)
            topic_word_prior = st.number_input(
                "Beta Prior",
                min_value=0.01,
                max_value=5.0,
                value=round(1.0 / n_topics, 3),
                step=0.05,
                disabled=not custom_beta,
                help="Dirichlet distribution hyperparameter for topic-word distributions (lower = sparser vocabulary)."
            ) if custom_beta else None

    # Train model button
    if st.button("🚀 Train LDA Model", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Fitting Latent Dirichlet Allocation (K={n_topics}) over {n_docs} documents..."):
                active_preproc = st.session_state.get('preprocessor')
                active_labels = st.session_state.get('custom_labels', {})

                modeler = TopicModeler(
                    n_topics=n_topics,
                    max_iter=max_iter,
                    learning_method=learning_method,
                    learning_decay=learning_decay,
                    doc_topic_prior=doc_topic_prior,
                    topic_word_prior=topic_word_prior,
                    min_df=int(min_df),
                    max_df=float(max_df),
                    max_features=max_features,
                    random_state=int(random_seed),
                    top_words_count=top_words_count,
                    preprocessor=active_preproc,
                    custom_labels=active_labels
                )
                modeler.fit(cleaned_df['clean_text'], preprocessor=active_preproc)

                # Generate Document-Topic assignments
                doc_topic_df = modeler.get_document_topic_df(cleaned_df, 'clean_text')

                # Run comprehensive evaluations
                eval_results = evaluate_model_comprehensively(
                    topic_modeler=modeler,
                    cleaned_df=cleaned_df,
                    text_column='clean_text',
                    top_n=top_words_count
                )

                # Store in session state
                st.session_state['topic_modeler'] = modeler
                st.session_state['doc_topic_df'] = doc_topic_df
                st.session_state['evaluation_results'] = eval_results

            st.success(f"LDA Model successfully trained in {modeler.training_time_sec} seconds! Discovered {n_topics} topics across {len(modeler.feature_names)} terms.")
            st.rerun()

        except Exception as e:
            st.error(f"Error during LDA model training: {e}")

    st.markdown("---")

    # Display results if model is trained
    if has_trained_model():
        tm = st.session_state['topic_modeler']
        eval_res = st.session_state.get('evaluation_results', {})

        st.subheader("🎯 Discovered Topics & Keywords")
        
        # Display training metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Discovered Topics", tm.n_topics)
        with m2:
            st.metric("Vocabulary Size", f"{len(tm.feature_names):,} terms")
        with m3:
            st.metric("Perplexity", eval_res.get('perplexity', 'N/A'))
        with m4:
            st.metric("Training Time", f"{tm.training_time_sec}s")

        # Feature Engineering: DTM inspection
        with st.expander("📐 Feature Engineering & Document-Term Matrix (DTM) Details", expanded=False):
            if tm.dtm is not None:
                n_docs_dtm, n_terms_dtm = tm.dtm.shape
                total_elements = n_docs_dtm * n_terms_dtm
                non_zero = tm.dtm.nnz
                sparsity = round(1.0 - (non_zero / total_elements), 4) if total_elements > 0 else 1.0

                dtm_c1, dtm_c2, dtm_c3, dtm_c4 = st.columns(4)
                with dtm_c1:
                    st.metric("DTM Shape", f"{n_docs_dtm} × {n_terms_dtm}")
                with dtm_c2:
                    st.metric("Non-Zero Entries", f"{non_zero:,}")
                with dtm_c3:
                    st.metric("Matrix Sparsity", f"{sparsity:.1%}")
                with dtm_c4:
                    st.metric("Density", f"{(1.0 - sparsity):.2%}")

                st.caption(
                    "Standard LDA processes this non-negative integer count matrix. "
                    "Each entry $C_{d,w}$ represents the count of word $w$ in document $d$."
                )

        # Disclaimer alert
        st.warning(
            "⚠️ **Interpretability Disclaimer:** Automatically generated labels below represent statistical "
            "term frequency distributions. They do not constitute verified ground-truth meanings. "
            "Please review and customize labels in the **Topic Exploration** tab."
        )

        # Topic summary cards
        for summary in tm.topic_summaries:
            with st.container():
                st.markdown(f"### {summary['topic_name']}: `{summary['auto_label'].replace(summary['topic_name'] + ': ', '')}`")
                
                kw_badges = " ".join([f"`{w}` ({wt:.3f})" for w, wt in zip(summary['top_words'][:8], summary['top_weights'][:8])])
                st.markdown(f"**Top Terms:** {kw_badges}")
                st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

        st.info("💡 Next steps: Dive into detailed keyword weights in **Topic Exploration** or examine document assignments in **Document Analysis**.")
