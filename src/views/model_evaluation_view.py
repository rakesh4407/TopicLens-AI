"""
src/views/model_evaluation_view.py
Quantitative & Qualitative Model Evaluation View for TopicLens AI.
Displays Perplexity, Log-Likelihood, Topic Diversity, Topic Coherence,
and critical pedagogical notes on metric limitations.
"""

import streamlit as st
import pandas as pd
from src.session_manager import has_trained_model


def render_model_evaluation_view():
    st.title("📈 Model Evaluation & Topic Quality")
    st.write(
        "Assess mathematical goodness-of-fit, topic coherence, word diversity, "
        "and understand the practical boundaries of statistical evaluation in NLP."
    )

    if not has_trained_model():
        st.warning("⚠️ No trained LDA model found. Please train a model in **Topic Modelling** first.")
        return

    tm = st.session_state['topic_modeler']
    eval_res = st.session_state.get('evaluation_results')

    if not eval_res:
        st.warning("Evaluation metrics are not yet available.")
        return

    # Critical Pedagogical Alert Box
    st.info(
        "🧠 **Fundamental NLP Principle:** " + eval_res.get('educational_note', '')
    )

    st.subheader("🎯 Quantitative Evaluation Metrics")

    # Metric KPI cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Model Perplexity",
            value=f"{eval_res.get('perplexity', 0.0):,}",
            help="Lower is better. Measures how surprised the model is by word occurrences in the corpus."
        )
    with m2:
        st.metric(
            label="Log-Likelihood",
            value=f"{eval_res.get('log_likelihood', 0.0):,}",
            help="Higher (less negative) is better. Measures log marginal probability of the observed document-term matrix."
        )
    with m3:
        st.metric(
            label="Topic Diversity",
            value=f"{eval_res.get('topic_diversity', 0.0):.3f}",
            help="Scale 0.0 to 1.0. Fraction of unique words among the top keywords across all topics (higher means less redundancy)."
        )
    with m4:
        st.metric(
            label="Mean UMass Coherence",
            value=f"{eval_res.get('mean_umass_coherence', 0.0):.3f}",
            help="Intrinsic co-occurrence coherence (Mimno et al., 2011). Values closer to 0 indicate tighter co-occurrence."
        )

    if eval_res.get('gensim_cv_coherence') is not None:
        st.metric("Gensim $C_v$ Coherence", f"{eval_res.get('gensim_cv_coherence'):.4f}")

    st.markdown("---")

    # Detailed Per-Topic Coherence Table
    st.subheader("📋 Per-Topic Coherence & Keywords Table")
    topic_eval_df = eval_res.get('topic_eval_df')
    if topic_eval_df is not None:
        st.dataframe(topic_eval_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Hyperparameter & Metric Sensitivity Guide
    st.subheader("💡 Hyperparameter Tuning Insights & Diagnostic Guide")
    with st.expander("📖 How to Interpret and Tune Topic Modelling Metrics", expanded=True):
        st.markdown("""
        | Metric | Ideal Trend | Mathematical Meaning | Practical Recommendation |
        |---|---|---|---|
        | **Perplexity** | **Lower** | $\\exp\\left(-\\frac{\\mathcal{L}(W)}{\\sum N_d}\\right)$ | Good for comparing models on held-out data, but empirical research shows lower perplexity can sometimes correlate with less human-interpretable topics. |
        | **Log-Likelihood** | **Higher (less negative)** | Probability of observing text counts given Dirichlet priors. | Increases as iterations converge. If stalled early, increase max iterations. |
        | **UMass Coherence** | **Less negative (closer to 0)** | Term pair co-occurrence log probability. | Highly negative values suggest top words rarely appear together in documents. |
        | **Topic Diversity** | **Closer to 1.0** | $\\frac{|\\text{Unique Top Words}|}{K \\times N}$ | If diversity is below 0.6, topics are sharing too many identical keywords; increase `min_df` or add custom stopwords. |
        """)
