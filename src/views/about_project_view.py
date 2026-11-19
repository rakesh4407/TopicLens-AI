"""
src/views/about_project_view.py
About Project View: Academic Capstone Details, Architecture, Mathematical Formulations,
and Industry Applications.
"""

import streamlit as st


def render_about_project_view():
    st.title("ℹ️ About Project: TopicLens AI")

    st.markdown("""
    <div style="background-color: #F8FAFC; border-left: 5px solid #2563EB; padding: 18px 24px; border-radius: 6px; margin-bottom: 24px;">
        <h3 style="margin-top: 0; color: #1E3A8A;">BCA AI & Data Science Capstone Project</h3>
        <p style="margin-bottom: 6px;"><strong>Project Name:</strong> TopicLens AI</p>
        <p style="margin-bottom: 6px;"><strong>Project Number:</strong> 9</p>
        <p style="margin-bottom: 6px;"><strong>Project Title:</strong> Text Topic Modelling</p>
        <p style="margin-bottom: 0;"><strong>Main Objective:</strong> Discover hidden topics and common themes from a collection of text documents.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🔬 Theory & Mathematics",
        "🏗️ System Architecture",
        "🌐 Real-World Applications",
        "⚠️ Limitations & Future Scope"
    ])

    with tabs[0]:
        st.subheader("📐 Mathematical Foundation: Latent Dirichlet Allocation (LDA)")
        st.markdown(r"""
        Latent Dirichlet Allocation (Blei, Ng, Jordan, 2003) is a generative probabilistic 3-level hierarchical Bayesian model for collections of discrete data.

        ### Generative Process:
        For each document $d \in \{1, \dots, D\}$ in corpus $\mathcal{D}$:
        1. Choose topic proportions $\theta_d \sim \text{Dirichlet}(\alpha)$, where $\alpha$ is a $K$-dimensional hyperparameter vector.
        2. For each word position $n \in \{1, \dots, N_d\}$ in document $d$:
           - Choose a latent topic $z_{dn} \sim \text{Multinomial}(\theta_d)$.
           - Choose a word $w_{dn} \sim \text{Multinomial}(\beta_{z_{dn}})$, where $\beta_k \sim \text{Dirichlet}(\eta)$ represents the vocabulary distribution for topic $k$.

        ### Joint Likelihood:
        $$p(W, Z, \Theta, \Phi \mid \alpha, \eta) = \prod_{k=1}^K p(\phi_k \mid \eta) \prod_{d=1}^D p(\theta_d \mid \alpha) \left( \prod_{n=1}^{N_d} p(z_{dn} \mid \theta_d) p(w_{dn} \mid \phi_{z_{dn}}) \right)$$

        ### Inference:
        Because the exact posterior distribution $p(\Theta, Z \mid W, \alpha, \eta)$ is intractable due to coupling between $\theta$ and $\beta$, Scikit-Learn utilizes **Online Variational Bayes** (Hoffman et al.) or **Batch Variational Expectation-Maximization**, maximizing the Evidence Lower Bound (ELBO).
        """)

    with tabs[1]:
        st.subheader("🏗️ Modular Engineering Architecture")
        st.markdown(r"""
        TopicLens AI is engineered following clean, decoupled modular patterns:
        - **`src/config.py`**: Centralized application configuration, default hyperparameters, and dataset registry.
        - **`src/data_loader.py`**: Multi-format ingestion (CSV, TXT, sample benchmarks), encoding detection, missing value diagnostics.
        - **`src/preprocessing.py`**: Deterministic text cleaning, regex URL/HTML stripping, NLTK stopword removal, WordNet lemmatization, duplicate pruning.
        - **`src/feature_engineering.py`**: `CountVectorizer` Document-Term Matrix (DTM) builder, sparsity metrics, vocabulary frequency inspector, and optional secondary TF-IDF analysis.
        - **`src/topic_model.py`**: Scikit-Learn `LatentDirichletAllocation` engine, Dirichlet priors ($\alpha, \beta/\eta$), automatic topic labeling, and Joblib serialization.
        - **`src/inference.py`**: Real-time topic inference engine for unseen text, confidence level scoring, and batch prediction.
        - **`src/evaluation.py`**: Quantitative Perplexity, Log-Likelihood, Topic Diversity, and intrinsic UMass Topic Coherence.
        - **`src/visualization.py`**: Plotly interactive charts (Topic prevalence, PCA 2D scatter, heatmaps) and Matplotlib Word Clouds.
        - **`src/export_manager.py`**: Multi-format artifact and report exporter (CSV, Joblib, Markdown & HTML).
        - **`src/utils.py`**: Analytical reporting generators and formatters.
        """)

    with tabs[2]:
        st.subheader("🌐 Real-World Industry Applications")
        st.markdown("""
        1. **E-Commerce & Customer Support:** Automatically categorizing thousands of daily customer feedback tickets into actionable operational buckets (e.g., Shipping delays, Software bugs, Billing discrepancies).
        2. **Academic & Scientific Literature Mining:** Mapping trending scientific breakthroughs across millions of PubMed or arXiv research paper abstracts without manual tagging.
        3. **Legal e-Discovery & Compliance:** Clustering subpoenaed corporate email archives to detect communications relevant to specific litigation claims.
        4. **News & Media Monitoring:** Tracking emerging global news themes and geopolitical events across syndicated news feeds.
        """)

    with tabs[3]:
        st.subheader("⚠️ Limitations & Future Enhancements")
        st.markdown("""
        ### Current Limitations:
        - **Bag-of-Words Assumption:** Standard LDA ignores grammatical syntax and word ordering ("not good" vs "good not").
        - **Fixed $K$ Topics:** The user must specify the number of topics $K$ a priori (Hierarchical Dirichlet Processes could automate $K$ selection).
        - **Short Text Sparsity:** On extremely short single-sentence texts (e.g. tweets), term co-occurrence matrices can be sparse.

        ### Future Roadmap:
        - Integrating Neural Topic Models (BERTopic / Top2Vec) with contextual transformer embeddings (BERT/RoBERTa).
        - Dynamic Topic Models (DTM) to visualize how topic vocabularies evolve over time across timestamps.
        - Guided / Seeded LDA allowing domain experts to provide prior anchor words for specific target topics.
        """)
