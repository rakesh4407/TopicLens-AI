# 🔍 TopicLens AI — Intelligent Text Topic Modelling & Discovery

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.10%2B-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BCA AI & Data Science Capstone Project**  
> **Project #9:** Text Topic Modelling  
> **Group:** Group 6  
> **Objective:** Uncover latent semantic topics, keyword importance distributions, and document thematic clustering across unstructured textual corpora.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Objectives](#-problem-statement--objectives)
3. [Technology Stack](#-technology-stack)
4. [System Architecture](#-system-architecture)
5. [Algorithm & Theoretical Formulation (LDA)](#-algorithm--theoretical-formulation-lda)
6. [Supported Datasets](#-supported-datasets)
7. [Text Preprocessing Pipeline](#-text-preprocessing-pipeline)
8. [Dashboard & Application Views](#-dashboard--application-views)
9. [Quantitative Model Evaluation](#-quantitative-model-evaluation)
10. [Installation & Setup](#-installation--setup)
11. [Running the Application](#-running-the-application)
12. [Running Automated Tests](#-running-automated-tests)
13. [How to Push Project to GitHub](#-how-to-push-project-to-github)
14. [Limitations & Future Roadmap](#-limitations--future-roadmap)
15. [Contributors](#-contributors)

---

## 🌟 Project Overview

**TopicLens AI** is an end-to-end, enterprise-grade Natural Language Processing (NLP) web application designed to automatically discover hidden themes, categorize documents, and extract high-value semantic insights from unstructured textual collections. Built with **Streamlit**, **Scikit-Learn**, and **Plotly**, the system allows data scientists, analysts, and students to ingest multi-format datasets (CSVs, TXT files, and benchmark corpora), execute configurable text preprocessing, train Latent Dirichlet Allocation (LDA) models over Document-Term Matrices (DTM), evaluate statistical coherence, explore interactive visualizations, test live text with real-time inference, and export reproducible model artifacts and executive reports.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
In the era of information overload, organizations accumulate massive repositories of unstructured text (customer reviews, research papers, support tickets, news articles) without predefined taxonomic labels. Manual reading and classification are cost-prohibitive, subjective, and unscalable. Unsupervised machine learning models are required to discover latent thematic structures directly from raw text distributions.

### Main Objectives
1. **Unsupervised Discovery:** Extract $K$ distinct topics and their representative keywords from document collections without requiring labeled supervision.
2. **Document Thematic Profiling:** Compute posterior probability distributions over topics for every document and identify the dominant thematic category.
3. **Interactive Visual Analytics:** Provide multi-dimensional visual perspectives, including 2D PCA topic space projections, heatmaps, and keyword importance bar charts.
4. **Human-in-the-Loop Validation:** Allow domain experts to review, validate, and customize automated topic labels.
5. **Real-Time Live Inference:** Enable users to classify custom unseen text snippets and view instant topic probabilities.
6. **Reproducibility & Serialization:** Export structured CSV assignments, Joblib model bundles, and analytical executive reports.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Core Language** | Python 3.10+ (tested on Python 3.13) |
| **GUI Framework** | Streamlit (Multi-page routing, stateful session management) |
| **NLP & Preprocessing** | NLTK (Stopwords, WordNet Lemmatizer, Tokenizers) |
| **Machine Learning** | Scikit-Learn (`LatentDirichletAllocation`, `CountVectorizer`, `PCA`) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualizations** | Plotly Express & Graph Objects, Matplotlib, WordCloud |
| **Model Persistence** | Joblib |
| **Testing** | Built-in Python `unittest` framework |

> **Important Engineering Principle:** As per rigorous NLP standards, **`CountVectorizer`** is utilized to build the discrete Document-Term Matrix (DTM) for standard LDA. `TfidfVectorizer` produces continuous normalized weights that violate the Dirichlet-Multinomial likelihood assumption of standard LDA; it is provided solely as an optional auxiliary analytical feature.

---

## 🏗️ System Architecture

```
TopicLens-AI/
│
├── .github/
│   └── workflows/              # GitHub Actions CI matrix (Ubuntu & Windows, Py 3.12 & 3.13)
│       └── test.yml
├── app.py                      # Main application entry point & unified router
├── requirements.txt            # Python dependencies (tested on 3.12 & 3.13)
├── README.md                   # Comprehensive documentation & capstone guide
├── REPORT_OUTLINE.md           # Formal academic capstone report structure
├── PRESENTATION_OUTLINE.md     # 15-slide capstone presentation outline
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── src/                        # Modular application source code
│   ├── __init__.py             # Package exports
│   ├── config.py               # Centralized configuration, defaults & dataset registry
│   ├── session_manager.py      # Streamlit session state persistence
│   ├── data_loader.py          # Multi-format dataset ingestion & validation
│   ├── preprocessing.py        # Configurable text cleaning & lemmatization
│   ├── feature_engineering.py  # CountVectorizer DTM & feature extraction
│   ├── topic_model.py          # Scikit-Learn LDA engine with prior tuning
│   ├── inference.py            # Real-time topic inference & batch predictor
│   ├── evaluation.py           # Perplexity, log-likelihood, UMass coherence
│   ├── visualization.py        # Plotly charts, heatmaps, PCA, word clouds
│   ├── export_manager.py       # Multi-format artifact & report exporter
│   ├── utils.py                # Report generation (Markdown/HTML) & CSV encoders
│   └── views/                  # UI view renderers
│       ├── home_view.py
│       ├── dataset_view.py
│       ├── preprocessing_view.py
│       ├── topic_modelling_view.py
│       ├── topic_exploration_view.py
│       ├── document_analysis_view.py
│       ├── visual_analytics_view.py
│       ├── model_evaluation_view.py
│       ├── export_results_view.py
│       └── about_project_view.py
│
├── pages/                      # Native Streamlit multipage routing
│   ├── 1_Dataset.py
│   ├── 2_Preprocessing.py
│   ├── 3_Topic_Modelling.py
│   ├── 4_Topic_Exploration.py
│   ├── 5_Document_Analysis.py
│   ├── 6_Visual_Analytics.py
│   ├── 7_Model_Evaluation.py
│   ├── 8_Export_Results.py
│   └── 9_About_Project.py
│
├── data/
│   └── sample/                 # Curated benchmark datasets
│       ├── 20newsgroups_subset.csv
│       ├── customer_feedback.csv
│       ├── tech_news.csv
│       └── biomedical_research.csv
│
├── models/                     # Serialized Joblib model bundles
├── outputs/                    # Exported analysis reports
└── tests/                      # Automated unit & integration tests (43 tests)
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_topic_model.py
    ├── test_inference.py
    ├── test_evaluation.py
    ├── test_exports.py
    └── test_e2e_pipeline.py
```

---

## 📐 Algorithm & Theoretical Formulation (LDA)

**Latent Dirichlet Allocation (Blei, Ng, Jordan, 2003)** is a hierarchical Bayesian generative model. It assumes that:
- Each document $d$ is a mixture of $K$ latent topics, governed by document-topic Dirichlet prior $\alpha$.
- Each topic $k$ is a discrete multinomial distribution over the vocabulary $V$, governed by topic-word Dirichlet prior $\eta$.

### Generative Story:
For each document $d \in \{1, \dots, D\}$:
1. Draw topic proportions $\theta_d \sim \text{Dirichlet}(\alpha)$.
2. For each word position $n \in \{1, \dots, N_d\}$:
   - Sample topic $z_{dn} \sim \text{Multinomial}(\theta_d)$.
   - Sample word $w_{dn} \sim \text{Multinomial}(\beta_{z_{dn}})$, where $\beta_k \sim \text{Dirichlet}(\eta)$.

### Posterior Estimation:
Because exact posterior marginalization over continuous $\theta$ and discrete $z$ is analytically intractable:
$$p(\Theta, Z \mid W, \alpha, \eta) = \frac{p(W, Z, \Theta \mid \alpha, \eta)}{\int \sum_Z p(W, Z, \Theta \mid \alpha, \eta) d\Theta}$$
Scikit-Learn optimizes the variational parameters via **Batch Variational Expectation-Maximization** or **Online Variational Bayes** (Hoffman et al.), iteratively maximizing the Evidence Lower Bound (ELBO).

---

## 📂 Supported Datasets

TopicLens AI supports multiple ingestion mechanisms:
1. **CSV Datasets:** Upload any CSV file. The application automatically detects text columns (via length and uniqueness heuristics) and allows the user to select the target text field.
2. **Multiple TXT Files:** Upload a batch of plain text files; they are merged into a unified DataFrame with document IDs and word counts.
3. **Pre-configured Benchmarks (1-Click Load):**
   - **20 Newsgroups Subset:** Real newsgroup posts across science, computing, autos, and politics.
   - **Customer Feedback & Reviews:** Multi-topic customer reviews covering Hardware, Customer Service, Shipping, and Software.
   - **Technology & AI News:** Multi-domain articles spanning Artificial Intelligence, Space Exploration, Cybersecurity, and Quantum Computing.

---

## 🧹 Text Preprocessing Pipeline

The preprocessing pipeline cleans unstructured text through 10 configurable stages:
1. **Case Normalization:** Converts all tokens to lowercase.
2. **HTML / XML Stripping:** Eliminates tags (`<p>`, `<div>`, `<br>`).
3. **URL & Link Removal:** Regex pattern `https?://\S+|www\.\S+`.
4. **Punctuation Stripping:** Removes punctuation characters to avoid token concatenation.
5. **Digit Filtering:** Strips standalone numbers and isolated digits.
6. **Tokenization:** Regex-based tokenization extracting alphabetical words ($\ge 3$ characters).
7. **Stopword Removal:** Eliminates 180+ standard English stopwords plus custom domain stopwords.
8. **Lemmatization:** Reduces inflected forms to root lemmas using NLTK `WordNetLemmatizer`.
9. **Empty Document Removal:** Discards documents that reduce to zero tokens after cleaning.
10. **Deduplication:** Flags and prunes duplicate records to prevent cluster bias.

---

## 🖥️ Dashboard & Application Views

The application provides 10 dedicated views:
1. **🏠 Home Dashboard:** High-level pipeline status cards, KPI metrics, active topic distributions, and quick-start demonstration triggers.
2. **📂 Dataset Upload:** CSV/TXT file upload, sample dataset selection, missing value reports, and word count distributions.
3. **🧹 Data Preprocessing:** Preprocessing toggles, execution button, before-and-after document comparisons, and vocabulary reduction metrics.
4. **⚙️ Topic Modelling:** Hyperparameter tuning ($K$, top words, min_df, max_df, max_features, iterations, random seed), LDA training, and initial topic overview.
5. **🔍 Topic Exploration:** Single-topic deep dives, human-in-the-loop label verification, horizontal keyword weight bar charts, topic word clouds, and representative exemplar documents.
6. **📄 Document Analysis:** Complete document-topic assignments table, keyword search, document-level probability inspector, and **Live Inference Sandbox** for custom text.
7. **📊 Visual Analytics:** 2D Document Topic Space Projection (PCA), Document-Topic probability heatmaps, macro distribution donut charts, and overall corpus word clouds.
8. **📈 Model Evaluation:** Perplexity, log-likelihood, topic diversity, intrinsic UMass topic coherence, and educational interpretability guidelines.
9. **💾 Export Results:** CSV downloads for document topics, keywords, and cleaned data; Joblib model bundle download; Markdown & HTML executive reports; and model bundle reloading.
10. **ℹ️ About Project:** Capstone academic specifications, theoretical formulas, engineering architecture, and real-world applications.

---

## 📊 Quantitative Model Evaluation

### Metrics Implemented:
1. **Perplexity:** $\exp\left(-\frac{\mathcal{L}(W)}{\sum N_d}\right)$ — measures document reconstruction error (lower is better).
2. **Log-Likelihood:** Marginal probability of the observed document-term matrix given Dirichlet priors (higher / less negative is better).
3. **Topic Diversity:** Proportion of unique terms across all top-$N$ topic words (scale $0.0$ to $1.0$, higher indicates non-redundant vocabularies).
4. **Intrinsic UMass Coherence (Mimno et al., 2011):**
   $$C_{\text{UMass}}(k) = \sum_{m=2}^{M} \sum_{l=1}^{m-1} \log \left( \frac{D(w_m, w_l) + \epsilon}{D(w_l)} \right)$$
   Measures empirical co-occurrence of top keywords within documents (values closer to 0 indicate tighter semantic co-occurrence).

> **Pedagogical Note:** Statistical evaluation metrics quantify mathematical fit. However, extensive NLP research confirms that mathematical scores do not independently guarantee that topics are conceptually meaningful to humans. Qualitative human-in-the-loop validation is essential.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/TopicLens-AI.git
cd TopicLens-AI
```

### Step 2: Create and Activate Virtual Environment (Recommended)
**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Corpora
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🧪 Running Automated Tests

TopicLens AI includes a complete automated test suite of **43 unit and integration tests** covering all pipeline layers:

```bash
python -m unittest discover tests -v
```

To run individual test suites:
```bash
python -m unittest tests/test_data_loader.py
python -m unittest tests/test_preprocessing.py
python -m unittest tests/test_feature_engineering.py
python -m unittest tests/test_topic_model.py
python -m unittest tests/test_inference.py
python -m unittest tests/test_evaluation.py
python -m unittest tests/test_exports.py
python -m unittest tests/test_e2e_pipeline.py
```

---

## 📤 How to Push Project to GitHub

Follow these steps to publish this capstone project to your personal GitHub repository:

### 1. Initialize Git in the Project Root
```bash
git init
```

### 2. Stage All Project Files
```bash
git add .
```

### 3. Create the Initial Commit
```bash
git commit -m "feat: initial commit of TopicLens AI complete topic modelling application"
```

### 4. Create a New Repository on GitHub
1. Log in to [GitHub](https://github.com/).
2. Click **New Repository** (green button).
3. Name the repository: `TopicLens-AI` (or `NLP-Project-9-Topic-Modelling`).
4. Set visibility to **Public**.
5. Leave "Add a README" and ".gitignore" **unchecked** (we already have them).
6. Click **Create repository**.

### 5. Link and Push to Remote
```bash
# Rename branch to main
git branch -M main

# Link remote URL (replace with your actual GitHub repo URL)
git remote add origin https://github.com/rakesh4407/TopicLens-AI.git

# Push code to GitHub
git push -u origin main
```

---

## ⚠️ Limitations & Future Roadmap

### Current Limitations
- **Bag-of-Words Order Blindness:** Standard LDA treats documents as unordered bags of words, ignoring syntactical nuance and negation phrases.
- **Predefined $K$:** The user must configure the number of topics $K$ manually.
- **Short Text Sparsity:** On single-line tweets or short search queries, term co-occurrence matrices can be sparse.

### Future Roadmap
- **Transformer-based Neural Topic Models:** Integrating BERTopic / Top2Vec leveraging contextual sentence-transformer embeddings.
- **Dynamic Topic Modelling (DTM):** Visualizing how topic vocabularies and prevalence evolve over temporal timestamps.
- **Seeded / Guided LDA:** Allowing domain analysts to inject prior anchor seed words for specific business topics.

---

## 👥 Contributors

- **Course:** Bachelor of Computer Applications (BCA) — Semester 5
- **Specialization:** AI & Data Science Capstone Project
- **Group:** Group 6
- **Project Assigned:** Project 9 — Text Topic Modelling
