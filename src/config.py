"""
src/config.py
Centralized application configuration for TopicLens AI.
All constants, defaults, and paths are defined here to avoid magic values scattered across the codebase.
"""

from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root and key directory paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = ROOT_DIR / "models"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
ASSETS_DIR: Path = ROOT_DIR / "assets"

# Ensure runtime directories exist
for _d in [MODELS_DIR, OUTPUTS_DIR, SAMPLE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Application Identity
# ---------------------------------------------------------------------------
APP_NAME: str = "TopicLens AI"
APP_TAGLINE: str = "Discover Hidden Meaning. Explore Every Topic."
APP_VERSION: str = "1.0.0"
PROJECT_NUMBER: int = 9
PROJECT_TITLE: str = "Text Topic Modelling"
DEGREE: str = "Bachelor of Computer Applications (AI & Data Science)"

# ---------------------------------------------------------------------------
# LDA Default Hyperparameters
# ---------------------------------------------------------------------------
LDA_DEFAULTS: Dict = {
    "n_topics": 4,
    "max_iter": 25,
    "learning_method": "batch",   # 'batch' or 'online'
    "learning_decay": 0.7,
    "doc_topic_prior": None,      # None → 1/n_topics (symmetric)
    "topic_word_prior": None,     # None → 1/n_topics (symmetric)
    "random_state": 42,
    "top_words_count": 10,
}

# ---------------------------------------------------------------------------
# CountVectorizer Default Settings
# ---------------------------------------------------------------------------
VECTORIZER_DEFAULTS: Dict = {
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 2000,
    "ngram_range": (1, 1),   # unigrams by default
    "token_pattern": r"\b[a-zA-Z]{3,}\b",
}

# ---------------------------------------------------------------------------
# Preprocessing Defaults
# ---------------------------------------------------------------------------
PREPROCESSING_DEFAULTS: Dict = {
    "lowercase": True,
    "remove_html": True,
    "remove_urls": True,
    "remove_punctuation": True,
    "remove_numbers": True,
    "remove_stopwords": True,
    "lemmatize": True,
    "min_word_len": 3,
    "drop_empty": True,
    "drop_duplicates": True,
    "custom_stopwords": [],
}

# ---------------------------------------------------------------------------
# Sample Dataset Registry
# ---------------------------------------------------------------------------
SAMPLE_DATASETS: List[Dict] = [
    {
        "id": "20newsgroups",
        "label": "20 Newsgroups — Computer Graphics, Space, Autos & Politics",
        "file": "20newsgroups_subset.csv",
        "text_col": "text",
        "description": (
            "A curated subset of the 20 Newsgroups benchmark corpus "
            "(available via scikit-learn). Contains posts from four groups: "
            "comp.graphics, sci.space, rec.autos, and talk.politics.misc."
        ),
        "is_synthetic": False,
    },
    {
        "id": "customer_feedback",
        "label": "Customer Reviews — Hardware, Support, Shipping, Software & Pricing",
        "file": "customer_feedback.csv",
        "text_col": "feedback_text",
        "description": (
            "Synthetic customer feedback reviews covering five operational categories. "
            "Created for demonstration and testing purposes; not real user-generated data."
        ),
        "is_synthetic": True,
    },
    {
        "id": "tech_news",
        "label": "Tech News Articles — AI, Space Exploration, Cybersecurity & Quantum",
        "file": "tech_news.csv",
        "text_col": "content",
        "description": (
            "Synthetic technology news summaries spanning Artificial Intelligence, "
            "Space Exploration, Cybersecurity, and Quantum Computing. "
            "Created for demonstration; not sourced from real publications."
        ),
        "is_synthetic": True,
    },
    {
        "id": "biomedical_research",
        "label": "Biomedical Research — Cardiology, Neuroscience, Oncology & Pharmacology",
        "file": "biomedical_research.csv",
        "text_col": "abstract_text",
        "description": (
            "Curated scientific abstracts covering five biomedical specialties: "
            "Cardiology, Neuroscience, Oncology, Pharmacology, and Epidemiology."
        ),
        "is_synthetic": False,
    },
]

# ---------------------------------------------------------------------------
# UI Colour Palette (consistent across all charts)
# ---------------------------------------------------------------------------
TOPIC_COLORS: List[str] = [
    "#2563EB",  # Blue
    "#7C3AED",  # Violet
    "#DB2777",  # Pink
    "#EA580C",  # Orange
    "#16A34A",  # Green
    "#0D9488",  # Teal
    "#4F46E5",  # Indigo
    "#9333EA",  # Purple
    "#C026D3",  # Fuchsia
    "#D97706",  # Amber
    "#059669",  # Emerald
    "#0891B2",  # Cyan
    "#4338CA",  # Deep Indigo
    "#6D28D9",  # Deep Violet
    "#BE185D",  # Rose
]

PLOTLY_TEMPLATE: str = "plotly_white"
