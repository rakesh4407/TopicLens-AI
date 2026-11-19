"""
src/utils.py
Utility functions for TopicLens AI:
- Comprehensive analytical report generation (Markdown & HTML)
- CSV export formatting
- Metric badge formatting
- Session state initialization and reset helpers
"""

from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_analysis_report_markdown(
    dataset_summary: Dict[str, Any],
    preprocessing_summary: Dict[str, Any],
    topic_modeler,
    evaluation_metrics: Dict[str, Any]
) -> str:
    """
    Generate an executive-ready, detailed Markdown analysis report of the topic modeling experiment.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_topics = topic_modeler.n_topics if topic_modeler else 0
    vocab_size = len(topic_modeler.feature_names) if topic_modeler and topic_modeler.feature_names else 0

    lines = [
        "# TopicLens AI — Executive Topic Discovery & Analytics Report",
        f"**Generated:** {now}  ",
        "**Project:** #9 Text Topic Modelling (BCA AI & Data Science Capstone)  ",
        "**Algorithm:** Latent Dirichlet Allocation (LDA) with Document-Term Matrix (CountVectorizer)  ",
        "\n---\n",
        "## 1. Executive Summary",
        (
            f"TopicLens AI analyzed a corpus of **{dataset_summary.get('total_documents', 0)} documents** "
            f"and successfully uncovered **{n_topics} latent thematic topics** across a refined vocabulary "
            f"of **{vocab_size} unique terms**."
        ),
        "\n## 2. Dataset Overview",
        f"- **Selected Text Column:** `{dataset_summary.get('text_column', 'N/A')}`",
        f"- **Total Documents:** {dataset_summary.get('total_documents', 0)}",
        f"- **Valid Non-Empty Documents:** {dataset_summary.get('valid_documents', 0)}",
        f"- **Average Document Length:** {dataset_summary.get('avg_word_count', 0)} words",
        f"- **Duplicate Documents:** {dataset_summary.get('duplicate_documents', 0)}",
        "\n## 3. Text Preprocessing Pipeline",
        f"- **Original Documents:** {preprocessing_summary.get('original_document_count', 0)}",
        f"- **Processed Documents:** {preprocessing_summary.get('final_document_count', 0)}",
        f"- **Empty Documents Removed:** {preprocessing_summary.get('removed_empty_count', 0)}",
        f"- **Original Vocabulary Size:** {preprocessing_summary.get('raw_vocab_size', 0)} terms",
        f"- **Preprocessed Vocabulary Size:** {preprocessing_summary.get('clean_vocab_size', 0)} terms",
        f"- **Vocabulary Reduction:** {preprocessing_summary.get('vocab_reduction_percent', 0)}%",
        f"- **Average Clean Doc Length:** {preprocessing_summary.get('avg_clean_doc_length', 0)} words",
        "\n## 4. Model Configuration & Parameters",
        f"- **Model Algorithm:** Scikit-Learn `LatentDirichletAllocation`",
        f"- **Vectorization Strategy:** `CountVectorizer` (Document-Term Matrix)",
        f"- **Discovered Topics ($K$):** {n_topics}",
        f"- **Learning Method:** `{topic_modeler.learning_method if topic_modeler else 'batch'}`",
        f"- **Max Iterations:** {topic_modeler.max_iter if topic_modeler else 25}",
        f"- **Random Seed:** {topic_modeler.random_state if topic_modeler else 42}",
        f"- **Training Latency:** {topic_modeler.training_time_sec if topic_modeler else 0.0} seconds",
        "\n## 5. Discovered Topics & Keywords",
    ]

    if topic_modeler and topic_modeler.topic_summaries:
        for summary in topic_modeler.topic_summaries:
            top_kw = ", ".join(summary['top_words'][:8])
            lines.append(f"### {summary['topic_name']}: {summary['auto_label']}")
            lines.append(f"- **Top Keywords:** {top_kw}")
            lines.append("- **Top Terms with Weights:**")
            for w, wt in zip(summary['top_words'][:5], summary['top_weights'][:5]):
                lines.append(f"  - `{w}`: {wt:.4f}")
            lines.append("")

    lines.extend([
        "## 6. Quantitative Model Evaluation",
        f"- **Model Perplexity:** {evaluation_metrics.get('perplexity', 'N/A')} *(Lower is better)*",
        f"- **Log-Likelihood:** {evaluation_metrics.get('log_likelihood', 'N/A')} *(Higher is better)*",
        f"- **Topic Diversity:** {evaluation_metrics.get('topic_diversity', 'N/A')} *(Scale 0.0 - 1.0)*",
        f"- **Mean UMass Topic Coherence:** {evaluation_metrics.get('mean_umass_coherence', 'N/A')} *(Less negative is better)*",
    ])

    if evaluation_metrics.get('gensim_cv_coherence') is not None:
        lines.append(f"- **Gensim $C_v$ Coherence:** {evaluation_metrics.get('gensim_cv_coherence')}")

    lines.extend([
        "\n### Interpretability Notice",
        (
            "> [!IMPORTANT]\n"
            "> Statistical metrics quantify how well the Dirichlet distributions approximate term co-occurrences. "
            "> However, they do not guarantee semantic soundness. Human validation remains indispensable."
        ),
        "\n## 7. Limitations & Recommendations",
        "- **Context Sensitivity:** Bag-of-Words representations do not capture complex word order or syntactical negation.",
        "- **Hyperparameter Tuning:** Exploring varying values of $K$ (e.g., 2 to 10) allows identifying optimal semantic granularity.",
        "- **Domain Dictionaries:** Adding domain-specific stop words eliminates non-informative high-frequency industry jargon."
    ])

    return "\n".join(lines)


def generate_analysis_report_html(markdown_content: str) -> str:
    """
    Convert markdown analysis report into a standalone, styled HTML document for presentation.
    """
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TopicLens AI Analysis Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1f2937;
        background-color: #f9fafb;
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 860px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    h1 {{ color: #1e3a8a; border-bottom: 2px solid #3b82f6; padding-bottom: 12px; }}
    h2 {{ color: #1e40af; margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }}
    h3 {{ color: #2563eb; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
    ul {{ padding-left: 24px; }}
    li {{ margin-bottom: 6px; }}
    .footer {{ margin-top: 40px; font-size: 0.85em; color: #6b7280; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <pre style="white-space: pre-wrap; font-family: inherit;">{markdown_content}</pre>
    <div class="footer">TopicLens AI — Project #9 Text Topic Modelling</div>
</div>
</body>
</html>
"""
    return html_template


def df_to_csv_download(df: pd.DataFrame) -> bytes:
    """
    Encode DataFrame to UTF-8 CSV bytes for Streamlit st.download_button.
    """
    return df.to_csv(index=False).encode('utf-8')
