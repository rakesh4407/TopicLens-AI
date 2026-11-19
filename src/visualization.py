"""
src/visualization.py
Interactive visualizations for TopicLens AI using Plotly, Matplotlib, and WordCloud.
Generates topic word bars, topic prevalence distributions, document-topic heatmaps,
2D PCA/t-SNE document projections, and Word Clouds.
"""

from io import BytesIO
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


# Aesthetic color palettes for topics
TOPIC_COLORS = [
    '#2563EB', '#7C3AED', '#DB2777', '#EA580C', '#16A34A',
    '#0D9488', '#4F46E5', '#9333EA', '#C026D3', '#D97706',
    '#059669', '#0891B2', '#4338CA', '#6D28D9', '#BE185D'
]


def plot_topic_distribution(doc_df: pd.DataFrame) -> Any:
    """
    Generate an interactive bar chart of dominant topic distribution across documents.
    """
    if 'Dominant_Topic_Label' not in doc_df.columns:
        return None

    counts = doc_df['Dominant_Topic_Label'].value_counts().reset_index()
    counts.columns = ['Topic', 'Document Count']
    counts['Percentage'] = (counts['Document Count'] / len(doc_df) * 100).round(1)

    if HAS_PLOTLY:
        fig = px.bar(
            counts,
            x='Topic',
            y='Document Count',
            text='Document Count',
            color='Topic',
            color_discrete_sequence=TOPIC_COLORS,
            title="<b>Dominant Topic Distribution Across Documents</b>",
            labels={'Document Count': 'Number of Documents', 'Topic': 'Discovered Topic'}
        )
        fig.update_traces(
            texttemplate='%{text} docs (%{customdata[0]}%)',
            customdata=counts[['Percentage']],
            textposition='outside'
        )
        fig.update_layout(
            xaxis_tickangle=-25,
            showlegend=False,
            height=450,
            margin=dict(l=40, r=40, t=60, b=80),
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(counts['Topic'], counts['Document Count'], color=TOPIC_COLORS[:len(counts)])
        ax.set_ylabel('Document Count')
        ax.set_title('Topic Distribution Across Documents')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        return fig


def plot_topic_pie(doc_df: pd.DataFrame) -> Any:
    """
    Generate a clean donut chart of topic proportions.
    """
    if 'Dominant_Topic_Label' not in doc_df.columns:
        return None

    counts = doc_df['Dominant_Topic_Label'].value_counts().reset_index()
    counts.columns = ['Topic', 'Count']

    if HAS_PLOTLY:
        fig = px.pie(
            counts,
            names='Topic',
            values='Count',
            hole=0.45,
            color='Topic',
            color_discrete_sequence=TOPIC_COLORS,
            title="<b>Topic Share Breakdown</b>"
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=450, template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
        return fig
    return None


def plot_topic_top_words(summary: Dict[str, Any], top_n: int = 10) -> Any:
    """
    Generate horizontal bar chart for top keywords in a single topic with importance weights.
    """
    words = summary['top_words'][:top_n]
    weights = summary['top_weights'][:top_n]

    # Sort ascending for horizontal bar chart display
    sorted_pairs = sorted(zip(words, weights), key=lambda x: x[1])
    s_words = [p[0] for p in sorted_pairs]
    s_weights = [p[1] for p in sorted_pairs]

    topic_title = summary.get('auto_label', f"Topic {summary.get('topic_number', 1)}")

    if HAS_PLOTLY:
        fig = go.Figure(go.Bar(
            x=s_weights,
            y=s_words,
            orientation='h',
            marker=dict(color='#3B82F6', line=dict(color='#1D4ED8', width=1)),
            text=[f"{w:.3f}" for w in s_weights],
            textposition='auto'
        ))
        fig.update_layout(
            title=f"<b>Top Terms: {topic_title}</b>",
            xaxis_title="Importance Weight (Component Value)",
            yaxis_title="Keywords",
            height=380,
            template="plotly_white",
            margin=dict(l=60, r=30, t=50, b=40)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(s_words, s_weights, color='#3B82F6')
        ax.set_title(topic_title)
        ax.set_xlabel('Weight')
        plt.tight_layout()
        return fig


def plot_all_topics_grid(topic_summaries: List[Dict[str, Any]], top_n: int = 8) -> Any:
    """
    Multi-bar comparison comparing top keywords across all discovered topics.
    """
    rows = []
    for s in topic_summaries:
        for w, wt in zip(s['top_words'][:top_n], s['top_weights'][:top_n]):
            rows.append({
                'Topic': s['topic_name'],
                'Word': w,
                'Weight': wt
            })
    df_grid = pd.DataFrame(rows)

    if HAS_PLOTLY:
        fig = px.bar(
            df_grid,
            x='Weight',
            y='Word',
            color='Topic',
            facet_col='Topic',
            facet_col_wrap=2,
            orientation='h',
            height=280 * ((len(topic_summaries) + 1) // 2),
            title="<b>Comparative Keyword Weights Across All Topics</b>",
            template="plotly_white"
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(showlegend=False)
        return fig
    return None


def plot_document_topic_heatmap(doc_df: pd.DataFrame, n_topics: int, max_docs: int = 25) -> Any:
    """
    Generate an interactive heatmap of topic probabilities for a sample of documents.
    """
    prob_cols = [f"Topic_{i + 1}_Prob" for i in range(n_topics) if f"Topic_{i + 1}_Prob" in doc_df.columns]
    if not prob_cols:
        return None

    sample_df = doc_df.head(max_docs)
    matrix = sample_df[prob_cols].values
    y_labels = [f"Doc #{i+1}" for i in range(len(sample_df))]
    x_labels = [f"Topic {i+1}" for i in range(n_topics)]

    if HAS_PLOTLY:
        fig = px.imshow(
            matrix,
            labels=dict(x="Topic", y="Document", color="Probability"),
            x=x_labels,
            y=y_labels,
            color_continuous_scale="Viridis",
            title=f"<b>Document-Topic Probability Heatmap (First {len(sample_df)} Documents)</b>",
            aspect="auto"
        )
        fig.update_layout(height=480, template="plotly_white")
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(matrix, cmap='viridis')
        fig.colorbar(cax)
        ax.set_xticks(range(n_topics))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(sample_df)))
        ax.set_yticklabels(y_labels)
        plt.tight_layout()
        return fig


def plot_document_pca_projection(doc_topic_dist: np.ndarray, doc_df: pd.DataFrame) -> Any:
    """
    Project high-dimensional document-topic vectors onto 2D plane using PCA.
    Visualizes document clusters and topic boundaries.
    """
    if doc_topic_dist is None or len(doc_topic_dist) < 3:
        return None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(doc_topic_dist)

    plot_data = pd.DataFrame({
        'PCA_1': coords[:, 0],
        'PCA_2': coords[:, 1],
        'Dominant_Topic': doc_df['Dominant_Topic_Label'] if 'Dominant_Topic_Label' in doc_df.columns else "Doc",
        'Probability': doc_df['Dominant_Topic_Probability'] if 'Dominant_Topic_Probability' in doc_df.columns else 1.0,
        'Doc_ID': range(1, len(coords) + 1)
    })

    explained_var = pca.explained_variance_ratio_ * 100

    if HAS_PLOTLY:
        fig = px.scatter(
            plot_data,
            x='PCA_1',
            y='PCA_2',
            color='Dominant_Topic',
            color_discrete_sequence=TOPIC_COLORS,
            size='Probability',
            hover_data=['Doc_ID', 'Dominant_Topic', 'Probability'],
            title=(
                f"<b>2D Document Topic Space Projection (PCA)</b> "
                f"<br><sup>Explains {explained_var[0]:.1f}% (PC1) and {explained_var[1]:.1f}% (PC2) of topic variance</sup>"
            ),
            template="plotly_white",
            height=500
        )
        fig.update_layout(
            legend_title_text='Dominant Topic',
            margin=dict(l=40, r=40, t=70, b=40)
        )
        return fig
    return None


def generate_topic_wordcloud(word_weight_dict: Dict[str, float]) -> Optional[plt.Figure]:
    """
    Generate a high-resolution WordCloud figure from a word-weight dictionary.
    """
    if not HAS_WORDCLOUD or not word_weight_dict:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        prefer_horizontal=0.9,
        max_words=60,
        random_state=42
    ).generate_from_frequencies(word_weight_dict)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def generate_overall_wordcloud(texts: List[str]) -> Optional[plt.Figure]:
    """
    Generate an overall corpus WordCloud from preprocessed document texts.
    """
    if not HAS_WORDCLOUD or not texts:
        return None

    combined_text = " ".join([str(t) for t in texts if isinstance(t, str)])
    if not combined_text.strip():
        return None

    wc = WordCloud(
        width=900,
        height=450,
        background_color='white',
        colormap='coolwarm',
        max_words=100,
        random_state=42
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig
