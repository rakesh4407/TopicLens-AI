"""
src/evaluation.py
Comprehensive Model Evaluation module for TopicLens AI.
Includes:
- Perplexity (Sklearn built-in)
- Log-Likelihood (Sklearn built-in)
- Intrinsic Topic Coherence (UMass and NPMI formulation from Mimno et al.)
- Gensim CoherenceModel wrapper (when Gensim is installed)
- Topic Diversity (proportion of unique words in top words across topics)
- Educational interpretation guidelines on metric limitations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_perplexity_and_log_likelihood(lda_model, dtm) -> Dict[str, float]:
    """
    Calculate Perplexity and Log-Likelihood of the fitted LDA model on the Document-Term Matrix.
    
    Interpretation:
    - Perplexity: Lower is better. Measures how surprised the model is by new/existing data.
    - Log-Likelihood: Higher (less negative) is better. Measures probability of observed word counts.
    """
    if lda_model is None or dtm is None:
        raise ValueError("Model and Document-Term Matrix must be provided.")

    log_likelihood = float(lda_model.score(dtm))
    perplexity = float(lda_model.perplexity(dtm))

    return {
        'perplexity': round(perplexity, 2),
        'log_likelihood': round(log_likelihood, 2)
    }


def compute_topic_diversity(topic_summaries: List[Dict[str, Any]], top_n: int = 10) -> float:
    """
    Calculate Topic Diversity (Inan et al. / Dieng et al.):
    The proportion of unique words among the top-N words across all topics.
    Score ranges from 0.0 (all topics share identical top words) to 1.0 (topics have totally distinct vocabularies).
    """
    if not topic_summaries:
        return 0.0

    all_top_words = []
    total_words_considered = 0

    for topic in topic_summaries:
        words = topic.get('top_words', [])[:top_n]
        all_top_words.extend(words)
        total_words_considered += len(words)

    if total_words_considered == 0:
        return 0.0

    unique_words = set(all_top_words)
    diversity = len(unique_words) / total_words_considered
    return round(float(diversity), 4)


def compute_umass_coherence(
    topic_words: List[List[str]],
    dtm,
    feature_names: List[str],
    top_n: int = 10,
    epsilon: float = 1e-12
) -> Tuple[float, List[float]]:
    """
    Compute Intrinsic UMass Topic Coherence (Mimno et al., 2011).
    Does not require external reference corpora.
    Formula:
        C_UMass = sum_{m=2}^{M} sum_{l=1}^{m-1} log( (D(w_m, w_l) + eps) / D(w_l) )
    
    Higher (less negative, closer to 0) indicates more coherent topics where top words frequently co-occur.
    """
    # Create word-to-column index lookup
    word_to_idx = {word: idx for idx, word in enumerate(feature_names)}
    
    # Convert DTM to binary co-occurrence matrix (docs containing words)
    binary_dtm = (dtm > 0).astype(int)
    
    # Document frequency vector: count of documents containing each word
    doc_freqs = np.asarray(binary_dtm.sum(axis=0)).flatten()
    
    topic_coherences = []

    for words in topic_words:
        selected_words = [w for w in words[:top_n] if w in word_to_idx]
        m_len = len(selected_words)
        if m_len < 2:
            topic_coherences.append(0.0)
            continue

        indices = [word_to_idx[w] for w in selected_words]
        sub_dtm = binary_dtm[:, indices].toarray()  # (n_docs, m_len)
        
        # Word co-occurrence matrix for this topic's top words
        co_occurrence = np.dot(sub_dtm.T, sub_dtm)
        
        score = 0.0
        pair_count = 0

        for m in range(1, m_len):
            for l in range(0, m):
                d_ml = co_occurrence[m, l]
                d_l = doc_freqs[indices[l]]
                
                # UMass score pair
                score += np.log((d_ml + epsilon) / (d_l + epsilon))
                pair_count += 1

        topic_score = score / max(pair_count, 1)
        topic_coherences.append(float(round(topic_score, 4)))

    mean_coherence = float(round(np.mean(topic_coherences), 4)) if topic_coherences else 0.0
    return mean_coherence, topic_coherences


def compute_gensim_coherence(
    topic_words: List[List[str]],
    tokenized_corpus: List[List[str]],
    coherence_measure: str = 'c_v'
) -> Optional[float]:
    """
    Calculate Gensim CoherenceModel score (c_v or u_mass) if Gensim is installed.
    """
    try:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.coherencemodel import CoherenceModel

        dictionary = Dictionary(tokenized_corpus)
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence=coherence_measure
        )
        return float(round(cm.get_coherence(), 4))
    except Exception:
        return None


def evaluate_model_comprehensively(
    topic_modeler,
    cleaned_df: pd.DataFrame,
    text_column: str = 'clean_text',
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Run full suite of quantitative evaluation metrics and educational interpretability checks.
    """
    if not topic_modeler.is_fitted:
        raise ValueError("Model is not fitted.")

    lda = topic_modeler.lda_model
    dtm = topic_modeler.dtm
    feature_names = topic_modeler.feature_names
    summaries = topic_modeler.topic_summaries

    # 1. Perplexity and Log-Likelihood
    stats = compute_perplexity_and_log_likelihood(lda, dtm)
    
    # 2. Topic Diversity
    diversity = compute_topic_diversity(summaries, top_n=top_n)

    # 3. Topic words list
    topic_words = [s['top_words'][:top_n] for s in summaries]

    # 4. Intrinsic UMass Coherence
    mean_umass, topic_umass_scores = compute_umass_coherence(
        topic_words=topic_words,
        dtm=dtm,
        feature_names=feature_names,
        top_n=top_n
    )

    # 5. Gensim Coherence (if available)
    tokenized_docs = [
        str(t).split() for t in cleaned_df[text_column] if str(t).strip()
    ]
    gensim_cv = compute_gensim_coherence(topic_words, tokenized_docs, 'c_v')

    # Topic-specific report dataframe
    topic_eval_rows = []
    for i, summary in enumerate(summaries):
        topic_eval_rows.append({
            'Topic': summary['topic_name'],
            'Suggested Label': summary['auto_label'],
            'UMass Coherence': topic_umass_scores[i] if i < len(topic_umass_scores) else None,
            'Top 5 Keywords': ", ".join(summary['top_words'][:5])
        })
    topic_eval_df = pd.DataFrame(topic_eval_rows)

    educational_note = (
        "Mathematical evaluation metrics (such as Perplexity, Log-Likelihood, and Coherence scores) "
        "provide statistical indicators of document reconstruction error and term co-occurrence. "
        "However, empirical studies in NLP show that higher mathematical scores do not necessarily "
        "guarantee that the discovered topics are conceptually meaningful, distinct, or actionable for humans. "
        "Human qualitative review (topic intruder testing, domain labeling) is always necessary alongside statistical metrics."
    )

    return {
        'perplexity': stats['perplexity'],
        'log_likelihood': stats['log_likelihood'],
        'topic_diversity': diversity,
        'mean_umass_coherence': mean_umass,
        'gensim_cv_coherence': gensim_cv,
        'topic_eval_df': topic_eval_df,
        'educational_note': educational_note,
        'n_topics': topic_modeler.n_topics,
        'vocab_size': len(feature_names),
        'n_documents': dtm.shape[0]
    }
