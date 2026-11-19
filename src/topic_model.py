"""
src/topic_model.py
Core Topic Modelling engine using Scikit-Learn's Latent Dirichlet Allocation (LDA)
and CountVectorizer for Document-Term Matrix (DTM) construction.
Includes topic-word extraction, document-topic distributions, dominant topic assignment,
automatic topic labeling, and model persistence via Joblib.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TopicModeler:
    """
    Manages Document-Term Matrix extraction and Latent Dirichlet Allocation (LDA) training,
    topic extraction, document-topic probability inference, and topic labeling.
    """

    def __init__(
        self,
        n_topics: int = 4,
        max_iter: int = 25,
        learning_method: str = 'batch',
        learning_decay: float = 0.7,
        doc_topic_prior: Optional[float] = None,
        topic_word_prior: Optional[float] = None,
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
        max_features: Optional[int] = 1500,
        random_state: int = 42,
        top_words_count: int = 10
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.random_state = random_state
        self.top_words_count = top_words_count

        self.vectorizer: Optional[CountVectorizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.lda_model: Optional[LatentDirichletAllocation] = None
        self.feature_names: List[str] = []
        self.dtm = None
        self.doc_topic_dist: Optional[np.ndarray] = None
        self.topic_summaries: List[Dict[str, Any]] = []
        self.is_fitted: bool = False
        self.training_time_sec: float = 0.0

    def fit(self, texts: Union[List[str], pd.Series]) -> "TopicModeler":
        """
        Build Document-Term Matrix with CountVectorizer and train LDA model.
        """
        import time
        start_time = time.time()

        # Input validation
        clean_texts = [str(t) for t in texts if isinstance(t, str) and str(t).strip()]
        if len(clean_texts) < self.n_topics:
            raise ValueError(
                f"Number of documents ({len(clean_texts)}) must be greater than or equal "
                f"to the number of topics ({self.n_topics})."
            )

        # 1. Document-Term Matrix via CountVectorizer (primary for standard LDA)
        self.vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        self.dtm = self.vectorizer.fit_transform(clean_texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        if len(self.feature_names) == 0:
            raise ValueError("Vocabulary is empty. Adjust min_df / max_df settings.")

        # 2. Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            learning_decay=self.learning_decay,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            random_state=self.random_state,
            evaluate_every=1,
            n_jobs=-1
        )
        self.doc_topic_dist = self.lda_model.fit_transform(self.dtm)

        # 3. Optional TF-IDF vectorizer for secondary comparative text analytics
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            self.tfidf_vectorizer.fit(clean_texts)
        except Exception:
            self.tfidf_vectorizer = None

        self.training_time_sec = round(time.time() - start_time, 2)
        self.is_fitted = True

        # Generate topic summaries
        self.topic_summaries = self._extract_topics(self.top_words_count)
        return self

    def _extract_topics(self, n_top_words: int) -> List[Dict[str, Any]]:
        """
        Extract top keywords, word weights, and generate human-readable suggested labels.
        """
        if not self.is_fitted or self.lda_model is None:
            return []

        summaries = []
        topic_word_matrix = self.lda_model.components_

        for topic_idx, topic in enumerate(topic_word_matrix):
            # Normalize to probabilities
            topic_prob = topic / topic.sum()
            top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
            
            top_words = [self.feature_names[i] for i in top_word_indices]
            top_weights = [float(round(topic[i], 4)) for i in top_word_indices]
            top_probabilities = [float(round(topic_prob[i], 4)) for i in top_word_indices]

            # Generate automatic label suggestion based on top 3 distinct words
            auto_label = f"Topic {topic_idx + 1}: " + ", ".join(top_words[:3]).title()

            summaries.append({
                'topic_id': topic_idx,
                'topic_number': topic_idx + 1,
                'topic_name': f"Topic {topic_idx + 1}",
                'auto_label': auto_label,
                'top_words': top_words,
                'top_weights': top_weights,
                'top_probabilities': top_probabilities,
                'word_weight_dict': dict(zip(top_words, top_weights)),
                'word_prob_dict': dict(zip(top_words, top_probabilities)),
                'disclaimer': (
                    "Note: Automatically generated labels represent statistical term prevalence "
                    "and should be reviewed and verified by a domain expert."
                )
            })

        return summaries

    def get_topic_keywords_df(self) -> pd.DataFrame:
        """
        Create a tidy DataFrame of topics and their top keywords.
        """
        rows = []
        for summary in self.topic_summaries:
            for rank, (word, weight, prob) in enumerate(
                zip(summary['top_words'], summary['top_weights'], summary['top_probabilities']),
                start=1
            ):
                rows.append({
                    'Topic ID': summary['topic_number'],
                    'Topic Label': summary['auto_label'],
                    'Rank': rank,
                    'Word': word,
                    'Component Weight': weight,
                    'Word Probability': prob
                })
        return pd.DataFrame(rows)

    def get_document_topic_df(self, original_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Combine original document DataFrame with topic probability distributions
        and dominant topic assignments.
        """
        if not self.is_fitted or self.doc_topic_dist is None:
            raise ValueError("Model has not been fitted yet.")

        result_df = original_df.copy()

        # Add topic probability columns
        for topic_idx in range(self.n_topics):
            col_name = f"Topic_{topic_idx + 1}_Prob"
            result_df[col_name] = np.round(self.doc_topic_dist[:, topic_idx], 4)

        # Dominant topic assignment
        dominant_topic_indices = np.argmax(self.doc_topic_dist, axis=1)
        dominant_topic_probs = np.max(self.doc_topic_dist, axis=1)

        result_df['Dominant_Topic_ID'] = dominant_topic_indices + 1
        result_df['Dominant_Topic_Label'] = [
            self.topic_summaries[idx]['auto_label'] for idx in dominant_topic_indices
        ]
        result_df['Dominant_Topic_Probability'] = np.round(dominant_topic_probs, 4)

        return result_df

    def transform_new_text(self, text: str) -> Dict[str, Any]:
        """
        Infer topic probabilities for a new single document or query.
        """
        if not self.is_fitted or self.vectorizer is None or self.lda_model is None:
            raise ValueError("Model is not fitted.")

        transformed_dtm = self.vectorizer.transform([text])
        doc_distribution = self.lda_model.transform(transformed_dtm)[0]

        dominant_idx = int(np.argmax(doc_distribution))
        dominant_prob = float(round(doc_distribution[dominant_idx], 4))

        distribution_dict = {
            f"Topic {i + 1}": float(round(prob, 4))
            for i, prob in enumerate(doc_distribution)
        }

        return {
            'text': text,
            'distribution': distribution_dict,
            'dominant_topic_id': dominant_idx + 1,
            'dominant_topic_label': self.topic_summaries[dominant_idx]['auto_label'],
            'dominant_probability': dominant_prob
        }

    def save(self, filepath: Union[str, Path]) -> str:
        """
        Persist trained model bundle to disk via joblib.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            'n_topics': self.n_topics,
            'max_iter': self.max_iter,
            'learning_method': self.learning_method,
            'learning_decay': self.learning_decay,
            'doc_topic_prior': self.doc_topic_prior,
            'topic_word_prior': self.topic_word_prior,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'top_words_count': self.top_words_count,
            'vectorizer': self.vectorizer,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'lda_model': self.lda_model,
            'feature_names': self.feature_names,
            'topic_summaries': self.topic_summaries,
            'training_time_sec': self.training_time_sec,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(bundle, path)
        return str(path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TopicModeler":
        """
        Load a trained model bundle from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        bundle = joblib.load(path)
        instance = cls(
            n_topics=bundle['n_topics'],
            max_iter=bundle['max_iter'],
            learning_method=bundle['learning_method'],
            learning_decay=bundle.get('learning_decay', 0.7),
            doc_topic_prior=bundle.get('doc_topic_prior', None),
            topic_word_prior=bundle.get('topic_word_prior', None),
            min_df=bundle['min_df'],
            max_df=bundle['max_df'],
            max_features=bundle['max_features'],
            random_state=bundle['random_state'],
            top_words_count=bundle['top_words_count']
        )
        instance.vectorizer = bundle['vectorizer']
        instance.tfidf_vectorizer = bundle.get('tfidf_vectorizer')
        instance.lda_model = bundle['lda_model']
        instance.feature_names = bundle['feature_names']
        instance.topic_summaries = bundle['topic_summaries']
        instance.training_time_sec = bundle.get('training_time_sec', 0.0)
        instance.is_fitted = True
        return instance
