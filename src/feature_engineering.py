"""
src/feature_engineering.py
Feature Engineering module for TopicLens AI.
Wraps CountVectorizer for Document-Term Matrix construction.
TF-IDF is available as a secondary exploratory feature only.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureEngineer:
    """
    Builds Document-Term Matrix (DTM) using CountVectorizer for LDA.
    TF-IDF is available as an optional secondary analysis feature.
    """

    def __init__(
        self,
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
        max_features: Optional[int] = 2000,
        ngram_range: Tuple[int, int] = (1, 1),
        token_pattern: str = r"\b[a-zA-Z]{3,}\b",
    ):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.token_pattern = token_pattern

        self.vectorizer: Optional[CountVectorizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.dtm: Optional[spmatrix] = None
        self.tfidf_matrix: Optional[spmatrix] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def build_dtm(self, texts: Union[List[str], pd.Series]) -> Tuple[spmatrix, List[str]]:
        """
        Fit CountVectorizer and build Document-Term Matrix.

        Returns:
            dtm: Sparse document-term matrix (count-based, for LDA)
            feature_names: Vocabulary term list
        """
        clean = [str(t) for t in texts if str(t).strip()]
        self.vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=self.token_pattern,
        )
        self.dtm = self.vectorizer.fit_transform(clean)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.is_fitted = True
        return self.dtm, self.feature_names

    def build_tfidf(self, texts: Union[List[str], pd.Series]) -> Tuple[spmatrix, List[str]]:
        """
        Fit TF-IDF vectorizer (secondary/exploratory use only — NOT used as LDA input).

        Returns:
            tfidf_matrix: Sparse TF-IDF matrix
            tfidf_feature_names: Vocabulary list
        """
        clean = [str(t) for t in texts if str(t).strip()]
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=self.token_pattern,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean)
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        return self.tfidf_matrix, tfidf_feature_names

    def get_dtm_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the constructed DTM.
        """
        if not self.is_fitted or self.dtm is None:
            raise ValueError("DTM has not been built yet. Call build_dtm() first.")

        n_docs, n_terms = self.dtm.shape
        total_elements = n_docs * n_terms
        non_zero = self.dtm.nnz
        sparsity = round(1.0 - (non_zero / total_elements), 4) if total_elements > 0 else 1.0

        return {
            "n_documents": n_docs,
            "vocabulary_size": n_terms,
            "total_elements": total_elements,
            "non_zero_elements": non_zero,
            "sparsity": sparsity,
            "density": round(1.0 - sparsity, 4),
            "avg_terms_per_doc": round(non_zero / n_docs, 2) if n_docs > 0 else 0.0,
        }

    def inspect_vocabulary(self, top_n: int = 30) -> pd.DataFrame:
        """
        Return the most frequent terms in the corpus as a DataFrame.
        """
        if not self.is_fitted or self.dtm is None or not self.feature_names:
            raise ValueError("DTM has not been built yet. Call build_dtm() first.")

        freq = np.asarray(self.dtm.sum(axis=0)).flatten()
        doc_freq = np.asarray((self.dtm > 0).sum(axis=0)).flatten()

        sorted_idx = np.argsort(freq)[::-1][:top_n]
        return pd.DataFrame({
            "Term": [self.feature_names[i] for i in sorted_idx],
            "Total Frequency": [int(freq[i]) for i in sorted_idx],
            "Document Frequency": [int(doc_freq[i]) for i in sorted_idx],
        })

    def transform(self, texts: Union[List[str], pd.Series]) -> spmatrix:
        """
        Transform new texts using the already-fitted CountVectorizer.
        """
        if not self.is_fitted or self.vectorizer is None:
            raise ValueError("Vectorizer is not fitted. Call build_dtm() first.")
        clean = [str(t) for t in texts if str(t).strip()]
        return self.vectorizer.transform(clean)
