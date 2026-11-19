"""
src/inference.py
Topic inference for new/unseen documents using a fitted TopicModeler instance.
Provides single-document and batch prediction capabilities.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class TopicInferencer:
    """
    Wraps a fitted TopicModeler to infer topic distributions
    for new unseen documents.
    """

    def __init__(self, topic_modeler):
        """
        Parameters
        ----------
        topic_modeler : TopicModeler
            A fully fitted TopicModeler instance.
        """
        if not topic_modeler.is_fitted:
            raise ValueError("TopicModeler must be fitted before creating an inferencer.")
        self.topic_modeler = topic_modeler

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Infer the topic distribution for a single document.
        Applies identical preprocessing configured during model training.

        Returns
        -------
        dict with keys:
          - text: original input
          - cleaned_text: preprocessed text
          - matched_terms: count of vocabulary terms found in document
          - is_oov: boolean indicating if input had zero vocabulary terms
          - distribution: {topic_label: probability}
          - dominant_topic_id: 1-based topic index
          - dominant_topic_label: verified or auto-generated label string
          - dominant_probability: float
          - confidence_level: 'High' / 'Medium' / 'Low' / 'Zero Match / Out-of-Vocabulary'
        """
        if not text or not str(text).strip():
            raise ValueError("Input text must be a non-empty string.")

        res = self.topic_modeler.transform_new_text(text)

        # Confidence band with explicit OOV / Zero Match handling
        if res.get('is_oov', False) or res.get('matched_terms', 0) == 0:
            confidence = "Zero Match / Out-of-Vocabulary"
        elif res['dominant_probability'] >= 0.60:
            confidence = "High"
        elif res['dominant_probability'] >= 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"

        res['confidence_level'] = confidence
        return res

    def batch_predict(
        self, texts: Union[List[str], pd.Series]
    ) -> pd.DataFrame:
        """
        Infer topic distributions for a list of documents.
        Applies identical model preprocessing pipeline to each text.

        Returns
        -------
        pd.DataFrame with columns: text_snippet, matched_terms, dominant_topic_id,
        dominant_topic_label, dominant_probability, confidence_level, Topic_1_Prob … Topic_K_Prob
        """
        tm = self.topic_modeler
        clean_inputs = [str(t) for t in texts if str(t).strip()]

        if not clean_inputs:
            return pd.DataFrame()

        rows = []
        for raw_text in clean_inputs:
            pred = self.predict(raw_text)
            row = {
                "text_snippet": raw_text[:120] + ("…" if len(raw_text) > 120 else ""),
                "matched_terms": pred.get("matched_terms", 0),
                "dominant_topic_id": pred["dominant_topic_id"],
                "dominant_topic_label": pred["dominant_topic_label"],
                "dominant_probability": pred["dominant_probability"],
                "confidence_level": pred["confidence_level"],
            }
            for topic_name, prob in pred["distribution"].items():
                row[f"{topic_name.replace(' ', '_')}_Prob"] = prob
            rows.append(row)

        return pd.DataFrame(rows)

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[str]:
        """
        Return top keywords for a given 1-based topic ID.
        """
        if topic_id < 1 or topic_id > self.topic_modeler.n_topics:
            raise ValueError(f"topic_id must be between 1 and {self.topic_modeler.n_topics}")
        summary = self.topic_modeler.topic_summaries[topic_id - 1]
        return summary["top_words"][:top_n]
