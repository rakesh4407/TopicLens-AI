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

        Returns
        -------
        dict with keys:
          - text: original input
          - distribution: {topic_label: probability}
          - dominant_topic_id: 1-based topic index
          - dominant_topic_label: auto-label string
          - dominant_probability: float
          - confidence_level: 'High' / 'Medium' / 'Low'
        """
        if not text or not str(text).strip():
            raise ValueError("Input text must be a non-empty string.")

        tm = self.topic_modeler
        dtm = tm.vectorizer.transform([str(text)])
        distribution = tm.lda_model.transform(dtm)[0]

        dominant_idx = int(np.argmax(distribution))
        dominant_prob = float(round(distribution[dominant_idx], 4))

        # Confidence band
        if dominant_prob >= 0.60:
            confidence = "High"
        elif dominant_prob >= 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"

        distribution_dict = {
            tm.topic_summaries[i]["topic_name"]: float(round(p, 4))
            for i, p in enumerate(distribution)
        }

        return {
            "text": text,
            "distribution": distribution_dict,
            "dominant_topic_id": dominant_idx + 1,
            "dominant_topic_label": tm.topic_summaries[dominant_idx]["auto_label"],
            "dominant_probability": dominant_prob,
            "confidence_level": confidence,
        }

    def batch_predict(
        self, texts: Union[List[str], pd.Series]
    ) -> pd.DataFrame:
        """
        Infer topic distributions for a list of documents.

        Returns
        -------
        pd.DataFrame with columns: text, dominant_topic_id, dominant_topic_label,
        dominant_probability, confidence_level, Topic_1_Prob … Topic_K_Prob
        """
        tm = self.topic_modeler
        clean = [str(t) for t in texts if str(t).strip()]

        if not clean:
            return pd.DataFrame()

        dtm = tm.vectorizer.transform(clean)
        distributions = tm.lda_model.transform(dtm)

        rows = []
        for i, dist in enumerate(distributions):
            dominant_idx = int(np.argmax(dist))
            dominant_prob = float(round(dist[dominant_idx], 4))
            confidence = (
                "High" if dominant_prob >= 0.60
                else ("Medium" if dominant_prob >= 0.35 else "Low")
            )
            row = {
                "text_snippet": clean[i][:120] + ("…" if len(clean[i]) > 120 else ""),
                "dominant_topic_id": dominant_idx + 1,
                "dominant_topic_label": tm.topic_summaries[dominant_idx]["auto_label"],
                "dominant_probability": dominant_prob,
                "confidence_level": confidence,
            }
            for t_idx, prob in enumerate(dist):
                row[f"Topic_{t_idx + 1}_Prob"] = float(round(prob, 4))
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
