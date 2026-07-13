"""Evaluation metrics for the sentiment model.

We care about catching negative reviews, so all metrics are computed
for the negative class, not overall accuracy.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

F1_GATE = 0.85  # project success gate from the problem statement


def precision_at_k(y_true_negative: np.ndarray, p_negative: np.ndarray, k: int) -> float:
    """Of the k reviews the model is most sure are negative, how many really are?

    This mirrors the support queue: the team works the top of the queue first.
    """
    top_k = np.argsort(p_negative)[::-1][:k]
    return float(np.mean(y_true_negative[top_k]))


def evaluate_negative_class(y_true: np.ndarray, p_negative: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all project metrics. y_true: 1 = positive, 0 = negative."""
    y_true_negative = (y_true == 0).astype(int)
    y_pred_negative = (p_negative >= threshold).astype(int)

    return {
        "f1_negative": f1_score(y_true_negative, y_pred_negative),
        "recall_negative": recall_score(y_true_negative, y_pred_negative),
        "precision_negative": precision_score(y_true_negative, y_pred_negative),
        "precision_at_100": precision_at_k(y_true_negative, p_negative, 100),
        "precision_at_1000": precision_at_k(y_true_negative, p_negative, 1000),
        "passes_gate": f1_score(y_true_negative, y_pred_negative) >= F1_GATE,
    }
