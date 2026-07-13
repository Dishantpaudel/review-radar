"""Load the trained model and score a single review."""

import pickle
from functools import lru_cache
from pathlib import Path

from src.urgency import route, urgency_score

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tfidf_logreg.pkl"


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_review(text: str, negative_threshold: float = 0.5) -> dict:
    model = load_model()
    p_negative = float(model.predict_proba([text])[0, 0])
    urgency = urgency_score(text)
    return {
        "p_negative": round(p_negative, 4),
        "urgency": urgency.score,
        "urgency_signals": urgency.signals,
        "route": route(p_negative, urgency.score, negative_threshold=negative_threshold),
    }


if __name__ == "__main__":
    import json
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "This movie was terrible, I want a refund!!!"
    print(json.dumps(score_review(text), indent=2))
