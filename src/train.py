"""Train the baseline sentiment model: TF-IDF + Logistic Regression.

Simple, fast, and strong — this is the model that goes to production.
Everything is tracked in MLflow so every experiment is reproducible.
"""

import pickle
from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_imdb
from src.evaluate import evaluate_negative_class

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "tfidf_logreg.pkl"


def build_pipeline(max_features: int = 50_000, C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def train(max_features: int = 50_000, C: float = 1.0) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("review-radar: baseline")

    train_df, test_df = load_imdb()

    with mlflow.start_run():
        mlflow.log_params({"model": "tfidf_logreg", "max_features": max_features, "C": C, "ngram_range": "1-2"})

        pipeline = build_pipeline(max_features=max_features, C=C)
        pipeline.fit(train_df["text"], train_df["label"])

        # P(negative) = probability of class 0
        p_negative = pipeline.predict_proba(test_df["text"])[:, 0]
        metrics = evaluate_negative_class(test_df["label"].to_numpy(), p_negative)

        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(MODEL_PATH))

    return metrics


if __name__ == "__main__":
    metrics = train()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
