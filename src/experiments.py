"""Experiment sweep: try several model configurations and track them all in MLflow.

This is the reason MLflow exists in this project. Each configuration is a run;
every run records its parameters, its metrics, and its model file. Afterwards you
open the MLflow UI, sort by f1_negative, and pick the winner with evidence
instead of memory.

Run:  python -m src.experiments
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import mlflow

from src.data import load_imdb
from src.evaluate import evaluate_negative_class
from src.train import build_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Each dict is one experiment run. We vary the vocabulary size, whether word
# pairs (bigrams) are used, and C (the regularization strength).
CONFIGS = [
    {"name": "A: unigrams, 3k words", "max_features": 3_000, "ngram": (1, 1), "C": 1.0},
    {"name": "B: unigrams, 50k words", "max_features": 50_000, "ngram": (1, 1), "C": 1.0},
    {"name": "C: + bigrams, 50k", "max_features": 50_000, "ngram": (1, 2), "C": 1.0},
    {"name": "D: + bigrams, weaker C=0.1", "max_features": 50_000, "ngram": (1, 2), "C": 0.1},
    {"name": "E: + bigrams, stronger C=10", "max_features": 50_000, "ngram": (1, 2), "C": 10.0},
    # F is the config that has to earn its place: it doubles training time and
    # model size, so it is only worth shipping if the sweep says so. Note that
    # F1 on IMDB is the wrong place to look for its benefit -- IMDB is clean
    # prose, so char n-grams have almost no misspellings to rescue there. The
    # payoff shows up in tests/test_typos.py, on the text real customers write.
    {"name": "F: word+char n-grams", "max_features": 50_000, "ngram": (1, 2), "C": 1.0, "use_char": True},
]


def run_sweep() -> list[dict]:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("review-radar: model sweep")

    train_df, test_df = load_imdb()
    results = []

    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["name"]):
            use_char = cfg.get("use_char", False)
            mlflow.log_params(
                {
                    "max_features": cfg["max_features"],
                    "ngram_range": f"{cfg['ngram'][0]}-{cfg['ngram'][1]}",
                    "C": cfg["C"],
                    "model": "tfidf_logreg",
                    "features": "word+char" if use_char else "word",
                }
            )

            pipeline = build_pipeline(
                max_features=cfg["max_features"],
                C=cfg["C"],
                ngram=cfg["ngram"],
                use_char=use_char,
            )
            pipeline.fit(train_df["text"], train_df["label"])

            p_negative = pipeline.predict_proba(test_df["text"])[:, 0]
            metrics = evaluate_negative_class(test_df["label"].to_numpy(), p_negative)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

            results.append({"name": cfg["name"], **metrics})
            print(f"{cfg['name']:<32} F1(neg) = {metrics['f1_negative']:.4f}")

    best = max(results, key=lambda r: r["f1_negative"])
    print(f"\nWinner: {best['name']}  (F1 = {best['f1_negative']:.4f})")
    return results


if __name__ == "__main__":
    run_sweep()
