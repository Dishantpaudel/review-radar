"""Train the baseline sentiment model: TF-IDF + Logistic Regression.

Simple, fast, and strong — this is the model that goes to production.
Everything is tracked in MLflow so every experiment is reproducible.
"""

import pickle
from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import load_imdb
from src.evaluate import evaluate_negative_class
from src.text import normalize

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "tfidf_logreg.pkl"


def build_pipeline(
    max_features: int = 50_000,
    C: float = 1.0,
    ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (3, 5),
    use_char: bool = True,
) -> Pipeline:
    """TF-IDF + Logistic Regression, over word *and* character n-grams.

    Word n-grams alone have one hard failure: a word the vocabulary has never
    seen contributes nothing at all. Real customers misspell precisely the words
    that carry the sentiment -- "vad", "borst", "terrable" -- so the signal
    drops out of exactly the reviews that matter most. Worse, "i dont like it"
    survives as the tokens "dont"/"like"/"it", and "like" is a *positive*
    feature, so a complaint can read as praise.

    Character n-grams fix this because a misspelling still shares most of its
    substrings with the word it meant: "borst" and "worst" have "ors", "rst" and
    "st " in common, so the model still sees most of the evidence. The two
    vectorizers are unioned rather than swapped -- word n-grams remain better on
    correctly spelled text, and bigrams are what capture "dont like" as a unit.

    Both share `normalize`, so the vectorizer and the urgency engine agree on
    what a word is; previously the engine folded "don't" to "dont" and the model
    did not, leaving them with two different vocabularies for one review.
    """
    word = TfidfVectorizer(
        preprocessor=normalize,
        max_features=max_features,
        ngram_range=ngram,
        sublinear_tf=True,
        min_df=2,
    )
    if not use_char:
        # Keeps the original single-vectorizer shape for the baseline configs.
        return Pipeline([("tfidf", word), ("clf", LogisticRegression(C=C, max_iter=1000))])

    char = TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",           # n-grams stay inside word boundaries
        ngram_range=char_ngram,
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
    )
    return Pipeline(
        [
            ("tfidf", FeatureUnion([("word", word), ("char", char)])),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def train(max_features: int = 50_000, C: float = 1.0, use_char: bool = True) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("review-radar: baseline")

    train_df, test_df = load_imdb()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "tfidf_logreg",
                "max_features": max_features,
                "C": C,
                "ngram_range": "1-2",
                "char_ngram_range": "3-5" if use_char else "none",
                "features": "word+char" if use_char else "word",
            }
        )

        pipeline = build_pipeline(max_features=max_features, C=C, use_char=use_char)
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
