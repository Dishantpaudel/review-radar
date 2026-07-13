"""Compare the baseline against a pretrained transformer.

Model: distilbert-base-uncased-finetuned-sst-2-english (no fine-tuning —
it ships already trained for sentiment). We score a fixed random sample
of the IMDB test set on CPU and log the same metrics to MLflow so the
two models are directly comparable.
"""

from pathlib import Path

import mlflow
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_imdb
from src.evaluate import evaluate_negative_class

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SIZE = 3000
BATCH_SIZE = 32


def score_texts(texts: list[str]) -> np.ndarray:
    """Return P(negative) for each text."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    p_negative = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            p_negative.extend(probs[:, 0].tolist())  # class 0 = NEGATIVE in SST-2
    return np.array(p_negative)


def main() -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("review-radar: transformer")

    _, test_df = load_imdb()
    sample = test_df.sample(SAMPLE_SIZE, random_state=42)

    with mlflow.start_run():
        mlflow.log_params({"model": MODEL_NAME, "sample_size": SAMPLE_SIZE, "fine_tuned": False})
        p_negative = score_texts(sample["text"].tolist())
        metrics = evaluate_negative_class(sample["label"].to_numpy(), p_negative)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    return metrics


if __name__ == "__main__":
    metrics = main()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
