"""Does adding character n-grams actually help? Measure it, do not assume it.

Trains two models differing in exactly one thing -- whether character n-grams are
unioned onto the word features -- and scores both on clean IMDB and on the same
reviews with typos injected.

Read the *short* row, not the headline. Measured over the whole test set the
answer looks like a rounding error: word-only loses just 0.007 F1 to a 15% typo
rate. That number is an artefact of the corpus. IMDB reviews average ~230 words,
so corrupting one word in seven still leaves ~195 clean words carrying the
sentiment, and the model coasts on the redundancy.

Real reviews are not 230 words. "its borst" is two, one of them misspelled, and
there is nothing else to fall back on -- word-only scores it 0.16, i.e.
confidently *positive*, because "borst" is out of vocabulary and contributes
literally nothing. Adding character n-grams moves it to 0.88. That is the effect
this module exists to expose, and averaging over long reviews hides it almost
completely.

Run:  python -m src.robustness
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import mlflow
import numpy as np

from src.data import load_imdb
from src.evaluate import evaluate_negative_class
from src.train import build_pipeline
from src.typos import corrupt_series

PROJECT_DIR = Path(__file__).resolve().parent.parent
TYPO_RATE = 0.15
SHORT_WORDS = 40  # a review with no redundancy left to hide behind


def run() -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("review-radar: typo robustness")

    train_df, test_df = load_imdb()
    y_test = test_df["label"].to_numpy()

    print(f"corrupting test set at rate={TYPO_RATE} ...")
    texts = test_df["text"].tolist()
    test_typo = corrupt_series(texts, rate=TYPO_RATE, seed=1234)

    # The slice where a typo actually costs something.
    short = np.array([len(t.split()) <= SHORT_WORDS for t in texts])
    print(f"short reviews (<= {SHORT_WORDS} words): {short.sum()} of {len(texts)}")

    results = {}
    for label, use_char in (("word only", False), ("word+char", True)):
        with mlflow.start_run(run_name=label):
            mlflow.log_params({"features": "word+char" if use_char else "word", "typo_rate": TYPO_RATE})

            print(f"training [{label}] ...")
            pipe = build_pipeline(use_char=use_char)
            pipe.fit(train_df["text"], train_df["label"])

            p_clean = pipe.predict_proba(texts)[:, 0]
            p_typo = pipe.predict_proba(test_typo)[:, 0]

            clean = evaluate_negative_class(y_test, p_clean)
            typo = evaluate_negative_class(y_test, p_typo)
            s_typo = evaluate_negative_class(y_test[short], p_typo[short])

            mlflow.log_metrics(
                {
                    "f1_clean": float(clean["f1_negative"]),
                    "f1_typo": float(typo["f1_negative"]),
                    "f1_typo_short": float(s_typo["f1_negative"]),
                    "f1_drop": float(clean["f1_negative"] - typo["f1_negative"]),
                }
            )
            results[label] = {
                "clean": clean["f1_negative"],
                "typo": typo["f1_negative"],
                "typo_short": s_typo["f1_negative"],
            }
            print(f"  {label:<10} clean={clean['f1_negative']:.4f}  typo={typo['f1_negative']:.4f}  typo/short={s_typo['f1_negative']:.4f}")

    print()
    print(f"{'':<12}{'F1 clean':>10}{'F1 typo':>10}{'typo/short':>12}")
    for label, r in results.items():
        print(f"{label:<12}{r['clean']:>10.4f}{r['typo']:>10.4f}{r['typo_short']:>12.4f}")

    w, c = results["word only"], results["word+char"]
    print()
    print(f"char n-grams, all reviews:   {(c['typo'] - w['typo']) * 100:+.2f} F1 points on typo'd text")
    print(f"char n-grams, short reviews: {(c['typo_short'] - w['typo_short']) * 100:+.2f} F1 points on typo'd text")
    print(f"cost on clean text:          {(c['clean'] - w['clean']) * 100:+.2f} F1 points")
    print()
    print(f"Read these with the sample size in mind: only {int(short.sum())} of {len(texts)} IMDB")
    print("test reviews are short enough for a typo to cost anything, so the short-review")
    print("column is suggestive, not decisive. That scarcity IS the finding -- IMDB is")
    print("long, edited prose, and it simply does not contain the kind of text this")
    print("change is for. It cannot answer the question on its own.")
    print()
    print("The decisive evidence is in tests/test_typos.py, on short reviews written the")
    print('way customers write them. There the effect is a flipped verdict, not a')
    print('fraction of a point: word-only scores "its borst" at 0.16 -- confidently')
    print("POSITIVE, because \"borst\" is out of vocabulary and contributes nothing at")
    print("all. word+char scores it 0.88. Same for \"i dont like it\" (0.46 -> 0.59),")
    print('where the only surviving in-vocabulary token is "like".')
    return results


if __name__ == "__main__":
    run()
