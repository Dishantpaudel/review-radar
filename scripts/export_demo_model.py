"""Export a compact word+char model for the presentation's in-browser demo.

The slide deck runs a real logistic-regression model in JavaScript so the
interactive demo works with no server. The shipped model is 100,000 features
(50k word + 50k char) and would be ~2.5 MB of JSON — far too big to embed. So
this trains a *compact* model with the same shape (word + char_wb n-grams, same
`normalize`) and exports it as JSON.

It also prints reference P(negative) for a set of probe strings, so the JS
vectoriser can be checked against Python — the deck must not claim behaviour the
real system does not have, and a mis-ported vectoriser would do exactly that.

Run:  python -m scripts.export_demo_model
Out:  scripts/demo_model.json  (paste into <script id="demoModel"> in the deck)
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import load_imdb
from src.text import normalize

OUT = Path(__file__).resolve().parent / "demo_model.json"

WORD_FEATURES = 12000
CHAR_FEATURES = 30000

PROBES = [
    "this movie was bad", "this movie was vad", "the film was worst",
    "the film was borst", "absolutely terrible acting", "absolutely terrable acting",
    "i dont like it", "i love it", "this movie was wonderful", "its boring", "its borst",
    "Charged twice, I want a REFUND, cancelling today!!!",
    "The ending felt rushed and the pacing dragged.",
]


def main():
    train_df, test_df = load_imdb()

    word = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                           max_features=WORD_FEATURES, sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                           max_features=CHAR_FEATURES, sublinear_tf=True, min_df=3)
    pipe = Pipeline([
        ("tfidf", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    pipe.fit(train_df["text"], train_df["label"])

    # Agreement with the shipped 50k model, on the same rounding the demo uses.
    from src.predict import load_model
    shipped = load_model()
    p_demo = pipe.predict_proba(test_df["text"])[:, 0]
    p_ship = shipped.predict_proba(test_df["text"])[:, 0]
    agree = float(((p_demo >= 0.5) == (p_ship >= 0.5)).mean())

    coef = pipe.named_steps["clf"].coef_[0]
    bias = float(pipe.named_steps["clf"].intercept_[0])
    n_word = len(word.vocabulary_)

    # FeatureUnion concatenates word block then char block; each TfidfVectorizer
    # L2-normalises its own block before concatenation, so the JS side must too.
    word_vocab = sorted(word.vocabulary_, key=lambda t: word.vocabulary_[t])
    char_vocab = sorted(char.vocabulary_, key=lambda t: char.vocabulary_[t])

    model = {
        "words": word_vocab,
        "word_idf": [round(float(x), 5) for x in word.idf_],
        "word_coef": [round(float(x), 5) for x in coef[:n_word]],
        "chars": char_vocab,
        "char_idf": [round(float(x), 5) for x in char.idf_],
        "char_coef": [round(float(x), 5) for x in coef[n_word:]],
        "bias": round(bias, 5),
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    print(f"agreement with shipped model: {agree:.3%}")
    print(f"features: {n_word} word + {len(char_vocab)} char")
    print(f"json size: {OUT.stat().st_size / 1024:.0f} KB")
    print("\n# reference P(negative) — JS must match these:")
    for t in PROBES:
        print(f"  {pipe.predict_proba([t])[0, 0]:.4f}  {t}")


if __name__ == "__main__":
    main()
