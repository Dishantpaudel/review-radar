# review-radar

Real-time customer review analysis for a movie website: read every review the moment it arrives, spot the angry ones, and route the urgent ones to a human — before the customer leaves.

Built as the final project for Industrial Machine Learning (Harbour.Space, 2026).

## The problem

A movie site gets hundreds of thousands of reviews a month. Nobody can read them all, so angry customers are noticed too late — after they already left. An angry review is an early warning; today it is ignored.

## What this system does

```
new review ──► sentiment model P(negative) ──► urgency scorer ──► route
                                                                  ├─ negative + urgent  → support team, now
                                                                  ├─ negative + calm    → product feedback backlog
                                                                  └─ positive           → analytics only
```

Two ideas from review feedback are built in:

1. **Negative ≠ urgent.** "The ending felt rushed" is useful feedback, not an emergency. "Charged twice, cancelling NOW!!!" needs a human in minutes. A transparent rule-based urgency scorer (churn phrases, billing words, anger words, shouting) separates the two, and every escalation shows exactly which signals fired.
2. **The decision threshold is chosen for money, not for F1.** Contacting a flagged customer costs $1; saving one is worth ~$9. Because value is 9× cost, the profit-optimal threshold is *lower* than the F1-optimal one — it pays to contact more people. Break-even precision is only C / $9 ≈ **11%**.

## Results

| Model | F1 (negative) | Recall (negative) | Precision (negative) | Notes |
|---|---|---|---|---|
| Constant guess | 0.50 | — | — | the score to beat |
| **TF-IDF + Logistic Regression** | **0.900** | 0.896 | 0.904 | trained here, ~2 ms inference, explainable |
| DistilBERT (pretrained SST-2, no fine-tune) | 0.898 | 0.913 | 0.885 | 3,000-review sample, CPU |

Both clear the project gate **F1(negative) ≥ 0.85**. The pretrained transformer ties the baseline; fine-tuning it on IMDB would push it to ~0.93 but needs a GPU. The baseline goes to production: ~100× cheaper to serve, fully explainable, precision@1000 = 1.00 (the top of the support queue is pure signal).

### Business threshold (notebook 04)

- Break-even precision: **0.111**
- Profit-optimal threshold: **0.44** (below the default 0.50, as theory predicts)
- Precision at the optimum: 0.87 — 8× above break-even, so the outreach program is a low-risk bet
- On this well-calibrated model the gain from tuning is modest (~$1.9k/month); with a noisier model or real-world class imbalance (only ~20% of production reviews are negative, vs 50% in IMDB) the gap grows. The method — sweep thresholds, maximize expected profit — is the point.

## Data

[IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011): 50,000 labeled movie reviews, balanced, 25k train / 25k test. Downloaded automatically on first run. It is a stand-in — in production the model would be retrained monthly on the site's own reviews.

## Project structure

```
src/
  data.py                 load IMDB
  train.py                TF-IDF + LogReg training, tracked in MLflow
  evaluate.py             F1/recall/precision for the negative class, precision@k
  compare_transformer.py  pretrained DistilBERT comparison
  threshold.py            profit-vs-threshold sweep, break-even math
  urgency.py              urgency scoring + routing matrix
  predict.py              load model, score one review
service/app.py            FastAPI: POST /review → {p_negative, urgency, route}
notebooks/01–04           EDA → baseline → transformer → business threshold
tests/                    urgency, threshold math, API
```

## How to run

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

python -m src.train                  # trains baseline, logs to MLflow, saves models/tfidf_logreg.pkl
python -m src.compare_transformer    # optional: DistilBERT comparison (CPU, ~10 min)
pytest                               # 10 tests

uvicorn service.app:app --port 8000  # live API
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/review -H "Content-Type: application/json" \
  -d "{\"text\": \"Terrible! Charged twice, I want a REFUND, cancelling today!!!\"}"
# → {"p_negative": 0.80, "urgency": 0.99, "route": "support_urgent", ...}
```

Experiment tracking: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000

## Roadmap

1. ✅ Baseline model — already passes the goal
2. ✅ Transformer comparison + live service + urgency routing + profit-optimal threshold
3. Monthly retraining on own reviews, drift monitoring, threshold re-tuned as costs change
4. Topic mining on negative reviews — find out *why* customers are unhappy, not just that they are
