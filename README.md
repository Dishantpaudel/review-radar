# review-radar

Real-time customer review analysis for a movie website: read every review the moment it arrives, spot the angry ones, and route the urgent ones to a human — before the customer leaves.

Built as the final project for Industrial Machine Learning (Harbour.Space, 2026).

## The problem

A movie site gets hundreds of thousands of reviews a month. Nobody can read them all, so angry customers are noticed too late — after they already left. An angry review is an early warning; today it is ignored.

## What this system does

```
new review ──► sentiment model P(negative) ──┐
                                             ├──► router ──► support team, now
          └──► urgency engine (rules) ───────┘              product feedback backlog
                                                            analytics
```

Each review is scored twice, because **"negative" and "urgent" are different questions**:

```python
if urgency >= 0.4 and p_negative >= 0.2:  return "support_urgent"
if p_negative >= 0.5:                     return "feedback_backlog"
return "analytics"
```

Two ideas from review feedback are built in:

1. **Negative ≠ urgent.** "The ending felt rushed" is useful feedback, not an emergency. "Charged twice, cancelling NOW!!!" needs a human in minutes. A transparent rule-based urgency engine (churn intent, billing terms, anger vocabulary, shouting) separates the two, and every escalation returns the exact terms that matched.
2. **The decision threshold is chosen for money, not for F1.** Contacting a flagged customer costs $1; saving one is worth ~$9. Because value is 9× cost, the profit-optimal threshold is *lower* than the F1-optimal one — it pays to contact more people. Break-even precision is only C / $9 ≈ **11%**.

Urgency escalates **independently** of the sentiment score, with sentiment acting only as a floor. That ordering is deliberate — see [Defects found](#defects-found-and-fixed).

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

## Defects found and fixed

Both defects below were found by **manually using the system**, not by the test suite. Both are now pinned by regression tests named after the failure.

### 1. The churn lexicon was too literal

`CHURN_PHRASES` matched fixed keywords only, so two sentences with identical meaning took opposite paths:

| Input | Urgency | Route |
|---|---|---|
| "I want to **cancel** my subscription" | 0.525 | `support_urgent` |
| "I **dont want** my subscription" | 0.075 | `feedback_backlog` ❌ |

**Fix:** added `CHURN_PATTERNS` — regexes for *compositional* churn intent, anchored to subscription/account nouns so `"I dont want to spoil the ending"` cannot false-positive. Apostrophes are normalised so `don't` and `dont` are one case.

### 2. The routing precedence was inverted

The original `route()` evaluated sentiment first:

```python
if p_negative < 0.5:  return "analytics"   # ← vetoed everything below it
```

The sentiment model is trained on IMDB **movie** reviews and is out of domain on billing language. It scored a departing customer at **0.319**:

```
"Charged twice and nobody replied. I want a REFUND, cancelling today!!!"
p_negative = 0.319   urgency = 0.828   →   "analytics"
```

That unreliable score silently overruled an urgency of 0.828 and routed a churning customer to analytics — the exact outcome this project exists to prevent.

**Fix:** urgency escalates independently, with sentiment reduced to a `clearly_positive` floor that stops a churn keyword inside praise from paging an agent.

> **The principle:** an unreliable component must never be able to veto a reliable one.

Tests grew **10 → 23** covering both defects and both false-positive guards.

## Data

[IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011): 50,000 labeled movie reviews, balanced, 25k train / 25k test. Downloaded automatically on first run. It is a stand-in — in production the model would be retrained monthly on the site's own reviews.

## Project structure

```
src/
  data.py                 load IMDB
  train.py                TF-IDF + LogReg training, tracked in MLflow
  evaluate.py             F1/recall/precision for the negative class, precision@k
  experiments.py          5-config MLflow sweep
  compare_transformer.py  pretrained DistilBERT comparison
  threshold.py            profit-vs-threshold sweep, break-even math
  urgency.py              urgency scoring + routing matrix
  predict.py              load model, score one review
service/app.py            FastAPI: POST /review → {p_negative, urgency, route}
notebooks/01–04           EDA → baseline → transformer → business threshold
tests/                    urgency, threshold math, API (23 tests)
presentation.html         slide deck
how-it-works.html         end-to-end walkthrough
```

## How to run

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

python -m src.train                  # trains baseline, logs to MLflow, saves models/tfidf_logreg.pkl
python -m src.compare_transformer    # optional: DistilBERT comparison (CPU, ~10 min)
pytest                               # 23 tests

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
